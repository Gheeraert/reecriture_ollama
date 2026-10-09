#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Réécriture française locale avec Ollama et Tkinter. Python 3.10+, sans dépendance externe."""
from __future__ import annotations

import json
import os
import queue
import re
import tempfile
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from urllib import error, request

from text_diff import DiffOp, DiffStats, compute_diff

OLLAMA_URL = "http://127.0.0.1:11434"
RECOMMENDED_MODELS = ("qwen3:14b", "gemma3:12b", "qwen3:8b")
MAX_GROUP_CHARS = 3200
MAX_GROUP_PARAGRAPHS = 8

STYLES = {
    "Correction légère": "Corrige uniquement les fautes d'orthographe, d'accord, de grammaire et de ponctuation. Ne reformule que si une phrase est incorrecte.",
    "Reformulation fidèle": "Améliore la clarté et la fluidité avec une reformulation sobre. Conserve toutes les idées, nuances et informations.",
    "Style universitaire": (
        "Tu es un rédacteur et éditeur français de très haut niveau.\n\n"
        "Ta mission est de produire une nouvelle version du passage fourni, en conservant rigoureusement son contenu intellectuel, mais en repensant profondément son expression.\n\n"
        "Objectif :\n"
        "- conserver les idées, les informations, les faits, les nuances, les relations logiques et chronologiques ;\n"
        "- transformer de manière nette la forme du passage ;\n"
        "- éviter une simple paraphrase locale.\n\n"
        "Consignes :\n\n"
        "1. Commence par comprendre la structure du raisonnement, sans l'afficher.\n"
        "2. Identifie les éléments non négociables :\n"
        "   - noms propres\n"
        "   - dates\n"
        "   - faits historiques\n"
        "   - distinctions conceptuelles\n"
        "   - termes techniques à conserver si nécessaire\n"
        "   - modalités du jugement\n"
        "3. Réécris ensuite le passage en repensant réellement son organisation stylistique.\n\n"
        "Tu dois :\n"
        "- modifier l'ordre des propositions lorsque c'est possible ;\n"
        "- redistribuer les informations entre les phrases ;\n"
        "- fusionner ou scinder des phrases ;\n"
        "- varier les structures syntaxiques ;\n"
        "- transformer les nominalisations en verbes, ou l'inverse, si cela aide ;\n"
        "- reformuler les articulations logiques sans en altérer le sens ;\n"
        "- éviter de suivre pas à pas la formulation originale.\n\n"
        "Tu ne dois pas :\n"
        "- procéder par simple substitution de synonymes ;\n"
        "- conserver la structure phrastique initiale si elle peut être refondue ;\n"
        "- remplacer un terme précis par un terme approximatif ;\n"
        "- ajouter des informations ;\n"
        "- affaiblir ou durcir une nuance ;\n"
        "- moderniser artificiellement le ton.\n\n"
        "Exigence spécifique :\n"
        "La nouvelle version doit être suffisamment différente dans sa formulation et dans son agencement pour qu'un lecteur ait le sentiment d'une véritable recomposition, tout en reconnaissant exactement la même pensée.\n\n"
        "Si le passage est en markdown, conserve parfaitement le markdown.\n\n"
        "N'affiche ni analyse ni commentaire.\n"
        "Fournis uniquement le texte réécrit."
    ),
    "Style littéraire": "Privilégie une prose française élégante, nuancée, expressive et rythmée, sans emphase ni invention.",
    "Alléger le texte": "Allège les répétitions et les lourdeurs, raccourcis les détours, sans supprimer d'information importante.",
}
INTENSITIES = {
    "Discrète": "Modifications minimales ; ne change pas inutilement le vocabulaire ou la syntaxe.",
    "Modérée": "Tu peux retravailler les constructions, mais sans effacer la voix de l'auteur.",
    "Soutenue": "Tu peux reformuler plus librement en préservant rigoureusement le sens, les faits et le ton général.",
}



class OllamaError(RuntimeError):
    pass


class Cancelled(Exception):
    pass


def api_json(path: str, payload: dict | None = None, timeout: int = 15) -> dict:
    try:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        req = request.Request(
            OLLAMA_URL + path, data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method="POST" if data else "GET",
        )
        with request.urlopen(req, timeout=timeout) as response:
            result = json.load(response)
        if "error" in result:
            raise OllamaError(str(result["error"]))
        return result
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:500]
        raise OllamaError(f"Ollama (HTTP {exc.code}) : {details}") from exc
    except (error.URLError, TimeoutError, ConnectionError) as exc:
        raise OllamaError("Ollama est inaccessible. Vérifie qu'il est installé et démarré (port 11434).") from exc


def api_stream(path: str, payload: dict, cancelled: threading.Event):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(OLLAMA_URL + path, data=data,
                          headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=600) as response:
            for raw_line in response:
                if cancelled.is_set():
                    raise Cancelled()
                if not raw_line.strip():
                    continue
                item = json.loads(raw_line.decode("utf-8"))
                if item.get("error"):
                    raise OllamaError(str(item["error"]))
                yield item
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:500]
        raise OllamaError(f"Ollama (HTTP {exc.code}) : {details}") from exc
    except (error.URLError, TimeoutError, ConnectionError) as exc:
        if cancelled.is_set():
            raise Cancelled() from exc
        raise OllamaError("Connexion Ollama interrompue ou délai d'attente dépassé.") from exc


def installed_models() -> list[str]:
    result = api_json("/api/tags", timeout=8)
    return sorted({m.get("name", "") for m in result.get("models", []) if m.get("name")}, key=str.lower)


def system_prompt(style: str, intensity: str, markdown: bool = False) -> str:
    instructions = [
        "Tu es une éditrice française expérimentée. Tu réécris exclusivement en français.",
        STYLES[style], INTENSITIES[intensity],
        "Ne crée aucun fait, aucune citation, aucune référence, aucun nom, aucune date ou source.",
        "Respecte les citations textuelles entre guillemets : ne les modifie pas.",
        "Respecte la voix de l'auteur. Ne donne ni explication, ni préface, ni commentaire.",
    ]
    if markdown:
        instructions.extend([
            "Tu travailles sur des fragments Markdown. Préserve strictement la structure : titres (#),",
            "listes, indentation, emphases, liens, renvois, notes, sauts de ligne signifiants.",
            "Toute balise de protection de forme ⟪VERROU_000⟫ doit être recopiée STRICTEMENT à l'identique.",
            "Ne crée ni ne supprime de paragraphe dans un fragment.",
            "Retourne uniquement l'objet JSON demandé, sans bloc de code Markdown.",
        ])
    else:
        instructions.append("Retourne uniquement le texte réécrit, sans guillemets englobants ni mise en forme ajoutée.")
    return "\n".join(instructions)


@dataclass
class Segment:
    kind: str  # "rewrite" ou "keep"
    text: str


FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
LINK_REFERENCE = re.compile(r"^ {0,3}\[[^\]\n]+\]:")
HORIZONTAL_RULE = re.compile(r"^ {0,3}(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,})$")
INLINE_PROTECTED = re.compile(
    r"!?\[[^\]\n]*\]\([^\n)]*\)|\[[^\]\n]+\]\[[^\]\n]*\]|`+[^`\n]+`+|\[\^[^\]\n]+\]|<https?://[^>\n]+>"
)
MARKER = re.compile(r"^(\s*(?:#{1,6}\s+|>\s*|(?:[-+*]|\d+[.)])\s+|\|))")


def _line_content(line: str) -> str:
    return line.rstrip("\r\n")


def tokenize_markdown(source: str) -> list[Segment]:
    """Sépare le texte à réécrire des zones à ne jamais transmettre au modèle."""
    lines = source.splitlines(keepends=True)
    result: list[Segment] = []
    i, n = 0, len(lines)

    def keep_until(end: int) -> None:
        nonlocal i
        result.append(Segment("keep", "".join(lines[i:end])))
        i = end

    if n and _line_content(lines[0]).strip() == "---":
        end = next((k + 1 for k in range(1, n)
                    if _line_content(lines[k]).strip() in ("---", "...")), None)
        if end:
            keep_until(end)

    while i < n:
        line = _line_content(lines[i])
        stripped = line.strip()
        if not stripped:
            j = i + 1
            while j < n and not _line_content(lines[j]).strip():
                j += 1
            keep_until(j)
            continue
        fence = FENCE.match(line)
        if fence:
            mark = fence.group(1)
            closing = re.compile(r"^ {0,3}" + re.escape(mark[0]) + "{" + str(len(mark)) + r",}\s*$")
            j = i + 1
            while j < n and not closing.match(_line_content(lines[j])):
                j += 1
            keep_until(min(j + 1, n))
            continue
        if stripped.startswith("$$"):
            j = i + 1
            if stripped != "$$" and stripped.endswith("$$"):
                keep_until(i + 1)
                continue
            while j < n and "$$" not in lines[j]:
                j += 1
            keep_until(min(j + 1, n))
            continue
        if "<!--" in line:
            j = i + 1
            if "-->" not in line:
                while j < n and "-->" not in lines[j]:
                    j += 1
                j = min(j + 1, n)
            keep_until(j)
            continue
        if re.match(r"^\s*<(?:script|style|pre|table|div|figure|svg)\b", line, re.I):
            opening = re.search(r"<([a-z]+)\b", line, re.I)
            tag = opening.group(1) if opening else "div"
            j = i + 1
            if f"</{tag}>" not in line.lower():
                while j < n and f"</{tag}>" not in lines[j].lower():
                    j += 1
                j = min(j + 1, n)
            keep_until(j)
            continue
        if (LINK_REFERENCE.match(line) or HORIZONTAL_RULE.match(line)
                or stripped.startswith("![") or line.startswith(("    ", "\t"))):
            keep_until(i + 1)
            continue
        if i + 1 < n and "|" in line and TABLE_SEPARATOR.match(_line_content(lines[i + 1])):
            j = i + 2
            while j < n and "|" in lines[j] and _line_content(lines[j]).strip():
                j += 1
            keep_until(j)
            continue
        # Un paragraphe Markdown ordinaire, terminé par une ligne blanche ou un bloc spécial.
        j = i + 1
        if line.startswith("#"):
            j = i + 1  # titre séparé du texte qui suit
        else:
            while j < n:
                next_line = _line_content(lines[j])
                if (not next_line.strip() or FENCE.match(next_line)
                        or next_line.lstrip().startswith(("# ", "## ", "### ", "#### ", "##### ", "###### "))
                        or next_line.strip().startswith(("$$", "<!--", "!["))
                        or LINK_REFERENCE.match(next_line) or HORIZONTAL_RULE.match(next_line)
                        or next_line.startswith(("    ", "\t"))
                        or (j + 1 < n and "|" in next_line and TABLE_SEPARATOR.match(_line_content(lines[j + 1])))):
                    break
                j += 1
        result.append(Segment("rewrite", "".join(lines[i:j])))
        i = j
    return result


def split_long_piece(content: str, cap: int = MAX_GROUP_CHARS) -> list[str]:
    if len(content) <= cap:
        return [content]
    pieces = []
    remaining = content
    while len(remaining) > cap:
        part = remaining[:cap]
        candidates = [part.rfind("\n"), part.rfind(". "), part.rfind("; "), part.rfind("! "), part.rfind("? "), part.rfind(" ")]
        split_at = max(candidates)
        if split_at < cap // 2:
            split_at = cap
        elif remaining[split_at:split_at + 1] != "\n":
            split_at += 1
        else:
            split_at += 1
        pieces.append(remaining[:split_at])
        remaining = remaining[split_at:]
    if remaining:
        pieces.append(remaining)
    return pieces


def prepare_segments(text: str) -> list[Segment]:
    tokens = tokenize_markdown(text)
    result = []
    for t in tokens:
        if t.kind == "rewrite":
            result.extend(Segment("rewrite", x) for x in split_long_piece(t.text))
        else:
            result.append(t)
    return result


def group_indices(segments: list[Segment]) -> list[list[int]]:
    """Groupe jusqu'à huit fragments sans jamais envoyer les régions protégées au modèle."""
    groups: list[list[int]] = []
    group: list[int] = []
    chars = 0
    for i, s in enumerate(segments):
        if s.kind != "rewrite":
            continue
        if group and (chars + len(s.text) > MAX_GROUP_CHARS or len(group) >= MAX_GROUP_PARAGRAPHS):
            groups.append(group)
            group, chars = [], 0
        group.append(i)
        chars += len(s.text)
    if group:
        groups.append(group)
    return groups


def mask_inline(text: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}
    def replace(match: re.Match) -> str:
        token = f"⟪VERROU_{len(protected):03d}⟫"
        protected[token] = match.group(0)
        return token
    return INLINE_PROTECTED.sub(replace, text), protected


def unmask_inline(text: str, protected: dict[str, str]) -> str:
    for key, value in protected.items():
        if text.count(key) != 1:
            raise ValueError(f"Marqueur protégé perdu ou dupliqué : {key}")
        text = text.replace(key, value)
    if "⟪VERROU_" in text:
        raise ValueError("Marqueur de protection inconnu dans la réponse")
    return text


def structural_signature(text: str) -> list[str]:
    """Vérification conservatrice des marqueurs structurels, pas validation Markdown intégrale."""
    return [m.group(1) for line in text.splitlines() if (m := MARKER.match(line))]


DEFAULT_OLLAMA_OPTIONS = {"temperature": 0.25, "top_p": 0.9, "repeat_penalty": 1.1, "num_predict": 4096}


def rewrite_markdown_group(model: str, segments: list[Segment], indices: list[int],
                           style: str, intensity: str, cancelled: threading.Event,
                           ollama_options: dict | None = None) -> tuple[list[str], list[str]]:
    masked = []
    protections = []
    for i in indices:
        t, p = mask_inline(segments[i].text)
        masked.append(t)
        protections.append(p)
    prompt = (
        "Réécris chaque fragment de cette liste, dans le même ordre et sans en omettre. "
        "Retourne un JSON de la forme {\"fragments\":[\"texte 1\",\"texte 2\"]}. "
        "Le tableau doit avoir exactement le même nombre d'éléments que la liste donnée. "
        "Préserve les éventuels sauts de ligne finaux. Voici les fragments :\n"
        + json.dumps(masked, ensure_ascii=False)
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt(style, intensity, markdown=True)},
            {"role": "user", "content": prompt},
        ],
        "think": False,
        "stream": True,
        "format": {
            "type": "object",
            "properties": {"fragments": {"type": "array", "items": {"type": "string"}}},
            "required": ["fragments"],
            "additionalProperties": False,
        },
        "options": {**DEFAULT_OLLAMA_OPTIONS, **(ollama_options or {}), "num_ctx": 8192},
    }
    if cancelled.is_set():
        raise Cancelled()
    body = "".join(item.get("message", {}).get("content", "")
                   for item in api_stream("/api/chat", payload, cancelled))
    if cancelled.is_set():
        raise Cancelled()
    try:
        data = json.loads(body)
        outputs = data["fragments"]
        if not isinstance(outputs, list) or len(outputs) != len(indices) or not all(isinstance(x, str) for x in outputs):
            raise ValueError("nombre de fragments différent")
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("Réponse JSON inattendue ou incomplète") from exc
    safe = []
    warnings = []
    for idx, out, locks in zip(indices, outputs, protections):
        original = segments[idx].text
        try:
            cleaned = unmask_inline(out, locks)
            if structural_signature(original) != structural_signature(cleaned):
                raise ValueError("structure Markdown modifiée")
            # Évite d'introduire des blocs de code ou d'en détruire par accident.
            if cleaned.count("```") != original.count("```"):
                raise ValueError("clôture Markdown inattendue")
            # Le modèle oublie souvent les sauts finaux ; on les rétablit précisément.
            end = re.search(r"[\r\n]*$", original).group(0)
            cleaned = cleaned.rstrip("\r\n") + end
            if not cleaned.strip() and original.strip():
                raise ValueError("fragment vide")
            safe.append(cleaned)
        except ValueError as exc:
            safe.append(original)
            warnings.append(f"Fragment {idx + 1} conservé sans changement ({exc}).")
    return safe, warnings


def rewrite_plain_stream(model: str, content: str, style: str, intensity: str,
                         cancelled: threading.Event, ollama_options: dict | None = None):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt(style, intensity)},
            {"role": "user", "content": content},
        ],
        "think": False,
        "stream": True,
        "options": {**DEFAULT_OLLAMA_OPTIONS, **(ollama_options or {}), "num_ctx": 8192},
    }
    for item in api_stream("/api/chat", payload, cancelled):
        chunk = item.get("message", {}).get("content", "")
        if chunk:
            yield chunk
        if item.get("done") and item.get("done_reason") == "length":
            raise OllamaError("Réponse tronquée par le modèle : essaie un texte plus court.")


def read_utf8_preserve_newlines(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return f.read()


def atomic_save(destination: Path, content: str, newline: str = "\n") -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".reecriture_", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(content.replace("\n", newline))
        os.replace(name, destination)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class RewriteApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Atelier de réécriture — Ollama (local)")
        self.geometry("1190x820")
        self.minsize(820, 600)
        self.jobs: queue.Queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.busy = False
        self.models: list[str] = []
        self.model = tk.StringVar()
        self.style = tk.StringVar(value="Reformulation fidèle")
        self.intensity = tk.StringVar(value="Modérée")
        self.temperature = tk.StringVar(value=str(DEFAULT_OLLAMA_OPTIONS["temperature"]))
        self.top_p = tk.StringVar(value=str(DEFAULT_OLLAMA_OPTIONS["top_p"]))
        self.repeat_penalty = tk.StringVar(value=str(DEFAULT_OLLAMA_OPTIONS["repeat_penalty"]))
        self.num_predict = tk.StringVar(value=str(DEFAULT_OLLAMA_OPTIONS["num_predict"]))
        self.download_model = tk.StringVar(value="qwen3:14b")
        self.source_path = tk.StringVar()
        self.target_path = tk.StringVar()
        self.status = tk.StringVar(value="Connexion à Ollama…")
        self._widgets()
        self.after(100, self._poll)
        self.refresh_models()

    def _widgets(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        opts = ttk.LabelFrame(self, text="Modèle et réglages")
        opts.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 5))
        for col in (1, 3):
            opts.columnconfigure(col, weight=1)
        ttk.Label(opts, text="Modèle installé :").grid(row=0, column=0, padx=(10, 5), pady=8, sticky="w")
        self.model_combo = ttk.Combobox(opts, textvariable=self.model, state="readonly", width=26)
        self.model_combo.grid(row=0, column=1, sticky="ew")
        self.refresh_btn = ttk.Button(opts, text="Actualiser", command=self.refresh_models)
        self.refresh_btn.grid(row=0, column=2, padx=6)
        ttk.Label(opts, text="Style :").grid(row=0, column=3, padx=5, sticky="e")
        self.style_combo = ttk.Combobox(opts, textvariable=self.style, values=list(STYLES), state="readonly", width=22)
        self.style_combo.grid(row=0, column=4, padx=(0, 10), sticky="ew")
        ttk.Label(opts, text="Intensité :").grid(row=1, column=0, padx=(10, 5), pady=8, sticky="w")
        self.intensity_combo = ttk.Combobox(opts, textvariable=self.intensity, values=list(INTENSITIES), state="readonly")
        self.intensity_combo.grid(row=1, column=1, sticky="ew")
        ttk.Label(opts, text="Installer un modèle :").grid(row=1, column=2, padx=6, sticky="e")
        self.download_combo = ttk.Combobox(opts, textvariable=self.download_model,
                                           values=RECOMMENDED_MODELS, width=23)
        self.download_combo.grid(row=1, column=3, sticky="ew", padx=6)
        self.pull_btn = ttk.Button(opts, text="Télécharger", command=self.pull_model)
        self.pull_btn.grid(row=1, column=4, padx=(3, 10))

        ttk.Label(opts, text="Température :").grid(row=2, column=0, padx=(10, 5), pady=8, sticky="w")
        ttk.Spinbox(opts, textvariable=self.temperature, from_=0.0, to=2.0, increment=0.05,
                   width=6, justify="center").grid(row=2, column=1, sticky="w")
        ttk.Label(opts, text="top_p :").grid(row=2, column=2, padx=6, sticky="e")
        ttk.Spinbox(opts, textvariable=self.top_p, from_=0.0, to=1.0, increment=0.05,
                   width=6, justify="center").grid(row=2, column=3, sticky="w", padx=6)
        ttk.Label(opts, text="repeat_penalty :").grid(row=3, column=0, padx=(10, 5), pady=8, sticky="w")
        ttk.Spinbox(opts, textvariable=self.repeat_penalty, from_=0.5, to=2.0, increment=0.05,
                   width=6, justify="center").grid(row=3, column=1, sticky="w")
        ttk.Label(opts, text="num_predict :").grid(row=3, column=2, padx=6, sticky="e")
        ttk.Spinbox(opts, textvariable=self.num_predict, from_=256, to=16384, increment=256,
                   width=7, justify="center").grid(row=3, column=3, sticky="w", padx=6)

        self.tabs = ttk.Notebook(self)
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=12, pady=5)
        self.tab_text = ttk.Frame(self.tabs)
        self.tab_file = ttk.Frame(self.tabs)
        self.tab_compare = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_text, text="Texte libre")
        self.tabs.add(self.tab_file, text="Fichier Markdown")
        self.tabs.add(self.tab_compare, text="Comparateur")
        self._tab_text()
        self._tab_file()
        self._tab_compare()

        foot = ttk.Frame(self)
        foot.grid(row=2, column=0, sticky="ew", padx=12, pady=(3, 12))
        foot.columnconfigure(0, weight=1)
        ttk.Label(foot, textvariable=self.status).grid(row=0, column=0, sticky="w", pady=4)
        self.cancel_btn = ttk.Button(foot, text="Annuler", command=self.cancel, state="disabled")
        self.cancel_btn.grid(row=0, column=1, padx=8)
        self.progress = ttk.Progressbar(foot, mode="determinate", maximum=100)
        self.progress.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _create_editors(self, parent: ttk.Frame, first: str, second: str):
        paned = ttk.Panedwindow(parent, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=6)
        widgets = []
        for heading in (first, second):
            group = ttk.LabelFrame(paned, text=heading)
            editor = tk.Text(group, wrap="word", undo=True, font=("Segoe UI", 11),
                             padx=10, pady=10, borderwidth=0)
            scroll = ttk.Scrollbar(group, orient="vertical", command=editor.yview)
            editor.configure(yscrollcommand=scroll.set)
            scroll.pack(side="right", fill="y")
            editor.pack(side="left", fill="both", expand=True)
            paned.add(group, weight=1)
            widgets.append(editor)
        return widgets[0], widgets[1]

    def _tab_text(self):
        self.tab_text.columnconfigure(0, weight=1)
        self.tab_text.rowconfigure(0, weight=1)
        frame = ttk.Frame(self.tab_text)
        frame.grid(row=0, column=0, sticky="nsew")
        self.text_input, self.text_output = self._create_editors(frame, "Texte original (français)", "Proposition réécrite")
        bar = ttk.Frame(self.tab_text)
        bar.grid(row=1, column=0, sticky="ew", padx=8, pady=7)
        self.rewrite_text_btn = ttk.Button(bar, text="Réécrire le texte", command=self.rewrite_text)
        self.rewrite_text_btn.pack(side="left")
        ttk.Button(bar, text="Copier le résultat", command=self.copy_text).pack(side="left", padx=10)
        ttk.Button(bar, text="Comparer les textes", command=lambda: self.open_comparator(
            self.text_input.get("1.0", "end-1c"), self.text_output.get("1.0", "end-1c"))).pack(side="left")
        ttk.Label(bar, text="Tu peux modifier le résultat avant de le copier.").pack(side="right")

    def _tab_file(self):
        self.tab_file.columnconfigure(0, weight=1)
        self.tab_file.rowconfigure(2, weight=1)
        src = ttk.Frame(self.tab_file)
        src.grid(row=0, column=0, sticky="ew", padx=8, pady=(10, 3))
        src.columnconfigure(1, weight=1)
        ttk.Label(src, text="Fichier source :").grid(row=0, column=0, padx=(0, 6))
        ttk.Entry(src, textvariable=self.source_path).grid(row=0, column=1, sticky="ew")
        self.browse_btn = ttk.Button(src, text="Parcourir…", command=self.choose_markdown)
        self.browse_btn.grid(row=0, column=2, padx=5)
        dest = ttk.Frame(self.tab_file)
        dest.grid(row=1, column=0, sticky="ew", padx=8, pady=(3, 5))
        dest.columnconfigure(1, weight=1)
        ttk.Label(dest, text="Fichier de sortie :").grid(row=0, column=0, padx=(0, 6))
        ttk.Entry(dest, textvariable=self.target_path).grid(row=0, column=1, sticky="ew")
        self.target_btn = ttk.Button(dest, text="Choisir…", command=self.choose_destination)
        self.target_btn.grid(row=0, column=2, padx=5)
        preview = ttk.Frame(self.tab_file)
        preview.grid(row=2, column=0, sticky="nsew")
        self.file_input, self.file_output = self._create_editors(preview, "Source Markdown", "Résultat Markdown")
        self.file_input.config(state="disabled")
        bottom = ttk.Frame(self.tab_file)
        bottom.grid(row=3, column=0, sticky="ew", padx=8, pady=(3, 8))
        self.rewrite_file_btn = ttk.Button(bottom, text="Réécrire et enregistrer un nouveau .md", command=self.rewrite_file)
        self.rewrite_file_btn.pack(side="left")
        ttk.Button(bottom, text="Enregistrer les modifications du résultat…", command=self.save_preview).pack(side="left", padx=10)
        ttk.Button(bottom, text="Comparer les textes", command=lambda: self.open_comparator(
            self.file_input.get("1.0", "end-1c"), self.file_output.get("1.0", "end-1c"))).pack(side="left")

    DIFF_COLORS = {"del": "#ffd6d6", "add": "#d6f5d9", "repl": "#ffe6b3"}

    def _tab_compare(self):
        self.tab_compare.columnconfigure(0, weight=1)
        self.tab_compare.rowconfigure(2, weight=1)
        self.diff_enabled = tk.BooleanVar(value=True)
        self._diff_guard = False

        stats_bar = ttk.Frame(self.tab_compare)
        stats_bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(10, 3))
        self.stats_var = tk.StringVar(value="Colle un texte original et sa réécriture, puis clique sur « Comparer les textes ».")
        ttk.Label(stats_bar, textvariable=self.stats_var).pack(side="left")

        bar = ttk.Frame(self.tab_compare)
        bar.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 5))
        self.compare_btn = ttk.Button(bar, text="Comparer les textes", command=self.run_diff)
        self.compare_btn.pack(side="left")
        ttk.Checkbutton(bar, text="Surlignage activé", variable=self.diff_enabled,
                        command=self._apply_highlight_visibility).pack(side="left", padx=12)
        ttk.Button(bar, text="Copier le texte réécrit (sans surlignage)",
                  command=self.copy_compare_rewritten).pack(side="left")

        paned = ttk.Panedwindow(self.tab_compare, orient="horizontal")
        paned.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        container = ttk.Frame(paned)
        paned.add(container, weight=1)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(1, weight=1)
        ttk.Label(container, text="Texte original").grid(row=0, column=0, sticky="w", padx=(10, 0))
        ttk.Label(container, text="Texte réécrit").grid(row=0, column=1, sticky="w", padx=(10, 0))

        self.compare_original = tk.Text(container, wrap="word", undo=True, font=("Segoe UI", 11),
                                        padx=10, pady=10, borderwidth=0)
        self.compare_rewritten = tk.Text(container, wrap="word", undo=True, font=("Segoe UI", 11),
                                         padx=10, pady=10, borderwidth=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self._compare_scroll_both)
        self.compare_original.configure(yscrollcommand=self._make_compare_yscroll(self.compare_rewritten, scrollbar))
        self.compare_rewritten.configure(yscrollcommand=self._make_compare_yscroll(self.compare_original, scrollbar))
        self.compare_original.grid(row=1, column=0, sticky="nsew")
        self.compare_rewritten.grid(row=1, column=1, sticky="nsew")
        scrollbar.grid(row=1, column=2, sticky="ns")

        for widget in (self.compare_original, self.compare_rewritten):
            widget.tag_configure("del", background=self.DIFF_COLORS["del"])
            widget.tag_configure("add", background=self.DIFF_COLORS["add"])
            widget.tag_configure("repl", background=self.DIFF_COLORS["repl"])

    def _compare_scroll_both(self, *args):
        self.compare_original.yview(*args)
        self.compare_rewritten.yview(*args)

    def _make_compare_yscroll(self, other: tk.Text, scrollbar: ttk.Scrollbar):
        def _set(first, last):
            scrollbar.set(first, last)
            if not self._diff_guard:
                self._diff_guard = True
                other.yview_moveto(first)
                self._diff_guard = False
        return _set

    def open_comparator(self, original: str, rewritten: str):
        self.compare_original.delete("1.0", "end")
        self.compare_original.insert("1.0", original)
        self.compare_rewritten.delete("1.0", "end")
        self.compare_rewritten.insert("1.0", rewritten)
        self.tabs.select(self.tab_compare)
        if original.strip() and rewritten.strip():
            self.run_diff()

    def run_diff(self):
        original = self.compare_original.get("1.0", "end-1c")
        rewritten = self.compare_rewritten.get("1.0", "end-1c")
        if not original.strip() or not rewritten.strip():
            messagebox.showwarning("Textes manquants", "Renseigne le texte original et le texte réécrit avant de comparer.")
            return
        self.stats_var.set("Comparaison en cours…")
        self._run(self._diff_worker, original, rewritten)

    def _diff_worker(self, original: str, rewritten: str):
        ops, stats = compute_diff(original, rewritten)
        self.jobs.put(("diff_result", (ops, stats)))

    def _render_diff(self, ops: list[DiffOp], stats: DiffStats):
        left, right = self.compare_original, self.compare_rewritten
        left.delete("1.0", "end")
        right.delete("1.0", "end")
        for op in ops:
            if op.tag == "equal":
                left.insert("end", op.original)
                right.insert("end", op.rewritten)
            elif op.tag == "delete":
                left.insert("end", op.original, ("del",))
            elif op.tag == "insert":
                right.insert("end", op.rewritten, ("add",))
            else:  # replace
                left.insert("end", op.original, ("repl",))
                right.insert("end", op.rewritten, ("repl",))
        self._apply_highlight_visibility()
        self.stats_var.set(
            f"Mots modifiés : {stats.percent_changed} % — "
            f"Ajoutés : {stats.words_added} — Supprimés : {stats.words_deleted} — Remplacés : {stats.words_replaced}"
        )

    def _apply_highlight_visibility(self):
        active = self.diff_enabled.get()
        for widget in (self.compare_original, self.compare_rewritten):
            for tag, color in self.DIFF_COLORS.items():
                widget.tag_configure(tag, background=color if active else "")

    def copy_compare_rewritten(self):
        result = self.compare_rewritten.get("1.0", "end-1c")
        if result:
            self.clipboard_clear()
            self.clipboard_append(result)
            self._status("Texte réécrit copié dans le presse-papiers (sans surlignage).")

    def _status(self, text: str):
        self.status.set(text)

    def _busy(self, value: bool):
        self.busy = value
        state = "disabled" if value else "normal"
        for w in (self.refresh_btn, self.pull_btn, self.rewrite_text_btn,
                  self.rewrite_file_btn, self.browse_btn, self.target_btn, self.compare_btn):
            w.configure(state=state)
        self.model_combo.configure(state="disabled" if value else "readonly")
        self.cancel_btn.configure(state="normal" if value else "disabled")
        if not value:
            self.progress.stop()
            self.progress.configure(mode="determinate")

    def _run(self, worker, *args):
        if self.busy:
            return
        self.cancel_event = threading.Event()
        self._busy(True)
        threading.Thread(target=self._worker, args=(worker, args), daemon=True).start()

    def _worker(self, worker, args):
        try:
            worker(*args)
        except Cancelled:
            self.jobs.put(("cancelled", None))
        except Exception as exc:
            self.jobs.put(("error", str(exc)))
        finally:
            self.jobs.put(("done", None))

    def _poll(self):
        try:
            while True:
                kind, data = self.jobs.get_nowait()
                if kind == "status":
                    self._status(data)
                elif kind == "models":
                    self.models = data
                    self.model_combo["values"] = data
                    current = self.model.get()
                    if current not in data:
                        best = next((x for x in RECOMMENDED_MODELS if x in data), data[0] if data else "")
                        self.model.set(best)
                    if not data:
                        self._status("Aucun modèle installé. Télécharge qwen3:14b ou qwen3:8b.")
                    else:
                        self._status(f"Ollama disponible — {len(data)} modèle(s) installé(s).")
                elif kind == "progress":
                    self.progress.configure(mode="determinate", value=data)
                elif kind == "text_chunk":
                    self.text_output.insert("end", data)
                    self.text_output.see("end")
                elif kind == "file_preview":
                    self.file_output.insert("end", data)
                    self.file_output.see("end")
                elif kind == "file_finish":
                    path, warnings = data
                    self._status(f"Terminé. Nouveau fichier : {path}" + (f" — {len(warnings)} avertissement(s)" if warnings else ""))
                    if warnings:
                        messagebox.showwarning("Vérification Markdown", "Certaines portions ont été conservées à l'identique pour éviter une altération :\n\n" + "\n".join(warnings[:15]))
                    else:
                        messagebox.showinfo("Fichier enregistré", f"Réécriture enregistrée dans :\n{path}")
                elif kind == "text_finish":
                    self._status("Réécriture terminée. Tu peux relire et copier le résultat.")
                    self.progress.configure(value=100)
                elif kind == "diff_result":
                    ops, stats = data
                    self._render_diff(ops, stats)
                elif kind == "cancelled":
                    self._status("Opération annulée. Aucun fichier incomplet n'a été enregistré.")
                elif kind == "error":
                    self._status("Erreur : " + data)
                    messagebox.showerror("Erreur", data)
                elif kind == "done":
                    self._busy(False)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def refresh_models(self):
        self._run(self._refresh_worker)

    def _refresh_worker(self):
        names = installed_models()
        self.jobs.put(("models", names))

    def pull_model(self):
        name = self.download_model.get().strip()
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.:/-]{0,120}", name):
            messagebox.showerror("Nom invalide", "Indique un identifiant de modèle Ollama valide.")
            return
        if not messagebox.askyesno("Télécharger un modèle", f"Télécharger « {name} » avec Ollama ?\n\nCela utilise Internet et plusieurs gigaoctets d'espace disque."):
            return
        self._run(self._pull_worker, name)

    def _pull_worker(self, name: str):
        self.jobs.put(("status", f"Téléchargement de {name}…"))
        for item in api_stream("/api/pull", {"name": name, "stream": True}, self.cancel_event):
            total, completed = item.get("total"), item.get("completed")
            if total:
                pct = max(0, min(100, round(100 * (completed or 0) / total)))
                self.jobs.put(("progress", pct))
                self.jobs.put(("status", f"{name} — {item.get('status', 'téléchargement')} : {pct} %"))
            else:
                self.jobs.put(("status", f"{name} — {item.get('status', 'en cours')}"))
        self.jobs.put(("models", installed_models()))

    def _selected_model(self) -> str | None:
        name = self.model.get()
        if not name or name not in self.models:
            messagebox.showwarning("Modèle manquant", "Choisis un modèle installé, ou télécharge-en un.")
            return None
        return name

    def _read_ollama_options(self) -> dict | None:
        try:
            temperature = float(self.temperature.get())
            top_p = float(self.top_p.get())
            repeat_penalty = float(self.repeat_penalty.get())
            num_predict = int(float(self.num_predict.get()))
            if not (0.0 <= temperature <= 2.0):
                raise ValueError("température hors de [0, 2]")
            if not (0.0 <= top_p <= 1.0):
                raise ValueError("top_p hors de [0, 1]")
            if not (0.1 <= repeat_penalty <= 3.0):
                raise ValueError("repeat_penalty hors de [0.1, 3]")
            if not (1 <= num_predict <= 32768):
                raise ValueError("num_predict hors de [1, 32768]")
        except ValueError as exc:
            messagebox.showerror("Paramètres Ollama invalides", f"Vérifie température, top_p, repeat_penalty et num_predict ({exc}).")
            return None
        return {"temperature": temperature, "top_p": top_p, "repeat_penalty": repeat_penalty, "num_predict": num_predict}

    def rewrite_text(self):
        model = self._selected_model()
        if not model:
            return
        options = self._read_ollama_options()
        if options is None:
            return
        source = self.text_input.get("1.0", "end-1c")
        if not source.strip():
            messagebox.showwarning("Texte absent", "Saisis d'abord un texte en français.")
            return
        self.text_output.delete("1.0", "end")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self._run(self._text_worker, model, source, self.style.get(), self.intensity.get(), options)

    def _text_worker(self, model, source, style, intensity, options):
        self.jobs.put(("status", f"Réécriture en cours avec {model}…"))
        # Traite également les longs collages sans déborder la fenêtre de contexte.
        for number, part in enumerate(split_long_piece(source, cap=9000)):
            if self.cancel_event.is_set():
                raise Cancelled()
            if number:
                self.jobs.put(("text_chunk", "\n\n"))
            for delta in rewrite_plain_stream(model, part, style, intensity, self.cancel_event, options):
                self.jobs.put(("text_chunk", delta))
        self.jobs.put(("text_finish", None))

    def copy_text(self):
        result = self.text_output.get("1.0", "end-1c")
        if result:
            self.clipboard_clear()
            self.clipboard_append(result)
            self._status("Résultat copié dans le presse-papiers.")

    def choose_markdown(self):
        choice = filedialog.askopenfilename(title="Choisir un fichier Markdown",
                                           filetypes=[("Fichiers Markdown", "*.md *.markdown"), ("Tous les fichiers", "*.*")])
        if not choice:
            return
        source = Path(choice)
        self.source_path.set(str(source))
        self.target_path.set(str(source.with_name(source.stem + "_reecrit" + source.suffix)))
        try:
            self._preview_source(read_utf8_preserve_newlines(source))
        except (OSError, UnicodeError) as exc:
            messagebox.showerror("Lecture impossible", str(exc))

    def _preview_source(self, contents):
        self.file_input.config(state="normal")
        self.file_input.delete("1.0", "end")
        self.file_input.insert("1.0", contents)
        self.file_input.config(state="disabled")
        self.file_output.delete("1.0", "end")

    def choose_destination(self):
        choice = filedialog.asksaveasfilename(title="Fichier Markdown de sortie", defaultextension=".md",
                                              filetypes=[("Markdown", "*.md"), ("Tous les fichiers", "*.*")],
                                              initialfile=Path(self.target_path.get()).name if self.target_path.get() else "texte_reecrit.md")
        if choice:
            self.target_path.set(choice)

    def rewrite_file(self):
        model = self._selected_model()
        if not model:
            return
        options = self._read_ollama_options()
        if options is None:
            return
        source = Path(self.source_path.get())
        target_str = self.target_path.get().strip()
        if not source.is_file() or not target_str:
            messagebox.showwarning("Fichier manquant", "Choisis un fichier source et une destination.")
            return
        target = Path(target_str)
        if source.resolve() == target.resolve():
            messagebox.showerror("Sécurité", "La sortie doit être différente de la source : l'original ne sera pas écrasé.")
            return
        if target.exists() and not messagebox.askyesno("Fichier existant", f"Le fichier de sortie existe déjà :\n{target}\n\nLe remplacer ?"):
            return
        try:
            contents = read_utf8_preserve_newlines(source)
        except (OSError, UnicodeError) as exc:
            messagebox.showerror("Lecture impossible", str(exc))
            return
        self._preview_source(contents)
        self.progress.configure(mode="determinate", value=0)
        self._run(self._file_worker, model, contents, target, self.style.get(), self.intensity.get(), options)

    def _file_worker(self, model: str, contents: str, target: Path, style: str, intensity: str, options: dict):
        newline = "\r\n" if contents.count("\r\n") > contents.count("\n") // 2 else "\n"
        contents = contents.replace("\r\n", "\n").replace("\r", "\n")
        segments = prepare_segments(contents)
        groups = group_indices(segments)
        warnings: list[str] = []
        for number, indices in enumerate(groups, start=1):
            if self.cancel_event.is_set():
                raise Cancelled()
            self.jobs.put(("status", f"{model} — groupe {number}/{len(groups)} ({len(indices)} fragments)…"))
            try:
                generated, alerts = rewrite_markdown_group(model, segments, indices, style, intensity, self.cancel_event, options)
            except ValueError as exc:
                warnings.append(f"Groupe {number} conservé ({exc}).")
                generated = [segments[idx].text for idx in indices]
                alerts = []
            warnings.extend(alerts)
            for idx, content in zip(indices, generated):
                segments[idx].text = content
            self.jobs.put(("progress", round(100 * number / len(groups))))
        if self.cancel_event.is_set():
            raise Cancelled()
        result = "".join(seg.text for seg in segments)
        # Toujours écrire seulement à la fin : pas de fichier partiellement réécrit.
        atomic_save(target, result, newline)
        self.jobs.put(("file_preview", result))
        self.jobs.put(("file_finish", (str(target), warnings)))

    def save_preview(self):
        text = self.file_output.get("1.0", "end-1c")
        if not text:
            messagebox.showwarning("Résultat vide", "Aucun résultat à enregistrer.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".md", filetypes=[("Markdown", "*.md")],
                                            title="Enregistrer le Markdown corrigé")
        if path:
            source = self.source_path.get().strip()
            if source and Path(path).resolve() == Path(source).resolve():
                messagebox.showerror("Sécurité", "L’original ne peut pas être écrasé. Choisis un autre nom.")
                return
            try:
                atomic_save(Path(path), text)
            except OSError as exc:
                messagebox.showerror("Enregistrement impossible", str(exc))
            else:
                self._status("Résultat enregistré : " + path)

    def cancel(self):
        self.cancel_event.set()
        self._status("Annulation demandée…")


def main():
    app = RewriteApp()
    app.mainloop()


if __name__ == "__main__":
    main()
