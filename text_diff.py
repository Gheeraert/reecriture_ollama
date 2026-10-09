#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comparaison fine de deux textes français, indépendante de Tkinter.

Le découpage se fait au niveau des mots (accents, apostrophes et traits
d'union compris) et de la ponctuation, en conservant les espaces et les
sauts de ligne afin de pouvoir reconstituer exactement le texte d'origine.
Fonctionne aussi bien sur du texte brut que sur des fragments Markdown : la
syntaxe Markdown est traitée comme de la ponctuation ordinaire et n'est
jamais modifiée par la comparaison elle-même.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

# Un « mot » peut contenir des apostrophes (droites ou typographiques) et des
# traits d'union internes : "aujourd'hui", "qu'il", "porte-manteau".
WORD_RE = re.compile(r"\w+(?:['’\-]\w+)*", re.UNICODE)
TOKEN_RE = re.compile(r"\w+(?:['’\-]\w+)*|[ \t]+|\n|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Découpe le texte en mots, ponctuation, espaces et sauts de ligne séparés."""
    return TOKEN_RE.findall(text)


def _is_word(token: str) -> bool:
    return bool(WORD_RE.fullmatch(token))


@dataclass
class DiffOp:
    tag: str  # "equal", "delete", "insert" ou "replace"
    original: str
    rewritten: str


@dataclass
class DiffStats:
    words_original: int
    words_rewritten: int
    words_added: int
    words_deleted: int
    words_replaced: int

    @property
    def percent_changed(self) -> float:
        """Part des mots d'origine supprimés ou remplacés (indicateur éditorial, pas une mesure de détection d'IA)."""
        total = max(self.words_original, 1)
        changed = self.words_deleted + self.words_replaced
        return round(100 * changed / total, 1)


def compute_diff(original: str, rewritten: str) -> tuple[list[DiffOp], DiffStats]:
    """Calcule les opérations de différence et les statistiques associées.

    `SequenceMatcher` repère les blocs communs indépendamment de leur position,
    ce qui limite (sans l'éliminer totalement) les faux positifs sur les
    passages simplement déplacés.
    """
    a = tokenize(original)
    b = tokenize(rewritten)
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    ops: list[DiffOp] = []
    added = deleted = replaced = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        seg_a = "".join(a[i1:i2])
        seg_b = "".join(b[j1:j2])
        if tag == "equal":
            ops.append(DiffOp("equal", seg_a, seg_b))
        elif tag == "delete":
            ops.append(DiffOp("delete", seg_a, ""))
            deleted += sum(1 for t in a[i1:i2] if _is_word(t))
        elif tag == "insert":
            ops.append(DiffOp("insert", "", seg_b))
            added += sum(1 for t in b[j1:j2] if _is_word(t))
        else:  # replace
            ops.append(DiffOp("replace", seg_a, seg_b))
            replaced += max(sum(1 for t in a[i1:i2] if _is_word(t)),
                            sum(1 for t in b[j1:j2] if _is_word(t)))
    stats = DiffStats(
        words_original=sum(1 for t in a if _is_word(t)),
        words_rewritten=sum(1 for t in b if _is_word(t)),
        words_added=added,
        words_deleted=deleted,
        words_replaced=replaced,
    )
    return ops, stats
