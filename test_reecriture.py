"""Tests autonomes (sans Ollama et sans interface graphique)."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from reecriture_ollama import (
    Cancelled, Segment, atomic_save, group_indices, mask_inline, prepare_segments,
    read_utf8_preserve_newlines, rewrite_markdown_group,
    split_long_piece, structural_signature, unmask_inline,
)

SAMPLE = """---
title: Essai de Bossuet
date: 2026-10-09
---

# Une introduction

Voici le **premier** paragraphe, qui renvoie à [Port-Royal](https://example.org/port-royal) et à `une_fonction()`.

- Premier argument.
- Deuxième argument.

```python
x = 1
print(x)
```

| Nom | Date |
| --- | --- |
| Bossuet | 1682 |

## Conclusion

Une dernière proposition. Voici une note[^1].

[^1]: Une note infrapaginale conservée.
"""


class TestMarkdown(unittest.TestCase):
    def test_tokenize_roundtrip_and_protections(self):
        segments = prepare_segments(SAMPLE)
        self.assertEqual("".join(s.text for s in segments), SAMPLE)
        keep = "".join(s.text for s in segments if s.kind == "keep")
        self.assertIn("title: Essai", keep)
        self.assertIn("print(x)", keep)
        self.assertIn("Bossuet | 1682", keep)
        self.assertIn("[^1]: Une note", keep)
        self.assertGreater(len(group_indices(segments)), 0)

    def test_masks_links_code_and_notes(self):
        s = "Voici [Port-Royal](https://ex.fr), `code` et la note[^5]."
        masked, locks = mask_inline(s)
        self.assertNotIn("https://ex.fr", masked)
        self.assertEqual(unmask_inline(masked, locks), s)
        with self.assertRaises(ValueError):
            unmask_inline("Quelque chose sans marqueurs", locks)

    def test_group_streamed_json_and_reconstruction(self):
        segments = prepare_segments("# Titre\n\nMon paragraphe et [lien](https://ex.fr).\n")
        indices = [i for i, s in enumerate(segments) if s.kind == "rewrite"]

        def mock_stream(endpoint, payload, cancelled):
            self.assertEqual(endpoint, "/api/chat")
            self.assertFalse(payload["think"])
            self.assertTrue(payload["stream"])
            paragraphs = json.loads(payload["messages"][1]["content"].split("Voici les fragments :\n")[1])
            response = {"fragments": [p.replace("Titre", "Titre revu").replace("Mon paragraphe", "Un paragraphe") for p in paragraphs]}
            raw = json.dumps(response, ensure_ascii=False)
            yield {"message": {"content": raw[:len(raw)//2]}}
            yield {"message": {"content": raw[len(raw)//2:]}, "done": True}

        with patch("reecriture_ollama.api_stream", mock_stream):
            rewritten, warnings = rewrite_markdown_group("qwen3:14b", segments, indices,
                                                          "Reformulation fidèle", "Modérée", threading.Event())
        self.assertFalse(warnings)
        for idx, val in zip(indices, rewritten):
            segments[idx].text = val
        output = "".join(s.text for s in segments)
        self.assertIn("# Titre revu", output)
        self.assertIn("Un paragraphe", output)
        self.assertIn("[lien](https://ex.fr)", output)

    def test_structure_guard_retains_original(self):
        segments = [Segment("rewrite", "# Mon titre\n")]
        def mock_stream(*_args):
            yield {"message": {"content": json.dumps({"fragments": ["Mon titre réécrit"]})}}
        with patch("reecriture_ollama.api_stream", mock_stream):
            rewritten, warnings = rewrite_markdown_group("m", segments, [0],
                                                          "Style universitaire", "Modérée", threading.Event())
        self.assertEqual(rewritten[0], segments[0].text)
        self.assertTrue(warnings)

    def test_long_pieces_lossless(self):
        s = "Texte long. " * 900
        pieces = split_long_piece(s, cap=600)
        self.assertEqual("".join(pieces), s)
        self.assertTrue(all(len(x) <= 600 for x in pieces))

    def test_atomic_save_and_newlines(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "original.md"
            dest = Path(d) / "sortie.md"
            source.write_bytes(b"# Original\r\n\r\nDu texte.\r\n")
            self.assertIn("\r\n", read_utf8_preserve_newlines(source))
            atomic_save(dest, "# Nouveau\n\nTexte.\n", "\r\n")
            self.assertEqual(dest.read_bytes(), b"# Nouveau\r\n\r\nTexte.\r\n")
            self.assertIn("Original", source.read_text())

    def test_cancelled_without_network(self):
        with patch("reecriture_ollama.api_stream") as stream:
            ev = threading.Event()
            ev.set()
            with self.assertRaises(Cancelled):
                rewrite_markdown_group("m", [Segment("rewrite", "texte")], [0],
                                       "Style universitaire", "Modérée", ev)
            stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
