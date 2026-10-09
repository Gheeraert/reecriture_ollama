# Atelier de réécriture — Ollama (local)

Une application Windows, en français, avec interface Tkinter, qui réécrit du texte français à l'aide d'un modèle Ollama fonctionnant **sur ton ordinateur**.

## 1. Installer les prérequis

- **Python 3.10 ou plus récent**, avec **Tkinter** (inclus dans l'installation standard de Python sous Windows).
- **Ollama pour Windows** : <https://ollama.com/download>.
- Lancer Ollama (il reste normalement actif dans la zone de notification Windows). Le programme contacte exclusivement **http://127.0.0.1:11434** pour la génération.
- Pas de `pip install` ni de `requirements.txt` : le programme utilise uniquement la bibliothèque standard de Python.

## 2. Lancer l'application

Double-clique sur `LANCER_WINDOWS.bat`, ou, depuis PowerShell dans le dossier du programme :

```powershell
py -3 reecriture_ollama.py
```

Si l'interface affiche « Ollama est inaccessible », lance d'abord Ollama. Dans certaines installations, tu peux aussi ouvrir un terminal et exécuter `ollama serve` (seulement si le serveur n'est pas déjà actif).

## 3. Installer et choisir un modèle

La liste « Modèle installé » est remplie automatiquement par Ollama. Pour en ajouter un, sélectionne ou saisis son nom dans « Installer un modèle », puis clique sur « Télécharger » : l'application indique la progression. **Le téléchargement d'un modèle utilise Internet.**

Suggestions pour une RTX 5070 Ti avec **16 Go de VRAM** :

| Modèle | Taille approximative du téléchargement | Usage conseillé |
|---|---:|---|
| `qwen3:14b` | 9,3 Go | Premier choix pour la reformulation française |
| `gemma3:12b` | 8,1 Go | Comparaison stylistique |
| `qwen3:8b` | 5,2 Go | Réécriture plus rapide |

La mémoire consommée en cours de génération est **supérieure** au poids du modèle sur disque. L'occupation VRAM réelle varie avec la fenêtre de contexte et la configuration d'Ollama.

Tu peux aussi télécharger un modèle depuis PowerShell :

```powershell
ollama pull qwen3:14b
```

## 4. Onglet « Texte libre »

Colle du texte français dans la colonne gauche. Choisis modèle, style et intensité ; clique sur « Réécrire le texte ». La réponse s'affiche progressivement à droite. Elle est modifiable et copiable.

Styles proposés : correction légère, reformulation fidèle, style universitaire, style littéraire et allègement du texte. Les très longs collages sont traités par morceaux et séparés par une ligne blanche : une relecture des jonctions est utile.

## 5. Onglet « Fichier Markdown »

Choisis un fichier `.md` ou `.markdown` et valide le fichier de sortie. Le nom proposé se termine par `_reecrit.md`. Le résultat est enregistré **uniquement si le traitement s'est achevé**. Le fichier source ne peut pas être sélectionné comme destination.

Le script :

- préserve sans les transmettre au modèle le frontmatter YAML initial, les blocs de code délimités par des accents graves ou tildes, les tableaux Markdown classiques, les commentaires HTML, certains blocs HTML, les formules délimitées par `$$`, les images seules, les définitions de références de liens et les notes définies en bas de page ;
- masque les liens Markdown en ligne, les références de liens, le code inline et les appels de notes avant de les envoyer au modèle, puis les rétablit à l'identique ;
- découpe les longs fichiers en groupes de fragments (environ 3 200 caractères au plus par groupe), pour limiter le contexte et maintenir une progression visible ;
- demande une réponse structurée en JSON, contrôle le nombre de fragments et les marqueurs Markdown essentiels ;
- conserve le fragment original lorsqu'un problème de structure est détecté, avec un avertissement à la fin.

**Limites :** cette sécurité est volontairement prudente mais ne constitue ni un parseur Markdown complet, ni une preuve de conservation des faits, des citations ou du style. Certains éléments complexes (HTML arbitraire, extensions Pandoc, tableaux atypiques, LaTeX avancé) pourront nécessiter une vérification manuelle. Il est recommandé de comparer l'original et la sortie avant publication, notamment pour une édition scientifique.

Le fichier généré est en UTF-8. Les fins de ligne Windows (`CRLF`) sont conservées si elles étaient utilisées dans la source.

## 6. Confidentialité et interruptions

- L'inférence s'effectue **localement** via l'API d'Ollama. Le programme n'envoie pas le contenu des documents à un service d'IA distant.
- Le téléchargement initial des modèles dépend de l'accès réseau à la bibliothèque Ollama.
- Le bouton « Annuler » interrompt le flux en cours ; pour le Markdown, aucun fichier final n'est écrit si l'opération est annulée avant l'enregistrement.
- La génération d'un grand Markdown peut prendre plusieurs minutes, selon le nombre de fragments et la vitesse du modèle.

## 7. Tests de base (facultatif)

```powershell
py -3 -m unittest -v test_reecriture.py
```

Les tests simulent les réponses d'Ollama : ils vérifient la conservation du Markdown, le masquage des liens, le traitement du JSON diffusé, la protection des titres, les textes longs et l'enregistrement sans écraser le fichier source. **Ils ne remplacent pas un essai réel avec Ollama**, qui n'était pas disponible dans l'environnement de création du programme.
