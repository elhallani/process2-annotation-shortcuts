"""Nettoyage des transcriptions PROCESS-2 (règle CLAUDE.md : retirer préfixes, marques et annotations ;
ne garder que les énoncés du participant). Ne jamais afficher le texte produit."""

import re
from pathlib import Path

PREFIX = re.compile(r"^\s*\*?([A-Za-z][A-Za-z0-9 _.\-]{0,15}):\s*")
ANNOTATIONS = [re.compile(p) for p in (r"\([^)\n]{0,80}\)", r"\[[^\]\n]{0,80}\]", r"<[^>\n]{1,40}>",
                                       r"\{[^}\n]{0,80}\}")]
PARTICIPANT_LABELS = {"pat", "patient", "participant", "p", "par", "subject", "subj", "s", "sub"}


def read_text(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")


def official_clean(text):
    """Reproduction exacte du prétraitement de PROCESS2_BASELINE_class.py (l. 420-425), y compris la
    concaténation des lignes sans espace."""
    out = ""
    for line in text.split("\n"):
        if line.strip():
            out += line.split("\t")[-1].replace("Pat:", "").replace("(Buzzer sounds)", "").strip()
    return out


def raw(text):
    """Version brute : lignes jointes par un espace, rien d'autre."""
    return " ".join(l.strip() for l in text.split("\n") if l.strip())


def turns(text):
    """Liste (label ou None, contenu) ; une ligne sans préfixe hérite du label précédent."""
    out, current = [], None
    for line in text.split("\n"):
        if not line.strip():
            continue
        m = PREFIX.match(line)
        if m:
            current = m.group(1).strip().lower()
            line = line[m.end():]
        out.append((current, line.strip()))
    return out


def strip_marks(s):
    for p in ANNOTATIONS:
        s = p.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def no_marks(text):
    """Préfixes de locuteur et annotations retirés ; tous les énoncés gardés."""
    return strip_marks(" ".join(c for _, c in turns(text)))


def participant_label(tr):
    """Label du participant dans un fichier dialogué : label connu de participant, sinon le plus loquace."""
    labels = {l for l, _ in tr if l is not None}
    known = labels & PARTICIPANT_LABELS
    if len(known) == 1:
        return known.pop(), "known_label"
    words = {}
    for l, c in tr:
        if l is not None:
            words[l] = words.get(l, 0) + len(c.split())
    return max(words, key=words.get), "most_words"


def participant_only(text):
    """Comme no_marks, mais pour un fichier dialogué (≥ 2 labels distincts) ne garde que le participant.
    Renvoie (texte, règle utilisée)."""
    tr = turns(text)
    labels = {l for l, _ in tr if l is not None}
    if len(labels) < 2:
        return no_marks(text), "monologue"
    who, rule = participant_label(tr)
    return strip_marks(" ".join(c for l, c in tr if l == who or l is None)), rule


def clean_for_model(text):
    """Nettoyage obligatoire pour toute baseline texte du projet."""
    return participant_only(text)[0]
