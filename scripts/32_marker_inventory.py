"""Aggregated inventory of transcription markers: speaker labels and annotations in (), [], <>, {}.

All 1200 transcripts, loaded through load_dataset(); no participant text is displayed. Protection under the data use
agreement: a form is named only if it occurs in at least MIN_FILES distinct files, digits are replaced by "N", and
rarer forms are only counted.
Outputs: results/32_marker_inventory/ (counts by diagnosis)."""

import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import TASKS, load_dataset  # noqa: E402
from src.text import PREFIX, read_text  # noqa: E402

OUT = ROOT / "results" / "32_marker_inventory"
MIN_FILES = 5
BRACKETS = {"( )": r"\(([^)\n]{0,80})\)", "[ ]": r"\[([^\]\n]{0,80})\]", "< >": r"<([^>\n]{1,40})>",
            "{ }": r"\{([^}\n]{0,80})\}"}
# catégories fixées a priori (sur la forme normalisée)
CATEGORIES = [("pause chronométrée", r"^N(\.N)?\s*(s|sec|secs|second|seconds)$"),
              ("pause non chronométrée", r"^(pause|long pause|silence|short pause)$"),
              ("événement non verbal", r"^(sighs?|laughs?|laughter|tuts?|coughs?|breath|breathes|clears throat|"
                                       r"sniffs?|yawns?|groans?|hums?|whistles?|clicks? tongue)$"),
              ("bruit / environnement", r"^(buzzer.*|noise|background noise|phone.*|beep.*|.*sounds?)$"),
              ("incertitude / inaudible", r"^(\?+|inaudible|unclear|unintelligible|.*\?)$")]


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"\d+", "N", s.strip().lower()))


def category(form):
    for name, p in CATEGORIES:
        if re.match(p, form):
            return name
    return "autre"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_dataset()
    pref_files, pref_occ = defaultdict(set), defaultdict(int)
    ann_files, ann_occ = defaultdict(set), defaultdict(int)
    n_files = 0
    for p in meta.itertuples():
        for t in TASKS:
            fid = (p.participant_id, t, p.diagnosis, p.Split)
            text = read_text(getattr(p, f"txt_{t}"))
            n_files += 1
            for line in text.splitlines():
                m = PREFIX.match(line)
                if m:
                    k = norm(m.group(1)) + ":"
                    pref_files[k].add(fid)
                    pref_occ[k] += 1
            for bname, rx in BRACKETS.items():
                for inner in re.findall(rx, text):
                    k = (bname, norm(inner))
                    ann_files[k].add(fid)
                    ann_occ[k] += 1

    # taxonomie : % de fichiers de chaque groupe contenant au moins un marqueur de la catégorie (fichiers uniques)
    n_grp = {g: 3 * int((meta.diagnosis == g).sum()) for g in ("HC", "MCI", "Dementia")}
    tax = defaultdict(set)
    for k, fs in pref_files.items():
        tax["1 Speaker labels: " + ("participant (Pat:)" if k == "pat:" else "other speaker (Oth:)" if k == "oth:"
                                    else "rare label")] |= fs
    for (bname, form), fs in ann_files.items():
        c = category(form) if len(fs) >= MIN_FILES else "rare"
        name = {"pause chronométrée": "2 Timed pauses", "événement non verbal": "3 Non-verbal vocal events",
                "incertitude / inaudible": "4 Uncertain or inaudible speech",
                "bruit / environnement": "5 Recording procedure or environment",
                "pause non chronométrée": "2 Untimed pauses", "rare": "6 Rare forms (not named)",
                "autre": "6 Other"}[c]
        tax[name] |= fs
    trows = []
    for name, fs in sorted(tax.items()):
        r = {"catégorie": name, "n_fichiers": len(fs)}
        for g in ("HC", "MCI", "Dementia"):
            r[f"pct_fichiers_{g}"] = round(100 * sum(1 for f in fs if f[2] == g) / n_grp[g], 1)
        trows.append(r)
    pd.DataFrame(trows).to_csv(OUT / "taxonomy_by_group.csv", index=False)
    print(pd.DataFrame(trows).to_string(index=False), "\n")

    def table(files, occ, key_cols):
        rows, rare_forms, rare_occ = [], 0, 0
        for k, fs in files.items():
            if len(fs) < MIN_FILES:
                rare_forms += 1
                rare_occ += occ[k]
                continue
            r = dict(zip(key_cols, k if isinstance(k, tuple) else (k,)))
            r |= {"n_occurrences": occ[k], "n_fichiers": len(fs), "n_participants": len({f[0] for f in fs})}
            for g in ("HC", "MCI", "Dementia"):
                r[f"fichiers_{g}"] = sum(1 for f in fs if f[2] == g)
            for task in TASKS:
                r[f"fichiers_{task}"] = sum(1 for f in fs if f[1] == task)
            rows.append(r)
        return pd.DataFrame(rows).sort_values("n_fichiers", ascending=False), rare_forms, rare_occ

    P, p_rare, p_rare_occ = table(pref_files, pref_occ, ["préfixe"])
    A, a_rare, a_rare_occ = table(ann_files, ann_occ, ["crochets", "forme"])
    A["catégorie"] = A.forme.map(category)
    cat = A.groupby("catégorie")[["n_occurrences", "n_fichiers"]].sum().sort_values("n_occurrences", ascending=False)
    P.to_csv(OUT / "prefixes.csv", index=False)
    A.to_csv(OUT / "annotations.csv", index=False)
    cat.to_csv(OUT / "annotation_categories.csv")
    summ = {"generated": datetime.now().isoformat(timespec="seconds"), "n_fichiers": n_files,
            "seuil_nommage_fichiers": MIN_FILES, "préfixes_rares_non_nommés": {"formes": p_rare, "occurrences": p_rare_occ},
            "annotations_rares_non_nommées": {"formes": a_rare, "occurrences": a_rare_occ}}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    # affichage : les formes « autre » (mots possibles entre crochets) ne sont jamais imprimées, seulement comptées
    cols = ["crochets", "forme", "catégorie", "n_occurrences", "n_fichiers", "fichiers_HC", "fichiers_MCI",
            "fichiers_Dementia"]
    print(P.to_string(index=False), "\n\n", A[A.catégorie != "autre"][cols].to_string(index=False),
          f"\n\nformes « autre » (non affichées) : {int((A.catégorie == 'autre').sum())} formes, "
          f"{int(A[A.catégorie == 'autre'].n_occurrences.sum())} occurrences\n\n", cat.to_string(), "\n",
          json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
