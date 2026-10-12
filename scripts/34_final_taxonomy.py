"""Final taxonomy of transcription markers (1200 files, training and test sets).

Category of each annotation form: the predefined rule for named forms (>= 5 files) and the majority vote of the three
local language models for rare forms; speaker labels added. Aggregated outputs: share of annotations by category,
and percentage of files and participants by group and split."""

import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import TASKS, load_dataset  # noqa: E402
from src.text import PREFIX, read_text  # noqa: E402

spec = importlib.util.spec_from_file_location("m33", ROOT / "scripts" / "33_llm_validate_taxonomy.py")
m33 = importlib.util.module_from_spec(spec)
saved, sys.argv = sys.argv, ["x"]
spec.loader.exec_module(m33)
sys.argv = saved
OUT = ROOT / "results" / "34_final_taxonomy"
NAMES = {"timed": "Timed pauses", "untimed": "Untimed pauses", "nonverbal": "Non-verbal vocal events",
         "uncertain": "Uncertain or inaudible speech", "procedure": "Recording procedure or environment",
         "word": "Other content"}
GROUPS = ("HC", "MCI", "Dementia")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    F = m33.forms()
    votes = pd.DataFrame({m["short"]: json.loads((m33.CACHE / f"{m['short']}.json").read_text())
                          for m in m33.m27.CFG["models"]})
    F["vote"] = votes.mode(axis=1)[0].values
    F["final"] = [r.règle if not r.rare else r.vote for r in F.itertuples()]
    cat_of = {(r.crochets, r.forme): r.final for r in F.itertuples()}
    meta = load_dataset()
    files = defaultdict(set)   # catégorie -> {(pid, task)}
    occ, occ_tr = Counter(), Counter()
    diag = dict(zip(meta.participant_id, meta.diagnosis))
    split = dict(zip(meta.participant_id, meta.Split))
    for p in meta.itertuples():
        for t in TASKS:
            text = read_text(getattr(p, f"txt_{t}"))
            for line in text.splitlines():
                m = PREFIX.match(line)
                if m:
                    lab = m.group(1).strip().lower()
                    k = "label_pat" if lab == "pat" else "label_oth" if lab == "oth" else "label_rare"
                    files[k].add((p.participant_id, t))
                    occ[k] += 1
                    occ_tr[k] += p.Split == "Train"
            for bname, rx in m32_brackets().items():
                for inner in re.findall(rx, text):
                    c = cat_of[(bname, m33.m32.norm(inner))]
                    files[c].add((p.participant_id, t))
                    occ[c] += 1
                    occ_tr[c] += p.Split == "Train"
    n_files = {(g, s): 3 * int(((meta.diagnosis == g) & (meta.Split == s)).sum()) for g in GROUPS
               for s in ("Train", "Test")}
    n_ann = sum(v for k, v in occ.items() if not k.startswith("label"))
    n_ann_tr = sum(v for k, v in occ_tr.items() if not k.startswith("label"))
    order = ["label_pat", "label_oth", "label_rare"] + list(NAMES)
    labels = {"label_pat": "Speaker label: participant (Pat:)", "label_oth": "Speaker label: other speaker (Oth:)",
              "label_rare": "Speaker label: rare"} | NAMES
    rows = []
    for k in order:
        fs = files.get(k, set())
        r = {"catégorie": labels[k], "occurrences": occ[k],
             "pct_des_annotations": None if k.startswith("label") else round(100 * occ[k] / n_ann, 1),
             "n_fichiers": len(fs), "occurrences_train": occ_tr[k],
             "pct_des_annotations_train": None if k.startswith("label") else round(100 * occ_tr[k] / n_ann_tr, 1)}
        for g in GROUPS:
            tot = n_files[(g, "Train")] + n_files[(g, "Test")]
            r[f"pct_fichiers_{g}"] = round(100 * sum(1 for f in fs if diag[f[0]] == g) / tot, 1)
            r[f"pct_participants_{g}"] = round(100 * len({f[0] for f in fs if diag[f[0]] == g})
                                               / (tot // 3), 1)
        for s in ("Train", "Test"):
            for g in GROUPS:
                r[f"pct_fichiers_{g}_{s}"] = round(100 * sum(1 for f in fs if diag[f[0]] == g and split[f[0]] == s)
                                                   / n_files[(g, s)], 1)
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "final_taxonomy.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"n_fichiers": 3 * len(meta), "n_annotations": n_ann,
                                                  "n_annotations_train": n_ann_tr,
                                                  "formes_rares_reclassées_par_vote_LLM": int(F.rare.sum())},
                                                 indent=2, ensure_ascii=False))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(T.to_string(index=False), f"\n\nannotations au total : {n_ann}")


def m32_brackets():
    return m33.m32.BRACKETS


if __name__ == "__main__":
    main()
