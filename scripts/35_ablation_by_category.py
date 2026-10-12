"""Ablation par catégorie d'annotation (TRAIN uniquement, 5 × 5 CV, TF-IDF + LR de 04_text_artifact).

Référence : transcription manuelle, préfixes de locuteur retirés, toutes les annotations conservées.
Versions : retrait d'UNE catégorie (taxonomie finale 34 : règle pour les formes nommées, vote LLM pour les rares) ;
retrait de toutes les annotations sauf les pauses chronométrées ; retrait de toutes les annotations.
ΔAUC = référence − version (perte), Nadeau–Bengio, Holm par tâche. Aucun texte affiché ni écrit.
"""

import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import TASKS, load_dataset  # noqa: E402
from src.stats import holm, nadeau_bengio  # noqa: E402
from src.text import read_text, turns  # noqa: E402


def _load(name, path):
    saved, sys.argv = sys.argv, ["x"]
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = saved
    return mod


m33 = _load("m33", "scripts/33_llm_validate_taxonomy.py")
t04 = _load("t04", "scripts/04_text_artifact.py")
OUT = ROOT / "results" / "35_ablation_by_category"
CATS = ["timed", "nonverbal", "uncertain", "procedure", "untimed"]


def category_map():
    F = m33.forms()
    votes = pd.DataFrame({m["short"]: json.loads((m33.CACHE / f"{m['short']}.json").read_text())
                          for m in m33.m27.CFG["models"]})
    F["final"] = [r.règle if not r.rare else v for r, v in zip(F.itertuples(), votes.mode(axis=1)[0])]
    return {(r.crochets, r.forme): r.final for r in F.itertuples()}


def strip(text, cat_of, remove):
    s = " ".join(c for _, c in turns(text))          # préfixes de locuteur retirés, tous les énoncés gardés
    for bname, rx in m33.m32.BRACKETS.items():
        full = {"( )": r"\(([^)\n]{0,80})\)", "[ ]": r"\[([^\]\n]{0,80})\]", "< >": r"<([^>\n]{1,40})>",
                "{ }": r"\{([^}\n]{0,80})\}"}[bname]
        s = re.sub(full, lambda m: " " if cat_of.get((bname, m33.m32.norm(m.group(1)))) in remove else m.group(0), s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cat_of = category_map()
    tr = load_dataset().query("Split == 'Train'").reset_index(drop=True)
    y3 = tr.diagnosis.map({"HC": 0, "MCI": 1, "Dementia": 2}).to_numpy()
    yb = (y3 > 0).astype(int)
    splits = t04.folds(y3, 5, 5)
    versions = {"référence (toutes annotations)": set()} | {f"sans {c}": {c} for c in CATS} | \
               {"seulement pauses chronométrées": set(CATS) - {"timed"}, "sans aucune annotation": set(CATS)}

    def fold_auc(texts):
        X = np.array(texts, dtype=object)
        a = []
        for rep in splits:
            for trn, tst in rep:
                a.append(roc_auc_score(yb[tst], t04.tfidf_model(False).fit(X[trn], yb[trn]).predict_proba(X[tst])[:, 1]))
        return np.array(a)

    rows = []
    for t in TASKS:
        raw = [read_text(p) for p in tr[f"txt_{t}"]]
        A = {v: fold_auc([strip(x, cat_of, rm) for x in raw]) for v, rm in versions.items()}
        ref = A["référence (toutes annotations)"]
        task_rows = []
        for v, a in A.items():
            d = ref - a
            task_rows.append({"task": t, "version": v, "AUC": a.mean(), "AUC_sd": a.std(ddof=1), "perte_vs_ref": d.mean(),
                              "p_NB": np.nan if v.startswith("référence") else nadeau_bengio(d, 256, 64)[1]})
        d = pd.DataFrame(task_rows)
        m = d.p_NB.notna()
        d.loc[m, "p_Holm"] = holm(d.loc[m, "p_NB"])
        rows.append(d)
        print(f"  {t} fait", flush=True)
    R = pd.concat(rows).round(4)
    R.to_csv(OUT / "ablation_by_category_train.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                                                  "référence": "manuel, préfixes retirés, annotations conservées",
                                                  "statut": "train uniquement"}, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 200)
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
