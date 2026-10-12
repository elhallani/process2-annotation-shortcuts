"""Fine-tuned DistilBERT with 5 x 5 cross-validation (training set).

Reuses 04_text_artifact unchanged (its hash is part of the test provenance) with five repetitions. Versions: raw
manual transcripts and transcripts with labels and annotations removed. Three-class head; binary score
1 - P(HC). Raw minus cleaned: corrected resampled t-test over 25 folds and paired permutation test."""

import importlib.util
import json
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

spec = importlib.util.spec_from_file_location("t04", ROOT / "scripts" / "04_text_artifact.py")
t04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t04)
t04.CONFIG["bert"]["n_repeats"] = 5
OUT = ROOT / "results" / "18_distilbert_5x5"
CACHE = ROOT / "results" / "cache" / "18_distilbert_5x5"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = load_dataset()
    meta = meta[meta.Split == "Train"].reset_index(drop=True)
    corpora, _ = t04.build_corpora(meta)
    b = t04.CONFIG["bert"]
    y3 = meta.diagnosis.map(t04.CLASSES).to_numpy()
    yb = (y3 > 0).astype(int)
    splits = t04.folds(y3, b["n_splits"], b["n_repeats"])
    rng = np.random.default_rng(t04.CONFIG["seed"])
    rows, diffs, log = [], [], []
    for t in TASKS:
        P3 = {}
        for v in b["versions"]:
            path = CACHE / f"P3_{t}_{v}.npy"
            if path.exists():
                P3[v] = np.load(path)
            else:
                P3[v] = t04.oof_bert(corpora[(t, v)], y3, splits, log)
                np.save(path, P3[v])
            print(f"  {t} {v} fait", flush=True)
        Pb = {v: np.stack([P3[v][..., 0], 1 - P3[v][..., 0]], -1) for v in b["versions"]}
        for tname, y, P, multi in [("HC vs MCI+Dementia", yb, Pb, False), ("3 classes", y3, P3, True)]:
            auc = {}
            for v in b["versions"]:
                a, bacc = t04.fold_metrics(y, P[v], multi, splits)
                auc[v] = a
                rows.append({"cible": tname, "task": t, "version": v, "AUC_mean": a.mean(), "AUC_sd": a.std(ddof=1),
                             "BAcc_mean": bacc.mean(), "BAcc_sd": bacc.std(ddof=1)})
            tval, p_nb = nadeau_bengio(auc["a_raw"] - auc["b_nomarks"], len(y) * .8, len(y) * .2)
            obs, p_perm = t04.paired_permutation(y, P["a_raw"], P["b_nomarks"], multi, rng, t04.CONFIG["n_permutations"])
            diffs.append({"cible": tname, "task": t, "dAUC_fold_mean": (auc["a_raw"] - auc["b_nomarks"]).mean(),
                          "p_NadeauBengio": p_nb, "dAUC_pooled": obs[0], "p_perm_pooled": p_perm[0]})
    res, dif = pd.DataFrame(rows).round(4), pd.DataFrame(diffs)
    for c in dif.cible.unique():
        m = dif.cible == c
        dif.loc[m, "p_NB_Holm"] = holm(dif.loc[m, "p_NadeauBengio"])
    dif = dif.round(4)
    res.to_csv(OUT / "results_train.csv", index=False)
    dif.to_csv(OUT / "paired_differences_train.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                                                  "config_bert": b, "n_new_fits": len(log),
                                                  "mean_s_per_fit": float(np.mean([l["seconds"] for l in log])) if log else None},
                                                 indent=2, ensure_ascii=False))
    pd.set_option("display.width", 200)
    print(res, "\n\n", dif)


if __name__ == "__main__":
    main()
