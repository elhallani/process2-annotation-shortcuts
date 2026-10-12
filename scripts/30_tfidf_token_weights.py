"""The TF-IDF + logistic-regression probe as a transparent instrument: weight of annotation tokens.

For each task, the probe (configuration of 04_text_artifact) is fitted on raw manual transcripts in each of the 25
training folds. A token is an annotation token when at least 90% of its training occurrences fall inside (), [], <>
or {} (rule fixed in advance). Aggregated outputs: share of the absolute weight carried by annotation tokens,
annotation tokens among the 20 strongest weights, and median rank and sign of selected annotation tokens.
Only annotation tokens are named, never content words."""

import importlib.util
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import TASKS, load_dataset  # noqa: E402
from src.text import ANNOTATIONS, raw, read_text  # noqa: E402

spec = importlib.util.spec_from_file_location("t04", ROOT / "scripts" / "04_text_artifact.py")
t04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t04)
OUT = ROOT / "results" / "30_tfidf_token_weights"
TOK = re.compile(r"(?u)\b\w\w+\b")   # motif par défaut de TfidfVectorizer
CLASSES = {"HC": 0, "MCI": 1, "Dementia": 2}
NAMED = ["seconds", "second", "sighs", "laughs", "tuts", "buzzer", "pat"]


def annotation_vocab(texts):
    inside, total = Counter(), Counter()
    for x in texts:
        low = x.lower()
        total.update(TOK.findall(low))
        for p in ANNOTATIONS:
            for m in p.findall(low):
                inside.update(TOK.findall(m))
    return {t for t, n in total.items() if inside[t] / n >= .9}, total


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr = load_dataset().query("Split == 'Train'").reset_index(drop=True)
    y3 = tr.diagnosis.map(CLASSES).to_numpy()
    yb = (y3 > 0).astype(int)
    splits = t04.folds(y3, 5, 5)
    rows, named = [], []
    for t in TASKS:
        texts = [raw(read_text(p)) for p in tr[f"txt_{t}"]]
        ann, total = annotation_vocab(texts)
        X = np.array(texts, dtype=object)
        for r, rep in enumerate(splits):
            for k, (trn, _) in enumerate(rep):
                m = t04.tfidf_model(False).fit(X[trn], yb[trn])
                vocab = m.named_steps["tfidfvectorizer"].get_feature_names_out()
                w = m.named_steps["logisticregression"].coef_[0]
                is_ann = np.array([v in ann for v in vocab])
                is_num = np.array([v.isdigit() for v in vocab])
                order = np.argsort(-np.abs(w))
                rank = np.empty(len(w), int)
                rank[order] = np.arange(1, len(w) + 1)
                top = order[:20]
                rows.append({"task": t, "fold": f"{r}.{k}", "n_vocab": len(vocab),
                             "pct_tokens_annotation": 100 * is_ann.mean(),
                             "pct_masse_coef_annotation": 100 * np.abs(w[is_ann]).sum() / np.abs(w).sum(),
                             "pct_masse_coef_nombres": 100 * np.abs(w[is_num]).sum() / np.abs(w).sum(),
                             "n_annotation_top20": int(is_ann[top].sum()),
                             "n_annotation_top20_risque_plus": int((is_ann[top] & (w[top] > 0)).sum())})
                for v in NAMED:
                    if v in ann and v in set(vocab):
                        i = int(np.where(vocab == v)[0][0])
                        named.append({"task": t, "token": v, "rang": int(rank[i]), "coef": float(w[i]),
                                      "n_vocab": len(vocab)})
        print(f"  {t} fait", flush=True)
    d = pd.DataFrame(rows)
    summ = d.groupby("task").agg(n_vocab=("n_vocab", "median"), pct_tokens_annotation=("pct_tokens_annotation", "mean"),
                                 pct_masse_coef_annotation=("pct_masse_coef_annotation", "mean"),
                                 pct_masse_coef_nombres=("pct_masse_coef_nombres", "mean"),
                                 n_annotation_top20=("n_annotation_top20", "mean"),
                                 n_annotation_top20_risque_plus=("n_annotation_top20_risque_plus", "mean")).round(2)
    nm = pd.DataFrame(named)
    nm = nm.groupby(["task", "token"]).agg(n_folds=("rang", "size"), rang_median=("rang", "median"),
                                          coef_median=("coef", "median"), pct_folds_coef_positif=(
                                              "coef", lambda c: 100 * (c > 0).mean()),
                                          n_vocab=("n_vocab", "median")).round(3).reset_index()
    summ.to_csv(OUT / "annotation_weight_share_train.csv")
    nm.to_csv(OUT / "named_annotation_tokens_train.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                                                  "statut": "train uniquement, 25 folds d'entraînement",
                                                  "règle_annotation": ">= 90 % des occurrences entre (), [], <>, {}",
                                                  "coef_positif": "augmente le risque (MCI+Dementia)"}, indent=2,
                                                 ensure_ascii=False))
    pd.set_option("display.width", 200)
    print(summ.to_string(), "\n\n", nm.to_string(index=False))


if __name__ == "__main__":
    main()
