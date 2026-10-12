"""Étape 0h — Signal texte « propre » : transcriptions manuelles vs Whisper (TRAIN uniquement).

Conditions (TF-IDF mots + LR pondérée, 5 × 5 folds identiques, comme 04_text_artifact) :
  m_raw        transcription manuelle brute
  m_clean      transcription manuelle nettoyée (src.text.clean_for_model : préfixes, annotations, participant seul)
  asr          Whisper medium (aucune annotation de transcripteur)
  asr_pauses   Whisper + pauses mesurées dans l'audio (pauses ≥ 1 s par minute, durée des pauses ≥ 1 s par minute)
Différences appariées testées par permutation (échange des prédictions hors-fold par participant).
Aucun texte affiché ni écrit.
"""

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import TASKS, load_dataset  # noqa: E402
from src.text import clean_for_model, raw, read_text  # noqa: E402

spec = importlib.util.spec_from_file_location("t04", ROOT / "scripts" / "04_text_artifact.py")
t04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t04)

OUT = ROOT / "results" / "08_text_asr"
ASR_DIR = ROOT / "data" / "derived" / "asr_whisper"
VAD = ROOT / "results" / "cache" / "01_audit" / "vad_segments.json"
CONFIG = {"seed": t04.CONFIG["seed"], "split": "Train", "n_splits": 5, "n_repeats": 5, "n_permutations": 1000,
          "tfidf": t04.CONFIG["tfidf"], "logreg": t04.CONFIG["logreg"],
          "pause_features": ["gaps_ge1s_per_min", "gaps_ge1s_s_per_min"],
          "comparisons": [["m_raw", "m_clean"], ["m_clean", "asr"], ["m_raw", "asr"], ["asr_pauses", "asr"]]}
CONDITIONS = ["m_raw", "m_clean", "asr", "asr_pauses"]


def pause_features(pid, task, vad, dur):
    segs = vad[f"{pid}__{task}"]
    gaps = np.array([b[0] - a[1] for a, b in zip(segs[:-1], segs[1:])])
    long = gaps[gaps >= 1.0]
    return len(long) / (dur / 60), long.sum() / (dur / 60)


def make_model(with_pauses):
    tf = CONFIG["tfidf"]
    vec = TfidfVectorizer(lowercase=tf["lowercase"], sublinear_tf=tf["sublinear_tf"], min_df=tf["min_df"],
                          ngram_range=tuple(tf["ngram_range"]))
    if not with_pauses:
        return make_pipeline(ColumnTransformer([("tfidf", vec, "text")]), LogisticRegression(**CONFIG["logreg"]))
    return make_pipeline(ColumnTransformer([("tfidf", vec, "text"),
                                            ("pauses", StandardScaler(), CONFIG["pause_features"])]),
                         LogisticRegression(**CONFIG["logreg"]))


def oof(X, y, with_pauses, splits):
    P = np.zeros((len(splits), len(y), len(np.unique(y))))
    for r, rep in enumerate(splits):
        for trn, tst in rep:
            P[r, tst] = make_model(with_pauses).fit(X.iloc[trn], y[trn]).predict_proba(X.iloc[tst])
    return P


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_dataset()
    meta = meta[meta.Split == CONFIG["split"]].reset_index(drop=True)
    feats = pd.read_csv(ROOT / "results" / "cache" / "01_audit" / "recording_features.csv",
                        dtype={"participant_id": str}).set_index(["participant_id", "task"])
    vad = json.loads(VAD.read_text())
    rng = np.random.default_rng(CONFIG["seed"])
    rows, diffs = [], []
    for t in TASKS:
        manual = [read_text(p) for p in meta[f"txt_{t}"]]
        asr = [(ASR_DIR / f"{pid}__{t}.txt").read_text(encoding="utf-8") for pid in meta.participant_id]
        pf = np.array([pause_features(pid, t, vad, feats.loc[(pid, t), "duration_s"]) for pid in meta.participant_id])
        base = pd.DataFrame({CONFIG["pause_features"][0]: pf[:, 0], CONFIG["pause_features"][1]: pf[:, 1]})
        X = {"m_raw": base.assign(text=[raw(x) for x in manual]),
             "m_clean": base.assign(text=[clean_for_model(x) for x in manual]),
             "asr": base.assign(text=asr), "asr_pauses": base.assign(text=asr)}
        for tname, (y, multi) in t04.targets(meta).items():
            splits = t04.folds(y, CONFIG["n_splits"], CONFIG["n_repeats"])
            P = {c: oof(X[c], y, c == "asr_pauses", splits) for c in CONDITIONS}
            for c in CONDITIONS:
                auc, bacc = t04.fold_metrics(y, P[c], multi, splits)
                rows.append({"cible": tname, "task": t, "condition": c, "AUC_mean": auc.mean(),
                             "AUC_sd": auc.std(ddof=1), "BAcc_mean": bacc.mean(), "BAcc_sd": bacc.std(ddof=1)})
            for c1, c2 in CONFIG["comparisons"]:
                obs, p = t04.paired_permutation(y, P[c1], P[c2], multi, rng, CONFIG["n_permutations"])
                diffs.append({"cible": tname, "task": t, "diff": f"{c1} − {c2}", "dAUC": obs[0], "p_dAUC": p[0],
                              "dBAcc": obs[1], "p_dBAcc": p[1]})
            print(f"  [{tname}] {t} : " + ", ".join(f"{r['condition']} {r['AUC_mean']:.3f}"
                                                    for r in rows[-len(CONDITIONS):]), flush=True)
    res, dif = pd.DataFrame(rows).round(4), pd.DataFrame(diffs).round(4)
    res.to_csv(OUT / "results_train.csv", index=False)
    dif.to_csv(OUT / "paired_differences_train.csv", index=False)
    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False))
    (OUT / "summary.json").write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                                                  "n_train": len(meta)}, indent=2))
    pd.set_option("display.width", 220)
    print(res, "\n", dif)


if __name__ == "__main__":
    main()
