"""Clinical fluency score with openai/whisper-large-v3-turbo instead of whisper-medium (exploratory).

Reuses 11_clinical_scores unchanged, with another model and output folder; evaluated on the training set only.
Outputs: data/derived/asr_whisper_turbo_ts/ (not versioned); results/cache/22_whisper_turbo/clinical_turbo.csv;
results/22_whisper_turbo/validation_train.csv."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import FLUENCY_TASKS, load_dataset, recordings  # noqa: E402
from src.stats import icc  # noqa: E402

spec = importlib.util.spec_from_file_location("m11", ROOT / "scripts" / "11_clinical_scores.py")
m11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m11)
m11.CONFIG["model"] = "openai/whisper-large-v3-turbo"
m11.TS_DIR = ROOT / "data" / "derived" / "asr_whisper_turbo_ts"
OUT, CACHE = ROOT / "results" / "22_whisper_turbo", ROOT / "results" / "cache" / "22_whisper_turbo"


def main():
    for d in (m11.TS_DIR, OUT, CACHE):
        d.mkdir(parents=True, exist_ok=True)
    meta = load_dataset()
    recs = recordings(meta)
    recs = recs[recs.task.isin(FLUENCY_TASKS)]
    m11.transcribe(recs)
    f = pd.read_csv(ROOT / "results" / "cache" / "05_features" / "features_per_recording.csv",
                    dtype={"participant_id": str}).set_index(["participant_id", "task"])
    old = pd.read_csv(ROOT / "results" / "cache" / "11_clinical" / "clinical_per_recording.csv",
                      dtype={"participant_id": str}).set_index(["participant_id", "task"])
    rows = []
    for r in recs.itertuples():
        segs = json.loads(m11.ts_path(r.participant_id, r.task).read_text(encoding="utf-8"))
        w = min(f.loc[(r.participant_id, r.task), "duration_s"], 45.0)
        v45, _, _ = m11.score(r.task, m11.timed_words(segs, w))
        vfull, _, _ = m11.score(r.task, m11.timed_words(segs, np.inf))
        o = old.loc[(r.participant_id, r.task)]
        rows.append({"participant_id": r.participant_id, "task": r.task, "diagnosis": r.diagnosis, "Split": r.Split,
                     "valid_per_s_turbo": v45 / w, "valid_full_turbo": vfull, "valid_full_medium": o.valid_full_asr,
                     "valid_full_manual": o.valid_full_manual})
    df = pd.DataFrame(rows)
    df[["participant_id", "task", "valid_per_s_turbo"]].to_csv(CACHE / "clinical_turbo.csv", index=False)
    tr = df[df.Split == "Train"]
    val = []
    for t in FLUENCY_TASKS:
        s = tr[tr.task == t]
        y = (s.diagnosis != "HC").astype(int)
        for asr in ("medium", "turbo"):
            col = f"valid_full_{asr}"
            val.append({"task": t, "asr": asr, "ICC_vs_manual": icc(s[[col, "valid_full_manual"]].to_numpy(float))[0],
                        "spearman_vs_manual": spearmanr(s[col], s.valid_full_manual)[0],
                        "mean_diff_vs_manual": (s[col] - s.valid_full_manual).mean(),
                        "AUC_count_HC_vs_MCI+Dem": roc_auc_score(y, -s[col])})
    val = pd.DataFrame(val).round(3)
    val.to_csv(OUT / "validation_train.csv", index=False)
    print(val.to_string(index=False))


if __name__ == "__main__":
    main()
