"""Sensitivity analysis: clinical fluency score on min(duration, 60 s) instead of 45 s.

Same rules as 11_clinical_scores. Writes results/cache/19_window60/clinical_60s.csv (column valid_per_s_60), used by
configs/model_clinical_v2_60s.yaml. The test set is scored for the cache only and never evaluated."""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("m11", ROOT / "scripts" / "11_clinical_scores.py")
m11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m11)

CACHE = ROOT / "results" / "cache" / "19_window60"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(ROOT / "results" / "cache" / "11_clinical" / "clinical_per_recording.csv", dtype={"participant_id": str})
    f = pd.read_csv(ROOT / "results" / "cache" / "05_features" / "features_per_recording.csv",
                    dtype={"participant_id": str}).set_index(["participant_id", "task"])
    rows = []
    for r in c.itertuples():
        segs = json.loads(m11.ts_path(r.participant_id, r.task).read_text(encoding="utf-8"))
        w = min(f.loc[(r.participant_id, r.task), "duration_s"], 60.0)
        v, _, _ = m11.score(r.task, m11.timed_words(segs, w))
        rows.append({"participant_id": r.participant_id, "task": r.task, "valid_per_s_60": v / w})
    pd.DataFrame(rows).to_csv(CACHE / "clinical_60s.csv", index=False)
    print("ok", len(rows))


if __name__ == "__main__":
    main()
