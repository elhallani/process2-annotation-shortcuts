"""Étape 1 — Mesures interprétables par tâche (définies dans configs/features.yaml).

Fenêtre min(durée, 45 s) ; unités = îlots Silero VAD (cache de 01_audit, mêmes paramètres) ;
F0 (contrôle) et noyaux syllabiques (CTD) via Praat / parselmouth.
Sorties : valeurs par enregistrement → results/cache/05_features/ (jamais affichées) ;
agrégats par tâche × diagnosis (train) → results/05_features/.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
import yaml
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import FLUENCY_TASKS, load_dataset, recordings  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs" / "features.yaml").read_text())
OUT = ROOT / "results" / "05_features"
CACHE = ROOT / "results" / "cache" / "05_features"
VAD = ROOT / "results" / "cache" / "01_audit" / "vad_segments.json"
N_JOBS = 6


def fcfg(group, name):
    return next(f for f in CFG["features"][group] if f["name"] == name)


def clip(segs, w):
    return [(s, min(e, w)) for s, e in segs if s < w]


def gaps(segs):
    return np.array([b[0] - a[1] for a, b in zip(segs[:-1], segs[1:])])


def f0_sd_semitones(snd, gender):
    p = fcfg("fluency", "f0_sd_semitones")
    pitch = snd.to_pitch(pitch_floor=p["pitch_floor_hz"][gender], pitch_ceiling=p["pitch_ceiling_hz"][gender])
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    return float(np.std(12 * np.log2(f0 / np.median(f0)))) if len(f0) >= 20 else np.nan


def syllable_nuclei(snd, segs):
    """de Jong & Wempe (2009), version simplifiée : pics d'intensité ≥ seuil (q99 − 25 dB), creux ≥ 2 dB
    avant le pic suivant, pic voisé (F0 > 0), dans une unité de parole."""
    inten = snd.to_intensity(minimum_pitch=50, time_step=0.01)
    t, db = inten.xs(), inten.values[0]
    thr = np.quantile(db, .99) - 25
    pitch = snd.to_pitch(time_step=0.01)
    peaks = [i for i in range(1, len(db) - 1) if db[i] > db[i - 1] and db[i] >= db[i + 1] and db[i] > thr]
    kept, last = [], None
    for i in peaks:
        if last is not None and db[last:i + 1].min() > min(db[last], db[i]) - 2:
            if db[i] > db[last]:
                kept[-1] = i
                last = i
            continue
        kept.append(i)
        last = i
    n = 0
    for i in kept:
        ti = t[i]
        if any(s <= ti <= e for s, e in segs) and pitch.get_value_at_time(ti) == pitch.get_value_at_time(ti):
            n += 1
    return n


def features(pid, task, wav, gender, segs_all):
    w_max = CFG["window"]["max_s"]
    y, sr = sf.read(str(wav), dtype="float32")
    dur = len(y) / sr
    w = min(dur, w_max)
    segs = clip(segs_all, w)
    snd = parselmouth.Sound(y[:int(w * sr)].astype(np.float64), sampling_frequency=sr)
    g = gaps(segs)
    r = {"participant_id": pid, "task": task, "duration_s": dur, "window_s": w, "short_recording": task in FLUENCY_TASKS and dur < w_max,
         "f0_sd_semitones": f0_sd_semitones(snd, gender)}
    if task in FLUENCY_TASKS:
        bin_s = fcfg("fluency", "units_per_bin")["bin_s"]
        r["units_per_s"] = len(segs) / w
        rates = []
        for b0 in np.arange(0, w, bin_s):
            b1 = min(b0 + bin_s, w)
            if b1 - b0 < 1.0:
                continue
            rates.append(sum(b0 <= s < b1 for s, _ in segs) / (b1 - b0))
        for k in range(int(w_max // bin_s)):
            r[f"units_per_s_bin{k + 1}"] = rates[k] if k < len(rates) else np.nan
        r["units_slope"] = float(np.polyfit(np.arange(len(rates)), rates, 1)[0]) if len(rates) >= 2 else np.nan
        r["median_gap_s"] = float(np.median(g)) if len(g) else np.nan
        r["long_silences_per_min"] = float((g > fcfg("fluency", "long_silences_per_min")["threshold_s"]).sum()) / (w / 60)
    else:
        r["pause_rate_per_min"] = float((g >= fcfg("ctd", "pause_rate_per_min")["pause_min_s"]).sum()) / (w / 60)
        phon = sum(e - s for s, e in segs)
        r["articulation_rate"] = syllable_nuclei(snd, segs) / phon if phon > 0 else np.nan
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = load_dataset()
    recs = recordings(meta)
    vad = json.loads(VAD.read_text())
    rows = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(features)(r.participant_id, r.task, r.wav, r.gender, vad[f"{r.participant_id}__{r.task}"])
        for r in recs.itertuples())
    df = pd.DataFrame(rows).merge(meta[["participant_id", "diagnosis", "Split", "gender", "age"]], on="participant_id")
    df.to_csv(CACHE / "features_per_recording.csv", index=False)
    num = [c for c in df.columns if df[c].dtype.kind == "f" and c not in ("duration_s", "window_s", "age")]
    tr = df[df.Split == "Train"]
    agg = tr.groupby(["task", "diagnosis"])[num].median().round(3)
    agg.to_csv(OUT / "median_by_task_diagnosis_train.csv")
    miss = df.groupby("task")[num].apply(lambda x: x.isna().sum())
    miss.to_csv(OUT / "missing_by_task.csv")
    summary = {"generated": datetime.now().isoformat(timespec="seconds"), "n_recordings": len(df),
               "n_short_recordings": int(df.short_recording.sum())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "config_used.yaml").write_text(yaml.safe_dump(CFG, allow_unicode=True, sort_keys=False))
    pd.set_option("display.width", 220)
    print(json.dumps(summary, indent=2), "\n", agg, "\n", miss)


if __name__ == "__main__":
    main()
