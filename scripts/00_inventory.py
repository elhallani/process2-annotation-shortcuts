"""Étape 0a — Inventaire structurel de PROCESS-2.

Vérifie la structure de data/PROCESS-2/ sans jamais afficher de donnée individuelle.
Toutes les sorties sont agrégées (comptes, percentiles) et écrites dans results/00_inventory/.
Les transcriptions (*.txt) ne sont jamais ouvertes : seules leur présence et leur taille non nulle
sont vérifiées. Les .wav ne sont lus que via leur en-tête (durée, fréquence, canaux).
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import DATA_DIR as DATA, METADATA_FILE, read_metadata  # noqa: E402

OUT = ROOT / "results" / "00_inventory"

CONFIG = {
    "data_dir": str(DATA.relative_to(ROOT)),
    "tasks": ["SFT", "PFT", "CTD"],
    "extensions": ["wav", "txt"],
    "fluency_tasks": ["SFT", "PFT"],
    "min_fluency_duration_s": 45.0,
    "percentiles": [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100],
    "dir_pattern": r"^PROCESS-2_rec__\d+$",
    "expected": {
        "n_participants": 400,
        "split_counts": {"Train": {"Dementia": 40, "MCI": 120, "HC": 160},
                         "Test": {"Dementia": 10, "MCI": 30, "HC": 40}},
        "metadata_columns": ["participant_id", "diagnosis", "age", "gender", "MMSE", "Split"],
        "sample_rate": 16000,
        "channels": 1,
        "duration_range_s": [8.6, 261.0],
    },
}


def check_dirs():
    """Inventaire des dossiers participants et des fichiers attendus (comptes uniquement)."""
    pattern = re.compile(CONFIG["dir_pattern"])
    all_dirs = sorted(p for p in DATA.iterdir() if p.is_dir())
    part_dirs = [d for d in all_dirs if pattern.match(d.name)]
    top_files = [p for p in DATA.iterdir() if p.is_file()]

    rows = []
    n_unexpected_files = 0
    for d in part_dirs:
        expected = {f"{d.name}__{t}.{e}" for t in CONFIG["tasks"] for e in CONFIG["extensions"]}
        present = {p.name for p in d.iterdir() if p.is_file()}
        n_unexpected_files += len(present - expected)
        row = {"participant_id": d.name}
        for t in CONFIG["tasks"]:
            for e in CONFIG["extensions"]:
                f = d / f"{d.name}__{t}.{e}"
                row[f"{t}_{e}_present"] = f.exists()
                row[f"{t}_{e}_nonempty"] = f.exists() and f.stat().st_size > 0
        rows.append(row)
    files = pd.DataFrame(rows)

    summary = {
        "n_subdirs_total": len(all_dirs),
        "n_participant_dirs": len(part_dirs),
        "n_subdirs_not_matching_pattern": len(all_dirs) - len(part_dirs),
        "top_level_files": sorted(p.suffix or "<no ext>" for p in top_files),
        "n_unexpected_files_in_participant_dirs": n_unexpected_files,
    }
    pres_cols = [c for c in files.columns if c.endswith("_present")]
    summary["n_participants_with_all_6_files"] = int(files[pres_cols].all(axis=1).sum())
    summary["missing_per_file_type"] = {c.replace("_present", ""): int((~files[c]).sum()) for c in pres_cols}
    summary["empty_per_file_type"] = {
        c.replace("_nonempty", ""): int((files[c.replace("_nonempty", "_present")] & ~files[c]).sum())
        for c in files.columns if c.endswith("_nonempty")
    }
    return files, summary


def check_metadata(participant_ids):
    meta = read_metadata(DATA)
    s = {
        "file": METADATA_FILE,
        "n_rows": len(meta),
        "columns_raw": list(pd.read_csv(DATA / METADATA_FILE, nrows=0).columns),
        "columns": list(meta.columns),
        "dtypes": {c: str(t) for c, t in meta.dtypes.items()},
        "missing_per_column": {c: int(v) for c, v in meta.isna().sum().items()},
        "expected_columns_absent": [c for c in CONFIG["expected"]["metadata_columns"] if c not in meta.columns],
    }
    if "participant_id" in meta.columns:
        ids = set(meta["participant_id"].astype(str))
        s["n_duplicate_participant_id"] = int(meta["participant_id"].duplicated().sum())
        s["n_meta_rows_without_folder"] = len(ids - set(participant_ids))
        s["n_folders_without_meta_row"] = len(set(participant_ids) - ids)
    for c in ["diagnosis", "gender", "Split"]:
        if c in meta.columns:
            s[f"{c}_levels"] = {str(k): int(v) for k, v in meta[c].value_counts(dropna=False).items()}
    return meta, s


def crosstabs(meta):
    ct = (meta.groupby(["Split", "diagnosis", "gender"], dropna=False).size()
          .rename("n").reset_index())
    ct_wide = pd.crosstab([meta["Split"], meta["diagnosis"]], meta["gender"], margins=True)
    ct_wide.to_csv(OUT / "counts_split_diagnosis_gender_wide.csv")
    ct.to_csv(OUT / "counts_split_diagnosis_gender.csv", index=False)

    # % d'hommes et disponibilité MMSE / âge par groupe (agrégés)
    g = meta.groupby(["Split", "diagnosis"], dropna=False)
    agg = pd.DataFrame({
        "n": g.size(),
        "pct_male": g["gender"].apply(lambda x: 100 * x.astype(str).str.lower().str.startswith("m").mean()),
        "age_mean": g["age"].mean(),
        "age_sd": g["age"].std(),
        "age_missing": g["age"].apply(lambda x: int(x.isna().sum())),
        "mmse_available_pct": g["MMSE"].apply(lambda x: 100 * x.notna().mean()),
        "mmse_n": g["MMSE"].count(),
        "mmse_mean": g["MMSE"].mean(),
        "mmse_sd": g["MMSE"].std(),
        "mmse_min": g["MMSE"].min(),
        "mmse_max": g["MMSE"].max(),
        # article (tab. 4) : HC train 27,57 ± 6,38 → valeur proche de 0 suspectée
        "mmse_n_below_10": g["MMSE"].apply(lambda x: int((x < 10).sum())),
    }).round(2)
    agg.to_csv(OUT / "group_summary.csv")

    split_diag = pd.crosstab(meta["Split"], meta["diagnosis"])
    exp = pd.DataFrame(CONFIG["expected"]["split_counts"]).T
    diff = (split_diag.reindex_like(exp).fillna(0).astype(int) - exp)
    return ct_wide, agg, split_diag, diff


def audio_durations(files, meta):
    rows = []
    for name in files["participant_id"]:
        for t in CONFIG["tasks"]:
            f = DATA / name / f"{name}__{t}.wav"
            if not f.exists():
                continue
            try:
                info = sf.info(str(f))
                rows.append({"participant_id": name, "task": t, "duration_s": info.duration,
                             "sr": info.samplerate, "channels": info.channels, "readable": True})
            except Exception:
                rows.append({"participant_id": name, "task": t, "duration_s": np.nan,
                             "sr": np.nan, "channels": np.nan, "readable": False})
    dur = pd.DataFrame(rows)
    dur = dur.merge(meta[["participant_id", "diagnosis", "Split"]], on="participant_id", how="left")

    pct = CONFIG["percentiles"]
    def pctl(x):
        x = x.dropna()
        return pd.Series({f"p{p}": np.percentile(x, p) for p in pct} | {"n": len(x), "mean": x.mean()})
    by_task = dur.groupby("task")["duration_s"].apply(pctl).unstack().round(2)
    by_task.to_csv(OUT / "duration_percentiles_by_task.csv")
    by_task_diag = dur.groupby(["task", "diagnosis"])["duration_s"].apply(pctl).unstack().round(2)
    by_task_diag.to_csv(OUT / "duration_percentiles_by_task_diagnosis.csv")

    thr = CONFIG["min_fluency_duration_s"]
    flu = dur[dur["task"].isin(CONFIG["fluency_tasks"])].copy()
    flu["short"] = flu["duration_s"] < thr
    short_tab = pd.crosstab([flu["task"]], [flu["Split"], flu["diagnosis"]],
                            values=flu["short"], aggfunc="sum", margins=True).fillna(0).astype(int)
    short_tab.to_csv(OUT / f"n_fluency_under_{int(thr)}s.csv")
    part_short = flu.groupby("participant_id")["short"].any()

    audio = {
        "n_wav_readable": int(dur["readable"].sum()),
        "n_wav_unreadable": int((~dur["readable"]).sum()),
        "sample_rates": {str(int(k)): int(v) for k, v in dur["sr"].dropna().value_counts().items()},
        "channels": {str(int(k)): int(v) for k, v in dur["channels"].dropna().value_counts().items()},
        "duration_min_s": round(float(dur["duration_s"].min()), 2),
        "duration_max_s": round(float(dur["duration_s"].max()), 2),
        "n_fluency_recordings_under_threshold": {t: int(flu.loc[flu.task == t, "short"].sum())
                                                 for t in CONFIG["fluency_tasks"]},
        "n_participants_with_any_fluency_under_threshold": int(part_short.sum()),
        "n_participants_with_both_fluencies_under_threshold":
            int(flu.groupby("participant_id")["short"].all().sum()),
    }
    return by_task, short_tab, audio


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files, dir_summary = check_dirs()
    meta, meta_summary = check_metadata(files["participant_id"].tolist())
    ct_wide, agg, split_diag, diff = crosstabs(meta)
    by_task, short_tab, audio = audio_durations(files, meta)

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "structure": dir_summary,
        "metadata": meta_summary,
        "split_x_diagnosis": split_diag.to_dict(),
        "split_x_diagnosis_minus_expected": diff.to_dict(),
        "audio": audio,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False))

    # Affichage console : agrégats uniquement
    pd.set_option("display.width", 160)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print("\n== Split x diagnostic x sexe ==\n", ct_wide)
    print("\n== Résumé par groupe ==\n", agg)
    print("\n== Durées par tâche (s) ==\n", by_task)
    print(f"\n== Fluences < {CONFIG['min_fluency_duration_s']} s ==\n", short_tab)
    print(f"\nSorties : {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    sys.exit(main())
