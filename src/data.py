"""Chargement unique de PROCESS-2 pour tous les scripts.

Ne parcourt jamais data/ de façon récursive : seuls les dossiers participants du premier niveau
(PROCESS-2_rec__XXX) sont considérés, ce qui ignore le cache HF (.cache/huggingface/...).
N'ouvre ni les .wav ni les .txt : renvoie uniquement les métadonnées et les chemins.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "PROCESS-2"
METADATA_FILE = "meta-info.csv"
TASKS = ("SFT", "PFT", "CTD")
FLUENCY_TASKS = ("SFT", "PFT")
DIAGNOSES = ("HC", "MCI", "Dementia")
DIR_PREFIX = "PROCESS-2_rec__"

EXPECTED_N = 400
EXPECTED_SPLIT = {"Train": {"Dementia": 40, "MCI": 120, "HC": 160},
                  "Test": {"Dementia": 10, "MCI": 30, "HC": 40}}


def read_metadata(data_dir=DATA_DIR):
    """meta-info.csv avec IDs → participant_id et Split normalisé en Train/Test."""
    meta = pd.read_csv(Path(data_dir) / METADATA_FILE)
    meta = meta.rename(columns={"IDs": "participant_id"})
    meta["participant_id"] = meta["participant_id"].astype(str)
    meta["Split"] = meta["Split"].astype(str).str.strip().str.title()
    return meta


def participant_dirs(data_dir=DATA_DIR):
    """Dossiers participants du premier niveau uniquement (dossiers cachés ignorés)."""
    return sorted(p for p in Path(data_dir).iterdir()
                  if p.is_dir() and not p.name.startswith(".") and p.name.startswith(DIR_PREFIX))


def load_dataset(data_dir=DATA_DIR, check=True):
    """Une ligne par participant : métadonnées + chemins wav_<TÂCHE> et txt_<TÂCHE>.

    Avec check=True, lève ValueError si la structure ou les effectifs diffèrent de l'attendu
    (400 participants ; Train 320 / Test 80 ; 40/120/160 et 10/30/40 Dementia/MCI/HC ; 6 fichiers chacun).
    """
    data_dir = Path(data_dir)
    meta = read_metadata(data_dir)
    dirs = {d.name: d for d in participant_dirs(data_dir)}
    for t in TASKS:
        for ext in ("wav", "txt"):
            meta[f"{ext}_{t}"] = [dirs[p] / f"{p}__{t}.{ext}" if p in dirs else None
                                  for p in meta["participant_id"]]
    if check:
        _check(meta, dirs)
    return meta


def recordings(df):
    """Format long : une ligne par (participant, tâche) avec le chemin du wav."""
    keep = [c for c in df.columns if not c.startswith(("wav_", "txt_"))]
    rows = df.melt(id_vars=keep, value_vars=[f"wav_{t}" for t in TASKS],
                   var_name="task", value_name="wav")
    rows["task"] = rows["task"].str.removeprefix("wav_")
    return rows.sort_values(["participant_id", "task"]).reset_index(drop=True)


def _check(meta, dirs):
    errors = []
    if len(meta) != EXPECTED_N or len(dirs) != EXPECTED_N:
        errors.append(f"{len(meta)} lignes de métadonnées / {len(dirs)} dossiers (attendu {EXPECTED_N})")
    if meta["participant_id"].duplicated().any():
        errors.append(f"{meta['participant_id'].duplicated().sum()} identifiants dupliqués")
    ids = set(meta["participant_id"])
    if ids != set(dirs):
        errors.append(f"{len(ids - set(dirs))} lignes sans dossier, {len(set(dirs) - ids)} dossiers sans ligne")
    counts = pd.crosstab(meta["Split"], meta["diagnosis"])
    for split, expected in EXPECTED_SPLIT.items():
        for diag, n in expected.items():
            got = int(counts.loc[split, diag]) if split in counts.index and diag in counts.columns else 0
            if got != n:
                errors.append(f"{split}/{diag} : {got} (attendu {n})")
    path_cols = [c for c in meta.columns if c.startswith(("wav_", "txt_"))]
    n_missing = sum(1 for c in path_cols for p in meta[c] if p is None or not p.exists())
    if n_missing:
        errors.append(f"{n_missing} fichiers wav/txt manquants")
    if errors:
        raise ValueError("Structure PROCESS-2 inattendue : " + " ; ".join(errors))
