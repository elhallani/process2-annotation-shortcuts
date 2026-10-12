"""Channel bandwidth by sex and voice-identity shortcut.

A. 95% spectral roll-off by diagnosis and sex; device cut-off frequency estimated on non-speech regions
   (within-session ICC and association with diagnosis).
B. Voice-identity shortcut (training set): mean ECAPA embedding per participant and SFT - PFT difference,
   PCA fitted in the fold and weighted logistic regression; targets HC vs MCI+dementia, three classes, and sex
   (identity control).
Inputs: caches of 01_audit. Outputs: results/02_channel_identity/ (noise spectra per recording in results/cache/)."""

import json
import sys
from datetime import datetime
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy.stats import chi2, mannwhitneyu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import FLUENCY_TASKS, TASKS, load_dataset, recordings  # noqa: E402
from src.stats import icc_bootstrap, permutation_cv  # noqa: E402

OUT = ROOT / "results" / "02_channel_identity"
CACHE_IN = ROOT / "results" / "cache" / "01_audit"
CACHE = ROOT / "results" / "cache" / "02_channel_identity"

CONFIG = {
    "seed": 42,
    "n_jobs": 6,
    "stft": {"n_fft": 512, "win_length": 400, "hop_length": 160, "sr": 16000},
    "nonspeech_margin_s": 0.2,
    "cutoff": {
        "definition": "plus petite fréquence f ≥ 1 kHz telle que le spectre moyen du bruit (lissé) reste "
                      "≤ niveau de référence − drop_db pour toute fréquence ≥ f ; Nyquist si jamais atteint",
        "reference_band_hz": [200, 2000], "min_hz": 1000, "smooth_bins": 5,
        "drop_db": 20.0, "drop_db_grid": [10.0, 20.0, 30.0],
    },
    "identity": {"pca_components": 20, "n_splits": 5, "n_repeats": 5, "n_permutations": 1000,
                 "model": "StandardScaler + PCA(20) + LogisticRegression(class_weight=balanced)",
                 "embedding": "moyenne des unités ECAPA L2-normalisées par tâche ; participant = moyenne des 3 tâches"},
}


# ---------------------------------------------------------------- A. Fréquence de coupure

def noise_ltas(wav, segs):
    c = CONFIG["stft"]
    y, sr = sf.read(str(wav), dtype="float32")
    S = np.abs(librosa.stft(y, n_fft=c["n_fft"], hop_length=c["hop_length"], win_length=c["win_length"])) ** 2
    t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=c["hop_length"])
    near = np.zeros(len(t), bool)
    for s, e in segs:
        near |= (t >= s - CONFIG["nonspeech_margin_s"]) & (t <= e + CONFIG["nonspeech_margin_s"])
    far = ~near
    return 10 * np.log10(S[:, far].mean(1) + 1e-12) if far.sum() >= 10 else np.full(S.shape[0], np.nan)


def cutoff_hz(ltas_db, freqs, drop_db):
    c = CONFIG["cutoff"]
    if np.isnan(ltas_db).any():
        return np.nan
    sm_db = np.convolve(ltas_db, np.ones(c["smooth_bins"]) / c["smooth_bins"], mode="same")
    band = (freqs >= c["reference_band_hz"][0]) & (freqs <= c["reference_band_hz"][1])
    ref = np.median(sm_db[band])
    above = sm_db > ref - drop_db
    cand = np.where(freqs >= c["min_hz"])[0]
    # dernière fréquence encore au-dessus du seuil ; la coupure est la suivante
    last_above = cand[above[cand]].max() if above[cand].any() else cand[0] - 1
    return float(freqs[last_above + 1]) if last_above + 1 < len(freqs) else float(freqs[-1])


def stage_ltas(recs):
    path = CACHE / "noise_ltas.npz"
    if path.exists():
        z = np.load(path)
        return z["ltas"]
    vad = json.loads((CACHE_IN / "vad_segments.json").read_text())
    ltas = Parallel(n_jobs=CONFIG["n_jobs"], verbose=2)(
        delayed(noise_ltas)(r.wav, vad[f"{r.participant_id}__{r.task}"]) for r in recs.itertuples())
    ltas = np.vstack(ltas).astype(np.float32)
    np.savez(path, ltas=ltas, pid=recs.participant_id.to_numpy(str), task=recs.task.to_numpy(str))
    return ltas


def logit_report(df, var, extra=("male",)):
    """Logit Dementia vs HC : var (z) + covariables ; test du rapport de vraisemblance de var | covariables."""
    d = df.dropna(subset=[var]).copy()
    d["z"] = (d[var] - d[var].mean()) / d[var].std()
    y = (d.diagnosis == "Dementia").astype(float)
    full = sm.Logit(y, sm.add_constant(d[["z", *extra]])).fit(disp=0)
    red = sm.Logit(y, sm.add_constant(d[list(extra)])).fit(disp=0) if extra else None
    only = sm.Logit(y, sm.add_constant(d[["z"]])).fit(disp=0)
    lr = 2 * (full.llf - red.llf) if red is not None else np.nan
    ci = full.conf_int().loc["z"]
    return {"variable": var, "covariables": "+".join(extra) or "-", "n": len(d),
            "OR_per_SD_alone": np.exp(only.params["z"]), "p_alone": only.pvalues["z"],
            "OR_per_SD_adjusted": np.exp(full.params["z"]), "OR_lo": np.exp(ci[0]), "OR_hi": np.exp(ci[1]),
            "p_wald_adjusted": full.pvalues["z"], "LR_stat": lr, "p_LR": chi2.sf(lr, 1) if extra else np.nan}


def by_diag_sex(df, var):
    t = df.groupby(["diagnosis", "gender"])[var].agg(n="count", median="median",
                                                     q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75))
    tests = []
    for g in ["male", "female"]:
        a = df[(df.gender == g) & (df.diagnosis == "Dementia")][var].dropna()
        b = df[(df.gender == g) & (df.diagnosis == "HC")][var].dropna()
        tests.append({"variable": var, "gender": g, "n_Dementia": len(a), "n_HC": len(b),
                      "median_Dementia": a.median(), "median_HC": b.median(),
                      "p_MannWhitney_Dem_vs_HC": mannwhitneyu(a, b).pvalue})
    return t.round(1), pd.DataFrame(tests)


# ---------------------------------------------------------------- B. Identité vocale

def participant_embeddings():
    z = np.load(CACHE_IN / "embeddings.npz")
    e = z["emb"] / np.linalg.norm(z["emb"], axis=1, keepdims=True)
    df = pd.DataFrame({"participant_id": z["pid"], "task": z["task"]})
    task_mean = {}
    for (pid, task), idx in df.groupby(["participant_id", "task"]).indices.items():
        task_mean[(pid, task)] = e[idx].mean(0)
    pids = sorted({p for p, _ in task_mean})
    complete = [p for p in pids if all((p, t) in task_mean for t in TASKS)]
    mean_emb = pd.DataFrame({p: np.mean([task_mean[(p, t)] for t in TASKS], 0) for p in complete}).T
    diff = [p for p in pids if all((p, t) in task_mean for t in FLUENCY_TASKS)]
    diff_emb = pd.DataFrame({p: task_mean[(p, "SFT")] - task_mean[(p, "PFT")] for p in diff}).T
    return mean_emb, diff_emb, len(pids)


def make_identity_model():
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), PCA(n_components=CONFIG["identity"]["pca_components"],
                                               random_state=CONFIG["seed"]),
                         LogisticRegression(class_weight="balanced", max_iter=5000))


def identity_test(meta, mean_emb, diff_emb):
    tr = meta[meta.Split == "Train"].set_index("participant_id")
    rows = []
    for ename, E in [("ECAPA moyen (3 tâches)", mean_emb), ("ECAPA SFT − PFT", diff_emb)]:
        ids = [p for p in tr.index if p in E.index]
        X = E.loc[ids].to_numpy(float)
        d = tr.loc[ids]
        targets = {"HC vs MCI+Dementia": ((d.diagnosis != "HC").astype(int).to_numpy(), False),
                   "3 classes": (d.diagnosis.map({"HC": 0, "MCI": 1, "Dementia": 2}).to_numpy(), True),
                   "sexe (contrôle)": ((d.gender == "male").astype(int).to_numpy(), False)}
        for tname, (y, multi) in targets.items():
            r = permutation_cv(X, y, make_identity_model, multi, CONFIG["seed"],
                               CONFIG["identity"]["n_permutations"], CONFIG["n_jobs"])
            rows.append({"embedding": ename, "cible": tname, "n": len(ids), **r})
            print(f"  identité [{ename}] {tname}: AUC {r['AUC_mean']:.3f}, BAcc {r['BAcc_mean']:.3f}", flush=True)
    return pd.DataFrame(rows).round(4)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(CONFIG["seed"])
    meta = load_dataset()
    recs = recordings(meta)
    feats = pd.read_csv(CACHE_IN / "recording_features.csv")
    meta_cols = meta[["participant_id", "diagnosis", "Split", "gender", "age"]]

    print("A. spectres du bruit", flush=True)
    ltas = stage_ltas(recs)
    freqs = librosa.fft_frequencies(sr=CONFIG["stft"]["sr"], n_fft=CONFIG["stft"]["n_fft"])
    for drop in CONFIG["cutoff"]["drop_db_grid"]:
        recs[f"cutoff_{int(drop)}db_hz"] = [cutoff_hz(l, freqs, drop) for l in ltas]
    main_cut = f"cutoff_{int(CONFIG['cutoff']['drop_db'])}db_hz"
    rec = (recs[["participant_id", "task"] + [c for c in recs.columns if c.startswith("cutoff_")]]
           .merge(feats[["participant_id", "task", "bw95_hz"]], on=["participant_id", "task"]))
    rec.to_csv(CACHE / "recording_cutoff.csv", index=False)
    part = rec.groupby("participant_id")[[c for c in rec.columns if c.endswith("_hz")]].mean().reset_index()
    part = part.merge(meta_cols, on="participant_id")
    part["male"] = (part.gender == "male").astype(float)
    train = part[part.Split == "Train"]

    # A1. bande passante par diagnostic × sexe (train)
    bw_tab, bw_tests = by_diag_sex(train, "bw95_hz")
    cut_tab, cut_tests = by_diag_sex(train, main_cut)
    bw_task = (rec.merge(meta_cols).query("Split == 'Train'")
               .groupby(["task", "gender", "diagnosis"])["bw95_hz"].median().unstack("diagnosis").round(0))
    bw_tab.to_csv(OUT / "bw95_by_diagnosis_sex_train.csv")
    bw_task.to_csv(OUT / "bw95_by_task_sex_diagnosis_train.csv")
    cut_tab.to_csv(OUT / "cutoff_by_diagnosis_sex_train.csv")
    tests = pd.concat([bw_tests, cut_tests]).round(4)
    tests.to_csv(OUT / "dementia_vs_hc_within_sex_train.csv", index=False)

    # A2. logit Dementia vs HC (train)
    dh = train[train.diagnosis.isin(["Dementia", "HC"])]
    logit = pd.DataFrame([logit_report(dh, "bw95_hz"), logit_report(dh, "bw95_hz", ("male", "age")),
                          logit_report(dh, main_cut), logit_report(dh, main_cut, ("male", "age"))]).round(4)
    logit.to_csv(OUT / "logit_dementia_vs_hc_train.csv", index=False)

    # A3. coupure : distribution, ICC, lien au diagnostic
    tr_rec = rec[rec.participant_id.isin(train.participant_id)]
    icc_rows = []
    for v in [c for c in rec.columns if c.endswith("_hz")]:
        x = tr_rec.pivot(index="participant_id", columns="task", values=v)[list(TASKS)].dropna().to_numpy(float)
        icc_rows.append({"variable": v, "n_participants": len(x), **icc_bootstrap(x, rng)})
    icc_df = pd.DataFrame(icc_rows).round(3)
    icc_df.to_csv(OUT / "icc_train.csv", index=False)
    edges = [0, 3000, 3600, 4200, 5000, 6000, 7000, 7900, 8001]
    cut_bins = pd.cut(tr_rec[main_cut], edges, right=False)
    dist = pd.crosstab(cut_bins, tr_rec.task).join(
        pd.crosstab(cut_bins, tr_rec.merge(meta_cols).diagnosis, normalize="columns").mul(100).round(1)
        .add_prefix("pct_"))
    dist.to_csv(OUT / "cutoff_distribution_train.csv")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for d in ["HC", "MCI", "Dementia"]:
        ids = train.loc[train.diagnosis == d, "participant_id"]
        ax.hist(tr_rec.loc[tr_rec.participant_id.isin(ids), main_cut], bins=np.arange(1000, 8300, 250),
                density=True, histtype="step", label=d)
    ax.set_xlabel("fréquence de coupure du bruit (Hz)")
    ax.set_ylabel("densité")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cutoff_hist_train.png", dpi=150)
    plt.close(fig)

    # B. identité vocale
    print("B. identité vocale (train)", flush=True)
    mean_emb, diff_emb, n_emb = participant_embeddings()
    ident = identity_test(meta, mean_emb, diff_emb)
    ident.to_csv(OUT / "identity_shortcut_train.csv", index=False)

    summary = {"generated": datetime.now().isoformat(timespec="seconds"),
               "n_recordings_without_noise_spectrum": int(np.isnan(ltas).any(1).sum()),
               "n_participants_with_embeddings": n_emb,
               "n_participants_mean_embedding": len(mean_emb), "n_participants_diff_embedding": len(diff_emb)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False))

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for name, t in [("bw95 diag × sexe (train)", bw_tab), ("bw95 tâche × sexe × diag", bw_task),
                    ("coupure diag × sexe", cut_tab), ("Tests Dem vs HC par sexe", tests),
                    ("Logit Dem vs HC", logit), ("ICC", icc_df), ("Distribution coupure", dist),
                    ("Identité vocale", ident)]:
        print(f"\n== {name} ==\n{t}")


if __name__ == "__main__":
    main()
