"""Étape 0b — Audit : VAD, début/fin des enregistrements, empreinte du canal, seconde voix, test de raccourci.

Sorties AGRÉGÉES dans results/01_audit/ ; valeurs par enregistrement uniquement dans results/cache/01_audit/
(non versionné, jamais affiché). Le test de raccourci n'utilise que le TRAIN.

Étapes (chacune mise en cache ; supprimer le fichier de cache pour recalculer) :
  A. VAD Silero + empreinte du canal par enregistrement (CPU, parallèle)
  B. Empreintes vocales ECAPA par segment (GPU)
  C. Regroupement des locuteurs par session (3 tâches d'un participant ensemble) → seconde voix
  D. Agrégats : profils début/fin, ICC du canal, proportions seconde voix, test de raccourci
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import FLUENCY_TASKS, TASKS, load_dataset, recordings  # noqa: E402

OUT = ROOT / "results" / "01_audit"
CACHE = ROOT / "results" / "cache" / "01_audit"
MODELS = ROOT / "models"

CONFIG = {
    "seed": 42,
    "sr": 16000,
    "vad": {"model": "silero-vad (pip)", "threshold": 0.5, "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 100, "speech_pad_ms": 30},
    "profile": {"window_s": 10.0, "bin_s": 0.1, "end_min_duration_s": 60.0},
    "channel": {"n_fft": 512, "win_length": 400, "hop_length": 160,
                "nonspeech_margin_s": 0.2, "nonspeech_margin_fallback_s": 0.05,
                "min_nonspeech_frames": 50, "bandwidth_energy_fraction": 0.95},
    "embedding": {"model": "speechbrain/spkrec-ecapa-voxceleb", "merge_gap_s": 0.3,
                  "min_unit_s": 0.8, "max_chunk_s": 3.0, "batch_size": 32},
    "clustering": {"method": "agglomerative, cosine, average linkage, par session (3 tâches)",
                   "similarity_threshold": 0.3, "similarity_grid": [0.2, 0.3, 0.4],
                   "second_voice_min_s": 3.0, "second_voice_grid_s": [1.0, 2.0, 3.0, 5.0, 10.0],
                   "main_speaker": "groupe de plus grande durée totale sur la session"},
    "shortcut": {"split": "Train", "n_splits": 5, "n_repeats": 5, "n_permutations": 1000,
                 "model": "SimpleImputer(median) + StandardScaler + LogisticRegression(class_weight=balanced)",
                 "channel_vars": ["noise_db", "snr_db", "bw95_hz", "noise_centroid_hz"]},
    "n_jobs": 6,
}
CHANNEL_VARS = CONFIG["shortcut"]["channel_vars"]


# ---------------------------------------------------------------- A. VAD + canal

_VAD = None


def _vad_model():
    global _VAD
    if _VAD is None:
        from silero_vad import load_silero_vad
        torch.set_num_threads(1)
        _VAD = load_silero_vad()
    return _VAD


def frame_mask(times, segs, pad=0.0):
    m = np.zeros(len(times), bool)
    for s, e in segs:
        m |= (times >= s - pad) & (times <= e + pad)
    return m


def channel_features(y, segs):
    c = CONFIG["channel"]
    S = np.abs(librosa.stft(y, n_fft=c["n_fft"], hop_length=c["hop_length"], win_length=c["win_length"])) ** 2
    freqs = librosa.fft_frequencies(sr=CONFIG["sr"], n_fft=c["n_fft"])
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=CONFIG["sr"], hop_length=c["hop_length"])
    power = S.mean(axis=0)
    speech = frame_mask(times, segs)
    far = ~frame_mask(times, segs, c["nonspeech_margin_s"])
    if far.sum() < c["min_nonspeech_frames"]:
        far = ~frame_mask(times, segs, c["nonspeech_margin_fallback_s"])
    eps = 1e-12
    f = {"n_nonspeech_frames": int(far.sum()), "n_speech_frames": int(speech.sum())}
    f["noise_db"] = 10 * np.log10(np.median(power[far]) + eps) if far.sum() >= 10 else np.nan
    f["speech_db"] = 10 * np.log10(np.median(power[speech]) + eps) if speech.sum() >= 10 else np.nan
    f["snr_db"] = f["speech_db"] - f["noise_db"]
    if speech.sum() >= 10:
        cum = np.cumsum(S[:, speech].mean(axis=1))
        f["bw95_hz"] = float(freqs[np.searchsorted(cum, c["bandwidth_energy_fraction"] * cum[-1])])
    else:
        f["bw95_hz"] = np.nan
    if far.sum() >= 10:
        ltas = S[:, far].mean(axis=1)
        f["noise_centroid_hz"] = float((freqs * ltas).sum() / (ltas.sum() + eps))
    else:
        f["noise_centroid_hz"] = np.nan
    return f


def process_recording(pid, task, wav):
    from silero_vad import get_speech_timestamps
    y, sr = sf.read(str(wav), dtype="float32")
    assert sr == CONFIG["sr"] and y.ndim == 1
    v = CONFIG["vad"]
    ts = get_speech_timestamps(torch.from_numpy(y), _vad_model(), sampling_rate=sr, return_seconds=True,
                               threshold=v["threshold"], min_speech_duration_ms=v["min_speech_duration_ms"],
                               min_silence_duration_ms=v["min_silence_duration_ms"],
                               speech_pad_ms=v["speech_pad_ms"])
    segs = [(float(t["start"]), float(t["end"])) for t in ts]
    rec = {"participant_id": pid, "task": task, "duration_s": len(y) / sr, "n_segments": len(segs),
           "speech_s": sum(e - s for s, e in segs),
           "first_onset_s": segs[0][0] if segs else np.nan}
    rec.update(channel_features(y, segs))
    return rec, segs


def stage_vad(recs):
    feat_path, vad_path = CACHE / "recording_features.csv", CACHE / "vad_segments.json"
    if feat_path.exists() and vad_path.exists():
        return pd.read_csv(feat_path), json.loads(vad_path.read_text())
    out = Parallel(n_jobs=CONFIG["n_jobs"], verbose=5)(
        delayed(process_recording)(r.participant_id, r.task, r.wav) for r in recs.itertuples())
    feats = pd.DataFrame([o[0] for o in out])
    vad = {f"{o[0]['participant_id']}__{o[0]['task']}": o[1] for o in out}
    feats.to_csv(feat_path, index=False)
    vad_path.write_text(json.dumps(vad))
    return feats, vad


# ---------------------------------------------------------------- B. ECAPA

def embedding_units(segs):
    """Segments VAD fusionnés (écart < merge_gap_s), gardés si ≥ min_unit_s, découpés en morceaux ≤ max_chunk_s."""
    e = CONFIG["embedding"]
    merged = []
    for s, t in segs:
        if merged and s - merged[-1][1] < e["merge_gap_s"]:
            merged[-1][1] = t
        else:
            merged.append([s, t])
    units = []
    for s, t in merged:
        d = t - s
        if d < e["min_unit_s"]:
            continue
        k = int(np.ceil(d / e["max_chunk_s"]))
        units += [(s + i * d / k, s + (i + 1) * d / k) for i in range(k)]
    return units


def stage_embeddings(recs, vad):
    path = CACHE / "embeddings.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return pd.DataFrame({"participant_id": z["pid"], "task": z["task"], "start": z["start"],
                             "end": z["end"]}), z["emb"]
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams(source=CONFIG["embedding"]["model"],
                                         savedir=str(MODELS / "spkrec-ecapa-voxceleb"),
                                         run_opts={"device": "cuda:0" if torch.cuda.is_available() else "cpu"})
    sr, bs = CONFIG["sr"], CONFIG["embedding"]["batch_size"]
    meta, embs = [], []
    for i, r in enumerate(recs.itertuples()):
        units = embedding_units(vad[f"{r.participant_id}__{r.task}"])
        if not units:
            continue
        y, _ = sf.read(str(r.wav), dtype="float32")
        chunks = [y[int(s * sr):int(t * sr)] for s, t in units]
        for b in range(0, len(chunks), bs):
            batch = chunks[b:b + bs]
            L = max(len(c) for c in batch)
            x = torch.zeros(len(batch), L)
            for j, c in enumerate(batch):
                x[j, :len(c)] = torch.from_numpy(c)
            lens = torch.tensor([len(c) / L for c in batch])
            with torch.no_grad():
                embs.append(enc.encode_batch(x, wav_lens=lens).squeeze(1).cpu().numpy())
        meta += [(r.participant_id, r.task, s, t) for s, t in units]
        if (i + 1) % 100 == 0:
            print(f"  ECAPA : {i + 1}/{len(recs)} enregistrements", flush=True)
    seg = pd.DataFrame(meta, columns=["participant_id", "task", "start", "end"])
    emb = np.concatenate(embs).astype(np.float32)
    np.savez(path, pid=seg.participant_id.to_numpy(str), task=seg.task.to_numpy(str),
             start=seg.start.to_numpy(), end=seg.end.to_numpy(), emb=emb)
    return seg, emb


# ---------------------------------------------------------------- C. Regroupement / seconde voix

def cluster_sessions(seg, emb, sim_thr):
    from sklearn.cluster import AgglomerativeClustering
    labels = np.full(len(seg), -1)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    for pid, idx in seg.groupby("participant_id").indices.items():
        if len(idx) < 2:
            labels[idx] = 0
            continue
        cl = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=1 - sim_thr).fit(emb[idx])
        lab = cl.labels_
        dur = pd.Series(seg["end"].to_numpy()[idx] - seg["start"].to_numpy()[idx]).groupby(lab).sum()
        main = dur.idxmax()
        # 0 = locuteur principal ; k ≥ 1 = autres groupes, classés par durée décroissante
        order = {c: k for k, c in enumerate([main] + [c for c in dur.sort_values(ascending=False).index
                                                       if c != main])}
        labels[idx] = [order[c] for c in lab]
    return labels


def second_voice(seg, labels, feats, min_s):
    """Par enregistrement : durée des groupes non principaux dont la durée sur la session ≥ min_s."""
    s = seg.assign(label=labels, dur=seg["end"] - seg["start"])
    other = s[s.label > 0]
    clus_tot = other.groupby(["participant_id", "label"])["dur"].sum()
    keep = clus_tot[clus_tot >= min_s].index
    other = other.set_index(["participant_id", "label"]).loc[lambda d: d.index.isin(keep)].reset_index()
    per_rec = other.groupby(["participant_id", "task"])["dur"].sum().rename("second_voice_s")
    # durée max d'un même groupe secondaire dans la tâche (critère « un second groupe dépasse le seuil »)
    per_rec_max = (other.groupby(["participant_id", "task", "label"])["dur"].sum()
                   .groupby(["participant_id", "task"]).max().rename("second_voice_max_cluster_s"))
    r = feats[["participant_id", "task", "duration_s"]].merge(per_rec, on=["participant_id", "task"], how="left")
    r = r.merge(per_rec_max, on=["participant_id", "task"], how="left").fillna(
        {"second_voice_s": 0.0, "second_voice_max_cluster_s": 0.0})
    r["sv_task_flag"] = r["second_voice_max_cluster_s"] >= min_s
    # trace en début (< 10 s) et en fin (10 dernières s) attribuée à un groupe secondaire retenu
    w = CONFIG["profile"]["window_s"]
    o = other.merge(feats[["participant_id", "task", "duration_s"]], on=["participant_id", "task"])
    start = o[o.start < w].groupby(["participant_id", "task"]).size().rename("sv_in_first_10s")
    end = o[o.end > o.duration_s - w].groupby(["participant_id", "task"]).size().rename("sv_in_last_10s")
    r = r.merge(start, on=["participant_id", "task"], how="left").merge(end, on=["participant_id", "task"], how="left")
    r[["sv_in_first_10s", "sv_in_last_10s"]] = r[["sv_in_first_10s", "sv_in_last_10s"]].fillna(0) > 0
    sess = (clus_tot[clus_tot >= min_s].groupby("participant_id").size() > 0).rename("sv_session_flag")
    return r, sess


def second_voice_strict_grid(seg, embn, lab, meta):
    """Variante stricte : groupe secondaire retenu seulement si son centroïde est peu similaire
    à celui du locuteur principal (sinon il s'agit probablement du même locuteur sur-segmenté)."""
    s = seg.assign(label=lab, d=seg.end - seg.start)
    rows = []
    for pid, g in s.groupby("participant_id"):
        c0 = embn[g.index[g.label == 0]].mean(0)
        c0 /= np.linalg.norm(c0)
        for k, gk in g[g.label > 0].groupby("label"):
            ck = embn[gk.index].mean(0)
            rows.append({"participant_id": pid, "sim": float(ck @ c0 / np.linalg.norm(ck)), "dur": gk.d.sum()})
    c = pd.DataFrame(rows)
    out = []
    for smax in [0.1, 0.2, 0.25, 0.3]:
        for T in [3, 5, 10]:
            flag = meta.participant_id.isin(c.loc[(c.sim < smax) & (c.dur >= T), "participant_id"])
            r = {"max_sim_to_main": smax, "min_dur_s": T, "pct_sessions": 100 * flag.mean()}
            r |= {f"pct_{d}_train": 100 * flag[(meta.diagnosis == d) & (meta.Split == "Train")].mean()
                  for d in ["HC", "MCI", "Dementia"]}
            out.append(r)
    out = pd.DataFrame(out)
    pct = [c for c in out.columns if c.startswith("pct_")]
    out[pct] = out[pct].round(1)
    return out


# ---------------------------------------------------------------- D. Agrégats

def activity_profiles(feats, vad, meta):
    p = CONFIG["profile"]
    nb = int(p["window_s"] / p["bin_s"])
    centers = (np.arange(nb) + 0.5) * p["bin_s"]
    rows = []
    for r in feats.merge(meta[["participant_id", "diagnosis"]]).itertuples():
        segs = vad[f"{r.participant_id}__{r.task}"]
        rows.append({"task": r.task, "diagnosis": r.diagnosis, "where": "start",
                     **dict(zip(range(nb), frame_mask(centers, segs).astype(float)))})
        if r.task in FLUENCY_TASKS and r.duration_s > p["end_min_duration_s"]:
            t = r.duration_s - p["window_s"] + centers
            rows.append({"task": r.task, "diagnosis": r.diagnosis, "where": "end",
                         **dict(zip(range(nb), frame_mask(t, segs).astype(float)))})
    df = pd.DataFrame(rows)
    prof = df.groupby(["where", "task"])[list(range(nb))].mean()
    n = df.groupby(["where", "task"]).size().rename("n")
    prof.columns = [f"{c:.2f}" for c in centers]
    prof = prof.join(n)
    prof_diag = df.groupby(["where", "task", "diagnosis"])[list(range(nb))].mean()
    prof_diag.columns = [f"{c:.2f}" for c in centers]
    return prof, prof_diag, centers


def plot_profiles(prof, centers):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, where in zip(axes, ["start", "end"]):
        for task in TASKS:
            if (where, task) in prof.index:
                row = prof.loc[(where, task)]
                x = centers if where == "start" else centers - CONFIG["profile"]["window_s"]
                ax.plot(x, row.drop("n").to_numpy(float), label=f"{task} (n={int(row['n'])})")
        ax.set_xlabel("temps depuis le début (s)" if where == "start" else "temps avant la fin (s)")
        ax.set_title("10 premières secondes" if where == "start" else "10 dernières s (fluences > 60 s)")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("proportion d'enregistrements en parole (VAD)")
    fig.tight_layout()
    fig.savefig(OUT / "speech_activity_profiles.png", dpi=150)
    plt.close(fig)


def icc(x):
    """ICC(2,1) accord absolu et ICC(3,1) cohérence ; x : n sujets × k tâches."""
    n, k = x.shape
    gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    icc3 = (msr - mse) / (msr + (k - 1) * mse)
    icc2 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    return icc2, icc3


def channel_icc(feats, meta, rng):
    tr = meta.loc[meta.Split == "Train", "participant_id"]
    rows = []
    for v in CHANNEL_VARS:
        w = feats[feats.participant_id.isin(tr)].pivot(index="participant_id", columns="task", values=v)[list(TASKS)]
        x = w.dropna().to_numpy(float)
        i2, i3 = icc(x)
        boots = np.array([icc(x[rng.integers(0, len(x), len(x))]) for _ in range(1000)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        rows.append({"variable": v, "n_participants": len(x), "ICC2_1": i2, "ICC2_1_lo": lo[0], "ICC2_1_hi": hi[0],
                     "ICC3_1": i3, "ICC3_1_lo": lo[1], "ICC3_1_hi": hi[1]})
    return pd.DataFrame(rows).round(3)


def shortcut_features(meta, feats, sv, sess):
    tr = meta[meta.Split == "Train"].set_index("participant_id")
    wide = lambda df, cols: df.pivot(index="participant_id", columns="task", values=cols)  # noqa: E731
    ch = wide(feats, CHANNEL_VARS)
    ch.columns = [f"{v}_{t}" for v, t in ch.columns]
    du = wide(feats, ["duration_s"])
    du.columns = [f"dur_{t}" for _, t in du.columns]
    s2 = np.log1p(wide(sv, ["second_voice_s"]))
    s2.columns = [f"log_sv_s_{t}" for _, t in s2.columns]
    s2["sv_session_flag"] = sess.reindex(s2.index).fillna(False).astype(float)
    base = pd.DataFrame({"male": (tr.gender.str.lower() == "male").astype(float), "age": tr.age.astype(float)})
    X = base.join(ch).join(du).join(s2)
    sets = {"sexe": ["male"], "âge": ["age"], "canal": list(ch.columns), "seconde voix": list(s2.columns),
            "durées": list(du.columns)}
    sets["tout"] = [c for cols in sets.values() for c in cols]
    return X.loc[tr.index], tr.diagnosis, sets


def cv_scores(X, y, multiclass, seed):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cv = RepeatedStratifiedKFold(n_splits=CONFIG["shortcut"]["n_splits"],
                                 n_repeats=CONFIG["shortcut"]["n_repeats"], random_state=seed)
    aucs, baccs = [], []
    for trn, tst in cv.split(X, y):
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=5000))
        m.fit(X[trn], y[trn])
        p = m.predict_proba(X[tst])
        aucs.append(roc_auc_score(y[tst], p, multi_class="ovr", average="macro") if multiclass
                    else roc_auc_score(y[tst], p[:, 1]))
        baccs.append(balanced_accuracy_score(y[tst], m.classes_[p.argmax(1)]))
    return np.array(aucs), np.array(baccs)


def shortcut_test(X, diag, sets):
    seed, n_perm = CONFIG["seed"], CONFIG["shortcut"]["n_permutations"]
    targets = {"HC vs MCI+Dementia": ((diag != "HC").astype(int).to_numpy(), False),
               "3 classes": (diag.map({"HC": 0, "MCI": 1, "Dementia": 2}).to_numpy(), True)}
    rows = []
    for tname, (y, multi) in targets.items():
        for sname, cols in sets.items():
            x = X[cols].to_numpy(float)
            auc, bacc = cv_scores(x, y, multi, seed)
            rng = np.random.default_rng(seed)
            perms = [rng.permutation(y) for _ in range(n_perm)]
            null = np.array(Parallel(n_jobs=CONFIG["n_jobs"])(
                delayed(lambda yp: tuple(s.mean() for s in cv_scores(x, yp, multi, seed)))(yp) for yp in perms))
            rows.append({"cible": tname, "variables": sname, "n_features": len(cols),
                         "AUC_mean": auc.mean(), "AUC_sd": auc.std(ddof=1),
                         "AUC_p_perm": (1 + (null[:, 0] >= auc.mean()).sum()) / (n_perm + 1),
                         "BAcc_mean": bacc.mean(), "BAcc_sd": bacc.std(ddof=1),
                         "BAcc_p_perm": (1 + (null[:, 1] >= bacc.mean()).sum()) / (n_perm + 1),
                         "null_AUC_p95": np.percentile(null[:, 0], 95),
                         "null_BAcc_p95": np.percentile(null[:, 1], 95)})
            print(f"  raccourci [{tname}] {sname}: AUC {auc.mean():.3f}, BAcc {bacc.mean():.3f}", flush=True)
    return pd.DataFrame(rows).round(4)


def pct_table(df, by, cols):
    g = df.groupby(by)
    t = g[cols].mean().mul(100).round(1)
    t.insert(0, "n", g.size())
    return t


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(CONFIG["seed"])
    meta = load_dataset()
    recs = recordings(meta)

    print("A. VAD + canal", flush=True)
    feats, vad = stage_vad(recs)
    print("B. ECAPA", flush=True)
    seg, emb = stage_embeddings(recs, vad)
    feats = feats.merge(meta[["participant_id", "diagnosis", "Split"]], on="participant_id")

    # --- VAD / couverture des empreintes
    seg_dur = seg.assign(d=seg.end - seg.start).groupby(["participant_id", "task"])["d"].sum().rename("embedded_s")
    f2 = feats.merge(seg_dur, on=["participant_id", "task"], how="left").fillna({"embedded_s": 0})
    f2["speech_ratio"] = f2.speech_s / f2.duration_s
    f2["embedded_fraction"] = f2.embedded_s / f2.speech_s.replace(0, np.nan)
    f2["speech_in_first_0.5s"] = f2.first_onset_s < 0.5
    q = lambda c: f2.groupby("task")[c].quantile([.05, .25, .5, .75, .95]).unstack().add_prefix(f"{c}_p")  # noqa: E731
    vad_summary = pd.concat([q("n_segments"), q("speech_ratio"), q("first_onset_s"), q("embedded_fraction")],
                            axis=1).round(3)
    vad_summary["pct_speech_in_first_0.5s"] = f2.groupby("task")["speech_in_first_0.5s"].mean().mul(100).round(1)
    vad_summary.to_csv(OUT / "vad_summary.csv")

    # --- C. seconde voix
    print("C. regroupement des locuteurs", flush=True)
    cl = CONFIG["clustering"]
    labels = {thr: cluster_sessions(seg, emb, thr) for thr in cl["similarity_grid"]}
    sv, sess = second_voice(seg, labels[cl["similarity_threshold"]], feats, cl["second_voice_min_s"])
    sv = sv.merge(meta[["participant_id", "diagnosis", "Split"]], on="participant_id")
    sv.to_csv(CACHE / "second_voice_per_recording.csv", index=False)
    sv_sess = meta[["participant_id", "diagnosis", "Split"]].assign(
        sv_session_flag=lambda d: d.participant_id.map(sess).fillna(False).astype(bool))
    sv_diag_task = pct_table(sv, ["Split", "diagnosis", "task"], ["sv_task_flag"]).unstack("task")
    sv_diag_task.columns = [f"{a}_{b}" for a, b in sv_diag_task.columns]
    sv_diag_task = sv_diag_task.join(pct_table(sv_sess, ["Split", "diagnosis"], ["sv_session_flag"])
                                     .rename(columns={"n": "n_sessions", "sv_session_flag": "pct_sessions"}))
    sv_diag_task.to_csv(OUT / "second_voice_by_diagnosis_task.csv")
    grid = []
    for thr in cl["similarity_grid"]:
        for ms in cl["second_voice_grid_s"]:
            r, s = second_voice(seg, labels[thr], feats, ms)
            r = r.merge(meta[["participant_id", "diagnosis"]])
            row = {"similarity_threshold": thr, "min_duration_s": ms,
                   "pct_sessions_flagged": 100 * meta.participant_id.map(s).fillna(False).astype(bool).mean()}
            row |= {f"pct_{t}_flagged": 100 * r.loc[r.task == t, "sv_task_flag"].mean() for t in TASKS}
            row |= {f"pct_sessions_{d}": 100 * meta.loc[meta.diagnosis == d, "participant_id"].map(s)
                    .fillna(False).astype(bool).mean() for d in ["HC", "MCI", "Dementia"]}
            grid.append(row)
    pd.DataFrame(grid).round(1).to_csv(OUT / "second_voice_threshold_grid.csv", index=False)
    # nombre de groupes et similarité au locuteur principal (diagnostic de qualité du regroupement)
    lab = labels[cl["similarity_threshold"]]
    embn = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    qual = []
    for pid, idx in seg.groupby("participant_id").indices.items():
        l = lab[idx]
        c0 = embn[idx][l == 0].mean(0)
        sims = [float(embn[idx][l == k].mean(0) @ c0 / np.linalg.norm(c0) /
                      np.linalg.norm(embn[idx][l == k].mean(0))) for k in set(l) - {0}]
        qual.append({"n_clusters": len(set(l)), "main_share": float(
            (seg.end.to_numpy()[idx] - seg.start.to_numpy()[idx])[l == 0].sum() /
            (seg.end.to_numpy()[idx] - seg.start.to_numpy()[idx]).sum()),
            "max_sim_other_to_main": max(sims) if sims else np.nan})
    qual = pd.DataFrame(qual).describe(percentiles=[.05, .25, .5, .75, .95]).round(3)
    qual.to_csv(OUT / "clustering_quality.csv")
    second_voice_strict_grid(seg, embn, lab, meta).to_csv(OUT / "second_voice_strict_grid.csv", index=False)

    # --- (b) début / fin
    prof, prof_diag, centers = activity_profiles(feats, vad, meta)
    prof.round(3).to_csv(OUT / "speech_activity_profile.csv")
    prof_diag.round(3).to_csv(OUT / "speech_activity_profile_by_diagnosis.csv")
    plot_profiles(prof, centers)
    sv["long_fluency"] = sv.task.isin(FLUENCY_TASKS) & (sv.duration_s > CONFIG["profile"]["end_min_duration_s"])
    trace = pct_table(sv, ["task"], ["sv_in_first_10s"])
    trace = trace.join(sv[sv.long_fluency].groupby("task").agg(
        n_long=("sv_in_last_10s", "size"), sv_in_last_10s=("sv_in_last_10s", lambda x: round(100 * x.mean(), 1))))
    trace.to_csv(OUT / "instruction_trace_by_task.csv")
    trace_diag = pct_table(sv, ["task", "diagnosis"], ["sv_in_first_10s"])
    trace_diag.to_csv(OUT / "instruction_trace_by_task_diagnosis.csv")

    # --- (c) canal
    ch_sum = feats.groupby(["task", "diagnosis"])[CHANNEL_VARS].median().round(2)
    ch_sum.to_csv(OUT / "channel_median_by_task_diagnosis.csv")
    ch_q = feats.groupby("task")[CHANNEL_VARS].quantile([.05, .25, .5, .75, .95]).round(2)
    ch_q.to_csv(OUT / "channel_quantiles_by_task.csv")
    n_digital_silence = int((feats.noise_db < -100).sum())
    iccs = channel_icc(feats, meta, rng)
    iccs.to_csv(OUT / "channel_icc_train.csv", index=False)

    # --- (e) raccourci
    print("D. test de raccourci (train)", flush=True)
    X, diag, sets = shortcut_features(meta, feats, sv, sess)
    sc = shortcut_test(X, diag, sets)
    sc.to_csv(OUT / "shortcut_test_train.csv", index=False)

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "n_recordings": len(feats), "n_embedding_units": len(seg),
        "n_recordings_noise_below_-100dB": n_digital_silence,
        "n_recordings_without_nonspeech_frames": int(feats.noise_db.isna().sum()),
        "pct_sessions_second_voice": round(100 * sv_sess.sv_session_flag.mean(), 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for name, t in [("VAD", vad_summary), ("Trace consigne", trace), ("Trace par diag", trace_diag),
                    ("Canal médianes", ch_sum), ("ICC canal (train)", iccs),
                    ("Seconde voix", sv_diag_task), ("Grille seconde voix", pd.DataFrame(grid).round(1)),
                    ("Qualité regroupement", qual), ("Raccourci (train)", sc)]:
        print(f"\n== {name} ==\n{t}")


if __name__ == "__main__":
    main()
