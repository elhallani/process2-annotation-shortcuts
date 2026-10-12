"""Non-speech control that isolates silence before encoding (training set).

In 24/25 the whole recording was encoded and only non-speech frames were pooled, so these frames carry speech
context through self-attention. Controls, same window and VAD:
  (a) isolated: non-speech segments (shrunk by 0.1 s on each side; segments < 0.2 s dropped) are cut out,
      concatenated with 5-ms fades and encoded on their own;
  (b) masked: speech is replaced by digital silence and the duration is kept; pooling over non-speech frames.
Reference covariate: non-speech duration per task, alone and fused with the embeddings. Same 5 x 5 folds and
pipeline (StandardScaler, PCA(32), weighted logistic regression) as 24_acoustic_models."""

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import TASKS, load_dataset, recordings  # noqa: E402
from src.stats import holm, nadeau_bengio  # noqa: E402

spec = importlib.util.spec_from_file_location("a24", ROOT / "scripts" / "24_acoustic_models.py")
a24 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a24)
OUT, CACHE = ROOT / "results" / "28_nonspeech_isolated", ROOT / "results" / "cache" / "28_nonspeech_isolated"
WAVLM_LAYERS, WHISPER_LAYERS = [8, 10, 12], [6, 8, 10]
MARGIN, MIN_SEG, FADE = .1, .2, .005


def nonspeech_segments(segs, dur):
    """Complément des segments de parole, rétréci de MARGIN de chaque côté."""
    out, t = [], 0.0
    for s, e in sorted(segs) + [(dur, dur)]:
        a, b = t + (MARGIN if t > 0 else 0), s - (MARGIN if s < dur else 0)
        if b - a >= MIN_SEG:
            out.append((a, b))
        t = max(t, e)
    return out


def isolate(y, sr, ns):
    f = int(FADE * sr)
    ramp = np.linspace(0, 1, f, dtype=np.float32)
    parts = []
    for a, b in ns:
        p = y[int(a * sr):int(b * sr)].copy()
        if len(p) > 2 * f:
            p[:f] *= ramp
            p[-f:] *= ramp[::-1]
        parts.append(p)
    x = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    if len(x) < int(.5 * sr):  # trop court pour l'encodeur : répété jusqu'à 0,5 s (compté)
        x = np.resize(x, int(.5 * sr)) if len(x) else np.zeros(int(.5 * sr), np.float32)
    return x


def masked(y, sr, segs):
    z = y.copy()
    for s, e in segs:
        z[int(s * sr):int(e * sr)] = 0
    return z


def whisper_hidden(enc, fe, y, sr, layers):
    chunks = []
    for c0 in range(0, len(y), 30 * sr):
        seg = y[c0:c0 + 30 * sr]
        with torch.no_grad():
            hw = enc(fe(seg, sampling_rate=sr, return_tensors="pt").input_features.cuda().half(),
                     output_hidden_states=True).hidden_states
        n = int(np.ceil(len(seg) / sr / .02))
        chunks.append({l: hw[l][0, :n].float().cpu().numpy() for l in layers})
    return {l: np.vstack([c[l] for c in chunks]) for l in layers}


def extract(recs):
    final = CACHE / "features.npz"
    if final.exists():
        return dict(np.load(final))
    from transformers import AutoFeatureExtractor, WavLMModel, WhisperFeatureExtractor, WhisperModel
    part = CACHE / "partial.npz"
    feats = {k: list(v) for k, v in np.load(part).items()} if part.exists() else {}
    done = len(feats.get("pid", []))
    print(f"reprise à {done}/{len(recs)}", flush=True)
    fe_w = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").cuda().eval()
    fe_s = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    wenc = WhisperModel.from_pretrained("openai/whisper-small", dtype=torch.float16).encoder.cuda().eval()
    for i, r in enumerate(recs.itertuples()):
        if i < done:
            continue
        y, sr = sf.read(str(r.wav), dtype="float32")
        y = y[:int(min(len(y) / sr, a24.W) * sr)]
        dur = len(y) / sr
        segs = [(s, min(e, a24.W)) for s, e in a24.VAD[f"{r.participant_id}__{r.task}"] if s < a24.W]
        ns = nonspeech_segments(segs, dur)
        xi, xm = isolate(y, sr, ns), masked(y, sr, segs)
        row = {"pid": r.participant_id, "task": r.task, "ns_seconds": sum(b - a for a, b in ns),
               "too_short": float(sum(b - a for a, b in ns) < .5)}
        with torch.no_grad():
            hi = wavlm(fe_w(xi, sampling_rate=sr, return_tensors="pt").input_values.cuda(),
                       output_hidden_states=True).hidden_states
            hm = wavlm(fe_w(xm, sampling_rate=sr, return_tensors="pt").input_values.cuda(),
                       output_hidden_states=True).hidden_states
        mm = a24.mask(hm[0].shape[1], .02, segs)
        for l in WAVLM_LAYERS:
            row[f"wavlm{l}_iso"] = a24.stats(hi[l][0].cpu().numpy(), np.ones(hi[l].shape[1], bool))
            row[f"wavlm{l}_mask"] = a24.stats(hm[l][0].cpu().numpy(), ~mm)
        wi, wm = whisper_hidden(wenc, fe_s, xi, sr, WHISPER_LAYERS), whisper_hidden(wenc, fe_s, xm, sr, WHISPER_LAYERS)
        mw = a24.mask(len(wm[WHISPER_LAYERS[0]]), .02, segs)
        for l in WHISPER_LAYERS:
            row[f"whisper{l}_iso"] = a24.stats(wi[l], np.ones(len(wi[l]), bool))
            row[f"whisper{l}_mask"] = a24.stats(wm[l], ~mw)
        for k, v in row.items():
            feats.setdefault(k, []).append(v)
        if (i + 1) % 40 == 0:
            np.savez(part, **{k: np.array(v) for k, v in feats.items()})
            print(f"  extraction {i + 1}/{len(recs)}", flush=True)
    out = {k: np.array(v) for k, v in feats.items()}
    np.savez(final, **out)
    part.unlink(missing_ok=True)
    return out


def main():
    for d in (OUT, CACHE):
        d.mkdir(parents=True, exist_ok=True)
    meta = load_dataset()
    tr = meta[meta.Split == "Train"].reset_index(drop=True)
    F = extract(recordings(tr))
    T = a24.m10.participant_table(tr)
    y3 = T.diagnosis.map({"HC": 0, "MCI": 1, "Dementia": 2}).to_numpy()
    yb = (y3 > 0).astype(int)
    splits = a24.t04.folds(y3, 5, 5)
    ns = pd.DataFrame({"s": F["ns_seconds"]}, index=pd.MultiIndex.from_arrays([F["pid"], F["task"]]))
    D = np.log1p(np.column_stack([ns.xs(t, level=1).loc[T.index, "s"].to_numpy() for t in TASKS]))
    P_main = np.load(ROOT / "results" / "cache" / "23_text_models" / "P_main.npy")

    def fold_auc(Pk):
        return np.array([roc_auc_score(yb[tst], Pk[r, tst][:, 1]) for r, rep in enumerate(splits) for _, tst in rep])

    # durée de non-parole seule
    P_dur = np.zeros((5, len(yb), 2))
    for r, rep in enumerate(splits):
        for trn, tst in rep:
            P_dur[r, tst] = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")).fit(
                D[trn], yb[trn]).predict_proba(D[tst])
    a_dur = fold_auc(P_dur)
    rows, comps = [{"modèle": "durée de non-parole (3 tâches, log)", "AUC": a_dur.mean(), "AUC_sd": a_dur.std(ddof=1)}], []
    keys = [f"wavlm{l}_{m}" for l in WAVLM_LAYERS for m in ("iso", "mask")] + \
           [f"whisper{l}_{m}" for l in WHISPER_LAYERS for m in ("iso", "mask")]
    for k in keys:
        X = a24.participant_matrix(F, k, T.index)
        P, Pd = np.zeros((5, len(yb), 2)), np.zeros((5, len(yb), 2))
        for r, rep in enumerate(splits):
            for trn, tst in rep:
                P[r, tst] = a24.EmbModel(None).fit(X[trn], yb[trn]).predict_proba(X[tst])
                Pd[r, tst] = (P[r, tst] + P_dur[r, tst]) / 2
        a, ad = fold_auc(P), fold_auc(Pd)
        rows.append({"modèle": k, "AUC": a.mean(), "AUC_sd": a.std(ddof=1), "AUC_fusion_durée": ad.mean()})
        for name, arr, ref, rn in [(k, a, a_dur, "durée seule"), (f"{k}+durée", ad, a_dur, "durée seule"),
                                   (k, a, fold_auc(P_main), "principal")]:
            d = arr - ref
            comps.append({"comparaison": f"{name} − {rn}", "dAUC": d.mean(), "p_NB": nadeau_bengio(d, 256, 64)[1]})
    comps = pd.DataFrame(comps)
    comps["p_Holm"] = holm(comps.p_NB)
    res = pd.DataFrame(rows).round(4)
    ref = pd.read_csv(ROOT / "results" / "24_acoustic_models" / "cv_results_train.csv")
    res.to_csv(OUT / "cv_results_train.csv", index=False)
    comps.round(4).to_csv(OUT / "comparisons_train.csv", index=False)
    summ = {"generated": datetime.now().isoformat(timespec="seconds"), "statut": "train uniquement, contrôle de non-parole isolée",
            "margin_s": MARGIN, "min_seg_s": MIN_SEG, "window_s": a24.W,
            "n_recordings": int(len(F["pid"])), "n_too_short_nonspeech": int(F["too_short"].sum()),
            "nonspeech_seconds_median_by_group": {g: float(np.median(np.expm1(D[(T.diagnosis == g).to_numpy()])))
                                                  for g in ("HC", "MCI", "Dementia")},
            "rappel_24_full_context": ref[ref.modèle.str.contains("NON_PAROLE")][["modèle", "AUC_diag"]].to_dict("records")}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 200)
    print(res.to_string(index=False), "\n\n", comps.round(4).to_string(index=False), "\n", json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
