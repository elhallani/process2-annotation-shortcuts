"""Nouvelles méthodes acoustiques — TRAIN uniquement, EXPLORATOIRE.
Modèles acoustiques complémentaires (étapes 3, 4, 6 + contrôle non-parole).

Par enregistrement, fenêtre min(durée, 45 s), statistiques moyenne + écart-type :
  - WavLM-base-plus (gelé, fp32), couches 8, 10, 12, sur les trames de PAROLE (VAD) ; et sur les trames de
    NON-PAROLE (contrôle de canal) ;
  - encodeur Whisper-small (gelé, fp16), couches 6, 8, 10, trames de parole seulement (padding exclu) ;
  - eGeMAPSv02 (openSMILE, fonctionnelles) sur la parole concaténée.
Participant = concaténation des 3 tâches. Modèles (mêmes 5 × 5 folds que le principal) :
  embeddings : StandardScaler + [projection hors de la direction « sexe » apprise dans le fold] + PCA(32) + LR ;
  eGeMAPS : z par sexe vs HC du fold + LR (C = 0,1).
Rapportés : AUC diagnostic (2 classes), AUC sexe (contrôle d'identité), Nadeau–Bengio vs principal, fusion à poids fixes.
"""

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import TASKS, load_dataset, recordings  # noqa: E402
from src.stats import holm, nadeau_bengio  # noqa: E402


def _load(name, path, argv=None):
    saved, sys.argv = sys.argv, [path] + (argv or [])
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = saved
    return mod


m10 = _load("m10", "scripts/10_model.py", ["--config", "configs/model_clinical_v2.yaml"])
t04 = m10.t04
OUT, CACHE = ROOT / "results" / "24_acoustic_models", ROOT / "results" / "cache" / "24_acoustic_models"
VAD = json.loads((ROOT / "results" / "cache" / "01_audit" / "vad_segments.json").read_text())
W = 45.0
WAVLM_LAYERS, WHISPER_LAYERS = [8, 10, 12], [6, 8, 10]


def mask(n_frames, hop, segs):
    t = (np.arange(n_frames) + .5) * hop
    m = np.zeros(n_frames, bool)
    for s, e in segs:
        m |= (t >= s) & (t <= e)
    return m


def stats(h, m):
    x = h[m] if m.sum() >= 5 else h
    return np.concatenate([x.mean(0), x.std(0)])


def extract(recs):
    path = CACHE / "features.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    import opensmile
    from transformers import AutoFeatureExtractor, WavLMModel, WhisperFeatureExtractor, WhisperModel
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)
    fe_w = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").cuda().eval()
    fe_s = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    wenc = WhisperModel.from_pretrained("openai/whisper-small", dtype=torch.float16).encoder.cuda().eval()
    feats = {f"wavlm{l}": [] for l in WAVLM_LAYERS} | {f"wavlm{l}_nonspeech": [] for l in WAVLM_LAYERS} | \
        {f"whisper{l}": [] for l in WHISPER_LAYERS} | {"egemaps": [], "pid": [], "task": []}
    for i, r in enumerate(recs.itertuples()):
        y, sr = sf.read(str(r.wav), dtype="float32")
        y = y[:int(min(len(y) / sr, W) * sr)]
        segs = [(s, min(e, W)) for s, e in VAD[f"{r.participant_id}__{r.task}"] if s < W]
        with torch.no_grad():
            hs = wavlm(fe_w(y, sampling_rate=sr, return_tensors="pt").input_values.cuda(),
                       output_hidden_states=True).hidden_states
        m = mask(hs[0].shape[1], .02, segs)
        for l in WAVLM_LAYERS:
            h = hs[l][0].cpu().numpy()
            feats[f"wavlm{l}"].append(stats(h, m))
            feats[f"wavlm{l}_nonspeech"].append(stats(h, ~m))
        # Whisper : morceaux de 30 s ; trames de 20 ms ; padding exclu
        chunks = []
        for c0 in range(0, len(y), 30 * sr):
            seg = y[c0:c0 + 30 * sr]
            inp = fe_s(seg, sampling_rate=sr, return_tensors="pt").input_features.cuda().half()
            with torch.no_grad():
                hw = wenc(inp, output_hidden_states=True).hidden_states
            n = int(np.ceil(len(seg) / sr / .02))
            chunks.append({l: hw[l][0, :n].float().cpu().numpy() for l in WHISPER_LAYERS})
        mw = mask(sum(len(c[WHISPER_LAYERS[0]]) for c in chunks), .02, segs)
        for l in WHISPER_LAYERS:
            feats[f"whisper{l}"].append(stats(np.vstack([c[l] for c in chunks]), mw))
        sp = np.concatenate([y[int(s * sr):int(e * sr)] for s, e in segs]) if segs else y
        feats["egemaps"].append(smile.process_signal(sp, sr).to_numpy()[0])
        feats["pid"].append(r.participant_id)
        feats["task"].append(r.task)
        if (i + 1) % 200 == 0:
            print(f"  extraction {i + 1}/{len(recs)}", flush=True)
    out = {k: np.array(v) for k, v in feats.items()}
    np.savez(path, **out)
    return out


def participant_matrix(F, key, pids):
    df = pd.DataFrame(F[key], index=pd.MultiIndex.from_arrays([F["pid"], F["task"]]))
    return np.hstack([df.xs(t, level=1).loc[pids].to_numpy() for t in TASKS])


class EmbModel:
    """StandardScaler → (option) retrait de la direction sexe apprise dans le fold → PCA(32) → LR pondérée."""
    def __init__(self, sex_tr=None):
        self.sex_tr = sex_tr

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        self.u = None
        if self.sex_tr is not None:
            w = LogisticRegression(C=0.1, max_iter=5000).fit(Z, self.sex_tr).coef_[0]
            self.u = w / np.linalg.norm(w)
            Z = Z - np.outer(Z @ self.u, self.u)
        self.pca = PCA(32, random_state=42).fit(Z)
        self.lr = LogisticRegression(class_weight="balanced", max_iter=5000).fit(self.pca.transform(Z), y)
        return self

    def predict_proba(self, X):
        Z = self.sc.transform(X)
        if self.u is not None:
            Z = Z - np.outer(Z @ self.u, self.u)
        return self.lr.predict_proba(self.pca.transform(Z))


def egemaps_z(Xtr, Xte, hc_tr, male_tr, male_te):
    """z par sexe relativement aux HC du train du fold."""
    Ztr, Zte = np.zeros_like(Xtr), np.zeros_like(Xte)
    for g in (0, 1):
        ref = Xtr[hc_tr & (male_tr == g)]
        mu, sd = ref.mean(0), ref.std(0) + 1e-8
        Ztr[male_tr == g] = (Xtr[male_tr == g] - mu) / sd
        Zte[male_te == g] = (Xte[male_te == g] - mu) / sd
    return Ztr, Zte


def main():
    for d in (OUT, CACHE):
        d.mkdir(parents=True, exist_ok=True)
    meta = load_dataset()
    tr_meta = meta[meta.Split == "Train"].reset_index(drop=True)
    F = extract(recordings(tr_meta))
    T = m10.participant_table(tr_meta)
    pids = T.index
    y3 = T.diagnosis.map({"HC": 0, "MCI": 1, "Dementia": 2}).to_numpy()
    yb, male = (y3 > 0).astype(int), T.male.to_numpy().astype(int)
    splits = t04.folds(y3, 5, 5)
    P_main = np.load(ROOT / "results" / "cache" / "23_text_models" / "P_main.npy")
    variants = {}
    for l in WAVLM_LAYERS:
        variants[f"wavlm_L{l}"] = (f"wavlm{l}", False)
        variants[f"wavlm_L{l}_sans_sexe"] = (f"wavlm{l}", True)
        variants[f"wavlm_L{l}_NON_PAROLE"] = (f"wavlm{l}_nonspeech", False)
    for l in WHISPER_LAYERS:
        variants[f"whisper_small_L{l}"] = (f"whisper{l}", False)
        variants[f"whisper_small_L{l}_sans_sexe"] = (f"whisper{l}", True)
    variants["egemaps_z_sexe"] = ("egemaps", None)
    X = {k: participant_matrix(F, k, pids) for k in {v[0] for v in variants.values()}}
    P = {(v, t): np.zeros((5, len(yb), 2)) for v in variants for t in ("diag", "sex")}
    for r, rep in enumerate(splits):
        for trn, tst in rep:
            for v, (key, rm_sex) in variants.items():
                Xtr, Xte = X[key][trn], X[key][tst]
                if rm_sex is None:  # eGeMAPS : z par sexe vs HC du fold, puis LR
                    Ztr, Zte = egemaps_z(Xtr, Xte, yb[trn] == 0, male[trn], male[tst])
                    for tgt, y in (("diag", yb), ("sex", male)):
                        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)
                        P[(v, tgt)][r, tst] = clf.fit(Ztr, y[trn]).predict_proba(Zte)
                else:  # embeddings : même chaîne (avec ou sans retrait de la direction sexe) pour les deux cibles
                    sex = male[trn] if rm_sex else None
                    for tgt, y in (("diag", yb), ("sex", male)):
                        P[(v, tgt)][r, tst] = EmbModel(sex).fit(Xtr, y[trn]).predict_proba(Xte)
        print(f"  répétition {r + 1}/5", flush=True)

    def fold_auc(y, Pk):
        return np.array([roc_auc_score(y[tst], Pk[r, tst][:, 1]) for r, rep in enumerate(splits) for _, tst in rep])

    a_main = fold_auc(yb, P_main)
    rows, comps = [], []
    for v in variants:
        a, s = fold_auc(yb, P[(v, "diag")]), fold_auc(male, P[(v, "sex")])
        fus = fold_auc(yb, (P_main + P[(v, "diag")]) / 2)
        rows.append({"modèle": v, "AUC_diag": a.mean(), "AUC_diag_sd": a.std(ddof=1), "AUC_sexe": s.mean(),
                     "AUC_fusion_avec_principal": fus.mean()})
        if "NON_PAROLE" not in v:
            for name, arr in [(v, a), (f"fusion({v})", fus)]:
                d = arr - a_main
                comps.append({"comparaison": f"{name} − principal", "dAUC": d.mean(), "p_NB": nadeau_bengio(d, 256, 64)[1]})
    comps = pd.DataFrame(comps)
    comps["p_Holm"] = holm(comps.p_NB)
    res = pd.DataFrame(rows).round(4)
    res.to_csv(OUT / "cv_results_train.csv", index=False)
    comps.round(4).to_csv(OUT / "comparisons_vs_main_train.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                                                  "statut": "exploratoire, train uniquement", "AUC_principal": a_main.mean(),
                                                  "window_s": W}, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 200)
    print(f"principal : {a_main.mean():.3f}\n", res.to_string(index=False), "\n\n", comps.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
