"""Étape 4b — Branche de comparaison WavLM (TRAIN uniquement).

microsoft/wavlm-base-plus gelé (fp32), couche 6 (sortie du 6e bloc), moyenne sur les trames de parole (VAD de
01_audit) dans la fenêtre min(durée, 45 s), par enregistrement.
Représentations par participant :
  wavlm_raw       concaténation [e_SFT, e_PFT, e_CTD] (contient identité vocale et canal)
  wavlm_centered  profil centré [e_t − moyenne des 3 tâches] (identité et canal communs retirés)
Modèles (5 × 5 folds identiques à 10_model) : StandardScaler + PCA(32) ajustée dans le fold + LR pondérée
(2 classes) / Frank & Hall (3 classes). Fusion : variables B_G (configs/model_clinical.yaml, normes dans le fold)
+ PCA(32) du profil centré dans un même modèle. Contrôle : prédiction du sexe.
"""

import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import TASKS, load_dataset, recordings  # noqa: E402


def _load(name, path, argv=None):
    saved = sys.argv
    sys.argv = [path] + (argv or [])
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = saved
    return mod


t04 = _load("t04", "scripts/04_text_artifact.py")
m10 = _load("m10", "scripts/10_model.py", ["--config", "configs/model_clinical.yaml"])

OUT = ROOT / "results" / "12_wavlm"
CACHE = ROOT / "results" / "cache" / "12_wavlm"
CONFIG = {"seed": 42, "model": "microsoft/wavlm-base-plus", "layer": 6, "window_s": 45.0, "pooling": "moyenne trames VAD",
          "pca_components": 32, "cv": {"n_splits": 5, "n_repeats": 5}, "n_permutations": 1000,
          "fusion_base": "B_G de configs/model_clinical.yaml",
          "comparisons": [["wavlm_centered", "wavlm_raw"], ["wavlm_centered", "B_G"], ["fusion", "B_G"]]}
CLASSES = {"HC": 0, "MCI": 1, "Dementia": 2}


def extract(recs):
    path = CACHE / "embeddings_layer6.npz"
    if path.exists():
        z = np.load(path)
        return pd.DataFrame(z["emb"], index=pd.MultiIndex.from_arrays([z["pid"], z["task"]]))
    from transformers import AutoFeatureExtractor, WavLMModel
    vad = json.loads((ROOT / "results" / "cache" / "01_audit" / "vad_segments.json").read_text())
    fe = AutoFeatureExtractor.from_pretrained(CONFIG["model"])
    model = WavLMModel.from_pretrained(CONFIG["model"]).cuda().eval()
    embs, t0 = [], time.time()
    for i, r in enumerate(recs.itertuples()):
        y, sr = sf.read(str(r.wav), dtype="float32")
        w = min(len(y) / sr, CONFIG["window_s"])
        y = y[:int(w * sr)]
        inp = fe(y, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            h = model(inp.input_values.cuda(), output_hidden_states=True).hidden_states[CONFIG["layer"]][0].cpu().numpy()
        t = (np.arange(len(h)) + 0.5) * 0.02
        mask = np.zeros(len(h), bool)
        for s, e in vad[f"{r.participant_id}__{r.task}"]:
            mask |= (t >= s) & (t <= e)
        embs.append(h[mask].mean(0) if mask.sum() >= 5 else h.mean(0))
        if (i + 1) % 200 == 0:
            print(f"  wavlm : {i + 1}/{len(recs)} ({(time.time() - t0) / (i + 1):.2f} s/enr.)", flush=True)
    emb = np.vstack(embs).astype(np.float32)
    np.savez(path, emb=emb, pid=recs.participant_id.to_numpy(str), task=recs.task.to_numpy(str))
    return pd.DataFrame(emb, index=pd.MultiIndex.from_arrays([recs.participant_id, recs.task]))


def representations(E, pids):
    per = {t: E.xs(t, level=1).loc[pids].to_numpy() for t in TASKS}
    mean = np.mean([per[t] for t in TASKS], 0)
    return {"wavlm_raw": np.hstack([per[t] for t in TASKS]),
            "wavlm_centered": np.hstack([per[t] - mean for t in TASKS])}


def emb_model():
    return make_pipeline(StandardScaler(), PCA(CONFIG["pca_components"], random_state=CONFIG["seed"]),
                         LogisticRegression(class_weight="balanced", C=1.0, max_iter=5000))


class EmbFrankHall(m10.FrankHall):
    def fit(self, X, y):
        self.m1, self.m2 = emb_model().fit(X, (y > 0).astype(int)), emb_model().fit(X, (y > 1).astype(int))
        return self


class Fusion:
    """Variables interprétables (standardisées) + PCA du profil WavLM ajustée sur le train du fold, une seule LR."""
    def __init__(self, n_tab, multiclass):
        self.n_tab, self.multi = n_tab, multiclass

    def _prep_fit(self, X):
        tab, emb = X[:, :self.n_tab], X[:, self.n_tab:]
        self.imp, self.sc = SimpleImputer(strategy="median").fit(tab), StandardScaler()
        self.sc_e, self.pca = StandardScaler().fit(emb), PCA(CONFIG["pca_components"], random_state=CONFIG["seed"])
        z = np.hstack([self.sc.fit_transform(self.imp.transform(tab)), self.pca.fit_transform(self.sc_e.transform(emb))])
        self.sc2 = StandardScaler().fit(z)
        return self.sc2.transform(z)

    def _prep(self, X):
        tab, emb = X[:, :self.n_tab], X[:, self.n_tab:]
        z = np.hstack([self.sc.transform(self.imp.transform(tab)), self.pca.transform(self.sc_e.transform(emb))])
        return self.sc2.transform(z)

    def fit(self, X, y):
        Z = self._prep_fit(X)
        lr = lambda: LogisticRegression(class_weight="balanced", C=1.0, max_iter=5000)  # noqa: E731
        self.m = [lr().fit(Z, y)] if not self.multi else [lr().fit(Z, (y > 0).astype(int)), lr().fit(Z, (y > 1).astype(int))]
        return self

    def predict_proba(self, X):
        Z = self._prep(X)
        if not self.multi:
            return self.m[0].predict_proba(Z)
        p1, p2 = self.m[0].predict_proba(Z)[:, 1], self.m[1].predict_proba(Z)[:, 1]
        p = np.column_stack([1 - p1, np.clip(p1 - p2, 0, None), p2])
        return p / p.sum(1, keepdims=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = load_dataset()
    E = extract(recordings(meta))
    tr = meta[meta.Split == "Train"].reset_index(drop=True)
    T = m10.participant_table(tr)
    R = representations(E, T.index)
    y3 = T.diagnosis.map(CLASSES).to_numpy()
    yb = (y3 > 0).astype(int)
    ysex = T.male.astype(int).to_numpy()
    cv = CONFIG["cv"]
    splits = t04.folds(y3, cv["n_splits"], cv["n_repeats"])
    names = ["wavlm_raw", "wavlm_centered", "B_G", "fusion"]
    P = {(n, t): np.zeros((cv["n_repeats"], len(T), 3 if t == "3c" else 2)) for n in names for t in ("bin", "3c", "sex")}
    for r, rep in enumerate(splits):
        for trn, tst in rep:
            F_tr, F_te = m10.fold_features(T.iloc[trn], T.iloc[tst])
            cols = m10.columns(F_tr, m10.CFG["models"]["B_G"])
            tab_tr, tab_te = F_tr[cols].to_numpy(float), F_te[cols].to_numpy(float)
            X = {"wavlm_raw": (R["wavlm_raw"][trn], R["wavlm_raw"][tst]),
                 "wavlm_centered": (R["wavlm_centered"][trn], R["wavlm_centered"][tst]),
                 "B_G": (tab_tr, tab_te),
                 "fusion": (np.hstack([tab_tr, R["wavlm_centered"][trn]]), np.hstack([tab_te, R["wavlm_centered"][tst]]))}
            for n in names:
                Xtr, Xte = X[n]
                if n.startswith("wavlm"):
                    mb, m3, ms = emb_model(), EmbFrankHall(), emb_model()
                elif n == "B_G":
                    mb, m3, ms = m10.lr(), m10.FrankHall(), m10.lr()
                else:
                    mb, m3, ms = Fusion(len(cols), False), Fusion(len(cols), True), Fusion(len(cols), False)
                P[(n, "bin")][r, tst] = mb.fit(Xtr, yb[trn]).predict_proba(Xte)
                P[(n, "3c")][r, tst] = m3.fit(Xtr, y3[trn]).predict_proba(Xte)
                P[(n, "sex")][r, tst] = ms.fit(Xtr, ysex[trn]).predict_proba(Xte)
        print(f"  répétition {r + 1}/{cv['n_repeats']} faite", flush=True)

    rng = np.random.default_rng(CONFIG["seed"])
    rows, diffs = [], []
    for n in names:
        for t, y, multi, tk in [("HC vs MCI+Dementia", yb, False, "bin"), ("3 classes", y3, True, "3c"),
                                ("sexe (contrôle)", ysex, False, "sex")]:
            auc, bacc = t04.fold_metrics(y, P[(n, tk)], multi, splits)
            row = {"modèle": n, "cible": t, "AUC_mean": auc.mean(), "AUC_sd": auc.std(ddof=1),
                   "BAcc_mean": bacc.mean(), "BAcc_sd": bacc.std(ddof=1)}
            if multi:
                f1 = m10.macro_f1(y, P[(n, tk)], splits)
                row |= {"macroF1_mean": f1.mean(), "macroF1_sd": f1.std(ddof=1)}
            rows.append(row)
    for a, b in CONFIG["comparisons"]:
        for t, y, multi, tk in [("HC vs MCI+Dementia", yb, False, "bin"), ("3 classes", y3, True, "3c")]:
            obs, p = t04.paired_permutation(y, P[(a, tk)], P[(b, tk)], multi, rng, CONFIG["n_permutations"])
            diffs.append({"comparaison": f"{a} − {b}", "cible": t, "dAUC": obs[0], "p_dAUC": p[0],
                          "dBAcc": obs[1], "p_dBAcc": p[1]})
    res, dif = pd.DataFrame(rows).round(4), pd.DataFrame(diffs).round(4)
    res.to_csv(OUT / "cv_results_train.csv", index=False)
    dif.to_csv(OUT / "paired_comparisons_train.csv", index=False)
    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False))
    (OUT / "summary.json").write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                                                  "n_train": len(T)}, indent=2))
    pd.set_option("display.width", 220)
    print(res, "\n\n", dif)


if __name__ == "__main__":
    main()
