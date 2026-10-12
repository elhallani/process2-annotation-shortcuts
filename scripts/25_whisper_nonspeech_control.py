"""Contrôle de raccourci (EXPLORATOIRE, train) : encodeur Whisper-small couches 8 et 10 sur les trames de NON-PAROLE
seulement (même fenêtre, même pooling que 24_acoustic_models). Si l'AUC reste proche de celle de la parole, la
représentation capte surtout l'environnement d'enregistrement."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_dataset, recordings  # noqa: E402

spec = importlib.util.spec_from_file_location("a24", ROOT / "scripts" / "24_acoustic_models.py")
a24 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a24)
LAYERS = [8, 10]


def main():
    path = a24.CACHE / "whisper_nonspeech.npz"
    meta = load_dataset()
    tr = meta[meta.Split == "Train"].reset_index(drop=True)
    recs = recordings(tr)
    if not path.exists():
        from transformers import WhisperFeatureExtractor, WhisperModel
        fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        enc = WhisperModel.from_pretrained("openai/whisper-small", dtype=torch.float16).encoder.cuda().eval()
        out = {f"whisper{l}_nonspeech": [] for l in LAYERS} | {"pid": [], "task": []}
        for r in recs.itertuples():
            y, sr = sf.read(str(r.wav), dtype="float32")
            y = y[:int(min(len(y) / sr, a24.W) * sr)]
            segs = [(s, min(e, a24.W)) for s, e in a24.VAD[f"{r.participant_id}__{r.task}"] if s < a24.W]
            chunks = []
            for c0 in range(0, len(y), 30 * sr):
                seg = y[c0:c0 + 30 * sr]
                with torch.no_grad():
                    hw = enc(fe(seg, sampling_rate=sr, return_tensors="pt").input_features.cuda().half(),
                             output_hidden_states=True).hidden_states
                n = int(np.ceil(len(seg) / sr / .02))
                chunks.append({l: hw[l][0, :n].float().cpu().numpy() for l in LAYERS})
            m = a24.mask(sum(len(c[LAYERS[0]]) for c in chunks), .02, segs)
            for l in LAYERS:
                out[f"whisper{l}_nonspeech"].append(a24.stats(np.vstack([c[l] for c in chunks]), ~m))
            out["pid"].append(r.participant_id)
            out["task"].append(r.task)
        np.savez(path, **{k: np.array(v) for k, v in out.items()})
    F = dict(np.load(path))
    T = a24.m10.participant_table(tr)
    y3 = T.diagnosis.map({"HC": 0, "MCI": 1, "Dementia": 2}).to_numpy()
    yb = (y3 > 0).astype(int)
    splits = a24.t04.folds(y3, 5, 5)
    rows = []
    for l in LAYERS:
        X = a24.participant_matrix(F, f"whisper{l}_nonspeech", T.index)
        P = np.zeros((5, len(yb), 2))
        for r, rep in enumerate(splits):
            for trn, tst in rep:
                P[r, tst] = a24.EmbModel(None).fit(X[trn], yb[trn]).predict_proba(X[tst])
        a = np.array([roc_auc_score(yb[tst], P[r, tst][:, 1]) for r, rep in enumerate(splits) for _, tst in rep])
        rows.append({"modèle": f"whisper_small_L{l}_NON_PAROLE", "AUC_diag": round(a.mean(), 4), "AUC_sd": round(a.std(ddof=1), 4)})
    res = pd.DataFrame(rows)
    res.to_csv(a24.OUT / "whisper_nonspeech_control_train.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
