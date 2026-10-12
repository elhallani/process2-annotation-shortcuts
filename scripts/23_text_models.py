"""Modèles texte sur transcriptions WHISPER (aucune annotation humaine) — TRAIN uniquement, EXPLORATOIRE.
Modèles texte complémentaires (étapes 0–2).

Textes : Whisper-medium horodaté (11_clinical pour SFT/PFT ; transcrit ici pour CTD, même réglage).
Pauses mesurées (VAD de 01_audit) insérées entre mots : « , » (0,5–2 s) et « ... » (> 2 s) ; temps des mots
interpolés dans leur segment Whisper.
Modèles, mêmes 5 × 5 folds que le modèle principal (t04.folds, graine 42) :
  distilbert_asr, distilbert_asr_pauses, roberta_asr_pauses (fine-tuning, 1 graine/fold, tête 3 classes,
  score binaire 1 − P(HC)) ; mpnet_asr_pauses (all-mpnet-base-v2 gelé + LR) ;
  fusion = moyenne (poids fixes 0,5/0,5) du modèle principal B_G_nocov et du meilleur modèle texte par tâche.
Comparaisons : Nadeau–Bengio contre B_G_nocov (AUC 2 classes), Holm.
"""

import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
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
m11 = _load("m11", "scripts/11_clinical_scores.py")
t04 = m10.t04
OUT, CACHE = ROOT / "results" / "23_text_models", ROOT / "results" / "cache" / "23_text_models"
VAD = json.loads((ROOT / "results" / "cache" / "01_audit" / "vad_segments.json").read_text())


def texts_with_pauses(pid, task, with_pauses):
    segs = json.loads(m11.ts_path(pid, task).read_text(encoding="utf-8"))
    words = m11.timed_words(segs, np.inf)
    if not with_pauses:
        return " ".join(w for w, *_ in words)
    gaps = [(a[1], b[0]) for a, b in zip(VAD[f"{pid}__{task}"][:-1], VAD[f"{pid}__{task}"][1:])]
    out = []
    for k, (w, t, _) in enumerate(words):
        out.append(w)
        if k + 1 < len(words):
            t2 = words[k + 1][1]
            g = sum(e - s for s, e in gaps if s >= t and e <= t2)
            out.append("..." if g > 2 else "," if g >= .5 else "")
    return " ".join(x for x in out if x)


def frozen_mpnet(texts):
    digest = hashlib.sha1("\n".join(texts).encode()).hexdigest()[:12]
    path = CACHE / f"mpnet_{digest}.npy"
    if path.exists():
        return np.load(path)
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    mdl = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").cuda().eval()
    embs = []
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i + 16], padding=True, truncation=True, max_length=384, return_tensors="pt").to("cuda")
        with torch.no_grad():
            h = mdl(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1)
        embs.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    E = np.vstack(embs)
    np.save(path, E)
    return E


def main():
    for d in (OUT, CACHE):
        d.mkdir(parents=True, exist_ok=True)
    meta = load_dataset()
    m11.transcribe(recordings(meta).query("task == 'CTD'"))  # horodatage CTD (même réglage Whisper-medium)
    tr = meta[meta.Split == "Train"].reset_index(drop=True)
    T = m10.participant_table(tr)
    y3 = T.diagnosis.map({"HC": 0, "MCI": 1, "Dementia": 2}).to_numpy()
    yb = (y3 > 0).astype(int)
    splits = t04.folds(y3, 5, 5)
    main_path = CACHE / "P_main.npy"
    if main_path.exists():
        P_main = np.load(main_path)
    else:
        P_all, _, _, _ = m10.run_cv(T)
        P_main = P_all[("B_G_nocov", "bin")]
        np.save(main_path, P_main)

    def fold_auc(P):
        return np.array([roc_auc_score(yb[tst], P[r, tst][:, 1]) for r, rep in enumerate(splits) for _, tst in rep])

    a_main = fold_auc(P_main)
    rows = [{"modèle": "B_G_nocov (principal)", "task": "-", "AUC": a_main.mean(), "AUC_sd": a_main.std(ddof=1)}]
    comps = []
    for t in TASKS:
        txt = {k: [texts_with_pauses(p, t, k == "pauses") for p in tr.participant_id] for k in ("plain", "pauses")}
        P = {}
        for name, model, key in [("distilbert_asr", "distilbert-base-uncased", "plain"),
                                 ("distilbert_asr_pauses", "distilbert-base-uncased", "pauses"),
                                 ("roberta_asr_pauses", "roberta-base", "pauses")]:
            path = CACHE / f"P3_{name}_{t}.npy"
            if path.exists():
                P3 = np.load(path)
            else:
                t04.CONFIG["bert"]["model"] = model
                t04.CONFIG["bert"]["lr"] = 2e-5 if model.startswith("roberta") else 5e-5
                P3 = t04.oof_bert(txt[key], y3, splits, [])
                np.save(path, P3)
            P[name] = np.stack([P3[..., 0], 1 - P3[..., 0]], -1)
            print(f"  {t} {name} fait", flush=True)
        E = frozen_mpnet(txt["pauses"])
        Pm = np.zeros((5, len(yb), 2))
        for r, rep in enumerate(splits):
            for trn, tst in rep:
                lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
                Pm[r, tst] = lr.fit(E[trn], yb[trn]).predict_proba(E[tst])
        P["mpnet_asr_pauses"] = Pm
        aucs = {k: fold_auc(v) for k, v in P.items()}
        best = max(aucs, key=lambda k: aucs[k].mean())
        P["fusion_" + best] = (P_main + P[best]) / 2
        aucs["fusion_" + best] = fold_auc(P["fusion_" + best])
        for k, a in aucs.items():
            rows.append({"modèle": k, "task": t, "AUC": a.mean(), "AUC_sd": a.std(ddof=1)})
            d = a - a_main
            comps.append({"comparaison": f"{k} ({t}) − principal", "dAUC": d.mean(), "p_NB": nadeau_bengio(d, 256, 64)[1]})
    comps = pd.DataFrame(comps)
    comps["p_Holm"] = holm(comps.p_NB)
    res = pd.DataFrame(rows).round(4)
    res.to_csv(OUT / "cv_results_train.csv", index=False)
    comps.round(4).to_csv(OUT / "comparisons_vs_main_train.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                                                  "statut": "exploratoire, train uniquement",
                                                  "pause_tokens": {", ": "0,5–2 s", "...": "> 2 s"}}, indent=2,
                                                 ensure_ascii=False))
    pd.set_option("display.width", 200)
    print(res.to_string(index=False), "\n\n", comps.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
