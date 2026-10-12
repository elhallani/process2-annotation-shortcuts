"""Étape 7 (partielle) — Équité par sexe et analyses de sensibilité du modèle principal (TRAIN uniquement).

Modèle : B_G de configs/model_clinical.yaml (G + CTD + âge/sexe ; normes réapprises dans chaque fold).
1. Équité : probabilités hors-fold moyennées sur les 5 répétitions (une prédiction par participant) ;
   AUC, BAcc, sensibilité, spécificité par sexe ; IC 95 % et écart hommes − femmes par bootstrap (2000).
2. Sensibilité : seconde voix (union pyannote ≥ 3 s, ECAPA strict, transcription dialoguée) ; fluence courte (< 45 s).
   (a) exclusion à l'évaluation seulement (mêmes prédictions) ; (b) validation croisée refaite sans les exclus.
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_dataset  # noqa: E402


def _load(name, path, argv=None):
    saved, sys.argv = sys.argv, [path] + (argv or [])
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = saved
    return mod


_ap = argparse.ArgumentParser()
_ap.add_argument("--config", default="configs/model_clinical.yaml")
_ap.add_argument("--model", default="B_G")
_ap.add_argument("--out", default="results/13_robustness")
ARGS = _ap.parse_args()
m10 = _load("m10", "scripts/10_model.py", ["--config", ARGS.config])
sv03 = _load("sv03", "scripts/03_second_voice.py")
OUT = ROOT / ARGS.out
CONFIG = {"seed": 42, "model": f"{ARGS.model} ({ARGS.config})", "n_boot": 2000,
          "second_voice_union": ["pyannote session ≥ 3 s", "ECAPA strict (sim < 0,25, ≥ 3 s)", "transcription dialoguée"],
          "short": "au moins une fluence < 45 s"}


def metrics(y, p):
    pred = (p >= .5).astype(int)
    return {"AUC": roc_auc_score(y, p), "BAcc": balanced_accuracy_score(y, pred),
            "sensibilité": pred[y == 1].mean(), "spécificité": 1 - pred[y == 0].mean()}


def fairness(T, p, rng):
    y = (T.diagnosis != "HC").astype(int).to_numpy()
    male = T.male.to_numpy().astype(bool)
    obs = {g: metrics(y[m], p[m]) for g, m in [("hommes", male), ("femmes", ~male), ("tous", np.ones_like(male))]}
    boots = {g: [] for g in obs} | {"écart": []}
    for _ in range(CONFIG["n_boot"]):
        i = rng.integers(0, len(y), len(y))
        yi, pi, mi = y[i], p[i], male[i]
        if min(len(set(yi[mi])), len(set(yi[~mi]))) < 2:
            continue
        mm, mf = metrics(yi[mi], pi[mi]), metrics(yi[~mi], pi[~mi])
        boots["hommes"].append(mm)
        boots["femmes"].append(mf)
        boots["tous"].append(metrics(yi, pi))
        boots["écart"].append({k: mm[k] - mf[k] for k in mm})
    rows = []
    for g in ["hommes", "femmes", "tous", "écart"]:
        b = pd.DataFrame(boots[g])
        for k in ["AUC", "BAcc", "sensibilité", "spécificité"]:
            val = obs["hommes"][k] - obs["femmes"][k] if g == "écart" else obs[g][k]
            lo, hi = np.percentile(b[k], [2.5, 97.5])
            rows.append({"groupe": g, "métrique": k, "valeur": val, "IC_lo": lo, "IC_hi": hi})
    n = pd.crosstab(T.male.map({1.0: "hommes", 0.0: "femmes"}), T.diagnosis)
    return pd.DataFrame(rows).round(3), n


def flags(meta, T):
    diar = pd.read_csv(ROOT / "results" / "cache" / "03_second_voice" / "diarization_flags_per_recording.csv",
                       dtype={"participant_id": str})
    pya = diar.groupby("participant_id").flag.any()
    ecapa = sv03.ecapa_strict_flags(meta)
    tr = pd.read_csv(ROOT / "results" / "cache" / "03_second_voice" / "transcript_markers.csv",
                     dtype={"participant_id": str})
    dia = tr.groupby("participant_id").dialogue.any()
    f = pd.read_csv(ROOT / "results" / "cache" / "05_features" / "features_per_recording.csv",
                    dtype={"participant_id": str})
    short = f.groupby("participant_id").short_recording.any()
    idx = T.index
    F = pd.DataFrame({"pyannote": pya.reindex(idx).fillna(False).astype(bool),
                      "ecapa_strict": ecapa.reindex(idx).fillna(False).astype(bool),
                      "transcript_dialogue": dia.reindex(idx).fillna(False).astype(bool),
                      "short": short.reindex(idx).fillna(False).astype(bool)}, index=idx)
    F["second_voice_union"] = F[["pyannote", "ecapa_strict", "transcript_dialogue"]].any(axis=1)
    F["any"] = F.second_voice_union | F.short
    return F


def cv_auc(T):
    P, splits, yb, y3 = m10.run_cv(T)
    t04 = m10.t04
    auc_b, bacc_b = t04.fold_metrics(yb, P[(ARGS.model, "bin")], False, splits)
    auc_3, bacc_3 = t04.fold_metrics(y3, P[(ARGS.model, "3c")], True, splits)
    return P, {"AUC_bin": auc_b.mean(), "AUC_bin_sd": auc_b.std(ddof=1), "BAcc_bin": bacc_b.mean(),
               "AUC_3c": auc_3.mean(), "BAcc_3c": bacc_3.mean()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(CONFIG["seed"])
    meta = load_dataset()
    tr = meta[meta.Split == "Train"].reset_index(drop=True)
    T = m10.participant_table(tr)
    print("CV complète", flush=True)
    P, full = cv_auc(T)
    p = P[(ARGS.model, "bin")][..., 1].mean(0)

    fair, n_sex = fairness(T, p, rng)
    fair.to_csv(OUT / "fairness_by_sex_train.csv", index=False)
    n_sex.to_csv(OUT / "n_by_sex_diagnosis_train.csv")

    F = flags(meta, T)
    counts = F.join(T.diagnosis).groupby("diagnosis")[list(F.columns)].sum()
    counts.loc["total"] = F.sum()
    counts.to_csv(OUT / "flag_counts_train.csv")
    y = (T.diagnosis != "HC").astype(int).to_numpy()
    sens = [{"analyse": "référence (tous)", "mode": "-", "n": len(T), **full,
             "AUC_pooled_eval": roc_auc_score(y, p)}]
    for fl in ["second_voice_union", "short", "any"]:
        keep = ~F[fl].to_numpy()
        sens.append({"analyse": f"sans {fl}", "mode": "(a) exclus à l'évaluation", "n": int(keep.sum()),
                     "AUC_pooled_eval": roc_auc_score(y[keep], p[keep])})
        print(f"CV sans {fl}", flush=True)
        _, res = cv_auc(T[keep])
        sens.append({"analyse": f"sans {fl}", "mode": "(b) CV refaite", "n": int(keep.sum()), **res})
    sens = pd.DataFrame(sens).round(4)
    sens.to_csv(OUT / "sensitivity_train.csv", index=False)

    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False))
    (OUT / "summary.json").write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                                                  "n_train": len(T)}, indent=2))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n== Effectifs ==\n", n_sex, "\n\n== Équité ==\n", fair, "\n\n== Marqueurs ==\n", counts,
          "\n\n== Sensibilité ==\n", sens)


if __name__ == "__main__":
    main()
