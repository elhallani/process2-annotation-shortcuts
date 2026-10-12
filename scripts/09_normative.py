"""Étapes 2–3 — Modèle normatif HC et indices D (dissociation SFT–PFT) / G (niveau global). TRAIN uniquement.

Normes ajustées sur les HC du train (Huber), scores des HC par cross-fitting (src/normative.py).
Descriptif seulement : aucune décision ni seuil n'est choisi ici ; l'évaluation prédictive (normes réapprises
dans chaque fold) relève de l'étape 4/6.
Sorties agrégées : results/09_normative/ ; scores par participant : results/cache/09_normative/.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_dataset  # noqa: E402
from src.normative import crossfit_scores  # noqa: E402

OUT = ROOT / "results" / "09_normative"
CACHE = ROOT / "results" / "cache" / "09_normative"
CONFIG = {
    "seed": 42, "split": "Train", "huber_epsilon": 1.35, "crossfit_splits": 5, "n_boot": 1000,
    # fixé avant tout résultat : mesure principale + mesures secondaires et leur transformation
    "primary": "units_per_s",
    "measures": {"units_per_s": "none", "median_gap_s": "log", "long_silences_per_min": "log1p",
                 "units_slope": "none"},
    "channel_vars": ["noise_db", "snr_db", "bw95_hz"],
    "dissociation_threshold": 1.96,
}
TRANSFORMS = {"none": lambda x: x, "log": np.log, "log1p": np.log1p}
INDICES = ["D", "D_simple", "G", "z_SFT", "z_PFT"]


def auc_ci(y, s, rng, n_boot):
    a = roc_auc_score(y, s)
    y, s = np.asarray(y), np.asarray(s)
    b = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            b.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(b, [2.5, 97.5])
    return a, lo, hi


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(CONFIG["seed"])
    meta = load_dataset()
    meta = meta[meta.Split == CONFIG["split"]].set_index("participant_id")
    feats = pd.read_csv(ROOT / "results" / "cache" / "05_features" / "features_per_recording.csv",
                        dtype={"participant_id": str})
    feats = feats[feats.participant_id.isin(meta.index)]
    chan = (pd.read_csv(ROOT / "results" / "cache" / "01_audit" / "recording_features.csv", dtype={"participant_id": str})
            .groupby("participant_id")[CONFIG["channel_vars"]].mean())
    base = pd.DataFrame({"age": meta.age.astype(float), "male": (meta.gender == "male").astype(float),
                         "diagnosis": meta.diagnosis})
    short = feats[feats.short_recording].participant_id.nunique()

    fits, groups, dissoc, robust, all_scores = [], [], [], [], []
    for meas, tr in CONFIG["measures"].items():
        w = feats.pivot(index="participant_id", columns="task", values=meas)[["SFT", "PFT"]]
        df = base.join(TRANSFORMS[tr](w).add_prefix("x_")).replace([np.inf, -np.inf], np.nan).dropna()
        is_hc = df.diagnosis == "HC"
        sc, full = crossfit_scores(df, is_hc, CONFIG["crossfit_splits"], CONFIG["seed"], CONFIG["huber_epsilon"])
        sc = sc.astype(float).join(df[["diagnosis"]])
        all_scores.append(sc.add_prefix(f"{meas}__").rename(columns={f"{meas}__diagnosis": "diagnosis"}))
        hc = df[is_hc]
        fits.append({"measure": meas, "transform": tr, "n": len(df), "n_HC": int(is_hc.sum()),
                     "n_dropped_missing": len(base) - len(df),
                     "r_SFT_PFT_HC": np.corrcoef(hc.x_SFT, hc.x_PFT)[0, 1], **full.coefficients()})
        # 1. indices par diagnostic + AUC (orientée : > 0,5 = plus élevé chez les patients)
        for idx in INDICES + ["absD"]:
            s = sc.D.abs() if idx == "absD" else sc[idx]
            row = {"measure": meas, "index": idx}
            for d in ["HC", "MCI", "Dementia"]:
                v = s[sc.diagnosis == d]
                row |= {f"median_{d}": v.median(), f"IQR_{d}": v.quantile(.75) - v.quantile(.25)}
            for name, pos in [("HC_vs_MCI+Dem", ["MCI", "Dementia"]), ("HC_vs_Dem", ["Dementia"]),
                              ("HC_vs_MCI", ["MCI"])]:
                m = sc.diagnosis.isin(["HC", *pos])
                a, lo, hi = auc_ci(sc.diagnosis[m].isin(pos).astype(int), s[m], rng, CONFIG["n_boot"])
                row |= {f"AUC_{name}": a, f"AUC_{name}_lo": lo, f"AUC_{name}_hi": hi}
            groups.append(row)
        # 2. dissociation bidirectionnelle : |D| > 1,96 par groupe et par sens
        thr = CONFIG["dissociation_threshold"]
        for d in ["HC", "MCI", "Dementia"]:
            v = sc.D[sc.diagnosis == d]
            dissoc.append({"measure": meas, "diagnosis": d, "n": len(v),
                           "pct_D_above": 100 * (v > thr).mean(), "pct_D_below": 100 * (v < -thr).mean()})
        # 3. robustesse chez les HC : lien avec le canal et le sexe (indices et mesures brutes)
        h = sc[is_hc].join(chan).join(df[["x_SFT", "x_PFT", "male"]])
        for idx in INDICES + ["x_SFT", "x_PFT"]:
            row = {"measure": meas, "index": idx}
            for c in CONFIG["channel_vars"]:
                row[f"rho_{c}"] = spearmanr(h[idx], h[c], nan_policy="omit")[0]
            row["AUC_sex"] = roc_auc_score(h.male, h[idx])
            robust.append(row)

    fit = pd.DataFrame(fits).round(4)
    grp = pd.DataFrame(groups).round(3)
    dis = pd.DataFrame(dissoc).round(1)
    rob = pd.DataFrame(robust).round(3)
    fit.to_csv(OUT / "normative_fit_HC_train.csv", index=False)
    grp.to_csv(OUT / "indices_by_diagnosis_train.csv", index=False)
    dis.to_csv(OUT / "dissociation_rates_train.csv", index=False)
    rob.to_csv(OUT / "robustness_HC_channel_sex_train.csv", index=False)
    pd.concat(all_scores, axis=1).loc[:, lambda d: ~d.columns.duplicated()].to_csv(CACHE / "scores_train.csv")
    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False))
    (OUT / "summary.json").write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                                                  "n_train": len(base), "n_participants_short_fluency": int(short)},
                                                 indent=2))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("== Ajustement normatif ==\n", fit)
    print("\n== Indices par diagnostic ==\n", grp[["measure", "index", "median_HC", "median_MCI", "median_Dementia",
                                                   "AUC_HC_vs_MCI+Dem", "AUC_HC_vs_MCI+Dem_lo", "AUC_HC_vs_MCI+Dem_hi",
                                                   "AUC_HC_vs_Dem", "AUC_HC_vs_MCI"]])
    print("\n== Dissociation |D| > 1,96 ==\n", dis)
    print("\n== Robustesse (HC) ==\n", rob)


if __name__ == "__main__":
    main()
