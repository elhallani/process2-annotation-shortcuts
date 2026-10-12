"""History of the successive decision-model configurations (training-set cross-validation).

Reads existing results and trains nothing. Output: results/20_config_history/config_history.csv."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("1", "2026-10-02", "configs/model.yaml", "results/10_model", "Unités VAD (purement acoustique), figé avant calcul", ["B_G", "C_G_D"]),
    ("2", "2026-10-02", "configs/model_clinical.yaml", "results/10_model_clinical",
     "Avenant 1 : score clinique Whisper (D n'ajoutait rien avec VAD)", ["B_G", "C_G_D"]),
    ("3", "2026-10-02", "configs/model_clinical_v2.yaml", "results/10_model_clinical_v2",
     "Avenant 2 : âge/sexe retirés du classifieur (parcimonie) → modèle principal B_G_nocov", ["B_G_nocov", "B_G", "C_G_D"]),
    ("S", "2026-10-02", "configs/model_clinical_v2_60s.yaml", "results/10_model_clinical_v2_60s",
     "Sensibilité (après test, train) : fenêtre 60 s", ["B_G_nocov", "B_G", "C_G_D"]),
]


def main():
    out = ROOT / "results" / "20_config_history"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for step, date, cfg, res, why, models in STEPS:
        r = pd.read_csv(ROOT / res / "cv_results_train.csv")
        for m in models:
            b = r[(r["modèle"] == m) & (r.cible == "HC vs MCI+Dementia")].iloc[0]
            t = r[(r["modèle"] == m) & (r.cible == "3 classes")].iloc[0]
            rows.append({"étape": step, "date": date, "config": cfg, "motif": why, "modèle": m,
                         "AUC_2c": round(b.AUC_mean, 3), "AUC_2c_sd": round(b.AUC_sd, 3), "BAcc_2c": round(b.BAcc_mean, 3),
                         "AUC_3c": round(t.AUC_mean, 3), "macroF1_3c": round(t.macroF1_mean, 3)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "config_history.csv", index=False)
    print(df.drop(columns=["config", "date"]).to_string(index=False))


if __name__ == "__main__":
    main()
