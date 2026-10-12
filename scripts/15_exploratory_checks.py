"""Vérifications EXPLORATOIRES (TRAIN uniquement, ajoutées après l'évaluation du test ; aucune décision n'en dépend).

A. Corrélation SFT–PFT chez les HC avec les comptages MANUELS (mots valides, enregistrement entier), comparée aux
   comptages Whisper (enregistrement entier et fenêtre 45 s) ; brute et partielle (âge, sexe).
B. AUC (HC vs MCI+Dementia) du comptage Whisper vs du comptage manuel, par tâche : différence appariée,
   IC bootstrap (2000) et permutation appariée (1000, échange des scores par participant).
E. Gain de la fusion (B_G + WavLM centré) sur B_G en 3 classes : macro-F1, accuracy équilibrée et AUC sur les
   prédictions hors-fold (mêmes 5 × 5 folds que 12_wavlm), permutation appariée (1000).
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import FLUENCY_TASKS, load_dataset  # noqa: E402


def _load(name, path, argv=None):
    saved, sys.argv = sys.argv, [path] + (argv or [])
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = saved
    return mod


m12 = _load("m12", "scripts/12_wavlm.py")
m10, t04 = m12.m10, m12.t04
OUT = ROOT / "results" / "15_exploratory"
SEED, N_PERM, N_BOOT = 42, 1000, 2000


def partial_r(x, y, Z):
    Z = np.column_stack([np.ones(len(x)), Z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return pearsonr(rx, ry)[0]


def check_a(c, meta):
    rows = []
    for col, label in [("valid_full_manual", "manuel, enregistrement entier"),
                       ("valid_full_asr", "Whisper, enregistrement entier"), ("valid_45s", "Whisper, fenêtre 45 s")]:
        w = c.pivot(index="participant_id", columns="task", values=col).join(meta)
        for d in ["HC", "MCI", "Dementia"]:
            s = w[w.diagnosis == d]
            rows.append({"comptage": label, "groupe": d, "n": len(s),
                         "pearson_r": pearsonr(s.SFT, s.PFT)[0], "spearman_rho": spearmanr(s.SFT, s.PFT)[0],
                         "partial_r_age_sexe": partial_r(s.SFT.to_numpy(float), s.PFT.to_numpy(float),
                                                         s[["age", "male"]].to_numpy(float))})
    return pd.DataFrame(rows).round(3)


def check_b(c, rng):
    rows = []
    for t in FLUENCY_TASKS:
        s = c[c.task == t]
        y = (s.diagnosis != "HC").astype(int).to_numpy()
        a, m = -s.valid_full_asr.to_numpy(float), -s.valid_full_manual.to_numpy(float)  # moins de mots = patient
        obs = roc_auc_score(y, a) - roc_auc_score(y, m)
        boots = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            if len(set(y[i])) == 2:
                boots.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], m[i]))
        null = []
        for _ in range(N_PERM):
            sw = rng.random(len(y)) < .5
            a2, m2 = np.where(sw, m, a), np.where(sw, a, m)
            null.append(roc_auc_score(y, a2) - roc_auc_score(y, m2))
        rows.append({"task": t, "n": len(y), "AUC_whisper": roc_auc_score(y, a), "AUC_manuel": roc_auc_score(y, m),
                     "dAUC_whisper_moins_manuel": obs, "IC_lo": np.percentile(boots, 2.5),
                     "IC_hi": np.percentile(boots, 97.5),
                     "p_perm": (1 + (np.abs(null) >= abs(obs)).sum()) / (N_PERM + 1)})
    return pd.DataFrame(rows).round(3)


def pooled_f1(y, P):
    return np.mean([f1_score(y, p.argmax(1), average="macro") for p in P])


def check_e(meta, rng):
    tr = meta.reset_index(drop=True)
    T = m10.participant_table(tr)
    E = m12.extract(None)
    R = m12.representations(E, T.index)["wavlm_centered"]
    y3 = T.diagnosis.map(m12.CLASSES).to_numpy()
    cv = m12.CONFIG["cv"]
    splits = t04.folds(y3, cv["n_splits"], cv["n_repeats"])
    P = {k: np.zeros((cv["n_repeats"], len(T), 3)) for k in ("B_G", "fusion")}
    for r, rep in enumerate(splits):
        for trn, tst in rep:
            F_tr, F_te = m10.fold_features(T.iloc[trn], T.iloc[tst])
            cols = m10.columns(F_tr, m10.CFG["models"]["B_G"])
            a, b = F_tr[cols].to_numpy(float), F_te[cols].to_numpy(float)
            P["B_G"][r, tst] = m10.FrankHall().fit(a, y3[trn]).predict_proba(b)
            P["fusion"][r, tst] = m12.Fusion(len(cols), True).fit(np.hstack([a, R[trn]]), y3[trn]).predict_proba(
                np.hstack([b, R[tst]]))
    obs_auc_bacc, p_auc_bacc = t04.paired_permutation(y3, P["fusion"], P["B_G"], True, rng, N_PERM)
    obs_f1 = pooled_f1(y3, P["fusion"]) - pooled_f1(y3, P["B_G"])
    null = []
    for _ in range(N_PERM):
        sw = rng.random(len(y3)) < .5
        Q1, Q2 = P["fusion"].copy(), P["B_G"].copy()
        Q1[:, sw], Q2[:, sw] = P["B_G"][:, sw], P["fusion"][:, sw]
        null.append(pooled_f1(y3, Q1) - pooled_f1(y3, Q2))
    return pd.DataFrame([{"comparaison": "fusion − B_G (3 classes, OOF regroupé)",
                          "macroF1_fusion": pooled_f1(y3, P["fusion"]), "macroF1_B_G": pooled_f1(y3, P["B_G"]),
                          "d_macroF1": obs_f1, "p_macroF1": (1 + (np.abs(null) >= abs(obs_f1)).sum()) / (N_PERM + 1),
                          "dAUC": obs_auc_bacc[0], "p_AUC": p_auc_bacc[0],
                          "dBAcc": obs_auc_bacc[1], "p_BAcc": p_auc_bacc[1]}]).round(4)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    meta = load_dataset()
    meta = meta[meta.Split == "Train"]
    mi = meta.set_index("participant_id")[["diagnosis", "age"]].assign(
        male=(meta.set_index("participant_id").gender == "male").astype(float), age=lambda d: d.age.astype(float))
    c = pd.read_csv(ROOT / "results" / "cache" / "11_clinical" / "clinical_per_recording.csv",
                    dtype={"participant_id": str})
    c = c[c.Split == "Train"]
    a, b = check_a(c, mi), check_b(c, rng)
    e = check_e(meta, rng)
    a.to_csv(OUT / "A_sft_pft_correlation_train.csv", index=False)
    b.to_csv(OUT / "B_whisper_vs_manual_auc_train.csv", index=False)
    e.to_csv(OUT / "E_fusion_3class_permutation_train.csv", index=False)
    (OUT / "README.json").write_text(json.dumps({"statut": "exploratoire, train uniquement, post-test",
                                                 "seed": SEED, "n_perm": N_PERM, "n_boot": N_BOOT}, indent=2,
                                                ensure_ascii=False))
    pd.set_option("display.width", 220)
    print("== A ==\n", a, "\n\n== B ==\n", b, "\n\n== E ==\n", e.T)


if __name__ == "__main__":
    main()
