"""Analyses complémentaires et corrections méthodologiques.

TRAIN uniquement, sauf F (marqué EXPLORATOIRE : pipeline figé de configs/final.yaml, aucune décision).
A. Ablation texte à régime de transcription égal : manuel brut vs manuel nettoyé, TF-IDF + LR, 5 × 5 CV,
   sur tous / régime « Pat: » (préfixe présent) / régime sans préfixe.
B. Test t corrigé de Nadeau–Bengio (différences par fold) + permutation appariée ; Holm par famille :
   famille « artefact » (brut − nettoyé, 3 tâches, tous) ; famille « modèle » (C − B, B_nocov − B, B_nocov − A).
C. GLM des pauses annotées sur TOUS les fichiers : interaction diagnostic × régime ; surdispersion (Poisson vs
   binomiale négative).
D. Calibration du modèle principal en CV (probabilités hors-fold moyennées sur les répétitions).
E. G (score clinique) ↔ MMSE chez MCI + Dementia ; AUC séparées HC vs MCI et HC vs Dementia (modèle principal, CV).
F. EXPLORATOIRE (test) : calibration du modèle principal figé ; vérifie que l'AUC test retrouvée = 0,811.
"""

import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import TASKS, load_dataset  # noqa: E402
from src.normative import crossfit_scores  # noqa: E402
from src.stats import holm, nadeau_bengio  # noqa: E402
from src.text import PREFIX, clean_for_model, raw, read_text  # noqa: E402


def _load(name, path, argv=None):
    saved, sys.argv = sys.argv, [path] + (argv or [])
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = saved
    return mod


m10 = _load("m10", "scripts/10_model.py", ["--config", "configs/model_clinical_v2.yaml"])
t04 = m10.t04
OUT = ROOT / "results" / "17_corrections"
SEED, N_PERM = 42, 1000
CLASSES = {"HC": 0, "MCI": 1, "Dementia": 2}


def per_fold_auc(y, P, splits):
    return np.array([roc_auc_score(y[tst], P[r, tst][:, 1]) for r, rep in enumerate(splits) for _, tst in rep])


def compare(name, family, y, P1, P2, splits, rng):
    a1, a2 = per_fold_auc(y, P1, splits), per_fold_auc(y, P2, splits)
    n = len(y)
    t, p_nb = nadeau_bengio(a1 - a2, n * 4 / 5, n / 5)
    obs, p_perm = t04.paired_permutation(y, P1, P2, False, rng, N_PERM)
    return {"comparaison": name, "famille": family, "n": n, "AUC_1_fold_mean": a1.mean(), "AUC_2_fold_mean": a2.mean(),
            "dAUC_fold_mean": (a1 - a2).mean(), "t_NB": t, "p_NadeauBengio": p_nb,
            "dAUC_pooled": obs[0], "p_perm_pooled": p_perm}


def tfidf_oof(texts, y, splits):
    X = pd.DataFrame({"text": texts, "a": 0.0, "b": 0.0})
    P = np.zeros((len(splits), len(y), 2))
    for r, rep in enumerate(splits):
        for trn, tst in rep:
            P[r, tst] = m08.make_model(False).fit(X.iloc[trn], y[trn]).predict_proba(X.iloc[tst])
    return P


m08 = _load("m08", "scripts/08_text_asr.py")


def part_a_b_text(tr, rng):
    conv = pd.read_csv(ROOT / "results" / "cache" / "03_second_voice" / "transcript_markers.csv",
                       dtype={"participant_id": str}).groupby("participant_id").has_speaker_prefix.mean()
    regime = tr.participant_id.map(conv) >= .5
    rows = []
    for t in TASKS:
        manual = [read_text(p) for p in tr[f"txt_{t}"]]
        texts = {"brut": [raw(x) for x in manual], "nettoyé": [clean_for_model(x) for x in manual]}
        for rname, mask in [("tous", np.ones(len(tr), bool)), ("régime « Pat: »", regime.to_numpy()),
                            ("régime sans préfixe", ~regime.to_numpy())]:
            idx = np.where(mask)[0]
            y3 = tr.diagnosis.iloc[idx].map(CLASSES).to_numpy()
            y = (y3 > 0).astype(int)
            splits = t04.folds(y3 if np.bincount(y3).min() >= 5 else y, 5, 5)
            P = {k: tfidf_oof([v[i] for i in idx], y, splits) for k, v in texts.items()}
            r = compare(f"brut − nettoyé, {t}, {rname}", "artefact" if rname == "tous" else "artefact (régime)",
                        y, P["brut"], P["nettoyé"], splits, rng)
            r |= {"task": t, "régime": rname, "n_HC": int((y == 0).sum()), "n_patients": int(y.sum())}
            rows.append(r)
            print(f"  A {t} {rname}: {r['AUC_1_fold_mean']:.3f} → {r['AUC_2_fold_mean']:.3f}", flush=True)
    return pd.DataFrame(rows)


def part_b_models(T, rng):
    P, splits, yb, y3 = m10.run_cv(T)
    rows = [compare("C_G_D − B_G", "modèle", yb, P[("C_G_D", "bin")], P[("B_G", "bin")], splits, rng),
            compare("B_G_nocov − B_G", "modèle", yb, P[("B_G_nocov", "bin")], P[("B_G", "bin")], splits, rng),
            compare("B_G_nocov − A_cov", "modèle", yb, P[("B_G_nocov", "bin")], P[("A_cov", "bin")], splits, rng)]
    return pd.DataFrame(rows), P, splits, yb, y3


def part_c_glm(tr):
    vad = json.loads((ROOT / "results" / "cache" / "01_audit" / "vad_segments.json").read_text())
    rx = re.compile(r"\((\d+)\s*seconds?\)", re.I)
    rows = []
    for p in tr.itertuples():
        for t in TASKS:
            text = read_text(getattr(p, f"txt_{t}"))
            segs = vad[f"{p.participant_id}__{t}"]
            gaps = np.array([b[0] - a[1] for a, b in zip(segs[:-1], segs[1:])])
            rows.append({"participant_id": p.participant_id, "task": t, "diagnosis": p.diagnosis,
                         "regime": int(any(PREFIX.match(l) for l in text.split("\n"))),
                         "n_annot": len(rx.findall(text)), "log_gaps": np.log1p((gaps >= 1.0).sum())})
    d = pd.DataFrame(rows)
    d["diagnosis"] = pd.Categorical(d.diagnosis, ["HC", "MCI", "Dementia"])
    # L'interaction diagnostic × régime est inestimable (≈ 0 annotation hors régime « Pat: » → séparation complète).
    # On estime donc (i) l'assignation au régime selon le diagnostic, (ii) la densité d'annotation dans le régime.
    out = []
    sess = d.groupby("participant_id").agg(regime=("regime", "max"), diagnosis=("diagnosis", "first")).reset_index()
    sess["diagnosis"] = pd.Categorical(sess.diagnosis, ["HC", "MCI", "Dementia"])
    lg = smf.logit("regime ~ diagnosis", sess).fit(disp=0)
    for term in ["diagnosis[T.MCI]", "diagnosis[T.Dementia]"]:
        lo, hi = lg.conf_int().loc[term]
        out.append({"modèle": "Logit : régime « Pat: » ~ diagnostic (sessions)", "terme": term,
                    "RR": np.exp(lg.params[term]), "RR_lo": np.exp(lo), "RR_hi": np.exp(hi), "p": lg.pvalues[term]})
    dp = d[d.regime == 1]
    f = "n_annot ~ log_gaps + C(task) + diagnosis"
    pois = smf.glm(f, dp, family=sm.families.Poisson()).fit()
    disp = pois.pearson_chi2 / pois.df_resid
    nb_alpha = smf.negativebinomial(f, dp).fit(disp=0).params["alpha"]
    m = smf.glm(f, dp, family=sm.families.Poisson()).fit(cov_type="cluster",
                                                         cov_kwds={"groups": pd.factorize(dp.participant_id)[0]})
    for term in ["diagnosis[T.MCI]", "diagnosis[T.Dementia]", "log_gaps"]:
        lo, hi = m.conf_int().loc[term]
        out.append({"modèle": "Poisson dans le régime « Pat: » (fichiers)", "terme": term, "RR": np.exp(m.params[term]),
                    "RR_lo": np.exp(lo), "RR_hi": np.exp(hi), "p": m.pvalues[term]})
    return pd.DataFrame(out), {"poisson_pearson_chi2_per_df_regime_Pat": disp, "negbin_alpha_regime_Pat": nb_alpha, "n_files": len(d),
                                 "n_files_regime_Pat": len(dp), "note": "OR (logit) dans la colonne RR pour le modèle de régime"}


def calib(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    slope_m = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
    icpt_m = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    return {"AUC": roc_auc_score(y, p), "Brier": brier_score_loss(y, p), "calibration_slope": slope_m.params[1],
            "calibration_intercept": icpt_m.params[0], "mean_pred": p.mean(), "observed_rate": y.mean()}


def reliability(y, p, n_bins=5):
    q = pd.qcut(p, n_bins, duplicates="drop")
    return pd.DataFrame({"p_moyenne": pd.Series(p).groupby(q).mean(), "taux_observé": pd.Series(y).groupby(q).mean(),
                         "n": pd.Series(y).groupby(q).size()}).round(3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if "--only-C" in sys.argv:
        tr = load_dataset().query("Split == 'Train'").reset_index(drop=True)
        C, cinfo = part_c_glm(tr)
        C.round(4).to_csv(OUT / "C_glm_interaction_train.csv", index=False)
        s = json.loads((OUT / "summary.json").read_text())
        s["C_overdispersion"] = cinfo
        (OUT / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False, default=float))
        print(C.round(3).to_string(), "\n", json.dumps(cinfo, indent=1, default=float))
        return
    rng = np.random.default_rng(SEED)
    meta = load_dataset()
    tr = meta[meta.Split == "Train"].reset_index(drop=True)
    T = m10.participant_table(tr)

    print("A. ablation texte par régime", flush=True)
    A = part_a_b_text(tr, rng)
    print("B. modèles", flush=True)
    B, P, splits, yb, y3 = part_b_models(T, rng)
    AB = pd.concat([A, B], ignore_index=True)
    for fam in AB.famille.unique():
        m = AB.famille == fam
        AB.loc[m, "p_NB_Holm"] = holm(AB.loc[m, "p_NadeauBengio"])
    AB.round(4).to_csv(OUT / "AB_comparisons_nadeau_bengio_train.csv", index=False)

    print("C. GLM", flush=True)
    C, cinfo = part_c_glm(tr)
    C.round(4).to_csv(OUT / "C_glm_interaction_train.csv", index=False)

    print("D–E. calibration et analyses complémentaires", flush=True)
    p = P[("B_G_nocov", "bin")][..., 1].mean(0)
    D = calib(yb, p)
    reliability(yb, p).to_csv(OUT / "D_reliability_train_cv.csv")
    E = {}
    for name, pos in [("HC_vs_MCI", 1), ("HC_vs_Dementia", 2)]:
        k = np.isin(y3, [0, pos])
        E[f"AUC_{name}_OOF"] = roc_auc_score((y3[k] == pos).astype(int), p[k])
    c = pd.read_csv(ROOT / "results" / "cache" / "11_clinical" / "clinical_per_recording.csv", dtype={"participant_id": str})
    x = c[c.Split == "Train"].pivot(index="participant_id", columns="task", values="valid_per_s").add_prefix("x_")
    mi = tr.set_index("participant_id")
    d = x.join(pd.DataFrame({"age": mi.age.astype(float), "male": (mi.gender == "male").astype(float),
                             "diagnosis": mi.diagnosis, "MMSE": mi.MMSE}))
    sc, _ = crossfit_scores(d, d.diagnosis == "HC", 5, SEED)
    s = sc.astype(float).join(d[["diagnosis", "MMSE"]])
    s = s[s.diagnosis.isin(["MCI", "Dementia"]) & s.MMSE.notna()]
    for idx in ["G", "D", "z_SFT", "z_PFT"]:
        rho, pv = spearmanr(s[idx], s.MMSE)
        E[f"spearman_{idx}_MMSE_MCI+Dem"], E[f"p_{idx}_MMSE"] = rho, pv
    E["n_MMSE_MCI+Dem"] = len(s)

    print("F. EXPLORATOIRE : calibration test", flush=True)
    te = meta[meta.Split == "Test"].reset_index(drop=True)
    Tte = m10.participant_table(te)
    F_tr, F_te = m10.fold_features(T, Tte)
    cols = m10.columns(F_tr, m10.CFG["models"]["B_G_nocov"])
    pt = m10.lr().fit(F_tr[cols].to_numpy(float), yb).predict_proba(F_te[cols].to_numpy(float))[:, 1]
    yt = (Tte.diagnosis != "HC").astype(int).to_numpy()
    Ft = calib(yt, pt)
    reliability(yt, pt, 4).to_csv(OUT / "F_EXPLORATORY_reliability_test.csv")

    summary = {"generated": datetime.now().isoformat(timespec="seconds"), "C_overdispersion": cinfo,
               "D_calibration_train_cv": D, "E": E, "F_EXPLORATORY_test_calibration": Ft,
               "note": "A–E : train uniquement ; F : test, exploratoire, pipeline figé, aucune décision"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(AB.round(3).to_string(), "\n\n", C.round(3).to_string(), "\n\n",
          json.dumps({k: v for k, v in summary.items() if k != "generated"}, indent=1, default=float))


if __name__ == "__main__":
    main()
