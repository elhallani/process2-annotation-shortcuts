"""Étape 4 — Modèle de décision (TRAIN uniquement, configs/model.yaml figé avant calcul).

Dans chaque fold (5 × 5, mêmes folds pour tous les modèles) :
  - normes HC réapprises sur les HC du train du fold (Huber) ; HC du train du fold scorés par cross-fitting interne ;
    participants du fold de test scorés par le modèle ajusté sur tous les HC du train du fold ;
  - indices G et D (4 mesures de fluence), z_SFT / z_PFT, z CTD (2 mesures) ;
  - régression logistique pondérée (2 classes) et ordinale Frank & Hall (3 classes).
Comparaison principale appariée : C_G_D vs B_G (D apporte-t-il quelque chose à G ?), test de permutation.
Sorties agrégées : results/10_model/.
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_dataset  # noqa: E402
from src.normative import COVARIATES, crossfit_scores, robust_sd  # noqa: E402

spec = importlib.util.spec_from_file_location("t04", ROOT / "scripts" / "04_text_artifact.py")
t04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t04)

_ap = argparse.ArgumentParser()
_ap.add_argument("--config", default="configs/model.yaml")
CFG = yaml.safe_load((ROOT / _ap.parse_args().config).read_text())
OUT = ROOT / CFG.get("output_dir", "results/10_model")
TRANSFORMS = {"none": lambda x: x, "log": np.log, "log1p": np.log1p}
CLASSES = {"HC": 0, "MCI": 1, "Dementia": 2}


# ---------------------------------------------------------------- données

def participant_table(meta):
    f = pd.read_csv(ROOT / "results" / "cache" / "05_features" / "features_per_recording.csv",
                    dtype={"participant_id": str})
    for path in CFG.get("extra_feature_files", []):  # mesures supplémentaires par (participant, tâche)
        extra = pd.read_csv(ROOT / path, dtype={"participant_id": str})
        f = f.merge(extra[["participant_id", "task", *[c for c in extra.columns if c in CFG["fluency_measures"]]]],
                    on=["participant_id", "task"], how="left")
    f = f[f.participant_id.isin(meta.participant_id)]
    t = meta.set_index("participant_id")[["diagnosis", "age"]].assign(
        male=(meta.set_index("participant_id").gender == "male").astype(float), age=lambda d: d.age.astype(float))
    for m, tr in CFG["fluency_measures"].items():
        w = f.pivot(index="participant_id", columns="task", values=m)
        for task in ("SFT", "PFT"):
            t[f"{m}__x_{task}"] = TRANSFORMS[tr](w[task])
    ctd = f[f.task == "CTD"].set_index("participant_id")
    for m, tr in CFG["ctd_measures"].items():
        t[f"ctd__{m}"] = TRANSFORMS[tr](ctd[m])
    return t


# ---------------------------------------------------------------- normes dans le fold

def z_single(train, test, col, is_hc, eps, n_splits, seed):
    """z d'une mesure vs HC (âge, sexe) : HC du train par cross-fitting, autres et test par le modèle complet."""
    def fit(d):
        d = d.dropna(subset=[col])
        m = make_pipeline(StandardScaler(), HuberRegressor(epsilon=eps, max_iter=1000)).fit(d[COVARIATES], d[col])
        return m, robust_sd(d[col] - m.predict(d[COVARIATES]))

    def score(m_sd, d):
        m, sd = m_sd
        return (d[col] - m.predict(d[COVARIATES])) / sd

    hc = train[is_hc]
    full = fit(hc)
    z_tr = score(full, train)
    for trn, tst in KFold(n_splits, shuffle=True, random_state=seed).split(hc):
        z_tr.loc[hc.index[tst]] = score(fit(hc.iloc[trn]), hc.iloc[tst])
    return z_tr, score(full, test)


def fold_features(train, test):
    n = CFG["normative"]
    eps, k, seed = n["huber_epsilon"], n["inner_crossfit_splits"], CFG["seed"]
    is_hc = train.diagnosis == "HC"
    F_tr, F_te = pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
    for m in CFG["fluency_measures"]:
        cols = {f"{m}__x_SFT": "x_SFT", f"{m}__x_PFT": "x_PFT"}
        tr_m, te_m = train[[*cols, *COVARIATES]].rename(columns=cols), test[[*cols, *COVARIATES]].rename(columns=cols)
        s_tr, full = crossfit_scores(tr_m, is_hc, k, seed, eps)
        s_te = full.scores(te_m)
        for idx in ["G", "D", "z_SFT", "z_PFT"]:
            F_tr[f"{idx}__{m}"], F_te[f"{idx}__{m}"] = s_tr[idx].astype(float), s_te[idx].astype(float)
    for m in CFG["ctd_measures"]:
        F_tr[f"ctd__{m}"], F_te[f"ctd__{m}"] = z_single(train, test, f"ctd__{m}", is_hc, eps, k, seed)
    for c in COVARIATES:
        F_tr[f"cov__{c}"], F_te[f"cov__{c}"] = train[c], test[c]
    return F_tr, F_te


def columns(F, blocks):
    prefix = {"cov": "cov__", "age": "cov__age", "G": "G__", "D": "D__", "ctd": "ctd__", "z_SFT": "z_SFT__",
              "z_PFT": "z_PFT__"}
    return [c for b in blocks for c in F.columns if c.startswith(prefix[b])]


# ---------------------------------------------------------------- classifieurs

def lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", C=1.0, max_iter=5000))


class FrankHall:
    """Ordinal (Frank & Hall 2001) : P(y > HC), P(y > MCI) par deux LR pondérées."""
    def fit(self, X, y):
        self.m1, self.m2 = lr().fit(X, (y > 0).astype(int)), lr().fit(X, (y > 1).astype(int))
        return self

    def predict_proba(self, X):
        p1, p2 = self.m1.predict_proba(X)[:, 1], self.m2.predict_proba(X)[:, 1]
        p = np.column_stack([1 - p1, np.clip(p1 - p2, 0, None), p2])
        return p / p.sum(1, keepdims=True)


# ---------------------------------------------------------------- évaluation

def run_cv(T):
    y3 = T.diagnosis.map(CLASSES).to_numpy()
    yb = (y3 > 0).astype(int)
    cv = CFG["cv"]
    splits = t04.folds(y3, cv["n_splits"], cv["n_repeats"])  # stratifié sur 3 classes, partagé par les 2 cibles
    P = {(m, t): np.zeros((cv["n_repeats"], len(T), 2 if t == "bin" else 3))
         for m in CFG["models"] for t in ("bin", "3c")}
    for r, rep in enumerate(splits):
        for trn, tst in rep:
            F_tr, F_te = fold_features(T.iloc[trn], T.iloc[tst])
            for m, blocks in CFG["models"].items():
                cols = columns(F_tr, blocks)
                Xtr, Xte = F_tr[cols].to_numpy(float), F_te[cols].to_numpy(float)
                P[(m, "bin")][r, tst] = lr().fit(Xtr, yb[trn]).predict_proba(Xte)
                P[(m, "3c")][r, tst] = FrankHall().fit(Xtr, y3[trn]).predict_proba(Xte)
        print(f"  répétition {r + 1}/{cv['n_repeats']} faite", flush=True)
    return P, splits, yb, y3


def macro_f1(y, P, splits):
    return np.array([f1_score(y[tst], P[r, tst].argmax(1), average="macro")
                     for r, rep in enumerate(splits) for _, tst in rep])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_dataset()
    meta = meta[meta.Split == CFG["split"]].reset_index(drop=True)
    T = participant_table(meta)
    P, splits, yb, y3 = run_cv(T)
    rng = np.random.default_rng(CFG["seed"])
    rows, diffs = [], []
    for m in CFG["models"]:
        for t, y, multi in [("HC vs MCI+Dementia", yb, False), ("3 classes", y3, True)]:
            key = (m, "bin" if not multi else "3c")
            auc, bacc = t04.fold_metrics(y, P[key], multi, splits)
            row = {"modèle": m, "cible": t, "AUC_mean": auc.mean(), "AUC_sd": auc.std(ddof=1),
                   "BAcc_mean": bacc.mean(), "BAcc_sd": bacc.std(ddof=1)}
            if multi:
                f1 = macro_f1(y, P[key], splits)
                row |= {"macroF1_mean": f1.mean(), "macroF1_sd": f1.std(ddof=1)}
            rows.append(row)
    comparisons = [CFG["main_comparison"], ["B_G", "A_cov"], ["C_G_D", "A_cov"], ["D_only", "A_cov"],
                   ["Z_tasks", "C_G_D"]]
    for a, b in comparisons:
        for t, y, multi, tk in [("HC vs MCI+Dementia", yb, False, "bin"), ("3 classes", y3, True, "3c")]:
            obs, p = t04.paired_permutation(y, P[(a, tk)], P[(b, tk)], multi, rng, CFG["n_permutations"])
            diffs.append({"comparaison": f"{a} − {b}", "cible": t, "dAUC": obs[0], "p_dAUC": p[0],
                          "dBAcc": obs[1], "p_dBAcc": p[1]})
    res = pd.DataFrame(rows).round(4)
    dif = pd.DataFrame(diffs).round(4)
    res.to_csv(OUT / "cv_results_train.csv", index=False)
    dif.to_csv(OUT / "paired_comparisons_train.csv", index=False)

    # coefficients standardisés du modèle principal ajusté sur tout le train (normes par cross-fitting)
    F_all, _ = fold_features(T, T.iloc[:1])
    cols = columns(F_all, CFG["models"]["C_G_D"])
    mdl = lr().fit(F_all[cols].to_numpy(float), yb)
    coef = pd.Series(mdl.named_steps["logisticregression"].coef_[0], index=cols).round(3)
    coef.to_csv(OUT / "coefficients_C_G_D_binary_train.csv", header=["coef_standardise"])

    (OUT / "config_used.yaml").write_text(yaml.safe_dump(CFG, allow_unicode=True, sort_keys=False))
    (OUT / "summary.json").write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                                                  "n_train": len(T), "n_features_C_G_D": len(cols)}, indent=2))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(res, "\n\n", dif, "\n\n", coef)


if __name__ == "__main__":
    main()
