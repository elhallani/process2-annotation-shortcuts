"""Outils statistiques partagés : ICC, validation croisée répétée, test de permutation."""

import numpy as np
from joblib import Parallel, delayed


def icc(x):
    """ICC(2,1) accord absolu et ICC(3,1) cohérence ; x : n sujets × k mesures."""
    n, k = x.shape
    gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    icc3 = (msr - mse) / (msr + (k - 1) * mse)
    icc2 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    return icc2, icc3


def icc_bootstrap(x, rng, n_boot=1000):
    """ICC(2,1), ICC(3,1) et leurs IC 95 % par bootstrap des sujets."""
    i2, i3 = icc(x)
    boots = np.array([icc(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {"ICC2_1": i2, "ICC2_1_lo": lo[0], "ICC2_1_hi": hi[0],
            "ICC3_1": i3, "ICC3_1_lo": lo[1], "ICC3_1_hi": hi[1]}


def cv_scores(X, y, make_model, multiclass, seed, n_splits=5, n_repeats=5):
    """AUC et accuracy équilibrée par fold (RepeatedStratifiedKFold) ; le modèle est réajusté dans chaque fold."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    aucs, baccs = [], []
    for trn, tst in cv.split(X, y):
        m = make_model().fit(X[trn], y[trn])
        p = m.predict_proba(X[tst])
        aucs.append(roc_auc_score(y[tst], p, multi_class="ovr", average="macro") if multiclass
                    else roc_auc_score(y[tst], p[:, 1]))
        baccs.append(balanced_accuracy_score(y[tst], m.classes_[p.argmax(1)]))
    return np.array(aucs), np.array(baccs)


def nadeau_bengio(diff, n_train, n_test):
    """Test t rééchantillonné corrigé (Nadeau & Bengio 2003) sur les différences par fold (J = répétitions × plis).
    Variance corrigée : (1/J + n_test/n_train) × s². Renvoie (t, p bilatéral)."""
    from scipy.stats import t as student
    d = np.asarray(diff, float)
    J = len(d)
    var = (1 / J + n_test / n_train) * d.var(ddof=1)
    tval = d.mean() / np.sqrt(var) if var > 0 else 0.0
    return tval, 2 * student.sf(abs(tval), J - 1)


def holm(pvals):
    """Correction de Holm–Bonferroni ; renvoie les p ajustés dans l'ordre d'origine."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def permutation_cv(X, y, make_model, multiclass, seed, n_perm=1000, n_jobs=6):
    """Scores observés (moyenne ± SD sur les folds) + p de permutation (étiquettes permutées, mêmes folds)."""
    auc, bacc = cv_scores(X, y, make_model, multiclass, seed)
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(y) for _ in range(n_perm)]
    null = np.array(Parallel(n_jobs=n_jobs)(
        delayed(lambda yp: tuple(s.mean() for s in cv_scores(X, yp, make_model, multiclass, seed)))(yp)
        for yp in perms))
    return {"AUC_mean": auc.mean(), "AUC_sd": auc.std(ddof=1),
            "AUC_p_perm": (1 + (null[:, 0] >= auc.mean()).sum()) / (n_perm + 1),
            "BAcc_mean": bacc.mean(), "BAcc_sd": bacc.std(ddof=1),
            "BAcc_p_perm": (1 + (null[:, 1] >= bacc.mean()).sum()) / (n_perm + 1),
            "null_AUC_p95": np.percentile(null[:, 0], 95), "null_BAcc_p95": np.percentile(null[:, 1], 95)}
