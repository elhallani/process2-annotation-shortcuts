"""Modèle normatif HC → indices de dissociation (D) et de niveau global (G).

Ajusté sur des HC uniquement (régression de Huber, prédicteurs standardisés) :
  D        : x_SFT ~ x_PFT + âge + sexe ;  D = résidu / σ_robuste(résidus HC)
  D_simple : (x_SFT − x_PFT) ~ âge + sexe (β = 1 imposé) ; même normalisation
  z_tâche  : x_tâche ~ âge + sexe ; z = résidu / σ_robuste
  G        : moyenne(z_SFT, z_PFT) / σ_robuste de cette moyenne chez les HC
σ_robuste = 1,4826 × MAD. Les HC de l'échantillon d'ajustement reçoivent des scores croisés (cross-fitting) :
le score d'un HC est calculé par un modèle ajusté sans lui.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COVARIATES = ["age", "male"]


def robust_sd(r):
    r = np.asarray(r, float)
    return 1.4826 * np.median(np.abs(r - np.median(r)))


def _huber(epsilon):
    return make_pipeline(StandardScaler(), HuberRegressor(epsilon=epsilon, max_iter=1000))


class NormativeModel:
    def __init__(self, epsilon=1.35):
        self.epsilon = epsilon

    def fit(self, hc):
        """hc : DataFrame avec x_SFT, x_PFT, age, male (HC uniquement)."""
        e = self.epsilon
        self.m_d = _huber(e).fit(hc[["x_PFT", *COVARIATES]], hc.x_SFT)
        self.m_diff = _huber(e).fit(hc[COVARIATES], hc.x_SFT - hc.x_PFT)
        self.m_z = {t: _huber(e).fit(hc[COVARIATES], hc[f"x_{t}"]) for t in ("SFT", "PFT")}
        self.sd_d = robust_sd(hc.x_SFT - self.m_d.predict(hc[["x_PFT", *COVARIATES]]))
        self.sd_diff = robust_sd(hc.x_SFT - hc.x_PFT - self.m_diff.predict(hc[COVARIATES]))
        self.sd_z = {t: robust_sd(hc[f"x_{t}"] - self.m_z[t].predict(hc[COVARIATES])) for t in ("SFT", "PFT")}
        zbar = self._z(hc).mean(axis=1)
        self.sd_g = robust_sd(zbar)
        return self

    def _z(self, df):
        return pd.DataFrame({f"z_{t}": (df[f"x_{t}"] - self.m_z[t].predict(df[COVARIATES])) / self.sd_z[t]
                             for t in ("SFT", "PFT")}, index=df.index)

    def scores(self, df):
        z = self._z(df)
        return pd.DataFrame({
            "D": (df.x_SFT - self.m_d.predict(df[["x_PFT", *COVARIATES]])) / self.sd_d,
            "D_simple": (df.x_SFT - df.x_PFT - self.m_diff.predict(df[COVARIATES])) / self.sd_diff,
            "G": z.mean(axis=1) / self.sd_g, **z}, index=df.index)

    def coefficients(self):
        """Coefficients dans l'échelle d'origine (β de x_PFT, γ âge, γ sexe) du modèle D."""
        sc, hub = self.m_d.named_steps["standardscaler"], self.m_d.named_steps["huberregressor"]
        coef = hub.coef_ / sc.scale_
        return dict(zip(["beta_PFT", "gamma_age", "gamma_male"], coef)) | {"sd_resid_D": self.sd_d}


def crossfit_scores(df, is_hc, n_splits=5, seed=42, epsilon=1.35):
    """Scores pour tout df : HC par cross-fitting (K plis sur les HC), non-HC par le modèle ajusté sur tous les HC."""
    out = pd.DataFrame(index=df.index, columns=["D", "D_simple", "G", "z_SFT", "z_PFT"], dtype=float)
    hc = df[is_hc]
    full = NormativeModel(epsilon).fit(hc)
    if (~is_hc).any():
        out.loc[~is_hc] = full.scores(df[~is_hc]).values
    for trn, tst in KFold(n_splits, shuffle=True, random_state=seed).split(hc):
        m = NormativeModel(epsilon).fit(hc.iloc[trn])
        out.loc[hc.index[tst]] = m.scores(hc.iloc[tst]).values
    return out, full
