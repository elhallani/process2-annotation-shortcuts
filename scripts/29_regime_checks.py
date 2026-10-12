"""Checks of the transcription-regime shortcut. Parts A-G and I: training set. Part H: test set,
exploratory (frozen pipeline of configs/final.yaml; no decision).

A. Session-level regime: mixed sessions, percentages by group, odds ratio of regime assignment by diagnosis.
B. Decision model within each regime; does its score predict the regime at equal diagnosis?
C. Is the regime predictable from audio at equal diagnosis (channel, model predictors, non-speech embeddings)?
D. Is the regime predictable from cleaned transcripts (word and character TF-IDF)?
E. Multi-task TF-IDF (concatenation and late fusion) on the folds of the decision model.
F. 95% CIs of the DistilBERT raw - cleaned differences.
G. Descriptive statistics for below-chance single-variable AUCs.
I. Style traits of cleaned transcripts by regime and a strict cleaning (regime and diagnosis prediction).
H. Test-set composition, models within each regime, three-class confusion matrix.
Aggregated outputs only."""

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as st
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import TASKS, load_dataset  # noqa: E402
from src.stats import holm, nadeau_bengio  # noqa: E402
from src.text import clean_for_model, raw, read_text  # noqa: E402


def _load(name, path, argv=None):
    saved, sys.argv = sys.argv, [path] + (argv or [])
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = saved
    return mod


m10 = _load("m10", "scripts/10_model.py", ["--config", "configs/model_clinical_v2.yaml"])
m08 = _load("m08", "scripts/08_text_asr.py")
t04 = m10.t04
OUT = ROOT / "results" / "29_regime_checks"
CACHE = ROOT / "results" / "cache"
ASR_DIR = ROOT / "data" / "derived" / "asr_whisper"
SEED, N_BOOT = 42, 2000
CLASSES = {"HC": 0, "MCI": 1, "Dementia": 2}
VAD = json.loads((CACHE / "01_audit" / "vad_segments.json").read_text())


def boot_auc(y, p, rng, n=N_BOOT):
    a = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            a.append(roc_auc_score(y[i], p[i]))
    return np.percentile(a, [2.5, 97.5])


def regime_table(split):
    m = pd.read_csv(CACHE / "03_second_voice" / "transcript_markers.csv", dtype={"participant_id": str})
    m = m[m.Split == split]
    s = m.groupby("participant_id").agg(n_prefix=("has_speaker_prefix", "sum"), diagnosis=("diagnosis", "first"))
    s["regime"] = (s.n_prefix >= 2).astype(int)        # définition retenue : majorité des 3 fichiers
    s["regime_any"] = (s.n_prefix >= 1).astype(int)
    s["mixed"] = s.n_prefix.isin([1, 2]).astype(int)
    return s


# ---------------------------------------------------------------- A

def part_a(R):
    rows = [{"diagnosis": g, "n_sessions": len(d), "pct_sessions_regime_Pat_majorité": 100 * d.regime.mean(),
             "pct_sessions_regime_Pat_au_moins_un": 100 * d.regime_any.mean(), "n_sessions_mixtes": int(d.mixed.sum()),
             "n_prefix_0_1_2_3": "/".join(str(int((d.n_prefix == k).sum())) for k in range(4))}
            for g, d in R.groupby("diagnosis")]
    d = R.assign(diagnosis=pd.Categorical(R.diagnosis, ["HC", "MCI", "Dementia"]))
    ors = []
    for col in ("regime", "regime_any"):
        f = smf.logit(f"{col} ~ diagnosis", d).fit(disp=0)
        ci = f.conf_int()
        for term in ("diagnosis[T.MCI]", "diagnosis[T.Dementia]"):
            ors.append({"définition": col, "terme": term, "OR": np.exp(f.params[term]),
                        "OR_lo": np.exp(ci.loc[term, 0]), "OR_hi": np.exp(ci.loc[term, 1]), "p": f.pvalues[term]})
    return pd.DataFrame(rows), pd.DataFrame(ors)


# ---------------------------------------------------------------- B

def part_b(T, R, yb, rng):
    P_main = np.load(CACHE / "23_text_models" / "P_main.npy")[..., 1]   # 5 répétitions × 320
    reg = R.loc[T.index, "regime"].to_numpy()
    p_mean = P_main.mean(0)
    rows = []
    for name, k in [("tous", np.ones(len(yb), bool)), ("régime « Pat: »", reg == 1), ("régime sans préfixe", reg == 0)]:
        a = [roc_auc_score(yb[k], P_main[r, k]) for r in range(len(P_main))]
        lo, hi = boot_auc(yb[k], p_mean[k], rng)
        rows.append({"sous-ensemble": name, "n_HC": int((yb[k] == 0).sum()), "n_patients": int(yb[k].sum()),
                     "AUC_principal_moy_rép": np.mean(a), "AUC_lo": lo, "AUC_hi": hi})
    ref = pd.read_csv(ROOT / "results" / "17_corrections" / "AB_comparisons_nadeau_bengio_train.csv")
    # score du modèle → régime, à diagnostic égal
    lk = []
    for g, k in [("HC", yb == 0), ("MCI+Dementia", yb == 1)]:
        lo, hi = boot_auc(reg[k], p_mean[k], rng)
        lk.append({"groupe": g, "n_Pat": int(reg[k].sum()), "n_sans": int((reg[k] == 0).sum()),
                   "AUC_score_principal→régime": roc_auc_score(reg[k], p_mean[k]), "AUC_lo": lo, "AUC_hi": hi})
    return pd.DataFrame(rows), pd.DataFrame(lk), ref


# ---------------------------------------------------------------- C, D

def regime_cv(X, reg, make, n_rep=5):
    """AUC de prédiction du régime, CV 5 × 5 stratifiée sur le régime ; X matrice ou DataFrame."""
    a = []
    for r in range(n_rep):
        for trn, tst in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(np.zeros(len(reg)), reg):
            Xtr = X.iloc[trn] if hasattr(X, "iloc") else X[trn]
            Xte = X.iloc[tst] if hasattr(X, "iloc") else X[tst]
            a.append(roc_auc_score(reg[tst], make().fit(Xtr, reg[trn]).predict_proba(Xte)[:, 1]))
    return np.mean(a), np.std(a, ddof=1)


def lr_dense():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))


def part_c(T, R, yb):
    a24 = _load("a24", "scripts/24_acoustic_models.py")
    reg = R.loc[T.index, "regime"].to_numpy()
    feats = pd.read_csv(CACHE / "01_audit" / "recording_features.csv", dtype={"participant_id": str})
    ch = feats.pivot(index="participant_id", columns="task", values=["noise_db", "snr_db", "bw95_hz"])
    blocks = {"canal (bruit, SNR, bw95 × 3 tâches)": ch.loc[T.index].to_numpy(float)}
    F, _ = m10.fold_features(T, T.iloc[:1])
    blocks["variables du modèle principal (6)"] = F[m10.columns(F, m10.CFG["models"]["B_G_nocov"])].to_numpy(float)
    dur = feats.pivot(index="participant_id", columns="task", values="duration_s")
    blocks["durées (3 tâches)"] = np.log(dur.loc[T.index].to_numpy(float))
    emb = {}
    z = np.load(CACHE / "24_acoustic_models" / "features.npz", allow_pickle=True)
    Fz = {k: z[k] for k in z.files}
    for l in (8, 12):
        emb[f"WavLM L{l} non-parole, contexte complet (24)"] = a24.participant_matrix(Fz, f"wavlm{l}_nonspeech", T.index)
    iso = CACHE / "28_nonspeech_isolated" / "features.npz"
    if iso.exists():
        Fi = dict(np.load(iso))
        for k in ("wavlm8_iso", "wavlm12_iso", "whisper8_iso"):
            emb[f"{k} non-parole ISOLÉE (28)"] = a24.participant_matrix(Fi, k, T.index)
    rows = []
    for g, k in [("HC", yb == 0), ("MCI+Dementia", yb == 1)]:
        for name, X in blocks.items():
            m, s = regime_cv(np.nan_to_num(X[k]), reg[k], lr_dense)
            rows.append({"groupe": g, "prédicteurs": name, "AUC_régime": m, "AUC_sd": s})
        for name, X in emb.items():
            m, s = regime_cv(X[k], reg[k], lambda: a24.EmbModel(None))
            rows.append({"groupe": g, "prédicteurs": name, "AUC_régime": m, "AUC_sd": s})
        print(f"  C {g} fait", flush=True)
    return pd.DataFrame(rows)


def texts(meta, t):
    manual = [read_text(p) for p in meta[f"txt_{t}"]]
    asr = [(ASR_DIR / f"{pid}__{t}.txt").read_text(encoding="utf-8") for pid in meta.participant_id]
    return {"manuel brut": [raw(x) for x in manual], "manuel nettoyé": [clean_for_model(x) for x in manual],
            "Whisper": asr}


def word_model():
    return t04.tfidf_model(False)


def char_model():
    return make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=False, min_df=2,
                                         sublinear_tf=True),
                         LogisticRegression(class_weight="balanced", C=1.0, max_iter=5000))


def part_d(tr, R, yb):
    reg = R.loc[tr.participant_id, "regime"].to_numpy()
    rows = []
    for t in TASKS:
        tx = texts(tr, t)
        for g, k in [("HC", yb == 0), ("MCI+Dementia", yb == 1)]:
            for v, xs in tx.items():
                X = np.array(xs, dtype=object)[k]
                for mname, mk in [("TF-IDF mots", word_model), ("TF-IDF caractères 1–3 (casse, ponctuation)", char_model)]:
                    m, s = regime_cv(X, reg[k], mk)
                    rows.append({"task": t, "groupe": g, "version": v, "modèle": mname, "AUC_régime": m, "AUC_sd": s})
        print(f"  D {t} fait", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- E

def part_e(tr, T, y3, yb):
    splits = t04.folds(y3, 5, 5)          # mêmes folds que le modèle principal
    feats = pd.read_csv(CACHE / "01_audit" / "recording_features.csv",
                        dtype={"participant_id": str}).set_index(["participant_id", "task"])
    cols, per_task = {}, {}
    for t in TASKS:
        tx = texts(tr, t)
        pf = np.array([m08.pause_features(p, t, VAD, feats.loc[(p, t), "duration_s"]) for p in tr.participant_id])
        for v, xs in tx.items():
            cols.setdefault(v, {})[f"text_{t}"] = xs
        cols.setdefault("Whisper + pauses mesurées", {})[f"text_{t}"] = tx["Whisper"]
        for j in range(2):
            cols["Whisper + pauses mesurées"][f"pause{j}_{t}"] = pf[:, j]
    tf = t04.CONFIG["tfidf"]

    def concat_model(with_p):
        parts = [(f"tf_{t}", TfidfVectorizer(lowercase=tf["lowercase"], sublinear_tf=tf["sublinear_tf"],
                                             min_df=tf["min_df"]), f"text_{t}") for t in TASKS]
        if with_p:
            parts.append(("p", StandardScaler(), [f"pause{j}_{t}" for t in TASKS for j in range(2)]))
        return make_pipeline(ColumnTransformer(parts), LogisticRegression(class_weight="balanced", max_iter=5000))

    def task_model(with_p, t):
        parts = [("tf", TfidfVectorizer(lowercase=tf["lowercase"], sublinear_tf=tf["sublinear_tf"],
                                        min_df=tf["min_df"]), f"text_{t}")]
        if with_p:
            parts.append(("p", StandardScaler(), [f"pause{j}_{t}" for j in range(2)]))
        return make_pipeline(ColumnTransformer(parts), LogisticRegression(class_weight="balanced", max_iter=5000))

    P_main = np.load(CACHE / "23_text_models" / "P_main.npy")
    fa = lambda P: np.array([roc_auc_score(yb[tst], P[r, tst][:, 1]) for r, rep in enumerate(splits) for _, tst in rep])
    a_main = fa(P_main)
    rows, comps = [], []
    for v, c in cols.items():
        X = pd.DataFrame(c)
        wp = v == "Whisper + pauses mesurées"
        Pc, Pl = np.zeros((5, len(yb), 2)), np.zeros((5, len(yb), 2))
        for r, rep in enumerate(splits):
            for trn, tst in rep:
                Pc[r, tst] = concat_model(wp).fit(X.iloc[trn], yb[trn]).predict_proba(X.iloc[tst])
                Pl[r, tst] = np.mean([task_model(wp, t).fit(X.iloc[trn], yb[trn]).predict_proba(X.iloc[tst])
                                      for t in TASKS], axis=0)
        for kind, P in (("concaténation", Pc), ("fusion tardive", Pl)):
            a = fa(P)
            rows.append({"version": v, "multi-tâches": kind, "AUC": a.mean(), "AUC_sd": a.std(ddof=1)})
            d = a - a_main
            comps.append({"comparaison": f"{v}, {kind} − principal", "dAUC": d.mean(),
                          "p_NB": nadeau_bengio(d, 256, 64)[1]})
        print(f"  E {v} fait", flush=True)
    comps = pd.DataFrame(comps)
    comps["p_Holm"] = holm(comps.p_NB)
    rows.append({"version": "modèle principal (sans annotation)", "multi-tâches": "3 tâches", "AUC": a_main.mean(),
                 "AUC_sd": a_main.std(ddof=1)})
    return pd.DataFrame(rows), comps


# ---------------------------------------------------------------- F, G

def part_f():
    d = pd.read_csv(ROOT / "results" / "18_distilbert_5x5" / "paired_differences_train.csv")
    tc = st.t.ppf(.975, 24)
    t_obs = st.t.ppf(1 - d.p_NadeauBengio.clip(1e-12, 1) / 2, 24)
    se = (d.dAUC_fold_mean.abs() / t_obs).replace(np.inf, np.nan)
    return d.assign(IC95_lo=d.dAUC_fold_mean - tc * se, IC95_hi=d.dAUC_fold_mean + tc * se)


def part_g(tr):
    feats = pd.read_csv(CACHE / "01_audit" / "recording_features.csv", dtype={"participant_id": str})
    c = pd.read_csv(CACHE / "11_clinical" / "clinical_per_recording.csv", dtype={"participant_id": str})
    rows = []
    for p in tr.itertuples():
        segs = VAD[f"{p.participant_id}__PFT"]
        gaps = np.array([b[0] - a[1] for a, b in zip(segs[:-1], segs[1:])])
        rows.append({"participant_id": p.participant_id, "diagnosis": p.diagnosis, "age": float(p.age),
                     "n_gaps_ge1s": int((gaps >= 1).sum()), "n_speech_units": len(segs)})
    d = pd.DataFrame(rows)
    d = d.merge(c[c.task == "PFT"][["participant_id", "valid_per_s"]], on="participant_id", how="left")
    d = d.merge(feats[feats.task == "PFT"][["participant_id", "duration_s"]], on="participant_id")
    d["gaps_per_unit"] = d.n_gaps_ge1s / d.n_speech_units.clip(lower=1)
    return d.groupby("diagnosis")[["age", "n_gaps_ge1s", "n_speech_units", "gaps_per_unit", "valid_per_s",
                                   "duration_s"]].median().round(3)


# ---------------------------------------------------------------- I

import re  # noqa: E402

FILLERS = {"um", "uh", "er", "erm", "ah", "hm", "hmm", "mm", "mmm", "uhm", "eh", "oh", "ooh", "em", "uh-huh"}
TOK = re.compile(r"[A-Za-z']+-?")


def style(text_clean, text_raw):
    toks = TOK.findall(text_clean)
    n = max(len(toks), 1)
    return {"n_tokens": len(toks), "hésitations_pour100": 100 * sum(t.lower().strip("-") in FILLERS for t in toks) / n,
            "fragments_pour100": 100 * sum(t.endswith("-") for t in toks) / n,
            "ponctuation_pour100": 100 * len(re.findall(r"[.,?!;:…\"]", text_clean)) / n,
            "majuscules_pour100": 100 * sum(t[0].isupper() for t in toks) / n,
            "lignes": sum(1 for l in text_raw.split("\n") if l.strip())}


def strict(text_clean):
    return " ".join(t.lower() for t in TOK.findall(text_clean)
                    if not t.endswith("-") and t.lower() not in FILLERS)


def part_i(tr, R, y3, yb):
    reg = R.loc[tr.participant_id, "regime"].to_numpy()
    splits = t04.folds(y3, 5, 5)
    srows, rrows, drows = [], [], []
    for t in TASKS:
        manual = [read_text(p) for p in tr[f"txt_{t}"]]
        cl = [clean_for_model(x) for x in manual]
        S = pd.DataFrame([style(c, m) for c, m in zip(cl, manual)]).assign(regime=reg, diagnosis=tr.diagnosis.values)
        for (g, r), d in S.groupby(["diagnosis", "regime"]):
            srows.append({"task": t, "diagnosis": g, "régime": "Pat:" if r else "sans préfixe", "n": len(d),
                          **d.drop(columns=["regime", "diagnosis"]).median().round(2).to_dict()})
        k = yb == 0
        Xs = S.drop(columns=["regime", "diagnosis"]).to_numpy(float)
        m, s = regime_cv(Xs[k], reg[k], lr_dense)
        rrows.append({"task": t, "prédicteurs": "6 traits de style (HC)", "AUC_régime": m, "AUC_sd": s})
        m, s = regime_cv(np.log1p(Xs[k][:, :1]), reg[k], lr_dense)
        rrows.append({"task": t, "prédicteurs": "longueur seule (HC)", "AUC_régime": m, "AUC_sd": s})
        st_txt = np.array([strict(c) for c in cl], dtype=object)
        m, s = regime_cv(st_txt[k], reg[k], word_model)
        rrows.append({"task": t, "prédicteurs": "TF-IDF mots, nettoyé STRICT (HC)", "AUC_régime": m, "AUC_sd": s})
        for v, xs in (("manuel nettoyé", np.array(cl, dtype=object)), ("manuel nettoyé strict", st_txt)):
            P = np.zeros((5, len(yb), 2))
            for r, rep in enumerate(splits):
                for trn, tst in rep:
                    P[r, tst] = word_model().fit(xs[trn], yb[trn]).predict_proba(xs[tst])
            a = np.array([roc_auc_score(yb[tst], P[r, tst][:, 1]) for r, rep in enumerate(splits) for _, tst in rep])
            drows.append({"task": t, "version": v, "AUC_diag": a.mean(), "AUC_sd": a.std(ddof=1)})
        print(f"  I {t} fait", flush=True)
    return pd.DataFrame(srows), pd.DataFrame(rrows), pd.DataFrame(drows)


# ---------------------------------------------------------------- H (EXPLORATOIRE, test)

def part_h(meta, T_tr, yb_tr, y3_tr, rng):
    te = meta[meta.Split == "Test"].reset_index(drop=True)
    T_te = m10.participant_table(te)
    R_te, R_tr = regime_table("Test"), regime_table("Train")
    comp = []
    for split, T, R in (("Train", T_tr, R_tr), ("Test", T_te, R_te)):
        c = pd.read_csv(CACHE / "11_clinical" / "clinical_per_recording.csv", dtype={"participant_id": str})
        f = pd.read_csv(CACHE / "01_audit" / "recording_features.csv", dtype={"participant_id": str})
        mm = meta.set_index("participant_id")
        for g in ("HC", "MCI", "Dementia"):
            pids = T.index[T.diagnosis == g]
            comp.append({"split": split, "diagnosis": g, "n": len(pids),
                         "pct_regime_Pat": 100 * R.loc[pids, "regime"].mean(),
                         "n_sessions_mixtes": int(R.loc[pids, "mixed"].sum()),
                         "MMSE_moy": mm.loc[pids, "MMSE"].astype(float).mean(),
                         "MMSE_n": int(mm.loc[pids, "MMSE"].notna().sum()),
                         "SFT_valid_per_s_med": c[(c.task == "SFT") & c.participant_id.isin(pids)].valid_per_s.median(),
                         "PFT_valid_per_s_med": c[(c.task == "PFT") & c.participant_id.isin(pids)].valid_per_s.median(),
                         "CTD_durée_med_s": f[(f.task == "CTD") & f.participant_id.isin(pids)].duration_s.median()})
    F_tr, F_te = m10.fold_features(T_tr, T_te)
    cols = m10.columns(F_tr, m10.CFG["models"]["B_G_nocov"])
    p = m10.lr().fit(F_tr[cols].to_numpy(float), yb_tr).predict_proba(F_te[cols].to_numpy(float))[:, 1]
    P3 = m10.FrankHall().fit(F_tr[cols].to_numpy(float), y3_tr).predict_proba(F_te[cols].to_numpy(float))
    yt = (T_te.diagnosis != "HC").astype(int).to_numpy()
    y3t = T_te.diagnosis.map(CLASSES).to_numpy()
    reg = R_te.loc[T_te.index, "regime"].to_numpy()
    rows = []
    tr = meta[meta.Split == "Train"].reset_index(drop=True)
    txt_tr = {t: texts(tr, t) for t in TASKS}
    txt_te = {t: texts(te, t) for t in TASKS}
    preds = {"modèle principal": p}
    for t in TASKS:
        for v in ("manuel brut", "manuel nettoyé"):
            preds[f"TF-IDF {v} {t}"] = word_model().fit(np.array(txt_tr[t][v], dtype=object), yb_tr).predict_proba(
                np.array(txt_te[t][v], dtype=object))[:, 1]
    for name, q in preds.items():
        for sub, k in [("tous", np.ones(len(yt), bool)), ("régime « Pat: »", reg == 1), ("régime sans préfixe", reg == 0)]:
            lo, hi = boot_auc(yt[k], q[k], rng)
            rows.append({"modèle": name, "sous-ensemble": sub, "n_HC": int((yt[k] == 0).sum()),
                         "n_patients": int(yt[k].sum()), "AUC": roc_auc_score(yt[k], q[k]), "AUC_lo": lo, "AUC_hi": hi})
    cm = pd.DataFrame(confusion_matrix(y3t, P3.argmax(1), labels=[0, 1, 2]),
                      index=[f"vrai {g}" for g in CLASSES], columns=[f"prédit {g}" for g in CLASSES])
    f1 = f1_score(y3t, P3.argmax(1), average=None, labels=[0, 1, 2])
    return pd.DataFrame(comp), pd.DataFrame(rows), cm, dict(zip(CLASSES, f1.round(3).tolist()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    meta = load_dataset()
    tr = meta[meta.Split == "Train"].reset_index(drop=True)
    T = m10.participant_table(tr)
    assert list(T.index) == list(tr.participant_id)
    y3 = T.diagnosis.map(CLASSES).to_numpy()
    yb = (y3 > 0).astype(int)
    R = regime_table("Train")
    only = sys.argv[1:] or list("ABCDEFGIH")

    if "A" in only:
        A1, A2 = part_a(R)
        A1.round(2).to_csv(OUT / "A_regime_sessions_train.csv", index=False)
        A2.round(4).to_csv(OUT / "A_regime_OR_train.csv", index=False)
        print(A1.round(2).to_string(), "\n", A2.round(3).to_string(), flush=True)
    if "B" in only:
        B1, B2, _ = part_b(T, R, yb, rng)
        B1.round(4).to_csv(OUT / "B_main_model_within_regime_train.csv", index=False)
        B2.round(4).to_csv(OUT / "B_main_score_predicts_regime_train.csv", index=False)
        print(B1.round(3).to_string(), "\n", B2.round(3).to_string(), flush=True)
    if "C" in only:
        C = part_c(T, R, yb)
        C.round(4).to_csv(OUT / "C_regime_from_audio_train.csv", index=False)
        print(C.round(3).to_string(), flush=True)
    if "D" in only:
        D = part_d(tr, R, yb)
        D.round(4).to_csv(OUT / "D_regime_from_text_train.csv", index=False)
        print(D.round(3).to_string(), flush=True)
    if "E" in only:
        E1, E2 = part_e(tr, T, y3, yb)
        E1.round(4).to_csv(OUT / "E_multitask_tfidf_train.csv", index=False)
        E2.round(4).to_csv(OUT / "E_multitask_vs_main_train.csv", index=False)
        print(E1.round(3).to_string(), "\n", E2.round(4).to_string(), flush=True)
    if "F" in only:
        Fd = part_f()
        Fd.round(4).to_csv(OUT / "F_distilbert_CI_train.csv", index=False)
        print(Fd.round(3).to_string(), flush=True)
    if "G" in only:
        G = part_g(tr)
        G.to_csv(OUT / "G_below_chance_descriptives_train.csv")
        print(G.to_string(), flush=True)
    if "I" in only:
        I1, I2, I3 = part_i(tr, R, y3, yb)
        I1.to_csv(OUT / "I_style_by_regime_train.csv", index=False)
        I2.round(4).to_csv(OUT / "I_regime_from_style_train.csv", index=False)
        I3.round(4).to_csv(OUT / "I_diag_strict_cleaning_train.csv", index=False)
        print(I1.to_string(), "\n", I2.round(3).to_string(), "\n", I3.round(3).to_string(), flush=True)
    if "H" in only:
        H1, H2, cm, f1 = part_h(meta, T, yb, y3, rng)
        H1.round(3).to_csv(OUT / "H_EXPLORATORY_composition_train_vs_test.csv", index=False)
        H2.round(4).to_csv(OUT / "H_EXPLORATORY_test_within_regime.csv", index=False)
        cm.to_csv(OUT / "H_EXPLORATORY_test_confusion_3class.csv")
        s = OUT / "summary.json"
        old = json.loads(s.read_text()) if s.exists() else {}
        old |= {"H_EXPLORATORY_F1_par_classe_test": f1}
        s.write_text(json.dumps(old, indent=2, ensure_ascii=False))
        print(H1.round(3).to_string(), "\n", H2.round(3).to_string(), "\n", cm, "\n", f1, flush=True)
    s = OUT / "summary.json"
    old = json.loads(s.read_text()) if s.exists() else {}
    old |= {"generated": datetime.now().isoformat(timespec="seconds"), "parties": only,
            "régime": "session ; « Pat: » si ≥ 2 des 3 fichiers portent un préfixe de locuteur",
            "note": "A–G train uniquement ; H test EXPLORATOIRE, pipeline figé, aucune décision"}
    s.write_text(json.dumps(old, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
