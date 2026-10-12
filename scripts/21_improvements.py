"""Pistes d'amélioration — TRAIN uniquement, EXPLORATOIRE (le test est consommé).

Nouvelles mesures sans annotation humaine (Whisper horodaté / Whisper CTD), fenêtre min(durée, 45 s) :
  SFT : switches/s et taille moyenne des clusters sémantiques (Troyer et al. 1997, approximés par la similarité
        Wu–Palmer WordNet entre animaux successifs ≥ 0,8), similarité moyenne entre mots successifs ;
  PFT : switches/s et taille moyenne des clusters phonémiques (mêmes deux premières lettres ou même rime finale) ;
  CTD : unités d'information (liste fixe de 25 concepts du Cookie Theft) distinctes par seconde et par mot.
Modèles (mêmes 5 × 5 folds, normes HC réapprises dans chaque fold, sans âge/sexe dans le classifieur) :
  base (= B_G_nocov), +clusters, +CIU, +les deux ; 3 classes : Frank & Hall vs décision en deux étapes.
Comparaisons : Nadeau–Bengio sur les folds ; aucune de ces variantes n'est évaluée sur le test.
"""

import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_dataset  # noqa: E402
from src.normative import COVARIATES, crossfit_scores  # noqa: E402
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
OUT = ROOT / "results" / "21_improvements"
CACHE = ROOT / "results" / "cache" / "21_improvements"
SEED = 42
WUP_THR = 0.8
# Unités d'information du Cookie Theft (liste fixe, concepts usuels de la littérature ; synonymes regroupés)
CIU = {"boy": ["boy", "son", "lad"], "girl": ["girl", "daughter", "sister"], "woman": ["woman", "mother", "mum", "mom", "lady"],
       "cookie": ["cookie", "cookies", "biscuit", "biscuits"], "jar": ["jar"], "stool": ["stool"],
       "fall": ["fall", "falling", "fell", "tipping", "topple", "toppling"], "sink": ["sink"],
       "water": ["water"], "overflow": ["overflow", "overflowing", "spilling", "spill", "flooding", "running over"],
       "dish": ["dish", "dishes", "plate", "plates", "cup", "cups"], "wash": ["wash", "washing", "drying", "dry", "wiping"],
       "cupboard": ["cupboard", "cabinet", "shelf"], "window": ["window"], "curtain": ["curtain", "curtains"],
       "floor": ["floor"], "garden": ["garden", "outside", "yard", "path"], "tap": ["tap", "faucet"],
       "kitchen": ["kitchen"], "reach": ["reach", "reaching", "taking", "stealing", "grabbing"],
       "hand": ["hand", "handing", "giving", "asking"], "shh": ["quiet", "shh", "finger", "lips"],
       "unaware": ["unaware", "not noticing", "doesn't notice", "daydreaming", "oblivious"],
       "puddle": ["puddle", "wet"], "apron": ["apron"]}
CIU_RX = {k: re.compile(r"\b(" + "|".join(re.escape(w) for w in v) + r")\b", re.I) for k, v in CIU.items()}


# ---------------------------------------------------------------- mesures

def sft_clusters(words):
    lem = [m11.animal_lemma(w.lower()) for w, *_ in words]
    lem = [l for l in lem if l]
    seq = [l for i, l in enumerate(lem) if i == 0 or l != lem[i - 1]]
    if len(seq) < 2:
        return 0, float(len(seq)), np.nan
    syn = [m11.WN.synsets(l, m11.WN.NOUN)[0] for l in seq]
    sims = []
    for a, b in zip(syn[:-1], syn[1:]):
        sims.append(max((x.wup_similarity(y) or 0) for x in m11.WN.synsets(a.lemmas()[0].name(), m11.WN.NOUN)[:3]
                        for y in m11.WN.synsets(b.lemmas()[0].name(), m11.WN.NOUN)[:3]))
    sims = np.array(sims)
    switches = int((sims < WUP_THR).sum())
    return switches, len(seq) / (switches + 1), float(sims.mean())


def pft_clusters(words):
    seq = [w.lower() for w, _, first in words if w.lower().startswith("p") and m11.is_english(w.lower())
           and not (w[0].isupper() and not first) and w.lower() not in m11.STOP]
    if len(seq) < 2:
        return 0, float(len(seq))
    same = [(a[:2] == b[:2]) or (len(a) > 2 and len(b) > 2 and a[-2:] == b[-2:]) for a, b in zip(seq[:-1], seq[1:])]
    switches = int(len(same) - sum(same))
    return switches, len(seq) / (switches + 1)


def build_measures(meta):
    path = CACHE / "new_measures.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"participant_id": str})
    f = pd.read_csv(ROOT / "results" / "cache" / "05_features" / "features_per_recording.csv",
                    dtype={"participant_id": str}).set_index(["participant_id", "task"])
    rows = []
    for pid in meta.participant_id:
        for t in ("SFT", "PFT"):
            segs = json.loads(m11.ts_path(pid, t).read_text(encoding="utf-8"))
            w = min(f.loc[(pid, t), "duration_s"], 45.0)
            words = m11.timed_words(segs, w)
            if t == "SFT":
                sw, cs, sim = sft_clusters(words)
                rows.append({"participant_id": pid, "task": t, "switch_per_s": sw / w, "cluster_size": cs,
                             "succ_similarity": sim})
            else:
                sw, cs = pft_clusters(words)
                rows.append({"participant_id": pid, "task": t, "switch_per_s": sw / w, "cluster_size": cs})
        text = (ROOT / "data" / "derived" / "asr_whisper" / f"{pid}__CTD.txt").read_text(encoding="utf-8")
        dur = f.loc[(pid, "CTD"), "duration_s"]
        n_words = max(len(text.split()), 1)
        n_ciu = sum(bool(rx.search(text)) for rx in CIU_RX.values())
        rows.append({"participant_id": pid, "task": "CTD", "ciu_per_s": n_ciu / dur, "ciu_per_word": n_ciu / n_words,
                     "ciu_count": n_ciu})
    d = pd.DataFrame(rows)
    d.to_csv(path, index=False)
    return d


# ---------------------------------------------------------------- modèle

def fold_features(train, test, fluency, ctd):
    is_hc = train.diagnosis == "HC"
    F_tr, F_te = pd.DataFrame(index=train.index), pd.DataFrame(index=test.index)
    for m in fluency:
        cols = {f"{m}__x_SFT": "x_SFT", f"{m}__x_PFT": "x_PFT"}
        a = train[[*cols, *COVARIATES]].rename(columns=cols)
        b = test[[*cols, *COVARIATES]].rename(columns=cols)
        s_tr, full = crossfit_scores(a, is_hc, 5, SEED)
        F_tr[f"G__{m}"], F_te[f"G__{m}"] = s_tr["G"].astype(float), full.scores(b)["G"].astype(float)
    for m in ctd:
        F_tr[f"ctd__{m}"], F_te[f"ctd__{m}"] = m10.z_single(train, test, f"ctd__{m}", is_hc, 1.35, 5, SEED)
    return F_tr, F_te


class TwoStage:
    """Étape 1 : HC vs patients ; étape 2 : MCI vs Dementia chez les patients (LR pondérées)."""
    def fit(self, X, y):
        self.s1 = m10.lr().fit(X, (y > 0).astype(int))
        self.s2 = m10.lr().fit(X[y > 0], (y[y > 0] == 2).astype(int))
        return self

    def predict_proba(self, X):
        p1, p2 = self.s1.predict_proba(X)[:, 1], self.s2.predict_proba(X)[:, 1]
        return np.column_stack([1 - p1, p1 * (1 - p2), p1 * p2])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = load_dataset()
    tr = meta[meta.Split == "Train"].reset_index(drop=True)
    T = m10.participant_table(tr)
    new = build_measures(tr)
    for m in ["switch_per_s", "cluster_size"]:
        w = new[new.task.isin(["SFT", "PFT"])].pivot(index="participant_id", columns="task", values=m)
        T[f"{m}__x_SFT"], T[f"{m}__x_PFT"] = w["SFT"], w["PFT"]
    c = new[new.task == "CTD"].set_index("participant_id")
    T["ctd__ciu_per_s"], T["ctd__ciu_per_word"] = np.log1p(c.ciu_per_s * 60), c.ciu_per_word
    base_flu = list(m10.CFG["fluency_measures"])
    base_ctd = list(m10.CFG["ctd_measures"])
    variants = {"base": (base_flu, base_ctd),
                "+clusters": (base_flu + ["switch_per_s", "cluster_size"], base_ctd),
                "+CIU": (base_flu, base_ctd + ["ciu_per_s", "ciu_per_word"]),
                "+clusters+CIU": (base_flu + ["switch_per_s", "cluster_size"], base_ctd + ["ciu_per_s", "ciu_per_word"])}
    y3 = T.diagnosis.map({"HC": 0, "MCI": 1, "Dementia": 2}).to_numpy()
    yb = (y3 > 0).astype(int)
    splits = t04.folds(y3, 5, 5)
    P = {(v, k): np.zeros((5, len(T), 2 if k == "bin" else 3)) for v in variants for k in ("bin", "fh", "2st")}
    for r, rep in enumerate(splits):
        for trn, tst in rep:
            for v, (flu, ctd) in variants.items():
                F_tr, F_te = fold_features(T.iloc[trn], T.iloc[tst], flu, ctd)
                Xtr, Xte = F_tr.to_numpy(float), F_te.to_numpy(float)
                P[(v, "bin")][r, tst] = m10.lr().fit(Xtr, yb[trn]).predict_proba(Xte)
                P[(v, "fh")][r, tst] = m10.FrankHall().fit(Xtr, y3[trn]).predict_proba(Xte)
                P[(v, "2st")][r, tst] = TwoStage().fit(Xtr, y3[trn]).predict_proba(Xte)
        print(f"  répétition {r + 1}/5", flush=True)

    def per_fold(y, Pk, fn):
        return np.array([fn(y[tst], Pk[r, tst]) for r, rep in enumerate(splits) for _, tst in rep])

    auc2 = lambda y, p: roc_auc_score(y, p[:, 1])  # noqa: E731
    auc3 = lambda y, p: roc_auc_score(y, p, multi_class="ovr")  # noqa: E731
    f13 = lambda y, p: f1_score(y, p.argmax(1), average="macro")  # noqa: E731
    bac = lambda y, p: balanced_accuracy_score(y, p.argmax(1))  # noqa: E731
    rows, comps = [], []
    for v in variants:
        a2 = per_fold(yb, P[(v, "bin")], auc2)
        rows.append({"variante": v, "décision 3 classes": "-", "AUC_2c": a2.mean(), "AUC_2c_sd": a2.std(ddof=1),
                     "BAcc_2c": per_fold(yb, P[(v, "bin")], bac).mean()})
        for k, name in [("fh", "Frank & Hall"), ("2st", "deux étapes")]:
            rows.append({"variante": v, "décision 3 classes": name, "AUC_3c": per_fold(y3, P[(v, k)], auc3).mean(),
                         "BAcc_3c": per_fold(y3, P[(v, k)], bac).mean(), "macroF1_3c": per_fold(y3, P[(v, k)], f13).mean()})
        if v != "base":
            d = a2 - per_fold(yb, P[("base", "bin")], auc2)
            comps.append({"comparaison": f"{v} − base (AUC 2 classes)", "delta": d.mean(),
                          "p_NB": nadeau_bengio(d, 256, 64)[1]})
    for v in variants:
        d = per_fold(y3, P[(v, "2st")], f13) - per_fold(y3, P[(v, "fh")], f13)
        comps.append({"comparaison": f"deux étapes − Frank & Hall, {v} (macro-F1 3 classes)", "delta": d.mean(),
                      "p_NB": nadeau_bengio(d, 256, 64)[1]})
    comps = pd.DataFrame(comps)
    comps["p_Holm"] = holm(comps.p_NB)
    res = pd.DataFrame(rows).round(4)
    res.to_csv(OUT / "cv_results_train.csv", index=False)
    comps.round(4).to_csv(OUT / "comparisons_train.csv", index=False)
    tr_new = new.merge(tr[["participant_id", "diagnosis"]])
    desc = tr_new.groupby(["task", "diagnosis"])[["switch_per_s", "cluster_size", "succ_similarity", "ciu_per_s",
                                                  "ciu_per_word", "ciu_count"]].median().round(3)
    desc.to_csv(OUT / "new_measures_median_train.csv")
    (OUT / "summary.json").write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                                                  "statut": "exploratoire, train uniquement", "wup_threshold": WUP_THR,
                                                  "ciu_concepts": list(CIU)}, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 220)
    print(desc, "\n\n", res, "\n\n", comps.round(4))


if __name__ == "__main__":
    main()
