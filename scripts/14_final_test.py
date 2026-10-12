"""Évaluation FINALE sur le test officiel — à exécuter UNE SEULE FOIS (configs/final.yaml).

Tout est ajusté sur les 320 participants du train (normes HC, classifieurs, PCA, TF-IDF), puis appliqué aux 80 du test.
IC 95 % par bootstrap des participants du test (2000). Le script refuse de s'exécuter si des résultats existent déjà.
Empreintes SHA-256 de la config et des scripts enregistrées avec les résultats.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import TASKS, load_dataset  # noqa: E402
from src.text import clean_for_model, raw, read_text  # noqa: E402

FINAL = yaml.safe_load((ROOT / "configs" / "final.yaml").read_text())
OUT = ROOT / "results" / "14_final_test"
CLASSES = {"HC": 0, "MCI": 1, "Dementia": 2}
HASHED = ["configs/final.yaml", "configs/model_clinical_v2.yaml", "configs/features.yaml", "scripts/14_final_test.py",
          "scripts/10_model.py", "scripts/12_wavlm.py", "scripts/08_text_asr.py", "scripts/04_text_artifact.py",
          "src/normative.py", "src/text.py", "src/data.py"]


def _load(name, path, argv=None):
    saved, sys.argv = sys.argv, [path] + (argv or [])
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = saved
    return mod


m10 = _load("m10", "scripts/10_model.py", ["--config", FINAL["main_model"]["config"]])
m12 = _load("m12", "scripts/12_wavlm.py")
m08 = _load("m08", "scripts/08_text_asr.py")


def metrics(y, P, multi):
    pred = P.argmax(1)
    if multi:
        return {"AUC": roc_auc_score(y, P, multi_class="ovr"), "BAcc": balanced_accuracy_score(y, pred),
                "macroF1": f1_score(y, pred, average="macro")}
    return {"AUC": roc_auc_score(y, P[:, 1]), "BAcc": balanced_accuracy_score(y, pred),
            "sensibilité": pred[y == 1].mean(), "spécificité": 1 - pred[y == 0].mean()}


def bootstrap(y, preds, multi, rng, main):
    """IC des métriques de chaque modèle + IC de la différence d'AUC (principal − modèle), mêmes rééchantillons."""
    obs = {k: metrics(y, P, multi) for k, P in preds.items()}
    boots = {k: [] for k in preds}
    n_cls = 3 if multi else 2
    for _ in range(FINAL["n_boot"]):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < n_cls:
            continue
        for k, P in preds.items():
            boots[k].append(metrics(y[i], P[i], multi))
    rows = []
    for k in preds:
        b = pd.DataFrame(boots[k])
        r = {"modèle": k}
        for mname, v in obs[k].items():
            lo, hi = np.percentile(b[mname], [2.5, 97.5])
            r |= {mname: v, f"{mname}_lo": lo, f"{mname}_hi": hi}
        d = np.array([x["AUC"] for x in boots[main]]) - np.array([x["AUC"] for x in boots[k]])
        r |= {"dAUC_vs_principal": obs[main]["AUC"] - obs[k]["AUC"],
              "dAUC_lo": np.percentile(d, 2.5), "dAUC_hi": np.percentile(d, 97.5)}
        rows.append(r)
    return pd.DataFrame(rows).round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", default=None, help="répertoire de sortie : faux test = 80 participants du TRAIN")
    args = ap.parse_args()
    global OUT
    if args.dry_run:
        OUT = Path(args.dry_run)
        OUT.mkdir(parents=True, exist_ok=True)
    elif OUT.exists() and any(OUT.iterdir()):
        sys.exit("results/14_final_test/ existe déjà : le test a déjà été évalué. Arrêt (évaluation unique).")
    else:
        OUT.mkdir(parents=True)
    rng = np.random.default_rng(FINAL["seed"])
    meta = load_dataset()
    if args.dry_run:  # essai à blanc : le test n'est pas lu
        from sklearn.model_selection import train_test_split
        tr_all = meta[meta.Split == "Train"]
        a, b = train_test_split(tr_all, test_size=80, stratify=tr_all.diagnosis, random_state=FINAL["seed"])
        trn_meta, tst_meta = a.reset_index(drop=True), b.reset_index(drop=True)
    else:
        trn_meta, tst_meta = (meta[meta.Split == s].reset_index(drop=True) for s in ("Train", "Test"))
    Ttr, Tte = m10.participant_table(trn_meta), m10.participant_table(tst_meta)
    y3_tr, y3_te = Ttr.diagnosis.map(CLASSES).to_numpy(), Tte.diagnosis.map(CLASSES).to_numpy()
    yb_tr, yb_te = (y3_tr > 0).astype(int), (y3_te > 0).astype(int)
    bin_, tri = {}, {}

    # acoustique interprétable (normes HC sur tout le train ; HC du train scorés par cross-fitting)
    F_tr, F_te = m10.fold_features(Ttr, Tte)
    for name in [FINAL["main_model"]["name"], *FINAL["secondary_models"]["acoustic"]]:
        cols = m10.columns(F_tr, m10.CFG["models"][name])
        Xtr, Xte = F_tr[cols].to_numpy(float), F_te[cols].to_numpy(float)
        bin_[name] = m10.lr().fit(Xtr, yb_tr).predict_proba(Xte)
        tri[name] = m10.FrankHall().fit(Xtr, y3_tr).predict_proba(Xte)

    # WavLM et fusion
    E = m12.extract(None)
    Rtr, Rte = m12.representations(E, Ttr.index), m12.representations(E, Tte.index)
    cols = m10.columns(F_tr, m10.CFG["models"]["B_G"])
    tab_tr, tab_te = F_tr[cols].to_numpy(float), F_te[cols].to_numpy(float)
    bin_["wavlm_centered"] = m12.emb_model().fit(Rtr["wavlm_centered"], yb_tr).predict_proba(Rte["wavlm_centered"])
    tri["wavlm_centered"] = m12.EmbFrankHall().fit(Rtr["wavlm_centered"], y3_tr).predict_proba(Rte["wavlm_centered"])
    Ftr, Fte = np.hstack([tab_tr, Rtr["wavlm_centered"]]), np.hstack([tab_te, Rte["wavlm_centered"]])
    bin_["fusion"] = m12.Fusion(len(cols), False).fit(Ftr, yb_tr).predict_proba(Fte)
    tri["fusion"] = m12.Fusion(len(cols), True).fit(Ftr, y3_tr).predict_proba(Fte)

    # texte, par tâche
    feats = pd.read_csv(ROOT / "results" / "cache" / "01_audit" / "recording_features.csv",
                        dtype={"participant_id": str}).set_index(["participant_id", "task"])
    vad = json.loads(m08.VAD.read_text())

    def text_X(mt, t):
        manual = [read_text(p) for p in mt[f"txt_{t}"]]
        asr = [(m08.ASR_DIR / f"{pid}__{t}.txt").read_text(encoding="utf-8") for pid in mt.participant_id]
        pf = np.array([m08.pause_features(pid, t, vad, feats.loc[(pid, t), "duration_s"]) for pid in mt.participant_id])
        base = pd.DataFrame({m08.CONFIG["pause_features"][0]: pf[:, 0], m08.CONFIG["pause_features"][1]: pf[:, 1]})
        return {"m_raw": base.assign(text=[raw(x) for x in manual]),
                "m_clean": base.assign(text=[clean_for_model(x) for x in manual]),
                "asr": base.assign(text=asr), "asr_pauses": base.assign(text=asr)}

    for t in TASKS:
        Xtr, Xte = text_X(trn_meta, t), text_X(tst_meta, t)
        for c in FINAL["secondary_models"]["text"]:
            bin_[f"text_{c}_{t}"] = m08.make_model(c == "asr_pauses").fit(Xtr[c], yb_tr).predict_proba(Xte[c])
            tri[f"text_{c}_{t}"] = m08.make_model(c == "asr_pauses").fit(Xtr[c], y3_tr).predict_proba(Xte[c])

    main_name = FINAL["main_model"]["name"]
    res_b = bootstrap(yb_te, bin_, False, rng, main_name)
    res_3 = bootstrap(y3_te, tri, True, rng, main_name)
    res_b.to_csv(OUT / "test_binary.csv", index=False)
    res_3.to_csv(OUT / "test_3class.csv", index=False)

    # équité par sexe (modèle principal)
    male = Tte.male.to_numpy().astype(bool)
    fair = []
    for g, m in [("hommes", male), ("femmes", ~male)]:
        r = bootstrap(yb_te[m], {main_name: bin_[main_name][m]}, False, rng, main_name).iloc[0].to_dict()
        fair.append({"groupe": g, "n": int(m.sum()), "n_patients": int(yb_te[m].sum()), **r})
    fair = pd.DataFrame(fair).drop(columns=["modèle", "dAUC_vs_principal", "dAUC_lo", "dAUC_hi"])
    fair.to_csv(OUT / "test_fairness_sex.csv", index=False)

    hashes = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in HASHED}
    (OUT / "provenance.json").write_text(json.dumps(
        {"generated": datetime.now().isoformat(timespec="seconds"), "n_train": len(Ttr), "n_test": len(Tte),
         "test_counts": Tte.diagnosis.value_counts().to_dict(), "sha256": hashes}, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("== 2 classes ==\n", res_b, "\n\n== 3 classes ==\n", res_3, "\n\n== Équité ==\n", fair)


if __name__ == "__main__":
    main()
