"""Étape 0e — Artefact d'annotation dans le texte (TRAIN uniquement).

Versions des transcriptions (src/text.py) :
  official  reproduction exacte du nettoyage de la baseline officielle (retire 'Pat:' et '(Buzzer sounds)',
            concatène les lignes sans espace)
  a_raw     brute (lignes jointes par un espace)
  b1_noprefix  préfixes de locuteur retirés seulement (annotations gardées)
  b_nomarks préfixes de locuteur et annotations ((), [], <>, {}) retirés
  c_partic  comme b, et pour les fichiers dialogués seuls les énoncés du participant sont gardés

TF-IDF (mots) + régression logistique pondérée, par tâche, 5 folds × 5 répétitions (mêmes folds pour toutes
les versions). Différences appariées testées par permutation (échange des prédictions hors-fold par participant).
Option --bert : DistilBERT (fine-tuning léger) sur a_raw et b_nomarks.
Aucun texte n'est affiché ni écrit : seulement des comptes et des métriques.
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import TASKS, load_dataset  # noqa: E402
from src.text import PREFIX, no_marks, official_clean, participant_only, raw, read_text, turns  # noqa: E402

OUT = ROOT / "results" / "04_text_artifact"

CONFIG = {
    "seed": 42, "split": "Train", "n_splits": 5, "n_repeats": 5, "n_permutations": 1000,
    "tfidf": {"lowercase": True, "sublinear_tf": True, "min_df": 2, "ngram_range": [1, 1]},
    "logreg": {"class_weight": "balanced", "C": 1.0, "max_iter": 5000},
    "comparisons": [["a_raw", "b_nomarks"], ["official", "b_nomarks"], ["b_nomarks", "c_partic"],
                    ["a_raw", "b1_noprefix"], ["b1_noprefix", "b_nomarks"]],
    # liste fixée a priori pour décrire les parenthèses (événements non verbaux / annotations de transcripteur)
    "nonverbal_keywords": ["laugh", "cough", "buzzer", "inaudible", "unclear", "unintelligible", "noise", "pause",
                           "sigh", "throat", "background", "interrupt", "crosstalk", "silence", "beep", "breath"],
    "bert": {"model": "distilbert-base-uncased", "max_len": 256, "batch_size": 8, "lr": 5e-5, "epochs": 4,
             "weight_decay": 0.01, "fp16": True, "n_splits": 5, "n_repeats": 1, "versions": ["a_raw", "b_nomarks"],
             "head": "3 classes ; score binaire = 1 − P(HC)", "loss": "entropie croisée pondérée (classes équilibrées)"},
}
VERSIONS = {"official": official_clean, "a_raw": raw,
            "b1_noprefix": lambda t: " ".join(c for _, c in turns(t)), "b_nomarks": no_marks,
            "c_partic": lambda t: participant_only(t)[0]}
CLASSES = {"HC": 0, "MCI": 1, "Dementia": 2}


def build_corpora(meta):
    """Textes par (tâche, version) en mémoire uniquement + statistiques agrégées."""
    corpora, stats = {}, []
    for t in TASKS:
        texts = [read_text(p) for p in meta[f"txt_{t}"]]
        for v, f in VERSIONS.items():
            corpora[(t, v)] = [f(x) for x in texts]
        rules = pd.Series([participant_only(x)[1] for x in texts])
        n_labels = pd.Series([len({l for l, _ in turns(x) if l is not None}) for x in texts])
        tok = {v: np.array([len(s.split()) for s in corpora[(t, v)]]) for v in VERSIONS}
        stats.append({"task": t, "n_files": len(texts),
                      "n_with_prefix": int(sum(any(PREFIX.match(l) for l in x.split("\n")) for x in texts)),
                      "n_dialogue_files": int((n_labels >= 2).sum()),
                      "participant_rule_known_label": int((rules == "known_label").sum()),
                      "participant_rule_most_words": int((rules == "most_words").sum()),
                      **{f"median_tokens_{v}": float(np.median(tok[v])) for v in VERSIONS},
                      "pct_tokens_removed_b_vs_a": round(100 * (1 - tok["b_nomarks"].sum() / tok["a_raw"].sum()), 2),
                      "pct_tokens_removed_c_vs_b": round(100 * (1 - tok["c_partic"].sum() / tok["b_nomarks"].sum()), 2)})
    return corpora, pd.DataFrame(stats)


def annotation_stats(meta):
    """Description agrégée des annotations entre parenthèses (aucun contenu écrit)."""
    kw = re.compile("|".join(CONFIG["nonverbal_keywords"]), re.I)
    rows = []
    for t in TASKS:
        for p, d in zip(meta[f"txt_{t}"], meta.diagnosis):
            text = read_text(p)
            par = re.findall(r"\(([^)\n]{0,80})\)", text)
            n_tok = len(raw(text).split())
            rows.append({"task": t, "diagnosis": d, "n_paren": len(par),
                         "n_paren_tokens": sum(len(x.split()) for x in par), "n_tokens": n_tok,
                         "n_nonverbal": sum(bool(kw.search(x)) for x in par),
                         "n_single_token": sum(len(x.split()) == 1 for x in par)})
    df = pd.DataFrame(rows)
    g = df.groupby(["task", "diagnosis"])
    out = pd.DataFrame({"n_files": g.size(),
                        "pct_files_with_paren": g.n_paren.apply(lambda x: 100 * (x > 0).mean()),
                        "paren_per_file_mean": g.n_paren.mean(),
                        "pct_tokens_in_paren": 100 * g.n_paren_tokens.sum() / g.n_tokens.sum(),
                        "pct_paren_nonverbal_kw": 100 * g.n_nonverbal.sum() / g.n_paren.sum().clip(lower=1),
                        "pct_paren_single_token": 100 * g.n_single_token.sum() / g.n_paren.sum().clip(lower=1)})
    return out.round(2)


# ---------------------------------------------------------------- TF-IDF

def tfidf_model(multiclass):
    tf = CONFIG["tfidf"]
    return make_pipeline(
        TfidfVectorizer(lowercase=tf["lowercase"], sublinear_tf=tf["sublinear_tf"], min_df=tf["min_df"],
                        ngram_range=tuple(tf["ngram_range"])),
        LogisticRegression(**CONFIG["logreg"]))


def folds(y, n_splits, n_repeats):
    return [list(StratifiedKFold(n_splits, shuffle=True, random_state=CONFIG["seed"] + r).split(np.zeros(len(y)), y))
            for r in range(n_repeats)]


def fold_metrics(y, P, multiclass, splits):
    """Métriques par fold (moyenne ± SD) à partir des probabilités hors-fold P (répétitions × n × C)."""
    aucs, baccs = [], []
    for r, rep in enumerate(splits):
        for _, tst in rep:
            p = P[r, tst]
            aucs.append(roc_auc_score(y[tst], p, multi_class="ovr") if multiclass else roc_auc_score(y[tst], p[:, 1]))
            baccs.append(balanced_accuracy_score(y[tst], p.argmax(1)))
    return np.array(aucs), np.array(baccs)


def pooled(y, P, multiclass):
    """AUC et BAcc sur les prédictions hors-fold regroupées, moyennées sur les répétitions."""
    a = np.mean([roc_auc_score(y, p, multi_class="ovr") if multiclass else roc_auc_score(y, p[:, 1]) for p in P])
    b = np.mean([balanced_accuracy_score(y, p.argmax(1)) for p in P])
    return np.array([a, b])


def oof_tfidf(texts, y, multiclass, splits):
    n_cls = len(np.unique(y))
    P = np.zeros((len(splits), len(y), n_cls))
    X = np.array(texts, dtype=object)
    for r, rep in enumerate(splits):
        for trn, tst in rep:
            P[r, tst] = tfidf_model(multiclass).fit(X[trn], y[trn]).predict_proba(X[tst])
    return P


def paired_permutation(y, P1, P2, multiclass, rng, n_perm):
    """H0 : les deux versions sont interchangeables. Échange aléatoire des prédictions par participant
    (le même échange pour toutes les répétitions) ; statistique = différence des métriques regroupées."""
    obs = pooled(y, P1, multiclass) - pooled(y, P2, multiclass)
    null = np.empty((n_perm, 2))
    for i in range(n_perm):
        swap = rng.random(len(y)) < .5
        Q1, Q2 = P1.copy(), P2.copy()
        Q1[:, swap], Q2[:, swap] = P2[:, swap], P1[:, swap]
        null[i] = pooled(y, Q1, multiclass) - pooled(y, Q2, multiclass)
    p = (1 + (np.abs(null) >= np.abs(obs)).sum(0)) / (n_perm + 1)
    return obs, p


def targets(meta):
    y3 = meta.diagnosis.map(CLASSES).to_numpy()
    return {"HC vs MCI+Dementia": ((y3 > 0).astype(int), False), "3 classes": (y3, True)}


def run_tfidf(meta, corpora):
    rng = np.random.default_rng(CONFIG["seed"])
    rows, diffs = [], []
    for tname, (y, multi) in targets(meta).items():
        splits = folds(y, CONFIG["n_splits"], CONFIG["n_repeats"])
        for t in TASKS:
            P = {v: oof_tfidf(corpora[(t, v)], y, multi, splits) for v in VERSIONS}
            for v in VERSIONS:
                auc, bacc = fold_metrics(y, P[v], multi, splits)
                pa, pb = pooled(y, P[v], multi)
                rows.append({"cible": tname, "task": t, "version": v, "AUC_mean": auc.mean(), "AUC_sd": auc.std(ddof=1),
                             "BAcc_mean": bacc.mean(), "BAcc_sd": bacc.std(ddof=1),
                             "AUC_pooled": pa, "BAcc_pooled": pb})
            for v1, v2 in CONFIG["comparisons"]:
                obs, p = paired_permutation(y, P[v1], P[v2], multi, rng, CONFIG["n_permutations"])
                diffs.append({"cible": tname, "task": t, "diff": f"{v1} − {v2}", "dAUC": obs[0], "p_dAUC": p[0],
                              "dBAcc": obs[1], "p_dBAcc": p[1]})
            print(f"  TF-IDF [{tname}] {t} : " + ", ".join(
                f"{r['version']} {r['AUC_mean']:.3f}" for r in rows[-len(VERSIONS):]), flush=True)
    return pd.DataFrame(rows).round(4), pd.DataFrame(diffs).round(4)


# ---------------------------------------------------------------- DistilBERT

def oof_bert(texts, y3, splits, log):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    b = CONFIG["bert"]
    dev = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(b["model"])
    enc = tok(list(texts), truncation=True, max_length=b["max_len"], padding="max_length", return_tensors="pt")
    P = np.zeros((len(splits), len(y3), 3))
    for r, rep in enumerate(splits):
        for k, (trn, tst) in enumerate(rep):
            t0 = time.time()
            torch.manual_seed(CONFIG["seed"] + 100 * r + k)
            model = AutoModelForSequenceClassification.from_pretrained(b["model"], num_labels=3).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=b["lr"], weight_decay=b["weight_decay"])
            w = torch.tensor(len(trn) / (3 * np.bincount(y3[trn], minlength=3)), dtype=torch.float32, device=dev)
            loss_fn = torch.nn.CrossEntropyLoss(weight=w)
            n_steps = b["epochs"] * int(np.ceil(len(trn) / b["batch_size"]))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / n_steps))
            scaler = torch.amp.GradScaler(enabled=b["fp16"])
            model.train()
            for _ in range(b["epochs"]):
                for idx in DataLoader(trn, batch_size=b["batch_size"], shuffle=True):
                    idx = idx.numpy()
                    with torch.autocast("cuda", dtype=torch.float16, enabled=b["fp16"]):
                        out = model(input_ids=enc["input_ids"][idx].to(dev),
                                    attention_mask=enc["attention_mask"][idx].to(dev))
                        loss = loss_fn(out.logits.float(), torch.tensor(y3[idx], device=dev))
                    opt.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=b["fp16"]):
                logits = torch.cat([model(input_ids=enc["input_ids"][tst[i:i + 32]].to(dev),
                                          attention_mask=enc["attention_mask"][tst[i:i + 32]].to(dev)).logits.float()
                                    for i in range(0, len(tst), 32)])
            P[r, tst] = torch.softmax(logits, -1).cpu().numpy()
            log.append({"seconds": time.time() - t0, "max_vram_gb": torch.cuda.max_memory_allocated() / 1e9})
            del model, opt
            torch.cuda.empty_cache()
    return P


def run_bert(meta, corpora):
    b = CONFIG["bert"]
    y3 = meta.diagnosis.map(CLASSES).to_numpy()
    yb = (y3 > 0).astype(int)
    splits = folds(y3, b["n_splits"], b["n_repeats"])
    rng = np.random.default_rng(CONFIG["seed"])
    rows, diffs, log = [], [], []
    for t in TASKS:
        P3 = {v: oof_bert(corpora[(t, v)], y3, splits, log) for v in b["versions"]}
        Pb = {v: np.stack([P3[v][..., 0], 1 - P3[v][..., 0]], -1) for v in b["versions"]}
        for tname, y, P, multi in [("HC vs MCI+Dementia", yb, Pb, False), ("3 classes", y3, P3, True)]:
            for v in b["versions"]:
                auc, bacc = fold_metrics(y, P[v], multi, splits)
                rows.append({"cible": tname, "task": t, "version": v, "AUC_mean": auc.mean(), "AUC_sd": auc.std(ddof=1),
                             "BAcc_mean": bacc.mean(), "BAcc_sd": bacc.std(ddof=1)})
            obs, p = paired_permutation(y, P["a_raw"], P["b_nomarks"], multi, rng, CONFIG["n_permutations"])
            diffs.append({"cible": tname, "task": t, "diff": "a_raw − b_nomarks", "dAUC": obs[0], "p_dAUC": p[0],
                          "dBAcc": obs[1], "p_dBAcc": p[1]})
        print(f"  DistilBERT {t} : fait ({np.mean([l['seconds'] for l in log]):.0f} s/fit, "
              f"VRAM max {max(l['max_vram_gb'] for l in log):.2f} Go)", flush=True)
    timing = {"n_fits": len(log), "mean_seconds_per_fit": float(np.mean([l["seconds"] for l in log])),
              "max_vram_gb": float(max(l["max_vram_gb"] for l in log))}
    return pd.DataFrame(rows).round(4), pd.DataFrame(diffs).round(4), timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert", action="store_true", help="ajoute DistilBERT (GPU)")
    ap.add_argument("--skip-tfidf", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_dataset()
    meta = meta[meta.Split == CONFIG["split"]].reset_index(drop=True)
    corpora, stats = build_corpora(meta)
    stats.to_csv(OUT / "corpus_stats_train.csv", index=False)
    ann = annotation_stats(meta)
    ann.to_csv(OUT / "parenthesis_stats_train.csv")
    print("== Parenthèses (train, agrégé) ==\n", ann)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("== Corpus (train) ==\n", stats)
    summary = {"generated": datetime.now().isoformat(timespec="seconds"), "n_train": len(meta)}
    if not args.skip_tfidf:
        res, diffs = run_tfidf(meta, corpora)
        res.to_csv(OUT / "tfidf_results_train.csv", index=False)
        diffs.to_csv(OUT / "tfidf_paired_differences_train.csv", index=False)
        print("\n== TF-IDF ==\n", res, "\n\n== Différences appariées ==\n", diffs)
    if args.bert:
        res_b, diffs_b, timing = run_bert(meta, corpora)
        res_b.to_csv(OUT / "distilbert_results_train.csv", index=False)
        diffs_b.to_csv(OUT / "distilbert_paired_differences_train.csv", index=False)
        summary["distilbert_timing"] = timing
        print("\n== DistilBERT ==\n", res_b, "\n\n== Différences ==\n", diffs_b, "\n", timing)
    (OUT / ("summary_bert.json" if args.bert else "summary.json")).write_text(json.dumps(summary, indent=2))
    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
