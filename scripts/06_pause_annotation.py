"""Étape 0f — Pauses annotées dans les transcriptions vs pauses mesurées dans l'audio (TRAIN uniquement).

Question : à pauses acoustiques égales, les transcriptions des HC contiennent-elles moins d'annotations
« (N second(s)) » ? Si oui, la densité d'annotation est un artefact de transcription lié au diagnostic.

Pauses annotées : regex « (N second(s)) » dans le .txt (enregistrement entier ; aucun texte écrit ni affiché).
Pauses acoustiques : silences entre îlots Silero VAD (cache 01_audit), enregistrement entier, seuil 1 s (grille 0,5–2 s).
Sorties agrégées dans results/06_pause_annotation/.
"""

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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import TASKS, load_dataset, recordings  # noqa: E402
from src.text import PREFIX, read_text  # noqa: E402

OUT = ROOT / "results" / "06_pause_annotation"
VAD = ROOT / "results" / "cache" / "01_audit" / "vad_segments.json"
CONFIG = {"split": "Train", "pause_regex": r"\((\d+)\s*seconds?\)", "acoustic_gap_s": 1.0,
          "gap_grid_s": [0.5, 1.0, 2.0], "glm": "Poisson, n_annot ~ log1p(n_gaps) + diagnosis + task, "
                                                "erreurs robustes groupées par participant"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_dataset()
    meta = meta[meta.Split == CONFIG["split"]]
    recs = recordings(meta).merge(meta[["participant_id"] + [f"txt_{t}" for t in TASKS]], on="participant_id")
    vad = json.loads(VAD.read_text())
    rx = re.compile(CONFIG["pause_regex"], re.I)
    rows = []
    for r in recs.itertuples():
        text = read_text(getattr(r, f"txt_{r.task}"))
        ann = [int(n) for n in rx.findall(text)]
        segs = vad[f"{r.participant_id}__{r.task}"]
        gaps = np.array([b[0] - a[1] for a, b in zip(segs[:-1], segs[1:])])
        row = {"participant_id": r.participant_id, "task": r.task, "diagnosis": r.diagnosis,
               "prefix_convention": any(PREFIX.match(l) for l in text.split("\n")),
               "n_annot": len(ann), "annot_s": sum(ann)}
        for g in CONFIG["gap_grid_s"]:
            row[f"n_gaps_{g}"] = int((gaps >= g).sum())
        row["gaps_s_1.0"] = float(gaps[gaps >= 1.0].sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    gcol = f"n_gaps_{CONFIG['acoustic_gap_s']}"

    # 1. Densité d'annotation vs pauses acoustiques, par tâche × diagnostic
    g = df.groupby(["task", "diagnosis"])
    desc = pd.DataFrame({
        "n_files": g.size(),
        "pct_files_with_pause_annot": g.n_annot.apply(lambda x: 100 * (x > 0).mean()),
        "median_n_annot": g.n_annot.median(),
        "median_acoustic_gaps_ge1s": g[gcol].median(),
        "annot_per_acoustic_gap": g.n_annot.sum() / g[gcol].sum(),
        "median_acoustic_gaps_in_files_without_annot": g.apply(lambda x: x.loc[x.n_annot == 0, gcol].median()),
        "spearman_annot_vs_gaps_annotated_files": g.apply(
            lambda x: spearmanr(x.loc[x.n_annot > 0, "n_annot"], x.loc[x.n_annot > 0, gcol])[0]
            if (x.n_annot > 0).sum() > 5 else np.nan),
    }).round(3)
    desc.to_csv(OUT / "annotation_vs_acoustic_by_task_diagnosis_train.csv")

    # 2. Par convention de transcription (préfixe de locuteur présent ou non)
    conv = df.groupby(["prefix_convention", "diagnosis"]).agg(
        n_files=("n_annot", "size"), pct_files_with_pause_annot=("n_annot", lambda x: 100 * (x > 0).mean()),
        annot_sum=("n_annot", "sum"), gaps_sum=(gcol, "sum"))
    conv["annot_per_acoustic_gap"] = conv.annot_sum / conv.gaps_sum
    conv = conv.drop(columns=["annot_sum", "gaps_sum"]).round(3)
    conv.to_csv(OUT / "annotation_by_convention_diagnosis_train.csv")

    # 3. GLM de Poisson : effet du diagnostic à pauses acoustiques égales
    d = df.assign(log_gaps=np.log1p(df[gcol]),
                  diagnosis=pd.Categorical(df.diagnosis, ["HC", "MCI", "Dementia"]))
    glm_rows = []
    for name, sub in [("tous les fichiers", d), ("fichiers annotés (≥ 1 pause notée)", d[d.n_annot > 0])]:
        m = smf.glm("n_annot ~ log_gaps + diagnosis + C(task)", sub, family=sm.families.Poisson()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(sub.participant_id)[0]})
        for term in ["diagnosis[T.MCI]", "diagnosis[T.Dementia]", "log_gaps"]:
            lo, hi = m.conf_int().loc[term]
            glm_rows.append({"échantillon": name, "n_files": len(sub), "terme": term,
                             "rate_ratio": np.exp(m.params[term]), "RR_lo": np.exp(lo), "RR_hi": np.exp(hi),
                             "p": m.pvalues[term]})
    glm = pd.DataFrame(glm_rows).round(4)
    glm.to_csv(OUT / "poisson_glm_train.csv", index=False)

    # 4. Pouvoir discriminant (AUC d'une seule variable, HC vs MCI+Dementia), par tâche
    auc = []
    for t in TASKS:
        s = df[df.task == t]
        y = (s.diagnosis != "HC").astype(int)
        for v in ["n_annot", gcol, "gaps_s_1.0"]:
            auc.append({"task": t, "variable": v, "AUC_HC_vs_MCI+Dem": roc_auc_score(y, s[v])})
    auc = pd.DataFrame(auc).round(3)
    auc.to_csv(OUT / "single_variable_auc_train.csv", index=False)

    # 5. Sensibilité au seuil acoustique
    sens = df.groupby("diagnosis")[[f"n_gaps_{g}" for g in CONFIG["gap_grid_s"]] + ["n_annot"]].sum()
    for gg in CONFIG["gap_grid_s"]:
        sens[f"annot_per_gap_{gg}"] = sens.n_annot / sens[f"n_gaps_{gg}"]
    sens.round(3).to_csv(OUT / "threshold_sensitivity_train.csv")

    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False))
    (OUT / "summary.json").write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                                                  "n_files": len(df)}, indent=2))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    for n, t in [("Annotation vs acoustique", desc), ("Par convention", conv), ("GLM Poisson", glm),
                 ("AUC une variable", auc), ("Sensibilité au seuil", sens.round(3))]:
        print(f"\n== {n} ==\n{t}")


if __name__ == "__main__":
    main()
