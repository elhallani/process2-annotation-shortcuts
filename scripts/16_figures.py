"""Figures de l'article (agrégées uniquement : médianes, intervalles, contours de densité ; aucun point individuel).

Fig. 1  Schéma de l'étude (audit + baseline sans annotation humaine)
Fig. 2  Raccourci d'annotation des pauses (régimes de transcription, densité, AUC annotées vs mesurées)
Fig. 3  Ablation texte : AUC par condition et par tâche (validation croisée train + test unique)
Fig. 4  Mesures interprétables : production par tranche de 15 s ; carte G–D (contours de densité par groupe)
Fig. 5  Comparaison des modèles sur le test (AUC, IC 95 %) et équité par sexe (train vs test)
Sorties : results/figures/fig{1..5}.png (300 dpi) et .pdf
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_dataset  # noqa: E402
from src.normative import crossfit_scores  # noqa: E402

R = ROOT / "results"
OUT = R / "figures"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GROUP = {"HC": "#2a78d6", "MCI": "#eb6834", "Dementia": "#1baf7a"}  # palette validée (3 couleurs, toutes paires)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "legend.frameon": False,
                     "font.family": "DejaVu Sans"})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1

def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    def box(x, y, w, h, title, body, edge=INK2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="white", ec=edge, lw=1))
        ax.text(x + .12, y + h - .18, title, fontsize=8.5, weight="bold", va="top", color=INK)
        ax.text(x + .12, y + h - .52, body, fontsize=7.2, va="top", color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))

    box(.1, 1.4, 2.0, 2.2, "PROCESS-2", "400 participants\nHC 200 · MCI 150\nDementia 50\n3 tasks per session:\nSFT, PFT, CTD\nTrain 320 · Test 80")
    box(2.6, 2.75, 3.3, 2.1, "A. Shortcut audit", "• demographics, channel, duration\n• second voice (pyannote,\n  ECAPA, blind listening)\n• transcription conventions\n• annotated vs measured pauses", GROUP["MCI"])
    box(2.6, .15, 3.3, 2.35, "B. Annotation-free baseline", "• Whisper → distinct valid\n  words (SFT, PFT)\n• VAD pauses (fluency, CTD)\n• HC norms (Huber) → G, D\n• weighted LR, norms refit\n  in every fold", GROUP["HC"])
    box(6.5, 1.4, 3.35, 2.2, "Evaluation", "• 5 × 5 CV on train\n• paired comparisons\n• official test: evaluated\n  once, bootstrap CIs\n• fairness by sex,\n  sensitivity (2nd voice)")
    arrow(2.1, 2.9, 2.6, 3.6)
    arrow(2.1, 2.1, 2.6, 1.4)
    arrow(5.9, 3.6, 6.5, 2.9)
    arrow(5.9, 1.4, 6.5, 2.1)
    save(fig, "fig1_overview")


# ---------------------------------------------------------------- Fig. 2

def fig2():
    conv = pd.read_csv(R / "06_pause_annotation" / "annotation_by_convention_diagnosis_train.csv")
    auc = pd.read_csv(R / "06_pause_annotation" / "single_variable_auc_train.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    groups = ["HC", "MCI", "Dementia"]
    for ax, col, ylab in [(axes[0], "pct_files_with_pause_annot", "Files with ≥ 1 annotated pause (%)"),
                          (axes[1], "annot_per_acoustic_gap", "Annotated pauses per\nmeasured pause ≥ 1 s")]:
        for j, g in enumerate(groups):
            vals = [conv[(conv.prefix_convention == pc) & (conv.diagnosis == g)][col].iloc[0] for pc in (False, True)]
            ax.bar(np.arange(2) + (j - 1) * .26, vals, width=.24, color=GROUP[g], label=g, edgecolor=SURF, lw=1)
        ax.set_xticks([0, 1], ["No \"Pat:\"\nprefix", "\"Pat:\"\nprefix"])
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=7.5, loc="upper left")
    axes[0].set_title("a  Two transcription regimes", loc="left", fontsize=9, color=INK)
    axes[1].set_title("b  Annotation density", loc="left", fontsize=9, color=INK)
    ax = axes[2]
    lab = {"n_annot": "Annotated pauses (text)", "n_gaps_1.0": "Pauses ≥ 1 s (audio)",
           "gaps_s_1.0": "Pause time ≥ 1 s (audio)"}
    tasks = ["SFT", "PFT", "CTD"]
    for k, (v, mk) in enumerate([("n_annot", "o"), ("n_gaps_1.0", "s"), ("gaps_s_1.0", "^")]):
        y = [auc[(auc.task == t) & (auc.variable == v)]["AUC_HC_vs_MCI+Dem"].iloc[0] for t in tasks]
        ax.plot(np.arange(3) + (k - 1) * .18, y, mk, ms=6, color=INK if v == "n_annot" else INK2,
                mfc=INK if v == "n_annot" else SURF, label=lab[v], lw=0)
    ax.axhline(.5, color=INK2, lw=.8, ls=":")
    ax.set_xticks(range(3), tasks)
    ax.set_ylim(.38, .75)
    ax.set_ylabel("Single-variable AUC\n(HC vs MCI+Dem.)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.16), ncol=1)
    ax.set_title("c  Text annotations \"beat\" audio", loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, "fig2_annotation_shortcut")


# ---------------------------------------------------------------- Fig. 3

def fig3():
    cv8 = pd.read_csv(R / "08_text_asr" / "results_train.csv")
    cv4 = pd.read_csv(R / "04_text_artifact" / "tfidf_results_train.csv")
    te = pd.read_csv(R / "14_final_test" / "test_binary.csv").set_index("modèle")
    conds = [("m_raw", "Manual, raw"), ("b1_noprefix", "Manual, speaker prefixes removed"),
             ("m_clean", "Manual, prefixes + annotations removed"), ("asr", "Whisper"),
             ("asr_pauses", "Whisper + measured pauses (audio)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), sharey=True)
    for ax, t in zip(axes, ["SFT", "PFT", "CTD"]):
        for i, (c, _) in enumerate(conds):
            y = len(conds) - 1 - i
            if c == "b1_noprefix":
                r = cv4[(cv4.task == t) & (cv4.version == c) & (cv4.cible == "HC vs MCI+Dementia")].iloc[0]
            else:
                r = cv8[(cv8.task == t) & (cv8.condition == c) & (cv8.cible == "HC vs MCI+Dementia")].iloc[0]
            ax.errorbar(r.AUC_mean, y + .14, xerr=r.AUC_sd, fmt="o", ms=5, color=INK2, mfc=SURF, lw=1, capsize=0)
            key = f"text_{c}_{t}"
            if key in te.index:
                e = te.loc[key]
                ax.errorbar(e.AUC, y - .14, xerr=[[e.AUC - e.AUC_lo], [e.AUC_hi - e.AUC]], fmt="D", ms=4.5,
                            color=INK, lw=1, capsize=0)
        ax.axvline(.5, color=INK2, lw=.8, ls=":")
        ax.set_xlim(.45, 1.0)
        ax.set_title(t, fontsize=9, color=INK)
        ax.set_xlabel("AUC (HC vs MCI+Dem.)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(conds)), [n for _, n in conds][::-1], fontsize=7.5)
    axes[0].plot([], [], "o", color=INK2, mfc=SURF, label="Train, 5 × 5 CV (mean ± SD)")
    axes[0].plot([], [], "D", color=INK, label="Official test (95% CI)")
    fig.legend(loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(.6, -.04))
    fig.tight_layout(rect=(0, .06, 1, 1))
    save(fig, "fig3_text_ablation")


# ---------------------------------------------------------------- Fig. 4

def fig4():
    meta = load_dataset()
    tr = meta[meta.Split == "Train"].set_index("participant_id")
    f = pd.read_csv(R / "cache" / "05_features" / "features_per_recording.csv", dtype={"participant_id": str})
    f = f[f.participant_id.isin(tr.index)]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    bins = ["units_per_s_bin1", "units_per_s_bin2", "units_per_s_bin3"]
    for ax, t in zip(axes[:2], ["SFT", "PFT"]):
        s = f[f.task == t]
        hc = s[s.diagnosis == "HC"][bins]
        x = np.array([7.5, 22.5, 37.5])
        ax.fill_between(x, hc.quantile(.25), hc.quantile(.75), color=GROUP["HC"], alpha=.15, lw=0,
                        label="HC: interquartile range")
        for g in GROUP:
            med = s[s.diagnosis == g][bins].median()
            ax.plot(x, med, "-o", color=GROUP[g], lw=2, ms=4, label=f"{g}: median")
        ax.set_xticks(x, ["0–15 s", "15–30 s", "30–45 s"])
        ax.set_xlim(3, 46)
        ax.set_title(f"{'a' if t == 'SFT' else 'b'}  {t}: output over time", loc="left", fontsize=9, color=INK)
    axes[0].set_ylabel("Speech units per second (VAD)")
    axes[1].legend(fontsize=6.5, loc="upper right")

    c = pd.read_csv(R / "cache" / "11_clinical" / "clinical_per_recording.csv", dtype={"participant_id": str})
    x = c[c.Split == "Train"].pivot(index="participant_id", columns="task", values="valid_per_s").add_prefix("x_")
    d = x.join(pd.DataFrame({"age": tr.age.astype(float), "male": (tr.gender == "male").astype(float),
                             "diagnosis": tr.diagnosis}))
    sc, _ = crossfit_scores(d, d.diagnosis == "HC", 5, 42)
    sc = sc.astype(float).join(d.diagnosis)
    ax = axes[2]
    gx, gy = np.mgrid[-4:3:200j, -4:3:200j]
    for g in GROUP:
        s = sc[sc.diagnosis == g][["G", "D"]].to_numpy().T
        k = gaussian_kde(s)
        z = k(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        lvl = np.quantile(k(s), .5)  # contour contenant ≈ 50 % du groupe
        ax.contour(gx, gy, z, levels=[lvl], colors=GROUP[g], linewidths=1.8)
        mg, md = np.median(s, axis=1)
        ax.plot(mg, md, "o", ms=7, color=GROUP[g], mec=SURF, mew=1.5)
        off = {"HC": (6, 6), "MCI": (6, -10), "Dementia": (-44, 8)}[g]
        ax.annotate(g, (mg, md), off, textcoords="offset points", fontsize=7, color=INK2)
    ax.axhline(0, color=INK2, lw=.6)
    ax.axvline(0, color=INK2, lw=.6)
    ax.set_xlim(-3.5, 2)
    ax.set_ylim(-3.5, 2.5)
    ax.set_xlabel("G: global level (z vs HC)")
    ax.set_ylabel("D: SFT | PFT dissociation (z)")
    ax.set_title("c  G–D map (clinical score)", loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, "fig4_interpretable_measures")


# ---------------------------------------------------------------- Fig. 5

def fig5():
    te = pd.read_csv(R / "14_final_test" / "test_binary.csv").set_index("modèle")
    rows = [("B_G_nocov", "Main: G + CTD (no demographics)", INK),
            ("B_G", "G + CTD + age, sex", INK2), ("C_G_D", "G + D + CTD + age, sex", INK2),
            ("wavlm_centered", "Frozen WavLM, centred profile", INK2), ("fusion", "Fusion G + WavLM", INK2),
            ("text_asr_pauses_CTD", "Text: Whisper + pauses (CTD)", INK2),
            ("text_m_clean_CTD", "Text: manual, cleaned (CTD)", INK2),
            ("text_m_raw_CTD", "Text: manual, raw (CTD) — artefact", GROUP["MCI"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for i, (k, name, col) in enumerate(rows):
        e = te.loc[k]
        y = len(rows) - 1 - i
        ax.errorbar(e.AUC, y, xerr=[[e.AUC - e.AUC_lo], [e.AUC_hi - e.AUC]], fmt="D" if i == 0 else "o",
                    ms=6 if i == 0 else 5, color=col, mfc=col if i in (0, len(rows) - 1) else SURF, lw=1.2, capsize=0)
        ax.text(1.005, y, f"{e.AUC:.2f}", fontsize=7, color=INK2, va="center")
    ax.axvline(.5, color=INK2, lw=.8, ls=":")
    ax.set_yticks(range(len(rows)), [r[1] for r in rows][::-1], fontsize=7.2)
    ax.set_xlim(.45, 1.0)
    ax.set_xlabel("Test AUC (HC vs MCI+Dem.), 95% CI")
    ax.grid(axis="y", visible=False)
    ax.set_title("a  Official test (n = 80)", loc="left", fontsize=9, color=INK)

    ax = axes[1]
    trf = pd.read_csv(R / "13_robustness_v2" / "fairness_by_sex_train.csv")
    tef = pd.read_csv(R / "14_final_test" / "test_fairness_sex.csv").set_index("groupe")
    for j, (g, mk) in enumerate([("hommes", "s"), ("femmes", "o")]):
        r = trf[(trf.groupe == g) & (trf.métrique == "AUC")].iloc[0]
        ax.errorbar(0 + (j - .5) * .2, r.valeur, yerr=[[r.valeur - r.IC_lo], [r.IC_hi - r.valeur]], fmt=mk, ms=5,
                    color=INK2, mfc=SURF if j else INK2, lw=1, capsize=0, label={"hommes": "Men", "femmes": "Women"}[g])
        e = tef.loc[g]
        ax.errorbar(1 + (j - .5) * .2, e.AUC, yerr=[[e.AUC - e.AUC_lo], [e.AUC_hi - e.AUC]], fmt=mk, ms=5,
                    color=INK2, mfc=SURF if j else INK2, lw=1, capsize=0)
    ax.axhline(.5, color=INK2, lw=.8, ls=":")
    ax.set_xticks([0, 1], ["Train (CV)", "Test"])
    ax.set_xlim(-.5, 1.5)
    ax.set_ylim(.45, 1.0)
    ax.set_ylabel("Main-model AUC")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="x", visible=False)
    ax.set_title("b  Fairness by sex", loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, "fig5_models_test")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5):
        f()
        print(f"{f.__name__} ok", flush=True)
    (OUT / "README.json").write_text(json.dumps({"contenu": "figures agrégées uniquement (aucun point individuel)",
                                                "palette": GROUP}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
