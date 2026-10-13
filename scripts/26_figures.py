"""Figures 2-7 (aggregated data only: medians, intervals, contours).

Elsevier sizes (190 mm double column, 90 mm single column), sans-serif text >= 6 pt, colour-blind-safe palette
(Okabe-Ito for groups; purple for the annotation shortcut; indigo for the annotation-free model), PNG at 300 dpi
and vector PDF. Figure 1 and the graphical abstract are produced by 31_fig1_overview.py.
  Fig. 2  Annotation shortcut: conventions, density, text vs audio, ablation by convention, taxonomy, ablation by category
  Fig. 3  Text ablation (training CV and single test evaluation) and token weights of the probe
  Fig. 4  Zero-shot local language models
  Fig. 5  Frozen speech embeddings and silence
  Fig. 6  Annotation-free interpretable model
  Fig. 7  Official test set and fairness by sex
Output: results/figures/."""

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_dataset  # noqa: E402
from src.normative import crossfit_scores  # noqa: E402

R = ROOT / "results"
OUT = R / "figures"
CACHE = R / "cache" / "26_figures"
MM = 1 / 25.4
W2 = 190 * MM  # Elsevier : double colonne 190 mm
W1 = 90 * MM   # simple colonne 90 mm
W15 = 140 * MM  # 1,5 colonne 140 mm
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
GROUP = {"HC": "#0072B2", "MCI": "#E69F00", "Dementia": "#009E73"}  # Okabe–Ito (daltonisme), réservé aux groupes
ACC_AUDIT, ACC_BASE, ACC_EVAL = "#AA3377", "#332288", "#52514e"  # raccourci / annotations ; modèle sans annotation ; neutre
plt.rcParams.update({
    # Normes Elsevier / Nature : sans empattement de métrique Arial, 7 pt (≥ 6 pt), labels de panneaux 8 pt gras
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": INK2, "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.2, "ps.fonttype": 42,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .5, "axes.axisbelow": True,
    "axes.linewidth": .7, "figure.facecolor": SURF, "axes.facecolor": SURF, "legend.frameon": False,
    "savefig.facecolor": SURF, "pdf.fonttype": 42})


def _load(name, path, argv=None):
    saved, sys.argv = sys.argv, [path] + (argv or [])
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = saved
    return mod


def panel(ax, letter, title):
    """Label de panneau : lettre minuscule en gras 8 pt, puis titre court en 7 pt (style des grandes revues)."""
    if letter:
        ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(0, 7), textcoords="offset points",
                    fontsize=8, weight="bold", color=INK, va="bottom", ha="left")
        ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(9, 7), textcoords="offset points",
                    fontsize=7, color=INK, va="bottom", ha="left")
    else:
        ax.set_title(title, loc="left", color=INK, pad=6, fontsize=7)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)





def fig2_annotation_shortcut():
    conv = pd.read_csv(R / "06_pause_annotation" / "annotation_by_convention_diagnosis_train.csv")
    auc = pd.read_csv(R / "06_pause_annotation" / "single_variable_auc_train.csv")
    ab = pd.read_csv(R / "17_corrections" / "AB_comparisons_nadeau_bengio_train.csv")
    tax = pd.read_csv(R / "34_final_taxonomy" / "final_taxonomy.csv").set_index("catégorie")
    abl = pd.read_csv(R / "35_ablation_by_category" / "ablation_by_category_train.csv")
    fig = plt.figure(figsize=(W2, 5.0))
    g1 = fig.add_gridspec(1, 3, wspace=.55, top=.9, bottom=.6, left=.08, right=.98)
    g2 = fig.add_gridspec(1, 3, wspace=.95, top=.4, bottom=.08, left=.08, right=.98, width_ratios=[.75, 1.15, 1.05])
    groups, tasks = list(GROUP), ["SFT", "PFT", "CTD"]

    # a, b : régime de transcription
    for ax, col, ylab, L, T in [(fig.add_subplot(g1[0, 0]), "pct_files_with_pause_annot",
                                 "Files with ≥ 1 annotated pause (%)", "a", "Annotations: one convention only"),
                                (fig.add_subplot(g1[0, 1]), "annot_per_acoustic_gap",
                                 "Annotations per measured pause ≥ 1 s", "b", "Where written, they track pauses")]:
        for j, g in enumerate(groups):
            vals = [conv[(conv.prefix_convention == pc) & (conv.diagnosis == g)][col].iloc[0] for pc in (False, True)]
            ax.bar(np.arange(2) + (j - 1) * .27, vals, width=.25, color=GROUP[g], label=g, edgecolor=SURF, lw=.6)
        ax.set_xticks([0, 1], ["not annotated", "annotated"])
        ax.set_xlabel("Transcription convention")
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
        panel(ax, L, T)
        if L == "a":
            ax.legend(loc="upper left", handlelength=1, fontsize=6.3)
    # c : texte vs audio
    ax = fig.add_subplot(g1[0, 2])
    for k, (v, mk, lab) in enumerate([("n_annot", "o", "annotated pauses (text)"),
                                      ("n_gaps_1.0", "s", "pause count (audio)"),
                                      ("gaps_s_1.0", "^", "pause time (audio)")]):
        y = [auc[(auc.task == t) & (auc.variable == v)]["AUC_HC_vs_MCI+Dem"].iloc[0] for t in tasks]
        filled = v == "n_annot"
        ax.plot(np.arange(3) + (k - 1) * .2, y, mk, ms=4.5, color=ACC_AUDIT if filled else INK2,
                mfc=ACC_AUDIT if filled else SURF, mew=.9, label=lab)
    ax.axhline(.5, color=MUTED, lw=.7, ls=":")
    ax.set_xticks(range(3), tasks)
    ax.set_ylim(.26, .75)
    ax.set_ylabel("Single-variable AUC")
    ax.legend(loc="lower right", handletextpad=.2, fontsize=6.2, borderaxespad=.1)
    ax.grid(axis="x", visible=False)
    panel(ax, "c", "Text annotations beat audio")

    # d : ablation à régime égal
    ax = fig.add_subplot(g2[0, 0])
    reg = {"tous": "all", "régime « Pat: »": "annot.", "régime sans préfixe": "not annot."}
    for i, t in enumerate(tasks):
        for j, (rk, rl) in enumerate(reg.items()):
            r = ab[(ab.task == t) & (ab["régime"] == rk)].iloc[0]
            ax.bar(i + (j - 1) * .27, r.dAUC_fold_mean, width=.25, color=[INK2, ACC_AUDIT, "#c9c8c2"][j],
                   edgecolor=SURF, lw=.6, label=rl if i == 0 else None)
    ax.axhline(0, color=INK2, lw=.7)
    ax.set_xticks(range(3), tasks)
    ax.set_ylim(-.01, .2)
    ax.set_ylabel("AUC loss, raw − cleaned")
    ax.legend(loc="upper center", ncol=3, handlelength=.8, columnspacing=.5, handletextpad=.3, fontsize=5.8,
              borderaxespad=.1)
    ax.grid(axis="x", visible=False)
    panel(ax, "d", "Loss only where written")

    # e : taxonomie
    cats = [("Timed pauses", "Timed pauses (N seconds)"), ("Non-verbal vocal events", "Non-verbal (sighs, laughs)"),
            ("Uncertain or inaudible speech", "Uncertain (?)"), ("Recording procedure or environment",
                                                               "Timer (buzzer sounds)")]
    ax = fig.add_subplot(g2[0, 1])
    for i, (k, lab) in enumerate(cats):
        for j, g in enumerate(groups):
            v = tax.loc[k, f"pct_fichiers_{g}_Train"]
            yy = len(cats) - 1 - i + (1 - j) * .26
            ax.barh(yy, v, height=.24, color=GROUP[g], edgecolor=SURF, lw=.5)
            ax.text(v + 1, yy, f"{v:.0f}", va="center", fontsize=5.8, color=INK2)
    ax.set_yticks(range(len(cats)), [f"{c[1]}\n{tax.loc[c[0], 'pct_des_annotations_train']:.0f}% of annotations"
                                     for c in cats][::-1], fontsize=6.2)
    ax.set_xlim(0, 95)
    ax.set_xlabel("Files containing the category (%)")
    ax.grid(axis="y", visible=False)
    panel(ax, "e", "What transcribers annotated")

    # f : ablation par catégorie
    vers = [("sans timed", "Timed\npauses"), ("sans procedure", "Timer"), ("sans nonverbal", "Non-\nverbal"),
            ("sans uncertain", "Uncer-\ntain"), ("sans aucune annotation", "All")]
    ax = fig.add_subplot(g2[0, 2])
    tcol = {"SFT": INK2, "PFT": MUTED, "CTD": "#c9c8c2"}
    for i, (v, lab) in enumerate(vers):
        for j, t in enumerate(tasks):
            r = abl[(abl.task == t) & (abl.version == v)].iloc[0]
            x = i + (j - 1) * .27
            ax.bar(x, r.perte_vs_ref, width=.25, color=tcol[t], edgecolor=SURF, lw=.5, label=t if i == 0 else None)
            if r.p_Holm < .05:
                ax.text(x, r.perte_vs_ref + .002, "*", ha="center", fontsize=7.5, color=ACC_AUDIT, weight="bold")
    ax.axhline(0, color=INK2, lw=.7)
    ax.axvspan(-.5, 1.5, color=ACC_AUDIT, alpha=.08, lw=0)
    ax.set_xticks(range(len(vers)), [v[1] for v in vers])
    ax.set_xlim(-.5, len(vers) - .5)
    ax.set_ylim(-.01, .12)
    ax.set_ylabel("AUC loss when removed")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=3, handlelength=1, columnspacing=.7, fontsize=6.2)
    panel(ax, "f", "Two categories carry it")
    save(fig, "fig2_annotation_shortcut")



def fig3_text_ablation():
    cv8 = pd.read_csv(R / "08_text_asr" / "results_train.csv")
    cv4 = pd.read_csv(R / "04_text_artifact" / "tfidf_results_train.csv")
    te = pd.read_csv(R / "14_final_test" / "test_binary.csv").set_index("modèle")
    conds = [("m_raw", "Manual, raw (with annotations)"),
             ("m_clean", "Manual, labels + annotations removed"), ("asr", "Whisper ASR (no annotation)"),
             ("asr_pauses", "Whisper + pauses measured in audio")]
    ccol = {"m_raw": ACC_AUDIT, "b1_noprefix": ACC_AUDIT, "m_clean": INK2, "asr": ACC_BASE, "asr_pauses": ACC_BASE}
    fig = plt.figure(figsize=(W2, 2.45))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, .2, .85], wspace=.08)
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, k], sharey=axes[0]) for k in (1, 2)]
    for a in axes[1:]:
        plt.setp(a.get_yticklabels(), visible=False)
    for ax, t, L in zip(axes, ["SFT", "PFT", "CTD"], "abc"):
        cvm = {}
        for i, (c, _) in enumerate(conds):
            y = len(conds) - 1 - i
            src = cv4[(cv4.task == t) & (cv4.version == c)] if c == "b1_noprefix" else cv8[(cv8.task == t) & (cv8.condition == c)]
            r = src[src.cible == "HC vs MCI+Dementia"].iloc[0]
            ax.errorbar(r.AUC_mean, y + .15, xerr=r.AUC_sd, fmt="o", ms=4, color=ccol[c], mfc=SURF, mew=1, lw=.9)
            cvm[c] = (r.AUC_mean, y + .15)
            key = f"text_{c}_{t}"
            if key in te.index:
                e = te.loc[key]
                ax.errorbar(e.AUC, y - .15, xerr=[[e.AUC - e.AUC_lo], [e.AUC_hi - e.AUC]], fmt="D", ms=3.8,
                            color=ccol[c], lw=.9)
        (xr, yr), (xc, yc) = cvm["m_raw"], cvm["m_clean"]
        x0 = .52
        ax.annotate("", (x0, yc), (x0, yr), arrowprops=dict(arrowstyle="-|>", color=ACC_AUDIT, lw=1.1,
                                                            shrinkA=0, shrinkB=0))
        ax.text(x0 + .012, (yr + yc) / 2, f"−{xr - xc:.2f}\nAUC", fontsize=6.5, color=ACC_AUDIT, weight="bold",
                va="center", linespacing=1)
        ax.axvline(.5, color=MUTED, lw=.7, ls=":")
        ax.set_xlim(.45, 1.0)
        ax.set_xlabel("AUC (HC vs MCI+Dem.)")
        ax.grid(axis="y", visible=False)
        panel(ax, L, {"SFT": "Semantic fluency", "PFT": "Phonemic fluency", "CTD": "Picture description"}[t])
    axes[0].set_yticks(range(len(conds)), [n for _, n in conds][::-1])
    # d : ce que lit la sonde (poids des tokens d'annotation, 25 folds d'entraînement, transcriptions brutes)
    sh = pd.read_csv(R / "30_tfidf_token_weights" / "annotation_weight_share_train.csv").set_index("task")
    tok = pd.read_csv(R / "30_tfidf_token_weights" / "named_annotation_tokens_train.csv")
    ax = fig.add_subplot(gs[0, 4])
    for i, t in enumerate(["SFT", "PFT", "CTD"]):
        y = 2 - i
        ax.barh(y + .17, sh.loc[t, "pct_tokens_annotation"], height=.32, color=MUTED)
        ax.barh(y - .17, sh.loc[t, "pct_masse_coef_annotation"], height=.32, color=ACC_AUDIT)
        r1 = int((tok[(tok.task == t) & (tok.token == "seconds")].rang_median).iloc[0])
        ax.text(sh.loc[t, "pct_masse_coef_annotation"] + .3, y - .17, f'"seconds": #{r1}', va="center",
                fontsize=6, color=ACC_AUDIT, weight="bold")
    ax.set_yticks(range(3), ["CTD", "PFT", "SFT"])
    ax.set_xlim(0, 12)
    ax.set_xlabel("Annotation tokens (%)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    extra = [Patch(color=MUTED, label="annotation tokens: % of vocabulary"),
             Patch(color=ACC_AUDIT, label="% of total |weight|")]
    panel(ax, "d", "The probe reads \"seconds\"")
    axes[0].plot([], [], "o", color=INK2, mfc=SURF, label="Train, 5 × 5 CV (mean ± SD)")
    axes[0].plot([], [], "D", color=INK2, label="Official test (95% CI)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h + extra, l + [e.get_label() for e in extra], loc="lower center", ncol=4, bbox_to_anchor=(.5, -.07),
               handlelength=1.2)
    fig.subplots_adjust(bottom=.24)
    save(fig, "fig3_text_ablation")



def words_per_bin(tr):
    path = CACHE / "valid_words_per_bin_train.csv"
    if path.exists():
        return pd.read_csv(path)
    m11 = _load("m11", "scripts/11_clinical_scores.py")
    f = pd.read_csv(R / "cache" / "05_features" / "features_per_recording.csv",
                    dtype={"participant_id": str}).set_index(["participant_id", "task"])
    rows = []
    for p in tr.itertuples():
        for t in ("SFT", "PFT"):
            dur = f.loc[(p.participant_id, t), "duration_s"]
            segs = json.loads(m11.ts_path(p.participant_id, t).read_text(encoding="utf-8"))
            words = m11.timed_words(segs, min(dur, 45.0))
            for k, (b0, b1) in enumerate([(0, 15), (15, 30), (30, 45)]):
                if dur < b1 - 5:
                    continue
                sel = [w for w in words if b0 <= w[1] < b1]
                rows.append({"diagnosis": p.diagnosis, "task": t, "bin": k,
                             "valid_per_s": m11.score(t, sel)[0] / (min(b1, dur) - b0)})
    d = pd.DataFrame(rows)  # une ligne par (participant, tâche, tranche) ; seules les médianes sont tracées
    d.to_csv(path, index=False)
    return d


def fig6_interpretable_model():
    meta = load_dataset()
    tr = meta[meta.Split == "Train"].reset_index(drop=True)
    wb = words_per_bin(tr)
    fig, axes = plt.subplots(1, 4, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1, 1, 1.1, .95], "wspace": .62})
    x = np.array([7.5, 22.5, 37.5])
    for ax, t, L in zip(axes[:2], ["SFT", "PFT"], "ab"):
        s = wb[wb.task == t]
        hc = s[s.diagnosis == "HC"].groupby("bin").valid_per_s
        ax.fill_between(x, hc.quantile(.25), hc.quantile(.75), color=GROUP["HC"], alpha=.14, lw=0, label="HC IQR")
        for g in GROUP:
            med = s[s.diagnosis == g].groupby("bin").valid_per_s.median()
            ax.plot(x, med.to_numpy(), "-o", color=GROUP[g], lw=1.6, ms=3.5, label=g)
        ax.set_xticks(x, ["0–15", "15–30", "30–45"])
        ax.set_xlabel("Time window (s)")
        panel(ax, L, {"SFT": "SFT: fewer words if impaired", "PFT": "PFT: same pattern"}[t])
    axes[0].set_ylabel("Distinct valid words per second")
    axes[1].legend(loc="upper right", handlelength=1.2)

    c = pd.read_csv(R / "cache" / "11_clinical" / "clinical_per_recording.csv", dtype={"participant_id": str})
    trm = tr.set_index("participant_id")
    xx = c[c.Split == "Train"].pivot(index="participant_id", columns="task", values="valid_per_s").add_prefix("x_")
    d = xx.join(pd.DataFrame({"age": trm.age.astype(float), "male": (trm.gender == "male").astype(float),
                              "diagnosis": trm.diagnosis}))
    sc, _ = crossfit_scores(d, d.diagnosis == "HC", 5, 42)
    sc = sc.astype(float).join(d.diagnosis)
    ax = axes[2]
    gx, gy = np.mgrid[-4.5:3:220j, -4.5:3:220j]
    off = {"HC": (10, 7), "MCI": (10, -9), "Dementia": (-8, -24)}
    for g in GROUP:
        s = sc[sc.diagnosis == g][["G", "D"]].to_numpy().T
        k = gaussian_kde(s)
        z = k(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        ax.contour(gx, gy, z, levels=[np.quantile(k(s), .5)], colors=GROUP[g], linewidths=1.4)
        mg, md = np.median(s, axis=1)
        ax.plot(mg, md, "o", ms=5, color=GROUP[g], mec=SURF, mew=1)
        ax.annotate(g, (mg, md), off[g], textcoords="offset points", fontsize=6.5, color=GROUP[g], weight="bold",
                    bbox=dict(boxstyle="round,pad=.15", fc="white", ec="none", alpha=.85), zorder=5)
    ax.axhline(0, color=MUTED, lw=.6)
    ax.axvline(0, color=MUTED, lw=.6)
    ax.set_xlim(-3.8, 2)
    ax.set_ylim(-3.6, 2.4)
    ax.set_xlabel("G: global level (z vs HC)")
    ax.set_ylabel("D: SFT | PFT dissociation (z)")
    panel(ax, "c", "Groups differ in G, not in D")

    m10 = _load("m10", "scripts/10_model.py", ["--config", "configs/model_clinical_v2.yaml"])
    T = m10.participant_table(tr)
    F, _ = m10.fold_features(T, T.iloc[:1])
    cols = m10.columns(F, m10.CFG["models"]["B_G_nocov"])
    yb = (T.diagnosis != "HC").astype(int).to_numpy()
    coef = pd.Series(m10.lr().fit(F[cols].to_numpy(float), yb).named_steps["logisticregression"].coef_[0], index=cols)
    names = {"G__valid_per_s": "Fluency", "G__median_gap_s": "Median gap", "G__long_silences_per_min": "Long silences",
             "G__units_slope": "Slope", "ctd__pause_rate_per_min": "CTD pauses", "ctd__articulation_rate": "CTD artic."}
    coef = coef.rename(names).sort_values()
    ax = axes[3]
    ax.barh(range(len(coef)), coef.to_numpy(), color=[INK2 if v > 0 else ACC_BASE for v in coef], height=.6)
    ax.set_yticks(range(len(coef)), coef.index)
    ax.axvline(0, color=INK2, lw=.7)
    ax.set_xlim(-1.15, .5)
    ax.text(-.05, len(coef) - .35, "← lower risk", ha="right", fontsize=6, color=ACC_BASE)
    ax.text(.05, len(coef) - .35, "higher →", ha="left", fontsize=6, color=INK2)
    ax.set_ylim(-.6, len(coef) - .1)
    ax.set_xlabel("Standardised coefficient")
    ax.grid(axis="y", visible=False)
    panel(ax, "d", "Fluency drives the score")
    pd.DataFrame({"coef": coef}).round(3).to_csv(OUT / "fig6d_coefficients_train.csv")
    save(fig, "fig6_interpretable_baseline")



def fig7_test_set():
    te = pd.read_csv(R / "14_final_test" / "test_binary.csv").set_index("modèle")
    rows = [("B_G_nocov", "Main: annotation-free G + CTD", "main"),
            ("B_G", "  + age and sex as predictors", "var"), ("C_G_D", "  + dissociation D", "var"),
            ("wavlm_centered", "Frozen WavLM (task-centred)", "deep"), ("fusion", "Fusion main + WavLM", "deep"),
            ("text_asr_pauses_CTD", "Text: Whisper + pauses (CTD)", "text"),
            ("text_m_clean_CTD", "Text: manual, cleaned (CTD)", "text"),
            ("text_m_raw_CTD", "Text: manual, raw (CTD)", "artefact")]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw={"width_ratios": [1.7, 1], "wspace": .35})
    ax = axes[0]
    for i, (k, name, kind) in enumerate(rows):
        e = te.loc[k]
        y = len(rows) - 1 - i
        col = {"main": ACC_BASE, "artefact": ACC_AUDIT}.get(kind, INK2)
        ax.errorbar(e.AUC, y, xerr=[[e.AUC - e.AUC_lo], [e.AUC_hi - e.AUC]], fmt="D" if kind == "main" else "o",
                    ms=5 if kind == "main" else 4, color=col, mfc=col if kind in ("main", "artefact") else SURF,
                    mew=.9, lw=1)
        ax.text(1.01, y, f"{e.AUC:.2f}", fontsize=6.5, color=INK2, va="center", transform=ax.get_yaxis_transform())
    ax.axvline(.5, color=MUTED, lw=.7, ls=":")
    m = te.loc["B_G_nocov"]
    ax.axvspan(m.AUC_lo, m.AUC_hi, color=ACC_BASE, alpha=.06, lw=0)
    ax.axvline(m.AUC, color=ACC_BASE, lw=.6, alpha=.5)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows][::-1])
    ax.set_xlim(.45, 1.0)
    ax.set_xlabel("Test AUC, HC vs MCI+Dementia (95% CI)")
    ax.grid(axis="y", visible=False)
    panel(ax, "a", "Test set (n = 80, once): annotation-free 0.81")
    ax = axes[1]
    trf = pd.read_csv(R / "13_robustness_v2" / "fairness_by_sex_train.csv")
    tef = pd.read_csv(R / "14_final_test" / "test_fairness_sex.csv").set_index("groupe")
    for j, (g, mk, lab) in enumerate([("hommes", "s", "Men"), ("femmes", "o", "Women")]):
        r = trf[(trf.groupe == g) & (trf["métrique"] == "AUC")].iloc[0]
        e = tef.loc[g]
        for xpos, v, lo, hi in [(0, r.valeur, r.IC_lo, r.IC_hi), (1, e.AUC, e.AUC_lo, e.AUC_hi)]:
            ax.errorbar(xpos + (j - .5) * .22, v, yerr=[[v - lo], [hi - v]], fmt=mk, ms=4.5, color=INK2,
                        mfc=INK2 if j == 0 else SURF, mew=.9, lw=.9, label=lab if xpos == 0 else None)
    ax.axhline(.5, color=MUTED, lw=.7, ls=":")
    ax.set_xticks([0, 1], ["Train (CV)", "Test"])
    ax.set_xlim(-.5, 1.5)
    ax.set_ylim(.45, 1.0)
    ax.set_ylabel("Main-model AUC")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    panel(ax, "b", "No stable sex gap")
    save(fig, "fig7_test_results")



def fig5_silence():
    a = pd.read_csv(R / "24_acoustic_models" / "cv_results_train.csv").set_index("modèle")
    w = pd.read_csv(R / "24_acoustic_models" / "whisper_nonspeech_control_train.csv").set_index("modèle")
    iso = pd.read_csv(R / "28_nonspeech_isolated" / "cv_results_train.csv").set_index("modèle")
    main = json.loads((R / "24_acoustic_models" / "summary.json").read_text())["AUC_principal"]
    dur = iso.loc["durée de non-parole (3 tâches, log)"].AUC
    items = [(f"WavLM L{l}", a.loc[f"wavlm_L{l}"].AUC_diag, a.loc[f"wavlm_L{l}_NON_PAROLE"].AUC_diag,
              iso.loc[f"wavlm{l}_iso"].AUC) for l in (8, 10, 12)] + \
            [(f"Whisper-small L{l}", a.loc[f"whisper_small_L{l}"].AUC_diag, w.loc[f"whisper_small_L{l}_NON_PAROLE"].AUC_diag,
              iso.loc[f"whisper{l}_iso"].AUC) for l in (8, 10)]
    fig, ax = plt.subplots(figsize=(W1, 2.75))
    for i, (name, sp, ns, ni) in enumerate(items):
        y = len(items) - 1 - i
        ax.plot([min(sp, ns, ni), max(sp, ns, ni)], [y, y], color=GRID, lw=2.2, zorder=1)
        ax.plot(sp, y, "o", ms=5, color=ACC_BASE, zorder=3, label="speech frames" if i == 0 else None)
        ax.plot(ns, y, "o", ms=5, color=INK2, mfc=SURF, mew=1.1, zorder=3,
                label="non-speech frames, full-context encoding" if i == 0 else None)
        ax.plot(ni, y, "o", ms=5, color=INK2, zorder=4,
                label="non-speech audio isolated before encoding" if i == 0 else None)
    ax.axvline(main, color=INK, lw=.8, ls="--")
    ax.text(main + .003, -.55, "main model", fontsize=6.3, color=INK2, va="bottom")
    ax.axvline(dur, color=MUTED, lw=.8, ls="-.")
    ax.text(dur - .003, -.55, "non-speech\nduration only", fontsize=6.3, color=INK2, va="bottom", ha="right")
    ax.axvline(.5, color=MUTED, lw=.7, ls=":")
    ax.set_yticks(range(len(items)), [n for n, *_ in items][::-1])
    ax.set_xlim(.55, .8)
    ax.set_ylim(-.7, len(items) + 1.2)
    ax.set_xlabel("Train CV AUC (HC vs MCI+Dem.)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", fontsize=6.2, handletextpad=.3, borderaxespad=.2)
    panel(ax, "", "Isolated silence ≈ amount of pausing")
    save(fig, "fig5_silence_shortcut")



def fig4_llm_zero_shot():
    cmp_ = pd.read_csv(R / "27_llm" / "paired_comparisons_train.csv")
    ab = pd.read_csv(R / "17_corrections" / "AB_comparisons_nadeau_bengio_train.csv")
    auc = pd.read_csv(R / "27_llm" / "auc_by_model_task_version_train.csv")
    models = [("qwen3_4b", "Qwen3-4B"), ("phi4_mini", "Phi-4-mini"), ("medgemma_4b", "MedGemma-4B")]
    tasks = ["SFT", "PFT", "CTD"]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.7), gridspec_kw={"width_ratios": [1.2, 1.05, .9], "wspace": 1.0})
    for ax, (cmpname, L, title) in zip(axes[:2], [("m_raw − m_clean", "a", "Annotations: opposite effects"),
                                                  ("asr_pauses − asr", "b", "Measured pauses: no effect")]):
        rows = []
        for t in tasks:
            if cmpname == "m_raw − m_clean":
                r = ab[(ab.task == t) & (ab["régime"] == "tous")].iloc[0]
                rows.append(("TF-IDF (supervised)", t, r.dAUC_fold_mean, None, None, "tfidf"))
            for m, lab in models:
                c = cmp_[(cmp_["modèle"] == m) & (cmp_.task == t) & (cmp_.comparaison == cmpname)].iloc[0]
                rows.append((lab, t, c.dAUC, c.IC_lo, c.IC_hi, "llm"))
        ylabels = []
        for i, (lab, t, d, lo, hi, kind) in enumerate(rows):
            y = len(rows) - 1 - i
            ylabels.append(f"{lab} · {t}")
            if kind == "tfidf":
                ax.plot(d, y, "D", ms=4.5, color=ACC_AUDIT, mec=ACC_AUDIT)
            else:
                sig = lo > 0 or hi < 0
                ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", ms=3.8, color=INK if sig else MUTED,
                            mfc=INK if sig else SURF, mew=.9, lw=.9)
        ax.axvline(0, color=INK2, lw=.7)
        ax.axvspan(-.03, .03, color=GRID, alpha=.5, lw=0)
        ax.set_yticks(range(len(rows)), ylabels[::-1], fontsize=6.2)
        ax.set_xlabel("ΔAUC (with − without)")
        if cmpname != "m_raw − m_clean":
            ax.set_xlim(-.065, .065)
            ax.set_xticks([-.05, 0, .05])
        ax.grid(axis="y", visible=False)
        panel(ax, L, title)
    ax = axes[2]
    best = auc[auc.version == "m_clean"].groupby("modèle").AUC_moyenne_variantes.max()
    cv8 = pd.read_csv(R / "08_text_asr" / "results_train.csv")
    cv8 = cv8[cv8.cible == "HC vs MCI+Dementia"]
    tf_raw = cv8[cv8.condition == "m_raw"].AUC_mean.max()
    tf_clean = cv8[cv8.condition == "m_clean"].AUC_mean.max()
    vals = [("Main model\n(annotation-free)", .746, ACC_BASE), ("TF-IDF, raw\n(annotated)", tf_raw, ACC_AUDIT),
            ("TF-IDF, cleaned", tf_clean, INK2)] + \
           [(f"{lab}, cleaned\n(zero-shot)", best[m], MUTED) for m, lab in models]
    for i, (lab, v, col) in enumerate(vals):
        ax.barh(len(vals) - 1 - i, v - .5, left=.5, color=col, height=.6)
        ax.text(v + .005, len(vals) - 1 - i, f"{v:.2f}", va="center", fontsize=6.3, color=INK2)
    ax.set_yticks(range(len(vals)), [v[0] for v in vals][::-1], fontsize=6.2)
    ax.set_xlim(.5, .85)
    ax.set_xlabel("Best train AUC")
    ax.grid(axis="y", visible=False)
    panel(ax, "c", "LLMs stay below")
    fig.text(.08, -.1, "Purple diamonds: supervised TF-IDF   Filled circles: 95% CI excludes 0   Open circles: CI includes 0   "
             "grey band: ±0.03 equivalence margin", fontsize=6.2, color=INK2)
    save(fig, "fig4_llm_zero_shot")



def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    for f in (fig2_annotation_shortcut, fig3_text_ablation, fig4_llm_zero_shot,
              fig5_silence, fig6_interpretable_model, fig7_test_set):
        f()
        print(f"{f.__name__} ok", flush=True)
    (OUT / "README.json").write_text(json.dumps({"content": "aggregated figures only (no individual data point)",
                                                "group_palette": GROUP, "double_column_width_in": W2},
                                               indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
