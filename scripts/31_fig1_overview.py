"""Figure 1 (illustrated study overview) and the graphical abstract (Elsevier format, 13 x 5.2 cm).

Drawn icons, numbered steps and coloured panels.

Aggregated numbers only, read from results/. Transcript icons are schematic; the "3 s" tags illustrate the format
of timed pause annotations and contain no real content. Coordinates in millimetres (190 x 150 mm)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("f26", ROOT / "scripts" / "26_figures.py")
f26 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f26)
plt, R, MM = f26.plt, f26.R, f26.MM
INK, INK2, MUTED, GRID = f26.INK, f26.INK2, f26.MUTED, f26.GRID
OR, BL = f26.ACC_AUDIT, f26.ACC_BASE                      # accents : raccourci / audit ; modèle sans annotation
GH, GM, GR = f26.GROUP["HC"], f26.GROUP["MCI"], f26.GROUP["Dementia"]   # couleurs de groupe uniquement
OR_BG, BL_BG, GY_BG = "#f8eaf2", "#ecebf6", "#f4f4f2"
W, H = 190, 150


# ---------------------------------------------------------------- primitives

def card(ax, x, y, w, h, fc, ec, lw=.8, ls="-", r=2.2, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z))


def txt(ax, x, y, s, size=6.5, color=INK, weight="normal", ha="left", va="center", style="normal", z=6):
    size = max(size, 6.0)  # norme Elsevier : ≥ 6 pt
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, style=style, zorder=z,
            linespacing=1.15)


def num(ax, x, y, k, color):
    ax.add_patch(Circle((x, y), 2.6, fc=color, ec="none", zorder=5))
    txt(ax, x, y, str(k), 7, "white", "bold", "center")


def arrow(ax, x1, y1, x2, y2, color=INK2, lw=1.1, rad=0.0, ms=7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4, shrinkA=0, shrinkB=0))


def doc(ax, x, y, w, h, color=INK2, tag=None, tag_rows=(1,), n=5):
    """Icône de transcription : feuille à coin plié + lignes ; tag = étiquette d'annotation sur certaines lignes."""
    c = min(w, h) * .28
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h - c), (x + w - c, y + h), (x, y + h)], closed=True,
                         fc="white", ec=color, lw=.7, zorder=5))
    ax.add_patch(Polygon([(x + w - c, y + h), (x + w - c, y + h - c), (x + w, y + h - c)], closed=True, fc=GRID,
                         ec=color, lw=.5, zorder=5))
    for i in range(n):
        yy = y + h - c - 1.3 - i * (h - c - 2) / max(n - 1, 1)
        lw_ = w * (.75 if i % 2 else .6)
        ax.plot([x + 1.2, x + 1.2 + lw_ * .55], [yy, yy], color=MUTED, lw=.9, solid_capstyle="round", zorder=6)
        if tag and i in tag_rows:
            card(ax, x + 1.3 + lw_ * .5, yy - 1.15, w * .6, 2.3, OR_BG, OR, lw=.5, r=.6, z=6)
            txt(ax, x + 1.3 + lw_ * .5 + w * .3, yy, tag, 4.6, OR, "bold", "center", z=7)


def wave(ax, x, y, w, h, color=INK2, seed=3, silences=()):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 160)
    env = .25 + .75 * np.abs(np.sin(t * 9.4)) ** .7
    for a, b in silences:
        env[(t >= a) & (t <= b)] = .05
    yy = y + h / 2 + (h / 2) * env * np.sin(t * 160) * rng.uniform(.6, 1, len(t))
    ax.plot(x + t * w, yy, color=color, lw=.6, zorder=6)


def db(ax, x, y, w, h, color):
    e = h * .22
    ax.add_patch(Rectangle((x, y + e / 2), w, h - e, fc=color, ec="none", zorder=5))
    for k, yy in enumerate([y + e / 2, y + h - e / 2]):
        ax.add_patch(Ellipse((x + w / 2, yy), w, e, fc=color if k == 0 else "#8f86d6", ec="white", lw=.5, zorder=6))
    for f in (.38, .68):
        ax.add_patch(Arc((x + w / 2, y + e / 2 + (h - e) * f), w, e, theta1=180, theta2=360, ec="white", lw=.5,
                         zorder=6))


def bell(ax, x, y, w, h, color):
    t = np.linspace(-3, 3, 100)
    g = np.exp(-t ** 2 / 2)
    ax.fill_between(x + (t + 3) / 6 * w, y, y + g * h, color=color, alpha=.25, lw=0, zorder=5)
    ax.plot(x + (t + 3) / 6 * w, y + g * h, color=color, lw=.9, zorder=6)
    ax.plot([x, x + w], [y, y], color=INK2, lw=.5, zorder=6)
    ax.plot([x + w * .2] * 2, [y, y + h * .55], color=OR, lw=.9, zorder=6)
    txt(ax, x + w * .2, y + h * .72, "z", 5.5, OR, "bold", "center")


def gauge(ax, x, y, r, frac=.72):
    cols = ["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"]
    for i, c in enumerate(cols):
        ax.add_patch(Wedge((x, y), r, 180 - (i + 1) * 36, 180 - i * 36, width=r * .35, fc=c, ec="white", lw=.4,
                           zorder=5))
    a = np.pi * (1 - frac)
    ax.plot([x, x + .8 * r * np.cos(a)], [y, y + .8 * r * np.sin(a)], color=INK, lw=1.1, zorder=6)
    ax.add_patch(Circle((x, y), r * .1, fc=INK, zorder=7))


def person(ax, x, y, s, color):
    ax.add_patch(Circle((x, y + s * .78), s * .2, fc=color, ec="none", zorder=5))
    ax.add_patch(Wedge((x, y), s * .42, 0, 180, fc=color, ec="none", zorder=5))


def lock(ax, x, y, s, color):
    ax.add_patch(Arc((x + s / 2, y + s * .6), s * .58, s * .75, theta1=0, theta2=180, ec=color, lw=1.6, zorder=5))
    ax.plot([x + s * .21] * 2, [y + s * .45, y + s * .6], color=color, lw=1.6, zorder=5)
    ax.plot([x + s * .79] * 2, [y + s * .45, y + s * .6], color=color, lw=1.6, zorder=5)
    card(ax, x, y, s, s * .6, color, color, r=.4, z=5)
    ax.add_patch(Circle((x + s / 2, y + s * .3), s * .09, fc="white", zorder=6))


def target(ax, x, y, r, color):
    for k, f in enumerate((1, .66, .33)):
        ax.add_patch(Circle((x, y), r * f, fc=color if k % 2 == 0 else "white", ec=color, lw=.5, zorder=5 + k))


def folds(ax, x, y, w, h, color):
    for i in range(5):
        for j in range(5):
            fc = color if i == j else "white"
            ax.add_patch(Rectangle((x + j * w / 5, y + (4 - i) * h / 5), w / 5 * .86, h / 5 * .8, fc=fc, ec=color,
                                   lw=.4, zorder=5))


# ---------------------------------------------------------------- figure

def transcript_box(ax, x, y, w, h, title, lines, ec, title_col):
    """Transcript shown as text (illustrative example, not corpus data)."""
    card(ax, x, y, w, h, "white", ec, lw=.8, r=1.2, z=4)
    txt(ax, x + 1.5, y + h - 2.2, title, 6.2, title_col, "bold")
    for k, segs in enumerate(lines):
        xx = x + 1.8
        yy = y + h - 6.2 - k * 3.6
        for s, col, bold in segs:
            ax.text(xx, yy, s, fontsize=7, color=col, family="monospace", weight="bold" if bold else "normal",
                    va="center", ha="left", zorder=7)
            xx += len(s) * 1.52


def main():
    sess = pd.read_csv(R / "29_regime_checks" / "A_regime_sessions_train.csv").set_index("diagnosis")
    ann = {g: sess.loc[g, "pct_sessions_regime_Pat_majorité"] for g in ("HC", "MCI", "Dementia")}
    tok = pd.read_csv(R / "30_tfidf_token_weights" / "named_annotation_tokens_train.csv")
    sft = tok[tok.task == "SFT"].set_index("token").coef_median
    first = int((tok[tok.token == "seconds"].rang_median == 1).sum())
    ab = pd.read_csv(R / "17_corrections" / "AB_comparisons_nadeau_bengio_train.csv")
    ab = ab[ab.task == "SFT"].set_index("régime")
    abl = pd.read_csv(R / "35_ablation_by_category" / "ablation_by_category_train.csv")
    abl = abl[abl.task == "SFT"].set_index("version")
    te = pd.read_csv(R / "14_final_test" / "test_binary.csv").set_index("modèle").loc["B_G_nocov"]

    W_, H_ = 190, 148
    fig = plt.figure(figsize=(W_ * MM, H_ * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_)
    ax.set_ylim(0, H_)
    ax.set_axis_off()

    # ------------------------------------------------------------ A. le problème
    card(ax, 2, 99, 186, 47, OR_BG, OR, lw=.9, ls=(0, (4, 2)))
    txt(ax, 6, 142, "A", 9, OR, "bold")
    txt(ax, 11, 142, "The problem: the same speech is transcribed in two ways, and the way depends on diagnosis",
        7.6, INK, "bold")
    txt(ax, 11, 137.6, "PROCESS-2: 400 participants (200 HC, 150 MCI, 50 dementia), three tasks; example below is "
        "illustrative", 6, INK2, style="italic")
    # parole
    card(ax, 5, 104, 33, 29, "white", MUTED, lw=.6, r=1.4)
    txt(ax, 21.5, 129.8, "What the participant says", 6.3, INK, "bold", "center")
    wave(ax, 7, 117, 29, 9, INK2, seed=4, silences=[(.36, .62)])
    txt(ax, 21.5, 114, "“cat … dog, lion”", 6.6, INK, ha="center", style="italic")
    txt(ax, 21.5, 109.6, "with a 3-s silence", 6, INK2, ha="center")
    txt(ax, 21.5, 106.6, "after “cat”", 6, INK2, ha="center")
    # deux transcriptions
    arrow(ax, 38.5, 124, 45.5, 126, INK2, 1.1)
    arrow(ax, 38.5, 113, 45.5, 110.5, INK2, 1.1)
    transcript_box(ax, 46, 118.5, 52, 15, "Convention 1: pauses written",
                   [[("Pat: cat ", INK, False), ("(3 seconds)", OR, True)], [("     dog lion", INK, False)]], OR, OR)
    transcript_box(ax, 46, 103, 52, 13.5, "Convention 2: pauses never written",
                   [[("cat dog lion", INK, False)]], MUTED, INK2)
    # qui reçoit quelle convention
    txt(ax, 104, 131, "Sessions transcribed with convention 1", 6.3, INK, "bold")
    for k, (g, lab) in enumerate([("HC", "HC"), ("MCI", "MCI"), ("Dementia", "Dementia")]):
        yy = 125.5 - k * 6.5
        txt(ax, 104, yy, lab, 6.3, INK2)
        x0, L = 117, 40
        ax.add_patch(Rectangle((x0, yy - 1.6), L * ann[g] / 100, 3.2, fc=OR, ec="none", zorder=5))
        ax.add_patch(Rectangle((x0 + L * ann[g] / 100, yy - 1.6), L * (1 - ann[g] / 100), 3.2, fc="#d9d8d3",
                               ec="none", zorder=5))
        txt(ax, x0 + 1, yy, f"{ann[g]:.0f}%", 6.2, "white", "bold")
    txt(ax, 104, 106.8, "Missing annotations are not missing pauses:", 6, INK2)
    txt(ax, 104, 103.8, "HC without them pause as much in the audio.", 6, INK2)
    # conséquence
    card(ax, 161, 104, 24, 29, "white", OR, lw=1, r=1.4)
    ax.add_patch(Polygon([(173, 129.5), (170, 124), (176, 124)], closed=True, fc=OR, ec="none", zorder=5))
    txt(ax, 173, 125.6, "!", 7, "white", "bold", "center")
    txt(ax, 173, 119.5, "A text model", 6.3, INK, "bold", "center")
    txt(ax, 173, 116.3, "can learn:", 6.3, INK, ha="center")
    txt(ax, 173, 111.5, "“seconds”", 7, OR, "bold", "center")
    txt(ax, 173, 107.5, "→ impaired", 7, OR, "bold", "center")
    arrow(ax, 157.5, 118.5, 160.7, 118.5, OR, 1.2)

    # ------------------------------------------------------------ B. l'audit
    card(ax, 2, 51, 186, 45, OR_BG, OR, lw=.9, ls=(0, (4, 2)))
    txt(ax, 6, 92, "B", 9, OR, "bold")
    txt(ax, 11, 92, "The audit: is the gain due to the annotations or to the speech?", 7.6, INK, "bold")
    steps = [(5, "1", "What does the model read?"), (51.5, "2", "Remove the annotations"),
             (98, "3", "Which annotations matter?"), (144.5, "4", "Other checks")]
    for x0, k, t in steps:
        card(ax, x0, 54, 41.5, 33.5, "white", "#e3b3cd", lw=.6, r=1.6)
        num(ax, x0 + 4, 83.5, k, OR)
        txt(ax, x0 + 8, 83.5, t, 6.6, INK, "bold")
    # 1 poids de la sonde
    z0 = 31
    txt(ax, z0 - .6, 78.4, "← healthy", 6, BL, ha="right")
    txt(ax, z0 + .6, 78.4, "impaired →", 6, OR)
    for k, key in enumerate(["seconds", "second", "buzzer"]):
        v = sft[key] / sft["seconds"]
        yy = 74.5 - k * 3.8
        ax.add_patch(Rectangle((z0 if v > 0 else z0 + v * 12, yy - 1.2), abs(v) * 12, 2.4, fc=OR, ec="none", zorder=5))
        txt(ax, 7, yy, f"“{key}”", 6.2, OR, "bold")
    ax.plot([z0, z0], [65.5, 77], color=INK2, lw=.6, zorder=6)
    txt(ax, 25.75, 61, f"“seconds” is the top word in", 6, INK, ha="center")
    txt(ax, 25.75, 57.8, f"{first}/3 tasks and 25/25 CV folds", 6, INK, "bold", "center")
    # 2 ablation par convention
    for k, (lab, key, col) in enumerate([("all sessions", "tous", INK), ("convention 1", "régime « Pat: »", OR),
                                         ("convention 2", "régime sans préfixe", MUTED)]):
        a, b = ab.loc[key, "AUC_1_fold_mean"], ab.loc[key, "AUC_2_fold_mean"]
        y0 = 78.5 - k * 6.6
        txt(ax, 54, y0, lab, 6.2, col, "bold")
        txt(ax, 90.5, y0, f"−{a - b:.2f}" if a - b > 0 else f"+{b - a:.2f}", 6.4, col, "bold", "right")
        xa, xb = 56 + (a - .55) * 110, 56 + (b - .55) * 110
        ax.plot([55, 90], [y0 - 2.7, y0 - 2.7], color=GRID, lw=.5, zorder=4)
        ax.add_patch(Circle((xa, y0 - 2.7), .85, fc=OR, ec="none", zorder=6))
        if abs(xa - xb) > 1.5:
            arrow(ax, xa - .9, y0 - 2.7, xb + .2, y0 - 2.7, col, 1.2, ms=5)
        ax.add_patch(Circle((xb, y0 - 2.7), .85, fc=INK2, ec="none", zorder=6))
    txt(ax, 72.25, 57.8, "AUC lost only where annotations exist", 6, INK, "bold", "center")
    # 3 ablation par catégorie
    for k, (lab, v, c) in enumerate([("timed pauses", "sans timed", OR), ("timer “buzzer”", "sans procedure", OR),
                                     ("sighs, laughs", "sans nonverbal", MUTED), ("uncertain “(?)”", "sans uncertain", MUTED)]):
        yy = 78.5 - k * 4.6
        loss = max(abl.loc[v, "perte_vs_ref"], 0)
        txt(ax, 100, yy, lab, 6.2, OR if c == OR else INK2, "bold" if c == OR else "normal")
        ax.add_patch(Rectangle((119, yy - 1.2), max(loss * 200, .3), 2.4, fc=c, ec="none", zorder=5))
        txt(ax, 120 + loss * 200, yy, f"−{loss:.2f}" if loss >= .005 else "0.00", 6.2, OR if c == OR else INK2, "bold" if c == OR else "normal")
    txt(ax, 118.75, 57.8, "two categories carry the shortcut", 6, INK, "bold", "center")
    # 4 autres contrôles
    for k, (a, b) in enumerate([("demographics, channel", "weak (AUC ≤ 0.56)"),
                                ("zero-shot language models", "no prior use of annotations"),
                                ("silence fed alone to audio", "≈ amount of pausing"),
                                ("cleaned transcripts", "other habits remain")]):
        yy = 78.6 - k * 6.1
        ax.add_patch(Circle((147.5, yy - .6), .9, fc=INK2, ec="none", zorder=5))
        txt(ax, 149.5, yy, a, 6, INK)
        txt(ax, 149.5, yy - 2.7, b, 6, INK2, "bold")

    # ------------------------------------------------------------ C. la solution
    card(ax, 2, 2, 186, 46, BL_BG, BL, lw=.9, ls=(0, (4, 2)))
    txt(ax, 6, 44, "C", 9, BL, "bold")
    txt(ax, 11, 44, "The control: a model that never reads a human annotation", 7.6, INK, "bold")
    boxes = [(5, "Audio only", "same input for everyone"), (40, "Voice activity", "pauses measured in audio"),
             (75, "Speech recognition", "valid words per second"), (110, "Healthy-control norms", "z-scores by age and sex"),
             (145, "Risk score", "6 visible weights")]
    for i, (x0, t, d) in enumerate(boxes):
        card(ax, x0, 21, 31, 18, "white", "#c3bfe6", lw=.6, r=1.4)
        txt(ax, x0 + 15.5, 36, t, 6.6, INK, "bold", "center")
        txt(ax, x0 + 15.5, 23.5, d, 6, INK2, ha="center")
        if i < len(boxes) - 1:
            arrow(ax, x0 + 31.3, 30, x0 + 34.7, 30, BL, 1.2)
    wave(ax, 8, 27, 25, 6, BL, seed=5, silences=[(.3, .45), (.7, .8)])
    for k, (a, b) in enumerate([(.0, .3), (.45, .7), (.8, 1.)]):
        ax.add_patch(Rectangle((43 + a * 25, 28), (b - a) * 25, 4, fc=BL, ec="none", alpha=.85, zorder=5))
    for k, (a, b) in enumerate([(.3, .45), (.7, .8)]):
        ax.add_patch(Rectangle((43 + a * 25, 28), (b - a) * 25, 4, fc="#d9d8d3", ec="none", zorder=5))
    ax.text(90.5, 30, "cat  dog  lion", fontsize=6.6, family="monospace", ha="center", va="center", color=INK, zorder=6)
    bell(ax, 117.5, 27, 16, 6.5, BL)
    gauge(ax, 160.5, 27.2, 5.5)
    card(ax, 40, 5, 110, 12.5, "white", BL, lw=1.1, r=1.6)
    txt(ax, 95, 13.6, f"Official test set, evaluated once:  AUC {te.AUC:.2f} [{te.AUC_lo:.2f}, {te.AUC_hi:.2f}]", 7.4,
        BL, "bold", "center")
    txt(ax, 95, 8.6, "same performance under both transcription conventions; all choices made on the training set",
        6.1, INK2, ha="center")

    for ext in ("png", "pdf"):
        fig.savefig(f26.OUT / f"fig1_system_overview.{ext}", dpi=400 if ext == "png" else None)
    plt.close(fig)
    print("fig1 ok", json.dumps({k: round(v, 1) for k, v in ann.items()}))


def graphical_abstract():
    """Graphical abstract, Elsevier format (13 x 5.2 cm): problem -> evidence -> annotation-free control."""
    sess = pd.read_csv(R / "29_regime_checks" / "A_regime_sessions_train.csv").set_index("diagnosis")
    no_ann = {g: 100 - sess.loc[g, "pct_sessions_regime_Pat_majorité"] for g in ("HC", "MCI", "Dementia")}
    tok = pd.read_csv(R / "30_tfidf_token_weights" / "named_annotation_tokens_train.csv")
    sft = tok[tok.task == "SFT"].set_index("token").coef_median
    te = pd.read_csv(R / "14_final_test" / "test_binary.csv").set_index("modèle").loc["B_G_nocov"]
    abl = pd.read_csv(R / "35_ablation_by_category" / "ablation_by_category_train.csv")
    abl = abl[abl.version.isin(["sans timed", "sans procedure"])].groupby("version").perte_vs_ref.max()
    GW, GH_ = 130, 52
    fig = plt.figure(figsize=(GW * MM, GH_ * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, GW)
    ax.set_ylim(0, GH_)
    ax.set_axis_off()
    txt(ax, GW / 2, 48.6, "Do speech benchmarks measure cognition, or how the speech was transcribed?", 7.6, INK,
        "bold", "center")
    cards = [(2, OR, OR_BG, "1", "Two ways to transcribe"), (45.5, OR, OR_BG, "2", "Text models learn them"),
             (89, BL, BL_BG, "3", "Annotation-free control")]
    for x0, ec, fc, k, title in cards:
        card(ax, x0, 8.5, 39, 36, fc, ec, lw=.9, ls=(0, (4, 2)), r=2)
        ax.add_patch(Circle((x0 + 4, 40.6), 2.2, fc=ec, ec="none", zorder=5))
        txt(ax, x0 + 4, 40.6, k, 7, "white", "bold", "center")
        txt(ax, x0 + 7.5, 40.6, title, 7, INK, "bold")
    arrow(ax, 41.3, 26.5, 45.2, 26.5, OR, 1.6, ms=9)
    arrow(ax, 84.8, 26.5, 88.7, 26.5, OR, 1.6, ms=9)

    # 1 : deux conventions, inégales entre groupes
    doc(ax, 6, 25, 10, 12, INK2, tag="3 s", tag_rows=(1, 3), n=5)
    doc(ax, 25, 25, 10, 12, INK2, n=5)
    txt(ax, 11, 22.8, "pauses written", 6, OR, "bold", "center")
    txt(ax, 30, 22.8, "no pauses", 6, INK2, "bold", "center")
    txt(ax, 4.5, 19.4, "sessions without annotations", 6, INK2)
    for k, (g, c) in enumerate([("HC", GH), ("MCI", GM), ("Dementia", GR)]):
        yy = 16.6 - k * 2.6
        txt(ax, 4.5, yy, g if g != "Dementia" else "Dem.", 6, INK2)
        ax.add_patch(Rectangle((12, yy - .9), no_ann[g] / 100 * 22, 1.8, fc=c, ec="none", zorder=5))
        txt(ax, 12.6 + no_ann[g] / 100 * 22, yy, f"{no_ann[g]:.0f}%", 6, INK, "bold")

    # 2 : la sonde transparente lit les annotations
    z0 = 66.5
    txt(ax, z0 - .6, 36.2, "← healthy", 6, BL, ha="right")
    txt(ax, z0 + .6, 36.2, "impaired →", 6, OR)
    for k, (lab, key) in enumerate([("seconds", "seconds"), ("second", "second"), ("buzzer", "buzzer")]):
        v = sft[key] / sft["seconds"]
        yy = 33 - k * 3.1
        ax.add_patch(Rectangle((z0 if v > 0 else z0 + v * 13, yy - 1.05), abs(v) * 13, 2.1, fc=OR, ec="none", zorder=5))
        txt(ax, 49, yy, f'"{lab}"', 6, OR, "bold")
    ax.plot([z0, z0], [25.5, 35], color=INK2, lw=.6, zorder=6)
    txt(ax, 65, 22, "removing annotations", 6, INK2, ha="center")
    txt(ax, 65, 18.4, "AUC −0.07 to −0.12", 9, OR, "bold", "center")
    txt(ax, 65, 14.6, f"timed pauses −{abl['sans timed']:.2f} · timer −{abl['sans procedure']:.2f}", 6, INK, ha="center")
    txt(ax, 65, 11.4, "only where annotations exist", 6, INK2, ha="center", style="italic")

    # 3 : contrôle sans annotation humaine
    card(ax, 92, 25.5, 13, 9, "white", "#c3bfe6", lw=.6, r=1.2, z=3)
    wave(ax, 93, 27, 11, 6, BL, seed=5, silences=[(.3, .45), (.7, .8)])
    txt(ax, 98.5, 23.2, "audio only", 6, INK2, ha="center")
    arrow(ax, 105.5, 30, 110.2, 30, BL, 1.1, ms=6)
    gauge(ax, 118.5, 28.2, 5.5)
    txt(ax, 118.5, 23.2, "risk score", 6, INK2, ha="center")
    txt(ax, 108.5, 37.2, "ASR + VAD → healthy norms", 6, INK2, ha="center")
    txt(ax, 108.5, 17.2, f"{te.AUC:.2f}", 15, BL, "bold", "center")
    txt(ax, 108.5, 12.6, f"test AUC [{te.AUC_lo:.2f}, {te.AUC_hi:.2f}]", 6.3, INK, "bold", "center")
    txt(ax, 108.5, 10.2, "same under both conventions", 6, INK2, ha="center", style="italic")

    # recommandation
    card(ax, 2, 1.2, 126, 5.6, GY_BG, MUTED, lw=.6, r=1.4)
    txt(ax, GW / 2, 4, "Measure pauses from audio  ·  document transcription conventions  ·  report annotation-free "
        "controls", 6.4, INK, "bold", "center")
    for ext in ("png", "pdf"):
        fig.savefig(f26.OUT / f"graphical_abstract_elsevier.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    print("graphical abstract ok")


if __name__ == "__main__":
    main()
    graphical_abstract()
