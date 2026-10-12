"""Fig. 1 — Vue d'ensemble de l'étude, en schéma illustré (icônes dessinées, étapes numérotées, panneaux colorés).
Chiffres agrégés uniquement, lus dans results/. Les « transcriptions » dessinées sont des icônes (lignes grises) et un
exemple de FORMAT d'annotation « (3 seconds) » : aucun contenu réel. Coordonnées en millimètres (190 × 150 mm).
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("f26", ROOT / "scripts" / "26_figures_v2.py")
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

def main():
    sess = pd.read_csv(R / "29_regime_checks" / "A_regime_sessions_train.csv").set_index("diagnosis")
    no_ann = {g: 100 - sess.loc[g, "pct_sessions_regime_Pat_majorité"] for g in ("HC", "MCI", "Dementia")}
    tok = pd.read_csv(R / "30_tfidf_token_weights" / "named_annotation_tokens_train.csv")
    share = pd.read_csv(R / "30_tfidf_token_weights" / "annotation_weight_share_train.csv")
    te = pd.read_csv(R / "14_final_test" / "test_binary.csv").set_index("modèle").loc["B_G_nocov"]
    first = int((tok[tok.token == "seconds"].rang_median == 1).sum())

    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    # ---- corpus (colonne gauche)
    card(ax, 2, 22, 38, 126, GY_BG, MUTED, lw=.7)
    txt(ax, 21, 143, "PROCESS-2 corpus", 7.5, INK, "bold", "center")
    db(ax, 15, 126, 12, 12, BL)
    txt(ax, 21, 121, "400 participants", 6.8, INK, "bold", "center")
    for k, (g, n, c) in enumerate([("HC", 200, GH), ("MCI", 150, GM), ("Dementia", 50, GR)]):
        yy = 114 - k * 6
        person(ax, 8, yy - 1.6, 4.2, c)
        txt(ax, 12, yy, f"{g}", 6.3, INK)
        txt(ax, 36, yy, f"{n}", 6.3, INK, "bold", "right")
    ax.plot([5, 37], [97.5, 97.5], color=GRID, lw=.6)
    txt(ax, 21, 93.5, "3 tasks, one session", 6.5, INK, "bold", "center")
    for k, (t, d) in enumerate([("SFT", "semantic fluency\n(animals, 60 s)"), ("PFT", "phonemic fluency\n(letter P, 60 s)"),
                                ("CTD", "picture description\n(Cookie Theft)")]):
        yy = 85 - k * 10
        card(ax, 5, yy - 3, 9, 6, "white", INK2, lw=.6, r=1)
        txt(ax, 9.5, yy, t, 6, INK, "bold", "center")
        txt(ax, 16, yy, d, 5.6, INK2)
    ax.plot([5, 37], [60.5, 60.5], color=GRID, lw=.6)
    card(ax, 5, 47, 32, 11, "white", INK2, lw=.6, r=1.2)
    wave(ax, 7, 49, 28, 7, INK2, silences=[(.3, .42), (.7, .8)])
    txt(ax, 21, 44, "audio (16 kHz)", 6, INK2, ha="center")
    doc(ax, 9.5, 28, 9, 11, INK2, tag="3 s", tag_rows=(2,), n=4)
    doc(ax, 23.5, 28, 9, 11, INK2, n=4)
    txt(ax, 21, 25, "manual transcripts", 6, INK2, ha="center")
    # split train / test
    card(ax, 2, 6, 38, 12, "white", MUTED, lw=.7)
    ax.add_patch(Rectangle((5, 8.5), 25.6, 4, fc=INK2, ec="none", zorder=5))
    ax.add_patch(Rectangle((30.6, 8.5), 6.4, 4, fc=OR, ec="none", zorder=5))
    txt(ax, 17.8, 10.5, "train 320", 5.8, "white", "bold", "center")
    txt(ax, 33.8, 10.5, "test 80", 5.0, "white", "bold", "center")
    txt(ax, 21, 15.3, "official split", 6, INK2, ha="center")

    # ---- A : audit (haut)
    card(ax, 45, 80, 143, 68, OR_BG, OR, lw=.9, ls=(0, (4, 2)))
    txt(ax, 49, 143, "A", 9, OR, "bold")
    txt(ax, 54, 143, "Shortcut audit: how much diagnostic information do transcription conventions carry?", 7.5, INK,
        "bold")
    cols = [48, 84, 120, 156]
    titles = ["Two transcription\nregimes", "Transparent probe\n(TF-IDF + LR)", "Controlled\nablation",
              "Other shortcut\nchecks"]
    for k, (x0, ti) in enumerate(zip(cols, titles)):
        card(ax, x0, 84, 31 if k < 3 else 29, 52, "white", "#e3b3cd", lw=.6, r=1.8)
        num(ax, x0 + 4, 131, k + 1, OR)
        txt(ax, x0 + 8, 131, ti, 6.4, INK, "bold")
        if k < 3:
            arrow(ax, x0 + 31.3, 110, x0 + 35.7, 110, OR, 1.2)
    # 1 : régimes
    doc(ax, 51, 107, 11, 15, INK2, tag="3 s", tag_rows=(1, 3), n=5)
    doc(ax, 66, 107, 11, 15, INK2, n=5)
    txt(ax, 56.5, 104.5, "annotated", 5.6, OR, "bold", "center")
    txt(ax, 71.5, 104.5, "not annotated", 5.6, INK2, "bold", "center")
    txt(ax, 50, 99.5, "sessions without annotations", 5.4, INK2)
    for k, (g, c) in enumerate([("HC", GH), ("MCI", GM), ("Dementia", GR)]):
        yy = 95.5 - k * 3.8
        txt(ax, 50, yy, g if g != "Dementia" else "Dem.", 5.4, INK2)
        ax.add_patch(Rectangle((58, yy - 1.2), no_ann[g] / 100 * 16, 2.4, fc=c, ec="none", zorder=5))
        txt(ax, 58.8 + no_ann[g] / 100 * 16, yy, f"{no_ann[g]:.0f}%", 5.4, INK, "bold")
    # 2 : sonde (poids signés, lus dans 30_tfidf_token_weights, SFT)
    txt(ax, 86, 122, "each word = one visible weight", 5.5, INK2, style="italic")
    sft = tok[tok.task == "SFT"].set_index("token").coef_median
    ref = sft["seconds"]
    z0 = 103.5
    txt(ax, z0 - .8, 118, "← healthy", 6, BL, ha="right")
    txt(ax, z0 + .8, 118, "impaired →", 6, OR)
    bars = [("seconds", sft["seconds"] / ref, OR), ("second", sft["second"] / ref, OR),
            ("buzzer", sft["buzzer"] / ref, OR), ("other words", .45, MUTED), ("", -.35, MUTED), ("", .3, MUTED)]
    for k, (lab, v, c) in enumerate(bars):
        yy = 113.5 - k * 3.7
        ax.add_patch(Rectangle((z0 if v > 0 else z0 + v * 10, yy - 1.2), abs(v) * 10, 2.4, fc=c, ec="none",
                               zorder=5, alpha=1 if c == OR else .8))
        if lab:
            txt(ax, 86, yy, lab, 6, OR if c == OR else MUTED, "bold" if c == OR else "normal")
    ax.plot([z0, z0], [91.5, 116], color=INK2, lw=.6, zorder=6)
    txt(ax, 99.5, 88.2, f'"seconds" = #1 in {first}/3 tasks\n× 25/25 folds; "buzzer" → HC', 6, OR, "bold",
        "center")
    # 3 : ablation contrôlée — par régime (haut) et par catégorie (bas), SFT
    ab = pd.read_csv(R / "17_corrections" / "AB_comparisons_nadeau_bengio_train.csv")
    ab = ab[ab.task == "SFT"].set_index("régime")
    txt(ax, 122, 124.5, "by regime (SFT AUC:", 6, INK2, style="italic")
    ax.add_patch(Circle((143.2, 124.5), .8, fc=OR, ec="none", zorder=6))
    txt(ax, 144.3, 124.5, "→", 6, INK2)
    ax.add_patch(Circle((147.6, 124.5), .8, fc=INK2, ec="none", zorder=6))
    txt(ax, 148.6, 124.5, ")", 6, INK2, style="italic")
    rows = [("all sessions", "tous", INK), ("annotated", "régime « Pat: »", OR),
            ("not annotated", "régime sans préfixe", MUTED)]
    for k, (lab, key, col) in enumerate(rows):
        a, b = ab.loc[key, "AUC_1_fold_mean"], ab.loc[key, "AUC_2_fold_mean"]
        d = b - a
        y0 = 120.5 - k * 6.3
        txt(ax, 122, y0, f"{lab}: {'−' if d < 0 else '+'}{abs(d):.2f}", 6, col, "bold")
        xa, xb = 124 + (a - .55) * 70, 124 + (b - .55) * 70
        ax.plot([124, 147], [y0 - 2.6, y0 - 2.6], color=GRID, lw=.5, zorder=4)
        ax.add_patch(Circle((xa, y0 - 2.6), .85, fc=OR, ec="none", zorder=6))
        if abs(xa - xb) > 1.5:
            arrow(ax, xa - .9, y0 - 2.6, xb + .2, y0 - 2.6, col, 1.2, ms=5)
        ax.add_patch(Circle((xb, y0 - 2.6), .85, fc=INK2, ec="none", zorder=6))
    ax.plot([122, 149], [102.2, 102.2], color=GRID, lw=.6)
    abl = pd.read_csv(R / "35_ablation_by_category" / "ablation_by_category_train.csv")
    abl = abl[abl.task == "SFT"].set_index("version")
    txt(ax, 122, 99.8, "by category (SFT AUC loss)", 6, INK2, style="italic")
    for k, (lab, v, c) in enumerate([("timed pauses", "sans timed", OR), ("timer marker", "sans procedure", OR),
                                     ("non-verbal", "sans nonverbal", MUTED), ("uncertain (?)", "sans uncertain", MUTED)]):
        yy = 96 - k * 3.3
        loss = max(abl.loc[v, "perte_vs_ref"], 0)
        txt(ax, 122, yy, lab, 6, c if c == OR else INK2, "bold" if c == OR else "normal")
        ax.add_patch(Rectangle((137, yy - 1.1), max(loss * 120, .3), 2.2, fc=c, ec="none", zorder=5))
        txt(ax, 137.8 + loss * 120, yy, f"{loss:.2f}", 6, c if c == OR else INK2, "bold" if c == OR else "normal")
    # 4 : autres contrôles
    checks = [("cleaned text → regime", "AUC 0.80–0.96 (style)", OR),
              ("annotation taxonomy", "LLM-validated, κ 0.97", INK2),
              ("demographics, channel", "AUC ≤ 0.56", INK2), ("second voices", "κ ≤ 0.47", INK2),
              ("zero-shot LLMs", "no prior shortcut", INK2), ("isolated silence", "= amount of pausing", INK2)]
    for k, (a, b, c) in enumerate(checks):
        yy = 122.5 - k * 6.5
        ax.add_patch(Circle((159.5, yy), 1.1, fc=c, ec="none", zorder=5))
        txt(ax, 162, yy + 1.2, a, 5.4, INK)
        txt(ax, 162, yy - 1.5, b, 5.4, c, "bold")

    # ---- B : baseline sans annotation (milieu)
    card(ax, 45, 28, 143, 48, BL_BG, BL, lw=.9, ls=(0, (4, 2)))
    txt(ax, 49, 71, "B", 9, BL, "bold")
    txt(ax, 54, 71, "Annotation-free interpretable control: speech → automatic measures → healthy norms → risk",
        7.5, INK, "bold")
    card(ax, 49, 33, 22, 33, "white", "#c3bfe6", lw=.6, r=1.6)
    wave(ax, 51, 50, 18, 9, BL, seed=5, silences=[(.25, .4), (.65, .78)])
    txt(ax, 60, 44, "audio only", 6, INK, "bold", "center")
    txt(ax, 60, 38.5, "no human\nannotation", 5.4, BL, "bold", "center")
    for k, (t, d, yy) in enumerate([("Whisper ASR", "word timestamps", 56), ("Silero VAD", "speech / pause", 40)]):
        card(ax, 76, yy - 5, 25, 11, "white", "#c3bfe6", lw=.6, r=1.4)
        txt(ax, 88.5, yy + 2, t, 6.2, INK, "bold", "center")
        txt(ax, 88.5, yy - 2, d, 5.5, INK2, ha="center")
        arrow(ax, 71.3, 49.5, 75.7, yy, BL, 1.0, rad=0)
    for k, (t, d, yy) in enumerate([("Fluency", "distinct valid\nwords per second", 56),
                                    ("Pauses", "median gap, long\nsilences, CTD rate", 40)]):
        card(ax, 106, yy - 5, 25, 11, "white", "#c3bfe6", lw=.6, r=1.4)
        txt(ax, 118.5, yy + 2.5, t, 6.2, INK, "bold", "center")
        txt(ax, 118.5, yy - 2, d, 5.2, INK2, ha="center")
        arrow(ax, 101.3, yy, 105.7, yy, BL, 1.0)
        arrow(ax, 131.3, yy, 136.7, 48, BL, 1.0)
    card(ax, 137, 33, 22, 33, "white", "#c3bfe6", lw=.6, r=1.6)
    txt(ax, 148, 61.5, "Healthy-control\nnorms", 6.2, INK, "bold", "center")
    bell(ax, 140, 44, 16, 9, BL)
    txt(ax, 148, 38.5, "age, sex\n(Huber, in fold)", 5.3, INK2, ha="center")
    arrow(ax, 159.3, 49.5, 163.2, 49.5, BL, 1.0)
    card(ax, 163.5, 33, 22, 33, "white", "#c3bfe6", lw=.6, r=1.6)
    txt(ax, 174.5, 61.5, "Logistic\nregression", 6.2, INK, "bold", "center")
    gauge(ax, 174.5, 46, 7)
    txt(ax, 174.5, 38.5, "6 weights,\nrisk score", 5.3, INK2, ha="center")

    # ---- C : protocole (bas)
    card(ax, 45, 2, 143, 22, GY_BG, MUTED, lw=.9, ls=(0, (4, 2)))
    txt(ax, 49, 19.5, "C", 9, INK2, "bold")
    txt(ax, 54, 19.5, "Evaluation protocol", 7.5, INK, "bold")
    folds(ax, 54, 4.5, 10, 10, INK2)
    txt(ax, 66, 10.5, "5 × 5 CV on train;\nnorms refitted per fold", 5.6, INK2)
    lock(ax, 95, 6.5, 6, INK2)
    txt(ax, 103, 10.5, "frozen, dated\nconfigurations", 5.6, INK2)
    target(ax, 128, 10, 4, OR)
    txt(ax, 134, 10.5, "official test\nused once", 5.6, INK2)
    card(ax, 152, 5, 33, 13, "white", BL, lw=1, r=1.6)
    txt(ax, 168.5, 14.5, "test AUC (HC vs impaired)", 5.4, INK2, ha="center")
    txt(ax, 168.5, 9, f"{te.AUC:.2f} [{te.AUC_lo:.2f}, {te.AUC_hi:.2f}]", 8, BL, "bold", "center")

    # ---- flèches du corpus vers A, B, C
    arrow(ax, 40.3, 110, 44.7, 110, INK2, 1.3)
    arrow(ax, 40.3, 50, 44.7, 50, INK2, 1.3)
    arrow(ax, 40.3, 12, 44.7, 12, INK2, 1.3)

    for ext in ("png", "pdf"):
        fig.savefig(f26.OUT / f"fig1_system_overview.{ext}", dpi=400 if ext == "png" else None)
    plt.close(fig)
    print("fig1 ok", json.dumps({k: round(v, 1) for k, v in no_ann.items()}))


if __name__ == "__main__":
    main()
