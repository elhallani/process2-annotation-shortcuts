"""Validation de la taxonomie des annotations (32_marker_inventory) par 3 LLM locaux, à l'aveugle.

Chaque forme d'annotation distincte (normalisée : minuscules, chiffres → N ; présentée avec N = 3) est classée par
chaque LLM dans 6 catégories, sans voir la règle. Score = log-probabilité de chaque lettre de réponse, moyennée sur
deux ordres des options (biais de position). Données strictement locales (DUA) : modèles open-weights, aucune API.
Sorties AGRÉGÉES : formes nommées (≥ 5 fichiers) avec règle et choix des LLM ; formes rares (< 5 fichiers) : seulement
des tableaux croisés de comptes (jamais la forme) ; accord règle–LLM et LLM–LLM (kappa de Cohen).
"""

import gc
import importlib.util
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import TASKS, load_dataset  # noqa: E402
from src.text import read_text  # noqa: E402


def _load(name, path, argv=None):
    saved, sys.argv = sys.argv, [path] + (argv or [])
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = saved
    return mod


m32 = _load("m32", "scripts/32_marker_inventory.py")
m27 = _load("m27", "scripts/27_llm_zero_shot.py")
OUT = ROOT / "results" / "33_llm_validate_taxonomy"
CACHE = ROOT / "results" / "cache" / "33_llm_validate_taxonomy"  # étiquettes par forme, dans l'ordre de forms()
CATS = [("timed", "a timed pause (a silence with its duration)"),
        ("untimed", "an untimed pause or silence"),
        ("nonverbal", "a non-verbal vocal event (e.g. sigh, laugh, cough, tongue click)"),
        ("uncertain", "uncertain or inaudible speech"),
        ("procedure", "a sound from the recording procedure or environment (device, timer, noise)"),
        ("word", "a spoken word or other speech content")]
RULE2CAT = {"pause chronométrée": "timed", "pause non chronométrée": "untimed", "événement non verbal": "nonverbal",
            "incertitude / inaudible": "uncertain", "bruit / environnement": "procedure", "autre": "word"}
LETTERS = "ABCDEF"
PROMPT = ("In a manual transcript of a speech recording, a transcriber wrote the following annotation between "
          "brackets:\n\n{form}\n\nWhat does this annotation most likely mark?\n{options}\nAnswer with a single letter.")


def forms():
    meta = load_dataset()
    files = defaultdict(set)
    for p in meta.itertuples():
        for t in TASKS:
            text = read_text(getattr(p, f"txt_{t}"))
            for bname, rx in m32.BRACKETS.items():
                for inner in re.findall(rx, text):
                    files[(bname, m32.norm(inner))].add((p.participant_id, t))
    rows = []
    for (b, f), fs in files.items():
        shown = b[0] + f.replace("N", "3") + b[-1]
        rows.append({"crochets": b, "forme": f, "présentée": shown, "n_fichiers": len(fs), "rare": len(fs) < m32.MIN_FILES,
                     "règle": RULE2CAT[m32.category(f)]})
    return pd.DataFrame(rows)


def classify(short, mid, F):
    import torch
    tok, model = m27.load_model(mid)
    ids = [m27.letter_ids(tok, L) for L in LETTERS]
    scores = np.zeros((len(F), len(CATS)))
    for order in (list(range(6)), list(range(6))[::-1]):
        opts = "\n".join(f"{LETTERS[i]}) {CATS[c][1]}" for i, c in enumerate(order))
        prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(form=s, options=opts)}],
                                           add_generation_prompt=True, tokenize=False) for s in F.présentée]
        for k in range(0, len(prompts), 4):
            lp = torch.log_softmax(m27.forward_last(tok, model, prompts[k:k + 4]), -1).cpu().numpy()
            for i, c in enumerate(order):
                scores[k:k + 4, c] += np.logaddexp.reduce(lp[:, ids[i]], axis=1)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  {short} fait", flush=True)
    return [CATS[j][0] for j in scores.argmax(1)]


def main():
    import subprocess
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    F = forms()
    if len(sys.argv) > 1:  # un seul modèle par processus (mémoire GPU libérée entre les modèles)
        m = next(m for m in m27.CFG["models"] if m["short"] == sys.argv[1])
        (CACHE / f"{m['short']}.json").write_text(json.dumps(classify(m["short"], m["id"], F)))
        return
    shorts = []
    for m in m27.CFG["models"]:
        path = CACHE / f"{m['short']}.json"
        if not path.exists():
            subprocess.run([sys.executable, "-u", __file__, m["short"]], check=True)
        F[m["short"]] = json.loads(path.read_text())
        shorts.append(m["short"])
    F["vote_LLM"] = F[shorts].mode(axis=1)[0]
    cats = [c for c, _ in CATS]
    agree = []
    for subset, d in (("formes nommées (≥ 5 fichiers)", F[~F.rare]), ("formes rares (< 5 fichiers)", F[F.rare]),
                      ("toutes, pondérées par fichiers", F)):
        w = d.n_fichiers.to_numpy() if "pondérées" in subset else None
        for a, b in [("règle", s) for s in shorts + ["vote_LLM"]] + [(shorts[i], shorts[j]) for i in range(3)
                                                                       for j in range(i + 1, 3)]:
            agree.append({"sous-ensemble": subset, "comparaison": f"{a} vs {b}", "n_formes": len(d),
                          "accord_%": 100 * np.average(d[a] == d[b], weights=w),
                          "kappa": cohen_kappa_score(d[a], d[b], labels=cats, sample_weight=w)})
    A = pd.DataFrame(agree).round(3)
    named = F[~F.rare][["crochets", "forme", "n_fichiers", "règle", *shorts, "vote_LLM"]].sort_values(
        "n_fichiers", ascending=False)
    rare_x = pd.crosstab(F[F.rare].règle, F[F.rare].vote_LLM).reindex(index=cats, columns=cats, fill_value=0)
    rare_files = F[F.rare].groupby("vote_LLM").n_fichiers.sum().reindex(cats, fill_value=0)
    named.to_csv(OUT / "named_forms_rule_vs_llm.csv", index=False)
    A.to_csv(OUT / "agreement.csv", index=False)
    rare_x.to_csv(OUT / "rare_forms_crosstab_rule_vs_vote.csv")
    (OUT / "summary.json").write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"), "modèles": shorts, "catégories": dict(CATS),
        "n_formes": len(F), "n_formes_rares": int(F.rare.sum()),
        "fichiers_formes_rares_par_catégorie_LLM": rare_files.astype(int).to_dict(),
        "note": "formes rares jamais écrites ; présentation N = 3 ; 2 ordres d'options"}, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 220)
    print(named.to_string(index=False), "\n\n", A.to_string(index=False), "\n\nformes rares, règle (lignes) × vote LLM:\n",
          rare_x.to_string(), "\n\nfichiers des formes rares par catégorie (vote LLM):\n", rare_files.to_string())


if __name__ == "__main__":
    main()
