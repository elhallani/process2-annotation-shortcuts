"""Zero-shot local language models (configs/llm.yaml, frozen before computation). Training set, no API.

Usage: python scripts/27_llm_zero_shot.py --model qwen3_4b      (computes and caches the scores)
       python scripts/27_llm_zero_shot.py --analyze             (aggregates all available models)

Score per participant = mean over six prompt variants (three paraphrases x two option orders) of
log P("impaired") - log P("healthy"), read on the first answer token (letters A/B). No text is displayed or
written to results/."""

import argparse
import faulthandler
import importlib.util
import signal
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import TASKS, load_dataset  # noqa: E402
from src.stats import holm  # noqa: E402
from src.text import clean_for_model, raw, read_text  # noqa: E402

faulthandler.register(signal.SIGUSR1)  # « kill -USR1 <pid> » affiche la pile en cas de blocage
CFG = yaml.safe_load((ROOT / "configs" / "llm.yaml").read_text())
OUT, CACHE = ROOT / "results" / "27_llm", ROOT / "results" / "cache" / "27_llm"
VAD = json.loads((ROOT / "results" / "cache" / "01_audit" / "vad_segments.json").read_text())


def _m11():
    spec = importlib.util.spec_from_file_location("m11", ROOT / "scripts" / "11_clinical_scores.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pause_label(g):
    n = max(1, int(round(g)))
    return f"({n} second{'s' if n > 1 else ''})"


def build_texts(meta, m11):
    """Textes des 5 versions par (tâche) ; en mémoire uniquement."""
    rng = np.random.default_rng(CFG["seed"])
    out = {}
    for t in TASKS:
        for v in CFG["versions"]:
            out[(t, v)] = []
        for p in meta.itertuples():
            manual = read_text(getattr(p, f"txt_{t}"))
            out[(t, "m_raw")].append(raw(manual))
            out[(t, "m_clean")].append(clean_for_model(manual))
            segs = json.loads(m11.ts_path(p.participant_id, t).read_text(encoding="utf-8"))
            words = m11.timed_words(segs, np.inf)
            out[(t, "asr")].append(" ".join(w for w, *_ in words))
            vg = VAD[f"{p.participant_id}__{t}"]
            gaps = [(a[1], b[0]) for a, b in zip(vg[:-1], vg[1:])]
            seq, labels = [], []
            for k, (w, tw, _) in enumerate(words):
                seq.append(w)
                if k + 1 < len(words):
                    g = sum(e - s for s, e in gaps if s >= tw and e <= words[k + 1][1])
                    if g >= 1.0:
                        seq.append(pause_label(g))
                        labels.append(pause_label(g))
            out[(t, "asr_pauses")].append(" ".join(seq))
            plain = [w for w, *_ in words]
            pos = sorted(rng.choice(len(plain) + 1, size=min(len(labels), len(plain) + 1), replace=False)) if plain else []
            lab = list(rng.permutation(labels))
            rnd, j = [], 0
            for k in range(len(plain) + 1):
                while j < len(pos) and pos[j] == k:
                    rnd.append(lab[j])
                    j += 1
                if k < len(plain):
                    rnd.append(plain[k])
            out[(t, "asr_random_pauses")].append(" ".join(rnd))
    return out


def load_model(mid):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    q = CFG["quantization"]
    bnb = BitsAndBytesConfig(load_in_4bit=q["load_in_4bit"], bnb_4bit_quant_type=q["quant_type"],
                             bnb_4bit_use_double_quant=q["double_quant"], bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(mid)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device_map = {"": 0}
    if mid in CFG.get("cpu_embeddings", []):
        # embeddings (non quantifiés, liés à lm_head) sur CPU pour libérer la mémoire GPU ; le reste sur GPU
        bnb.llm_int8_enable_fp32_cpu_offload = True
        from transformers import AutoConfig
        if "ForConditionalGeneration" in str(AutoConfig.from_pretrained(mid).architectures):  # Gemma 3 multimodal
            p = "model.language_model"
            device_map = {"model.vision_tower": "cpu", "model.multi_modal_projector": "cpu", f"{p}.embed_tokens": "cpu",
                          f"{p}.layers": 0, f"{p}.norm": 0, f"{p}.rotary_emb": 0, "lm_head": "cpu"}
        else:
            device_map = {"model.embed_tokens": "cpu", "model.layers": 0, "model.norm": 0, "model.rotary_emb": 0,
                          "lm_head": "cpu"}
    try:
        model = AutoModelForCausalLM.from_pretrained(mid, quantization_config=bnb, device_map=device_map)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(mid, quantization_config=bnb, device_map={"": 0})
    return tok, model.eval()


def letter_ids(tok, letter):
    ids = {tok.encode(letter, add_special_tokens=False)[0], tok.encode(" " + letter, add_special_tokens=False)[-1]}
    return sorted(ids)


def forward_last(tok, model, chunk):
    """Logits de la dernière position ; en cas de manque de mémoire GPU, traite les textes un par un
    (résultat identique : chaque texte est indépendant, padding à gauche)."""
    import torch

    def run(texts):
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=CFG["max_input_tokens"], add_special_tokens=False).to("cuda")
        with torch.no_grad():
            return model(**enc, logits_to_keep=1).logits[:, -1].float()  # dernière position seulement
    try:
        return run(chunk)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return torch.cat([run([c]) for c in chunk])


def score_model(short, mid, texts, n):
    import torch
    tok, model = load_model(mid)
    A, B = letter_ids(tok, "A"), letter_ids(tok, "B")
    opts = CFG["options"]
    variants = [(p, o) for p in CFG["prompts"] for o in CFG["option_orders"]]
    t0 = time.time()
    for t in TASKS:
        for v in CFG["versions"]:
            path = CACHE / f"{short}__{t}__{v}.npy"
            if path.exists():
                continue
            S = np.zeros((len(variants), n))
            for vi, (prompt, order) in enumerate(variants):
                msgs = [[{"role": "user", "content": prompt.format(task=CFG["tasks"][t], text=x,
                                                                   opt_a=opts[order[0]], opt_b=opts[order[1]])}]
                        for x in texts[(t, v)]]
                for i in range(0, n, CFG["batch_size"]):
                    chunk = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                             for m in msgs[i:i + CFG["batch_size"]]]
                    lg = forward_last(tok, model, chunk)
                    lp = torch.log_softmax(lg, -1)
                    la, lb = torch.logsumexp(lp[:, A], -1), torch.logsumexp(lp[:, B], -1)
                    # score = log P(atteint) − log P(sain), quel que soit l'ordre des options
                    s = (lb - la) if order[1] == "impaired" else (la - lb)
                    S[vi, i:i + len(chunk)] = s.cpu().numpy()
                    if (i // CFG["batch_size"]) % 20 == 0:
                        print(f"    batch {t} {v} variante {vi} {i}/{n}", flush=True)
            np.save(path, S)
            print(f"  {short} {t} {v} fait ({(time.time() - t0) / 60:.0f} min)", flush=True)
    del model
    torch.cuda.empty_cache()


def paired(y, s1, s2, rng, n_boot=2000, n_perm=1000):
    obs = roc_auc_score(y, s1) - roc_auc_score(y, s2)
    b = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            b.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    null = []
    z1, z2 = (s1 - s1.mean()) / s1.std(), (s2 - s2.mean()) / s2.std()  # échelles comparables avant échange
    for _ in range(n_perm):
        sw = rng.random(len(y)) < .5
        null.append(roc_auc_score(y, np.where(sw, z2, z1)) - roc_auc_score(y, np.where(sw, z1, z2)))
    return obs, np.percentile(b, 2.5), np.percentile(b, 97.5), (1 + (np.abs(null) >= abs(obs)).sum()) / (n_perm + 1)


def analyze(meta):
    rng = np.random.default_rng(CFG["seed"])
    y = (meta.diagnosis != "HC").astype(int).to_numpy()
    conv = pd.read_csv(ROOT / "results" / "cache" / "03_second_voice" / "transcript_markers.csv",
                       dtype={"participant_id": str}).groupby("participant_id").has_speaker_prefix.mean()
    regime = (meta.participant_id.map(conv) >= .5).to_numpy()
    rows, comps = [], []
    for mdl in CFG["models"]:
        sh = mdl["short"]
        if not all((CACHE / f"{sh}__{t}__{v}.npy").exists() for t in TASKS for v in CFG["versions"]):
            continue
        for t in TASKS:
            S = {v: np.load(CACHE / f"{sh}__{t}__{v}.npy") for v in CFG["versions"]}
            for v, M in S.items():
                per = [roc_auc_score(y, r) for r in M]
                rows.append({"modèle": sh, "task": t, "version": v, "AUC_moyenne_variantes": roc_auc_score(y, M.mean(0)),
                             "AUC_variantes_min": min(per), "AUC_variantes_max": max(per),
                             "AUC_regime_Pat": roc_auc_score(y[regime], M.mean(0)[regime]),
                             "AUC_regime_sans_prefixe": roc_auc_score(y[~regime], M.mean(0)[~regime])})
            for a, b, fam in [("m_raw", "m_clean", "primaire"), ("asr_pauses", "asr", "contrefactuel"),
                              ("asr_pauses", "asr_random_pauses", "position"), ("m_clean", "asr", "manuel vs ASR")]:
                obs, lo, hi, p = paired(y, S[a].mean(0), S[b].mean(0), rng)
                comps.append({"modèle": sh, "task": t, "comparaison": f"{a} − {b}", "famille": fam, "dAUC": obs,
                              "IC_lo": lo, "IC_hi": hi, "p_perm": p,
                              "équivalent_±0.03": bool(lo > -CFG["equivalence_margin"] and hi < CFG["equivalence_margin"])})
                if fam == "primaire":
                    for rn, msk in [("régime « Pat: »", regime), ("sans préfixe", ~regime)]:
                        o2, l2, h2, p2 = paired(y[msk], S[a].mean(0)[msk], S[b].mean(0)[msk], rng)
                        comps.append({"modèle": sh, "task": t, "comparaison": f"{a} − {b} ({rn})", "famille": "régime",
                                      "dAUC": o2, "IC_lo": l2, "IC_hi": h2, "p_perm": p2})
    res, cmp_ = pd.DataFrame(rows), pd.DataFrame(comps)
    if len(cmp_):
        cmp_["p_Holm"] = np.nan
        for (mdl, fam), g in cmp_.groupby(["modèle", "famille"]):
            cmp_.loc[g.index, "p_Holm"] = holm(g.p_perm)
    OUT.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(OUT / "auc_by_model_task_version_train.csv", index=False)
    cmp_.round(4).to_csv(OUT / "paired_comparisons_train.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                                                  "statut": "exploratoire, train, zero-shot, inférence locale",
                                                  "modèles_analysés": sorted(res["modèle"].unique().tolist()) if len(res) else []},
                                                 indent=2, ensure_ascii=False))
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False), "\n\n", cmp_.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = load_dataset()
    meta = meta[meta.Split == CFG["split"]].reset_index(drop=True)
    if args.model:
        mdl = next(m for m in CFG["models"] if m["short"] == args.model)
        texts = build_texts(meta, _m11())
        score_model(mdl["short"], mdl["id"], texts, len(meta))
    if args.analyze:
        analyze(meta)


if __name__ == "__main__":
    main()
