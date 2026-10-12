"""Clinical fluency score (distinct valid words) from time-stamped Whisper transcripts.

A. Whisper-medium on SFT and PFT with segment timestamps -> data/derived/asr_whisper_ts/ (not versioned).
B. Score within min(duration, 45 s), word times interpolated within their segment:
   SFT: animal names (WordNet hypernym animal.n.01, bigrams tested first), distinct by lemma;
   PFT: English words starting with "p" (WordNet or the NLTK word list), proper nouns excluded, distinct by stem;
   plus repetitions and intrusions.
C. Validation (training set): the same scoring rules on cleaned manual transcripts versus Whisper; correlation with
   VAD units.
D. Normative G and D indices for the clinical score (src/normative.py).
Only counts and metrics are written to results/."""

import json
import re
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import FLUENCY_TASKS, load_dataset, recordings  # noqa: E402
from src.normative import crossfit_scores  # noqa: E402
from src.stats import icc  # noqa: E402
from src.text import clean_for_model, read_text  # noqa: E402

TS_DIR = ROOT / "data" / "derived" / "asr_whisper_ts"
OUT = ROOT / "results" / "11_clinical"
CACHE = ROOT / "results" / "cache" / "11_clinical"
CONFIG = {
    "seed": 42, "model": "openai/whisper-medium", "language": "en", "window_s": 45.0,
    "word_timing": "interpolation uniforme dans le segment Whisper",
    "sft_rule": "nom dont un synset WordNet a animal.n.01 dans ses hyperonymes ; le mot « animal(s) » exclu ; "
                "bigrammes (ex. polar bear) testés avant les unigrammes ; distinct par lemme",
    "pft_rule": "commence par p ; présent dans WordNet (toute catégorie) ou nltk words ; nom propre exclu "
                "(majuscule hors début de segment) ; distinct par racine Porter",
    "fillers": ["um", "uh", "er", "erm", "ah", "oh", "hmm", "mm", "mhm", "okay", "ok", "yeah", "yes", "no", "well",
                "so", "like", "right", "oops", "sorry"],
}


# ---------------------------------------------------------------- A. Whisper horodaté

def ts_path(pid, task):
    return TS_DIR / f"{pid}__{task}.json"


def transcribe(recs):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    todo = [r for r in recs.itertuples() if not ts_path(r.participant_id, r.task).exists()]
    if not todo:
        return
    dev = torch.device("cuda:0")
    proc = WhisperProcessor.from_pretrained(CONFIG["model"])
    model = WhisperForConditionalGeneration.from_pretrained(CONFIG["model"], dtype=torch.float16).to(dev).eval()
    t0 = time.time()
    for i, r in enumerate(todo):
        y, sr = sf.read(str(r.wav), dtype="float32")
        inp = proc(y, sampling_rate=sr, return_tensors="pt", truncation=False, padding="longest",
                   return_attention_mask=True)
        if inp.input_features.shape[-1] < 3000:
            inp = proc(y, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
        with torch.no_grad():
            out = model.generate(input_features=inp.input_features.to(dev, torch.float16),
                                 attention_mask=inp.attention_mask.to(dev), language=CONFIG["language"],
                                 task="transcribe", return_timestamps=True, condition_on_prev_tokens=False,
                                 return_segments=True)
        segs = [{"start": float(s["start"]), "end": float(s["end"]),
                 "text": proc.decode(s["tokens"], skip_special_tokens=True).strip()} for s in out["segments"][0]]
        ts_path(r.participant_id, r.task).write_text(json.dumps(segs), encoding="utf-8")
        if (i + 1) % 100 == 0 or i + 1 == len(todo):
            el = time.time() - t0
            print(f"  whisper-ts : {i + 1}/{len(todo)} (reste ≈ {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min)",
                  flush=True)


# ---------------------------------------------------------------- B. Score

WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def lexicon():
    import nltk
    for p in ("wordnet", "omw-1.4", "words", "stopwords"):
        nltk.download(p, quiet=True)
    from nltk.corpus import stopwords, wordnet as wn, words
    from nltk.stem import PorterStemmer
    return wn, {w.lower() for w in words.words()}, set(stopwords.words("english")) | set(CONFIG["fillers"]), \
        PorterStemmer()


WN, WORDS, STOP, STEM = lexicon()
ANIMAL = WN.synset("animal.n.01")


@lru_cache(maxsize=None)
def animal_lemma(w):
    """Lemme si w (mot ou bigramme « a_b ») désigne un animal, sinon None."""
    if w in ("animal", "animals"):
        return None
    lemma = WN.morphy(w, WN.NOUN) or w
    for s in WN.synsets(lemma, WN.NOUN):
        if s == ANIMAL or ANIMAL in s.closure(lambda x: x.hypernyms()):
            return lemma
    return None


@lru_cache(maxsize=None)
def is_english(w):
    return w in WORDS or bool(WN.synsets(w)) or bool(WN.morphy(w))


def timed_words(segs, window):
    """(mot original, temps, début_de_segment) pour les mots dont le temps interpolé < window."""
    out = []
    for s in segs:
        ws = WORD.findall(s["text"])
        for k, w in enumerate(ws):
            t = s["start"] + (k + 0.5) / max(len(ws), 1) * (s["end"] - s["start"])
            if t < window:
                out.append((w, t, k == 0))
    return out


def score_sft(words):
    low = [w.lower() for w, *_ in words]
    valid, used = [], set()
    i = 0
    while i < len(low):
        if i + 1 < len(low) and (lem := animal_lemma(f"{low[i]}_{low[i + 1]}")):
            valid.append(lem)
            i += 2
            continue
        if (lem := animal_lemma(low[i])):
            valid.append(lem)
        elif low[i] not in STOP:
            used.add(i)
        i += 1
    return len(set(valid)), len(valid) - len(set(valid)), len(used)


def score_pft(words):
    valid, intr = [], 0
    for w, _, first in words:
        lw = w.lower()
        if lw in STOP:
            continue
        if lw.startswith("p") and is_english(lw) and not (w[0].isupper() and not first):
            valid.append(STEM.stem(lw))
        else:
            intr += 1
    return len(set(valid)), len(valid) - len(set(valid)), intr


def score(task, words):
    return (score_sft if task == "SFT" else score_pft)(words)


def manual_words(text):
    return [(w, 0.0, k == 0) for k, w in enumerate(WORD.findall(clean_for_model(text)))]


# ---------------------------------------------------------------- main

def main():
    TS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = load_dataset()
    recs = recordings(meta)
    recs = recs[recs.task.isin(FLUENCY_TASKS)].merge(meta[["participant_id", "txt_SFT", "txt_PFT"]], on="participant_id")
    transcribe(recs)

    feats = pd.read_csv(ROOT / "results" / "cache" / "05_features" / "features_per_recording.csv",
                        dtype={"participant_id": str}).set_index(["participant_id", "task"])
    rows = []
    for r in recs.itertuples():
        segs = json.loads(ts_path(r.participant_id, r.task).read_text(encoding="utf-8"))
        f = feats.loc[(r.participant_id, r.task)]
        w = min(f.duration_s, CONFIG["window_s"])
        v45, rep45, intr45 = score(r.task, timed_words(segs, w))
        vfull, _, _ = score(r.task, timed_words(segs, np.inf))
        vman, _, _ = score(r.task, manual_words(read_text(getattr(r, f"txt_{r.task}"))))
        rows.append({"participant_id": r.participant_id, "task": r.task, "diagnosis": r.diagnosis, "Split": r.Split,
                     "window_s": w, "valid_45s": v45, "valid_per_s": v45 / w, "repetitions_45s": rep45,
                     "intrusions_45s": intr45, "valid_full_asr": vfull, "valid_full_manual": vman,
                     "vad_units_45s": f.units_per_s * w})
    df = pd.DataFrame(rows)
    df.to_csv(CACHE / "clinical_per_recording.csv", index=False)
    tr = df[df.Split == "Train"]

    # C. validation
    val = []
    for t in FLUENCY_TASKS:
        s = tr[tr.task == t]
        val.append({"task": t, "n": len(s),
                    "spearman_asr_vs_manual_full": spearmanr(s.valid_full_asr, s.valid_full_manual)[0],
                    "ICC21_asr_vs_manual_full": icc(s[["valid_full_asr", "valid_full_manual"]].to_numpy(float))[0],
                    "mean_diff_asr_minus_manual": (s.valid_full_asr - s.valid_full_manual).mean(),
                    "spearman_vad_units_vs_valid_45s": spearmanr(s.vad_units_45s, s.valid_45s)[0]})
    val = pd.DataFrame(val).round(3)
    val.to_csv(OUT / "validation_train.csv", index=False)
    desc = tr.groupby(["task", "diagnosis"])[["valid_45s", "repetitions_45s", "intrusions_45s", "valid_full_asr",
                                               "valid_full_manual"]].median()
    desc.to_csv(OUT / "median_counts_by_task_diagnosis_train.csv")

    # D. indices normatifs pour le score clinique (descriptif, train)
    m = meta[meta.Split == "Train"].set_index("participant_id")
    x = tr.pivot(index="participant_id", columns="task", values="valid_per_s").add_prefix("x_")
    d = x.join(pd.DataFrame({"age": m.age.astype(float), "male": (m.gender == "male").astype(float),
                             "diagnosis": m.diagnosis}))
    is_hc = d.diagnosis == "HC"
    sc, full = crossfit_scores(d, is_hc, 5, CONFIG["seed"])
    sc = sc.astype(float)
    hc = d[is_hc]
    norm = {"r_SFT_PFT_HC": float(np.corrcoef(hc.x_SFT, hc.x_PFT)[0, 1]), **full.coefficients()}
    aucs = []
    for idx in ["G", "D", "D_simple", "z_SFT", "z_PFT"]:
        row = {"index": idx}
        for name, pos in [("HC_vs_MCI+Dem", ["MCI", "Dementia"]), ("HC_vs_Dem", ["Dementia"]), ("HC_vs_MCI", ["MCI"])]:
            k = d.diagnosis.isin(["HC", *pos])
            row[f"AUC_{name}"] = roc_auc_score(d.diagnosis[k].isin(pos), sc[idx][k])
        for g in ["HC", "MCI", "Dementia"]:
            row[f"median_{g}"] = sc[idx][d.diagnosis == g].median()
        aucs.append(row)
    aucs = pd.DataFrame(aucs).round(3)
    aucs.to_csv(OUT / "normative_indices_auc_train.csv", index=False)

    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False))
    (OUT / "summary.json").write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                                                  "n_recordings": len(df), "normative": norm}, indent=2,
                                                 ensure_ascii=False, default=float))
    pd.set_option("display.width", 220)
    print("== Validation ==\n", val, "\n\n== Médianes ==\n", desc, "\n\n== Normes ==\n",
          json.dumps(norm, indent=1, default=float), "\n\n== Indices ==\n", aucs)


if __name__ == "__main__":
    main()
