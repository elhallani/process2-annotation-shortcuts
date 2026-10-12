"""Étape 0g — Transcriptions automatiques Whisper (contrôle sans annotation de transcripteur).

openai/whisper-medium (comme la baseline officielle), anglais, fp16 sur GPU, décodage long-form natif.
Les transcriptions sont des données dérivées des participants : écrites dans data/derived/asr_whisper/
(non versionné, jamais affiché). Reprise possible : les fichiers déjà présents sont sautés.
Sortie agrégée : results/07_asr_whisper/ (WER vs transcription manuelle nettoyée, par tâche × diagnostic, train).
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_dataset, recordings  # noqa: E402
from src.text import clean_for_model, read_text  # noqa: E402

ASR_DIR = ROOT / "data" / "derived" / "asr_whisper"
OUT = ROOT / "results" / "07_asr_whisper"
CONFIG = {"model": "openai/whisper-medium", "language": "en", "task": "transcribe", "dtype": "float16",
          "decoding": "generate long-form séquentiel, return_timestamps=True, condition_on_prev_tokens=False, batch 1", "output_dir": "data/derived/asr_whisper/{participant_id}__{task}.txt",
          "wer_reference": "transcription manuelle passée par src.text.clean_for_model, minuscules, sans ponctuation"}


def asr_path(pid, task):
    return ASR_DIR / f"{pid}__{task}.txt"


def transcribe(recs):
    # Appel direct de generate (long-form séquentiel natif de Whisper) : la pipeline transformers passe par
    # torchcodec, dont la bibliothèque CUDA ne se charge pas dans cet environnement.
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    todo = [r for r in recs.itertuples() if not asr_path(r.participant_id, r.task).exists()]
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
        if inp.input_features.shape[-1] < 3000:  # < 30 s : format court standard
            inp = proc(y, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
        with torch.no_grad():
            ids = model.generate(input_features=inp.input_features.to(dev, torch.float16),
                                 attention_mask=inp.attention_mask.to(dev), language=CONFIG["language"],
                                 task=CONFIG["task"], return_timestamps=True, condition_on_prev_tokens=False)
        text = proc.batch_decode(ids, skip_special_tokens=True)[0]
        asr_path(r.participant_id, r.task).write_text(text.strip() + "\n", encoding="utf-8")
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            el = time.time() - t0
            print(f"  whisper : {i + 1}/{len(todo)} ({el / (i + 1):.1f} s/enr., reste ≈ "
                  f"{el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min)", flush=True)


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", s.lower())).strip().split()


def wer(ref, hyp):
    d = np.arange(len(hyp) + 1)
    for i, rw in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[-1], len(ref)


def main():
    ASR_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_dataset()
    recs = recordings(meta).merge(meta[["participant_id", "txt_SFT", "txt_PFT", "txt_CTD"]], on="participant_id")
    transcribe(recs)
    rows = []
    for r in recs[recs.Split == "Train"].itertuples():
        ref = norm(clean_for_model(read_text(getattr(r, f"txt_{r.task}"))))
        hyp = norm(asr_path(r.participant_id, r.task).read_text(encoding="utf-8"))
        e, n = wer(ref, hyp)
        rows.append({"task": r.task, "diagnosis": r.diagnosis, "errors": e, "ref_words": n,
                     "hyp_words": len(hyp), "empty_hyp": len(hyp) == 0})
    df = pd.DataFrame(rows)
    g = df.groupby(["task", "diagnosis"])
    tab = pd.DataFrame({"n": g.size(), "WER_pooled": g.errors.sum() / g.ref_words.sum(),
                        "median_ref_words": g.ref_words.median(), "median_hyp_words": g.hyp_words.median(),
                        "n_empty_hyp": g.empty_hyp.sum()}).round(3)
    tab.to_csv(OUT / "wer_by_task_diagnosis_train.csv")
    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False))
    (OUT / "summary.json").write_text(json.dumps(
        {"generated": datetime.now().isoformat(timespec="seconds"),
         "n_asr_files": len(list(ASR_DIR.glob("*.txt"))),
         "WER_pooled_train": round(float(df.errors.sum() / df.ref_words.sum()), 3)}, indent=2))
    print(tab)


if __name__ == "__main__":
    main()
