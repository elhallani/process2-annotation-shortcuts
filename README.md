# Annotation shortcuts in speech-based cognitive impairment detection

Code and aggregated results for the study *Annotation Shortcuts in Speech-Based Cognitive Impairment Detection:
An Audit of PROCESS-2 and an Interpretable Annotation-Free Baseline* (El Hallani, Chakhtouna and Adib).

The study audits the PROCESS-2 corpus (400 participants; semantic fluency, phonemic fluency and Cookie Theft
picture description). It shows that timed pause annotations follow two transcription conventions whose use depends
on diagnosis. Supervised text classifiers exploit this, mainly through timed pauses and a timer marker. An
interpretable model that never reads a human annotation reaches a test AUC of 0.81.

## Data

**No data are included in this repository.** PROCESS-2 is available to researchers under a data use agreement from
its authors and cannot be redistributed. Place the corpus in `data/PROCESS-2/` (never versioned). All loading goes
through `src/data.py::load_dataset()`. Participant-level intermediate files are written to `results/cache/`, which
is also excluded. Only aggregated outputs (counts, means, metrics, figures) are versioned in `results/`.

## Repository layout

```
configs/    frozen, dated analysis configurations (YAML)
src/        reusable modules: data loading, normative models, statistics, transcript cleaning
scripts/    one script per analysis step, numbered in execution order (00–35)
results/    aggregated results and figures (results/figures_v2/)
```

## Main steps

| Scripts | Content |
|---|---|
| 00–03 | Inventory, audit of explicit confounds, recording channel, second voices |
| 04, 06, 08, 17 | Transcription conventions, pause annotations, text ablation (raw, cleaned, Whisper) |
| 05, 07, 09–11 | Acoustic features, Whisper ASR, healthy-control norms, annotation-free model |
| 14 | Official test set, evaluated once (refuses to run twice; SHA-256 provenance) |
| 24, 25, 28 | Frozen speech embeddings and the non-speech control (silence isolated before encoding) |
| 27, 33 | Zero-shot local language models; blind validation of the annotation taxonomy |
| 29, 30, 32, 34, 35 | Transcription-regime checks, probe token weights, marker inventory, taxonomy, ablation by category |
| 26, 31 | Figures |

## Environment

Python 3.11 (conda environment `disco`), PyTorch with CUDA, scikit-learn, statsmodels, transformers, openSMILE,
Silero VAD, Whisper, pyannote. All computations ran on a single 4-GB GPU; local language models use 4-bit
quantisation. No commercial API was used to process the data.

## Protocol

All choices were made by 5 × 5 cross-validation on the 320 training participants, with norms and normalisations
refitted inside each fold. Configurations were frozen before each analysis, and the official test set (80
participants) was evaluated once.
