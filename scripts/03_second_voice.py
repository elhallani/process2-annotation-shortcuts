"""Étape 0d — Seconde voix : conventions des transcriptions, diarisation pyannote, vérification manuelle.

Usage : python scripts/03_second_voice.py --stage {transcripts,diarize,sample,kappa,all}

  transcripts  Comptes agrégés des marques de locuteur / annotations dans les .txt (aucun contenu affiché ni écrit).
  diarize      Diarisation pyannote (speaker-diarization-community-1) des 1200 enregistrements ;
               enregistrement marqué si un locuteur non principal totalise ≥ 3 s ; session marquée si ≥ 1 tâche marquée.
  sample       30 identifiants du TRAIN (15 sessions marquées, 15 non), ordre aléatoire, écrits dans
               results/01_audit/manual_check_ids.txt SANS le statut (écoute en aveugle). Clé dans results/cache/.
  kappa        Accord (kappa de Cohen) entre notes/manual_check.csv (colonnes participant_id, second_voice ∈ {0,1})
               et les marqueurs automatiques (pyannote, ECAPA strict, transcriptions).
"""

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import TASKS, load_dataset, recordings  # noqa: E402

OUT = ROOT / "results" / "03_second_voice"
CACHE = ROOT / "results" / "cache" / "03_second_voice"
IDS_FILE = ROOT / "results" / "01_audit" / "manual_check_ids.txt"
MANUAL_FILE = ROOT / "notes" / "manual_check.csv"

CONFIG = {
    "seed": 42,
    "transcripts": {
        "encoding": "utf-8 (errors=replace)",
        "patterns": {
            "speaker_prefix": r"(?m)^\s*\*?([A-Za-z][A-Za-z0-9 _.\-]{0,15}):",
            "square_brackets": r"\[[^\]\n]{0,80}\]",
            "parentheses": r"\([^)\n]{0,80}\)",
            "angle_tags": r"<[^>\n]{1,40}>",
            "curly_braces": r"\{[^}\n]{0,80}\}",
            "timestamps": r"\b\d{1,2}:\d{2}(?::\d{2})?\b",
        },
        "dialogue_rule": "≥ 2 préfixes de locuteur distincts dans le fichier",
    },
    "diarization": {"model": "pyannote/speaker-diarization-community-1", "output": "exclusive_speaker_diarization",
                    "second_speaker_min_s": 3.0, "grid_s": [1.0, 2.0, 3.0, 5.0, 10.0]},
    "manual_check": {"split": "Train", "n_flagged": 15, "n_unflagged": 15, "flag": "session pyannote ≥ 3 s"},
    "ecapa_strict": {"similarity_threshold": 0.3, "max_sim_to_main": 0.25, "min_s": 3.0},
}


# ---------------------------------------------------------------- transcriptions

def transcript_markers(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    pats = CONFIG["transcripts"]["patterns"]
    r = {f"has_{k}": bool(re.search(p, text)) for k, p in pats.items()}
    prefixes = {m.strip().lower() for m in re.findall(pats["speaker_prefix"], text)}
    r["n_distinct_prefixes"] = len(prefixes)
    r["n_lines"] = sum(1 for l in text.splitlines() if l.strip())
    r["n_tokens"] = len(text.split())
    r["empty"] = r["n_tokens"] == 0
    return r


def stage_transcripts(meta):
    rows = []
    for p in meta.itertuples():
        for t in TASKS:
            rows.append({"participant_id": p.participant_id, "task": t, "diagnosis": p.diagnosis,
                         "Split": p.Split, **transcript_markers(getattr(p, f"txt_{t}"))})
    df = pd.DataFrame(rows)
    df["dialogue"] = df.n_distinct_prefixes >= 2
    df.to_csv(CACHE / "transcript_markers.csv", index=False)
    flags = [c for c in df.columns if c.startswith("has_")] + ["dialogue", "empty"]
    g = df.groupby(["task", "diagnosis"])
    tab = g[flags].mean().mul(100).round(1)
    tab.insert(0, "n_files", g.size())
    tab.to_csv(OUT / "transcript_markers_by_task_diagnosis.csv")
    prefix_dist = pd.crosstab(df.task, df.n_distinct_prefixes.clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3+"}))
    prefix_dist.to_csv(OUT / "transcript_n_speaker_prefixes_by_task.csv")
    size = df.groupby("task")[["n_lines", "n_tokens"]].quantile([.05, .5, .95]).unstack().round(0)
    size.to_csv(OUT / "transcript_size_by_task.csv")
    sess = df.groupby("participant_id")["dialogue"].any()
    convention_vs_channel(meta, df)
    print("== Marques dans les transcriptions (% de fichiers) ==\n", tab)
    print("\n== Nombre de préfixes de locuteur distincts par fichier ==\n", prefix_dist)
    print("\n== Taille (lignes, tokens) ==\n", size)
    print(f"\nSessions avec ≥ 1 transcription dialoguée : {sess.sum()} / {len(sess)}")
    return df


def convention_vs_channel(meta, df):
    """Convention de transcription (préfixe de locuteur) comme indicateur de source : lien au diagnostic et au canal (train)."""
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score
    conv = df.groupby("participant_id")["has_speaker_prefix"].mean().rename("prefix_conv")
    ch = (pd.read_csv(ROOT / "results" / "cache" / "01_audit" / "recording_features.csv", dtype={"participant_id": str})
          .groupby("participant_id")[["noise_db", "snr_db", "bw95_hz"]].mean())
    x = meta[meta.Split == "Train"].set_index("participant_id").join(conv).join(ch)
    consistent = df[df.Split == "Train"].groupby("participant_id")["has_speaker_prefix"].agg(lambda s: s.all() or not s.any())
    dh = x[x.diagnosis != "MCI"]
    res = {"pct_sessions_consistent_convention": round(100 * consistent.mean(), 1),
           "pct_sessions_with_prefix_by_diag": (x.groupby("diagnosis").prefix_conv.apply(lambda s: (s >= .5).mean())
                                                .mul(100).round(1).to_dict()),
           "AUC_convention_HC_vs_MCI+Dem": round(roc_auc_score(x.diagnosis != "HC", x.prefix_conv), 3),
           "AUC_convention_Dem_vs_HC": round(roc_auc_score(dh.diagnosis == "Dementia", dh.prefix_conv), 3)}
    g = x.prefix_conv >= .5
    for v in ["noise_db", "snr_db", "bw95_hz"]:
        res[f"{v}_median_with_vs_without_prefix"] = [round(x.loc[g, v].median(), 1), round(x.loc[~g, v].median(), 1)]
        res[f"{v}_p_MannWhitney"] = round(mannwhitneyu(x.loc[g, v], x.loc[~g, v]).pvalue, 4)
    (OUT / "transcript_convention_vs_diagnosis_channel_train.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


# ---------------------------------------------------------------- diarisation

def stage_diarize(meta):
    import torch
    from pyannote.audio import Pipeline
    path = CACHE / "diarization.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    recs = recordings(meta)
    todo = [r for r in recs.itertuples() if f"{r.participant_id}__{r.task}" not in done]
    if todo:
        pipe = Pipeline.from_pretrained(CONFIG["diarization"]["model"])
        pipe.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        for i, r in enumerate(todo):
            y, sr = sf.read(str(r.wav), dtype="float32")
            out = pipe({"waveform": torch.from_numpy(y)[None], "sample_rate": sr})
            ann = getattr(out, CONFIG["diarization"]["output"], out)
            durs = {}
            for seg, _, spk in ann.itertracks(yield_label=True):
                durs[spk] = durs.get(spk, 0.0) + seg.duration
            done[f"{r.participant_id}__{r.task}"] = sorted(durs.values(), reverse=True)
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                path.write_text(json.dumps(done))
                print(f"  diarisation : {len(done)}/{len(recs)}", flush=True)
    return done


def diar_flags(meta, diar, min_s):
    rows = [{"participant_id": k.rsplit("__", 1)[0], "task": k.rsplit("__", 1)[1],
             "n_speakers": len(v), "second_s": v[1] if len(v) > 1 else 0.0,
             "others_s": sum(v[1:])} for k, v in diar.items()]
    rec = pd.DataFrame(rows).merge(meta[["participant_id", "diagnosis", "Split"]], on="participant_id")
    rec["flag"] = rec.second_s >= min_s
    sess = rec.groupby("participant_id")["flag"].any().rename("session_flag")
    return rec, sess


def summarize_diarization(meta, diar, transcripts):
    c = CONFIG["diarization"]
    rec, sess = diar_flags(meta, diar, c["second_speaker_min_s"])
    rec.to_csv(CACHE / "diarization_flags_per_recording.csv", index=False)
    m = meta[["participant_id", "diagnosis", "Split"]].assign(session_flag=lambda d: d.participant_id.map(sess))
    by_task = rec.pivot_table(index=["Split", "diagnosis"], columns="task", values="flag", aggfunc="mean").mul(100)
    by_task = by_task.add_prefix("pct_flag_").join(m.groupby(["Split", "diagnosis"]).agg(
        n=("session_flag", "size"), pct_sessions=("session_flag", lambda x: 100 * x.mean()))).round(1)
    by_task.to_csv(OUT / "pyannote_second_voice_by_diagnosis_task.csv")
    nspk = pd.crosstab(rec.task, rec.n_speakers.clip(upper=4).map(lambda k: f"{k}" if k < 4 else "4+"))
    nspk.to_csv(OUT / "pyannote_n_speakers_by_task.csv")
    grid = []
    for ms in c["grid_s"]:
        r, s = diar_flags(meta, diar, ms)
        row = {"min_second_speaker_s": ms, "pct_sessions": 100 * s.mean()}
        row |= {f"pct_{t}": 100 * r.loc[r.task == t, "flag"].mean() for t in TASKS}
        row |= {f"pct_sessions_{d}_train": 100 * meta.loc[(meta.diagnosis == d) & (meta.Split == "Train"),
                                                         "participant_id"].map(s).mean()
                for d in ["HC", "MCI", "Dementia"]}
        grid.append(row)
    grid = pd.DataFrame(grid).round(1)
    grid.to_csv(OUT / "pyannote_threshold_grid.csv", index=False)
    agree = {}
    if transcripts is not None:
        from sklearn.metrics import cohen_kappa_score
        j = rec.merge(transcripts[["participant_id", "task", "dialogue"]], on=["participant_id", "task"])
        agree["kappa_recording_pyannote_vs_transcript_dialogue"] = cohen_kappa_score(j.flag, j.dialogue)
        agree["crosstab_recording"] = pd.crosstab(j.flag, j.dialogue).to_dict()
    print("== pyannote : % marqués ==\n", by_task, "\n\n== nb locuteurs ==\n", nspk, "\n\n== grille ==\n", grid)
    print("\n== accord avec les transcriptions ==\n", json.dumps(agree, indent=1, default=str))
    return sess, agree


# ---------------------------------------------------------------- tirage et kappa

def stage_sample(meta, sess):
    mc = CONFIG["manual_check"]
    tr = meta[meta.Split == mc["split"]].assign(flag=lambda d: d.participant_id.map(sess).fillna(False))
    rng = np.random.default_rng(CONFIG["seed"])
    pos = tr[tr.flag].participant_id.to_numpy()
    neg = tr[~tr.flag].participant_id.to_numpy()
    pick = np.concatenate([rng.choice(pos, min(mc["n_flagged"], len(pos)), replace=False),
                           rng.choice(neg, min(mc["n_unflagged"], len(neg)), replace=False)])
    rng.shuffle(pick)
    IDS_FILE.write_text("\n".join(pick) + "\n")
    pd.DataFrame({"participant_id": pick, "pyannote_session_flag": [p in set(pos) for p in pick]}).to_csv(
        CACHE / "manual_check_key.csv", index=False)
    print(f"{len(pick)} identifiants écrits dans {IDS_FILE.relative_to(ROOT)} "
          f"({min(mc['n_flagged'], len(pos))} marqués, {min(mc['n_unflagged'], len(neg))} non marqués ; "
          f"ordre aléatoire, statut non indiqué).")


def ecapa_strict_flags(meta):
    spec = importlib.util.spec_from_file_location("audit01", ROOT / "scripts" / "01_audit.py")
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    z = np.load(ROOT / "results" / "cache" / "01_audit" / "embeddings.npz")
    seg = pd.DataFrame({"participant_id": z["pid"], "task": z["task"], "start": z["start"], "end": z["end"]})
    emb = z["emb"]
    embn = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    e = CONFIG["ecapa_strict"]
    s = seg.assign(label=a.cluster_sessions(seg, emb, e["similarity_threshold"]), d=seg.end - seg.start)
    flagged = set()
    for pid, g in s.groupby("participant_id"):
        c0 = embn[g.index[g.label == 0]].mean(0)
        c0 /= np.linalg.norm(c0)
        for _, gk in g[g.label > 0].groupby("label"):
            ck = embn[gk.index].mean(0)
            if ck @ c0 / np.linalg.norm(ck) < e["max_sim_to_main"] and gk.d.sum() >= e["min_s"]:
                flagged.add(pid)
    return meta.participant_id.isin(flagged).set_axis(meta.participant_id)


def kappa_ci(a, b, rng, n_boot=2000):
    from sklearn.metrics import cohen_kappa_score
    a, b = np.asarray(a, int), np.asarray(b, int)
    k = cohen_kappa_score(a, b)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        if len(set(a[i])) > 1 or len(set(b[i])) > 1:
            boots.append(cohen_kappa_score(a[i], b[i]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"kappa": round(k, 3), "ci95": [round(lo, 3), round(hi, 3)], "n": len(a),
            "agreement_pct": round(100 * (a == b).mean(), 1),
            "confusion_manual_x_auto": pd.crosstab(a, b).to_dict()}


def stage_kappa(meta, sess, transcripts):
    man = pd.read_csv(MANUAL_FILE)
    man["participant_id"] = man.participant_id.astype(str).str.strip()
    v = man.second_voice.astype(str).str.strip().str.lower()
    man["manual"] = v.map({"1": 1, "0": 0, "oui": 1, "non": 0, "yes": 1, "no": 0, "true": 1, "false": 0})
    ids = set(IDS_FILE.read_text().split())
    man = man[man.participant_id.isin(ids) & man.manual.notna()]
    rng = np.random.default_rng(CONFIG["seed"])
    res = {"n_rated": len(man), "n_expected": len(ids)}
    res["pyannote_session"] = kappa_ci(man.manual, man.participant_id.map(sess).fillna(False), rng)
    res["ecapa_strict_session"] = kappa_ci(man.manual, man.participant_id.map(ecapa_strict_flags(meta)), rng)
    if transcripts is not None:
        dia = transcripts.groupby("participant_id")["dialogue"].any()
        res["transcript_dialogue_session"] = kappa_ci(man.manual, man.participant_id.map(dia).fillna(False), rng)
    (OUT / "manual_check_kappa.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(res, indent=2, ensure_ascii=False, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["transcripts", "diarize", "sample", "kappa", "all"])
    stage = ap.parse_args().stage
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = load_dataset()
    tpath = CACHE / "transcript_markers.csv"
    transcripts = None
    if stage in ("transcripts", "all"):
        transcripts = stage_transcripts(meta)
    elif tpath.exists():
        transcripts = pd.read_csv(tpath, dtype={"participant_id": str})
    sess = None
    if stage in ("diarize", "sample", "kappa", "all"):
        diar = stage_diarize(meta)
        sess, agree = summarize_diarization(meta, diar, transcripts)
        if stage in ("diarize", "all"):
            (OUT / "summary.json").write_text(json.dumps(
                {"generated": datetime.now().isoformat(timespec="seconds"),
                 "pct_sessions_pyannote": round(100 * sess.mean(), 1), **agree}, indent=2, default=str))
    if stage in ("sample", "all"):
        stage_sample(meta, sess)
    if stage == "kappa":
        stage_kappa(meta, sess, transcripts)
    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
