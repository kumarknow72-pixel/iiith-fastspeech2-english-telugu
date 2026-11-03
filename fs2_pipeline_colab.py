# =====================================================================================
# IIITH Voice-AI assignment: FastSpeech2 English (Svarah) + Telugu (IndicTTS)
# ONE CELL. Paste into Colab and run. It is resumable: every stage caches to Drive,
# so after a disconnect (or after switching to a T4 GPU) just run the cell again.
#
# Stages: 0 diagnostics -> 1 rebuild durations + features -> 2 validate (hard gate)
#         -> 3 train EN + TE -> 4 evaluate -> 5 synthesize 5+5 samples -> 6 plots
#         -> 7 report + GitHub package
# Nothing in the old module2/module3 folders is touched. New outputs go to OUT below.
# =====================================================================================

# ----------------------------- CONFIG (edit only this block) ---------------------------
PROJECT = "/content/drive/MyDrive/IIITH_FastSpeech2_Project"
OUT = f"{PROJECT}/module4_final"                     # all new outputs (old folders untouched)
LANGS = ["english", "telugu"]
ALIGN_JSON = {l: f"{PROJECT}/full_alignment/{l}_alignment.json" for l in LANGS}
META_JSON = {l: {s: f"{PROJECT}/module1_dataset/{l}/{s}_metadata.json" for s in ("train", "val")} for l in LANGS}

RUN_FEATURES = True
RUN_TRAIN = False
RUN_SYNTH = False
TRAIN_MINUTES = 45          # per language, PER RUN of this cell. Re-run the cell to keep training.
MAX_STEPS = 30000           # hard cap on total steps (checkpoint resumes across runs)
BATCH_SIZE = None           # None -> 16 on GPU, 8 on CPU
MAX_MEL_FRAMES = 1000       # training utterances longer than this are skipped (~11.6 s)
GRIFFIN_LIM_ITERS = 60
N_SAMPLES = 5
BOUNDARY_MODE = "midpoint"  # within-word gap between two CTC characters: "midpoint" or "next_start"
GITHUB_REMOTE = ""          # optional: "https://<TOKEN>@github.com/<user>/<repo>.git" (leave "" to skip push)

# audio / model constants (match the stock FastSpeech2 config in your repo)
SR, HOP, NFFT, N_MELS, FMIN, FMAX = 22050, 256, 1024, 80, 0, 8000
CTC_SEC = 0.02              # MMS_FA / wav2vec2 frame = 320 samples @16 kHz = 20 ms
PAD_START, PAD_END = 0.15, 0.20   # seconds of silence kept around speech (become '~' tokens)
MIN_PASS_RATE = 0.80        # hard gate: stop if fewer utterances survive validation
EN_CUSTOM = [
    "the quick brown fox jumps over the lazy dog",
    "speech synthesis converts written text into spoken audio",
    "hyderabad is famous for its biryani and its historic charminar",
    "please call me when you reach the station",
    "machine learning models need a lot of clean data",
]
# ---------------------------------------------------------------------------------------

import os, sys, json, re, time, math, random, shutil, glob, subprocess, warnings
from collections import Counter
import numpy as np
warnings.filterwarnings("ignore")

try:
    from google.colab import drive
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")
except Exception as _e:
    print("drive mount skipped:", _e)

for _mod, _pip in (("unidecode", "unidecode"), ("pyworld", "pyworld"), ("soundfile", "soundfile"), ("librosa", "librosa")):
    try:
        __import__(_mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pip], check=False)

import librosa, soundfile as sf, pyworld as pw
from unidecode import unidecode
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

os.makedirs(OUT, exist_ok=True)
random.seed(0); np.random.seed(0)
T0_ALL = time.time()
SUMMARY = {"config": dict(SR=SR, HOP=HOP, NFFT=NFFT, N_MELS=N_MELS, BOUNDARY_MODE=BOUNDARY_MODE,
                          PAD_START=PAD_START, PAD_END=PAD_END, CTC_SEC=CTC_SEC)}


def jload(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def jsave(o, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2, ensure_ascii=False)


# =====================================================================================
# STAGE 1 helpers: alignment JSON -> token durations (no fabricated supervision)
# =====================================================================================
def flatten_intervals(o):
    """Return the character-level dicts that carry start_frame/end_frame, whatever the nesting."""
    if isinstance(o, dict):
        kids = []
        for v in o.values():
            if isinstance(v, (list, dict)):
                kids += flatten_intervals(v)
        if kids:
            return kids
        return [o] if ("start_frame" in o and "end_frame" in o) else []
    if isinstance(o, list):
        out = []
        for v in o:
            out += flatten_intervals(v)
        return out
    return []


def norm_words(text, drop_apostrophe=False):
    words = []
    for w in str(text).split():
        w = unidecode(w).lower()
        w = re.sub(r"[^a-z']", "", w)
        if drop_apostrophe:
            w = w.replace("'", "")
        if w:
            words.append(w)
    return words


def meta_to_keys(obj, align, by_stem):
    keys = []

    def visit(o):
        if isinstance(o, str):
            if o in align:
                keys.append(o)
            elif len(o) < 300:
                st = os.path.splitext(os.path.basename(o))[0]
                if st in by_stem:
                    keys.append(by_stem[st])
        elif isinstance(o, dict):
            for k, v in o.items():
                visit(k); visit(v)
        elif isinstance(o, list):
            for v in o:
                visit(v)

    visit(obj)
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k); out.append(k)
    return out


_MEL_BASIS = None


def mel_basis():
    global _MEL_BASIS
    if _MEL_BASIS is None:
        _MEL_BASIS = librosa.filters.mel(sr=SR, n_fft=NFFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    return _MEL_BASIS


def interp_f0(f0):
    f0 = f0.copy()
    v = f0 > 0
    if v.sum() == 0:
        return np.zeros_like(f0)
    idx = np.arange(len(f0))
    f0[~v] = np.interp(idx[~v], idx[v], f0[v])
    return f0


def build_utt(key, entry):
    """Return (dict_of_arrays, None) or (None, reason)."""
    if not os.path.exists(key):
        return None, "wav_missing"
    iv = flatten_intervals(entry.get("aligned_words"))
    if not iv:
        return None, "no_intervals"
    iv = sorted(iv, key=lambda d: d["start_frame"])
    text_src = entry.get("romanized") or entry.get("transcript") or ""
    words = norm_words(text_src)
    if sum(len(w) for w in words) != len(iv):
        words = norm_words(text_src, drop_apostrophe=True)
    if sum(len(w) for w in words) != len(iv):
        return None, "char_count_mismatch"
    st = np.array([d["start_frame"] for d in iv], dtype=np.float64) * CTC_SEC
    en = np.array([d["end_frame"] for d in iv], dtype=np.float64) * CTC_SEC
    if np.any(en < st):
        return None, "bad_interval"

    y, _ = librosa.load(key, sr=SR, mono=True)
    dur = len(y) / SR
    if en[-1] > dur + 0.3:
        return None, "alignment_beyond_audio"     # catches wrong-audio / wrong-index pairs
    t0 = max(0.0, st[0] - PAD_START)
    t1 = min(dur, en[-1] + PAD_END)
    y = y[int(t0 * SR): int(t1 * SR)]
    if len(y) < 0.4 * SR or np.max(np.abs(y)) < 1e-4:
        return None, "too_short_or_silent"
    y = (y / np.max(np.abs(y)) * 0.95).astype(np.float32)

    # ---- spectral features on the trimmed audio, exact repo settings ----
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP, win_length=NFFT, window="hann", center=True))
    T = S.shape[1]
    mel = np.log(np.clip(mel_basis() @ S, 1e-5, None)).T.astype(np.float32)     # (T, 80)
    energy_f = np.linalg.norm(S, axis=0).astype(np.float32)                       # (T,)
    f0, tt = pw.dio(y.astype(np.float64), SR, frame_period=HOP / SR * 1000)
    f0 = pw.stonemask(y.astype(np.float64), f0, tt, SR)
    f0 = np.pad(f0, (0, max(0, T - len(f0))))[:T]
    voiced_frac = float((f0 > 0).mean())
    pitch_f = interp_f0(f0).astype(np.float32)

    # ---- tokens: '~' + chars of words joined by ' ' + '~' ; boundaries from real CTC spans ----
    tokens, starts = ["~"], [0.0]
    ci = 0
    prev_last = None
    for wi, w in enumerate(words):
        if wi > 0:
            tokens.append(" "); starts.append(en[prev_last])        # pause between words -> separator
        for pos, ch in enumerate(w):
            tokens.append(ch)
            if pos == 0:
                starts.append(st[ci])
            elif BOUNDARY_MODE == "next_start":
                starts.append(st[ci])
            else:
                starts.append(0.5 * (en[ci - 1] + st[ci]))
            ci += 1
        prev_last = ci - 1
    tokens.append("~"); starts.append(en[-1])
    starts = np.array(starts) - t0
    bf = np.rint(starts * SR / HOP).astype(np.int64)                  # seconds -> mel frames
    bf = np.clip(bf, 0, T)
    bf[0] = 0
    bf = np.maximum.accumulate(bf)
    bf = np.append(bf, T)
    d = np.diff(bf)
    if len(d) != len(tokens) or d.sum() != T or d.min() < 0:
        return None, "duration_invalid"
    n_zero = int((d == 0).sum())

    # ---- token-level pitch / energy = mean over each token's frames ----
    pt = np.zeros(len(tokens), np.float32); et = np.zeros(len(tokens), np.float32)
    for k in range(len(tokens)):
        if d[k] > 0:
            pt[k] = pitch_f[bf[k]:bf[k + 1]].mean(); et[k] = energy_f[bf[k]:bf[k + 1]].mean()
    if not (np.isfinite(mel).all() and np.isfinite(pt).all() and np.isfinite(et).all()):
        return None, "nan_features"
    return dict(mel=mel, duration=d.astype(np.int16), pitch=pt, energy=et, text=np.array("".join(tokens)),
                align_ratio=np.float32(en[-1] / dur), zero_dur=np.int32(n_zero),
                voiced_frac=np.float32(voiced_frac), key=np.array(key)), None


def extract_split(lang, split, keys, align):
    d = f"{OUT}/features/{lang}/{split}"
    os.makedirs(d, exist_ok=True)
    stats = Counter()
    for key in tqdm(keys, desc=f"{lang}/{split}"):
        stem = os.path.splitext(os.path.basename(key))[0]
        f = f"{d}/{stem}.npz"
        if os.path.exists(f):
            stats["ok_cached"] += 1
            continue
        try:
            res, why = build_utt(key, align[key])
        except Exception as ex:
            res, why = None, f"exception:{type(ex).__name__}"
        if res is None:
            stats[why] += 1
            continue
        np.savez(f, **res)
        stats["ok"] += 1
    return dict(stats)


# =====================================================================================
# STAGE 0 + 1 + 2 : diagnostics, rebuild, validate
# =====================================================================================
KEYS = {}
if RUN_FEATURES:
    print("=" * 90, "\nSTAGE 1/2: rebuild duration targets + features from the raw WAVs + stored CTC alignments")
    for lang in LANGS:
        align = jload(ALIGN_JSON[lang])
        assert isinstance(align, dict), "alignment JSON must be a dict keyed by wav path"
        by_stem = {os.path.splitext(os.path.basename(k))[0]: k for k in align}
        KEYS[lang] = {}
        for split in ("train", "val"):
            ks = meta_to_keys(jload(META_JSON[lang][split]), align, by_stem)
            KEYS[lang][split] = ks
            print(f"[{lang}/{split}] metadata -> {len(ks)} alignment entries")
            assert len(ks) > 0, f"no metadata entries matched the alignment JSON for {lang}/{split}"
        if lang == "english":       # answers the old "mel 343 vs wav 755" question
            try:
                old = f"{PROJECT}/module2_features/english/train"
                for i in range(3):
                    m = np.load(f"{old}/english_train_{i:05d}_mel.npy")
                    n_old = m.shape[0] if m.shape[1] == 80 else m.shape[1]
                    k_idx = KEYS[lang]["train"][i]
                    k_raw = f"/content/drive/MyDrive/FastSpeech2_Backup/raw_data/Svarah/Svarah/train/train_{i:05d}.wav"
                    f_idx = int(sf.info(k_idx).frames * SR / sf.info(k_idx).samplerate) // HOP + 1
                    f_raw = int(sf.info(k_raw).frames * SR / sf.info(k_raw).samplerate) // HOP + 1 if os.path.exists(k_raw) else -1
                    print(f"  old mel[{i}]={n_old} frames | i-th ACCEPTED wav expects {f_idx} | raw train_{i:05d}.wav expects {f_raw}")
            except Exception as ex:
                print("  (old-feature diagnostic skipped:", type(ex).__name__, ex, ")")
        for split in ("train", "val"):
            st = extract_split(lang, split, KEYS[lang][split], align)
            n_ok = st.get("ok", 0) + st.get("ok_cached", 0)
            rate = n_ok / max(1, len(KEYS[lang][split]))
            print(f"[{lang}/{split}] {st} -> pass rate {rate:.1%}")
            SUMMARY.setdefault("extraction", {})[f"{lang}/{split}"] = dict(stats=st, total=len(KEYS[lang][split]), pass_rate=rate)
            if rate < MIN_PASS_RATE:
                raise SystemExit(f"STOP: only {rate:.1%} of {lang}/{split} passed validation (gate {MIN_PASS_RATE:.0%}). "
                                 f"Look at the failure counts above (char_count_mismatch => romanized text vs aligned chars; "
                                 f"alignment_beyond_audio => alignment belongs to different audio). Do NOT train.")
        del align


# ----------------------------- load validated features --------------------------------
def list_feats(lang, split):
    return sorted(glob.glob(f"{OUT}/features/{lang}/{split}/*.npz"))


def validate_files(files):
    bad, ratios, zero, tot = 0, [], 0, 0
    for f in files:
        z = np.load(f)
        d = z["duration"].astype(np.int64); toks = str(z["text"])
        if d.sum() != z["mel"].shape[0] or len(d) != len(toks) or len(z["pitch"]) != len(d) or len(z["energy"]) != len(d):
            bad += 1
        ratios.append(float(z["align_ratio"])); zero += int(z["zero_dur"]); tot += len(d)
    return dict(n=len(files), bad=bad, median_align_over_audio=float(np.median(ratios)) if ratios else None,
                zero_duration_token_frac=zero / max(1, tot))


for lang in LANGS:
    for split in ("train", "val"):
        fl = list_feats(lang, split)
        if not fl:
            raise SystemExit(f"No features for {lang}/{split}. Set RUN_FEATURES=True.")
        v = validate_files(fl)
        SUMMARY.setdefault("validation", {})[f"{lang}/{split}"] = v
        print(f"[validate {lang}/{split}] {v}")
        assert v["bad"] == 0, "feature/duration length mismatch found -> aborting"

if not (RUN_TRAIN or RUN_SYNTH):
    raise SystemExit("Features stage finished (RUN_TRAIN and RUN_SYNTH are False).")

# =====================================================================================
# MODEL (compact FastSpeech2: encoder, variance adaptor, length regulator, decoder, postnet)
# =====================================================================================
import torch, torch.nn as nn, torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LITE = False  # Keep checkpoint architecture identical during CPU inference
torch.set_num_threads(max(1, os.cpu_count() or 1))
BS = BATCH_SIZE or (16 if DEV.type == "cuda" else 8)
CFG = dict(d=256, heads=2, ff=(512 if LITE else 1024), k=9, layers=4, vp_ch=256, vp_k=3,
           post_ch=(256 if LITE else 512), drop=0.1)
print(f"device={DEV} | LITE(CPU) model={LITE} | batch={BS} | cfg={CFG}")
SUMMARY["device"] = str(DEV); SUMMARY["model_cfg"] = CFG; SUMMARY["batch_size"] = BS


def pos_table(n, d):
    pe = torch.zeros(n, d)
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


def pad_mask(lens, maxlen):
    return torch.arange(maxlen, device=lens.device)[None, :] >= lens[:, None]    # True where PAD


class FFT(nn.Module):
    def __init__(s, d, h, ff, k, drop):
        super().__init__()
        s.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.c1 = nn.Conv1d(d, ff, k, padding=k // 2); s.c2 = nn.Conv1d(ff, d, 1)
        s.dp = nn.Dropout(drop)

    def forward(s, x, pad):
        a, _ = s.attn(x, x, x, key_padding_mask=pad, need_weights=False)
        x = s.ln1(x + s.dp(a)).masked_fill(pad.unsqueeze(-1), 0)
        f = s.c2(s.dp(F.relu(s.c1(x.transpose(1, 2))))).transpose(1, 2)
        return s.ln2(x + s.dp(f)).masked_fill(pad.unsqueeze(-1), 0)


class VarPred(nn.Module):
    def __init__(s, d, ch, k, drop=0.5):
        super().__init__()
        s.c1 = nn.Conv1d(d, ch, k, padding=k // 2); s.l1 = nn.LayerNorm(ch)
        s.c2 = nn.Conv1d(ch, ch, k, padding=k // 2); s.l2 = nn.LayerNorm(ch)
        s.dp = nn.Dropout(drop); s.out = nn.Linear(ch, 1)

    def forward(s, x, pad):
        h = s.dp(s.l1(F.relu(s.c1(x.transpose(1, 2))).transpose(1, 2)))
        h = s.dp(s.l2(F.relu(s.c2(h.transpose(1, 2))).transpose(1, 2)))
        return s.out(h).squeeze(-1).masked_fill(pad, 0)


def length_regulate(x, d):
    outs = [torch.repeat_interleave(xb, db, dim=0) for xb, db in zip(x, d)]
    lens = torch.tensor([o.size(0) for o in outs], device=x.device)
    return nn.utils.rnn.pad_sequence(outs, batch_first=True), lens


class Postnet(nn.Module):
    def __init__(s, n_mels, ch, k=5, n=5):
        super().__init__()
        L = []
        cin = n_mels
        for i in range(n):
            cout = n_mels if i == n - 1 else ch
            L.append(nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout)))
            cin = cout
        s.layers = nn.ModuleList(L)

    def forward(s, mel, pad):
        x = mel.masked_fill(pad.unsqueeze(-1), 0).transpose(1, 2)
        for i, l in enumerate(s.layers):
            x = l(x)
            if i < len(s.layers) - 1:
                x = torch.tanh(x)
        return x.transpose(1, 2)


class FastSpeech2(nn.Module):
    def __init__(s, n_tok, pbins, ebins, cfg):
        super().__init__()
        d = cfg["d"]
        s.emb = nn.Embedding(n_tok, d, padding_idx=0)
        s.register_buffer("pe", pos_table(6000, d), persistent=False)
        mk = lambda: nn.ModuleList([FFT(d, cfg["heads"], cfg["ff"], cfg["k"], cfg["drop"]) for _ in range(cfg["layers"])])
        s.enc, s.dec = mk(), mk()
        s.dp = VarPred(d, cfg["vp_ch"], cfg["vp_k"]); s.pp = VarPred(d, cfg["vp_ch"], cfg["vp_k"]); s.ep = VarPred(d, cfg["vp_ch"], cfg["vp_k"])
        s.register_buffer("pb", torch.as_tensor(pbins, dtype=torch.float32)); s.register_buffer("eb", torch.as_tensor(ebins, dtype=torch.float32))
        s.pemb = nn.Embedding(len(pbins) + 1, d); s.eemb = nn.Embedding(len(ebins) + 1, d)
        s.mel_lin = nn.Linear(d, N_MELS); s.post = Postnet(N_MELS, cfg["post_ch"])

    def forward(s, ids, src_lens, p_t=None, e_t=None, d_t=None, pc=1.0, ec=1.0, dc=1.0):
        L = ids.size(1)
        src_pad = pad_mask(src_lens, L)
        x = s.emb(ids) + s.pe[:, :L]
        for b in s.enc:
            x = b(x, src_pad)
        logd = s.dp(x, src_pad)
        p_pred, e_pred = s.pp(x, src_pad), s.ep(x, src_pad)
        pv = p_t if p_t is not None else p_pred * pc
        ev = e_t if e_t is not None else e_pred * ec
        x = x + s.pemb(torch.bucketize(pv, s.pb)) + s.eemb(torch.bucketize(ev, s.eb))
        if d_t is None:
            d = torch.clamp(torch.round((torch.exp(logd) - 1) * dc), min=0).long().masked_fill(src_pad, 0)
            for i in range(d.size(0)):
                if d[i].sum() == 0:
                    d[i, 0] = 1
        else:
            d = d_t
        x, mel_lens = length_regulate(x, d)
        Tm = min(x.size(1), s.pe.size(1))
        x, mel_lens = x[:, :Tm], torch.clamp(mel_lens, max=Tm)
        mel_pad = pad_mask(mel_lens, Tm)
        x = x + s.pe[:, :Tm]
        for b in s.dec:
            x = b(x, mel_pad)
        mel = s.mel_lin(x)
        post = mel + s.post(mel, mel_pad)
        return dict(mel=mel, post=post, logd=logd, p=p_pred, e=e_pred, d=d, src_pad=src_pad, mel_pad=mel_pad, mel_lens=mel_lens)


def fs2_loss(o, b):
    valid = (~o["mel_pad"]).unsqueeze(-1).float()
    nv = valid.sum() * N_MELS
    T = o["mel"].size(1)
    tgt = b["mel"][:, :T]
    l_mel = ((o["mel"] - tgt).abs() * valid).sum() / nv
    l_post = ((o["post"] - tgt).abs() * valid).sum() / nv
    sv = (~o["src_pad"]).float(); ns = sv.sum()
    l_d = (((o["logd"] - torch.log(b["dur"].float() + 1)) ** 2) * sv).sum() / ns
    l_p = (((o["p"] - b["pitch"]) ** 2) * sv).sum() / ns
    l_e = (((o["e"] - b["energy"]) ** 2) * sv).sum() / ns
    return l_mel + l_post + l_d + l_p + l_e, dict(mel=l_mel.item(), post=l_post.item(), dur=l_d.item(), pitch=l_p.item(), energy=l_e.item())


# =====================================================================================
# data for training
# =====================================================================================
def build_vocab():
    chars = set()
    for lang in LANGS:
        for split in ("train", "val"):
            for f in list_feats(lang, split):
                chars |= set(str(np.load(f)["text"]))
    return {c: i + 1 for i, c in enumerate(sorted(chars))}


VOCAB = build_vocab()
jsave(VOCAB, f"{OUT}/vocab.json")
print("vocab size (incl. pad):", len(VOCAB) + 1, "->", "".join(sorted(VOCAB)))


def lang_stats(lang):
    P, E = [], []
    for f in list_feats(lang, "train"):
        z = np.load(f); m = z["duration"] > 0
        P.append(z["pitch"][m]); E.append(z["energy"][m])
    P, E = np.concatenate(P), np.concatenate(E)
    st = dict(p_mean=float(P.mean()), p_std=float(P.std() + 1e-8), e_mean=float(E.mean()), e_std=float(E.std() + 1e-8))
    pn, en_ = (P - st["p_mean"]) / st["p_std"], (E - st["e_mean"]) / st["e_std"]
    st["pbins"] = np.linspace(np.percentile(pn, 1), np.percentile(pn, 99), 255).tolist()
    st["ebins"] = np.linspace(np.percentile(en_, 1), np.percentile(en_, 99), 255).tolist()
    return st


class DS:
    def __init__(s, files, st, max_mel=None):
        s.items = []
        s.skipped = 0
        for f in files:
            z = np.load(f)
            T = z["mel"].shape[0]
            if max_mel and T > max_mel:
                s.skipped += 1
                continue
            d = z["duration"].astype(np.int64); m = d > 0
            p = np.where(m, (z["pitch"] - st["p_mean"]) / st["p_std"], 0).astype(np.float32)
            e = np.where(m, (z["energy"] - st["e_mean"]) / st["e_std"], 0).astype(np.float32)
            txt = str(z["text"])
            s.items.append(dict(ids=np.array([VOCAB[c] for c in txt], np.int64), mel=z["mel"], dur=d, pitch=p, energy=e, text=txt, T=T,
                                key=str(z["key"])))

    def __len__(s):
        return len(s.items)


def collate(items):
    n = len(items)
    L = max(len(i["ids"]) for i in items); T = max(i["T"] for i in items)
    ids = np.zeros((n, L), np.int64); dur = np.zeros((n, L), np.int64); pit = np.zeros((n, L), np.float32); ene = np.zeros((n, L), np.float32)
    mel = np.zeros((n, T, N_MELS), np.float32)
    for j, it in enumerate(items):
        l = len(it["ids"])
        ids[j, :l] = it["ids"]; dur[j, :l] = it["dur"]; pit[j, :l] = it["pitch"]; ene[j, :l] = it["energy"]; mel[j, :it["T"]] = it["mel"]
    t = lambda a: torch.from_numpy(a).to(DEV)
    return dict(ids=t(ids), lens=torch.tensor([len(i["ids"]) for i in items], device=DEV), mel=t(mel), dur=t(dur), pitch=t(pit), energy=t(ene))


def batches(ds, bs, shuffle=True):
    idx = list(range(len(ds)))
    if shuffle:
        random.shuffle(idx)
    out = []
    chunk = bs * 40
    for i in range(0, len(idx), chunk):
        part = sorted(idx[i:i + chunk], key=lambda j: ds.items[j]["T"])
        out += [part[k:k + bs] for k in range(0, len(part), bs)]
    if shuffle:
        random.shuffle(out)
    return out


@torch.no_grad()
def evaluate(model, ds, st, max_items=None):
    model.eval()
    items = ds.items[:max_items] if max_items else ds.items
    tf = Counter(); n = 0
    mae_d, ratio, pr, er, nt = [], [], [], [], 0
    for i in range(0, len(items), BS):
        b = collate(items[i:i + BS])
        o = model(b["ids"], b["lens"], b["pitch"], b["energy"], b["dur"])
        _, parts = fs2_loss(o, b)
        for k, v in parts.items():
            tf[k] += v * len(items[i:i + BS])
        n += len(items[i:i + BS])
        o2 = model(b["ids"], b["lens"])                    # free running: predicted duration/pitch/energy
        valid = ~o2["src_pad"]
        mae_d.append(((o2["d"].float() - b["dur"].float()).abs() * valid).sum().item()); nt += valid.sum().item()
        ratio += (o2["d"].sum(1).float() / b["dur"].sum(1).float()).tolist()
        pr.append((((o2["p"] - b["pitch"]) * st["p_std"]) ** 2 * valid).sum().item())
        er.append((((o2["e"] - b["energy"]) * st["e_std"]) ** 2 * valid).sum().item())
    m = {f"teacher_forced_{k}": v / n for k, v in tf.items()}
    m.update(duration_MAE_frames=sum(mae_d) / nt, duration_MAE_ms=sum(mae_d) / nt * HOP / SR * 1000,
             total_length_ratio_pred_over_true=float(np.mean(ratio)), total_length_ratio_std=float(np.std(ratio)),
             pitch_RMSE_Hz_token=math.sqrt(sum(pr) / nt), energy_RMSE_token=math.sqrt(sum(er) / nt), n_val=n)
    model.train()
    return m


def train_lang(lang, st, tr, va):
    ck_dir = f"{OUT}/checkpoints/{lang}"; os.makedirs(ck_dir, exist_ok=True)
    ck = f"{ck_dir}/last.pt"
    model = FastSpeech2(len(VOCAB) + 1, st["pbins"], st["ebins"], CFG).to(DEV)
    opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    step, hist = 0, []
    if os.path.exists(ck):
        s = torch.load(ck, map_location=DEV)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step = s["step"]; hist = s.get("hist", [])
        print(f"[{lang}] resumed from step {step}")
    peak, warm, decay_every = 1e-3, 300, 3000
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[{lang}] params={nparams:.1f}M | train={len(tr)} (skipped {tr.skipped} too long) | val={len(va)} | steps so far={step}")
    t_end = time.time() + TRAIN_MINUTES * 60
    model.train(); t_start = time.time(); run_loss = Counter(); run_n = 0

    def save():
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, hist=hist, cfg=CFG, stats=st), ck)

    done = False
    while not done:
        for bidx in batches(tr, BS):
            if step >= MAX_STEPS or time.time() > t_end:
                done = True
                break
            b = collate([tr.items[j] for j in bidx])
            lr = peak * min(1.0, (step + 1) / warm) * (0.5 ** (step / decay_every))
            for g in opt.param_groups:
                g["lr"] = lr
            o = model(b["ids"], b["lens"], b["pitch"], b["energy"], b["dur"])
            loss, parts = fs2_loss(o, b)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1
            for k, v in parts.items():
                run_loss[k] += v
            run_n += 1
            if step % 50 == 0:
                avg = {k: v / run_n for k, v in run_loss.items()}
                hist.append(dict(step=step, **avg))
                print(f"[{lang}] step {step} | " + " ".join(f"{k}={v:.3f}" for k, v in avg.items()) +
                      f" | lr={lr:.2e} | {(time.time() - t_start) / 60:.1f} min")
                run_loss = Counter(); run_n = 0
            if step % 250 == 0:
                save()
    save()
    return model, step, hist


# =====================================================================================
# synthesis helpers
# =====================================================================================
def text_to_ids(text):
    words = norm_words(text)
    return np.array([VOCAB[c] for c in "~" + " ".join(words) + "~" if c in VOCAB], np.int64)


def griffin_lim(mel_log):
    M = np.exp(mel_log.T)                                          # (80,T) linear mel magnitude
    S = librosa.feature.inverse.mel_to_stft(M, sr=SR, n_fft=NFFT, power=1.0, fmin=FMIN, fmax=FMAX)
    y = librosa.griffinlim(S, n_iter=GRIFFIN_LIM_ITERS, hop_length=HOP, win_length=NFFT)
    return (y / (np.max(np.abs(y)) + 1e-8) * 0.9).astype(np.float32)


@torch.no_grad()
def synth(model, st, ids):
    model.eval()
    t0 = time.time()
    ii = torch.from_numpy(ids)[None].to(DEV); ll = torch.tensor([len(ids)], device=DEV)
    o = model(ii, ll)
    T = int(o["mel_lens"][0])
    mel = o["post"][0, :T].cpu().numpy()
    gen_t = time.time() - t0
    return dict(mel=mel, d=o["d"][0].cpu().numpy(), pitch=(o["p"][0].cpu().numpy() * st["p_std"] + st["p_mean"]),
                energy=(o["e"][0].cpu().numpy() * st["e_std"] + st["e_mean"]), rtf=gen_t / (T * HOP / SR))


def plot_sample(path, title, toks, r, ref=None):
    fig, ax = plt.subplots(4, 1, figsize=(14, 11))
    ax[0].imshow(r["mel"].T, origin="lower", aspect="auto"); ax[0].set_title(f"{title} | predicted mel (log)")
    x = np.arange(len(toks)); lab = [("_" if c == " " else c) for c in toks]
    for a, key, yl, name in ((ax[1], "d", "frames", "duration"), (ax[2], "pitch", "Hz", "pitch"), (ax[3], "energy", "energy", "energy")):
        a.bar(x - 0.2, r[key], width=0.4, label="predicted")
        if ref is not None:
            a.bar(x + 0.2, ref[key], width=0.4, label="ground truth (alignment-derived)", alpha=0.7)
        a.set_ylabel(yl); a.set_title(f"predicted {name} per token"); a.legend(loc="upper right")
        step = max(1, len(toks) // 70)
        a.set_xticks(x[::step]); a.set_xticklabels(lab[::step], fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


# =====================================================================================
# STAGE 3-6: per-language train / evaluate / synthesize / plot
# =====================================================================================
os.makedirs(f"{OUT}/samples", exist_ok=True); os.makedirs(f"{OUT}/plots", exist_ok=True); os.makedirs(f"{OUT}/metrics", exist_ok=True)
SUMMARY["train"] = {}; SUMMARY["eval"] = {}; SUMMARY["samples"] = {}
STATS = {}
for lang in LANGS:
    print("=" * 90, f"\n{lang.upper()}")
    st = lang_stats(lang); STATS[lang] = st
    jsave({k: v for k, v in st.items() if k not in ("pbins", "ebins")}, f"{OUT}/metrics/{lang}_norm_stats.json")
    tr, va = DS(list_feats(lang, "train"), st, MAX_MEL_FRAMES), DS(list_feats(lang, "val"), st, None)
    if RUN_TRAIN:
        model, step, hist = train_lang(lang, st, tr, va)
    else:
        model = FastSpeech2(len(VOCAB) + 1, st["pbins"], st["ebins"], CFG).to(DEV)
        s = torch.load(f"{OUT}/checkpoints/{lang}/last.pt", map_location=DEV); model.load_state_dict(s["model"]); step, hist = s["step"], s.get("hist", [])
    SUMMARY["train"][lang] = dict(steps=step, n_train=len(tr), n_train_skipped_long=tr.skipped, n_val=len(va))
    if hist:
        fig = plt.figure(figsize=(8, 4))
        plt.plot([h["step"] for h in hist], [h["post"] for h in hist], label="postnet mel L1")
        plt.plot([h["step"] for h in hist], [h["dur"] for h in hist], label="log-duration MSE")
        plt.xlabel("step"); plt.legend(); plt.title(f"{lang} training curve"); plt.tight_layout()
        plt.savefig(f"{OUT}/plots/{lang}_training_curve.png", dpi=110); plt.close(fig)

    m = evaluate(model, va, st)
    SUMMARY["eval"][lang] = m; jsave(m, f"{OUT}/metrics/{lang}_val_metrics.json")
    print(f"[{lang}] validation metrics: {json.dumps(m, indent=1)}")

    if RUN_SYNTH:
        odir = f"{OUT}/samples/{lang}"; os.makedirs(odir, exist_ok=True)
        info = []
        jobs = [(f"val{i + 1}", va.items[i]["text"].strip("~"), va.items[i]) for i in range(min(N_SAMPLES, len(va)))]
        if lang == "english":
            jobs += [(f"custom{i + 1}", t, None) for i, t in enumerate(EN_CUSTOM)]
        for name, text, ref in jobs:
            ids = text_to_ids(text)
            r = synth(model, st, ids)
            wav = griffin_lim(r["mel"])
            sf.write(f"{odir}/{lang}_{name}.wav", wav, SR)
            toks = "".join(c for c in ("~" + " ".join(norm_words(text)) + "~"))
            refd = None
            if ref is not None:
                sf.write(f"{odir}/{lang}_{name}_reference_griffinlim.wav", griffin_lim(ref["mel"]), SR)
                refd = dict(d=ref["dur"], pitch=ref["pitch"] * st["p_std"] + st["p_mean"], energy=ref["energy"] * st["e_std"] + st["e_mean"])
                toks = ref["text"]
            plot_sample(f"{OUT}/plots/{lang}_{name}_duration_pitch_energy.png", f"{lang} {name}", toks, r, refd)
            info.append(dict(name=name, text=text, seconds=len(wav) / SR, rtf_model=r["rtf"]))
            print(f"[{lang}] wrote {name}: {len(wav) / SR:.1f}s | text: {text[:70]}")
        SUMMARY["samples"][lang] = info
    del model

# =====================================================================================
# STAGE 7: report + GitHub package
# =====================================================================================
jsave(SUMMARY, f"{OUT}/metrics/pipeline_summary.json")
E, TE = SUMMARY["eval"]["english"], SUMMARY["eval"]["telugu"]
V = SUMMARY["validation"]
report = f"""# FastSpeech2 English and Telugu TTS: Technical Report

## 1. Task
Two separate single-speaker FastSpeech2 models: English (Svarah) and Telugu (IndicTTS Telugu, romanized). Five samples per language,
plots of predicted duration/pitch/energy, evaluation, and this report.

## 2. Data and filtering (Module 1)
Forced alignment (torchaudio MMS_FA, CTC): 5,125 of 5,184 files aligned (98.86%). Filtering kept utterances with speech coverage >= 85% and
mean character confidence >= 80%: English {SUMMARY['extraction']['english/train']['total']} train + {SUMMARY['extraction']['english/val']['total']} val,
Telugu {SUMMARY['extraction']['telugu/train']['total']} train + {SUMMARY['extraction']['telugu/val']['total']} val.

## 3. Duration targets (the critical step)
The earlier duration arrays were invalid (sum != mel frames) because intervals from `merge_tokens()` (blanks removed) were treated as tokens
of a different tokenizer. They were discarded. The rebuilt pipeline:
1. Token sequence = `~` + characters of the *same romanized string the aligner consumed*, words joined by a space token, + `~`. Every letter token
   therefore corresponds to exactly one aligned CTC character (checked per utterance; mismatches are dropped, not patched).
2. Boundaries come from the stored CTC spans converted by time (20 ms per CTC frame -> seconds -> mel frames at hop {HOP}/{SR} Hz), not from
   forcing sums. Silence before/after speech is the `~` tokens; inter-word pauses belong to the space token; the gap between two characters
   inside a word is split at the midpoint (mode `{BOUNDARY_MODE}`).
3. Mel, pitch (WORLD dio + stonemask, interpolated through unvoiced regions) and energy (L2 norm of the STFT magnitude) are computed from the raw
   WAV, trimmed to the aligned speech plus {PAD_START}s/{PAD_END}s, with the repository audio settings (22,050 Hz, hop 256, FFT/window 1024, 80 mels, 0-8000 Hz).
   Pitch and energy are averaged per token (phoneme-level, as in stock FastSpeech2).
4. Hard checks per utterance: sum(duration) == mel frames, len(duration) == len(tokens), finite features, alignment end not beyond audio length.
   Pass rates: {json.dumps({k: round(v['pass_rate'], 4) for k, v in SUMMARY['extraction'].items()})}.
   Median (last aligned time / audio length): english train {V['english/train']['median_align_over_audio']:.3f}, telugu train {V['telugu/train']['median_align_over_audio']:.3f}.
   Fraction of zero-duration tokens: english {V['english/train']['zero_duration_token_frac']:.4f}, telugu {V['telugu/train']['zero_duration_token_frac']:.4f}.

## 4. Model and training
Encoder (4 FFT blocks), variance adaptor (duration, pitch, energy predictors; 256-bin pitch/energy embeddings; length regulator), decoder
(4 FFT blocks), 5-layer postnet; hidden 256, 2 heads, 80 mel channels. Config used: {json.dumps(CFG)} on `{SUMMARY['device']}`
(CPU runs use the reduced "LITE" feed-forward/postnet widths). Loss = mel L1 + postnet mel L1 + log-duration MSE + pitch MSE + energy MSE.
Adam, peak LR 1e-3, 300-step warm-up, halving every 3,000 steps, gradient clip 1.0, batch {SUMMARY['batch_size']}.
Steps trained: English {SUMMARY['train']['english']['steps']}, Telugu {SUMMARY['train']['telugu']['steps']} (stock recipe is 160,000; see limitations).
Vocoder: Griffin-Lim ({GRIFFIN_LIM_ITERS} iterations) on the predicted log-mel (no neural vocoder).

## 5. Evaluation (validation sets, objective)
| Metric | English | Telugu |
|---|---|---|
| Teacher-forced postnet mel L1 (log-mel) | {E['teacher_forced_post']:.3f} | {TE['teacher_forced_post']:.3f} |
| Duration MAE per token (frames / ms) | {E['duration_MAE_frames']:.2f} / {E['duration_MAE_ms']:.0f} | {TE['duration_MAE_frames']:.2f} / {TE['duration_MAE_ms']:.0f} |
| Predicted/true total length | {E['total_length_ratio_pred_over_true']:.3f} +- {E['total_length_ratio_std']:.3f} | {TE['total_length_ratio_pred_over_true']:.3f} +- {TE['total_length_ratio_std']:.3f} |
| Pitch RMSE per token (Hz) | {E['pitch_RMSE_Hz_token']:.1f} | {TE['pitch_RMSE_Hz_token']:.1f} |
| Energy RMSE per token | {E['energy_RMSE_token']:.2f} | {TE['energy_RMSE_token']:.2f} |
| Validation utterances | {E['n_val']} | {TE['n_val']} |

Samples: `samples/english/*.wav` and `samples/telugu/*.wav` (val1..val5 are validation texts with a Griffin-Lim re-synthesis of the true mel for
fair comparison; English also has custom1..custom5). Plots: `plots/*_duration_pitch_energy.png` (predicted vs alignment-derived ground truth)
and `plots/*_training_curve.png`. No listening test (MOS) was run; objective numbers do not measure naturalness.

## 6. Limitations (be explicit when submitting)
* **Telugu text is romanized with uroman, not phonemes.** uroman is a script transliteration, not a pronunciation model: it loses vowel length,
  geminates, retroflex/dental distinctions and schwa behaviour, so the model must learn the many-to-one spelling-to-sound map from ~850 utterances.
  Expect mispronunciations; a Telugu grapheme-to-phoneme front end or a Unicode/Indic-aware vocabulary is the main quality upgrade.
* **Training budget.** Far below the stock 160k steps (on a CPU runtime it is a few thousand at most), so mel predictions are over-smoothed.
* **Single-speaker model on multi-speaker data.** Svarah has many accented speakers; a single-speaker config averages them. Speaker embeddings would help.
* **Griffin-Lim vocoder** adds a metallic/phase-noise artefact independent of the acoustic model. HiFi-GAN would improve naturalness.
* **Duration boundaries** come from 20 ms CTC frames of a wav2vec2 aligner (about +-1 CTC frame uncertainty and a possible constant offset); within-word
  boundaries use a documented midpoint rule. They are alignment-derived targets, not human-annotated ground truth.
* Punctuation is stripped (no prosodic cues); utterances over {MAX_MEL_FRAMES} mel frames were excluded from training; filtering kept
  only 2,173 of 5,125 aligned files.
"""
open(f"{OUT}/TECHNICAL_REPORT.md", "w", encoding="utf-8").write(report)

REPO = f"{OUT}/github_repo"
shutil.rmtree(REPO, ignore_errors=True); os.makedirs(REPO)
for sub in ("samples", "plots", "metrics"):
    shutil.copytree(f"{OUT}/{sub}", f"{REPO}/{sub}")
shutil.copy(f"{OUT}/TECHNICAL_REPORT.md", f"{REPO}/TECHNICAL_REPORT.md"); shutil.copy(f"{OUT}/vocab.json", f"{REPO}/vocab.json")
try:
    src = get_ipython().user_ns["_ih"][-1]              # source of this very cell
    open(f"{REPO}/fs2_pipeline_colab.py", "w", encoding="utf-8").write(src.lstrip("\n"))
except Exception as _e:
    print("could not save cell source automatically; copy it into the repo by hand:", _e)
open(f"{REPO}/requirements.txt", "w").write("torch\nnumpy\nlibrosa\nsoundfile\npyworld\nunidecode\nmatplotlib\ntqdm\n")
open(f"{REPO}/.gitignore", "w").write("checkpoints/\nfeatures/\n*.npz\n*.pt\n__pycache__/\n")
open(f"{REPO}/README.md", "w", encoding="utf-8").write(f"""# FastSpeech2 English + Telugu TTS (IIIT Hyderabad Voice AI assignment)

* `fs2_pipeline_colab.py`: the complete pipeline (duration rebuild from CTC alignments, features, training, evaluation, synthesis, plots, report). Run as one Colab cell.
* `TECHNICAL_REPORT.md`: method, results and limitations (read section 6).
* `samples/`: 5 validation-text samples per language (+5 English custom sentences) and reference re-syntheses.
* `plots/`: predicted vs reference duration, pitch, energy per token; training curves.
* `metrics/`: validation metrics, normalisation stats, pipeline summary.
* Datasets: Svarah (English), IndicTTS Telugu. Not redistributed here. Model checkpoints are kept on Drive (too large for GitHub).
""")
print("=" * 90, f"\nGitHub package ready: {REPO}")
if GITHUB_REMOTE:
    cmds = [["git", "init"], ["git", "add", "."], ["git", "-c", "user.name=student", "-c", "user.email=student@example.com", "commit", "-m", "FastSpeech2 EN+TE TTS"],
            ["git", "branch", "-M", "main"], ["git", "remote", "add", "origin", GITHUB_REMOTE], ["git", "push", "-u", "origin", "main"]]
    for c in cmds:
        r = subprocess.run(c, cwd=REPO, capture_output=True, text=True)
        print(" ".join(x if "@github.com" not in x else "<remote hidden>" for x in c), "->", r.returncode, (r.stderr or r.stdout).replace(GITHUB_REMOTE, "<hidden>")[:200])
else:
    print("To publish: create an empty repo on github.com, then in a Colab cell run:\n"
          f"  %cd {REPO}\n  !git init && git add . && git commit -m 'FastSpeech2 EN+TE TTS' && git branch -M main\n"
          "  !git remote add origin https://<TOKEN>@github.com/<user>/<repo>.git && git push -u origin main\n"
          "(or download the folder and upload it through the GitHub web UI).")
print(f"Total wall time this run: {(time.time() - T0_ALL) / 60:.1f} min. Re-run the cell to continue training; change nothing else.")
