# FastSpeech2 English and Telugu TTS: Technical Report

## 1. Task
Two separate single-speaker FastSpeech2 models: English (Svarah) and Telugu (IndicTTS Telugu, romanized). Five samples per language,
plots of predicted duration/pitch/energy, evaluation, and this report.

## 2. Data and filtering (Module 1)
Forced alignment (torchaudio MMS_FA, CTC): 5,125 of 5,184 files aligned (98.86%). Filtering kept utterances with speech coverage >= 85% and
mean character confidence >= 80%: English 1107 train + 123 val,
Telugu 848 train + 95 val.

## 3. Duration targets (the critical step)
The earlier duration arrays were invalid (sum != mel frames) because intervals from `merge_tokens()` (blanks removed) were treated as tokens
of a different tokenizer. They were discarded. The rebuilt pipeline:
1. Token sequence = `~` + characters of the *same romanized string the aligner consumed*, words joined by a space token, + `~`. Every letter token
   therefore corresponds to exactly one aligned CTC character (checked per utterance; mismatches are dropped, not patched).
2. Boundaries come from the stored CTC spans converted by time (20 ms per CTC frame -> seconds -> mel frames at hop 256/22050 Hz), not from
   forcing sums. Silence before/after speech is the `~` tokens; inter-word pauses belong to the space token; the gap between two characters
   inside a word is split at the midpoint (mode `midpoint`).
3. Mel, pitch (WORLD dio + stonemask, interpolated through unvoiced regions) and energy (L2 norm of the STFT magnitude) are computed from the raw
   WAV, trimmed to the aligned speech plus 0.15s/0.2s, with the repository audio settings (22,050 Hz, hop 256, FFT/window 1024, 80 mels, 0-8000 Hz).
   Pitch and energy are averaged per token (phoneme-level, as in stock FastSpeech2).
4. Hard checks per utterance: sum(duration) == mel frames, len(duration) == len(tokens), finite features, alignment end not beyond audio length.
   Pass rates: {"english/train": 1.0, "english/val": 1.0, "telugu/train": 1.0, "telugu/val": 1.0}.
   Median (last aligned time / audio length): english train 0.963, telugu train 0.965.
   Fraction of zero-duration tokens: english 0.0018, telugu 0.0001.

## 4. Model and training
Encoder (4 FFT blocks), variance adaptor (duration, pitch, energy predictors; 256-bin pitch/energy embeddings; length regulator), decoder
(4 FFT blocks), 5-layer postnet; hidden 256, 2 heads, 80 mel channels. Config used: {"d": 256, "heads": 2, "ff": 1024, "k": 9, "layers": 4, "vp_ch": 256, "vp_k": 3, "post_ch": 512, "drop": 0.1} on `Tesla T4 GPU training; CPU checkpoint inference`
(The trained checkpoints use feed-forward width 1024 and postnet width 512; CPU inference retains the same architecture). Loss = mel L1 + postnet mel L1 + log-duration MSE + pitch MSE + energy MSE.
Adam, peak LR 1e-3, 300-step warm-up, halving every 3,000 steps, gradient clip 1.0, batch 16.
Steps trained: English 7778, Telugu 6016 (stock recipe is 160,000; see limitations).
Vocoder: Griffin-Lim (20 iterations) on the predicted log-mel (no neural vocoder).

## 5. Evaluation (validation sets, objective)
| Metric | English | Telugu |
|---|---|---|
| Teacher-forced postnet mel L1 (log-mel) | 0.997 | 0.752 |
| Duration MAE per token (frames / ms) | 3.00 / 35 | 1.99 / 23 |
| Predicted/true total length | 0.864 +- 0.220 | 0.949 +- 0.088 |
| Pitch RMSE per token (Hz) | 64.5 | 38.7 |
| Energy RMSE per token | 29.88 | 21.22 |
| Validation utterances | 123 | 95 |

Audio samples were generated during the experiments but are not
included in this public repository pending confirmation of dataset
redistribution permissions. English experiments included five validation
samples and five custom samples; Telugu experiments included five
validation samples.

Plots: `plots/*_duration_pitch_energy.png` (predicted vs alignment-derived
targets) and `plots/*_training_curve.png`. No listening test (MOS) was
run; objective numbers do not measure naturalness.

## 6. Limitations (be explicit when submitting)
* **Telugu text is romanized with uroman, not phonemes.** uroman is a script transliteration, not a pronunciation model: it loses vowel length,
  geminates, retroflex/dental distinctions and schwa behaviour, so the model must learn the many-to-one spelling-to-sound map from ~850 utterances.
  Expect mispronunciations; a Telugu grapheme-to-phoneme front end or a Unicode/Indic-aware vocabulary is the main quality upgrade.
* **Training budget.** Training was limited to 7,778 English steps and 6,016 Telugu steps, far below the 160k-step stock recipe; this limits acoustic-model quality.
* **Single-speaker model on multi-speaker data.** Svarah has many accented speakers; a single-speaker config averages them. Speaker embeddings would help.
* **Griffin-Lim vocoder** adds a metallic/phase-noise artefact independent of the acoustic model. HiFi-GAN would improve naturalness.
* **Duration boundaries** come from 20 ms CTC frames of a wav2vec2 aligner (about +-1 CTC frame uncertainty and a possible constant offset); within-word
  boundaries use a documented midpoint rule. They are alignment-derived targets, not human-annotated ground truth.
* Punctuation is stripped (no prosodic cues); utterances over 1000 mel frames were excluded from training; filtering kept
  only 2,173 of 5,125 aligned files.
