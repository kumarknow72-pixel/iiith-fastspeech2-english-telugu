# FastSpeech2 English and Telugu TTS: Corrected Technical Report

## 1. Task and repository scope
The project implements separate English (Svarah) and Telugu (IndicTTS Telugu, romanized) FastSpeech2 experiments. The public repository contains the pipeline code, evaluation metrics, visualizations and documentation. Generated audio and trained checkpoints are not included in the public repository; see README.md and the limitations below.

## 2. Data and filtering
Forced alignment used torchaudio MMS_FA (CTC): 5,125 of 5,184 files were aligned (98.86%). Filtering retained utterances with speech coverage >= 85% and mean character confidence >= 80%.

- Eligible after filtering: English 1,107 train + 123 validation; Telugu 848 train + 95 validation.
- Actually used for training after the 1,000-mel-frame length limit: English 979 (1,107 eligible minus 128 skipped); Telugu 219 (848 eligible minus 629 skipped).
- Validation sets: 123 English and 95 Telugu.

The eligible counts are not the same as the counts actually used for training. The Telugu training set was especially reduced by the length filter.

## 3. Duration targets and feature preparation
An earlier duration-array approach was discarded because duration sums did not match mel-frame counts. The rebuilt pipeline uses the same romanized character sequence consumed by the aligner, with start/end marker tokens and spaces retained. Character-to-alignment mismatches are dropped rather than patched.

1. CTC boundaries are converted from 20 ms frames to seconds and then to mel frames using 22,050 Hz sampling and hop length 256.
2. Silence before and after speech is assigned to start/end marker tokens; inter-word pauses belong to the space token; within-word character gaps are split at the midpoint.
3. Mel, pitch and energy are computed from raw audio using 22,050 Hz, hop 256, FFT/window 1,024, 80 mel channels and 0–8,000 Hz.
4. Pitch uses WORLD DIO plus StoneMask, interpolated through unvoiced regions. Energy is the L2 norm of STFT magnitude.
5. Per-utterance checks include duration sum equals mel-frame count, duration length equals token length, finite features and alignment end within audio length.

Feature validation pass rates: 100% for English train/validation and Telugu train/validation.

Median aligned-time/audio ratios: English train 0.9627, English validation 0.9608, Telugu train 0.9647, Telugu validation 0.9689. Zero-duration token fractions: English train 0.00184, Telugu train 0.000106.

## 4. Model and training
The model uses a 4-block FFT encoder, variance adaptor with duration, pitch and energy predictors, length regulator, 4-block FFT decoder and 5-layer postnet. Hidden size 256; 2 attention heads; feed-forward width 1,024; FFT kernel 9; pitch/energy embedding channels 256; predictor kernel 3; postnet channels 512; dropout 0.1.

- Loss: mel L1 + postnet mel L1 + log-duration MSE + pitch MSE + energy MSE.
- Optimizer: Adam; peak learning rate 1e-3; 300-step warm-up; learning-rate halving every 3,000 steps; gradient clipping 1.0; batch size 16.
- Training steps: English 7,778; Telugu 6,016.
- Device: Tesla T4 GPU training; CPU checkpoint inference.

Vocoder: Griffin–Lim with 20 iterations on predicted log-mel spectrograms; no neural vocoder was used.

## 5. Objective evaluation

| Metric | English | Telugu |
|---|---:|---:|
| Teacher-forced postnet mel L1 | 0.9974 | 0.7519 |
| Duration MAE per token (frames / ms) | 3.00 / 34.85 | 1.99 / 23.09 |
| Predicted/true total length (mean ± SD) | 0.864 ± 0.220 | 0.949 ± 0.088 |
| Pitch RMSE per token (Hz) | 64.50 | 38.66 |
| Energy RMSE per token | 29.88 | 21.22 |
| Validation utterances | 123 | 95 |

The repository contains English validation plots, five English custom-sample plots, five Telugu validation plots, and English and Telugu training curves. The pipeline summary records five English validation samples, five English custom samples and five Telugu validation samples. It does not record five Telugu custom samples.

No Mean Opinion Score (MOS) listening test was run. Objective metrics do not directly measure perceived naturalness. Audio files and model checkpoints are not included in the public repository. README.md notes that original datasets and checkpoints are not included, and generated Telugu audio is withheld pending licence confirmation.

## 6. Limitations
- Telugu uses uroman transliteration rather than a pronunciation-aware phoneme front end. Transliteration can lose vowel length, gemination, retroflex/dental distinctions and schwa behavior.
- Training stopped at 7,778 English and 6,016 Telugu steps; only 979 English and 219 Telugu training utterances were used after the length filter. This limits model quality.
- Svarah contains multiple accented speakers, while this setup uses a single-speaker configuration.
- Griffin–Lim can introduce metallic or phase-noise artifacts; a neural vocoder such as HiFi-GAN may improve naturalness.
- Durations derive from 20 ms CTC frames and midpoint assignment within words; they are alignment-derived targets, not human-annotated ground truth.
- Punctuation is stripped, reducing prosodic cues; utterances longer than 1,000 mel frames were excluded.

## 7. Repository contents and reproducibility
The reviewed public repository contains README.md, TECHNICAL_REPORT.md, fs2_pipeline_colab.py, requirements.txt, vocab.json, metrics/ and plots/. The repository tree reviewed for this report does not contain separate preprocess.py, train.py, inference.py or config.yaml files, nor trained checkpoint files. The single pipeline script is the main implementation entry point.

Repository: https://github.com/kumarknow72-pixel/iiith-fastspeech2-english-telugu

Submission note: This report reflects the committed repository files and metrics reviewed. Confirm the assignment's required deliverables separately; this report does not claim a full 160,000-step training run, MOS evaluation, or five Telugu custom audio outputs.
