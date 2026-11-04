# IIITH Voice AI Assignment — FastSpeech2 English and Telugu

Two separate FastSpeech2-style TTS models:
- English: Svarah
- Telugu: IndicTTS Telugu

## Project structure
- English/: English model files and checkpoint
- Telugu/: Telugu model files and checkpoint
- Results/English/: English generated test audios and plots
- Results/Telugu/: Telugu generated test audios and plots
- Report/: Technical report

## Audio
Five assignment test utterances were synthesized for each language.

## Important
The existing checkpoints were reused for inference. No retraining was performed during final test synthesis.
The generated audio uses Griffin-Lim reconstruction.
See the technical report for methodology, metrics and limitations.
