# FastSpeech2 English and Telugu TTS Project


---

## Dataset Acknowledgements and Attribution

We gratefully acknowledge the following datasets and their creators.

### 1. English Dataset — Svarah

- **Dataset:** Svarah
- **Provider:** AI4Bharat
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Licence:** https://creativecommons.org/licenses/by/4.0/

The Svarah dataset was used for English text-to-speech model training and evaluation.

Users redistributing material covered by this licence must provide appropriate attribution, a link to the licence, and indicate whether changes were made.

### 2. Telugu Dataset — IndicTTS

- **Dataset:** IndicTTS Telugu
- **Provider:** IIT Madras
- **Licence:** Subject to the original IndicTTS licence.

We acknowledge IIT Madras for providing the IndicTTS resource used in our Telugu text-to-speech experiments.

Please consult the original dataset licence before redistributing dataset material or generated audio.

### 3. Project Disclaimer

This repository contains our implementation, experimental results, evaluation metrics, visualizations and technical documentation.

- Original datasets are not included.
- Trained model checkpoints are not included.
- Generated English model-output WAV samples are included under `samples/english/` (five validation-text samples and five custom-text samples).
- Generated Telugu audio samples are withheld pending confirmation of redistribution permissions.

This is an independent academic experimental project and is not an official product of AI4Bharat or IIT Madras.

## Project overview

This repository contains a reproducible experimental FastSpeech2 pipeline for English (Svarah) and Telugu (IndicTTS, romanized text), together with evaluation metrics, plots and a technical report.

### Repository contents

- `fs2_pipeline_colab.py` — consolidated Google Colab pipeline for feature preparation, validation, training, evaluation, synthesis and plots.
- `TECHNICAL_REPORT.md` — methods, dataset filtering, model configuration, objective metrics and limitations.
- `metrics/` — committed objective evaluation outputs.
- `plots/` — committed training and validation visualizations.
- `samples/english/` — five generated validation-text WAV samples and five generated custom-text WAV samples.
- `vocab.json` and `requirements.txt` — vocabulary and Python dependencies.

## Reproducing the experiments

1. Open `fs2_pipeline_colab.py` in Google Colab.
2. Mount Google Drive and make sure the project data, alignment JSON files and metadata files are available at the paths configured near the top of the script.
3. Review the configuration block before running. The default settings do **not** start training or synthesis:
   - `RUN_FEATURES = True`
   - `RUN_TRAIN = False`
   - `RUN_SYNTH = False`
4. To train, set `RUN_TRAIN = True` and run the pipeline with a GPU runtime. Training resumes from the language-specific checkpoint when one exists.
5. To synthesize after both language checkpoints have been created, set `RUN_FEATURES = False`, `RUN_TRAIN = False`, and `RUN_SYNTH = True`. This synthesis stage expects checkpoints at `module4_final/checkpoints/english/last.pt` and `module4_final/checkpoints/telugu/last.pt` under the configured Drive project directory.
6. The script writes generated WAV files and plots under `module4_final/samples/` and `module4_final/plots/` in Google Drive. Review the outputs by listening before sharing them.

The synthesis code creates five validation-text samples per language. It additionally creates five English custom-text samples. Do not describe those extra English samples as Telugu samples.

## Current artifact status

The committed repository contains code, metrics, plots, documentation and 10 generated English WAV samples under `samples/english/` (five validation-text samples and five custom-text samples). Trained model checkpoint files are not included.

Generated Telugu audio is not redistributed here until the applicable IndicTTS terms have been checked. Do not upload source dataset audio or other dataset material unless its licence explicitly permits redistribution. The English files are model-generated outputs, not original Svarah recordings. Their presence does not constitute a subjective quality or MOS evaluation.

## Important limitations

- Telugu text is romanized; this is not a pronunciation-aware Telugu phoneme frontend.
- The Telugu training set was substantially reduced by the configured utterance-length filter.
- Griffin–Lim is used for waveform reconstruction; no neural vocoder or MOS listening study is reported.
- The technical report documents the actual training steps, evaluation metrics and known limitations. See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for details.

## Dataset acknowledgements and attribution

### English — Svarah
- Provider: AI4Bharat
- Licence: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

The Svarah dataset was used for English text-to-speech training and evaluation. Provide appropriate attribution and indicate changes when redistributing material covered by CC BY 4.0.

### Telugu — IndicTTS
- Provider: IIT Madras
- Licence: Subject to the original IndicTTS terms.

Consult the original dataset licence before redistributing dataset material or generated audio.

## Disclaimer

This is an independent academic experimental project. It is not an official product of AI4Bharat, IIT Madras or IIIT Hyderabad. Dataset names and institutional acknowledgements do not imply endorsement.
