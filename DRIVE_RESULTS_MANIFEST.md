# Verified Drive Results Manifest

The following inventory was visually checked in the user's shared Google Drive submission folder on 4 November 2026. This manifest records Drive presence only; it does not claim these binary files have been uploaded to this GitHub repository.

## English

- Checkpoint: `English/checkpoints/Last.pt`
- Generated audio: `E1.wav`, `E2a.wav`, `E2b.wav`, `E3.wav`, `E4.wav` (5 files)
- Variance-adaptor plots: 5 duration, 5 pitch, 5 energy images
- Training curve: `training_loss.png`
- Inference metadata: `inference_results.json`

## Telugu

- Checkpoint: `Telugu/checkpoints/Last.pt`
- Generated audio: `T1.wav`, `T2a.wav`, `T2b.wav`, `T3.wav`, `T4.wav` (5 files)
- Variance-adaptor plots: 5 duration, 5 pitch, 5 energy images
- Training curve: `training_loss.png`
- Inference metadata: `inference_results.json`

## Totals

- 2 trained checkpoints present in Drive.
- 10 generated test WAV files present in Drive.
- 30 variance-adaptor plots present in Drive.
- 2 training-loss plots present in Drive.

## Important distinctions

- This inventory confirms the files were present in the shared Drive folders inspected; it does not independently validate audio quality or model correctness.
- The Telugu `test_generated_audios` folder directly under the language model folder was empty in one inspection, but the Telugu test audio files were present under `Results/Telugu/test_generated_audios`.
- The Drive report folder contains `Technical_Report.md` and `Technical_Report.pdf`.
- No validation-loss curve was verified. The displayed plots were training-loss curves.
- The six language-specific `train.py`, `preprocess.py`, and `inference.py` files have not yet been verified or added here.
