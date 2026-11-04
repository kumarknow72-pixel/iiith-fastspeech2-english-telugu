# IIITH FastSpeech 2 Assignment — Completion Tracker

Working branch: `assignment-compliant`. The `main` branch is unchanged.

## Verified in the user's shared Google Drive (4 Nov 2026)

- [x] English and Telugu trained checkpoints present in Drive.
- [x] Five assignment test WAVs per language (E1, E2a, E2b, E3, E4; T1, T2a, T2b, T3, T4).
- [x] Per language: five duration, five pitch, and five energy plots (30 total).
- [x] Training-loss plot for each language.
- [x] Language config files and inference-results JSON files visible in the Drive results structure.
- [x] Technical_Report.md and Technical_Report.pdf present in Drive.
- [x] Drive inventory documented in `DRIVE_RESULTS_MANIFEST.md`.

## Added to this GitHub branch

- [x] Existing shared Colab pipeline `fs2_pipeline_colab.py` now accepts environment switches for preprocessing, training, and synthesis.
- [x] English and Telugu folders each contain `preprocess.py`, `train.py`, `inference.py`, and `config.yaml`.
- [x] README documents the entry points and their limitations.

## Important code limitation — do not overclaim

The six Python files are **thin stage launchers**, not a complete refactor into two independent language-specific implementations. The existing pipeline uses a combined English+Telugu vocabulary and loops over both languages; therefore either language launcher's execution operates on the paired pipeline. Inference launchers run the pipeline's built-in validation/custom synthesis. The assignment-specific E1–E4 and T1–T4 outputs were verified in Drive, not created by these launchers.

## Still outstanding / verify before final submission

- [ ] Refactor and test truly independent per-language preprocessing/training/inference modules if the evaluator strictly requires language-isolated scripts.
- [ ] Add the final report files to GitHub only after reviewing contents and ensuring redistribution is permitted.
- [ ] Confirm exact train/validation audio hours and split independence from source manifests/logs.
- [ ] A genuine validation-loss curve was not verified. Do not label the training-loss curve as validation loss.
- [ ] Verify dataset, audio, and checkpoint redistribution permissions before publishing binary artifacts.

## Safety and reproducibility

- The Drive manifest confirms file presence only, not audio quality or independent model validation.
- Do not upload datasets, audio, or checkpoints without confirming licences/redistribution terms.
- Check the Colab Drive paths and dependencies before running. Training launchers resume training and should only be run intentionally.
