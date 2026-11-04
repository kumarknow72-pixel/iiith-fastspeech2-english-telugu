# IIITH FastSpeech 2 Assignment — Completion Tracker

This is the working submission branch. The `main` branch is intentionally left unchanged.

## Verified in the user's shared Google Drive submission folder (4 Nov 2026)

- [x] English checkpoint exists: `English/checkpoints/Last.pt`.
- [x] Telugu checkpoint exists: `Telugu/checkpoints/Last.pt`.
- [x] English test audio: E1, E2a, E2b, E3, E4 (5 WAV files).
- [x] Telugu test audio: T1, T2a, T2b, T3, T4 (5 WAV files).
- [x] English variance-adaptor plots: 5 duration + 5 pitch + 5 energy.
- [x] Telugu variance-adaptor plots: 5 duration + 5 pitch + 5 energy.
- [x] English and Telugu training-loss images are present.
- [x] Both language configuration files and inference-results JSON files were visible in the Drive results structure.
- [x] README, requirements file, and Technical_Report.pdf/Technical_Report.md are present in the Drive submission folder.

See [DRIVE_RESULTS_MANIFEST.md](DRIVE_RESULTS_MANIFEST.md) for the verified Drive inventory.

## Still to complete or verify in this GitHub branch

- [ ] Add and review separate language-specific `preprocess.py`, `train.py`, and `inference.py` entry points. The existing `fs2_pipeline_colab.py` is a monolithic Colab pipeline; it is not six separate entry points.
- [ ] Add the final technical report (Markdown/PDF) to this repository after checking the report contents and redistribution constraints.
- [ ] Decide whether to publish checkpoints/audio/plots. Checkpoint files are large; generated audio and checkpoints must not be redistributed until the relevant dataset/model terms are confirmed. The Drive inventory is not the same as a GitHub upload.
- [ ] Add a genuine validation-loss curve only if per-step validation-loss history exists. Do not relabel training loss as validation loss.
- [ ] Verify exact train/validation audio hours, split independence, and actual examples used from original logs before making precise dataset-hour claims.

## Reproducibility and integrity notes

- Do not claim that the six separate entry-point files exist until they are added and tested.
- Do not claim that Drive files are hosted in this repository unless the corresponding GitHub paths are visible.
- Do not upload source datasets or copyrighted audio/checkpoints without confirming redistribution permission.
- Keep `main` unchanged until this branch has been reviewed.
