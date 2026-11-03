# IIITH FastSpeech 2 Assignment — Completion Tracker

This branch is a safe working branch for bringing the existing project in line with the assignment. The `main` branch is unchanged.

## Verified repository gaps to close

- [ ] Separate English and Telugu entry points: `preprocess.py`, `train.py`, `inference.py`, and language-specific configuration.
- [ ] Preserve the existing trained checkpoints and record their exact paths and hashes.
- [ ] Add validation-loss tracking during training and export validation-loss plots.
- [ ] Generate audio for every exact English and Telugu assignment test sentence.
- [ ] Generate per-test duration, pitch, and energy plots for both languages.
- [ ] Verify actual train/validation audio hours, split independence, and the number of examples actually used by each training run.
- [ ] Reconcile the technical report with the actual code and run logs.
- [ ] Produce the requested PDF report and final README with reproducible commands.
- [ ] Confirm dataset and checkpoint redistribution permissions before publishing those artifacts.

## Important reproducibility notes

- Do not claim that a model was retrained, a test utterance was synthesized, or a plot was generated unless the corresponding output exists and has been checked.
- Do not upload source datasets or WAV files unless their licences permit redistribution.
- Keep the original `main` branch intact until the assignment branch has been reviewed.

## Work that requires the user's Colab/Drive environment

The current repository connection does not expose the user's Google Drive datasets or active Colab runtime. Therefore, the following must be run in the connected Colab environment: dataset-hour verification, any required retraining, assignment-sentence synthesis, and export of genuine loss curves, audio, and variance-adaptor plots.
