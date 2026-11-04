"""Telugu entry point for the shared FastSpeech2 Colab pipeline.

IMPORTANT: English and Telugu checkpoints share a combined vocabulary. This launcher
therefore executes the paired pipeline for BOTH languages, not a language-isolated run.
See README.md before running.
"""
from pathlib import Path
import os
import runpy

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "fs2_pipeline_colab.py"
if not PIPELINE.exists():
    raise FileNotFoundError(f"Missing shared pipeline: {PIPELINE}")

os.environ["FS2_RUN_FEATURES"] = "0"
os.environ["FS2_RUN_TRAIN"] = "1"
os.environ["FS2_RUN_SYNTH"] = "0"

print("Requested entry point: Telugu / train")
print("Pipeline scope: paired English + Telugu (shared vocabulary/checkpoints).")
try:
    runpy.run_path(str(PIPELINE), run_name="__main__")
except SystemExit as exc:
    # Preprocessing-only mode intentionally exits after validating cached features.
    if "train" == "preprocess" and isinstance(exc.code, str):
        print("Preprocessing stage finished; review validation output above.")
    else:
        raise
