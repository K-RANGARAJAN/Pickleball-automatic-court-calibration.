#!/usr/bin/env bash
# Launch the court-calibration app using your TRAINED model (epoch65.pt).
# macOS/Linux:  bash run_with_model.sh
#
# CALIB_MODEL takes priority over Roboflow, so this uses your local YOLO weights.
# Edit the paths/port below if they differ.

export CALIB_MODEL="${CALIB_MODEL:-../epoch65.pt}"   # weights one level up in model_run_relabel/
export PORT="${PORT:-5001}"

# export HTTPS=1   # uncomment for phone camera over LAN without a tunnel

echo "Starting court-calibration app with model: $CALIB_MODEL"
python app_calib.py
