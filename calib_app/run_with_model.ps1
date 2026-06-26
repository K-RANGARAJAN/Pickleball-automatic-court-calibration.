# Launch the court-calibration app using your TRAINED model (epoch65.pt).
# Windows PowerShell:  ./run_with_model.ps1
#
# CALIB_MODEL takes priority over Roboflow, so this uses your local YOLO weights.
# Edit the two values below if your paths/port differ.

$env:CALIB_MODEL = "D:\Pickleball_dataset\model_run_relabel\epoch65.pt"
$env:PORT = "5001"

# (optional) HTTPS for phone camera over LAN — uncomment if not using a tunnel:
# $env:HTTPS = "1"

Write-Host "Starting court-calibration app with model: $env:CALIB_MODEL"
python app_calib.py
