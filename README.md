# Pickleball Automatic Court Calibration

Automatic near-half court ROI detection for pickleball footage using YOLO-pose keypoint estimation and homography-based calibration.

Given a camera view of a pickleball court, the system detects 12 court keypoints, computes a homography to a canonical court model (real-world feet), and projects the near-half region of interest (net → near baseline) back into the image — locked on within ~10 seconds of video.

---

## Pipeline overview

```
Screen recordings
      │
      ▼
extract_clip_frames.py        — sample frames, letterbox to 1920×1080
      │
      ▼
pb_relabel_tool/              — run Roboflow model, manually correct keypoints in browser
      │
      ▼
train_pose.py                 — train YOLOv11-pose on the labeled dataset
      │
      ▼
calib_app/                    — web app: upload video or use camera, get locked ROI overlay
```

---

## Court keypoints (12 points)

```
  0 ——————— 1 ——————— 2      ← far baseline
  |                   |
  5 ——————— 4 ——————— 3      ← far kitchen line
  
  ═══════════════════════    ← net (y = 22 ft)
  
  6 ——————— 7 ——————— 8      ← near kitchen line
  |                   |
  11 —————— 10 ——————— 9     ← near baseline
```

Visibility: **2** = visible, **1** = occluded (position known), **0** = not labeled.

---

## Components

### `court_calib.py`
Core calibration library. Takes 12 image keypoints, fits a homography to the canonical court model (20 × 44 ft) via RANSAC, and returns:
- Near-half ROI polygon (image coordinates)
- Net line endpoints
- Re-projection error (pixels and feet)

Usable standalone — no model required if you supply keypoints from labels or predictions.

### `calib_app/`
Flask web app. Upload a video or use a webcam; the app searches for a good lock frame over a configurable window (default 10 s) and freezes the best calibration for the session.

See [`calib_app/README.md`](calib_app/README.md) for full usage and tuning options.

### `pb_relabel_tool/`
Browser-based keypoint annotation tool. Runs the Roboflow model over a folder of images, lets you drag/correct the 12 court keypoints per image, and exports a YOLO-pose dataset ready for training.

See [`pb_relabel_tool/README.md`](pb_relabel_tool/README.md) for the full workflow.

### `extract_clip_frames.py`
Samples every Nth frame from `.mov` clips, letterboxes to 1920×1080 (preserving aspect ratio), and writes a manifest CSV for traceability.

```bash
python extract_clip_frames.py --src /path/to/clips --out clips_frames --step 6
```

### `train_pose.py`
Trains a YOLOv11-pose model on the labeled court keypoint dataset.

```bash
python train_pose.py --smoke                        # 1-epoch sanity check first
python train_pose.py --model yolo11m-pose.pt        # full run (200 epochs, imgsz=1280)
python train_pose.py --model yolo11x-pose.pt --batch 16   # max accuracy
```

Output: `runs/pose/pb_court_pose/weights/best.pt`

### `validate_labels.py`
Validates YOLO-pose label files for format errors, out-of-range coordinates, and missing visibility flags. Writes a CSV report.

### `split_clip_aware.py`
Splits the dataset into train/val while keeping frames from the same source clip in the same split (prevents data leakage).

### `rename_new_frames.py` / `pb_relabel_tool/pad_rename.py`
Utility scripts for merging newly extracted frames into the master `img_XXXXX` naming sequence.

---

## Quickstart

### 1. Label and build a dataset

```bash
cd pb_relabel_tool
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python run_inference.py          # predict with Roboflow model
python app.py                    # correct keypoints in browser
python export_yolo.py            # export YOLO-pose dataset
```

### 2. Train

```bash
pip install ultralytics
python train_pose.py --smoke                  # verify pipeline
python train_pose.py                          # full training run
```

### 3. Run the calibration app

```bash
pip install flask opencv-python-headless numpy
cd calib_app

# stub mode (GT labels, no model needed):
python app_calib.py

# with trained model:
CALIB_MODEL=/path/to/best.pt python app_calib.py
```

Open `http://127.0.0.1:5000`, upload a video or use the camera, and press **Start**.

---

## Requirements

- Python 3.8+
- `opencv-python`, `numpy`
- `flask` (calib app)
- `ultralytics` (training + model inference)
- GPU recommended for training (CPU works for inference)
