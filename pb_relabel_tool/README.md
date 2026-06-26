# Pickleball court keypoint relabel tool

Run the Roboflow model `pb-9bsin/4` over a folder of images, correct the predicted
court keypoints by hand in your browser, and export a training-ready **YOLO‑pose**
dataset.

```
run_inference.py   ->  app.py (browser editor)  ->  export_yolo.py
   (predict)              (drag / fix points)          (train-ready dataset)
```

The 12 keypoints (model order):

| idx | name | idx | name |
|----|------|----|------|
| 0 | background baseline left   | 6  | foreground kitchen-line left  |
| 1 | background baseline mid    | 7  | foreground kitchen-line mid   |
| 2 | background baseline right  | 8  | foreground kitchen-line right |
| 3 | background kitchen-line right | 9  | foreground baseline right  |
| 4 | background kitchen-line mid   | 10 | foreground baseline mid    |
| 5 | background kitchen-line left  | 11 | foreground baseline left   |

Visibility per point: **2 = visible**, **1 = occluded** (position known, hidden),
**0 = not labeled** (ignored during training).

---

## 1. Setup (once)

Requires Python 3.8+ (works on 3.13 — inference is a plain HTTP call, no SDK).
The bundled `venv/` is a Windows environment; on macOS/Linux create a fresh one:

```bash
cd pb_relabel_tool
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Your Roboflow key / model are already in `config.json`. Edit if needed.

## 2. Put your images in place

Default location is a folder named `images` next to `config.json`:

```
model_run_relabel/
  pb_relabel_tool/
    config.json, run_inference.py, app.py, export_yolo.py, ...
  images/                <-- drop your .jpg/.png here
```

Or point at any folder: add `--images "D:\path\to\images"` to every command,
or set `"images_dir"` in `config.json` to an absolute path.

## 3. Run inference

```bash
python run_inference.py
```

This writes, inside `work_dir` (default `pb_relabel/`):

- `annotations.json` — the editable source of truth (12 points + visibility per image)
- `raw_predictions/<image>.json` — the raw Roboflow response (backup)
- `labels/<image>.txt` — initial YOLO‑pose labels from the model

Handy flags: `--limit 20` (quick test), `--only-new`, `--overwrite`.

## 4. Correct the points (browser editor)

```bash
python app.py
```

A browser tab opens at `http://127.0.0.1:5000`. For each image:

- **Drag** any point to its correct spot. Dragging a grey (not‑labeled) point
  marks it visible automatically.
- **Click a point**, then press **V** (or use the right‑hand list) to cycle its
  visibility: visible → occluded → not‑labeled.
- **Wheel** to zoom toward the cursor, **drag empty space** to pan, **F** to fit.
- **← / →** move between images (auto‑saves). **✓ Reviewed + Next** marks the
  image done and advances. Press number keys **0–9** to select a point.

Every save updates `annotations.json` **and** the YOLO `.txt` for that image, so
you can stop and resume anytime. The left list shows progress
(green = reviewed, orange = no detection).

> Nothing leaves your machine — the editor is a local server that only reads and
> writes files under `work_dir`. The internet is only used in step 3 (inference).

## 5. Export the training dataset

```bash
python export_yolo.py                 # everything with a usable label
python export_yolo.py --reviewed-only # only images you marked reviewed
```

Creates `pb_relabel/dataset/`:

```
dataset/
  images/train  images/val
  labels/train  labels/val
  data.yaml          # nc=1 (court), kpt_shape [12,3], flip_idx, names
```

Train (Ultralytics):

```bash
pip install ultralytics
yolo pose train data="…/pb_relabel/dataset/data.yaml" model=yolo11n-pose.pt epochs=100 imgsz=1280
```

`flip_idx` is set so horizontal‑flip augmentation correctly swaps left/right
keypoints.

---

## Files

| file | purpose |
|------|---------|
| `config.json` | API key, model id, folder paths, val split |
| `common.py` | keypoint defs, parsing, YOLO conversion (shared) |
| `run_inference.py` | batch predict → annotations + initial labels |
| `app.py` + `templates/index.html` | the browser correction editor |
| `export_yolo.py` | build train/val YOLO‑pose dataset + `data.yaml` |

## Notes & tips

- **Bounding box is automatic** — it's computed to enclose your labeled points,
  so you only ever move the 12 keypoints.
- **Resume / re-run:** `run_inference.py` keeps images you've already marked
  reviewed (use `--overwrite` to force re-prediction).
- **No detection on an image?** It still appears in the editor with 12 grey
  placeholder points — just drag them into place.
- **Windows symlinks:** `export_yolo.py` symlinks images by default and falls
  back to copying automatically; add `--copy` to always copy.
