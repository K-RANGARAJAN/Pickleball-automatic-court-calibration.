"""
Shared constants and helpers for the pickleball court keypoint relabel tool.

Keypoint order (class_id -> name), exactly as defined by the Roboflow model pb-9bsin/4:
    0  background baseline left
    1  background baseline mid
    2  background baseline right
    3  background kitchen-line right
    4  background kitchen-line mid
    5  background kitchen-line left
    6  foreground kitchen-line left
    7  foreground kitchen-line mid
    8  foreground kitchen-line right
    9  foreground baseline right
    10 foreground baseline mid
    11 foreground baseline left

Visibility convention (YOLO-pose / COCO style):
    2 = labeled and visible
    1 = labeled but occluded / not visible (position still known)
    0 = not labeled (position unknown -> ignored in training)
"""

import base64
import json
import os

# ---------------------------------------------------------------------------
# Keypoint definitions
# ---------------------------------------------------------------------------
KPT_NAMES = [
    "background baseline left",       # 0
    "background baseline mid",        # 1
    "background baseline right",      # 2
    "background kitchen-line right",  # 3
    "background kitchen-line mid",    # 4
    "background kitchen-line left",   # 5
    "foreground kitchen-line left",   # 6
    "foreground kitchen-line mid",    # 7
    "foreground kitchen-line right",  # 8
    "foreground baseline right",      # 9
    "foreground baseline mid",        # 10
    "foreground baseline left",       # 11
]
NUM_KPTS = len(KPT_NAMES)

# Left<->right mirror mapping for YOLO 'flip_idx' (horizontal-flip augmentation).
# left<->right swap, mid stays, foreground/background unchanged.
FLIP_IDX = [2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9]

# Court wireframe edges (pairs of keypoint indices) - for drawing context only.
SKELETON = [
    # background baseline (far):  0-1-2
    [0, 1], [1, 2],
    # background kitchen line:  5-4-3
    [5, 4], [4, 3],
    # foreground kitchen line:  6-7-8
    [6, 7], [7, 8],
    # foreground baseline (near):  11-10-9
    [11, 10], [10, 9],
    # left sideline:  0 -> 5 -> 6 -> 11
    [0, 5], [5, 6], [6, 11],
    # right sideline: 2 -> 3 -> 8 -> 9
    [2, 3], [3, 8], [8, 9],
    # center service lines (no center line through the kitchen):
    [1, 4], [7, 10],
]

# 12 visually distinct colours (hex) for the editor overlay.
COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#9A6324",
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "api_url": "https://serverless.roboflow.com",
    "api_key": "GD2Vh60x6e9XGMNStpkD",
    "model_id": "pb-9bsin/4",
    "images_dir": "images",       # relative to the config file's folder (or absolute)
    "work_dir": "pb_relabel",     # where outputs are written
    "confidence_threshold": 0.0,  # keypoints below this -> kept but flagged occluded; 0 = keep all visible
    "bbox_pad_frac": 0.02,        # padding added around keypoint extent for the YOLO box
    "val_split": 0.2,             # fraction of images used for validation on export
    "seed": 42,
}


def load_config(config_path):
    """Load config.json, filling any missing keys with defaults."""
    cfg = dict(DEFAULT_CONFIG)
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def resolve_dirs(cfg, base_dir):
    """Resolve images_dir and work_dir to absolute paths relative to base_dir."""
    def _abs(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))
    return _abs(cfg["images_dir"]), _abs(cfg["work_dir"])


def work_paths(work_dir):
    """Standard sub-paths inside the working directory."""
    return {
        "annotations": os.path.join(work_dir, "annotations.json"),
        "raw": os.path.join(work_dir, "raw_predictions"),
        "labels": os.path.join(work_dir, "labels"),
        "dataset": os.path.join(work_dir, "dataset"),
    }


# ---------------------------------------------------------------------------
# Roboflow hosted inference over plain HTTP (no inference-sdk, so any Python 3.x)
# ---------------------------------------------------------------------------
RETRY_STATUS = {429, 500, 502, 503, 504}


def infer_image(session, image_path, api_url, api_key, model_id, timeout=60,
                retries=4, backoff=2.0):
    """POST one image (base64) to the hosted Roboflow model; return parsed JSON.

    Retries transient server errors (429/5xx) with exponential backoff — the free
    serverless tier returns 503 under load. Raises only after the last attempt fails so
    the caller can count a genuine failure.
    """
    import time

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    url = f"{api_url}/{model_id}"
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = session.post(url, params={"api_key": api_key}, data=b64,
                                headers={"Content-Type": "application/x-www-form-urlencoded"},
                                timeout=timeout)
            if resp.status_code in RETRY_STATUS and attempt < retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001 - connection blips are also retried
            last_exc = e
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise
    raise last_exc


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------
def list_images(images_dir):
    if not os.path.isdir(images_dir):
        return []
    files = [f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS)]
    files.sort()
    return files


def image_size(path):
    """Return (width, height) using Pillow."""
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (w, h)


# ---------------------------------------------------------------------------
# Annotations.json IO  (source of truth: pixel coords + visibility)
# ---------------------------------------------------------------------------
# {
#   "model_id": "...",
#   "kpt_names": [...],
#   "images": {
#       "<filename>": {
#           "width": W, "height": H, "reviewed": false,
#           "keypoints": [ {"x": px, "y": py, "v": 2}, ... 12 ... ]
#       }, ...
#   }
# }

def empty_keypoints(width, height):
    """12 placeholder points on a grid, all marked not-labeled (v=0)."""
    pts = []
    for i in range(NUM_KPTS):
        col = i % 3
        row = i // 3
        x = width * (0.25 + 0.25 * col)
        y = height * (0.2 + 0.2 * row)
        pts.append({"x": round(x, 2), "y": round(y, 2), "v": 0})
    return pts


def load_annotations(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"model_id": "", "kpt_names": KPT_NAMES, "images": {}}


def save_annotations(ann, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Roboflow keypoint-prediction parsing  (defensive)
# ---------------------------------------------------------------------------
def parse_roboflow_result(result, width, height, conf_thresh=0.0):
    """
    Convert a Roboflow keypoint-detection result into a list of 12 keypoints
    [{x,y,v}, ...] in PIXEL coordinates, ordered by class_id 0..11.

    Handles: multiple detections, out-of-order keypoints, name-keyed keypoints,
    normalized coordinates, and missing keypoints (left as v=0 placeholders).
    Returns (keypoints, found_any).
    """
    kpts = empty_keypoints(width, height)
    found = [False] * NUM_KPTS

    preds = (result or {}).get("predictions") or []
    kp_preds = [p for p in preds if isinstance(p, dict) and p.get("keypoints")]
    if not kp_preds:
        return kpts, False

    best = max(kp_preds, key=lambda p: p.get("confidence", 0.0))
    raw_kps = best.get("keypoints", [])

    # Detect normalized coordinates (all values <= ~1.5)
    max_coord = 0.0
    for k in raw_kps:
        max_coord = max(max_coord, abs(k.get("x", 0)), abs(k.get("y", 0)))
    normalized = 0 < max_coord <= 1.5

    name_to_idx = {n.lower(): i for i, n in enumerate(KPT_NAMES)}

    for order, k in enumerate(raw_kps):
        idx = k.get("class_id", None)
        if idx is None or not (0 <= int(idx) < NUM_KPTS):
            cname = str(k.get("class", "")).lower().strip()
            idx = name_to_idx.get(cname, order if order < NUM_KPTS else None)
        if idx is None:
            continue
        idx = int(idx)
        x = float(k.get("x", 0.0))
        y = float(k.get("y", 0.0))
        if normalized:
            x *= width
            y *= height
        conf = float(k.get("confidence", 1.0))
        v = 2 if conf >= conf_thresh else 1
        kpts[idx] = {"x": round(x, 2), "y": round(y, 2), "v": v}
        found[idx] = True

    return kpts, any(found)


# ---------------------------------------------------------------------------
# Bounding box + YOLO-pose label generation
# ---------------------------------------------------------------------------
def compute_bbox(keypoints, width, height, pad_frac=0.02):
    """
    Axis-aligned bbox (pixels) enclosing all labeled (v>=1) keypoints, with
    fractional padding, clamped to the image. Returns (xmin,ymin,xmax,ymax)
    or None if fewer than 2 labeled points.
    """
    xs, ys = [], []
    for k in keypoints:
        if k.get("v", 0) >= 1:
            xs.append(k["x"])
            ys.append(k["y"])
    if len(xs) < 2:
        return None
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad_x = (xmax - xmin) * pad_frac + width * 0.005
    pad_y = (ymax - ymin) * pad_frac + height * 0.005
    xmin = max(0.0, xmin - pad_x)
    ymin = max(0.0, ymin - pad_y)
    xmax = min(float(width), xmax + pad_x)
    ymax = min(float(height), ymax + pad_y)
    return (xmin, ymin, xmax, ymax)


def keypoints_to_yolo_line(keypoints, width, height, pad_frac=0.02, class_id=0):
    """
    Build one YOLO-pose label line (normalized):
        class cx cy w h  px0 py0 v0 ... px11 py11 v11
    Not-labeled keypoints (v=0) -> '0 0 0' (Ultralytics convention).
    Returns None when there is no usable bounding box.
    """
    bbox = compute_bbox(keypoints, width, height, pad_frac)
    if bbox is None:
        return None
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2.0 / width
    cy = (ymin + ymax) / 2.0 / height
    bw = (xmax - xmin) / width
    bh = (ymax - ymin) / height

    def c01(v):
        return min(1.0, max(0.0, v))

    parts = [str(class_id),
             f"{c01(cx):.6f}", f"{c01(cy):.6f}",
             f"{c01(bw):.6f}", f"{c01(bh):.6f}"]
    for k in keypoints:
        v = int(k.get("v", 0))
        if v == 0:
            parts += ["0.000000", "0.000000", "0"]
        else:
            parts += [f"{c01(k['x'] / width):.6f}", f"{c01(k['y'] / height):.6f}", str(v)]
    return " ".join(parts)


def write_yolo_label(keypoints, width, height, out_txt_path, pad_frac=0.02):
    """Write (or clear) the YOLO-pose .txt label for one image. Returns True if a box was written."""
    line = keypoints_to_yolo_line(keypoints, width, height, pad_frac)
    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        if line:
            f.write(line + "\n")
    return line is not None


def label_name_for(image_filename):
    """foo.jpg -> foo.txt"""
    return os.path.splitext(image_filename)[0] + ".txt"
