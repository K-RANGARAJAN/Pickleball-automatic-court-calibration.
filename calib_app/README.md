# Court Calibration App — near-half ROI

Web app: a user places the camera at the recommended spot, picks **Upload video**
or **Use camera**, presses **Start**, and the court locks on within ~10 seconds.
Once locked, the near-half region of interest (net → near baseline) is overlaid.

## How it works

```
browser (upload / webcam)  --frames-->  Flask backend
                                           |
                              predict 12 keypoints   <-- the only part that needs the model
                                           |
                              court_calib.py  (RANSAC homography, tested)
                                           |
                              near-half ROI polygon + reprojection error
                                           |
browser  <--ROI + status---  (Searching / Locked / Failed)
```

**Lock-on:** during the ~10s window the server scores every frame by re-projection
error and keeps the lowest-error one (with ≥6 RANSAC inliers). If the best frame
beats the threshold (`lock_thresh_px`, default 8px) it **freezes that homography**
and reuses it for the whole session — the camera is fixed, so one good calibration
holds. If nothing beats the threshold in the window → **Failed: reposition camera**.

## Run it now (before training finishes)

The predictor is stubbed to read your **ground-truth labels**, so the entire app
is testable today on your existing images.

```bash
pip install flask opencv-python-headless numpy
cd calib_app
python app_calib.py
# open http://127.0.0.1:5000  -> Upload video -> Start
```

(For a pure stub test you can also point a short video made from your labelled
frames; the homography/ROI/lock-on logic all exercise without the model.)

## Integrate the trained model (~the one change)

When `best.pt` is ready, set one environment variable — **no code change**:

```bash
pip install ultralytics
CALIB_MODEL=/path/to/best.pt python app_calib.py
```

`predict_keypoints()` in `app_calib.py` switches from the GT-label stub to running
YOLO at imgsz=1280 automatically. That's the whole integration.

## Tuning the lock-on

In `app_calib.py` → `SESSION`:

| field            | meaning                                            | default |
|------------------|----------------------------------------------------|---------|
| `lock_window_s`  | how long to search before giving up                | 10.0    |
| `lock_thresh_px` | max re-projection error accepted as a valid lock   | 8.0     |

Your ground-truth baseline was ~3px median, p95 ~4px. Expect the trained model to
be a bit higher, especially on occluded points — start at 8px and tighten if locks
look loose, loosen if good camera placements fail to lock.

## Files

| file           | purpose                                                        |
|----------------|----------------------------------------------------------------|
| `app_calib.py` | Flask backend, lock-on logic, predictor seam                   |
| `index.html`   | browser UI: source select, Start, lock status, ROI overlay     |
| `../court_calib.py` | the tested homography / RANSAC / ROI pipeline (reused)     |

## Status / what's left

- [x] Calibration pipeline (homography + RANSAC + ROI + reprojection) — tested on 4,389 labels
- [x] Web UI (upload + camera + Start + lock status + overlay)
- [x] Backend lock-on (10s window, freeze best frame)
- [ ] Swap stub → `best.pt` once training finishes (~30h) and you confirm accuracy
- [ ] Final pass: per-frame eval on a held-out test clip, confirm reprojection error
      tracks the ground-truth baseline
