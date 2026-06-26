"""
Court calibration web app — backend.

Pipeline:  frame -> predict 12 keypoints -> court_calib (RANSAC homography) ->
           near-half ROI + reprojection error.

Lock-on:   client sends frames during a ~10s window; server scores each frame by
           reprojection error and keeps the BEST (lowest-error, enough inliers).
           When a good-enough frame is found, the homography is FROZEN and reused
           for the rest of the session (camera is fixed at the recommended spot).

The model is the ONLY part waiting on training. Until best.pt exists, PREDICTOR
reads ground-truth labels so the whole UI is testable today. Swap one function.

Run:
    pip install flask opencv-python-headless numpy ultralytics
    python app_calib.py
    open http://127.0.0.1:5000
"""

import base64
import os
import sys
import time

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))                    # court_calib.py
sys.path.insert(0, os.path.join(_here, "..", "pb_relabel_tool")) # common.py
import court_calib as cc
import common as C   # tested Roboflow inference + keypoint parsing

# --------------------------------------------------------------------------- #
# PREDICTOR — the integration seam.  Three modes, picked by env vars:
#   CALIB_MODEL   = path/to/best.pt    -> local YOLO  (final, after training)
#   ROBOFLOW_KEY  = <api key>          -> hosted Roboflow pb-9bsin/4 (temporary, live now)
#   (neither)                          -> ground-truth label stub (offline testing)
# --------------------------------------------------------------------------- #
USE_MODEL    = os.environ.get("CALIB_MODEL", "")
ROBOFLOW_KEY = os.environ.get("ROBOFLOW_KEY", "")
ROBOFLOW_ID  = os.environ.get("ROBOFLOW_MODEL", "pb-9bsin/4")
ROBOFLOW_URL = os.environ.get("ROBOFLOW_URL", "https://serverless.roboflow.com")

_model = None
_rf_session = None


def _load_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(USE_MODEL)
    return _model


def _rf():
    global _rf_session
    if _rf_session is None:
        import requests
        _rf_session = requests.Session()
    return _rf_session


def predict_keypoints(frame_bgr, gt_label_path=None):
    """
    Return list of 12 (x, y, v) for a frame.

    Priority: local YOLO (CALIB_MODEL) > Roboflow (ROBOFLOW_KEY) > GT-label stub.

    >>> SWAP POINT: when best.pt is ready, set CALIB_MODEL=path/to/best.pt and the
        app uses the local model with no other change (it takes priority). <<<
    """
    if USE_MODEL:
        r = _load_model()(frame_bgr, imgsz=1280, verbose=False)[0]
        return cc.model_points_from_result(r)

    if ROBOFLOW_KEY:
        h, w = frame_bgr.shape[:2]
        # write frame to a temp jpg, send to hosted model, parse via tested common.py
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tmp = tf.name
        try:
            cv2.imwrite(tmp, frame_bgr)
            result = C.infer_image(_rf(), tmp, ROBOFLOW_URL, ROBOFLOW_KEY, ROBOFLOW_ID)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        api_img = (result or {}).get("image") or {}
        W = int(api_img.get("width") or w)
        H = int(api_img.get("height") or h)
        kpts, found = C.parse_roboflow_result(result, W, H, 0.0)
        if not found:
            raise RuntimeError("roboflow returned no keypoints")
        # common.py gives [{x,y,v}, ...]; court_calib wants (x,y,v) tuples
        return [(k["x"], k["y"], k["v"]) for k in kpts]

    if gt_label_path and os.path.exists(gt_label_path):
        h, w = frame_bgr.shape[:2]
        return cc.load_label_points(gt_label_path, w, h)

    raise RuntimeError("no predictor configured (set CALIB_MODEL or ROBOFLOW_KEY, "
                       "or pass a GT label)")


# --------------------------------------------------------------------------- #
# Session state (single-session for now)
# --------------------------------------------------------------------------- #
SESSION = {
    "locked": False,
    "H_inv": None,
    "poly": None,
    "best_err": float("inf"),
    "lock_started": None,
    "lock_window_s": 10.0,
    "lock_thresh_px": 8.0,    # accept a lock only if best reprojection error < this
}


def reset_session():
    SESSION.update(locked=False, H_inv=None, poly=None,
                   best_err=float("inf"), lock_started=None)


def _decode(data_url):
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


app = Flask(__name__)            # no static_folder mapping at root (it caused a 403 on /)
_APP_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return send_from_directory(_APP_DIR, "index.html")


@app.route("/api/start", methods=["POST"])
def api_start():
    reset_session()
    SESSION["lock_started"] = time.time()
    return jsonify({"ok": True, "window_s": SESSION["lock_window_s"]})


@app.route("/api/frame", methods=["POST"])
def api_frame():
    """
    Receive one frame during lock-on (or after).
    body: { image: dataURL, gt_label?: path }   gt_label only used in stub mode.
    Returns lock status + ROI polygon + error so the client can draw + show state.
    """
    data = request.get_json(force=True)
    frame = _decode(data["image"])
    gt = data.get("gt_label")

    # already locked -> just return the frozen ROI (cheap; no re-inference needed)
    if SESSION["locked"]:
        return jsonify({"locked": True, "polygon": SESSION["poly"],
                        "reproj_px": round(SESSION["best_err"], 2), "status": "locked"})

    # within lock window: score this frame
    elapsed = time.time() - (SESSION["lock_started"] or time.time())
    try:
        kps = predict_keypoints(frame, gt)
        H, H_inv, inl, used = cc.compute_homography(kps)
        m = cc.reprojection_error(kps, H, H_inv, used, inl)
        err = m["mean_px"]
        n_in = int(inl.sum())
        # DIAGNOSTIC: show what each frame produced
        print(f"  [frame] reproj={err:.1f}px  inliers={n_in}/{len(used)}  "
              f"(lock if <{SESSION['lock_thresh_px']}px & inliers>=6)", flush=True)
        if err < SESSION["best_err"] and n_in >= 6:
            SESSION["best_err"] = err
            SESSION["H_inv"] = H_inv.tolist()
            SESSION["poly"] = cc.near_half_polygon(H_inv).tolist()
    except Exception as e:
        err = None
        print(f"  [frame] PREDICT/HOMOGRAPHY FAILED: {repr(e)[:140]}", flush=True)

    # decide: lock, keep searching, or fail
    window_done = elapsed >= SESSION["lock_window_s"]
    if SESSION["best_err"] < SESSION["lock_thresh_px"] and (window_done or SESSION["best_err"] < 4.0):
        SESSION["locked"] = True
        return jsonify({"locked": True, "polygon": SESSION["poly"],
                        "reproj_px": round(SESSION["best_err"], 2), "status": "locked"})
    if window_done:
        return jsonify({"locked": False, "status": "failed",
                        "best_px": (round(SESSION["best_err"], 2)
                                    if SESSION["best_err"] < 1e9 else None),
                        "message": "Could not lock on. Reposition camera to the recommended spot."})
    return jsonify({"locked": False, "status": "searching",
                    "elapsed": round(elapsed, 1),
                    "best_px": (round(SESSION["best_err"], 2)
                                if SESSION["best_err"] < 1e9 else None),
                    "preview_polygon": SESSION["poly"]})


@app.route("/api/predict_frame", methods=["POST"])
def api_predict_frame():
    """
    Manual-adjust step 1: predict the 12 keypoints for ONE frame (no locking).
    body: { image: dataURL }
    Returns the points so the client can show them as draggable dots.
    """
    data = request.get_json(force=True)
    frame = _decode(data["image"])
    try:
        kps = predict_keypoints(frame, data.get("gt_label"))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 200
    # also send what the homography would give as-is, for an immediate preview
    pts = [[round(float(x), 1), round(float(y), 1), int(v)] for (x, y, v) in kps]
    out = {"ok": True, "points": pts}
    try:
        tup = [(p[0], p[1], p[2]) for p in pts]
        H, H_inv, inl, used = cc.compute_homography(tup)
        m = cc.reprojection_error(tup, H, H_inv, used, inl)
        out["polygon"] = cc.near_half_polygon(H_inv).tolist()
        out["reproj_px"] = round(m["mean_px"], 2)
    except Exception:
        out["polygon"] = None
    return jsonify(out)


@app.route("/api/confirm_points", methods=["POST"])
def api_confirm_points():
    """
    Manual-adjust step 2: take the user's (corrected) 12 points, compute and
    FREEZE the homography, return the ROI. The session is then locked.
    body: { points: [[x,y,v], ...12] }
    """
    data = request.get_json(force=True)
    pts = data.get("points") or []
    if len(pts) != 12:
        return jsonify({"ok": False, "error": "expected 12 points"}), 200
    tup = [(float(p[0]), float(p[1]), int(p[2]) if len(p) > 2 else 2) for p in pts]
    try:
        H, H_inv, inl, used = cc.compute_homography(tup)
        m = cc.reprojection_error(tup, H, H_inv, used, inl)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 200
    SESSION["locked"] = True
    SESSION["H_inv"] = H_inv.tolist()
    SESSION["poly"] = cc.near_half_polygon(H_inv).tolist()
    SESSION["best_err"] = m["mean_px"]
    return jsonify({"ok": True, "locked": True, "polygon": SESSION["poly"],
                    "reproj_px": round(m["mean_px"], 2)})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    reset_session()
    return jsonify({"ok": True})


if __name__ == "__main__":
    if USE_MODEL:
        mode = f"LOCAL YOLO ({USE_MODEL})"
    elif ROBOFLOW_KEY:
        mode = f"ROBOFLOW ({ROBOFLOW_ID})"
    else:
        mode = "STUB (ground-truth labels)"
    port = int(os.environ.get("PORT", "5000"))
    # phones require HTTPS for camera access. HTTPS=1 -> serve with a self-signed cert,
    # bind 0.0.0.0 so a phone on the same wifi can reach it.
    use_https = os.environ.get("HTTPS", "") == "1"

    # find this machine's LAN IP to print a phone-openable URL
    import socket
    lan_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    scheme = "https" if use_https else "http"
    print(f"\n  Court calibration app — predictor: {mode}")
    print(f"  on this computer:  {scheme}://127.0.0.1:{port}")
    if use_https:
        print(f"  on your phone:     {scheme}://{lan_ip}:{port}   (same wifi; accept the security warning)\n")
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True, ssl_context="adhoc")
    else:
        print(f"  (camera on phone needs HTTPS=1 — see note)\n")
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
