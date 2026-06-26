"""
Step 2 - The correction editor (local web app).

Opens in your browser. For each image you see the model's predicted keypoints
overlaid on a court wireframe. Drag any point to correct it, cycle a point's
visibility (visible / occluded / not-labeled), and move to the next image.
Every save updates annotations.json AND the YOLO-pose .txt label for that image.

Usage:
    python app.py                 # http://127.0.0.1:5000
    python app.py --port 5001
    python app.py --images PATH   # override images folder

Nothing leaves your machine - the server is local and only reads/writes the
files under your work_dir.
"""

import argparse
import os
import threading

from flask import Flask, jsonify, request, send_file, abort, Response

import common as C

app = Flask(__name__)
STATE = {}            # filled in main()
LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
def _ann_path():
    return C.work_paths(STATE["work_dir"])["annotations"]


def _labels_dir():
    return C.work_paths(STATE["work_dir"])["labels"]


def _ensure_record(fname):
    """Return the annotation record for fname, creating a placeholder if needed."""
    imgs = STATE["ann"]["images"]
    if fname not in imgs:
        path = os.path.join(STATE["images_dir"], fname)
        try:
            w, h = C.image_size(path)
        except Exception:
            w, h = 1280, 720
        imgs[fname] = {"width": w, "height": h, "reviewed": False,
                       "predicted": False, "keypoints": C.empty_keypoints(w, h)}
    return imgs[fname]


# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "templates", "index.html"), encoding="utf-8") as f:
        return Response(f.read(), mimetype="text/html")


@app.route("/api/meta")
def meta():
    return jsonify({
        "kpt_names": C.KPT_NAMES,
        "colors": C.COLORS,
        "skeleton": C.SKELETON,
        "num_kpts": C.NUM_KPTS,
        "model_id": STATE["cfg"].get("model_id", ""),
        "images_dir": STATE["images_dir"],
        "work_dir": STATE["work_dir"],
    })


@app.route("/api/list")
def list_imgs():
    imgs = STATE["images"]
    recs = STATE["ann"]["images"]
    out = []
    for f in imgs:
        r = recs.get(f, {})
        out.append({
            "name": f,
            "reviewed": bool(r.get("reviewed", False)),
            "predicted": bool(r.get("predicted", False)),
            "has_record": f in recs,
        })
    return jsonify({"images": out, "count": len(out)})


@app.route("/api/image")
def image():
    name = request.args.get("name", "")
    if name not in STATE["images_set"]:
        abort(404)
    return send_file(os.path.join(STATE["images_dir"], name))


@app.route("/api/ann")
def get_ann():
    name = request.args.get("name", "")
    if name not in STATE["images_set"]:
        abort(404)
    with LOCK:
        rec = _ensure_record(name)
        return jsonify({"name": name, **rec})


@app.route("/api/save", methods=["POST"])
def save():
    data = request.get_json(force=True)
    name = data.get("name", "")
    if name not in STATE["images_set"]:
        return jsonify({"ok": False, "error": "unknown image"}), 400
    kpts = data.get("keypoints")
    if not isinstance(kpts, list) or len(kpts) != C.NUM_KPTS:
        return jsonify({"ok": False, "error": "expected 12 keypoints"}), 400

    # sanitize
    clean = []
    for k in kpts:
        clean.append({
            "x": round(float(k.get("x", 0)), 2),
            "y": round(float(k.get("y", 0)), 2),
            "v": int(k.get("v", 0)) if int(k.get("v", 0)) in (0, 1, 2) else 0,
        })

    with LOCK:
        rec = _ensure_record(name)
        rec["keypoints"] = clean
        rec["reviewed"] = bool(data.get("reviewed", rec.get("reviewed", False)))
        w, h = rec["width"], rec["height"]
        # write YOLO label
        wrote = C.write_yolo_label(
            clean, w, h,
            os.path.join(_labels_dir(), C.label_name_for(name)),
            STATE["cfg"].get("bbox_pad_frac", 0.02),
        )
        C.save_annotations(STATE["ann"], _ann_path())

    return jsonify({"ok": True, "name": name, "wrote_label": wrote,
                    "reviewed": rec["reviewed"]})


# --------------------------------------------------------------------------- #
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Pickleball keypoint correction editor")
    ap.add_argument("--config", default=os.path.join(here, "config.json"))
    ap.add_argument("--images", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    base_dir = os.path.dirname(os.path.abspath(args.config)) if os.path.exists(args.config) else here
    images_dir, work_dir = C.resolve_dirs(cfg, base_dir)
    if args.images:
        images_dir = os.path.abspath(args.images)

    if not os.path.isdir(images_dir):
        raise SystemExit(f"[error] images folder not found: {images_dir}")

    images = C.list_images(images_dir)
    if not images:
        raise SystemExit(f"[error] no images in {images_dir}")

    ann = C.load_annotations(C.work_paths(work_dir)["annotations"])
    ann.setdefault("images", {})
    os.makedirs(C.work_paths(work_dir)["labels"], exist_ok=True)

    STATE.update({
        "cfg": cfg, "images_dir": images_dir, "work_dir": work_dir,
        "images": images, "images_set": set(images), "ann": ann,
    })

    reviewed = sum(1 for f in images if ann["images"].get(f, {}).get("reviewed"))
    url = f"http://{args.host}:{args.port}"
    print(f"\n  Pickleball keypoint editor")
    print(f"  images:   {images_dir}  ({len(images)} files, {reviewed} reviewed)")
    print(f"  labels:   {C.work_paths(work_dir)['labels']}")
    print(f"\n  Open:  {url}\n")

    if not args.no_browser:
        try:
            import webbrowser
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        except Exception:
            pass

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
