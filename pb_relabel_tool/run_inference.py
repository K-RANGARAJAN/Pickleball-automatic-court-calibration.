"""
Step 1 - Batch inference.

Runs the Roboflow model over every image in the images folder, saves the raw
prediction JSON for each image, and builds:
  - annotations.json   (source of truth: 12 keypoints per image, pixel coords + visibility)
  - labels/<name>.txt  (initial YOLO-pose labels from the model predictions)

Usage:
    python run_inference.py                  # uses config.json next to this file
    python run_inference.py --images PATH    # override images folder
    python run_inference.py --only-new       # skip images already in annotations.json
    python run_inference.py --limit 20       # only process first N (handy for a test run)

The corrected labels are produced later by the editor (app.py); this step just
gives every image a starting point.
"""

import argparse
import json
import os
import sys
import time

import common as C


def get_args():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Batch Roboflow keypoint inference.")
    ap.add_argument("--config", default=os.path.join(here, "config.json"))
    ap.add_argument("--images", default=None, help="override images folder")
    ap.add_argument("--only-new", action="store_true",
                    help="skip images already present in annotations.json")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-run inference even for reviewed images (default: keep reviewed)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N images (0 = all)")
    return ap.parse_args(), here


def main():
    args, here = get_args()
    cfg = C.load_config(args.config)
    base_dir = os.path.dirname(os.path.abspath(args.config)) if os.path.exists(args.config) else here
    images_dir, work_dir = C.resolve_dirs(cfg, base_dir)
    if args.images:
        images_dir = os.path.abspath(args.images)
    paths = C.work_paths(work_dir)

    if not os.path.isdir(images_dir):
        sys.exit(f"[error] images folder not found: {images_dir}\n"
                 f"        Put your images there, or pass --images PATH, "
                 f"or edit 'images_dir' in config.json.")

    images = C.list_images(images_dir)
    if not images:
        sys.exit(f"[error] no images found in {images_dir}")
    if args.limit:
        images = images[:args.limit]

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(paths["raw"], exist_ok=True)
    os.makedirs(paths["labels"], exist_ok=True)

    ann = C.load_annotations(paths["annotations"])
    ann["model_id"] = cfg["model_id"]
    ann["kpt_names"] = C.KPT_NAMES
    ann.setdefault("images", {})

    # Lazy import so --help works without requests installed.
    import requests
    session = requests.Session()

    print(f"images:  {images_dir}  ({len(images)} files)")
    print(f"output:  {work_dir}")
    print(f"model:   {cfg['model_id']}\n")

    done = skipped = failed = 0
    for i, fname in enumerate(images, 1):
        rec = ann["images"].get(fname)
        if rec and rec.get("reviewed") and not args.overwrite:
            skipped += 1
            print(f"[{i}/{len(images)}] {fname}  (reviewed - kept)")
            continue
        if rec and args.only_new:
            skipped += 1
            continue

        img_path = os.path.join(images_dir, fname)
        try:
            w, h = C.image_size(img_path)
        except Exception as e:
            failed += 1
            print(f"[{i}/{len(images)}] {fname}  !! cannot open image: {e}")
            continue

        try:
            result = C.infer_image(session, img_path, cfg["api_url"],
                                   cfg["api_key"], cfg["model_id"])
        except Exception as e:
            failed += 1
            print(f"[{i}/{len(images)}] {fname}  !! inference failed: {repr(e)[:160]}")
            # still create an empty placeholder so it shows up in the editor
            ann["images"].setdefault(fname, {
                "width": w, "height": h, "reviewed": False,
                "keypoints": C.empty_keypoints(w, h),
            })
            continue

        # save raw prediction for audit / re-parsing
        with open(os.path.join(paths["raw"], fname + ".json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        # prefer the size reported by the API if present
        api_img = (result or {}).get("image") or {}
        W = int(api_img.get("width") or w)
        H = int(api_img.get("height") or h)

        kpts, found = C.parse_roboflow_result(result, W, H, cfg.get("confidence_threshold", 0.0))
        ann["images"][fname] = {
            "width": W, "height": H,
            "reviewed": False,
            "predicted": bool(found),
            "keypoints": kpts,
        }

        # initial YOLO label
        C.write_yolo_label(kpts, W, H,
                           os.path.join(paths["labels"], C.label_name_for(fname)),
                           cfg.get("bbox_pad_frac", 0.02))

        done += 1
        tag = "ok" if found else "no keypoints detected"
        print(f"[{i}/{len(images)}] {fname}  ({tag})")

        if done % 20 == 0:
            C.save_annotations(ann, paths["annotations"])  # periodic checkpoint

    C.save_annotations(ann, paths["annotations"])
    print(f"\nDone. inferred={done}  skipped={skipped}  failed={failed}")
    print(f"annotations: {paths['annotations']}")
    print(f"\nNext:  python app.py   (then open the printed http://127.0.0.1:5000 link)")


if __name__ == "__main__":
    main()
