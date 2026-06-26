"""
Batch re-projection-error validator.

Runs the homography on every label and reports the re-projection error
distribution. Frames with HIGH error are suspect: a keypoint mislabeled, or the
12 points in an inconsistent order (the silent failure that corrupts training).

Usage:
    python validate_labels.py --images images --labels pb_relabel/labels
    python validate_labels.py --images images --labels pb_relabel/labels --flag-px 12 --save-bad calib_bad
"""

import argparse
import os
import csv
import cv2
import numpy as np
import court_calib as cc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--flag-px", type=float, default=12.0,
                    help="frames with mean reprojection error above this are flagged")
    ap.add_argument("--save-bad", default=None,
                    help="if set, render flagged frames' overlays into this folder")
    ap.add_argument("--ransac-px", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    labels = sorted(f for f in os.listdir(args.labels) if f.endswith(".txt"))
    if args.limit:
        labels = labels[: args.limit]

    if args.save_bad:
        os.makedirs(args.save_bad, exist_ok=True)

    rows = []
    means = []
    bad = []
    errored = []
    for lf in labels:
        stem = os.path.splitext(lf)[0]
        ip = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = os.path.join(args.images, stem + ext)
            if os.path.exists(cand):
                ip = cand
                break
        if ip is None:
            errored.append((stem, "no image"))
            continue
        try:
            img = cv2.imread(ip)
            h, w = img.shape[:2]
            kps = cc.load_label_points(os.path.join(args.labels, lf), w, h)
            H, Hinv, inl, used = cc.compute_homography(kps, args.ransac_px)
            m = cc.reprojection_error(kps, H, Hinv, used, inl)
            rows.append((stem, m["mean_px"], m["max_px"], m["n_points"], int(inl.sum())))
            means.append(m["mean_px"])
            if m["mean_px"] > args.flag_px:
                bad.append((stem, m["mean_px"]))
                if args.save_bad:
                    out, _, _ = cc.draw_near_half(img, kps, args.ransac_px)
                    cv2.imwrite(os.path.join(args.save_bad, f"{stem}_{m['mean_px']:.0f}px.jpg"), out)
        except Exception as e:
            errored.append((stem, str(e)[:60]))

    means = np.array(means)
    print(f"\nValidated {len(rows)} frames "
          f"({len(errored)} errored, {len(bad)} flagged > {args.flag_px}px)")
    if len(means):
        print(f"  reprojection mean_px:  median={np.median(means):.2f}  "
              f"mean={means.mean():.2f}  p95={np.percentile(means,95):.2f}  "
              f"max={means.max():.2f}")
    if bad:
        print(f"\n  worst frames (likely mislabeled / wrong point order):")
        for stem, e in sorted(bad, key=lambda x: -x[1])[:20]:
            print(f"    {stem}: {e:.1f} px")
    if errored:
        print(f"\n  could not process ({len(errored)}), e.g.:")
        for stem, why in errored[:10]:
            print(f"    {stem}: {why}")

    with open("validate_labels_report.csv", "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["stem", "mean_px", "max_px", "n_points", "n_inliers"])
        wtr.writerows(rows)
    print("\nfull report -> validate_labels_report.csv")


if __name__ == "__main__":
    main()
