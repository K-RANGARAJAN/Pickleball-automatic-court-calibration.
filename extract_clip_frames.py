"""
Extract frames from the screen-recording clips and letterbox them to 1920x1080.

- Samples every STEP-th frame (default 6).
- Source clips are 2940x1912 (3:2). They are scaled to fit inside 1920x1080
  (16:9) PRESERVING aspect ratio, then padded with black bars (letterbox) so
  the court geometry is never stretched.
- Output: temp names new_00001.jpg, new_00002.jpg, ... in OUT_DIR, plus a
  manifest CSV so any frame can be traced back to its source clip/frame.
  (Fold these into your master img_XXXXX sequence later with pad_rename.py.)

Usage:
    python extract_clip_frames.py
    python extract_clip_frames.py --src "/path/to/clips" --out clips_frames --step 6
"""

import argparse
import cv2
import glob
import os


def letterbox(img, tw, th):
    """Resize img to fit inside (tw, th) preserving aspect, pad with black bars."""
    h, w = img.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (th - nh) // 2
    bottom = th - nh - top
    left = (tw - nw) // 2
    right = tw - nw - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def main():
    ap = argparse.ArgumentParser(description="Extract + letterbox clip frames to 1920x1080.")
    ap.add_argument("--src", default=".",
                    help="folder containing the .mov clips (default: current dir)")
    ap.add_argument("--out", default="clips_frames",
                    help="output folder for extracted frames")
    ap.add_argument("--step", type=int, default=6,
                    help="keep every Nth frame (default: 6)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality 1-100")
    ap.add_argument("--ext", default="mov", help="clip extension to match (default: mov)")
    args = ap.parse_args()

    clips = sorted(glob.glob(os.path.join(args.src, f"*.{args.ext}")))
    if not clips:
        # also try uppercase extension
        clips = sorted(glob.glob(os.path.join(args.src, f"*.{args.ext.upper()}")))
    if not clips:
        raise SystemExit(f"No *.{args.ext} clips found in {os.path.abspath(args.src)}")

    os.makedirs(args.out, exist_ok=True)
    man_path = os.path.join(args.out, "_source_manifest.csv")
    man = open(man_path, "w")
    man.write("out_name,source_clip,source_frame_index\n")

    idx = 0
    print(f"Found {len(clips)} clip(s). Extracting every {args.step}th frame -> "
          f"{args.width}x{args.height} letterboxed JPG.\n")
    for clip in clips:
        cap = cv2.VideoCapture(clip)
        if not cap.isOpened():
            print(f"  [warn] could not open {clip}, skipping")
            continue
        fno = saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fno % args.step == 0:
                idx += 1
                name = f"new_{idx:05d}.jpg"
                cv2.imwrite(os.path.join(args.out, name),
                            letterbox(frame, args.width, args.height),
                            [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                man.write(f"{name},{os.path.basename(clip)},{fno}\n")
                saved += 1
            fno += 1
        cap.release()
        print(f"  {os.path.basename(clip)[:46]:48s} read {fno:5d}f  saved {saved}")

    man.close()
    print("-" * 72)
    print(f"DONE. {idx} frames -> {os.path.abspath(args.out)}/")
    print(f"manifest: {man_path}")
    print("\nNext: fold new_*.jpg into your master img_XXXXX sequence with pad_rename.py,")
    print("then label them (court keypoints will sit in a horizontal band between the bars).")


if __name__ == "__main__":
    main()
