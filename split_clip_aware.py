"""
Clip-aware train/val split for a FLAT folder of YOLO-pose data.

Takes a flat folder of images + matching .txt labels and splits into:
    <out>/images/train  <out>/images/val
    <out>/labels/train  <out>/labels/val
keeping WHOLE GROUPS (clips/sources) on one side only, so near-duplicate
consecutive frames never leak across train/val (which would inflate val scores).

How frames are grouped into "clips" — pick with --group-by:
  prefix   : group by filename prefix before the trailing number
             (img_09285.jpg -> group "img_")  [DEFAULT]
  regex    : group by a custom regex's first capture group (--pattern)
  chunk    : group every N consecutive frames (sorted) into one clip (--chunk N)
             use this when filenames carry no clip identity but ARE in capture order

Examples:
    # frames named clipA_0001.jpg, clipB_0001.jpg ... -> group by the clip name
    python split_clip_aware.py --src his_flat --out pb_dataset --group-by regex --pattern '^(.*?)_\d+'

    # all named img_00001.jpg in capture order, no clip id -> chunk every 200 frames
    python split_clip_aware.py --src his_flat --out pb_dataset --group-by chunk --chunk 200

    # default: split on the text before the trailing number
    python split_clip_aware.py --src his_flat --out pb_dataset

Safe: copies (doesn't move) by default. Use --move to move, --symlink to link.
Prints the plan and the exact group->split assignment.
"""

import argparse
import os
import random
import re
import shutil
from collections import defaultdict

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
NUM_TAIL = re.compile(r"^(?P<prefix>.*?)(?P<num>\d+)$")


def stem(fn):
    return os.path.splitext(fn)[0]


def group_key(name, mode, pattern, chunk, ordered_index):
    s = stem(name)
    if mode == "prefix":
        m = NUM_TAIL.match(s)
        return m.group("prefix") if m else s
    if mode == "regex":
        m = re.search(pattern, s)
        return m.group(1) if m else s
    if mode == "chunk":
        return f"chunk_{ordered_index // chunk:05d}"
    return s


def place(src, dst, how):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    if how == "move":
        shutil.move(src, dst)
    elif how == "symlink":
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="flat folder with images + .txt labels")
    ap.add_argument("--out", default="pb_dataset", help="output dataset root")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--group-by", choices=["prefix", "regex", "chunk"], default="prefix")
    ap.add_argument("--pattern", default=r"^(.*?)_\d+", help="regex (first group = clip id) for --group-by regex")
    ap.add_argument("--chunk", type=int, default=200, help="frames per group for --group-by chunk")
    ap.add_argument("--labels-src", default=None,
                    help="separate labels folder if labels aren't beside images")
    ap.add_argument("--seed", type=int, default=42)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--move", action="store_true")
    g.add_argument("--symlink", action="store_true")
    args = ap.parse_args()

    how = "move" if args.move else ("symlink" if args.symlink else "copy")
    labels_src = args.labels_src or args.src

    imgs = sorted(f for f in os.listdir(args.src) if f.lower().endswith(IMG_EXTS))
    if not imgs:
        raise SystemExit(f"no images in {args.src}")

    # pair images with labels; warn on any missing label (would become a bg image)
    pairs, missing = [], []
    for i, im in enumerate(imgs):
        lbl = stem(im) + ".txt"
        if os.path.exists(os.path.join(labels_src, lbl)):
            pairs.append((im, lbl, i))
        else:
            missing.append(im)
    if missing:
        print(f"[WARN] {len(missing)} images have NO label (will be SKIPPED), e.g. {missing[:5]}")

    # group
    groups = defaultdict(list)
    for im, lbl, idx in pairs:
        groups[group_key(im, args.group_by, args.pattern, args.chunk, idx)].append((im, lbl))
    keys = sorted(groups)
    print(f"{len(pairs)} labeled frames in {len(keys)} groups (group-by={args.group_by})")

    # assign whole groups to val until ~val-frac of frames reached
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    target_val = int(round(len(pairs) * args.val_frac))
    val_keys, n_val = set(), 0
    for k in keys:
        if n_val >= target_val:
            break
        val_keys.add(k)
        n_val += len(groups[k])

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    n_tr = n_vl = 0
    for k in keys:
        split = "val" if k in val_keys else "train"
        for im, lbl in groups[k]:
            place(os.path.join(args.src, im), os.path.join(args.out, "images", split, im), how)
            place(os.path.join(labels_src, lbl), os.path.join(args.out, "labels", split, lbl), how)
            if split == "val":
                n_vl += 1
            else:
                n_tr += 1

    print(f"\ntrain: {n_tr} frames   val: {n_vl} frames   ({n_vl/(n_tr+n_vl):.1%} val)")
    print(f"val groups held out: {len(val_keys)} / {len(keys)}")
    print(f"output: {os.path.abspath(args.out)}/  (now add data.yaml here)")


if __name__ == "__main__":
    main()
