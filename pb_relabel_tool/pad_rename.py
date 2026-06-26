"""Zero-pad image numbers so frames sort in true numeric (== capture) order.

Renames  img_<n>.jpg -> img_<n zero-padded to WIDTH>.jpg  and keeps everything matched:
  - the image in images_dir
  - its YOLO label   <work_dir>/labels/<stem>.txt
  - its key in       <work_dir>/annotations.json   (preserves reviewed corrections)

After this, alphabetical order = numeric order, so one court's consecutive frames are
adjacent in the editor and "Copy prev (C)" flies through near-duplicate frames.

Safe by default: prints the plan and changes NOTHING unless --apply is given. Aborts on any
collision. Writes a reverse map (pad_rename_undo.csv) so the rename can be undone.

Usage:
    python pad_rename.py                 # dry run: show what would change
    python pad_rename.py --apply         # actually rename
    python pad_rename.py --width 5 --apply
    python pad_rename.py --undo pad_rename_undo.csv --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import common as C

NAME_RE = re.compile(r"^(?P<prefix>.*?)(?P<num>\d+)(?P<ext>\.[A-Za-z0-9]+)$")


def parse_name(fname: str):
    """img_12.jpg -> ('img_', 12, '.jpg'); returns None if no trailing number."""
    m = NAME_RE.match(fname)
    if not m:
        return None
    return m.group("prefix"), int(m.group("num")), m.group("ext")


def padded(fname: str, width: int) -> str | None:
    p = parse_name(fname)
    if not p:
        return None
    prefix, num, ext = p
    return f"{prefix}{num:0{width}d}{ext}"


def build_plan(images_dir: str, width: int):
    """List of (old_image, new_image) for files that actually change. Raises on collision."""
    imgs = [f for f in os.listdir(images_dir)
            if f.lower().endswith(C.IMG_EXTS) and os.path.isfile(os.path.join(images_dir, f))]
    plan = []
    seen_new: dict[str, str] = {}
    for f in imgs:
        new = padded(f, width)
        if new is None:
            print(f"  [skip] no trailing number: {f}")
            continue
        if new in seen_new and seen_new[new] != f:
            raise SystemExit(f"[abort] collision: '{f}' and '{seen_new[new]}' both -> '{new}'")
        seen_new[new] = f
        if new != f:
            plan.append((f, new))
    # a target must not already exist as a *different* current file we're not renaming
    current = set(imgs)
    untouched = current - {old for old, _ in plan}
    for _, new in plan:
        if new in untouched:
            raise SystemExit(f"[abort] target '{new}' already exists and isn't being renamed")
    return sorted(plan)


def stem(fname: str) -> str:
    return os.path.splitext(fname)[0]


def apply_plan(plan, images_dir, labels_dir, ann_path, undo_csv):
    # two-phase rename via temp suffix to dodge any A->B, B->C chains safely
    tmp = ".padtmp__"
    # images + labels: phase 1 -> temp
    for old, new in plan:
        os.rename(os.path.join(images_dir, old), os.path.join(images_dir, old + tmp))
        lo = os.path.join(labels_dir, stem(old) + ".txt")
        if os.path.exists(lo):
            os.rename(lo, lo + tmp)
    # phase 2 temp -> final
    for old, new in plan:
        os.rename(os.path.join(images_dir, old + tmp), os.path.join(images_dir, new))
        lo = os.path.join(labels_dir, stem(old) + ".txt") + tmp
        if os.path.exists(lo):
            os.rename(lo, os.path.join(labels_dir, stem(new) + ".txt"))

    # annotations.json keys
    if os.path.exists(ann_path):
        ann = C.load_annotations(ann_path)
        imgs_map = ann.get("images", {})
        remap = {old: new for old, new in plan}
        new_images = {}
        for k, v in imgs_map.items():
            new_images[remap.get(k, k)] = v
        ann["images"] = new_images
        C.save_annotations(ann, ann_path)

    with open(undo_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["current_name", "previous_name"])
        for old, new in plan:
            w.writerow([new, old])


def run_undo(undo_csv, images_dir, labels_dir, ann_path):
    with open(undo_csv, encoding="utf-8") as f:
        rows = [r for r in csv.reader(f)][1:]
    plan = [(cur, prev) for cur, prev in rows]   # rename current -> previous
    apply_plan(plan, images_dir, labels_dir, ann_path, undo_csv + ".reundo")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(here, "config.json"))
    ap.add_argument("--width", type=int, default=4, help="zero-pad to this many digits")
    ap.add_argument("--apply", action="store_true", help="actually rename (default: dry run)")
    ap.add_argument("--undo", default=None, help="reverse a previous rename using its undo csv")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    images_dir, work_dir = C.resolve_dirs(cfg, base)
    labels_dir = C.work_paths(work_dir)["labels"]
    ann_path = C.work_paths(work_dir)["annotations"]

    if not os.path.isdir(images_dir):
        sys.exit(f"[error] images dir not found: {images_dir}")

    if args.undo:
        if not args.apply:
            sys.exit("--undo needs --apply to actually run")
        run_undo(args.undo, images_dir, labels_dir, ann_path)
        print(f"undone using {args.undo}")
        return

    plan = build_plan(images_dir, args.width)
    print(f"images:      {images_dir}")
    print(f"labels:      {labels_dir}")
    print(f"annotations: {ann_path}")
    print(f"pad width:   {args.width}")
    print(f"files to rename: {len(plan)}\n")
    for old, new in plan[:8]:
        print(f"  {old:>16}  ->  {new}")
    if len(plan) > 8:
        print(f"  ... and {len(plan) - 8} more")

    if not args.apply:
        print("\n(dry run — nothing changed. Re-run with --apply to rename.)")
        return

    undo_csv = os.path.join(here, "pad_rename_undo.csv")
    apply_plan(plan, images_dir, labels_dir, ann_path, undo_csv)
    print(f"\nrenamed {len(plan)} images (+ matching labels + annotations keys)")
    print(f"undo map: {undo_csv}   (python pad_rename.py --undo {undo_csv} --apply)")


if __name__ == "__main__":
    main()
