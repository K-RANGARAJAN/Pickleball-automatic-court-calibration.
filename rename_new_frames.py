"""
Rename extracted clip frames (new_00001.jpg, new_00002.jpg, ...) into the master
sequence starting at img_9285.jpg, img_9286.jpg, ...

- Renames in ascending numeric order of the new_ files (so capture order is kept).
- Safe by default: prints the plan and changes NOTHING unless --apply is given.
- Aborts if any target name already exists (no clobbering).
- Writes rename_new_frames_undo.csv so the rename can be reversed.

Usage:
    python rename_new_frames.py                 # dry run (shows first/last few)
    python rename_new_frames.py --apply         # actually rename
    python rename_new_frames.py --dir clips_frames --start 9285 --width 4 --apply
"""

import argparse
import csv
import os
import re

NEW_RE = re.compile(r"^new_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser(description="Rename new_* frames to img_<start>.jpg ...")
    ap.add_argument("--dir", default="clips_frames", help="folder with the new_*.jpg files")
    ap.add_argument("--start", type=int, default=9285, help="first img number (default: 9285)")
    ap.add_argument("--width", type=int, default=4,
                    help="zero-pad width for the number (default: 4 -> img_9285)")
    ap.add_argument("--apply", action="store_true", help="perform the rename (default: dry run)")
    args = ap.parse_args()

    d = args.dir
    if not os.path.isdir(d):
        raise SystemExit(f"folder not found: {os.path.abspath(d)}")

    # collect + sort new_ files by their numeric index
    files = []
    for f in os.listdir(d):
        m = NEW_RE.match(f)
        if m:
            files.append((int(m.group(1)), f, m.group(2)))
    files.sort(key=lambda t: t[0])
    if not files:
        raise SystemExit(f"no new_*.jpg files in {os.path.abspath(d)}")

    # build plan
    plan = []  # (old, new)
    n = args.start
    for _, fname, ext in files:
        new = f"img_{n:0{args.width}d}.{ext.lower()}"
        plan.append((fname, new))
        n += 1

    # collision check against existing files in the same folder
    existing = set(os.listdir(d))
    targets = {new for _, new in plan}
    clashes = [new for new in targets if new in existing and new not in {o for o, _ in plan}]
    if clashes:
        raise SystemExit(f"ABORT: {len(clashes)} target name(s) already exist, e.g. {clashes[:5]}")

    first_old, first_new = plan[0]
    last_old, last_new = plan[-1]
    print(f"{len(plan)} files in {os.path.abspath(d)}")
    print(f"  {first_old} -> {first_new}")
    print(f"  {last_old} -> {last_new}")

    if not args.apply:
        print("\nDRY RUN — nothing changed. Re-run with --apply to rename.")
        return

    # rename via temp suffix first to avoid any in-place overwrite edge cases
    tmp = [(o, o + ".tmp_rename") for o, _ in plan]
    for o, t in tmp:
        os.rename(os.path.join(d, o), os.path.join(d, t))
    for (o, new), (_, t) in zip(plan, tmp):
        os.rename(os.path.join(d, t), os.path.join(d, new))

    undo = os.path.join(d, "rename_new_frames_undo.csv")
    with open(undo, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["current_name", "previous_name"])
        for o, new in plan:
            w.writerow([new, o])

    print(f"\nRenamed {len(plan)} files. Undo map: {undo}")


if __name__ == "__main__":
    main()
