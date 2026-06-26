"""
Train YOLOv11-pose on the pickleball court keypoint dataset (12 kpts).

Prereq:
    pip install ultralytics
    Dataset assembled at pb_dataset/ with images/{train,val}, labels/{train,val}, data.yaml

Run:
    python train_pose.py                      # full run, settings below
    python train_pose.py --smoke              # 1-epoch sanity check (do this FIRST)
    python train_pose.py --model yolo11x-pose.pt --batch 16

The smoke test catches format errors / OOM in minutes before you commit the long run.
"""

import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pb_dataset/data.yaml")
    ap.add_argument("--model", default="yolo11m-pose.pt",
                    help="yolo11m-pose.pt (balanced) or yolo11x-pose.pt (max accuracy)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=-1,
                    help="-1 = auto (uses ~60%% VRAM); set explicitly on a big GPU, e.g. 16")
    ap.add_argument("--device", default="0", help="'0' single GPU, '0,1,2,3' multi-GPU, 'cpu'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default="pb_court_pose")
    ap.add_argument("--smoke", action="store_true", help="1-epoch sanity run on small imgsz")
    args = ap.parse_args()

    model = YOLO(args.model)

    if args.smoke:
        model.train(
            data=args.data, epochs=1, imgsz=640, batch=8,
            device=args.device, workers=4,
            name="pb_smoke_test", project="runs/pose", exist_ok=True,
            plots=False,
        )
        print("\nSmoke test done. If this completed and read all images, the pipeline is valid.")
        return

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,

        # --- optimizer / schedule ---
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        weight_decay=0.0005,
        patience=50,                # early-stop if no val improvement for 50 epochs

        # --- augmentation ---
        # fliplr is ON and SAFE because data.yaml flip_idx maps L<->R court points.
        fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5,                  # courts are near-level; keep rotation small
        translate=0.1,
        scale=0.3,
        shear=2.0,
        mosaic=0.0,                 # OFF: mosaic mangles single-court geometry/keypoints
        mixup=0.0,                  # OFF: same reason
        rect=False,

        # --- bookkeeping ---
        name=args.name,
        project="runs/pose",
        exist_ok=True,
        save=True,
        plots=True,
        val=True,
    )
    print(f"\nDone. Best weights: runs/pose/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
