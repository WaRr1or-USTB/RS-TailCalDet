"""Train the shared-localization YOLO26x hierarchical detector from the promoted HRSC checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = PROJECT_ROOT / "ultralytics-main"
DEFAULT_MODEL = ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26x-hierarchical.yaml"
DEFAULT_DATA = PROJECT_ROOT / "data_v2" / "dataset.yaml"
DEFAULT_PRETRAINED = PROJECT_ROOT / "runs" / "detect" / "yolo26x_hrsc_ft_e40" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--name", default="yolo26x_hier_broadg1_e60")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--broad-loss-gain", type=float, default=1.0)
    parser.add_argument("--lr0", type=float, default=2e-5)
    parser.add_argument("--lrf", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.broad_loss_gain <= 0:
        raise ValueError("--broad-loss-gain must be positive")
    for path in (args.data, args.pretrained):
        if not path.resolve().exists():
            raise FileNotFoundError(path.resolve())
    sys.path.insert(0, str(ULTRALYTICS_ROOT))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics_config"))
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    model.load(str(args.pretrained.resolve()))
    head = model.model.model[-1]
    head.broad_loss_gain = args.broad_loss_gain
    head.broad_fusion_gain = 0.0
    print(
        f"head={type(head).__name__} broad_loss_gain={head.broad_loss_gain} "
        f"fine_classes={head.nc} broad_classes={head.broad_nc}"
    )
    if args.dry_run:
        model.info()
        print("dry_run=ok")
        return
    output = PROJECT_ROOT / "runs" / "detect" / args.name
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    results = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name=args.name,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=5e-4,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        cos_lr=True,
        seed=args.seed,
        deterministic=True,
        cache=False,
        amp=True,
        mosaic=1.0,
        close_mosaic=10,
        freeze=args.freeze,
        plots=False,
        save=True,
        resume=args.resume,
    )
    print(results)


if __name__ == "__main__":
    main()
