"""Train one YOLO26 crop classifier used to verify and refine shared detector candidates."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = PROJECT_ROOT / "ultralytics-main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", required=True, choices=("ship", "aircraft", "vehicle"))
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data_hierarchical" / "experts")
    parser.add_argument("--model", default="yolo26m-cls.yaml")
    parser.add_argument("--pretrained", type=Path, default=None, help="Optional classifier or detector checkpoint.")
    parser.add_argument("--name", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = (args.data_root / args.expert).resolve()
    if not (dataset / "train").is_dir() or not (dataset / "val").is_dir():
        raise FileNotFoundError(f"classification dataset missing train/val folders: {dataset}")
    sys.path.insert(0, str(ULTRALYTICS_ROOT))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics_config"))
    from ultralytics import YOLO

    model = YOLO(args.model)
    if args.pretrained:
        model.load(str(args.pretrained.resolve()))
    if args.dry_run:
        model.info()
        print(f"expert={args.expert} data={dataset} dry_run=ok")
        return
    name = args.name or f"yolo26m_cls_{args.expert}_ctx_e{args.epochs}"
    output = PROJECT_ROOT / "runs" / "classify" / name
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    results = model.train(
        data=str(dataset),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(PROJECT_ROOT / "runs" / "classify"),
        name=name,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.01,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        patience=25,
        cos_lr=True,
        seed=args.seed,
        deterministic=True,
        cache=False,
        amp=True,
        plots=False,
        save=True,
    )
    print(results)


if __name__ == "__main__":
    main()
