from pathlib import Path
import argparse
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = PROJECT_ROOT / "ultralytics-main"
DATA_YAML = PROJECT_ROOT / "data_v2" / "dataset.yaml"
YOLO_CONFIG_DIR = PROJECT_ROOT / ".ultralytics_config"


def ensure_dataset_yaml_path(data_yaml: Path):
    dataset_path = str(data_yaml.parent).replace("\\", "/")
    text = data_yaml.read_text(encoding="utf-8")
    lines = text.splitlines()
    changed = False
    for i, line in enumerate(lines):
        if line.strip().startswith("path:"):
            lines[i] = f"path: {dataset_path}"
            changed = True
            break
    if not changed:
        lines.insert(0, f"path: {dataset_path}")
    new_text = "\n".join(lines) + "\n"
    if new_text != text:
        data_yaml.write_text(new_text, encoding="utf-8")


def apply_gpu_memory_fraction(fraction: float) -> None:
    if fraction <= 0:
        return
    if fraction > 1:
        raise ValueError("--gpu-memory-fraction must be between 0 and 1.")

    import torch

    if not torch.cuda.is_available():
        print("gpu_memory_fraction=skipped_cuda_unavailable")
        return

    device_count = torch.cuda.device_count()
    for device_idx in range(device_count):
        torch.cuda.set_per_process_memory_fraction(fraction, device=device_idx)
    print(f"gpu_memory_fraction={fraction} visible_cuda_devices={device_count}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO baseline for the Challenge Cup dataset.")
    parser.add_argument(
        "--model",
        default=str(ULTRALYTICS_ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26.yaml"),
        help="Model YAML or .pt weights. Prefer yolo26n.pt/yolo26s.pt if pretrained weights are available.",
    )
    parser.add_argument("--data", default=str(DATA_YAML))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--name", default="yolo26n_baseline_img1024")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--freeze", type=int, default=0, help="Freeze the first N model layers during training.")
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.0,
        help="Optional per-process CUDA memory fraction for each visible GPU, e.g. 0.43 on 16GB V100.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Ultralytics training plots. Disabled by default to avoid matplotlib/numpy post-processing crashes.",
    )
    parser.add_argument("--pretrained", default="", help="Optional .pt weights loaded after constructing the model.")
    parser.add_argument("--dry-run", action="store_true", help="Only load the model and print basic info; do not train.")
    parser.add_argument(
        "--iou-loss",
        default="CIoU",
        choices=["CIoU", "WIoU"],
        help="Bounding-box IoU loss. WIoU = Wise-IoU v3 (enables the loss-level switch).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_yaml = Path(args.data).expanduser().resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset YAML not found: {data_yaml}")
    ensure_dataset_yaml_path(data_yaml)
    if args.iou_loss.upper() == "WIoU":
        os.environ["YOLO_IoU_LOSS"] = "WIoU"
    sys.path.insert(0, str(ULTRALYTICS_ROOT))
    YOLO_CONFIG_DIR.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
    apply_gpu_memory_fraction(args.gpu_memory_fraction)

    from ultralytics import YOLO

    model = YOLO(args.model)
    if args.pretrained:
        model.load(args.pretrained)
    if args.dry_run:
        model.info()
        print(f"data={data_yaml}")
        print(f"model={args.model}")
        print(f"pretrained={args.pretrained or None}")
        print(f"freeze={args.freeze}")
        print("dry_run=ok")
        return

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name=args.name,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        cos_lr=args.cos_lr,
        seed=args.seed,
        deterministic=True,
        cache=False,
        amp=args.amp,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        plots=args.plots,
        freeze=args.freeze,
        save=True,
        resume=args.resume,
    )
    print(results)


if __name__ == "__main__":
    main()
