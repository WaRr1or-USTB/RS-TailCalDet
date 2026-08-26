from pathlib import Path
import os
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = PROJECT_ROOT / "ultralytics-main"
DATA_YAML = PROJECT_ROOT / "data_v2" / "dataset.yaml"
YOLO_CONFIG_DIR = PROJECT_ROOT / ".ultralytics_config"


def count_files(path: Path, suffix: str) -> int:
    return len(list(path.glob(f"*{suffix}")))


def main() -> int:
    print(f"project_root={PROJECT_ROOT}")
    print(f"ultralytics_root={ULTRALYTICS_ROOT}")
    print(f"data_yaml={DATA_YAML}")

    if not ULTRALYTICS_ROOT.exists():
        print("ERROR: ultralytics-main not found.")
        return 1
    if not DATA_YAML.exists():
        print("ERROR: data_v2/dataset.yaml not found.")
        return 1

    for split in ("train", "val"):
        images = PROJECT_ROOT / "data_v2" / "images" / split
        labels = PROJECT_ROOT / "data_v2" / "labels" / split
        image_count = count_files(images, ".jpg")
        label_count = count_files(labels, ".txt")
        print(f"{split}: images={image_count} labels={label_count}")
        if image_count != label_count:
            print(f"ERROR: {split} image/label count mismatch.")
            return 1
        if image_count == 0:
            print(f"ERROR: {split} split is empty.")
            return 1

    sys.path.insert(0, str(ULTRALYTICS_ROOT))
    YOLO_CONFIG_DIR.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
    print(f"YOLO_CONFIG_DIR={os.environ['YOLO_CONFIG_DIR']}")

    try:
        import torch
    except Exception as exc:
        print("ERROR: failed to import torch.")
        print(repr(exc))
        traceback.print_exc()
        return 2

    try:
        import ultralytics
    except Exception as exc:
        print("ERROR: failed to import ultralytics from local source.")
        print(repr(exc))
        traceback.print_exc()
        return 3

    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device_count={torch.cuda.device_count()}")
        print(f"cuda_device_0={torch.cuda.get_device_name(0)}")
    print(f"ultralytics={ultralytics.__version__}")
    print("setup_check=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
