#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-$ROOT/data_v2/report_ablation_v16}"
DATA_YAML="$DATA_DIR/dataset_official.yaml"
BASE_MODEL="${BASE_MODEL:-$ROOT/yolo26x.pt}"
RUN_NAME="${RUN_NAME:-yolo26x_report_baseline_official60_e120_s2026_b6}"
BATCH="${BATCH:-6}"
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-0}"

export PYTHONPATH="$ROOT/ultralytics-main"
export YOLO_CONFIG_DIR="$ROOT/.ultralytics_config"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python scripts/prepare_report_ablation_datasets.py \
  --variant official \
  --output-dir "$DATA_DIR" \
  --overwrite

test -f "$BASE_MODEL"
test -f "$DATA_YAML"

RUN_DIR="$ROOT/runs/detect/$RUN_NAME"
if [ -e "$RUN_DIR" ]; then
  echo "ERROR: output already exists: $RUN_DIR" >&2
  echo "Resume explicitly or choose a new RUN_NAME; this script will not overwrite it." >&2
  exit 1
fi

python scripts/train_yolo_baseline.py \
  --model "$BASE_MODEL" \
  --data "$DATA_YAML" \
  --epochs 120 \
  --imgsz 1024 \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --name "$RUN_NAME" \
  --patience 30 \
  --optimizer auto \
  --lr0 0.01 \
  --lrf 0.01 \
  --weight-decay 0.0005 \
  --warmup-epochs 3 \
  --close-mosaic 10 \
  --mosaic 1.0 \
  --seed 2026 \
  --no-plots \
  2>&1 | tee "$ROOT/runs/${RUN_NAME}.log"

test -f "$RUN_DIR/weights/best.pt"
sha256sum "$RUN_DIR/weights/best.pt"
echo "REPORT_OFFICIAL_BASELINE=PASSED"
