#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-$ROOT/runs/detect/yolo26x_report60_cal10_e120_s2026_b6/weights/best.pt}"
SPLIT_DATA="${SPLIT_DATA:-$ROOT/data_v2/report_split_v16/dataset_report.yaml}"
CAL_BIG="${CAL_BIG:-$ROOT/data_big_report_calibration_s2026}"
VAL_BIG="${VAL_BIG:-$ROOT/data_big_report_validation_s2026}"
THRESH_DIR="${THRESH_DIR:-$ROOT/runs/detect/report_phase1_thresholds_r925_repro_s2026}"
THRESHOLDS="$THRESH_DIR/class_thresholds.json"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-report_v16_repro_r925}"
DEVICE="${DEVICE:-0}"

export PYTHONPATH="$ROOT/ultralytics-main"
export YOLO_CONFIG_DIR="$ROOT/.ultralytics_config"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

require_file() {
  if [ ! -f "$1" ]; then
    echo "ERROR: required file not found: $1" >&2
    exit 1
  fi
}

require_new_dir() {
  if [ -e "$1" ]; then
    echo "ERROR: output already exists: $1" >&2
    echo "Use a new output name; locked evaluation artifacts will not be overwritten." >&2
    exit 1
  fi
}

require_file "$MODEL"
require_file "$SPLIT_DATA"
require_new_dir "$CAL_BIG"
require_new_dir "$VAL_BIG"
require_new_dir "$THRESH_DIR"

python scripts/build_big_mosaic_dataset.py \
  --source-data "$SPLIT_DATA" \
  --source-split calibration \
  --output-root "$CAL_BIG" \
  --variants scene_grouped,mixed_stress,sparse \
  --seed 2026

python scripts/build_big_mosaic_dataset.py \
  --source-data "$SPLIT_DATA" \
  --source-split validation \
  --output-root "$VAL_BIG" \
  --variants scene_grouped,mixed_stress,sparse \
  --seed 2026

python scripts/optimize_class_thresholds.py \
  --model "$MODEL" \
  --data "$CAL_BIG/dataset.yaml" \
  --ultralytics-root "$ROOT/ultralytics-main" \
  --split val_scene_grouped \
  --output-dir "$THRESH_DIR" \
  --device "$DEVICE" \
  --batch 1 \
  --imgsz 1024 \
  --tile-size 800 \
  --stride 800 \
  --pred-conf 0.001 \
  --pred-iou 0.70 \
  --global-iou 0.70 \
  --max-det 300 \
  --base-threshold 0.001 \
  --threshold-stop 0.95 \
  --threshold-step 0.001 \
  --min-recall 0.925

require_file "$THRESHOLDS"

for split in val_scene_grouped val_mixed_stress val_sparse; do
  output="$ROOT/runs/detect/${OUTPUT_PREFIX}_${split}_s2026"
  require_new_dir "$output"
  python scripts/evaluate_big_images_sliding.py \
    --model "$MODEL" \
    --data "$VAL_BIG/dataset.yaml" \
    --ultralytics-root "$ROOT/ultralytics-main" \
    --class-thresholds "$THRESHOLDS" \
    --split "$split" \
    --output-dir "$output" \
    --device "$DEVICE" \
    --batch 1 \
    --imgsz 1024 \
    --tile-size 800 \
    --stride 800 \
    --pred-conf 0.001 \
    --pred-iou 0.70 \
    --global-iou 0.70 \
    --max-det 300 \
    --target-recall 0.85 \
    --max-fdr 0.20 \
    --time-limit 20.0
done

sha256sum "$MODEL" "$THRESHOLDS"
echo "REPORT_LOCKED_EVALUATION=PASSED"
