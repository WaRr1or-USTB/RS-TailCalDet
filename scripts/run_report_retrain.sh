#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
DATA_YAML="${DATA_YAML:-$ROOT/data_v2/report_split_v16/dataset_report.yaml}"
BASE_MODEL="${BASE_MODEL:-$ROOT/yolo26x.pt}"
PHASE1_NAME="${PHASE1_NAME:-yolo26x_report60_cal10_e120_s2026_b6}"
PHASE2_NAME="${PHASE2_NAME:-yolo26x_report60_cal10_ft_e40_s2026_b6}"
BATCH="${BATCH:-6}"
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-0}"
SEED="${SEED:-2026}"

export PYTHONPATH="$ROOT/ultralytics-main"
export YOLO_CONFIG_DIR="$ROOT/.ultralytics_config"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

require_file() {
  if [ ! -f "$1" ]; then
    echo "ERROR: required file not found: $1" >&2
    exit 1
  fi
}

run_phase1() {
  local run_dir="$ROOT/runs/detect/$PHASE1_NAME"
  if [ -f "$run_dir/weights/best.pt" ] && [ -f "$run_dir/results.csv" ]; then
    echo "PHASE1_ALREADY_COMPLETE=$run_dir/weights/best.pt"
    return
  fi
  if [ -e "$run_dir" ]; then
    echo "ERROR: incomplete phase-1 output exists: $run_dir" >&2
    echo "Resume it explicitly or choose a new PHASE1_NAME; this script will not overwrite it." >&2
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
    --name "$PHASE1_NAME" \
    --patience 30 \
    --optimizer auto \
    --lr0 0.01 \
    --lrf 0.01 \
    --weight-decay 0.0005 \
    --warmup-epochs 3 \
    --close-mosaic 10 \
    --mosaic 1.0 \
    --seed "$SEED" \
    --no-plots \
    2>&1 | tee "$ROOT/runs/${PHASE1_NAME}.log"

  require_file "$run_dir/weights/best.pt"
  echo "REPORT_PHASE1_COMPLETE=$run_dir/weights/best.pt"
}

run_phase2() {
  local phase1_weight="$ROOT/runs/detect/$PHASE1_NAME/weights/best.pt"
  local run_dir="$ROOT/runs/detect/$PHASE2_NAME"
  require_file "$phase1_weight"
  if [ -f "$run_dir/weights/best.pt" ] && [ -f "$run_dir/results.csv" ]; then
    echo "PHASE2_ALREADY_COMPLETE=$run_dir/weights/best.pt"
    return
  fi
  if [ -e "$run_dir" ]; then
    echo "ERROR: incomplete phase-2 output exists: $run_dir" >&2
    echo "Resume it explicitly or choose a new PHASE2_NAME; this script will not overwrite it." >&2
    exit 1
  fi

  python scripts/train_yolo_baseline.py \
    --model "$phase1_weight" \
    --data "$DATA_YAML" \
    --epochs 40 \
    --imgsz 1024 \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --name "$PHASE2_NAME" \
    --patience 15 \
    --optimizer AdamW \
    --lr0 0.0001 \
    --lrf 0.01 \
    --weight-decay 0.0005 \
    --warmup-epochs 2 \
    --close-mosaic 10 \
    --mosaic 1.0 \
    --seed "$SEED" \
    --no-plots \
    2>&1 | tee "$ROOT/runs/${PHASE2_NAME}.log"

  require_file "$run_dir/weights/best.pt"
  echo "REPORT_PHASE2_COMPLETE=$run_dir/weights/best.pt"
}

require_file "$DATA_YAML"
require_file "$BASE_MODEL"

case "$MODE" in
  phase1) run_phase1 ;;
  phase2) run_phase2 ;;
  all) run_phase1; run_phase2 ;;
  *)
    echo "Usage: bash scripts/run_report_retrain.sh [phase1|phase2|all]" >&2
    exit 2
    ;;
esac
