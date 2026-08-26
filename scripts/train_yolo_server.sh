#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SERVER_GPUS="${SERVER_GPUS:-0}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-1024}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-4}"
GPU_MEMORY_FRACTION="${GPU_MEMORY_FRACTION:-0.45}"
RUN_NAME="${1:-yolo26_server_b${BATCH}_img${IMGSZ}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-}"
PRETRAINED="${PRETRAINED:-}"
PLOTS="${PLOTS:-false}"

IFS=',' read -r -a GPU_LIST <<< "$SERVER_GPUS"
TRAIN_DEVICES=""
for GPU_INDEX in "${!GPU_LIST[@]}"; do
  if [ -z "$TRAIN_DEVICES" ]; then
    TRAIN_DEVICES="$GPU_INDEX"
  else
    TRAIN_DEVICES="${TRAIN_DEVICES},${GPU_INDEX}"
  fi
done

export CUDA_VISIBLE_DEVICES="$SERVER_GPUS"
export YOLO_CONFIG_DIR="$PWD/.ultralytics_config"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p runs

echo "project=$PWD"
echo "server_gpus=$SERVER_GPUS"
echo "train_devices=$TRAIN_DEVICES"
echo "epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH workers=$WORKERS gpu_memory_fraction=$GPU_MEMORY_FRACTION"
echo "model=${MODEL:-default} pretrained=${PRETRAINED:-none} plots=$PLOTS"
nvidia-smi -i "$SERVER_GPUS"

"$PYTHON_BIN" scripts/check_yolo_setup.py

nvidia-smi -i "$SERVER_GPUS" -l 10 > "runs/${RUN_NAME}_nvidia_smi.log" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" >/dev/null 2>&1 || true' EXIT

MODEL_ARGS=()
if [ -n "$MODEL" ]; then
  MODEL_ARGS=(--model "$MODEL")
fi

PRETRAINED_ARGS=()
if [ -n "$PRETRAINED" ]; then
  PRETRAINED_ARGS=(--pretrained "$PRETRAINED")
fi

PLOTS_ARG="--no-plots"
if [ "$PLOTS" = "true" ]; then
  PLOTS_ARG="--plots"
fi

"$PYTHON_BIN" scripts/train_yolo_baseline.py \
  "${MODEL_ARGS[@]}" \
  "${PRETRAINED_ARGS[@]}" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --device "$TRAIN_DEVICES" \
  --gpu-memory-fraction "$GPU_MEMORY_FRACTION" \
  --name "$RUN_NAME" \
  "$PLOTS_ARG" \
  2>&1 | tee "runs/${RUN_NAME}.log"
