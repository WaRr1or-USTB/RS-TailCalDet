#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_source="$project_root/runs/detect/yolo26x_report60_cal10_e120_s2026_b6/weights/best.pt"
threshold_source="$project_root/runs/detect/report_phase1_thresholds_r925_s2026/class_thresholds.json"

test -f "$model_source"
test -f "$threshold_source"
command -v conda >/dev/null

mkdir -p "$project_root/models"
cp "$model_source" "$project_root/models/best.pt"
cp "$threshold_source" "$project_root/models/class_thresholds.json"
conda env export --no-builds | sed '/^prefix:/d' > "$project_root/environment.yml"

if grep -Eiq 'win-64|pywin32|^[[:space:]]*prefix:|[A-Za-z]:\\' "$project_root/environment.yml"; then
    echo "ERROR: environment.yml contains Windows-only content" >&2
    exit 1
fi

python -m json.tool "$project_root/models/class_thresholds.json" >/dev/null
sha256sum \
    "$project_root/models/best.pt" \
    "$project_root/models/class_thresholds.json" \
    "$project_root/environment.yml"
echo "DOCKER_DELIVERY_ASSETS=PASSED"
