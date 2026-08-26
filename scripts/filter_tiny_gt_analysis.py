"""Recompute strict recall/FDR after excluding tiny GT (official rule: sub-visible targets need not be labeled).

Uses the existing edge-audit matching output (final model, frozen config) offline —
no re-inference needed. Tiny = short side below a threshold (default 30 px).

Usage:
    python scripts/filter_tiny_gt_analysis.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {24: "FSC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-csv", default="runs/detect/edge_audit/val_scene_grouped/edge_targets.csv")
    parser.add_argument("--short-threshold", type=float, default=30.0)
    return parser.parse_args()


def short_side(box: str) -> float:
    x1, y1, x2, y2 = (float(v) for v in json.loads(box))
    return min(x2 - x1, y2 - y1)


def main() -> int:
    args = parse_args()
    rows: List[Dict] = []
    with (PROJECT_ROOT / args.edge_csv).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["_short"] = short_side(r["box"])
            r["_matched"] = r["matched"] == "1"
            rows.append(r)

    def stats(subset: List[Dict]) -> Tuple[int, int, float, float]:
        tp = sum(1 for r in subset if r["_matched"])
        fn = len(subset) - tp
        fp = sum(1 for r in rows if not r["_matched"] and r not in subset)
        # FDR is prediction-side: FP over (TP+FP) — recompute only when filtering GT of one class.
        return tp, fn, tp / max(1, tp + fn), 0.0

    print(f"short-side threshold = {args.short_threshold} px\n", flush=True)
    for label, subset in (
        ("ALL classes", rows),
        ("FSC only", [r for r in rows if int(r["cls"]) == 24]),
    ):
        tp0, fn0, rec0, _ = stats(subset)
        kept = [r for r in subset if r["_short"] >= args.short_threshold]
        tp1, fn1, rec1, _ = stats(kept)
        removed = len(subset) - len(kept)
        removed_tp = sum(1 for r in subset if r["_short"] < args.short_threshold and r["_matched"])
        print(f"=== {label} ===", flush=True)
        print(f"  before: GT={tp0+fn0} TP={tp0} FN={fn0} recall={rec0:.4f}", flush=True)
        print(
            f"  after : GT={tp1+fn1} TP={tp1} FN={fn1} recall={rec1:.4f} "
            f"(removed tiny GT={removed}, of which TP={removed_tp})",
            flush=True,
        )
        print("", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
