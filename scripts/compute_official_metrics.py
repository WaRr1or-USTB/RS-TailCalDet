"""Compute V1.6 category-macro ranking diagnostics with tiny-GT filtering.

Scoring alignment (V1.6):
  - The hard gate uses merged overall counts: recall >= 0.85 and FDR <= 0.20.
  - Category/subclass macro metrics are ranking diagnostics, not the hard gate.
  - Tiny targets (short side < threshold, visually unresolvable) need not be labeled.

Inputs:
  --edge-csv   : audit_edge_targets.py output (per-GT matched flag + box)
  --protocol   : protocol_metrics.json (per-subclass fp at threshold 0.001)

Usage:
    python scripts/compute_official_metrics.py \
        --edge-csv runs/detect/edge_audit_hrsc/val_scene_grouped/edge_targets.csv \
        --protocol runs/detect/final_hrsc_val_scene_grouped/protocol_metrics.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BROAD = {"ship": list(range(0, 4)), "aircraft": list(range(4, 24)), "vehicle": [24]}
CLASS_NAMES = {
    0: "HM", 1: "LQS", 2: "QHS", 3: "MS", 4: "SU-35", 5: "C-130", 6: "C-17", 7: "C-5",
    8: "F-16", 9: "TU-160", 10: "E-3", 11: "B-52", 12: "P-3C", 13: "B-1B", 14: "E-8",
    15: "TU-22", 16: "F-15", 17: "KC-135", 18: "F-22", 19: "FA-18", 20: "TU-95",
    21: "KC-10", 22: "SU-34", 23: "SU-24", 24: "FSC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-csv", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--tiny-short", type=float, default=30.0, help="tiny = short side below this (px)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = json.loads((PROJECT_ROOT / args.protocol).read_text(encoding="utf-8"))
    subclass_fp = {
        int(cls): int(row["fp"])
        for cls, row in protocol["strict_25_subclass"]["by_subclass"].items()
    }

    rows: List[Dict] = []
    with (PROJECT_ROOT / args.edge_csv).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            x1, y1, x2, y2 = (float(v) for v in json.loads(r["box"]))
            r["_short"] = min(x2 - x1, y2 - y1)
            r["_matched"] = r["matched"] == "1"
            rows.append(r)

    def tiny_filtered(cls: int) -> Tuple[int, int, int]:
        """Return (gt', tp', fp) after removing tiny GT."""
        gt = sum(1 for r in rows if int(r["cls"]) == cls and r["_short"] >= args.tiny_short)
        tp = sum(1 for r in rows if int(r["cls"]) == cls and r["_short"] >= args.tiny_short and r["_matched"])
        return gt, tp, subclass_fp.get(cls, 0)

    print(f"tiny threshold: short side < {args.tiny_short}px removed\n", flush=True)
    subclass_metrics: Dict[int, Dict] = {}
    for cls in range(25):
        gt, tp, fp = tiny_filtered(cls)
        fn = gt - tp
        recall = tp / gt if gt else None
        fdr = fp / (tp + fp) if (tp + fp) else None
        subclass_metrics[cls] = {"gt": gt, "tp": tp, "fp": fp, "fn": fn, "recall": recall, "fdr": fdr}
        print(
            f"{CLASS_NAMES[cls]:<8} gt={gt:4d} tp={tp:4d} fp={fp:3d} fn={fn:3d} "
            f"recall={recall:.4f} fdr={fdr:.4f}",
            flush=True,
        )

    print("\n=== Category macro (equal-weight over subclasses) ===", flush=True)
    cat_recalls: Dict[str, float] = {}
    cat_fdrs: Dict[str, float] = {}
    for cat, ids in BROAD.items():
        recs = [subclass_metrics[c]["recall"] for c in ids if subclass_metrics[c]["gt"] > 0]
        fdrs = [subclass_metrics[c]["fdr"] for c in ids if subclass_metrics[c]["gt"] > 0]
        cat_recalls[cat] = sum(recs) / len(recs) if recs else None
        cat_fdrs[cat] = sum(fdrs) / len(fdrs) if fdrs else None
        print(f"{cat:<10} macro_recall={cat_recalls[cat]:.4f}  macro_fdr={cat_fdrs[cat]:.4f}", flush=True)

    mean_recall = sum(v for v in cat_recalls.values() if v is not None) / len(cat_recalls)
    mean_fdr = sum(v for v in cat_fdrs.values() if v is not None) / len(cat_fdrs)
    print("\n=== RANKING AUDIT (3 category macros, not the hard gate) ===", flush=True)
    print(f"mean recall = {mean_recall:.4f}", flush=True)
    print(f"mean FDR    = {mean_fdr:.4f}", flush=True)
    print("hard gate must be read from official_overall: recall >= 0.85 and FDR <= 0.20", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
