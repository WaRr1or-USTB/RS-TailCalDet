"""Generate report-ready result charts from frozen RS-TailCalDet artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "results"
TRAINING_CSV = ROOT / "runs" / "detect" / "yolo26x_report60_cal10_e120_s2026_b6" / "results.csv"
CALIBRATION_DIR = ROOT / "runs" / "detect" / "report_phase1_thresholds_r925_s2026"
LOCKED_DIRS = {
    "Scene-grouped": ROOT / "runs" / "detect" / "report_v16_locked_r925_val_scene_grouped_s2026",
    "Mixed-stress": ROOT / "runs" / "detect" / "report_v16_locked_r925_val_mixed_stress_s2026",
    "Sparse": ROOT / "runs" / "detect" / "report_v16_locked_r925_val_sparse_s2026",
}

NAVY = "#163B65"
BLUE = "#3274A1"
TEAL = "#2A9D8F"
ORANGE = "#E07A5F"
RED = "#C44536"
GOLD = "#D4A72C"
GRAY = "#6B7280"
LIGHT = "#E8EEF4"
TEXT = "#17212B"
WHITE = "#FFFFFF"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


F_TITLE = font(48, True)
F_SUBTITLE = font(25)
F_PANEL = font(30, True)
F_AXIS = font(22)
F_TICK = font(19)
F_SMALL = font(17)


def canvas(title: str, subtitle: str, width: int = 2200, height: int = 1300) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((80, 45), title, fill=NAVY, font=F_TITLE)
    draw.text((82, 112), subtitle, fill=GRAY, font=F_SUBTITLE)
    return image, draw


def finish(image: Image.Image, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_DIR / filename, format="PNG", dpi=(300, 300), optimize=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(handle)]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_line_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    x_values: Sequence[float],
    series: Sequence[tuple[str, Sequence[float], str]],
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=18, outline=LIGHT, width=3, fill="#FBFCFE")
    draw.text((left + 28, top + 20), title, fill=TEXT, font=F_PANEL)
    plot_left, plot_top = left + 90, top + 88
    plot_right, plot_bottom = right - 30, bottom - 70
    values = [value for _, data, _ in series for value in data]
    lo = min(values) if y_min is None else y_min
    hi = max(values) if y_max is None else y_max
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    for index in range(5):
        y = plot_top + (plot_bottom - plot_top) * index / 4
        value = hi - (hi - lo) * index / 4
        draw.line((plot_left, y, plot_right, y), fill=LIGHT, width=2)
        draw.text((left + 12, y - 11), f"{value:.2f}", fill=GRAY, font=F_SMALL)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=GRAY, width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=GRAY, width=3)
    x_lo, x_hi = min(x_values), max(x_values)
    for name, data, color in series:
        points = []
        for x_value, y_value in zip(x_values, data):
            x = plot_left + (x_value - x_lo) / max(x_hi - x_lo, 1) * (plot_right - plot_left)
            y = plot_bottom - (y_value - lo) / (hi - lo) * (plot_bottom - plot_top)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=5, joint="curve")
        for point in points[:: max(1, len(points) // 18)]:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)
    legend_x = plot_left
    for name, _, color in series:
        draw.line((legend_x, bottom - 34, legend_x + 35, bottom - 34), fill=color, width=6)
        draw.text((legend_x + 45, bottom - 48), name, fill=TEXT, font=F_SMALL)
        legend_x += 205
    draw.text((plot_left, plot_bottom + 18), f"{int(x_lo)}", fill=GRAY, font=F_SMALL)
    draw.text((plot_right - 35, plot_bottom + 18), f"{int(x_hi)}", fill=GRAY, font=F_SMALL)


def grouped_bar_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], str]],
    limit: float | None = None,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=18, outline=LIGHT, width=3, fill="#FBFCFE")
    draw.text((left + 28, top + 20), title, fill=TEXT, font=F_PANEL)
    plot_left, plot_top = left + 90, top + 90
    plot_right, plot_bottom = right - 35, bottom - 120
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=GRAY, width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=GRAY, width=3)
    for index in range(6):
        value = index * 20
        y = plot_bottom - value / 100 * (plot_bottom - plot_top)
        draw.line((plot_left, y, plot_right, y), fill=LIGHT, width=2)
        draw.text((left + 28, y - 11), str(value), fill=GRAY, font=F_SMALL)
    if limit is not None:
        y = plot_bottom - limit / 100 * (plot_bottom - plot_top)
        draw.line((plot_left, y, plot_right, y), fill=RED, width=4)
        draw.text((plot_left + 12, y - 32), f"Limit {limit:.0f}%", fill=RED, font=F_SMALL)
    group_width = (plot_right - plot_left) / len(labels)
    bar_width = min(72, group_width / (len(series) + 0.8))
    for group_index, label in enumerate(labels):
        center = plot_left + (group_index + 0.5) * group_width
        total_width = len(series) * bar_width
        start = center - total_width / 2
        for series_index, (_, values, color) in enumerate(series):
            value = values[group_index] * 100
            x0 = start + series_index * bar_width
            x1 = x0 + bar_width * 0.78
            y0 = plot_bottom - value / 100 * (plot_bottom - plot_top)
            draw.rectangle((x0, y0, x1, plot_bottom), fill=color)
            draw.text((x0 - 3, y0 - 25), f"{value:.1f}", fill=TEXT, font=F_SMALL)
        text_width = draw.textbbox((0, 0), label, font=F_TICK)[2]
        draw.text((center - text_width / 2, plot_bottom + 24), label, fill=TEXT, font=F_TICK)
    legend_x = plot_left
    for name, _, color in series:
        draw.rectangle((legend_x, bottom - 48, legend_x + 26, bottom - 22), fill=color)
        draw.text((legend_x + 36, bottom - 52), name, fill=TEXT, font=F_SMALL)
        legend_x += 220


def generate_training_curves() -> None:
    rows = read_csv(TRAINING_CSV)
    epochs = [float(row["epoch"]) for row in rows]
    image, draw = canvas(
        "Training and Calibration Curves",
        "RS-TailCalDet phase-1 training; validation metrics are measured on the 10% calibration split.",
    )
    panels = [
        ((70, 180, 1070, 700), "Training losses", [("Box", "train/box_loss", BLUE), ("Cls", "train/cls_loss", ORANGE)]),
        ((1130, 180, 2130, 700), "Validation losses", [("Box", "val/box_loss", BLUE), ("Cls", "val/cls_loss", ORANGE)]),
        ((70, 750, 1070, 1240), "Precision and recall", [("Precision", "metrics/precision(B)", TEAL), ("Recall", "metrics/recall(B)", GOLD)]),
        ((1130, 750, 2130, 1240), "Mean average precision", [("mAP50", "metrics/mAP50(B)", BLUE), ("mAP50-95", "metrics/mAP50-95(B)", RED)]),
    ]
    for box, title, definitions in panels:
        series = [(name, [float(row[key]) for row in rows], color) for name, key, color in definitions]
        plot_line_panel(draw, box, title, epochs, series)
    finish(image, "training_curves.png")


def locked_metrics() -> dict[str, dict]:
    return {name: read_json(path / "protocol_metrics.json") for name, path in LOCKED_DIRS.items()}


def generate_overall_chart(metrics: dict[str, dict]) -> None:
    labels = list(metrics)
    recalls = [metrics[label]["official_overall"]["recall"] for label in labels]
    fdrs = [metrics[label]["official_overall"]["fdr"] for label in labels]
    image, draw = canvas(
        "Locked 30% Validation: Overall Gate Metrics",
        "Three broad categories are merged using the official V1.6 matching policy; official hidden-test results may differ.",
    )
    grouped_bar_panel(draw, (70, 190, 1070, 1220), "Overall Recall (%)", labels, [("RS-TailCalDet", recalls, TEAL)], 85)
    grouped_bar_panel(draw, (1130, 190, 2130, 1220), "Overall FDR (%)", labels, [("RS-TailCalDet", fdrs, ORANGE)], 20)
    finish(image, "locked_overall_metrics.png")


def generate_category_chart(metrics: dict[str, dict]) -> None:
    selected = ["Scene-grouped", "Mixed-stress"]
    labels = ["Ship", "Aircraft", "Vehicle"]
    colors = [BLUE, ORANGE]
    recall_series = []
    fdr_series = []
    for name, color in zip(selected, colors):
        categories = metrics[name]["official_by_category"]["categories"]
        recall_series.append((name, [categories[label.lower()]["recall"] for label in labels], color))
        fdr_series.append((name, [categories[label.lower()]["fdr"] for label in labels], color))
    image, draw = canvas(
        "Locked 30% Validation: Broad-category Metrics",
        "Pooled category metrics; sparse layout is omitted because it contains no vehicle ground truth.",
    )
    grouped_bar_panel(draw, (70, 190, 1070, 1220), "Recall by category (%)", labels, recall_series)
    grouped_bar_panel(draw, (1130, 190, 2130, 1220), "FDR by category (%)", labels, fdr_series, 20)
    finish(image, "locked_category_metrics.png")


def generate_subclass_chart(metrics: dict[str, dict]) -> None:
    subclasses = metrics["Scene-grouped"]["strict_25_subclass"]["by_subclass"]
    rows = [(int(class_id), values) for class_id, values in subclasses.items()]
    rows.sort(key=lambda item: item[0])
    width, height = 2400, 2300
    image, draw = canvas(
        "Locked 30% Validation: 25-class Diagnostic Metrics",
        "Exact subclass matching on the scene-grouped layout; these diagnostics are not the overall hard-gate metric.",
        width,
        height,
    )
    left_label = 260
    recall_left, recall_right = 430, 1280
    fdr_left, fdr_right = 1450, 2180
    top, row_height = 210, 76
    draw.text((recall_left, 165), "Recall", fill=NAVY, font=F_PANEL)
    draw.text((fdr_left, 165), "FDR", fill=NAVY, font=F_PANEL)
    for index, (_, values) in enumerate(rows):
        y = top + index * row_height
        if index % 2 == 0:
            draw.rectangle((55, y - 8, width - 55, y + row_height - 10), fill="#F6F8FB")
        name = values["name"]
        draw.text((70, y + 10), name, fill=TEXT, font=F_TICK)
        color = BLUE if index <= 3 else TEAL if index <= 23 else ORANGE
        recall = float(values["recall"])
        fdr = float(values["fdr"])
        draw.rectangle((recall_left, y + 13, recall_left + (recall_right - recall_left) * recall, y + 43), fill=color)
        draw.rectangle((fdr_left, y + 13, fdr_left + (fdr_right - fdr_left) * fdr, y + 43), fill=RED)
        draw.text((recall_right + 15, y + 8), f"{recall * 100:.1f}%", fill=TEXT, font=F_SMALL)
        draw.text((2200, y + 8), f"{fdr * 100:.1f}%", fill=TEXT, font=F_SMALL)
    finish(image, "locked_subclass_metrics.png")


def generate_calibration_chart() -> None:
    trace = read_csv(CALIBRATION_DIR / "optimization_trace.csv")
    rounds = [float(row["round"]) for row in trace]
    recalls = [float(row["recall"]) for row in trace]
    fdrs = [float(row["fdr"]) for row in trace]
    summary = read_json(CALIBRATION_DIR / "summary.json")
    baseline, optimized = summary["baseline"], summary["optimized"]
    image, draw = canvas(
        "Class-threshold Calibration",
        "Thresholds are selected on the independent 10% calibration split; model weights remain frozen.",
    )
    plot_line_panel(
        draw,
        (70, 190, 1300, 1220),
        "Optimization trace",
        rounds,
        [("Recall", recalls, TEAL), ("FDR", fdrs, RED)],
        0,
        1,
    )
    labels = ["Before", "After"]
    grouped_bar_panel(
        draw,
        (1360, 190, 2130, 1220),
        "Before / after (%)",
        labels,
        [
            ("Recall", [baseline["recall"], optimized["recall"]], TEAL),
            ("FDR", [baseline["fdr"], optimized["fdr"]], RED),
        ],
    )
    finish(image, "threshold_calibration.png")


def generate_timing_chart() -> None:
    image, draw = canvas(
        "RTX 3090 Inference Time on 10000 x 10000 Images",
        "Official timing scope: after image read/decode through global NMS; 800 x 800 tiles, 169 tiles per image, FP32.",
    )
    left, top, right, bottom = 150, 220, 2080, 1140
    draw.line((left, top, left, bottom), fill=GRAY, width=3)
    draw.line((left, bottom, right, bottom), fill=GRAY, width=3)
    for seconds in range(0, 21, 5):
        y = bottom - seconds / 20 * (bottom - top)
        draw.line((left, y, right, y), fill=LIGHT, width=2)
        draw.text((75, y - 12), str(seconds), fill=GRAY, font=F_TICK)
    limit_y = top
    draw.line((left, limit_y, right, limit_y), fill=RED, width=5)
    draw.text((right - 210, limit_y + 12), "20 s limit", fill=RED, font=F_AXIS)
    colors = [BLUE, TEAL, ORANGE]
    for split_index, ((name, directory), color) in enumerate(zip(LOCKED_DIRS.items(), colors)):
        rows = read_csv(directory / "per_image_times.csv")
        center = left + (split_index + 0.5) * (right - left) / len(LOCKED_DIRS)
        values = [float(row["seconds"]) for row in rows]
        for index, value in enumerate(values):
            offset = (index - (len(values) - 1) / 2) * 20
            y = bottom - value / 20 * (bottom - top)
            draw.ellipse((center + offset - 9, y - 9, center + offset + 9, y + 9), fill=color)
        maximum = max(values)
        y_max = bottom - maximum / 20 * (bottom - top)
        draw.line((center - 125, y_max, center + 125, y_max), fill=NAVY, width=4)
        draw.text((center - 70, y_max - 38), f"max {maximum:.2f}s", fill=NAVY, font=F_SMALL)
        label_width = draw.textbbox((0, 0), name, font=F_AXIS)[2]
        draw.text((center - label_width / 2, bottom + 30), name, fill=TEXT, font=F_AXIS)
    finish(image, "inference_timing.png")


def write_summary_csv(metrics: dict[str, dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "split", "images", "ground_truth", "tp", "fp", "fn", "overall_recall", "overall_fdr",
        "ship_recall", "ship_fdr", "aircraft_recall", "aircraft_fdr", "vehicle_ground_truth",
        "vehicle_recall", "vehicle_fdr", "mean_seconds", "max_seconds", "max_end_to_end_seconds",
    ]
    with (OUTPUT_DIR / "performance_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, data in metrics.items():
            overall = data["official_overall"]
            categories = data["official_by_category"]["categories"]
            timing = data["v16_ranking_inputs"]["timing"]
            per_image = read_csv(LOCKED_DIRS[name] / "per_image_times.csv")
            writer.writerow(
                {
                    "split": name,
                    "images": timing["images"],
                    "ground_truth": overall["ground_truth"],
                    "tp": overall["tp"],
                    "fp": overall["fp"],
                    "fn": overall["fn"],
                    "overall_recall": overall["recall"],
                    "overall_fdr": overall["fdr"],
                    "ship_recall": categories["ship"]["recall"],
                    "ship_fdr": categories["ship"]["fdr"],
                    "aircraft_recall": categories["aircraft"]["recall"],
                    "aircraft_fdr": categories["aircraft"]["fdr"],
                    "vehicle_ground_truth": categories["vehicle"]["ground_truth"],
                    "vehicle_recall": categories["vehicle"]["recall"],
                    "vehicle_fdr": categories["vehicle"]["fdr"],
                    "mean_seconds": timing["mean_seconds"],
                    "max_seconds": timing["max_seconds"],
                    "max_end_to_end_seconds": max(float(row["end_to_end_seconds"]) for row in per_image),
                }
            )


def verify_inputs(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen artifact(s): " + ", ".join(str(path) for path in missing))


def main() -> int:
    verify_inputs(
        [TRAINING_CSV, CALIBRATION_DIR / "summary.json", CALIBRATION_DIR / "optimization_trace.csv"]
        + [directory / "protocol_metrics.json" for directory in LOCKED_DIRS.values()]
        + [directory / "per_image_times.csv" for directory in LOCKED_DIRS.values()]
    )
    metrics = locked_metrics()
    generate_training_curves()
    generate_overall_chart(metrics)
    generate_category_chart(metrics)
    generate_subclass_chart(metrics)
    generate_calibration_chart()
    generate_timing_chart()
    write_summary_csv(metrics)
    print(f"REPORT_RESULTS={OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
