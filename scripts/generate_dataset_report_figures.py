"""Generate report-ready dataset figures from the frozen 60/10/30 split."""

from __future__ import annotations

import ast
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data_v2"
SPLIT_ROOT = DATA_ROOT / "report_split_v16"
OUTPUT_DIR = ROOT / "docs" / "results"
ASSIGNMENTS = SPLIT_ROOT / "assignments.csv"
SUMMARY = SPLIT_ROOT / "split_summary.json"

NAVY = "#163B65"
BLUE = "#3274A1"
TEAL = "#2A9D8F"
ORANGE = "#E07A5F"
RED = "#C44536"
GOLD = "#D4A72C"
GRAY = "#667085"
LIGHT = "#E8EEF4"
TEXT = "#17212B"
WHITE = "#FFFFFF"
SPLIT_COLORS = {"train": BLUE, "calibration": GOLD, "validation": TEAL}
CATEGORY_COLORS = {"ship": BLUE, "aircraft": TEAL, "vehicle": ORANGE}

# Restrained editorial palette for paper figures. The existing representative
# sample plate intentionally keeps the original annotation colors.
PAPER_INK = "#263445"
PAPER_MUTED = "#667384"
PAPER_GRID = "#E5E9EE"
PAPER_BLUE = "#476F95"
PAPER_BLUE_LIGHT = "#9CB4C9"
PAPER_SLATE = "#6F8091"
PAPER_WARM = "#C87958"
PAPER_PALE = "#F6F8FA"
PAPER_SPLIT_COLORS = {
    "train": PAPER_BLUE,
    "calibration": PAPER_SLATE,
    "validation": PAPER_WARM,
}
PAPER_CATEGORY_COLORS = {
    "ship": PAPER_SLATE,
    "aircraft": PAPER_BLUE,
    "vehicle": PAPER_WARM,
}
PAPER_SCALE_COLORS = {
    "small": PAPER_WARM,
    "medium": PAPER_BLUE_LIGHT,
    "large": PAPER_BLUE,
}

# Compact, color-blind-friendly palette used by the report's paper figures.
NPG_BLUE = "#3C5488"
NPG_RED = "#E64B35"
NPG_TEAL = "#00A087"
NPG_CYAN = "#4DBBD5"
FIG_INK = "#252A34"
FIG_MUTED = "#6B7280"
FIG_GRID = "#D9DEE7"
FIG_BG = "#FFFFFF"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/simhei.ttf" if bold else "C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def latin_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


F_TITLE = font(48, True)
F_SUBTITLE = font(25)
F_PANEL = font(30, True)
F_AXIS = font(22)
F_TICK = font(19)
F_SMALL = font(17)
FIG_TITLE = font(36, True)
FIG_SUBTITLE = font(20)
FIG_LABEL = font(20)
FIG_LABEL_BOLD = font(20, True)
FIG_TICK = font(16)
FIG_SMALL = font(15)
PAPER_LATIN = latin_font(19)
PAPER_LATIN_SMALL = latin_font(16)
PAPER_LATIN_BOLD = latin_font(19, True)


def canvas(title: str, subtitle: str, width: int = 2200, height: int = 1300) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((80, 45), title, fill=NAVY, font=F_TITLE)
    draw.text((82, 112), subtitle, fill=GRAY, font=F_SUBTITLE)
    return image, draw


def paper_canvas(
    title: str,
    subtitle: str,
    width: int = 2200,
    height: int = 1250,
) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((105, 62), title, fill=PAPER_INK, font=F_TITLE)
    draw.text((107, 128), subtitle, fill=PAPER_MUTED, font=F_SUBTITLE)
    draw.line((105, 180, width - 105, 180), fill=PAPER_GRID, width=2)
    return image, draw


def figure_canvas(
    title: str,
    subtitle: str,
    width: int,
    height: int,
) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), FIG_BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 42), title, fill=FIG_INK, font=FIG_TITLE)
    draw.text((72, 94), subtitle, fill=FIG_MUTED, font=FIG_SUBTITLE)
    draw.line((70, 142, width - 70, 142), fill=FIG_GRID, width=2)
    return image, draw


def mix_color(color: str, amount: float) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    rgb = tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))
    return tuple(round(255 - (255 - channel) * amount) for channel in rgb)


def rotated_text(
    image: Image.Image,
    xy: tuple[int, int],
    text: str,
    color: str,
    angle: float = 55,
) -> None:
    box = ImageDraw.Draw(image).textbbox((0, 0), text, font=FIG_SMALL)
    layer = Image.new("RGBA", (box[2] - box[0] + 12, box[3] - box[1] + 12), (255, 255, 255, 0))
    ImageDraw.Draw(layer).text((6, 4), text, fill=color, font=FIG_SMALL)
    layer = layer.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(layer, xy, layer)


def centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    y: float,
    text: str,
    fill: str,
    text_font: ImageFont.ImageFont,
) -> None:
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((center_x - (box[2] - box[0]) / 2, y), text, fill=fill, font=text_font)


def finish(image: Image.Image, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_DIR / filename, format="PNG", dpi=(300, 300), optimize=True)


def load_inputs() -> tuple[dict, list[dict[str, str]]]:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if isinstance(summary["class_names"], dict):
        summary["class_names"] = [summary["class_names"][str(index)] for index in range(len(summary["class_names"]))]
    with ASSIGNMENTS.open(encoding="utf-8-sig", newline="") as handle:
        assignments = list(csv.DictReader(handle))
    return summary, assignments


def official_rows(assignments: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in assignments if row["sample_kind"] == "official"]


def class_counts_by_split(assignments: list[dict[str, str]], class_count: int) -> dict[str, list[int]]:
    counts = {split: [0] * class_count for split in ("train", "calibration", "validation")}
    for row in official_rows(assignments):
        values = ast.literal_eval(row["class_counts"])
        counts[row["assigned_split"]] = [a + int(b) for a, b in zip(counts[row["assigned_split"]], values)]
    return counts


def local_image_path(server_path: str) -> Path:
    normalized = server_path.replace("\\", "/")
    marker = "/data_v2/"
    if marker not in normalized:
        raise ValueError(f"Cannot map image path: {server_path}")
    return DATA_ROOT / normalized.split(marker, 1)[1]


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_index = parts.index("images")
    parts[image_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def parse_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        labels.append((int(float(values[0])), *(float(value) for value in values[1:5])))
    return labels


def generate_split_figure(summary: dict) -> None:
    counts = summary["official_split_counts"]
    ratios = summary["official_split_ratios"]
    labels = [("train", "训练集"), ("calibration", "校准集"), ("validation", "锁定验证集")]
    width, height = 1800, 720
    image, draw = figure_canvas(
        "官方数据的场景级互斥划分",
        "60%训练 / 10%校准 / 30%锁定验证；同一场景组仅进入一个子集。",
        width,
        height,
    )
    split_colors = {"train": NPG_BLUE, "calibration": NPG_TEAL, "validation": NPG_RED}
    left, right = 105, 1695
    bar_top, bar_bottom = 250, 340
    x = left
    total = sum(counts.values())
    for key, name in labels:
        segment_width = (right - left) * counts[key] / total
        draw.rectangle((x, bar_top, x + segment_width, bar_bottom), fill=split_colors[key])
        centered_text(draw, x + segment_width / 2, bar_top + 28, f"{ratios[key] * 100:.1f}%", WHITE, FIG_LABEL_BOLD)
        x += segment_width

    column_width = (right - left) / 3
    for index, (key, name) in enumerate(labels):
        center = left + (index + 0.5) * column_width
        marker_x = center - 145
        draw.ellipse((marker_x, 421, marker_x + 16, 437), fill=split_colors[key])
        draw.text((marker_x + 30, 409), name, fill=FIG_INK, font=FIG_LABEL_BOLD)
        draw.text((marker_x + 30, 455), f"{counts[key]:,} 张", fill=FIG_INK, font=font(30, True))
        draw.text((marker_x + 30, 502), f"{ratios[key] * 100:.2f}%", fill=split_colors[key], font=FIG_LABEL)

    draw.line((left, 575, right, 575), fill=FIG_GRID, width=2)
    draw.text((left, 606), "场景交叉：train/cal = 0，train/val = 0，cal/val = 0", fill=FIG_MUTED, font=FIG_SMALL)
    footer = f"N = {summary['official_images']:,} images    seed = {summary['seed']}    外部与派生样本仅用于训练"
    footer_width = draw.textbbox((0, 0), footer, font=FIG_SMALL)[2]
    draw.text((right - footer_width, 606), footer, fill=FIG_MUTED, font=FIG_SMALL)
    finish(image, "dataset_split.png")


def generate_class_distribution(summary: dict, counts: dict[str, list[int]]) -> None:
    names = summary["class_names"]
    values = counts["train"]
    ordered = sorted(range(len(names)), key=lambda index: values[index], reverse=True)
    width, height = 2200, 1320
    image, draw = figure_canvas(
        "官方训练集25类目标长尾分布",
        "类别按标注框数量降序排列；纵轴为对数刻度，仅统计60%官方训练子集。",
        width,
        height,
    )
    left, right = 125, 2080
    top, bottom = 210, 930
    maximum = max(values)
    positive = [value for value in values if value > 0]
    ratio = maximum / min(positive)
    axis_min, axis_max = 1, 2000
    for tick in (1, 10, 100, 1000):
        y = bottom - math.log10(tick / axis_min) / math.log10(axis_max / axis_min) * (bottom - top)
        draw.line((left, y, right, y), fill=FIG_GRID, width=2)
        draw.text((65, y - 10), f"{tick:,}", fill=FIG_MUTED, font=FIG_TICK)
    draw.line((left, bottom, right, bottom), fill=FIG_INK, width=2)

    slot = (right - left) / len(ordered)
    bar_width = slot * 0.70
    for rank, class_id in enumerate(ordered):
        center = left + (rank + 0.5) * slot
        if class_id <= 3:
            color = NPG_TEAL
        elif class_id <= 23:
            color = NPG_BLUE
        else:
            color = NPG_RED
        bar_top = bottom - math.log10(values[class_id] / axis_min) / math.log10(axis_max / axis_min) * (bottom - top)
        draw.rectangle((center - bar_width / 2, bar_top, center + bar_width / 2, bottom), fill=color)
        value_text = f"{values[class_id]:,}"
        value_box = draw.textbbox((0, 0), value_text, font=FIG_SMALL)
        draw.text((center - (value_box[2] - value_box[0]) / 2, bar_top - 27), value_text, fill=FIG_MUTED, font=FIG_SMALL)
        rotated_text(image, (round(center - 13), bottom + 15), f"{class_id:02d} {names[class_id]}", FIG_INK)

    legend_y = 1190
    legend = [("舰船", NPG_TEAL), ("飞机", NPG_BLUE), ("车辆/FSC", NPG_RED)]
    legend_x = 670
    for label, color in legend:
        draw.rectangle((legend_x, legend_y, legend_x + 24, legend_y + 24), fill=color)
        draw.text((legend_x + 36, legend_y - 2), label, fill=FIG_INK, font=FIG_LABEL)
        legend_x += 250
    draw.text((125, 1242), f"imbalance ratio = {ratio:.1f}:1", fill=NPG_RED, font=FIG_SMALL)
    finish(image, "class_distribution.png")


def generate_broad_distribution(counts: dict[str, list[int]]) -> None:
    split_labels = [("train", "训练"), ("calibration", "校准"), ("validation", "验证")]
    categories = [
        ("舰船", "ship", range(0, 4)),
        ("飞机", "aircraft", range(4, 24)),
        ("车辆", "vehicle", range(24, 25)),
    ]
    totals = {
        split: [sum(counts[split][class_id] for class_id in class_ids) for _, _, class_ids in categories]
        for split, _ in split_labels
    }
    width, height = 1800, 880
    image, draw = figure_canvas(
        "官方数据三大类标注规模",
        "三个子集共享对数横轴；条末给出原始标注框数量。",
        width,
        height,
    )
    panel_lefts = [80, 650, 1220]
    panel_width = 500
    category_colors = [NPG_TEAL, NPG_BLUE, NPG_RED]
    for panel_index, ((split, split_name), panel_left) in enumerate(zip(split_labels, panel_lefts)):
        plot_left, plot_right = panel_left + 105, panel_left + panel_width - 18
        plot_top, plot_bottom = 265, 650
        centered_text(draw, panel_left + panel_width / 2, 185, split_name, FIG_INK, FIG_LABEL_BOLD)
        if panel_index:
            draw.line((panel_left - 35, 180, panel_left - 35, 740), fill=FIG_GRID, width=2)
        for tick in (10, 100, 1000, 10000):
            x = plot_left + math.log10(tick / 10) / 3 * (plot_right - plot_left)
            draw.line((x, plot_top, x, plot_bottom), fill=FIG_GRID, width=1)
            centered_text(draw, x, plot_bottom + 18, f"{tick:,}", FIG_MUTED, FIG_TICK)
        for row, ((category_name, _, _), value, color) in enumerate(zip(categories, totals[split], category_colors)):
            y = plot_top + row * 125 + 22
            draw.text((panel_left + 5, y + 10), category_name, fill=FIG_INK, font=FIG_LABEL)
            x1 = plot_left + max(0, math.log10(max(value, 10) / 10) / 3) * (plot_right - plot_left)
            draw.line((plot_left, y + 21, x1, y + 21), fill=color, width=18)
            draw.ellipse((x1 - 13, y + 8, x1 + 13, y + 34), fill=color)
            value_text = f"{value:,}"
            value_width = draw.textbbox((0, 0), value_text, font=FIG_SMALL)[2]
            value_x = x1 + 20 if x1 + value_width + 20 <= plot_right else x1 - value_width - 20
            draw.text((value_x, y + 4), value_text, fill=FIG_INK, font=FIG_SMALL)
        centered_text(draw, panel_left + panel_width / 2, 730, f"总计 {sum(totals[split]):,} 个标注框", FIG_MUTED, FIG_SMALL)
    finish(image, "broad_category_distribution.png")


def scale_counts(assignments: list[dict[str, str]]) -> tuple[dict[str, Counter], int]:
    counts: dict[str, Counter] = defaultdict(Counter)
    scanned = 0
    for row in official_rows(assignments):
        image_path = local_image_path(row["image"])
        labels = parse_labels(label_path(image_path))
        with Image.open(image_path) as image:
            width, height = image.size
        for _, _, _, box_width, box_height in labels:
            area = box_width * width * box_height * height
            scale = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
            counts[row["assigned_split"]][scale] += 1
        scanned += 1
    return counts, scanned


def generate_scale_distribution(assignments: list[dict[str, str]]) -> dict[str, Counter]:
    counts, scanned = scale_counts(assignments)
    scales = [("small", "小目标\n<32×32"), ("medium", "中目标\n32×32–96×96"), ("large", "大目标\n>96×96")]
    split_labels = [("train", "训练"), ("calibration", "校准"), ("validation", "验证")]
    percentages = {
        split: [counts[split][key] / max(sum(counts[split].values()), 1) for key, _ in scales]
        for split, _ in split_labels
    }
    width, height = 1800, 830
    image, draw = figure_canvas(
        "官方数据目标尺度分布",
        "单元格同时给出占比与标注框数量；尺度按原图像素面积划分。",
        width,
        height,
    )
    left, top = 300, 245
    cell_width, cell_height = 440, 145
    scale_colors = {"small": NPG_RED, "medium": NPG_CYAN, "large": NPG_BLUE}
    column_labels = ["小目标  <32×32", "中目标  32×32–96×96", "大目标  >96×96"]
    for column, label in enumerate(column_labels):
        centered_text(draw, left + column * cell_width + cell_width / 2, 185, label, FIG_INK, FIG_LABEL_BOLD)
    for row, (split, split_name) in enumerate(split_labels):
        y = top + row * cell_height
        draw.text((105, y + 54), split_name, fill=FIG_INK, font=FIG_LABEL_BOLD)
        for column, (key, _) in enumerate(scales):
            x = left + column * cell_width
            percentage = percentages[split][column]
            fill = mix_color(scale_colors[key], 0.18 + 0.82 * min(percentage / 0.65, 1.0))
            draw.rectangle((x, y, x + cell_width - 8, y + cell_height - 8), fill=fill, outline=WHITE, width=3)
            text_color = WHITE if percentage > 0.32 else FIG_INK
            centered_text(draw, x + (cell_width - 8) / 2, y + 31, f"{percentage * 100:.1f}%", text_color, font(30, True))
            centered_text(draw, x + (cell_width - 8) / 2, y + 82, f"n = {counts[split][key]:,}", text_color, FIG_SMALL)
        total = sum(counts[split].values())
        draw.text((left + 3 * cell_width + 30, y + 43), f"N = {total:,}", fill=FIG_MUTED, font=FIG_LABEL)
    draw.text((105, 735), f"已扫描官方影像 {scanned:,} 张；框面积由原始图像尺寸与YOLO归一化标注计算。", fill=FIG_MUTED, font=FIG_SMALL)
    finish(image, "object_scale_distribution.png")
    return counts


def generate_dataset_statistics_overview(
    assignments: list[dict[str, str]],
    counts: dict[str, list[int]],
) -> None:
    """Create an LVIS Figure 6-style three-panel dataset statistics plate."""

    split_labels = [("train", "Train", "#1F77B4"), ("calibration", "Calibration", "#FF7F0E"), ("validation", "Validation", "#2CA02C")]
    category_cardinality: dict[str, Counter] = defaultdict(Counter)
    relative_sizes: dict[str, list[float]] = defaultdict(list)
    for row in official_rows(assignments):
        split = row["assigned_split"]
        per_class = ast.literal_eval(row["class_counts"])
        category_cardinality[split][sum(int(value) > 0 for value in per_class)] += 1
        for _, _, _, box_width, box_height in parse_labels(label_path(local_image_path(row["image"]))):
            relative_sizes[split].append(math.sqrt(max(box_width * box_height, 0.0)))

    width, height = 2100, 850
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    panel_lefts = [90, 750, 1410]
    panel_width = 545
    plot_top, plot_bottom = 80, 500

    def axes(panel_left: int, x_ticks: list[float], x_labels: list[str], y_ticks: list[float], y_labels: list[str]) -> tuple[int, int]:
        plot_left, plot_right = panel_left + 75, panel_left + panel_width - 15
        for value, label in zip(y_ticks, y_labels):
            y = plot_bottom - (math.log10(value) - math.log10(y_ticks[0])) / (math.log10(y_ticks[-1]) - math.log10(y_ticks[0])) * (plot_bottom - plot_top)
            draw.line((plot_left, y, plot_right, y), fill="#E6E6E6", width=1)
            label_width = draw.textbbox((0, 0), label, font=PAPER_LATIN_SMALL)[2]
            draw.text((plot_left - label_width - 10, y - 9), label, fill="#333333", font=PAPER_LATIN_SMALL)
        for index, (value, label) in enumerate(zip(x_ticks, x_labels)):
            x = plot_left + index / max(len(x_ticks) - 1, 1) * (plot_right - plot_left)
            draw.line((x, plot_top, x, plot_bottom), fill="#EEEEEE", width=1)
            centered_text(draw, x, plot_bottom + 10, label, "#333333", PAPER_LATIN_SMALL)
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#777777", width=2)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#777777", width=2)
        return plot_left, plot_right

    def y_map(value: float, y_min: float, y_max: float) -> float:
        return plot_bottom - (math.log10(value) - math.log10(y_min)) / (math.log10(y_max) - math.log10(y_min)) * (plot_bottom - plot_top)

    # (a) Distribution of the number of represented categories per image.
    panel_left = panel_lefts[0]
    max_categories = max(max(counter) for counter in category_cardinality.values())
    x_values_a = list(range(1, max_categories + 1))
    plot_left, plot_right = axes(panel_left, x_values_a, [str(value) for value in x_values_a], [0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    for split, label, color in split_labels:
        total_images = sum(category_cardinality[split].values())
        points = []
        for index, category_count in enumerate(x_values_a):
            percentage = category_cardinality[split][category_count] / max(total_images, 1) * 100
            if percentage <= 0:
                continue
            x = plot_left + index / max(len(x_values_a) - 1, 1) * (plot_right - plot_left)
            points.append((x, y_map(percentage, 0.01, 100)))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    draw.text((panel_left - 8, 44), "Percent of images", fill="#333333", font=PAPER_LATIN_SMALL)
    centered_text(draw, (plot_left + plot_right) / 2, 540, "Number of categories", "#333333", PAPER_LATIN)
    legend_y = 125
    for split, label, color in split_labels:
        draw.line((plot_left + 22, legend_y, plot_left + 60, legend_y), fill=color, width=3)
        draw.text((plot_left + 70, legend_y - 10), label, fill="#333333", font=PAPER_LATIN_SMALL)
        legend_y += 31

    # (b) Sorted category instance counts on the official training split.
    panel_left = panel_lefts[1]
    sorted_counts = sorted(counts["train"], reverse=True)
    x_ticks_b = [0, 5, 10, 15, 20, 24]
    plot_left, plot_right = axes(panel_left, x_ticks_b, [str(value) for value in x_ticks_b], [1, 10, 100, 1000, 10000], ["1", "10", "100", "1,000", "10,000"])
    points = []
    for index, value in enumerate(sorted_counts):
        x = plot_left + index / (len(sorted_counts) - 1) * (plot_right - plot_left)
        points.append((x, y_map(value, 1, 10000)))
    draw.line(points, fill="#1F77B4", width=3)
    for x, y in points:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#FF7F0E")
    draw.text((panel_left - 8, 44), "Number of instances", fill="#333333", font=PAPER_LATIN_SMALL)
    centered_text(draw, (plot_left + plot_right) / 2, 540, "Sorted category index", "#333333", PAPER_LATIN)

    # (c) Relative object size distribution, analogous to LVIS mask-size statistics.
    panel_left = panel_lefts[2]
    bin_count = 20
    x_values_c = [(index + 0.5) / bin_count for index in range(bin_count)]
    x_ticks_c = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plot_left, plot_right = axes(panel_left, x_ticks_c, [f"{value:.1f}" for value in x_ticks_c], [0.01, 0.1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    for split, label, color in split_labels:
        histogram = [0] * bin_count
        for value in relative_sizes[split]:
            histogram[min(int(value * bin_count), bin_count - 1)] += 1
        total_instances = sum(histogram)
        points = []
        for index, count in enumerate(histogram):
            percentage = count / max(total_instances, 1) * 100
            if percentage < 0.01:
                continue
            x = plot_left + x_values_c[index] * (plot_right - plot_left)
            points.append((x, y_map(percentage, 0.01, 100)))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
    draw.text((panel_left - 8, 44), "Percent of instances", fill="#333333", font=PAPER_LATIN_SMALL)
    centered_text(draw, (plot_left + plot_right) / 2, 540, "Relative bounding-box size", "#333333", PAPER_LATIN)
    legend_y = 125
    for split, label, color in split_labels:
        draw.line((plot_left + 245, legend_y, plot_left + 283, legend_y), fill=color, width=3)
        draw.text((plot_left + 293, legend_y - 10), label, fill="#333333", font=PAPER_LATIN_SMALL)
        legend_y += 31

    captions = [
        "(a) 每幅影像所含类别数分布。",
        "(b) 每类实例数揭示明显长尾。",
        "(c) 相对目标框尺度分布。",
    ]
    for panel_left, caption in zip(panel_lefts, captions):
        draw.text((panel_left, 615), caption, fill="#222222", font=FIG_LABEL)
    centered_text(draw, width / 2, 760, "数据集统计", "#222222", FIG_LABEL_BOLD)
    finish(image, "dataset_statistics_overview.png")


def draw_annotated_sample(image_path: Path, names: list[str], size: tuple[int, int]) -> Image.Image:
    with Image.open(image_path) as source:
        source = source.convert("RGB")
    width, height = source.size
    labels = parse_labels(label_path(image_path))
    draw = ImageDraw.Draw(source)
    line_width = max(3, round(min(width, height) / 220))
    for class_id, x_center, y_center, box_width, box_height in labels:
        x0 = (x_center - box_width / 2) * width
        y0 = (y_center - box_height / 2) * height
        x1 = (x_center + box_width / 2) * width
        y1 = (y_center + box_height / 2) * height
        category = "ship" if class_id <= 3 else "aircraft" if class_id <= 23 else "vehicle"
        color = CATEGORY_COLORS[category]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)
        label = names[class_id]
        text_box = draw.textbbox((0, 0), label, font=font(max(14, round(min(width, height) / 42)), True))
        text_width, text_height = text_box[2], text_box[3]
        text_y = max(0, y0 - text_height - 8)
        draw.rectangle((x0, text_y, x0 + text_width + 10, text_y + text_height + 6), fill=color)
        draw.text((x0 + 5, text_y + 1), label, fill=WHITE, font=font(max(14, round(min(width, height) / 42)), True))
    source.thumbnail(size, Image.Resampling.LANCZOS)
    return source


def generate_sample_figure(summary: dict) -> None:
    names = summary["class_names"]
    samples = [
        ("images/train/02-PAN-20250405-013-430-L00000001339-CCD26_4_crop3.jpg", "舰船样本：HM"),
        ("images/train/01-PAN-20250210-441-302-L00000100998-CCD26_5_crop1.jpg", "舰船样本：MS"),
        ("images/train/MAR20_2485.jpg", "飞机样本：A1_SU-35"),
        ("images/train/MAR20_2342.jpg", "飞机样本：A15_F-22"),
        ("images/train/fsc_AGZ-N42.80-E141.65-lv20-Bing_crop0001.jpg", "车辆样本：FSC"),
        ("images/train/fsc_AGZ-N42.80-E141.65-lv20-Google_crop0001.jpg", "小目标与复杂背景"),
    ]
    width, height = 2400, 1760
    image, draw = canvas(
        "真实训练样本与标注示例",
        "全部样本来自60%官方训练子集；彩色框为原始YOLO水平框标注。",
        width,
        height,
    )
    panel_width, panel_height = 720, 675
    start_x, start_y = 90, 205
    gap_x, gap_y = 85, 90
    for index, (relative_path, caption) in enumerate(samples):
        row, column = divmod(index, 3)
        x0 = start_x + column * (panel_width + gap_x)
        y0 = start_y + row * (panel_height + gap_y)
        draw.rounded_rectangle((x0, y0, x0 + panel_width, y0 + panel_height), radius=18, fill="#F8FAFC", outline=LIGHT, width=3)
        annotated = draw_annotated_sample(DATA_ROOT / relative_path, names, (650, 570))
        paste_x = int(x0 + (panel_width - annotated.width) / 2)
        paste_y = int(y0 + 28 + (570 - annotated.height) / 2)
        image.paste(annotated, (paste_x, paste_y))
        caption_width = draw.textbbox((0, 0), caption, font=F_AXIS)[2]
        draw.text((x0 + (panel_width - caption_width) / 2, y0 + 615), caption, fill=TEXT, font=F_AXIS)
    finish(image, "representative_training_samples.png")


def write_statistics(summary: dict, counts: dict[str, list[int]], scales: dict[str, Counter]) -> None:
    fields = ["split", "images", "objects", "ship_objects", "aircraft_objects", "vehicle_objects", "small", "medium", "large"]
    with (OUTPUT_DIR / "dataset_statistics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split in ("train", "calibration", "validation"):
            values = counts[split]
            writer.writerow(
                {
                    "split": split,
                    "images": summary["official_split_counts"][split],
                    "objects": sum(values),
                    "ship_objects": sum(values[0:4]),
                    "aircraft_objects": sum(values[4:24]),
                    "vehicle_objects": values[24],
                    "small": scales[split]["small"],
                    "medium": scales[split]["medium"],
                    "large": scales[split]["large"],
                }
            )


def main() -> int:
    for path in (ASSIGNMENTS, SUMMARY, DATA_ROOT / "images", DATA_ROOT / "labels"):
        if not path.exists():
            raise FileNotFoundError(path)
    summary, assignments = load_inputs()
    counts = class_counts_by_split(assignments, len(summary["class_names"]))
    generate_split_figure(summary)
    generate_class_distribution(summary, counts)
    generate_broad_distribution(counts)
    scales = generate_scale_distribution(assignments)
    generate_dataset_statistics_overview(assignments, counts)
    sample_figure = OUTPUT_DIR / "representative_training_samples.png"
    if not sample_figure.exists():
        generate_sample_figure(summary)
    write_statistics(summary, counts, scales)
    print(f"DATASET_REPORT_FIGURES={OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
