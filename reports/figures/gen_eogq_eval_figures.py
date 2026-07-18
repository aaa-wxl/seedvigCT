import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.evaluate_eog_corruption import _collect_predictions, _load_model
from seedvig.raw_dataset import RawSeedVIGSequenceDataset


ROOT = Path(__file__).resolve().parent
RUN_ROOT = Path(r"D:\seedvig-CT\runs")
PREFIX = "raw_eeg_eog_cross_eog_query_binary_group_subject_reg_f"
DATA_ROOT = Path(r"D:\eeg-eog\data\SEED-VIG")
CACHE_DIR = Path(r"D:\seedvig-CT\cache\seedvig_raw_eeg")
FONT_DIR = Path(r"C:\Windows\Fonts")


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ("timesbd.ttf" if bold else "times.ttf")), size)


def load_fold(fold, device):
    run_dir = RUN_ROOT / f"{PREFIX}{fold}"
    metrics = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    dataset = RawSeedVIGSequenceDataset(
        root_path=DATA_ROOT,
        cache_dir=CACHE_DIR,
        include_eog=True,
        sequence_length=int(metrics.get("sequence_length", 8)),
        split="test",
        split_strategy="group_subject",
        fold=fold,
        label_mode=metrics.get("label_mode", "binary"),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model = _load_model(run_dir, metrics, "eeg_eog", device)
    pred, target = _collect_predictions(model, loader, device, "eeg_eog", "none", 0.0, 0)
    confusion = np.array(json.loads(metrics["test_confusion_matrix"]), dtype=np.int64)
    return confusion, pred.numpy(), target.numpy()


def xy(box, x, y):
    left, top, right, bottom = box
    return left + x * (right - left), bottom - y * (bottom - top)


def draw_center(draw, box, text, fnt, fill):
    lines = text.split("\n")
    heights = [draw.textbbox((0, 0), line, font=fnt)[3] for line in lines]
    y0 = box[1] + (box[3] - box[1] - sum(heights) - 6 * (len(lines) - 1)) / 2
    for line, h in zip(lines, heights):
        bbox = draw.textbbox((0, 0), line, font=fnt)
        draw.text((box[0] + (box[2] - box[0] - (bbox[2] - bbox[0])) / 2, y0), line, font=fnt, fill=fill)
        y0 += h + 6


def draw_figure(confusion, pred, target):
    img = Image.new("RGB", (2100, 930), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    dark = (31, 41, 55)
    gray = (105, 113, 124)
    draw.text((90, 55), "Classification and Regression Outputs", font=font(40, True), fill=dark)
    draw.text((90, 105), "Binary labels are obtained by thresholding continuous PERCLOS predictions at 0.35.", font=font(24), fill=gray)

    # Confusion matrix.
    draw.text((155, 185), "Classification confusion matrix", font=font(28, True), fill=dark)
    matrix_box = (190, 270, 690, 770)
    row_pct = confusion / confusion.sum(axis=1, keepdims=True) * 100
    labels = ["Alert", "Fatigue"]
    for i in range(2):
        draw.text((85, matrix_box[1] + i * 250 + 100), labels[i], font=font(24), fill=dark)
        draw.text((matrix_box[0] + i * 250 + 85, 790), labels[i], font=font(24), fill=dark)
        for j in range(2):
            val = row_pct[i, j] / 100
            color = (232 - int(120 * val), 242 - int(80 * val), 252)
            cell = (matrix_box[0] + j * 250, matrix_box[1] + i * 250, matrix_box[0] + (j + 1) * 250, matrix_box[1] + (i + 1) * 250)
            draw.rounded_rectangle(cell, radius=10, fill=color, outline=(255, 255, 255), width=5)
            txt_color = "white" if val > 0.55 else dark
            draw_center(draw, cell, f"{confusion[i, j]}\n{row_pct[i, j]:.1f}%", font(30, True), txt_color)
    draw.text((320, 850), "Predicted class", font=font(24), fill=dark)
    draw.text((75, 235), "True", font=font(22), fill=dark)

    # Regression scatter.
    draw.text((1130, 185), "PERCLOS regression fit", font=font(28, True), fill=dark)
    draw.text((1130, 225), "gray: ideal    coral: fitted", font=font(21), fill=gray)
    plot_box = (1100, 270, 1850, 770)
    draw.rectangle(plot_box, outline=(210, 215, 221), width=2)
    for tick in np.linspace(0, 1, 6):
        x, y0 = xy(plot_box, float(tick), 0)
        _, y1 = xy(plot_box, 0, float(tick))
        draw.line((x, plot_box[3], x, plot_box[3] + 10), fill=gray, width=2)
        draw.line((plot_box[0] - 10, y1, plot_box[0], y1), fill=gray, width=2)
        draw.text((x - 12, plot_box[3] + 18), f"{tick:.1f}", font=font(18), fill=gray)
        draw.text((plot_box[0] - 55, y1 - 10), f"{tick:.1f}", font=font(18), fill=gray)
    rng = np.random.default_rng(7)
    idx = rng.choice(len(target), min(3500, len(target)), replace=False)
    for t, p in zip(target[idx], pred[idx]):
        x, y = xy(plot_box, float(t), float(p))
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(42, 157, 143, 45))
    x0, y0 = xy(plot_box, 0, 0)
    x1, y1 = xy(plot_box, 1, 1)
    draw.line((x0, y0, x1, y1), fill=(140, 140, 140), width=3)
    coef = np.polyfit(target, pred, 1)
    a0, b0 = xy(plot_box, 0, float(coef[1]))
    a1, b1 = xy(plot_box, 1, float(coef[0] + coef[1]))
    draw.line((a0, b0, a1, b1), fill=(231, 111, 81), width=5)
    draw.text((1340, 850), "True PERCLOS", font=font(24), fill=dark)
    draw.text((965, 250), "Predicted", font=font(22), fill=dark)
    img.save(ROOT / "eogq_classification_regression.png", quality=95)
    img.save(ROOT / "eogq_classification_regression.pdf", "PDF", resolution=300.0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confusions, preds, targets = [], [], []
    for fold in range(5):
        confusion, pred, target = load_fold(fold, device)
        confusions.append(confusion)
        preds.append(pred)
        targets.append(target)
    draw_figure(np.sum(confusions, axis=0), np.concatenate(preds), np.concatenate(targets))


if __name__ == "__main__":
    main()
