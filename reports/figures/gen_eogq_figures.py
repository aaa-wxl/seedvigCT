from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SRC = Path(r"C:\Users\ASUS\.codex\generated_images\019f5981-ca9f-7fb1-966e-b532f80d71fb\call_XG2h7b3xtBUwtCZG2uleDsww.png")
FONT_DIR = Path(r"C:\Windows\Fonts")


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ("arialbd.ttf" if bold else "arial.ttf")), size)


def text_center(draw, box, text, size=26, bold=False, fill=(30, 39, 55)):
    f = font(size, bold)
    lines = text.split("\n")
    bounds = [draw.textbbox((0, 0), line, font=f) for line in lines]
    heights = [b[3] - b[1] for b in bounds]
    y = box[1] + (box[3] - box[1] - sum(heights) - 8 * (len(lines) - 1)) / 2
    for line, h, b in zip(lines, heights, bounds):
        w = b[2] - b[0]
        draw.text((box[0] + (box[2] - box[0] - w) / 2 - b[0], y - b[1]), line, font=f, fill=fill)
        y += h + 8


def architecture():
    img = Image.open(SRC).convert("RGB")
    draw = ImageDraw.Draw(img)
    # Left branch: EEG evidence source.
    text_center(draw, (175, 856, 393, 918), "Raw EEG", 21, True)
    text_center(draw, (176, 751, 393, 806), "EEG CNN\nPatch Encoder", 16, True)
    text_center(draw, (175, 617, 392, 696), "Window Transformer\nEncoder x1", 15, True)
    text_center(draw, (176, 493, 391, 542), "EEG Tokens\nfor K,V", 18, True)

    # Right branch: EOG query and anchor pathway.
    text_center(draw, (1126, 869, 1360, 924), "Raw EOG", 21, True)
    text_center(draw, (1127, 760, 1360, 815), "EOG CNN\nPatch Encoder", 16, True)
    text_center(draw, (1127, 653, 1360, 707), "EOG Tokens", 19, True)
    text_center(draw, (1127, 551, 1360, 607), "MeanPool", 20, True)
    text_center(draw, (1127, 444, 1360, 499), "EOG Window\nEmbedding", 17, True)

    # Fusion block: EOG as Query, EEG as Key/Value.
    text_center(draw, (653, 790, 745, 842), "K,V", 21, True)
    text_center(draw, (879, 790, 972, 842), "Q", 21, True)
    text_center(draw, (653, 681, 972, 740), "Multi-Head\nCross-Attention", 19, True)
    text_center(draw, (653, 599, 972, 657), "Cross-Modal Gate", 20, True)
    text_center(draw, (653, 526, 972, 583), "Add & Norm\nEOG + g*EEG", 18, True)
    text_center(draw, (668, 451, 956, 506), "Fused Sequence", 20, True)
    draw.text((567, 432), "Positional\nEncoding", font=font(17), fill=(25, 25, 25))

    # Prediction head.
    text_center(draw, (747, 241, 1042, 312), "Temporal Transformer\nEncoder x1", 20, True)
    text_center(draw, (806, 147, 986, 196), "Linear", 19, True)
    text_center(draw, (806, 49, 987, 106), "Sigmoid", 19, True)
    text_center(draw, (521, 50, 699, 106), "Threshold", 19, True)
    draw.text((805, 10), "PERCLOS\nOutput", font=font(17), fill=(25, 25, 25))
    draw.text((534, 10), "Binary\nMetrics", font=font(17), fill=(25, 25, 25))
    img.save(ROOT / "eogq_model_architecture_image2_v2.png", quality=95)
    img.save(ROOT / "eogq_model_architecture_image2_v2.pdf", "PDF", resolution=300.0)


def barh(draw, x, y, w, h, values, labels, title, lo, hi, colors):
    draw.text((x, y), title, font=font(24, True), fill=(25, 35, 45))
    y0 = y + 45
    for i, (v, label) in enumerate(zip(values, labels)):
        yy = y0 + i * 43
        draw.text((x, yy + 4), label, font=font(15), fill=(50, 55, 65))
        bw = int((v - lo) / (hi - lo) * w)
        draw.rounded_rectangle((x + 145, yy, x + 145 + bw, yy + h), radius=6, fill=colors[i])
        if "Ours" in label or (title.endswith("Pearson") and label == "EOG-only"):
            draw.text((x + 153 + bw, yy + 2), f"{v:.3f}", font=font(15, True), fill=(120, 55, 35))


def results():
    labels = ["EEG-only", "EOG-only", "EEG-Q", "EEG-Q+gate", "EOG-Q no gate", "EOG-Q (Ours)", "Bi-dir+gate"]
    colors = [(184, 192, 204)] * len(labels)
    colors[4] = (122, 157, 192)
    colors[5] = (231, 111, 81)
    img = Image.new("RGB", (2100, 1020), "white")
    draw = ImageDraw.Draw(img)
    draw.text((70, 35), "Final Results on SEED-VIG Cross-subject Validation", font=font(34, True), fill=(25, 35, 45))
    draw.text((70, 82), "EOG-Q/EEG-KV with gate is highlighted. Values are fold averages.", font=font(22), fill=(85, 90, 100))

    metrics = [("Acc", [0.6378, 0.8250, 0.7976, 0.8152, 0.8178, 0.8367, 0.8180]),
               ("Macro-F1", [0.5711, 0.7883, 0.7768, 0.7903, 0.7942, 0.8165, 0.7964]),
               ("BalAcc", [0.6002, 0.7820, 0.7828, 0.7902, 0.7915, 0.8104, 0.7947])]
    x0, y0 = 90, 170
    for gi, (name, vals) in enumerate(metrics):
        x = x0 + gi * 650
        draw.text((x, y0), name + " higher", font=font(26, True), fill=(25, 35, 45))
        for i, (v, label) in enumerate(zip(vals, labels)):
            yy = y0 + 55 + i * 46
            draw.text((x, yy + 5), label, font=font(18), fill=(50, 55, 65))
            bw = int((v - 0.54) / (0.86 - 0.54) * 350)
            draw.rounded_rectangle((x + 150, yy, x + 150 + bw, yy + 28), radius=7, fill=colors[i])
            if i == 5:
                draw.text((x + 158 + bw, yy + 2), f"{v:.3f}", font=font(17, True), fill=(120, 55, 35))

    barh(draw, 120, 620, 500, 24, [0.2345, 0.1432, 0.1615, 0.1460, 0.1503, 0.1423, 0.1519], labels, "RMSE lower", 0.13, 0.24, colors)
    barh(draw, 1110, 620, 500, 24, [0.5428, 0.8536, 0.8213, 0.8472, 0.8249, 0.8454, 0.8324], labels, "Pearson higher", 0.52, 0.87, colors)
    img.save(ROOT / "eogq_final_results.png", quality=95)
    img.save(ROOT / "eogq_final_results.pdf", "PDF", resolution=300.0)


if __name__ == "__main__":
    architecture()
    results()
