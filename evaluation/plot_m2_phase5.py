"""Plot saved phase5 metrics; no inference or statistics beyond recorded means."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.m2_annotations import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="docs/m2/phase5")
    args = parser.parse_args()
    root = Path(args.run)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    modes = ("off", "raw", "open_close")
    labels = ("H0-M1\n(off)", "B2\n(raw)", "B3\n(open-close)")
    colors = ("#72808e", "#267b91", "#df8c32")
    for column, split in enumerate(("validation", "test")):
        report = read_json(root / split / "results.json")
        for row, metric, title in ((0, "color_reference_similarity", "HSV147 color similarity"),
                                   (1, "mask_iou_end_to_end", "End-to-end mask IoU")):
            values = [report["summary"][m][metric + "_mean"] for m in modes]
            count = report["summary"]["off"][metric + "_n"]
            ax = axes[row, column]
            bars = ax.bar(labels, values, color=colors, width=.6)
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
            ax.set_ylim(0, 1.08)
            ax.set_title(f"{split.title()} | {title} | references n={count}", fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("M2 fixed-ROI ablation on shared reviewed images", fontsize=15)
    fig.supxlabel("Color: detection miss=0, mask fallback uses ROI. Mask IoU: miss/null mask=0. Shared regression, not product retrieval.", fontsize=9)
    fig.savefig(root / "ablation.png", dpi=170)
    plt.close(fig)
    print(root / "ablation.png")


if __name__ == "__main__":
    main()
