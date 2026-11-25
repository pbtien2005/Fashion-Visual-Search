"""Matplotlib charts for the embedding benchmark; each function saves one PNG into reports/.

Colour follows the model (fixed slot order, never re-assigned when a model is missing),
one y-scale per chart, recessive grid, direct value labels on bar ends.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import MODELS
from .run_models import STORED
from .scoring import CONFIG_LABELS, CONFIG_ORDER, model_label

# Validated categorical palette, light mode (fixed order). The harness check is neutral gray.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MODEL_COLORS = {key: SLOTS[i] for i, key in enumerate(MODELS)} | {STORED: "#8a8984"}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = {"win": "#2a78d6", "loss": "#e34948", "tie": "#f0efec"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160,
})


def _save(fig, out_dir, name):
    path = Path(out_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


def _models(frame):
    order = [STORED, *MODELS]
    return [key for key in order if key in set(frame["model"])]


def _grouped_hbars(ax, frame, metric, configs, fmt="{:.2f}"):
    models = _models(frame)
    height = 0.8 / len(models)
    for i, model in enumerate(models):
        rows = frame[frame["model"] == model].set_index("config")
        for j, config in enumerate(configs):
            if config not in rows.index:
                continue
            value = rows.loc[config, metric]
            y = j + (i - (len(models) - 1) / 2) * height
            ax.barh(y, value, height=height * 0.85, color=MODEL_COLORS[model],
                    label=model_label(model) if j == 0 or config == configs[-1] else None)
            ax.text(value + 0.005, y, fmt.format(value), va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(range(len(configs)), [CONFIG_LABELS[c] for c in configs])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)


def _legend(fig, frame, **kwargs):
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m in _models(frame)]
    fig.legend(handles, [model_label(m) for m in _models(frame)], **kwargs)


def hit_mrr(summary, split, out_dir, configs=None):
    frame = summary[summary["split"] == split]
    configs = [c for c in CONFIG_ORDER if c in set(frame["config"]) and (configs is None or c in configs)]
    frame = frame[frame["config"].isin(configs)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.9 + 1.25 * len(configs) * len(_models(frame)) / 4),
                             sharey=True)
    n = int(frame["has_match"].max())
    for ax, metric, title in ((axes[0], "hit_at_5", f"HitRate@5 ({split}, {n} ảnh has_match)"),
                              (axes[1], "mrr_at_5", f"MRR@5 ({split})")):
        _grouped_hbars(ax, frame, metric, configs)
        ax.set_title(title)
        ax.set_xlim(0, min(1.0, frame[metric].max() * 1.25 + 0.05))
    _legend(fig, frame, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Chất lượng truy xuất theo model và cấu hình (nhãn nghiêm ngặt)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, f"01_hitrate_mrr_{split}.png")


def strict_vs_judged(summary, split, out_dir, configs=("C2_embedding_only", "B0_current")):
    frame = summary[(summary["split"] == split) & summary["config"].isin(configs)]
    fig, axes = plt.subplots(1, len(configs), figsize=(6.2 * len(configs), 0.8 + 0.42 * len(_models(frame))),
                             sharex=True)
    for ax, config in zip(np.atleast_1d(axes), configs):
        rows = frame[frame["config"] == config].set_index("model")
        models = [m for m in _models(frame) if m in rows.index]
        for y, model in enumerate(models):
            strict, judged = rows.loc[model, "hit_at_5"], rows.loc[model, "judged_hit_at_5"]
            color = MODEL_COLORS[model]
            ax.plot([strict, judged], [y, y], color=GRID, linewidth=2, zorder=1)
            ax.scatter([strict], [y], s=60, color=SURFACE, edgecolor=color, linewidth=2, zorder=2)
            ax.scatter([judged], [y], s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.text(max(strict, judged) + 0.02, y, f"unjudged@5 = {int(rows.loc[model, 'unjudged_at_5'])}",
                    va="center", fontsize=8, color=MUTED)
        ax.set_yticks(range(len(models)), [model_label(m) for m in models])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(CONFIG_LABELS[config])
        ax.set_xlabel("HitRate@5")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK_2,
                          markersize=8, markeredgewidth=2),
               plt.Line2D([], [], marker="o", linestyle="", color=INK_2, markersize=8)]
    fig.legend(handles, ["Nghiêm ngặt (sản phẩm chưa chấm = trượt)", "Chỉ tính sản phẩm đã chấm"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Ảnh hưởng của sản phẩm chưa chấm nhãn ({split})", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, f"02_strict_vs_judged_{split}.png")


def validation_vs_test(summary, out_dir, config="C2_embedding_only", metric="hit_at_5"):
    frame = summary[summary["config"] == config].pivot_table(index="model", columns="split", values=metric)
    if not {"validation", "test"} <= set(frame.columns):
        return None
    models = [m for m in [STORED, *MODELS] if m in frame.index]
    fig, ax = plt.subplots(figsize=(8, 0.8 + 0.42 * len(models)))
    for y, model in enumerate(models):
        val, test = frame.loc[model, "validation"], frame.loc[model, "test"]
        ax.plot([val, test], [y, y], color=GRID, linewidth=2, zorder=1)
        ax.scatter([val], [y], s=60, color=SURFACE, edgecolor=MODEL_COLORS[model], linewidth=2, zorder=2)
        ax.scatter([test], [y], s=60, color=MODEL_COLORS[model], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(max(val, test) + 0.02, y, f"val {val:.2f} · test {test:.2f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(models)), [model_label(m) for m in models])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric.replace("_at_", "@").replace("hit", "HitRate").replace("mrr", "MRR"))
    ax.set_title(f"Validation (vòng rỗng) và test (chấm đặc): {CONFIG_LABELS[config]}")
    fig.tight_layout()
    return _save(fig, out_dir, f"03_validation_vs_test_{config}.png")


def paired(paired_frame, split, out_dir, configs=("C2_embedding_only", "B0_current")):
    frame = paired_frame[(paired_frame["split"] == split) & paired_frame["config"].isin(configs)]
    if frame.empty:
        return None
    fig, axes = plt.subplots(1, len(configs), figsize=(7.4 * len(configs), 0.9 + 0.42 * frame["model"].nunique()),
                             sharex=True)
    limit = int(frame[["win", "loss"]].to_numpy().max()) + 3
    for ax, config in zip(np.atleast_1d(axes), configs):
        rows = frame[frame["config"] == config].set_index("model")
        models = [m for m in [STORED, *MODELS] if m in rows.index]
        for y, model in enumerate(models):
            win, loss = rows.loc[model, "win"], rows.loc[model, "loss"]
            ax.barh(y, win, color=DIVERGING["win"], height=0.6)
            ax.barh(y, -loss, color=DIVERGING["loss"], height=0.6)
            ax.text(win + 0.3, y, f"+{win}", va="center", fontsize=8, color=INK_2)
            ax.text(-loss - 0.3, y, f"−{loss}", va="center", ha="right", fontsize=8, color=INK_2)
            # Own column right of the plot so it never collides with the "+n" bar labels.
            ax.text(1.02, y, f"hòa {rows.loc[model, 'tie']} · p={rows.loc[model, 'sign_test_p']:.2f}",
                    transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8, color=MUTED,
                    clip_on=False)
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.set_yticks(range(len(models)), [model_label(m) for m in models])
        ax.invert_yaxis()
        ax.set_xlim(-limit, limit)
        ax.set_title(CONFIG_LABELS[config])
        ax.set_xlabel("Số truy vấn RR@5 tốt hơn (+) / kém hơn (−) FashionCLIP")
    fig.suptitle(f"So từng truy vấn với FashionCLIP nhúng lại ({split}, kiểm định dấu)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, f"04_paired_vs_fashionclip_{split}.png")


def config_gain(summary, split, out_dir, pairs=(("B0_current", "F2_B0_whole"), ("C2_embedding_only", "F2_C2_whole")),
                metric="hit_at_5"):
    """Dumbbell per model: base configuration (hollow) -> v1 fallback configuration (filled)."""
    frame = summary[summary["split"] == split]
    fig, axes = plt.subplots(1, len(pairs), figsize=(6.6 * len(pairs), 0.9 + 0.42 * len(_models(frame))),
                             sharex=True)
    n = int(frame["has_match"].max())
    for ax, (base, new) in zip(np.atleast_1d(axes), pairs):
        table = frame[frame["config"].isin((base, new))].pivot_table(index="model", columns="config", values=metric)
        models = [m for m in _models(frame) if m in table.index and table.loc[m].notna().all()]
        for y, model in enumerate(models):
            before, after = table.loc[model, base], table.loc[model, new]
            color = MODEL_COLORS[model]
            ax.plot([before, after], [y, y], color=GRID, linewidth=2, zorder=1)
            ax.scatter([before], [y], s=60, color=SURFACE, edgecolor=color, linewidth=2, zorder=2)
            ax.scatter([after], [y], s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.text(max(before, after) + 0.02, y, f"{before * n:.0f} → {after * n:.0f} / {n}",
                    va="center", fontsize=8, color=INK_2)
        ax.set_yticks(range(len(models)), [model_label(m) for m in models])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("HitRate@5" if metric == "hit_at_5" else metric)
        ax.set_title(f"{CONFIG_LABELS[base]} → {CONFIG_LABELS[new]}", fontsize=10.5)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK_2,
                          markersize=8, markeredgewidth=2),
               plt.Line2D([], [], marker="o", linestyle="", color=INK_2, markersize=8)]
    fig.legend(handles, ["Cấu hình gốc (v0)", "Thêm nhóm ảnh nguyên (v1)"], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Mức tăng khi thêm nhóm kết quả theo ảnh nguyên ({split})", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, f"08_fallback_gain_{split}.png")


def class_heatmap(table, split, config, out_dir):
    if table.empty:
        return None
    models = [m for m in [STORED, *MODELS] if m in table.columns]
    table = table[models]
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(1.3 * len(models) + 2.5, 0.45 * len(table) + 1.6))
    image = ax.imshow(table.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for (i, j), value in np.ndenumerate(table.to_numpy(dtype=float)):
        if not np.isnan(value):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color=SURFACE if value > 0.55 else INK)
    ax.set_xticks(range(len(models)), [model_label(m) for m in models], rotation=30, ha="right")
    ax.set_yticks(range(len(table)), table.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, fraction=0.03, label="HitRate@5")
    ax.set_title(f"HitRate@5 theo loại trang phục ({split}, {CONFIG_LABELS[config]})")
    fig.tight_layout()
    return _save(fig, out_dir, f"05_class_heatmap_{split}_{config}.png")


def latency(summary, split, out_dir, config="C2_embedding_only"):
    frame = summary[(summary["split"] == split) & (summary["config"] == config)].set_index("model")
    models = [m for m in MODELS if m in frame.index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.9 + 0.42 * len(models)), sharey=True)
    for ax, column, title, fmt in (
        (axes[0], "embed_ms_mean", "Nhúng + màu mỗi truy vấn, mean sau ảnh đầu (ms)", "{:.0f} ms"),
        (axes[1], "catalog_embed_seconds", "Nhúng toàn bộ catalog (giây)", "{:.1f} s"),
    ):
        values = frame.loc[models, column].astype(float)
        ax.barh(range(len(models)), values, color=[MODEL_COLORS[m] for m in models], height=0.6)
        for y, value in enumerate(values):
            ax.text(value * 1.01, y, fmt.format(value), va="center", fontsize=8, color=INK_2)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlim(0, values.max() * 1.25)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(models)), [model_label(m) for m in models])
    axes[0].invert_yaxis()
    fig.suptitle(f"Chi phí tính toán ({split}, {CONFIG_LABELS[config]}, GPU)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, f"06_latency_{split}.png")


# Pre-threshold, like every other chart: the scorer's "rejected" is a top-5 hit that the
# 0.70 / 0.60 thresholds (tuned for FashionCLIP) then reject, so it is counted as a hit here.
FAILURES = ["hit", "rank_miss", "candidate_miss", "wrong_class", "no_detection"]
FAILURE_LABELS = {"hit": "Trúng top-5 (trước ngưỡng)", "rank_miss": "Có ứng viên, xếp hạng trượt",
                  "candidate_miss": "Không có trong top-100", "wrong_class": "YOLO sai lớp",
                  "no_detection": "YOLO không phát hiện"}


def failures(failure_frame, split, config, out_dir):
    frame = failure_frame[(failure_frame["split"] == split) & (failure_frame["config"] == config)]
    if frame.empty:
        return None
    frame = frame.set_index("model").copy()
    frame["hit"] = frame.get("hit", 0) + frame.get("rejected", 0)
    models = [m for m in [STORED, *MODELS] if m in frame.index]
    columns = [c for c in FAILURES if c in frame.columns and frame[c].sum() > 0]
    fig, ax = plt.subplots(figsize=(10, 0.9 + 0.45 * len(models)))
    left = np.zeros(len(models))
    for column in columns:
        values = frame.loc[models, column].to_numpy(dtype=float)
        # Colour follows the failure category, not its position among the columns present.
        ax.barh(range(len(models)), values, left=left, color=SLOTS[FAILURES.index(column)], height=0.6,
                edgecolor=SURFACE, linewidth=2, label=FAILURE_LABELS[column])
        for y, (start, value) in enumerate(zip(left, values)):
            if value >= 2:
                ax.text(start + value / 2, y, f"{int(value)}", ha="center", va="center", fontsize=8,
                        color=SURFACE)
        left += values
    ax.set_yticks(range(len(models)), [model_label(m) for m in models])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Số truy vấn has_match")
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.32 - 0.02 * len(models)))
    ax.set_title(f"Nguyên nhân trượt ({split}, {CONFIG_LABELS[config]})")
    fig.tight_layout()
    return _save(fig, out_dir, f"07_failures_{split}_{config}.png")


def summary_table(summary, split):
    """Wide table for display and Markdown export."""
    frame = summary[summary["split"] == split].copy()
    frame["model_label"] = frame["model"].map(model_label)
    order = {key: i for i, key in enumerate([STORED, *MODELS])}
    frame = frame.sort_values(["config", "model"], key=lambda s: s.map(
        order if s.name == "model" else {c: i for i, c in enumerate(CONFIG_ORDER)}))
    columns = ["config", "model_label", "dimension", "hit_at_5", "mrr_at_5", "precision_at_5", "joint_at_5",
               "candidate_hit_at_100", "judged_hit_at_5", "judged_mrr_at_5", "unjudged_at_5", "target_recall",
               "embed_ms_mean"]
    return frame[columns].reset_index(drop=True)


def to_markdown(frame: pd.DataFrame) -> str:
    def cell(value):
        if isinstance(value, float):
            return "" if np.isnan(value) else f"{value:.3f}"
        return str(value)

    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)
