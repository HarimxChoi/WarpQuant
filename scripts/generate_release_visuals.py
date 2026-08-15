from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "release"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#172235"
MUTED = "#637083"
BLUE = "#2F64C8"
BLUE_SOFT = "#E8EFFB"
WARM = "#F6F4EF"
WHITE = "#FFFFFF"
GRID = "#DCE2E8"

FORMATS = ["BF16", "Q4_K_M", "IQ3_S", "WarpQuant"]
BPW = [16.0, 4.92, 3.6940, 3.6165]
PPL = [6.9548, 6.9656, 7.1820, 7.4737]
ARC = [52.17, 50.84, 52.17, 56.86]
MMLU = [43.07, 42.90, 42.97, 42.72]
COMMONSENSE = [79.23, 79.23, 78.83, 78.83]


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUT / f"{stem}.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": 600,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
        }
    )


def pareto() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.8), facecolor=WARM)
    fig.subplots_adjust(top=0.78, bottom=0.18, wspace=0.28)
    fig.suptitle("Qwen3.8-27B · quality–memory trade-off", x=0.07, y=0.94, ha="left", fontsize=23)
    fig.text(0.07, 0.875, "Text-backbone payload. Lower bpw is smaller; arrows mark the preferred direction.", color=MUTED, fontsize=11)

    for ax in axes:
        ax.set_facecolor(WHITE)
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

    colors = ["#9AA4B2", "#6C7B8F", "#7A8EAE", BLUE]
    sizes = [90, 105, 110, 190]

    for name, x, y, color, size in zip(FORMATS, BPW, ARC, colors, sizes):
        axes[0].scatter(x, y, s=size, color=color, edgecolor=WHITE, linewidth=1.4, zorder=3)
        axes[0].annotate(name, (x, y), xytext=(7, 7), textcoords="offset points", fontsize=9, weight=600 if name == "WarpQuant" else 400)
    axes[0].set_title("ARC-Challenge accuracy", loc="left", fontsize=14, pad=14)
    axes[0].set_xlabel("Text bpw  ← smaller")
    axes[0].set_ylabel("Accuracy (%)  ↑ better")
    axes[0].set_xlim(3.1, 16.8)
    axes[0].set_ylim(49.5, 57.7)

    frontier = sorted(zip(BPW, PPL))
    axes[1].plot([p[0] for p in frontier], [p[1] for p in frontier], color=BLUE, alpha=0.4, linewidth=2.2)
    for name, x, y, color, size in zip(FORMATS, BPW, PPL, colors, sizes):
        axes[1].scatter(x, y, s=size, color=color, edgecolor=WHITE, linewidth=1.4, zorder=3)
        offset = (7, -15) if name == "WarpQuant" else (7, 7)
        axes[1].annotate(name, (x, y), xytext=offset, textcoords="offset points", fontsize=9, weight=600 if name == "WarpQuant" else 400)
    axes[1].set_title("WikiText-2 PPL Pareto frontier", loc="left", fontsize=14, pad=14)
    axes[1].set_xlabel("Text bpw  ← smaller")
    axes[1].set_ylabel("Perplexity  ↓ better")
    axes[1].set_xlim(3.1, 16.8)
    axes[1].set_ylim(6.82, 7.58)

    fig.text(0.07, 0.055, "The PPL panel traces the non-dominated memory–quality frontier; ARC shows task accuracy at the same payloads.", fontsize=10.5, color=MUTED)
    save(fig, "qwen38-pareto")


def benchmark_card() -> None:
    fig = plt.figure(figsize=(14.2, 8.0), facecolor=WARM)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.06, 0.91, "WarpQuant · Qwen3.8-27B", fontsize=25, weight=600)
    ax.text(0.06, 0.858, "INT3 LLM quantization · text-backbone benchmark", fontsize=12, color=MUTED)

    columns = ["Format", "Text bpw", "Payload", "WT2 PPL ↓", "ARC ↑", "MMLU ↑", "Commonsense ↑"]
    rows = [
        ["BF16", "16.00", "50.11 GiB", "6.9548", "52.17", "43.07", "79.23"],
        ["Q4_K_M", "4.92", "15.41 GiB", "6.9656", "50.84", "42.90", "79.23"],
        ["IQ3_S", "3.6940", "11.57 GiB", "7.1820", "52.17", "42.97", "78.83"],
        ["WarpQuant R16E4H4", "3.6165", "11.32 GiB", "7.4737", "56.86", "42.72", "78.83"],
    ]
    x = [0.06, 0.33, 0.45, 0.58, 0.70, 0.80, 0.90]
    top = 0.75
    row_h = 0.105
    ax.add_patch(FancyBboxPatch((0.045, top - 0.03), 0.91, 0.085, boxstyle="round,pad=0.006,rounding_size=0.008", facecolor=INK, edgecolor="none"))
    for xpos, label in zip(x, columns):
        ax.text(xpos, top + 0.012, label, color=WHITE, fontsize=9.5, weight=600, ha="left" if label == "Format" else "center")

    for row_i, row in enumerate(rows):
        y = top - (row_i + 1) * row_h
        face = BLUE_SOFT if row_i == 3 else WHITE
        edge = BLUE if row_i == 3 else GRID
        ax.add_patch(FancyBboxPatch((0.045, y - 0.025), 0.91, 0.085, boxstyle="round,pad=0.006,rounding_size=0.008", facecolor=face, edgecolor=edge, linewidth=1.4 if row_i == 3 else 0.8))
        for col_i, (xpos, value) in enumerate(zip(x, row)):
            ax.text(xpos, y + 0.016, value, fontsize=10.5, weight=600 if row_i == 3 else 400, color=BLUE if row_i == 3 else INK, ha="left" if col_i == 0 else "center")

    ax.text(0.06, 0.19, "3.6165 bpw", fontsize=25, color=BLUE, weight=600)
    ax.text(0.06, 0.145, "lowest text payload", fontsize=10.5, color=MUTED)
    ax.text(0.34, 0.19, "11.32 GiB", fontsize=25, color=BLUE, weight=600)
    ax.text(0.34, 0.145, "text-backbone payload", fontsize=10.5, color=MUTED)
    ax.text(0.62, 0.19, "56.86%", fontsize=25, color=BLUE, weight=600)
    ax.text(0.62, 0.145, "ARC-Challenge · 299", fontsize=10.5, color=MUTED)
    ax.text(0.06, 0.065, "Commonsense = macro average of fixed 1,000-example HellaSwag, WinoGrande, and PIQA screens.", fontsize=9.5, color=MUTED)
    save(fig, "qwen38-benchmark-card")


def social_cover() -> None:
    fig = plt.figure(figsize=(16, 9), facecolor=INK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.065, 0.89, "W A R P Q U A N T", color="#8BB0FF", fontsize=13, weight=600)
    ax.text(0.065, 0.78, "Dual-Domain INT3\nLLM Quantization", color=WHITE, fontsize=37, weight=600, va="top", linespacing=1.05)
    ax.text(0.065, 0.54, "Hadamard rotation  ×  Output-Fisher sensitivity", color="#B9C5D6", fontsize=15)

    stages = [("01", "Rotate", "Hadamard domain"), ("02", "Quantize", "Block-GPTQ · INT3"), ("03", "Recover", "Output-Fisher columns")]
    for i, (num, title, sub) in enumerate(stages):
        x = 0.065 + i * 0.205
        ax.add_patch(FancyBboxPatch((x, 0.29), 0.17, 0.15, boxstyle="round,pad=0.012,rounding_size=0.014", facecolor="#223049", edgecolor="#3D4E6A"))
        ax.text(x + 0.018, 0.397, num, color="#8BB0FF", fontsize=9, weight=600)
        ax.text(x + 0.018, 0.353, title, color=WHITE, fontsize=14, weight=600)
        ax.text(x + 0.018, 0.316, sub, color="#AEBACB", fontsize=9)
        if i < 2:
            ax.text(x + 0.185, 0.36, "→", color="#71819A", fontsize=18, ha="center")

    ax.add_patch(FancyBboxPatch((0.70, 0.17), 0.245, 0.68, boxstyle="round,pad=0.018,rounding_size=0.022", facecolor=WARM, edgecolor="none"))
    ax.text(0.73, 0.79, "Qwen3.8-27B", color=MUTED, fontsize=11, weight=600)
    metrics = [("3.6165", "text bpw"), ("11.32 GiB", "payload"), ("56.86%", "ARC-Challenge")]
    for i, (value, label) in enumerate(metrics):
        y = 0.65 - i * 0.18
        ax.text(0.73, y, value, color=BLUE, fontsize=28, weight=600)
        ax.text(0.73, y - 0.045, label, color=MUTED, fontsize=10)
        if i < 2:
            ax.plot([0.73, 0.91], [y - 0.085, y - 0.085], color=GRID, linewidth=1)
    ax.text(0.065, 0.105, "harimxchoi.github.io/projects/warpquant", color="#8B99AD", fontsize=10)
    save(fig, "warpquant-social-cover")


if __name__ == "__main__":
    setup()
    pareto()
    benchmark_card()
    social_cover()
