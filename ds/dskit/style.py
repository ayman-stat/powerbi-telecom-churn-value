"""House chart style: one palette, insight-led titles, consistent export."""
from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = {
    "ink": "#1B2A41", "navy": "#1F3A5F", "teal": "#0A7C8C", "amber": "#E8A33D", "coral": "#E4572E",
    "sage": "#5B8C5A", "purple": "#6C5B7B", "grey": "#8A94A6", "light": "#D9DEE7", "paper": "#FFFFFF",
}
CYCLE = [PALETTE[k] for k in ("navy", "teal", "amber", "coral", "sage", "purple", "grey")]
POS, NEG, NEUTRAL = PALETTE["teal"], PALETTE["coral"], PALETTE["grey"]


def set_style() -> None:
    """Apply the portfolio style to every matplotlib figure in the session."""
    fonts = ["Segoe UI", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]
    mpl.rcParams.update({
        "figure.figsize": (10, 5), "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "figure.facecolor": PALETTE["paper"], "axes.facecolor": PALETTE["paper"],
        "font.family": "sans-serif", "font.sans-serif": fonts, "font.size": 10.5,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
        "axes.labelsize": 10.5, "axes.labelcolor": PALETTE["ink"], "axes.edgecolor": PALETTE["light"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#EEF1F5", "grid.linewidth": 0.8, "xtick.color": "#4A5568", "ytick.color": "#4A5568",
        "axes.prop_cycle": mpl.cycler(color=CYCLE), "legend.frameon": False, "legend.fontsize": 9.5,
        "lines.linewidth": 2.0, "patch.edgecolor": "none",
    })


def headline(ax, title: str, subtitle: str | None = None) -> None:
    """Insight-led title (the finding, not the chart type) with an optional grey subtitle for the method."""
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=PALETTE["ink"], pad=24 if subtitle else 12)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.5, color="#5A6478", va="bottom")


def savefig(fig, name: str, folder: str) -> str:
    """Save a figure as PNG for the README and model card; returns the path."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, facecolor=fig.get_facecolor())
    return path


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def fmt_money(x: float, symbol: str = "$", digits: int = 0) -> str:
    a = abs(x)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{'-' if x < 0 else ''}{symbol}{a / div:.1f}{suf}"
    return f"{'-' if x < 0 else ''}{symbol}{a:,.{digits}f}"


def close(fig=None) -> None:
    plt.close(fig) if fig is not None else plt.close("all")
