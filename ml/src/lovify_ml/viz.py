"""Plot helpers, so every notebook chart looks like it belongs to the same project."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import schema as S

INK = "#1d1d1f"
MUTED = "#8a8a8e"
ACCENT = "#1db954"      # the obvious one
WARN = "#e8604c"
GRID = "#e6e6ea"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "grid.color": GRID, "axes.grid": True, "grid.linewidth": .8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "font.size": 10, "legend.frameon": False,
})


def bar_results(df: pd.DataFrame, value: str, label: str, title: str,
                reference: float | None = None, reference_label: str = "chance",
                ax=None, highlight: list[str] | None = None):
    """Horizontal bars with a reference line -- the chance-rate line is the
    whole point of most of these charts."""
    ax = ax or plt.subplots(figsize=(7, 0.42 * len(df) + 1.2))[1]
    d = df.sort_values(value)
    colors = [WARN if (highlight and l in highlight) else ACCENT for l in d[label]]
    ax.barh(d[label], d[value], color=colors, height=.68)
    if reference is not None:
        ax.axvline(reference, color=INK, ls="--", lw=1.2)
        ax.text(reference, -0.9, f"  {reference_label} = {reference:.3f}",
                color=INK, fontsize=9, va="top")
    for y, v in zip(range(len(d)), d[value]):
        ax.text(v, y, f"  {v:.3f}", va="center", fontsize=9, color=MUTED)
    ax.set_title(title)
    ax.set_xlabel(value)
    ax.margins(x=.18)
    return ax


def taste_mix_pie(mix: pd.Series, title: str = "Your taste mix", ax=None):
    ax = ax or plt.subplots(figsize=(5.6, 5.6))[1]
    cmap = plt.get_cmap("tab20")
    ax.pie(mix.to_numpy(), labels=list(mix.index), autopct="%1.1f%%",
           colors=[cmap(i % 20) for i in range(len(mix))],
           textprops={"fontsize": 9}, startangle=90)
    ax.set_title(title)
    ax.grid(False)
    return ax


def lift_chart(lift: pd.Series, title: str = "Lift over the average listener", ax=None):
    """The honest version of the pie chart: how far from 1.0 each sign sits."""
    ax = ax or plt.subplots(figsize=(7, 4.4))[1]
    d = lift.sort_values()
    colors = [ACCENT if v >= 1 else MUTED for v in d]
    ax.barh(list(d.index), d.to_numpy() - 1, left=1, color=colors, height=.68)
    ax.axvline(1.0, color=INK, lw=1.3)
    ax.set_title(title)
    ax.set_xlabel("x the average listener  (1.0 = exactly average)")
    for y, v in enumerate(d):
        ax.text(v, y, f"  {v:.2f}x", va="center", fontsize=9, color=MUTED)
    ax.margins(x=.16)
    return ax


def feature_distributions(frames: dict[str, pd.DataFrame], features: list[str] | None = None,
                          ncols: int = 4):
    """Overlaid histograms -- the picture that shows why a mean vector is a lie."""
    features = features or S.AUDIO_FEATURES[:8]
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    colors = [ACCENT, WARN, "#5b7cfa", "#f2b705"]
    for ax, f in zip(axes, features):
        for i, (name, df) in enumerate(frames.items()):
            ax.hist(df[f].to_numpy(float), bins=28, alpha=.55,
                    color=colors[i % len(colors)], label=name, density=True)
        ax.set_title(f, fontsize=10)
        ax.set_yticks([])
    for ax in axes[len(features):]:
        ax.set_visible(False)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def null_distribution(null: np.ndarray, score: float, percentile: float,
                      title: str = "Where this pair falls among random pairs", ax=None):
    ax = ax or plt.subplots(figsize=(7, 3.6))[1]
    ax.hist(null, bins=60, color=MUTED, alpha=.6)
    ax.axvline(score, color=WARN, lw=2.2)
    ax.text(score, ax.get_ylim()[1] * .92, f"  this pair: {percentile:.0f}th pct",
            color=WARN, fontsize=10, va="top")
    ax.set_title(title)
    ax.set_xlabel("blended compatibility score")
    ax.set_ylabel("random pairs")
    return ax
