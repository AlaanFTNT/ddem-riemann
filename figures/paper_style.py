"""Shared matplotlib style for Paper 1 figures.

Imported by all build_figureN.py scripts so the eight figures share a single
visual language: serif font, deep-blue/orange/teal palette, consistent grid,
no top/right spines, single white background.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Color palette — used across all figures
COLORS = {
    "primary":    "#1f4e79",   # deep blue: model curves, DDEM
    "secondary":  "#d35400",   # orange-red: data, comparison curves
    "tertiary":   "#16a085",   # teal: derived quantities, secondary curves
    "highlight":  "#f39c12",   # gold: emphasis, best-fit markers
    "control":    "#7f7f7f",   # neutral grey: controls, null lines
    "tier1":      "#5b9bd5",   # light blue: Tier 1
    "tier2":      "#70ad47",   # green: Tier 2 (accessible)
    "tier3":      "#c00000",   # dark red: Tier 3 (out of reach)
    "accent":     "#8e44ad",   # purple: alternative emphasis
    "lightbg":    "#fafafa",   # axes background
    "grid":       "#cccccc",
}

# Standard figure sizes (inches)
SIZE_SINGLE_COL = (6.5, 4.0)   # full-width single panel
SIZE_DOUBLE_COL = (13.0, 4.5)  # double width or two-panel
SIZE_SQUARE     = (6.0, 6.0)   # corner plot or schematic
SIZE_TALL       = (6.5, 8.0)   # multi-row panel

def apply_style():
    """Apply the unified rcParams. Call at the top of each figure script."""
    mpl.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "font.size":        10,
        "axes.labelsize":   11,
        "axes.titlesize":   12,
        "axes.titleweight": "normal",
        "legend.fontsize":  9,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "axes.linewidth":   1.0,
        "lines.linewidth":  1.4,
        "figure.dpi":       130,
        "savefig.dpi":      220,
        "figure.facecolor": "white",
        "savefig.facecolor":"white",
        "axes.facecolor":   COLORS["lightbg"],
        "grid.color":       COLORS["grid"],
        "grid.linewidth":   0.5,
        "grid.alpha":       0.7,
        "axes.grid":        True,
        "axes.grid.which":  "both",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.1,
    })

def label_panel(ax, letter, x=-0.10, y=1.05):
    """Add (a), (b), ... panel label."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")

def add_figure_caption(fig, caption, y=-0.02):
    """Caption below figure (matplotlib supratitle alternative)."""
    fig.text(0.5, y, caption, ha="center", va="top",
             fontsize=9, style="italic", wrap=True)
