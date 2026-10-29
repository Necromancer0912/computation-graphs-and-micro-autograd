"""
Autograd From Scratch (`src/`).

Pure Python and NumPy reverse-mode automatic differentiation engine:
- `ScalarValue`: Scalar computational DAG node with operator overloads and DFS topological backprop.
- `TensorValue`: N-dimensional matrix tensor engine with broadcasting un-reduction (`_unbroadcast`).
"""

from .autograd_engine import (
    ScalarValue,
    TensorValue,
    _unbroadcast,
)

unbroadcast_grad = _unbroadcast

from .plot_theme import (
    AMBER,
    BG_AXES,
    BG_CANVAS,
    CLASS_COLORS,
    CMAP_DECISION,
    CMAP_MULTICLASS,
    CMAP_SURFACE,
    CORAL,
    CYAN,
    EMERALD,
    MAGENTA,
    PALETTE,
    PURPLE,
    SLATE_BLUE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    apply_github_dark_style,
    plot_decision_boundary,
    plot_loss_landscape,
    save_fig,
    style_3d_axis,
)

__all__ = [
    "ScalarValue",
    "TensorValue",
    "unbroadcast_grad",
    "apply_github_dark_style",
    "style_3d_axis",
    "plot_decision_boundary",
    "plot_loss_landscape",
    "save_fig",
    "PALETTE",
    "CLASS_COLORS",
    "BG_CANVAS",
    "BG_AXES",
    "TEXT_PRIMARY",
    "TEXT_MUTED",
    "CYAN",
    "CORAL",
    "AMBER",
    "EMERALD",
    "PURPLE",
    "SLATE_BLUE",
    "MAGENTA",
    "CMAP_SURFACE",
    "CMAP_DECISION",
    "CMAP_MULTICLASS",
]
