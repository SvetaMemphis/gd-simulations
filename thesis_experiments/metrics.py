from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .core_torch import ReLUNet1D, forward_numpy


def find_decision_boundaries(
    model: ReLUNet1D,
    x_range: Tuple[float, float] = (-10, 10),
    resolution: int = 20000,
) -> List[float]:
    """
    Find x where f(x)=0 by scanning a grid and detecting sign changes.
    """
    x_vals = np.linspace(x_range[0], x_range[1], int(resolution), dtype=float)
    outputs = forward_numpy(model, x_vals)

    sign_changes = np.where((outputs[:-1] * outputs[1:]) <= 0)[0]

    boundaries: List[float] = []
    for i in sign_changes:
        if outputs[i] == 0:
            boundaries.append(float(x_vals[i]))
        elif outputs[i + 1] == 0:
            boundaries.append(float(x_vals[i + 1]))
        else:
            t = -outputs[i] / (outputs[i + 1] - outputs[i])
            boundary = x_vals[i] + t * (x_vals[i + 1] - x_vals[i])
            boundaries.append(float(boundary))
    return boundaries


def compute_margin(
    model: ReLUNet1D,
    x: np.ndarray,
    y: np.ndarray,
    resolution: int = 20000,
    x_range: Tuple[float, float] = (-10, 10),
) -> float:
    """
    1D margin: min_i distance from x_i to closest decision boundary, if correctly classified.
    If any point misclassified -> margin 0. If no boundary but all correct -> inf.
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    preds = np.sign(forward_numpy(model, x))
    # treat exact zero as correct only if y=0 (not our case), so set zeros to +1
    preds[preds == 0] = 1.0

    if not np.all(preds == y):
        return 0.0

    boundaries = find_decision_boundaries(model, x_range=x_range, resolution=resolution)
    if len(boundaries) == 0:
        return float("inf")

    b = np.asarray(boundaries)[None, :]
    dist = np.abs(x[:, None] - b)
    return float(np.min(np.min(dist, axis=1)))


def compute_margin_gap(
    model: ReLUNet1D,
    x: np.ndarray,
    y: np.ndarray,
    optimal_margin: float = 1.0,
    margin: Optional[float] = None,
    resolution: int = 20000,
    x_range: Tuple[float, float] = (-10, 10),
) -> float:
    if margin is None:
        margin = compute_margin(model, x, y, resolution=resolution, x_range=x_range)
    if margin == float("inf"):
        return 0.0
    return max(0.0, float(optimal_margin - margin))