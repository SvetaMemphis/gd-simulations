from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from .core_torch import ReLUNet1D


def init_experiment1_k2(
    seed: int = 42,
    dtype: torch.dtype = torch.float32,
    w1_init: float = 1.0,
    b1_init: float = 1.0,
    w2_init: float = -10.0,
    b2_init: float = 10.0,
) -> ReLUNet1D:
    """
    Experiment 1 (k=2) thesis-style:
      v fixed = [1, -1]
      allow user-defined init for (w1,b1,w2,b2)
    """
    # seed currently not used for deterministic init, but kept for API symmetry
    _ = seed

    w = np.array([w1_init, w2_init], dtype=float)
    b = np.array([b1_init, b2_init], dtype=float)
    v = np.array([1.0, -1.0], dtype=float)

    model = ReLUNet1D(k=2, w_init=w, b_init=b, v_init=v, freeze_v=True, dtype=dtype)
    return model

def init_5f_run(
    rng: np.random.Generator,
    dtype: torch.dtype = torch.float32,
) -> Tuple[ReLUNet1D, float, float, float, float]:
    """
    Exp 5f init:
      w ~ N(0,2), b=0, v=[1,-1] fixed.
    Returns (model, w1_0,b1_0,w2_0,b2_0)
    """
    std = float(np.sqrt(2.0))  # N(0,2)
    w1_0 = float(rng.normal(0.0, std))
    w2_0 = float(rng.normal(0.0, std))
    b1_0 = float(rng.normal(0.0, std))
    b2_0 = float(rng.normal(0.0, std))

    w = np.array([w1_0, w2_0], dtype=float)
    b = np.array([b1_0, b2_0], dtype=float)
    v = np.array([1.0, -1.0], dtype=float)

    model = ReLUNet1D(k=2, w_init=w, b_init=b, v_init=v, freeze_v=True, dtype=dtype)
    return model, w1_0, b1_0, w2_0, b2_0


def init_6e_rich_k20(
    rng: np.random.Generator,
    k: int = 20,
    dtype: torch.dtype = torch.float32,
) -> ReLUNet1D:
    """
    Exp 6e rich init:
      v_j ∈ {-1, +1} (ensure both signs), fixed
      w,b ~ N(0,2)
    """
    while True:
        v = rng.choice([-1.0, 1.0], size=k).astype(float)
        if np.any(v > 0) and np.any(v < 0):
            break

    std = float(np.sqrt(2.0))
    w = rng.normal(0.0, std, size=k).astype(float)
    b = rng.normal(0.0, std, size=k).astype(float)

    model = ReLUNet1D(k=k, w_init=w, b_init=b, v_init=v, freeze_v=True, dtype=dtype)
    return model