from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class ReLUNet1D(nn.Module):
    """
    1D ReLU network:
      f(x) = sum_{j=1..k} v_j * ReLU(w_j x + b_j)
    """

    def __init__(
        self,
        k: int,
        w_init: Optional[np.ndarray] = None,
        b_init: Optional[np.ndarray] = None,
        v_init: Optional[np.ndarray] = None,
        freeze_v: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.k = int(k)
        self.dtype = dtype
        self.device = device if device is not None else torch.device("cpu")

        # parameters
        self.w = nn.Parameter(torch.randn(self.k, dtype=dtype, device=self.device))
        self.b = nn.Parameter(torch.randn(self.k, dtype=dtype, device=self.device))
        self.v = nn.Parameter(torch.randn(self.k, dtype=dtype, device=self.device))

        if w_init is not None:
            self.w.data = torch.as_tensor(w_init, dtype=dtype, device=self.device).clone()
        if b_init is not None:
            self.b.data = torch.as_tensor(b_init, dtype=dtype, device=self.device).clone()
        if v_init is not None:
            self.v.data = torch.as_tensor(v_init, dtype=dtype, device=self.device).clone()

        if freeze_v:
            self.v.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (n,) or (n,1)
        returns: shape (n,)
        """
        x = x.reshape(-1)  # (n,)
        pre = self.w[:, None] * x[None, :] + self.b[:, None]  # (k,n)
        act = torch.relu(pre)
        out = (self.v[:, None] * act).sum(dim=0)  # (n,)
        return out


def exponential_loss(y: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    # mean exp(-y*f(x))
    return torch.mean(torch.exp(-y * preds))


@torch.no_grad()
def forward_numpy(model: ReLUNet1D, x_np: np.ndarray) -> np.ndarray:
    x_t = torch.as_tensor(x_np, dtype=model.dtype, device=model.device).reshape(-1)
    y_t = model(x_t)
    return y_t.detach().cpu().numpy()


def train_gd(
    model: ReLUNet1D,
    x: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float,
    num_iterations: int,
    freeze_v: bool = False,
) -> List[float]:
    """
    Simple SGD with autograd.
    If freeze_v=True, we ensure v is not trainable (even if created trainable).
    Returns list of loss values (python floats).
    """
    if freeze_v:
        model.v.requires_grad_(False)

    params: Iterable[torch.nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=float(learning_rate))

    losses: List[float] = []
    for _t in range(int(num_iterations)):
        optimizer.zero_grad(set_to_none=True)
        preds = model(x)
        loss = exponential_loss(y, preds)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return losses


def gd_step_wb_only(
    model: ReLUNet1D,
    x: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float,
) -> float:
    """
    One SGD step updating only w,b (v fixed).
    Returns loss value after the update (float).
    """
    model.v.requires_grad_(False)
    model.w.requires_grad_(True)
    model.b.requires_grad_(True)

    optimizer = torch.optim.SGD([model.w, model.b], lr=float(learning_rate))

    optimizer.zero_grad(set_to_none=True)
    preds = model(x)
    loss = exponential_loss(y, preds)
    loss.backward()
    optimizer.step()

    return float(loss.detach().cpu().item())