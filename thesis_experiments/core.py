import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass
class NetworkParams:
    """Parameters for a 1D ReLU network with k neurons: Phi(x)=sum_j v_j ReLU(w_j x + b_j)."""

    w: np.ndarray  # shape (k,)
    b: np.ndarray  # shape (k,)
    v: np.ndarray  # shape (k,)

    def __post_init__(self):
        assert len(self.w) == len(self.b) == len(self.v)

    @property
    def k(self) -> int:
        return len(self.w)

    def copy(self) -> "NetworkParams":
        return NetworkParams(w=self.w.copy(), b=self.b.copy(), v=self.v.copy())


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def network_forward(params: NetworkParams, x: np.ndarray) -> np.ndarray:
    """
    Forward pass for 1D inputs.

    Args:
        params: network parameters
        x: array-like of shape (n,)

    Returns:
        outputs of shape (n,)
    """
    x = np.asarray(x).reshape(-1)
    # (k, n) pre-activations
    pre = params.w[:, None] * x[None, :] + params.b[:, None]
    act = relu(pre)
    return (params.v[:, None] * act).sum(axis=0)


def exponential_loss(y: np.ndarray, predictions: np.ndarray) -> float:
    """Mean exponential loss: mean_i exp(-y_i * f(x_i))."""
    return float(np.mean(np.exp(-y * predictions)))


def compute_gradients(
    params: NetworkParams,
    x: np.ndarray,
    y: np.ndarray,
    loss_fn: Callable = exponential_loss,
) -> NetworkParams:
    """
    Compute gradients of mean exponential loss wrt (w,b,v).

    Note: `loss_fn` is currently unused (kept for API compatibility).
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    k = params.k
    n = len(x)

    pre = params.w[:, None] * x[None, :] + params.b[:, None]  # (k, n)
    relu_pre = relu(pre)  # (k, n)
    outputs = (params.v[:, None] * relu_pre).sum(axis=0)  # (n,)

    # dL/dout_i = -(1/n) y_i exp(-y_i out_i)
    dL_dout = -(1.0 / n) * y * np.exp(-y * outputs)  # (n,)

    indicator = (pre > 0).astype(float)  # (k, n)

    grad_v = (dL_dout[None, :] * relu_pre).sum(axis=1)  # (k,)
    grad_w = (dL_dout[None, :] * params.v[:, None] * indicator * x[None, :]).sum(axis=1)  # (k,)
    grad_b = (dL_dout[None, :] * params.v[:, None] * indicator).sum(axis=1)  # (k,)

    return NetworkParams(w=grad_w, b=grad_b, v=grad_v)


def gradient_descent_step(
    params: NetworkParams,
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    loss_fn: Callable = exponential_loss,
) -> Tuple[NetworkParams, float]:
    grads = compute_gradients(params, x, y, loss_fn)
    new_params = NetworkParams(
        w=params.w - learning_rate * grads.w,
        b=params.b - learning_rate * grads.b,
        v=params.v if params.k == 2 else params.v - learning_rate * grads.v,
    )
    loss = loss_fn(np.asarray(y), network_forward(new_params, np.asarray(x)))
    return new_params, loss

