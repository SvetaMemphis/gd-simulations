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


def network_forward(params: NetworkParams, X: np.ndarray) -> np.ndarray:
    """
    Forward pass for d-dimensional inputs.

    Args:
        params: network parameters with w shape (k,d), b shape (k,), v shape (k,)
        X: array-like of shape (n,d) or (d,) for a single point

    Returns:
        outputs of shape (n,)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        # X = X.reshape(1, -1)
        X = X.reshape(-1, 1)

    # pre-activations: (k,n) = (k,d) @ (d,n) + (k,1)
    pre = params.w @ X.T + params.b[:, None]          # (k,n)
    act = np.maximum(pre, 0.0)                        # (k,n)
    out = (params.v[:, None] * act).sum(axis=0)       # (n,)
    return out.reshape(-1)


def exponential_loss(y: np.ndarray, predictions: np.ndarray, clip: float = 50.0) -> float:
    """
    Mean exponential loss: mean_i exp(-y_i * f(x_i)), with clipping for numerical stability.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)

    z = -y * predictions
    z = np.clip(z, -clip, clip)
    return float(np.mean(np.exp(z)))

def compute_gradients(
    params: NetworkParams,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: Callable = exponential_loss,
) -> NetworkParams:
    """
    Compute gradients of mean exponential loss wrt (w,b,v) for dD network.

    L = mean_i exp(-y_i f(x_i)), f(x)=sum_j v_j ReLU(<w_j,x> + b_j)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        # X = X.reshape(1, -1)
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1)

    n, d = X.shape
    k = params.k

    pre = params.w @ X.T + params.b[:, None]              # (k,n)
    relu_pre = np.maximum(pre, 0.0)                       # (k,n)
    outputs = (params.v[:, None] * relu_pre).sum(axis=0)  # (n,)

    # dL_dout = -(1.0 / n) * y * np.exp(-y * outputs)       # (n,)
    z = -y * outputs
    z = np.clip(z, -50.0, 50.0)          # same clip as loss
    exp_z = np.exp(z)

    dL_dout = -(1.0 / n) * y * exp_z
    indicator = (pre > 0).astype(float)                   # (k,n)

    grad_v = (dL_dout[None, :] * relu_pre).sum(axis=1)    # (k,)

    scale = (dL_dout[None, :] * params.v[:, None] * indicator)  # (k,n)
    grad_w = scale @ X                                    # (k,n)@(n,d)->(k,d)
    grad_b = scale.sum(axis=1)                            # (k,)

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


def train_gd(
    initial_params: NetworkParams,
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    num_iterations: int,
    loss_fn: Callable = exponential_loss,
    track_boundaries: bool = True,
    track_margin: bool = False,
    optimal_margin: float = 1.0,
    x_range: Tuple[float, float] = (-10, 10),
):
    """
    Train network using GD.

    Returns:
        (params_history, losses, boundary_counts, margins, margin_gaps)
    """
    from .metrics import compute_margin, compute_margin_gap, find_decision_boundaries

    params = initial_params.copy()
    params_history: List[NetworkParams] = [params.copy()]
    losses: List[float] = []
    boundary_counts: List[int] = []
    margins: Optional[List[float]] = [] if track_margin else None
    margin_gaps: Optional[List[float]] = [] if track_margin else None

    for _t in range(num_iterations):
        params, loss = gradient_descent_step(params, x, y, learning_rate, loss_fn)
        params_history.append(params.copy())
        losses.append(loss)

        if track_boundaries:
            boundary_counts.append(len(find_decision_boundaries(params, x_range=x_range)))

        if track_margin:
            m = compute_margin(params, x, y, x_range=x_range)
            gaps = compute_margin_gap(params, x, y, optimal_margin=optimal_margin, margin=m, x_range=x_range)
            margins.append(m)  # type: ignore[union-attr]
            margin_gaps.append(gaps)  # type: ignore[union-attr]

    return params_history, losses, boundary_counts, margins, margin_gaps
    

