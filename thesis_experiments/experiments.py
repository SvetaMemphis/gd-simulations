from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

from .core_torch import ReLUNet1D, exponential_loss, forward_numpy, gd_step_wb_only, train_gd, get_device
from .datasets import create_dataset_symmetric_2point
from .init_utils import init_5f_run, init_6e_rich_k20, init_experiment1_k2
from .metrics import compute_margin, compute_margin_gap, find_decision_boundaries

device = get_device()
print("Using device:", device)

model = init_experiment1_k2(...).to(device)

x = torch.tensor(x_np, dtype=torch.float32, device=device)
y = torch.tensor(y_np, dtype=torch.float32, device=device)

def experiment_1_k2(
    num_iterations: int = 2000,
    learning_rate: float = 0.01,
    seed: int = 42,
    w1_init: float = 1.0,
    b1_init: float = 1.0,
    w2_init: float = -10.0,
    b2_init: float = 10.0,
    x_range: Tuple[float, float] = (-10.0, 10.0),
) -> Dict[str, object]:
    """
    Experiment 1, restricted to k=2:
      - dataset x=[-1,1], y=[-1,1]
      - v fixed = [1,-1]
      - train only w,b (v frozen)
      - track loss, decision boundary count, margin, margin gap
      - saves a PNG
    """
    print("\n=== Experiment 1 (PyTorch): k=2 ===")

    # model = init_experiment1_k2(seed=seed, dtype=torch.float32)
    model = init_experiment1_k2(
    seed=seed,
    dtype=torch.float32,
    w1_init=w1_init,
    b1_init=b1_init,
    w2_init=w2_init,
    b2_init=b2_init,
)
    x_np, y_np = create_dataset_symmetric_2point()

    x = torch.tensor(x_np, dtype=model.dtype)
    y = torch.tensor(y_np, dtype=model.dtype)

    losses: List[float] = []
    boundary_counts: List[int] = []
    margins: List[float] = []
    margin_gaps: List[float] = []
    optimal_margin = 1.0

    for t in range(num_iterations):
        loss_t = gd_step_wb_only(model, x, y, learning_rate=learning_rate)
        losses.append(loss_t)

        bc = len(find_decision_boundaries(model, x_range=x_range))
        boundary_counts.append(int(bc))

        m = compute_margin(model, x_np, y_np, x_range=x_range)
        margins.append(float(m))

        mg = compute_margin_gap(model, x_np, y_np, optimal_margin=optimal_margin, margin=m, x_range=x_range)
        margin_gaps.append(float(mg))

        if (t + 1) % 500 == 0:
            print(f"t={t+1}, loss={loss_t:.6e}, boundaries={bc}, margin={m:.6f}, gap={mg:.6f}")

    print("Final params:")
    print("  w:", model.w.detach().cpu().numpy())
    print("  b:", model.b.detach().cpu().numpy())
    print("  v:", model.v.detach().cpu().numpy())

    # plot
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))

    axes[0, 0].plot(losses)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Loss vs Iteration")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True)

    axes[0, 1].plot(boundary_counts)
    axes[0, 1].set_title("Decision Boundaries vs Iteration")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("# boundaries")
    axes[0, 1].grid(True)

    axes[2, 0].plot(margins)
    axes[2, 0].axhline(1.0, linestyle="--", alpha=0.5, label="optimal margin (1.0)")
    axes[2, 0].legend()
    axes[2, 0].set_title("Margin vs Iteration")
    axes[2, 0].set_xlabel("Iteration")
    axes[2, 0].set_ylabel("Margin")
    axes[2, 0].grid(True)

    gap_abs = np.abs(np.asarray(margin_gaps, dtype=float))
    eps = 1e-12
    axes[2, 1].plot(np.maximum(gap_abs, eps))
    axes[2, 1].set_yscale("log")
    axes[2, 1].set_title("Absolute Margin Gap vs Iteration")
    axes[2, 1].set_xlabel("Iteration")
    axes[2, 1].set_ylabel("|optimal - margin|")
    axes[2, 1].grid(True)

    # function plot
    x_plot = np.linspace(-3, 3, 2000)
    f_plot = forward_numpy(model, x_plot)
    axes[1, 0].plot(x_plot, f_plot, label="Network output")
    axes[1, 0].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].scatter(
        x_np,
        [0] * len(x_np),
        c=["red" if yi < 0 else "blue" for yi in y_np],
        s=100,
        zorder=5,
        label="Data points",
    )
    axes[1, 0].set_title("Final Network Output")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("f(x)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # parameter evolution not tracked here; show w,b as constants at end
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.05, 0.95,
        f"Final parameters:\n"
        f"w={model.w.detach().cpu().numpy()}\n"
        f"b={model.b.detach().cpu().numpy()}\n"
        f"v={model.v.detach().cpu().numpy()}",
        va="top",
        family="monospace"
    )

    plt.tight_layout()
    out_path = "experiment_1_k2_torch.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot: {out_path}")

    return {
        "losses": losses,
        "boundary_counts": boundary_counts,
        "margins": margins,
        "margin_gaps": margin_gaps,
        "plot_path": out_path,
    }


def experiment_5f_hit_linear_condition_with_low_loss(
    num_runs: int = 10000,
    max_iterations: int = 10_000_000,
    learning_rate: float = 0.05,
    tol: float = 1e-3,
    loss_threshold: float = 0.5,
    seed: int = 42,
) -> None:
    """
    Exp 5f (PyTorch) — PRINTS EVERY RUN.
    """

    print("\n=== Experiment 5f (PyTorch) ===")
    print(f"num_runs={num_runs}, max_iterations={max_iterations}, lr={learning_rate}")
    print(f"condition: |w1+b1+w2-b2|<{tol} AND loss<{loss_threshold}")
    print("abort: if t==10000 and loss>=loss_threshold")

    rng = np.random.default_rng(seed)
    x_np, y_np = create_dataset_symmetric_2point()
    x_t = torch.tensor(x_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.float32)

    hit_times = np.full(num_runs, -1, dtype=int)
    metric_values: List[float] = []

    count_hit = 0
    count_loss_abort = 0
    count_max_iterations = 0

    for r in range(num_runs):

        print("\n" + "=" * 60)
        print(f"RUN {r+1}/{num_runs}")
        print("=" * 60)

        model, w1_0, b1_0, w2_0, b2_0 = init_5f_run(rng, dtype=torch.float32)

        print(f"Init:")
        print(f"  w1_0={w1_0:.6f}, b1_0={b1_0:.6f}")
        print(f"  w2_0={w2_0:.6f}, b2_0={b2_0:.6f}")

        t = 0
        stop_reason = "unknown"
        loss_last = None
        expr_last = None

        while t <= max_iterations:

            with torch.no_grad():
                preds = model(x_t)
                loss_t = float(exponential_loss(y_t, preds).cpu().item())

                w1 = float(model.w[0].cpu().item())
                b1 = float(model.b[0].cpu().item())
                w2 = float(model.w[1].cpu().item())
                b2 = float(model.b[1].cpu().item())

                expr_t = abs(w1 + b1 + w2 - b2)

            loss_last = loss_t
            expr_last = expr_t

            # abort rule
            if t == 10000 and not (loss_t < loss_threshold):
                stop_reason = "loss-abort"
                count_loss_abort += 1
                break

            # success condition
            if loss_t < loss_threshold and expr_t < tol:
                hit_times[r] = t
                stop_reason = "hit"
                count_hit += 1

                metric_min = min(abs(b2 - b1), abs(w1_0 + w2_0) / 2.0)
                metric_values.append(float(metric_min))
                break

            if t == max_iterations:
                stop_reason = "max-iterations"
                count_max_iterations += 1
                break

            _ = gd_step_wb_only(model, x_t, y_t, learning_rate=float(learning_rate))
            t += 1

        print(f"\nStop reason: {stop_reason}")
        print(f"Iterations: {t}")
        print(f"Final loss: {loss_last:.6e}")
        print(f"Final |w1+b1+w2-b2|: {expr_last:.6e}")

        print(f"Final params:")
        print(f"  w1={model.w[0].item():.6f}, b1={model.b[0].item():.6f}")
        print(f"  w2={model.w[1].item():.6f}, b2={model.b[1].item():.6f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Hit: {count_hit}/{num_runs}")
    print(f"Failed (loss-abort): {count_loss_abort}/{num_runs}")
    print(f"Failed (max-iterations): {count_max_iterations}/{num_runs}")

def experiment_6e_overparam_cluster_then_collapse_compare_margins(
    k: int = 20,
    d: int = 50,
    n: int = 1000,
    radius: float = 0.5,
    learning_rate: float = 0.01,
    max_pre_collapse_iters: int = 200_000,
    post_collapse_iters: int = 200_000,
    collapse_loss_threshold: float = 1e-5,
    seed: int = 42,
    track_every: int = 100,
    print_points_per_cluster: int = 10,
    print_neurons: int = 20,
    save_margins_path: str = "experiment_6e_margins.png",
    save_function_at_collapse_path: str = "experiment_6e_functions_at_collapse.png",
    save_rich_1d_path: str = "experiment_6e_rich_1d_no_collapse.png",
) -> Dict[str, object]:
    """
    Exp 6e (PyTorch):
      - init k=20, v in {-1,+1} fixed, w,b ~ N(0,2)
      - data: x_raw ~ Unif(B(0,radius)) in R^d, y ∈ {-1,+1}, then x[0]+=y
      - TRAIN using only x1 = x[0] with the 1D network, update w,b only
      - pre-collapse stops when loss < collapse_loss_threshold (or max_pre_collapse_iters)
      - collapse by behavior on dataset; remove dead neurons; group by m_plus>m_minus
      - IMPORTANT: collapse uses SUMS of w,b in each group (like your latest code variant)
      - compare margins pre+post rich vs collapsed
      - save:
          save_rich_1d_path
          save_function_at_collapse_path
          save_margins_path
    """
    print("\n=== Experiment 6e (PyTorch) ===")
    print(f"k={k}, d={d}, n={n}, radius={radius}, lr={learning_rate}")

    rng = np.random.default_rng(seed)
    model_rich = init_6e_rich_k20(rng, k=k, dtype=torch.float32)

    # sample uniform from d-ball
    def sample_uniform_ball(d_: int, R_: float) -> np.ndarray:
        u = rng.normal(size=d_)
        u /= (np.linalg.norm(u) + 1e-12)
        r = R_ * (rng.uniform() ** (1.0 / d_))
        return r * u

    X_full = np.zeros((n, d), dtype=float)
    y_full = rng.choice([-1.0, 1.0], size=n).astype(float)

    for i in range(n):
        x_raw = sample_uniform_ball(d, radius)
        x = x_raw.copy()
        x[0] += y_full[i]
        X_full[i] = x

    # print sample points
    def show_point(vec: np.ndarray) -> str:
        head = ", ".join(f"{vec[j]: .4f}" for j in range(min(6, vec.shape[0])))
        if vec.shape[0] > 6:
            return "[" + head + ", ...]"
        return "[" + head + "]"

    neg_idxs = np.where(y_full < 0)[0]
    pos_idxs = np.where(y_full > 0)[0]

    print("\nSampled data points (first coordinate shifted by label):")
    print("Negative label samples (y=-1):")
    for i in range(min(print_points_per_cluster, len(neg_idxs))):
        idx = int(neg_idxs[i])
        print(f"  {i:02d}: x={show_point(X_full[idx])}")

    print("\nPositive label samples (y=+1):")
    for i in range(min(print_points_per_cluster, len(pos_idxs))):
        idx = int(pos_idxs[i])
        print(f"  {i:02d}: x={show_point(X_full[idx])}")

    # train on x1
    x_rich_np = X_full[:, 0].astype(float)
    y_rich_np = y_full.astype(float)

    x_rich = torch.tensor(x_rich_np, dtype=torch.float32)
    y_rich = torch.tensor(y_rich_np, dtype=torch.float32)

    def print_neurons(title: str, model: ReLUNet1D, max_print: int):
        print(f"\n=== {title} ===")
        m = min(max_print, model.k)
        v = model.v.detach().cpu().numpy()
        w = model.w.detach().cpu().numpy()
        b = model.b.detach().cpu().numpy()
        for j in range(m):
            print(f"j={j:02d}: v={int(v[j]):+d}, w={w[j]: .6f}, b={b[j]: .6f}")
        if m < model.k:
            print(f"... (printed first {m} of {model.k})")
        print("=========================")

    print_neurons("Initialization (rich)", model_rich, print_neurons)

    # tracking
    t_all: List[int] = [0]
    margin_rich_all: List[float] = [float(compute_margin(model_rich, x_rich_np, y_rich_np, x_range=(-2, 2)))]
    margin_simple_all: List[float] = [float("nan")]

    t_collapse = None
    loss_at_collapse = None

    # pre-collapse train (w,b only)
    for t in range(1, max_pre_collapse_iters + 1):
        loss_t = gd_step_wb_only(model_rich, x_rich, y_rich, learning_rate=float(learning_rate))

        if t % track_every == 0:
            t_all.append(t)
            margin_rich_all.append(float(compute_margin(model_rich, x_rich_np, y_rich_np, x_range=(-2, 2))))
            margin_simple_all.append(float("nan"))

        if t % 10000 == 0:
            print(f"[pre-collapse] t={t}, loss={loss_t:.6e}")

        if loss_t < collapse_loss_threshold:
            t_collapse = t
            loss_at_collapse = loss_t
            break

    if t_collapse is None:
        t_collapse = max_pre_collapse_iters
        with torch.no_grad():
            preds = model_rich(x_rich)
            loss_at_collapse = float(exponential_loss(y_rich, preds).cpu().item())
        print("WARNING: Did not reach collapse_loss_threshold within max_pre_collapse_iters.")

    print(f"\n*** COLLAPSE at t={t_collapse} (loss={loss_at_collapse:.6e}) ***")

    print_neurons("Rich params at collapse", model_rich, print_neurons)

    # rich-only function plot + center values + zero crossings
    def count_zero_crossings(model: ReLUNet1D, x_min=-2.0, x_max=2.0, num=6000) -> int:
        xg = np.linspace(x_min, x_max, num)
        fg = forward_numpy(model, xg)
        s = np.sign(fg)
        s[s == 0] = 1
        return int(np.sum(s[:-1] * s[1:] < 0))

    xg = np.linspace(-2.0, 2.0, 1200)
    fg = forward_numpy(model_rich, xg)
    f_m1 = float(forward_numpy(model_rich, np.array([-1.0]))[0])
    f_p1 = float(forward_numpy(model_rich, np.array([1.0]))[0])
    crossings = count_zero_crossings(model_rich)

    plt.figure(figsize=(10, 6))
    plt.plot(xg, fg, label="f_rich(x) (k=20)")
    plt.axhline(0.0, linestyle="--")
    plt.scatter([-1.0, 1.0], [f_m1, f_p1], s=80, marker="X", label="centers x=-1, x=1")
    plt.title(f"Rich network at collapse (no collapse) | zero-crossings={crossings}")
    plt.xlabel("x (1D projection: first coordinate)")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_rich_1d_path, dpi=200)
    plt.close()
    print(f"Saved rich-only plot: {save_rich_1d_path}")
    print(f"f_rich(-1)={f_m1:.6f}, f_rich(1)={f_p1:.6f}, zero_crossings={crossings}")

    # collapse
    with torch.no_grad():
        pre = model_rich.w[:, None] * x_rich[None, :] + model_rich.b[:, None]  # (k,n)
        pre_np = pre.detach().cpu().numpy()

    alive_mask = np.any(pre_np > 0.0, axis=1)
    alive_idxs = np.where(alive_mask)[0]
    dead_idxs = np.where(~alive_mask)[0]
    print(f"\nDead neurons: {len(dead_idxs)} / {k} (alive={len(alive_idxs)})")

    if len(alive_idxs) == 0:
        raise RuntimeError("All neurons are dead — cannot collapse.")

    pos_mask = (y_rich_np > 0)
    neg_mask = (y_rich_np < 0)

    pre_alive = pre_np[alive_idxs, :]
    m_plus = pre_alive[:, pos_mask].mean(axis=1)
    m_minus = pre_alive[:, neg_mask].mean(axis=1)

    group_plus_alive = alive_idxs[np.where(m_plus > m_minus)[0]]
    group_minus_alive = alive_idxs[np.where(m_plus <= m_minus)[0]]

    if len(group_plus_alive) == 0 or len(group_minus_alive) == 0:
        print("Degenerate grouping (alive only) — fallback split.")
        mid = len(alive_idxs) // 2
        group_plus_alive = alive_idxs[:mid]
        group_minus_alive = alive_idxs[mid:]

    w_np = model_rich.w.detach().cpu().numpy()
    b_np = model_rich.b.detach().cpu().numpy()

    # IMPORTANT: sum (not mean)
    w_pos = float(w_np[group_plus_alive].sum())
    b_pos = float(b_np[group_plus_alive].sum())
    w_neg = float(w_np[group_minus_alive].sum())
    b_neg = float(b_np[group_minus_alive].sum())

    print("\nCollapsed groups (alive only):")
    print(f"  group_plus size={len(group_plus_alive)} -> (w,b)=({w_pos:.6f}, {b_pos:.6f})")
    print(f"  group_minus size={len(group_minus_alive)} -> (w,b)=({w_neg:.6f}, {b_neg:.6f})")

    model_simple = ReLUNet1D(
        k=2,
        w_init=np.array([w_pos, w_neg], dtype=float),
        b_init=np.array([b_pos, b_neg], dtype=float),
        v_init=np.array([1.0, -1.0], dtype=float),
        freeze_v=True,
        dtype=torch.float32,
    )

    x_simple_np = np.array([-1.0, 1.0], dtype=float)
    y_simple_np = np.array([-1.0, 1.0], dtype=float)

    x_simple = torch.tensor(x_simple_np, dtype=torch.float32)
    y_simple = torch.tensor(y_simple_np, dtype=torch.float32)

    # plot functions at collapse
    x_plot = np.linspace(-2.0, 2.0, 800)
    f_rich_c = forward_numpy(model_rich, x_plot)
    f_simp_c = forward_numpy(model_simple, x_plot)

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, f_rich_c, label="Rich f(x) at collapse (k=20)")
    plt.plot(x_plot, f_simp_c, label="Collapsed f(x) at collapse (k=2)")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("x (1D projection: first coordinate)")
    plt.ylabel("f(x)")
    plt.title("Function at collapse: rich vs collapsed")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_function_at_collapse_path, dpi=200)
    plt.close()
    print(f"Saved function-at-collapse plot: {save_function_at_collapse_path}")

    # ensure we have a margin value for collapsed at t_collapse
    if t_all[-1] != t_collapse:
        t_all.append(int(t_collapse))
        margin_rich_all.append(float(compute_margin(model_rich, x_rich_np, y_rich_np, x_range=(-2, 2))))
        margin_simple_all.append(float(compute_margin(model_simple, x_simple_np, y_simple_np, x_range=(-2, 2))))
    else:
        margin_simple_all[-1] = float(compute_margin(model_simple, x_simple_np, y_simple_np, x_range=(-2, 2)))

    # post-collapse train both (w,b only)
    for s in range(1, post_collapse_iters + 1):
        _ = gd_step_wb_only(model_rich, x_rich, y_rich, learning_rate=float(learning_rate))
        _ = gd_step_wb_only(model_simple, x_simple, y_simple, learning_rate=float(learning_rate))

        if s % track_every == 0:
            t_all.append(int(t_collapse + s))
            margin_rich_all.append(float(compute_margin(model_rich, x_rich_np, y_rich_np, x_range=(-2, 2))))
            margin_simple_all.append(float(compute_margin(model_simple, x_simple_np, y_simple_np, x_range=(-2, 2))))

    # margin plot
    plt.figure(figsize=(10, 6))
    plt.plot(t_all, margin_rich_all, label="Rich margin (k=20)")
    plt.plot(t_all, margin_simple_all, label="Collapsed margin (k=2)")
    plt.axvline(int(t_collapse), color="red", linewidth=2, label="Collapse time")
    plt.xlabel("Iteration")
    plt.ylabel("Margin")
    plt.title("Margins: rich vs collapsed (red line = collapse time)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_margins_path, dpi=200)
    plt.close()
    print(f"Saved margins plot: {save_margins_path}")

    return {
        "t_collapse": int(t_collapse),
        "loss_at_collapse": float(loss_at_collapse),
        "save_rich_1d_path": save_rich_1d_path,
        "save_function_at_collapse_path": save_function_at_collapse_path,
        "save_margins_path": save_margins_path,
        "group_plus_size": int(len(group_plus_alive)),
        "group_minus_size": int(len(group_minus_alive)),
    }