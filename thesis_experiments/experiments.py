import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Callable

import csv

from .core import network_forward, train_gd, exponential_loss, gradient_descent_step, NetworkParams, compute_gradients
from .datasets import create_dataset
from .init_utils import initialize_network, print_initial_params
from .metrics import compute_margin


def experiment_1_arbitrary_neurons(
    k: int = 5,
    num_iterations: int = 1000,
    learning_rate: float = 0.01,
    init_type: str = "random",
    seed: int = 42,
    w1_init: float = 1.0,
    b1_init: float = 1.0,
    w2_init: float = -10.0,
    b2_init: float = 10.0,
    v_binary: bool = False,
):
    """
    Experiment 1: Simulate GD on arbitrary number of neurons with arbitrary initialization.
    """
    print(f"\n=== Experiment 1: {k} neurons, {init_type} initialization ===")

    # params = initialize_network(k, init_type=init_type, seed=seed)
    params = initialize_network(
    k,
    init_type=init_type,
    seed=seed,
    w1_init=w1_init,
    b1_init=b1_init,
    w2_init=w2_init,
    b2_init=b2_init,
    v_binary=v_binary,
)
    if k == 2:
        params.v = np.array([1.0, -1.0], dtype=float)

    print_initial_params(params, "Initial Parameters")

    x, y = create_dataset(symmetric=True)
    print(f"Dataset: x={x}, y={y}")

    # params_history, losses, boundary_counts, margins, _ = train_gd(
    #     params, x, y, learning_rate, num_iterations, track_boundaries=True
    # )

    params_history, losses, boundary_counts, margins, margin_gaps = train_gd(
        params,
        x,
        y,
        learning_rate,
        num_iterations,
        track_boundaries=True,
        track_margin=True,
        optimal_margin=1.0,   # for your symmetric dataset [-1,1], this is correct
    )


    print(f"\nTraining completed:")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Final boundaries: {boundary_counts[-1]}")
    print(f"  Final parameters:")
    print(f"    w: {params_history[-1].w}")
    print(f"    b: {params_history[-1].b}")
    print(f"    v: {params_history[-1].v}")

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))

    # Loss
    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss vs Iteration")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True)

    # Margin
    axes[2, 0].plot(margins)
    axes[2, 0].set_xlabel("Iteration")
    axes[2, 0].set_ylabel("Margin")
    axes[2, 0].set_title("Margin vs Iteration")
    axes[2, 0].grid(True)
    axes[2, 0].axhline(1.0, linestyle="--", alpha=0.5, label="optimal margin (1.0)")
    axes[2, 0].legend()

    # Margin gap (use absolute value; guard for log scale)
    gap_abs = np.abs(np.asarray(margin_gaps, dtype=float))
    eps = 1e-12  # avoid log(0)
    axes[2, 1].plot(np.maximum(gap_abs, eps))
    axes[2, 1].set_xlabel("Iteration")
    axes[2, 1].set_ylabel("|optimal margin − margin|")
    axes[2, 1].set_title("Absolute margin gap vs Iteration")
    axes[2, 1].set_yscale("log")  # now always safe
    axes[2, 1].grid(True)


    axes[0, 1].plot(boundary_counts)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Number of Decision Boundaries")
    axes[0, 1].set_title("Decision Boundaries vs Iteration")
    axes[0, 1].grid(True)

    x_plot = np.linspace(-3, 3, 1000)
    y_plot = network_forward(params_history[-1], x_plot)
    axes[1, 0].plot(x_plot, y_plot, label="Network output")
    axes[1, 0].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].axvline(0, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].scatter(
        x,
        [0] * len(x),
        c=["red" if yi < 0 else "blue" for yi in y],
        s=100,
        zorder=5,
        label="Data points",
    )
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("Phi(x)")
    axes[1, 0].set_title("Final Network Output")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    for j in range(min(20, k)):
        w_vals = [p.w[j] for p in params_history]
        axes[1, 1].plot(w_vals, label=f"w_{j+1}")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Parameter Value")
    axes[1, 1].set_title("Parameter Evolution")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(f"experiment_1_k{k}_{init_type}.png", dpi=150)
    print(f"\nPlot saved to experiment_1_k{k}_{init_type}.png")

    return params_history, losses, boundary_counts, margins, margin_gaps


def experiment_2_boundary_count(
    k: int = 2,
    num_iterations: int = 1000,
    learning_rate: float = 0.01,
    init_type: str = "thesis",
    w1_init: float = 1.0,
    b1_init: float = 1.0,
    M: float = 10.0,
):
    """
    Experiment 2: Compute number of x-axis intersections as a function of iteration.
    """
    print(f"\n=== Experiment 2: Boundary count over time ===")

    params = initialize_network(k, init_type=init_type, w1_init=w1_init, b1_init=b1_init, M=M)
    x, y = create_dataset(symmetric=True)

    _, losses, boundary_counts, _, _ = train_gd(
        params, x, y, learning_rate, num_iterations, track_boundaries=True, x_range=(-10, 10)
    )

    plt.figure(figsize=(10, 6))
    plt.plot(boundary_counts, linewidth=2)
    plt.xlabel("Iteration t", fontsize=12)
    plt.ylabel("Number of Decision Boundaries", fontsize=12)
    plt.title(f"Number of x-axis Intersections vs Iteration (k={k})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig("experiment_2_boundary_count.png", dpi=150)
    print("Plot saved to experiment_2_boundary_count.png")
    print(f"Initial boundaries: {boundary_counts[0]}")
    print(f"Final boundaries: {boundary_counts[-1]}")
    print(f"Max boundaries: {max(boundary_counts)}")

    return boundary_counts


def experiment_3_robust_case(
    k: int = 2,
    num_iterations: int = 1000,
    learning_rate: float = 0.01,
    shifts: List[float] = [0.0, -0.4, -0.8, -0.999],
    init_type: str = "thesis",
    w1_init: float = 1.0,
    b1_init: float = 1.0,
    M: float = 10.0,
):
    """
    Experiment 3: Test robust case (training set moved closer to origin).
    """
    print(f"\n=== Experiment 3: Robust case (shifted dataset) ===")

    results = {}

    for shift in shifts:
        print(f"\nTesting shift = {shift}")

        params = initialize_network(k, init_type=init_type, w1_init=w1_init, b1_init=b1_init, M=M)
        print("\n  Initial parameters:")
        print(f"    w: {params.w}")
        print(f"    b: {params.b}")
        print(f"    v: {params.v}")

        x, y = create_dataset(symmetric=True, shift=shift)
        print(f"  Dataset: x={x}, y={y}")

        params_history, losses, boundary_counts, _, _ = train_gd(
            params, x, y, learning_rate, num_iterations, track_boundaries=True
        )

        results[shift] = {
            "losses": losses,
            "boundary_counts": boundary_counts,
            "final_params": params_history[-1],
            "x": x,
        }

        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Final boundaries: {boundary_counts[-1]}")
        final_params = params_history[-1]

        print(f"  Final parameters:")
        print(f"    w: {final_params.w}")
        print(f"    b: {final_params.b}")
        print(f"    v: {final_params.v}")


    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for shift in shifts:
        axes[0, 0].plot(results[shift]["losses"], label=f"shift={shift}")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss vs Iteration (Different Shifts)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    for shift in shifts:
        axes[0, 1].plot(results[shift]["boundary_counts"], label=f"shift={shift}")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Number of Decision Boundaries")
    axes[0, 1].set_title("Boundaries vs Iteration (Different Shifts)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    x_plot = np.linspace(-2, 2, 1000)
    for shift in shifts:
        y_plot = network_forward(results[shift]["final_params"], x_plot)
        axes[1, 0].plot(x_plot, y_plot, label=f"shift={shift}")
    axes[1, 0].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].axvline(0, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("Phi(x)")
    axes[1, 0].set_title("Final Network Outputs")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    for shift in shifts:
        x_data = results[shift]["x"]
        axes[1, 1].scatter(x_data, [shift] * len(x_data), label=f"shift={shift}", s=100, alpha=0.7)
    axes[1, 1].axvline(0, color="k", linestyle="--", alpha=0.5)
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("Shift")
    axes[1, 1].set_title("Data Point Positions")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("experiment_3_robust_case.png", dpi=150)
    print("\nPlot saved to experiment_3_robust_case.png")

    return results


def experiment_4_non_symmetric(
    k: int = 2,
    num_iterations: int = 1000,
    learning_rate: float = 0.01,
    init_type: str = "random",
    seed: int = 42,
):
    """
    Experiment 4: Simulations over non-symmetric data.
    """
    print(f"\n=== Experiment 4: Non-symmetric data ===")

    configs = [
        {"x": np.array([-0.5, 1.5]), "y": np.array([-1.0, 1.0]), "name": "asymmetric_1"},
        {"x": np.array([-1.5, 0.5]), "y": np.array([-1.0, 1.0]), "name": "asymmetric_2"},
        {"x": np.array([-0.8, 1.2]), "y": np.array([-1.0, 1.0]), "name": "asymmetric_3"},
    ]

    results = {}

    for config in configs:
        print(f"\nTesting {config['name']}: x={config['x']}, y={config['y']}")
        params = initialize_network(k, init_type=init_type, seed=seed)

        params_history, losses, boundary_counts, _, _ = train_gd(
            params, config["x"], config["y"], learning_rate, num_iterations, track_boundaries=True
        )

        results[config["name"]] = {
            "losses": losses,
            "boundary_counts": boundary_counts,
            "final_params": params_history[-1],
            "x": config["x"],
            "y": config["y"],
        }
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Final boundaries: {boundary_counts[-1]}")

    print(f"\nTesting symmetric case for comparison")
    params_sym = initialize_network(k, init_type=init_type, seed=seed)
    x_sym, y_sym = create_dataset(symmetric=True)
    params_history_sym, losses_sym, boundary_counts_sym, _, _ = train_gd(
        params_sym, x_sym, y_sym, learning_rate, num_iterations, track_boundaries=True
    )
    results["symmetric"] = {
        "losses": losses_sym,
        "boundary_counts": boundary_counts_sym,
        "final_params": params_history_sym[-1],
        "x": x_sym,
        "y": y_sym,
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for name in results:
        axes[0, 0].plot(results[name]["losses"], label=name)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss vs Iteration (Symmetric vs Non-symmetric)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    for name in results:
        axes[0, 1].plot(results[name]["boundary_counts"], label=name)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Number of Decision Boundaries")
    axes[0, 1].set_title("Boundaries vs Iteration")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    x_plot = np.linspace(-2, 2, 1000)
    for name in results:
        y_plot = network_forward(results[name]["final_params"], x_plot)
        axes[1, 0].plot(x_plot, y_plot, label=name, alpha=0.7)
    axes[1, 0].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].axvline(0, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("Phi(x)")
    axes[1, 0].set_title("Final Network Outputs")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    for name in results:
        x_data = results[name]["x"]
        y_data = results[name]["y"]
        colors = ["red" if yi < 0 else "blue" for yi in y_data]
        axes[1, 1].scatter(x_data, [0] * len(x_data), c=colors, label=name, s=100, alpha=0.7)
    axes[1, 1].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[1, 1].axvline(0, color="k", linestyle="--", alpha=0.5)
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y (dummy)")
    axes[1, 1].set_title("Data Point Positions")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("experiment_4_non_symmetric.png", dpi=150)
    print("\nPlot saved to experiment_4_non_symmetric.png")

    return results


def experiment_5_overparameterized(
    k_values: List[int] = [2, 5, 10, 20, 50],
    num_iterations: int = 2000,
    learning_rate: float = 0.01,
    num_runs: int = 5,
    seed: int = 42,
):
    """
    Experiment 5a: Over-parameterized regime.
    """
    print(f"\n=== Experiment 5a: Over-parameterized regime ===")

    x, y = create_dataset(symmetric=True)
    optimal_margin = 1.0

    results = {}
    for k in k_values:
        print(f"\nTesting k={k} neurons")
        margins_all = []
        margin_gaps_all = []
        losses_all = []

        for run in range(num_runs):
            params = initialize_network(k, init_type="random", seed=seed + run)
            _, losses, _, margins, margin_gaps = train_gd(
                params,
                x,
                y,
                learning_rate,
                num_iterations,
                track_boundaries=False,
                track_margin=True,
                optimal_margin=optimal_margin,
            )
            margins_all.append(margins)
            margin_gaps_all.append(margin_gaps)
            losses_all.append(losses)

        results[k] = {"margins": margins_all, "margin_gaps": margin_gaps_all, "losses": losses_all}

        final_margins = [m[-1] if len(m) > 0 else 0 for m in margins_all]
        final_gaps = [g[-1] if len(g) > 0 else optimal_margin for g in margin_gaps_all]
        print(f"  Final margin: {np.mean(final_margins):.6f} ± {np.std(final_margins):.6f}")
        print(f"  Final margin gap: {np.mean(final_gaps):.6f} ± {np.std(final_gaps):.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for k in k_values:
        avg_margins = np.mean(results[k]["margins"], axis=0)
        axes[0, 0].plot(avg_margins, label=f"k={k}")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Margin")
    axes[0, 0].set_title("Margin Convergence (Over-parameterized)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    for k in k_values:
        avg_gaps = np.mean(results[k]["margin_gaps"], axis=0)
        axes[0, 1].plot(avg_gaps, label=f"k={k}")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Margin Gap")
    axes[0, 1].set_title("Margin Gap Convergence")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    final_margins_mean = [
        np.mean([m[-1] if len(m) > 0 else 0 for m in results[k]["margins"]]) for k in k_values
    ]
    final_margins_std = [
        np.std([m[-1] if len(m) > 0 else 0 for m in results[k]["margins"]]) for k in k_values
    ]
    axes[1, 0].errorbar(k_values, final_margins_mean, yerr=final_margins_std, marker="o", capsize=5)
    axes[1, 0].set_xlabel("Number of Neurons (k)")
    axes[1, 0].set_ylabel("Final Margin")
    axes[1, 0].set_title("Final Margin vs Over-parameterization")
    axes[1, 0].grid(True)

    convergence_iters = {}
    for k in k_values:
        iters = []
        for gaps in results[k]["margin_gaps"]:
            for i, gap in enumerate(gaps):
                if gap < 0.1:
                    iters.append(i)
                    break
        convergence_iters[k] = iters if iters else [num_iterations] * num_runs

    conv_mean = [np.mean(convergence_iters[k]) for k in k_values]
    conv_std = [np.std(convergence_iters[k]) for k in k_values]
    axes[1, 1].errorbar(k_values, conv_mean, yerr=conv_std, marker="o", capsize=5)
    axes[1, 1].set_xlabel("Number of Neurons (k)")
    axes[1, 1].set_ylabel("Iterations to Margin Gap < 0.1")
    axes[1, 1].set_title("Convergence Rate vs Over-parameterization")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("experiment_5a_overparameterized.png", dpi=150)
    print("\nPlot saved to experiment_5a_overparameterized.png")
    return results


def experiment_5b_highdimensional_clustered(
    d_values: List[int] = [1, 2, 5, 10],
    num_clusters: int = 2,
    points_per_cluster: int = 10,
    num_iterations: int = 2000,
    learning_rate: float = 0.01,
    k: int = 10,
    seed: int = 42,
):
    """
    Experiment 5b: High-dimensional clustered data (simulated via 1D projections).
    """
    print(f"\n=== Experiment 5b: High-dimensional clustered data ===")

    np.random.seed(seed)
    results = {}

    for d in d_values:
        print(f"\nTesting effective dimension d={d}")

        x_list = []
        y_list = []
        for cluster_id in range(num_clusters):
            center = (cluster_id * 2 - 1) * (1.0 + 0.1 * d)
            cluster_points = center + np.random.randn(points_per_cluster) * (0.1 + 0.05 * d)
            x_list.extend(cluster_points)
            y_list.extend([1.0 if cluster_id == 0 else -1.0] * points_per_cluster)

        x = np.array(x_list)
        y = np.array(y_list)

        optimal_margin = (
            np.min([abs(xi - xj) for i, xi in enumerate(x) for j, xj in enumerate(x) if y[i] != y[j]])
            / 2
        )

        print(f"  Dataset size: {len(x)} points")
        print(f"  Optimal margin (approx): {optimal_margin:.6f}")

        margins_all = []
        margin_gaps_all = []
        for run in range(3):
            params = initialize_network(k, init_type="random", seed=seed + run)
            _, _, _, margins, margin_gaps = train_gd(
                params,
                x,
                y,
                learning_rate,
                num_iterations,
                track_boundaries=False,
                track_margin=True,
                optimal_margin=optimal_margin,
            )
            margins_all.append(margins)
            margin_gaps_all.append(margin_gaps)

        results[d] = {
            "margins": margins_all,
            "margin_gaps": margin_gaps_all,
            "x": x,
            "y": y,
            "optimal_margin": optimal_margin,
        }

        final_margins = [m[-1] if len(m) > 0 else 0 for m in margins_all]
        print(f"  Final margin: {np.mean(final_margins):.6f} ± {np.std(final_margins):.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for d in d_values:
        avg_margins = np.mean(results[d]["margins"], axis=0)
        axes[0, 0].plot(avg_margins, label=f"d={d}")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Margin")
    axes[0, 0].set_title("Margin Convergence (High-dimensional)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    for d in d_values:
        avg_gaps = np.mean(results[d]["margin_gaps"], axis=0)
        axes[0, 1].plot(avg_gaps, label=f"d={d}")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Margin Gap")
    axes[0, 1].set_title("Margin Gap Convergence")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    final_margins_mean = [
        np.mean([m[-1] if len(m) > 0 else 0 for m in results[d]["margins"]]) for d in d_values
    ]
    final_margins_std = [
        np.std([m[-1] if len(m) > 0 else 0 for m in results[d]["margins"]]) for d in d_values
    ]
    axes[1, 0].errorbar(d_values, final_margins_mean, yerr=final_margins_std, marker="o", capsize=5)
    axes[1, 0].set_xlabel("Effective Dimension (d)")
    axes[1, 0].set_ylabel("Final Margin")
    axes[1, 0].set_title("Final Margin vs Dimension")
    axes[1, 0].grid(True)

    if 1 in d_values:
        x_plot = np.linspace(results[1]["x"].min() - 0.5, results[1]["x"].max() + 0.5, 1000)
        params_final = initialize_network(k, init_type="random", seed=seed + 2)
        params_history_final, _, _, _, _ = train_gd(
            params_final, results[1]["x"], results[1]["y"], learning_rate, num_iterations, track_boundaries=False
        )
        y_plot = network_forward(params_history_final[-1], x_plot)
        axes[1, 1].plot(x_plot, y_plot, alpha=0.7, label="Network output")
        axes[1, 1].scatter(
            results[1]["x"],
            [0] * len(results[1]["x"]),
            c=["blue" if yi > 0 else "red" for yi in results[1]["y"]],
            s=50,
            alpha=0.7,
            label="Data points",
        )
        axes[1, 1].axhline(0, color="k", linestyle="--", alpha=0.5)
        axes[1, 1].set_xlabel("x")
        axes[1, 1].set_ylabel("Phi(x)")
        axes[1, 1].set_title("Clustered Data (d=1)")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("experiment_5b_highdimensional.png", dpi=150)
    print("\nPlot saved to experiment_5b_highdimensional.png")

    return results


def experiment_5c_margin_convergence_rate(
    k: int = 2,
    num_runs: int = 50,
    max_iterations: int = 10000,
    learning_rate: float = 0.01,
    delta: float = 0.01,
    epsilon: float = 0.01,
    seed: int = 42,
):
    """
    Experiment 5c: random initializations; aggregate iterations until margin_gap<delta and loss<epsilon.
    """
    print(f"\n=== Experiment 5c: Margin convergence from random initializations ===")
    print(f"  Target: margin gap < {delta}, loss < {epsilon}")
    print(f"  Running {num_runs} random initializations...")

    x, y = create_dataset(symmetric=True)
    optimal_margin = 1.0

    iterations_to_margin = []
    iterations_to_loss = []
    iterations_to_both = []
    failed_margin = 0
    failed_loss = 0
    failed_both = 0

    for run in range(num_runs):
        params = initialize_network(k, init_type="random", seed=seed + run)
        _, losses, _, _margins, margin_gaps = train_gd(
            params,
            x,
            y,
            learning_rate,
            max_iterations,
            track_boundaries=False,
            track_margin=True,
            optimal_margin=optimal_margin,
        )

        margin_reached = False
        for i, gap in enumerate(margin_gaps):
            if gap < delta:
                iterations_to_margin.append(i)
                margin_reached = True
                break
        if not margin_reached:
            failed_margin += 1

        loss_reached = False
        for i, loss in enumerate(losses):
            if loss < epsilon:
                iterations_to_loss.append(i)
                loss_reached = True
                break
        if not loss_reached:
            failed_loss += 1

        both_reached = False
        for i in range(min(len(margin_gaps), len(losses))):
            if margin_gaps[i] < delta and losses[i] < epsilon:
                iterations_to_both.append(i)
                both_reached = True
                break
        if not both_reached:
            failed_both += 1

        if (run + 1) % 10 == 0:
            print(f"  Completed {run + 1}/{num_runs} runs...")

    print(f"\nResults:")
    print(f"  Margin gap < {delta}:")
    if iterations_to_margin:
        print(f"    Success: {len(iterations_to_margin)}/{num_runs}")
        print(f"    Mean iterations: {np.mean(iterations_to_margin):.1f}")
        print(f"    Median iterations: {np.median(iterations_to_margin):.1f}")
        print(f"    Std: {np.std(iterations_to_margin):.1f}")
        print(f"    Min: {np.min(iterations_to_margin)}, Max: {np.max(iterations_to_margin)}")
    else:
        print(f"    Failed: {failed_margin}/{num_runs}")

    print(f"\n  Loss < {epsilon}:")
    if iterations_to_loss:
        print(f"    Success: {len(iterations_to_loss)}/{num_runs}")
        print(f"    Mean iterations: {np.mean(iterations_to_loss):.1f}")
        print(f"    Median iterations: {np.median(iterations_to_loss):.1f}")
        print(f"    Std: {np.std(iterations_to_loss):.1f}")
        print(f"    Min: {np.min(iterations_to_loss)}, Max: {np.max(iterations_to_loss)}")
    else:
        print(f"    Failed: {failed_loss}/{num_runs}")

    print(f"\n  Both conditions:")
    if iterations_to_both:
        print(f"    Success: {len(iterations_to_both)}/{num_runs}")
        print(f"    Mean iterations: {np.mean(iterations_to_both):.1f}")
        print(f"    Median iterations: {np.median(iterations_to_both):.1f}")
        print(f"    Std: {np.std(iterations_to_both):.1f}")
        print(f"    Min: {np.min(iterations_to_both)}, Max: {np.max(iterations_to_both)}")
    else:
        print(f"    Failed: {failed_both}/{num_runs}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if iterations_to_margin:
        axes[0].hist(iterations_to_margin, bins=20, edgecolor="black", alpha=0.7)
        axes[0].axvline(np.mean(iterations_to_margin), color="r", linestyle="--", label=f"Mean: {np.mean(iterations_to_margin):.1f}")
        axes[0].set_xlabel("Iterations to Margin Gap < δ")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"Margin Convergence (δ={delta})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    if iterations_to_loss:
        axes[1].hist(iterations_to_loss, bins=20, edgecolor="black", alpha=0.7)
        axes[1].axvline(np.mean(iterations_to_loss), color="r", linestyle="--", label=f"Mean: {np.mean(iterations_to_loss):.1f}")
        axes[1].set_xlabel("Iterations to Loss < ε")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title(f"Loss Convergence (ε={epsilon})")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    if iterations_to_both:
        axes[2].hist(iterations_to_both, bins=20, edgecolor="black", alpha=0.7)
        axes[2].axvline(np.mean(iterations_to_both), color="r", linestyle="--", label=f"Mean: {np.mean(iterations_to_both):.1f}")
        axes[2].set_xlabel("Iterations to Both Conditions")
        axes[2].set_ylabel("Frequency")
        axes[2].set_title("Both Conditions Met")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment_5c_margin_convergence.png", dpi=150)
    print("\nPlot saved to experiment_5c_margin_convergence.png")

    return {
        "iterations_to_margin": iterations_to_margin,
        "iterations_to_loss": iterations_to_loss,
        "iterations_to_both": iterations_to_both,
        "failed_margin": failed_margin,
        "failed_loss": failed_loss,
        "failed_both": failed_both,
    }


def experiment_5d_mixture(
    k: int = 20,
    num_iterations: int = 2000,
    learning_rate: float = 0.01,
    num_runs: int = 5,
    seed: int = 42,
):
    """
    Experiment 5d: Mixture of all settings (over-parameterized + non-symmetric + clustered).
    """
    print(f"\n=== Experiment 5d: Mixture of all settings ===")

    np.random.seed(seed)
    x_list = []
    y_list = []

    center1 = -0.8
    cluster1 = center1 + np.random.randn(5) * 0.1
    x_list.extend(cluster1)
    y_list.extend([-1.0] * 5)

    center2 = 1.2
    cluster2 = center2 + np.random.randn(5) * 0.1
    x_list.extend(cluster2)
    y_list.extend([1.0] * 5)

    x = np.array(x_list)
    y = np.array(y_list)

    optimal_margin = (
        np.min([abs(xi - xj) for i, xi in enumerate(x) for j, xj in enumerate(x) if y[i] != y[j]]) / 2
    )

    print(f"  Over-parameterized: k={k}")
    print(f"  Non-symmetric clustered data: {len(x)} points")
    print(f"  Optimal margin (approx): {optimal_margin:.6f}")

    margins_all = []
    margin_gaps_all = []
    losses_all = []

    for run in range(num_runs):
        params = initialize_network(k, init_type="random", seed=seed + run)
        params_history, losses, _, margins, margin_gaps = train_gd(
            params,
            x,
            y,
            learning_rate,
            num_iterations,
            track_boundaries=False,
            track_margin=True,
            optimal_margin=optimal_margin,
        )
        margins_all.append(margins)
        margin_gaps_all.append(margin_gaps)
        losses_all.append(losses)

    final_margins = [m[-1] if len(m) > 0 else 0 for m in margins_all]
    final_gaps = [g[-1] if len(g) > 0 else optimal_margin for g in margin_gaps_all]
    print(f"\nResults:")
    print(f"  Final margin: {np.mean(final_margins):.6f} ± {np.std(final_margins):.6f}")
    print(f"  Final margin gap: {np.mean(final_gaps):.6f} ± {np.std(final_gaps):.6f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    avg_margins = np.mean(margins_all, axis=0)
    std_margins = np.std(margins_all, axis=0)
    axes[0, 0].plot(avg_margins, label="Mean")
    axes[0, 0].fill_between(
        range(len(avg_margins)), avg_margins - std_margins, avg_margins + std_margins, alpha=0.3
    )
    axes[0, 0].axhline(optimal_margin, color="r", linestyle="--", label="Optimal")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Margin")
    axes[0, 0].set_title("Margin Convergence (Mixture)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    avg_gaps = np.mean(margin_gaps_all, axis=0)
    std_gaps = np.std(margin_gaps_all, axis=0)
    axes[0, 1].plot(avg_gaps, label="Mean")
    axes[0, 1].fill_between(range(len(avg_gaps)), avg_gaps - std_gaps, avg_gaps + std_gaps, alpha=0.3)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Margin Gap")
    axes[0, 1].set_title("Margin Gap Convergence")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    avg_losses = np.mean(losses_all, axis=0)
    std_losses = np.std(losses_all, axis=0)
    axes[1, 0].plot(avg_losses, label="Mean")
    axes[1, 0].fill_between(range(len(avg_losses)), avg_losses - std_losses, avg_losses + std_losses, alpha=0.3)
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("Loss Convergence")
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    x_plot = np.linspace(x.min() - 0.5, x.max() + 0.5, 1000)
    params_final = initialize_network(k, init_type="random", seed=seed)
    params_history_final, _, _, _, _ = train_gd(
        params_final, x, y, learning_rate, num_iterations, track_boundaries=False, track_margin=False
    )
    y_plot = network_forward(params_history_final[-1], x_plot)
    axes[1, 1].plot(x_plot, y_plot, alpha=0.7, label="Network output")
    axes[1, 1].scatter(
        x, [0] * len(x), c=["red" if yi < 0 else "blue" for yi in y], s=100, alpha=0.7, label="Data points"
    )
    axes[1, 1].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("Phi(x)")
    axes[1, 1].set_title("Final Network on Clustered Data")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("experiment_5d_mixture.png", dpi=150)
    print("\nPlot saved to experiment_5d_mixture.png")

    return {"margins": margins_all, "margin_gaps": margin_gaps_all, "losses": losses_all, "x": x, "y": y}


def example_initialization_options():
    """
    Small demo printing several initialization schemes.
    """
    print("\n" + "=" * 60)
    print("Example: Initialization Options")
    print("=" * 60)

    k = 4

    print("\n1. Binary initialization (w_j = +1/-1):")
    params1 = initialize_network(k=k, init_type="binary", v_binary=True, seed=42)
    print_initial_params(params1, "Binary w_j initialization")

    print("\n2. Explicit w_binary list:")
    params2 = initialize_network(k=k, w_binary=[1, -1, 1, -1], v_binary=True, seed=42)
    print_initial_params(params2, "Explicit w_binary=[1, -1, 1, -1]")

    print("\n3. Custom explicit arrays:")
    w_custom = np.array([1.0, -1.0, 0.5, -0.5])
    b_custom = np.array([0.1, -0.1, 0.2, -0.2])
    v_custom = np.array([1.0, -1.0, 1.0, -1.0])
    params3 = initialize_network(k=k, w_init=w_custom, b_init=b_custom, v_init=v_custom)
    print_initial_params(params3, "Custom explicit arrays")

    print("\n4. Thesis initialization (k=2):")
    params4 = initialize_network(k=2, init_type="thesis", w1_init=1.0, b1_init=1.0, M=10.0)
    print_initial_params(params4, "Thesis initialization")

    print("\n5. Random initialization:")
    params5 = initialize_network(k=k, init_type="random", seed=42)
    print_initial_params(params5, "Random initialization")

    print("\n" + "=" * 60)

# def experiment_5f_hit_linear_condition_with_low_loss(
#     num_runs: int = 10000,
#     max_iterations: int = 10_000_000,
#     learning_rate: float = 0.05,
#     tol: float = 1e-3,
#     loss_threshold: float = 0.5,
#     seed: int = 42,
# ):
#     """
#     Run num_runs times with random init (standard normal w,b), k=2, and fixed v=[1,-1].
#     Run GD until BOTH:
#         |w1 + b1 + w2 - b2| < tol
#         loss < loss_threshold
#     Record hitting iteration for each run and plot histogram.

#     Extra rule: if after 10,000 iterations we still don't have loss < loss_threshold, abort this run.
#     """

#     rng = np.random.default_rng(seed)

#     # Use your standard dataset
#     x, y = create_dataset(symmetric=True)

#     hit_times = np.full(num_runs, -1, dtype=int)

#     for r in range(num_runs):
#         # random standard normal init for w,b
#         # w1_0 = float(rng.standard_normal())
#         w1_0 = float(rng.normal(loc=0.0, scale=np.sqrt(2)))
#         b1_0 = 0
#         # b1_0 = float(rng.standard_normal())
#         # w2_0 = float(rng.standard_normal())
#         w2_0 = float(rng.normal(loc=0.0, scale=np.sqrt(2)))
#         b2_0 = 0
#         # b2_0 = float(rng.standard_normal())

#         params = initialize_network(
#             k=2,
#             init_type="thesis",
#             seed=None,
#             w1_init=w1_0,
#             b1_init=b1_0,
#             w2_init=w2_0,
#             b2_init=b2_0,
#         )

#         # Force v1=1, v2=-1 and keep it fixed
#         params.v = np.array([1.0, -1.0], dtype=float)

#         t = 0
#         while t <= max_iterations:
#             # Print every 100,000 iterations (including t=0)
#             if t % 100000 == 0:
#                 w1 = float(params.w[0])
#                 b1 = float(params.b[0])
#                 w2 = float(params.w[1])
#                 b2 = float(params.b[1])

#                 print(f"Run {r} | t={t}")
#                 print(f"  w1={w1:.6f}, b1={b1:.6f}")
#                 print(f"  w2={w2:.6f}, b2={b2:.6f}")
#                 print(f"  w1+b1+w2-b2 = {w1+b1+w2-b2:.6e}")
#                 print("-" * 40)

#             preds = network_forward(params, x)
#             loss_t = float(exponential_loss(y, preds))

#             # NEW: if after 10k iterations loss still not below threshold -> abort run
#             if t == 10000 and not (loss_t < loss_threshold):
#                 print(f"Run {r}: aborting (loss={loss_t:.6f} not < {loss_threshold}) at t=10000")
#                 break

#             # Only consider the hit condition when loss is below threshold
#             if loss_t < loss_threshold:
#                 w1 = float(params.w[0])
#                 b1 = float(params.b[0])
#                 w2 = float(params.w[1])
#                 b2 = float(params.b[1])

#                 expr = abs(w1 + b1 + w2 - b2)
#                 if expr < tol:
#                     hit_times[r] = t
#                     break

#             # GD step
#             params, _ = gradient_descent_step(params, x, y, learning_rate=learning_rate)
#             # re-freeze v after update
#             params.v = np.array([1.0, -1.0], dtype=float)
#             t += 1
#         print("Stopping because:", 
#             "hit condition" if hit_times[r] != -1
#             else "loss-abort" if t == 10000
#             else "max-iterations")

#         if hit_times[r] == -1:
#             print(f"Run {r}: did NOT hit within limits (t={t}, max_iterations={max_iterations})")
#         else:
#             print(f"Run {r}: hit at t={hit_times[r]}")

#     # Summary + histogram
#     successful = hit_times[hit_times >= 0]

#     print("\n=== Experiment 5f summary ===")
#     print(f"num_runs={num_runs}, max_iterations={max_iterations}, lr={learning_rate}")
#     print(f"condition: |w1+b1+w2-b2| < {tol} AND loss < {loss_threshold}")
#     print(f"successful runs: {len(successful)}/{num_runs}")

#     if len(successful) > 0:
#         print(f"mean hit time: {successful.mean():.2f}")
#         print(f"median hit time: {np.median(successful):.2f}")
#         print(f"min hit time: {successful.min()}")
#         print(f"max hit time: {successful.max()}")

#         plt.figure(figsize=(10, 6))
#         plt.hist(successful, bins=40)
#         plt.title(rf"Hit time for |w1+b1+w2-b2|<{tol} (only when loss<{loss_threshold})")
#         plt.xlabel("Iterations")
#         plt.ylabel("Count")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig("experiment_5f_hit_time_hist.png", dpi=200)
#         plt.close()

#         print("Saved histogram: experiment_5f_hit_time_hist.png")
#     else:
#         print("No successful hits; no histogram saved.")


def experiment_5f_hit_linear_condition_with_low_loss(
    num_runs: int = 10000,
    max_iterations: int = 10_000_000,
    learning_rate: float = 0.05,
    tol: float = 1e-3,
    loss_threshold: float = 0.5,
    seed: int = 42,
):
    """
    Runs num_runs times with random init:
        w ~ N(0,2), b=0, k=2, v=[1,-1] fixed.

    Stop when:
        |w1 + b1 + w2 - b2| < tol
        AND loss < loss_threshold

    Abort rule:
        If at t=10000 loss still not < loss_threshold → abort.

    Writes:
        experiment_5f_runs.csv
        experiment_5f_summary.txt
        experiment_5f_hit_time_hist.png
        experiment_5f_metric_hist.png
        experiment_5f_hit_time_hist.csv
        experiment_5f_metric_hist.csv
    """

    rng = np.random.default_rng(seed)
    x, y = create_dataset(symmetric=True)

    hit_times = np.full(num_runs, -1, dtype=int)
    metric_values = []

    count_hit = 0
    count_loss_abort = 0
    count_max_iterations = 0

    runs_csv_path = "experiment_5f_runs.csv"
    summary_txt_path = "experiment_5f_summary.txt"
    hist1_csv_path = "experiment_5f_hit_time_hist.csv"
    hist2_csv_path = "experiment_5f_metric_hist.csv"

    with open(runs_csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "run", "stop_reason", "t_last", "t_hit",
                "w1_0", "b1_0", "w2_0", "b2_0",
                "w1_T", "b1_T", "w2_T", "b2_T",
                "loss_last", "expr_last",
                "metric_min"
            ],
        )
        writer.writeheader()

        for r in range(num_runs):
            print(f"Run: {r}")

            # Initialization
            w1_0 = float(rng.normal(0.0, np.sqrt(2)))
            b1_0 = 0.0
            w2_0 = float(rng.normal(0.0, np.sqrt(2)))
            b2_0 = 0.0

            params = initialize_network(
                k=2,
                init_type="thesis",
                seed=None,
                w1_init=w1_0,
                b1_init=b1_0,
                w2_init=w2_0,
                b2_init=b2_0,
            )

            params.v = np.array([1.0, -1.0], dtype=float)

            t = 0
            stop_reason = None

            while t <= max_iterations:

                preds = network_forward(params, x)
                loss_t = float(exponential_loss(y, preds))

                w1 = float(params.w[0])
                b1 = float(params.b[0])
                w2 = float(params.w[1])
                b2 = float(params.b[1])

                expr_t = abs(w1 + b1 + w2 - b2)

                # Abort rule
                if t == 10000 and not (loss_t < loss_threshold):
                    stop_reason = "loss-abort"
                    count_loss_abort += 1
                    break

                # Hit condition
                if loss_t < loss_threshold and expr_t < tol:
                    hit_times[r] = t
                    stop_reason = "hit condition"
                    count_hit += 1

                    metric_min = min(abs(b2 - b1), abs(w1_0 + w2_0) / 2.0)
                    metric_values.append(metric_min)
                    break

                if t == max_iterations:
                    stop_reason = "max-iterations"
                    count_max_iterations += 1
                    break

                params, _ = gradient_descent_step(
                    params, x, y, learning_rate=learning_rate
                )
                params.v = np.array([1.0, -1.0], dtype=float)

                t += 1

            # Final parameters
            w1_T = float(params.w[0])
            b1_T = float(params.b[0])
            w2_T = float(params.w[1])
            b2_T = float(params.b[1])

            metric_val = ""
            if stop_reason == "hit condition":
                metric_val = metric_values[-1]

            writer.writerow({
                "run": r,
                "stop_reason": stop_reason,
                "t_last": t,
                "t_hit": hit_times[r],
                "w1_0": w1_0,
                "b1_0": b1_0,
                "w2_0": w2_0,
                "b2_0": b2_0,
                "w1_T": w1_T,
                "b1_T": b1_T,
                "w2_T": w2_T,
                "b2_T": b2_T,
                "loss_last": loss_t,
                "expr_last": expr_t,
                "metric_min": metric_val,
            })

    # Safety check
    assert count_hit + count_loss_abort + count_max_iterations == num_runs

    # -------------------------
    # Histogram 1: Hit times
    # -------------------------
    successful_hits = hit_times[hit_times >= 0]

    hist1_counts = None
    hist1_edges = None
    if len(successful_hits) > 0:
        hist1_counts, hist1_edges = np.histogram(successful_hits, bins=40)

        plt.figure(figsize=(8, 5))
        plt.hist(successful_hits, bins=40)
        plt.title("Hit time histogram")
        plt.xlabel("Iterations")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("experiment_5f_hit_time_hist.png", dpi=200)
        plt.close()

        with open(hist1_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bin_left", "bin_right", "count"])
            for i in range(len(hist1_counts)):
                writer.writerow([
                    hist1_edges[i],
                    hist1_edges[i+1],
                    hist1_counts[i]
                ])

    # -------------------------
    # Histogram 2: Metric
    # -------------------------
    metric_arr = np.array(metric_values)

    hist2_counts = None
    hist2_edges = None
    if len(metric_arr) > 0:
        hist2_counts, hist2_edges = np.histogram(metric_arr, bins=40)

        plt.figure(figsize=(8, 5))
        plt.hist(metric_arr, bins=40)
        plt.title("Histogram of min(|b2-b1|, |w1^0+w2^0|/2)")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("experiment_5f_metric_hist.png", dpi=200)
        plt.close()

        with open(hist2_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bin_left", "bin_right", "count"])
            for i in range(len(hist2_counts)):
                writer.writerow([
                    hist2_edges[i],
                    hist2_edges[i+1],
                    hist2_counts[i]
                ])

    # -------------------------
    # Summary file
    # -------------------------
    with open(summary_txt_path, "w") as f:
        f.write("=== Experiment 5f summary ===\n")
        f.write(f"num_runs={num_runs}\n")
        f.write(f"max_iterations={max_iterations}\n")
        f.write(f"learning_rate={learning_rate}\n")
        f.write(f"tol={tol}, loss_threshold={loss_threshold}\n\n")

        f.write(f"Hit condition: {count_hit}/{num_runs}\n")
        f.write(f"Failed (loss-abort): {count_loss_abort}/{num_runs}\n")
        f.write(f"Failed (max-iterations): {count_max_iterations}/{num_runs}\n\n")

        if hist1_counts is not None:
            f.write("Hit-time histogram edges:\n")
            f.write(",".join(map(str, hist1_edges)) + "\n")
            f.write("Hit-time histogram counts:\n")
            f.write(",".join(map(str, hist1_counts)) + "\n\n")

        if hist2_counts is not None:
            f.write("Metric histogram edges:\n")
            f.write(",".join(map(str, hist2_edges)) + "\n")
            f.write("Metric histogram counts:\n")
            f.write(",".join(map(str, hist2_counts)) + "\n")

    print("\n=== Experiment 5f summary ===")
    print(f"Hit condition: {count_hit}/{num_runs}")
    print(f"Failed (loss-abort): {count_loss_abort}/{num_runs}")
    print(f"Failed (max-iterations): {count_max_iterations}/{num_runs}")
    print("Files written:")
    print(" - experiment_5f_runs.csv")
    print(" - experiment_5f_summary.txt")
    if hist1_counts is not None:
        print(" - experiment_5f_hit_time_hist.png")
        print(" - experiment_5f_hit_time_hist.csv")
    if hist2_counts is not None:
        print(" - experiment_5f_metric_hist.png")
        print(" - experiment_5f_metric_hist.csv")

# # 

def collapse_2relu_1d_with_3_constraints_signs(
    params_rich: "NetworkParams",
    x_fit: np.ndarray,              # [-1, +1]
    y_fit: np.ndarray,              # [-1, +1] (sanity)
    x0_search_min: float = -2.0,
    x0_search_max: float =  2.0,
    x0_search_num: int = 4001,
    act_eps: float = 1e-9,
    d: int = 10,
    h1_grid_num: int = 2000,
    h1_min: float = -0.999,
    h1_max: float = -1e-4,
) -> Tuple["NetworkParams", Dict]:
    """
    Build f(t)=ReLU(w1 t + b1) - ReLU(w2 t + b2) with:
      - f(+1)=g(+1), f(-1)=g(-1), f(t0)=0
      - single-point-only: neuron1 active only at +1, neuron2 active only at -1
      - sign constraints: w1>0, b1>0, w2<0, b2>0
    """

    x_fit = np.asarray(x_fit, dtype=float).reshape(-1)
    if x_fit.shape[0] != 2:
        raise ValueError("Expected exactly 2 fit points (typically [-1,+1]).")

    def g(tt: np.ndarray) -> np.ndarray:
        tt = np.asarray(tt, dtype=float).reshape(-1)
        X_axis = np.zeros((tt.shape[0], d), dtype=float)
        X_axis[:, 0] = tt
        return network_forward(params_rich, X_axis).astype(float).reshape(-1)

    # teacher root
    t0 = find_root_on_grid_1d_fn(g, x_min=x0_search_min, x_max=x0_search_max, num=x0_search_num)

    gp = float(g(np.array([+1.0]))[0])  # >0
    gm = float(g(np.array([-1.0]))[0])  # <0
    if not (gp > 0 and gm < 0):
        raise RuntimeError(f"Need g(+1)>0 and g(-1)<0. Got g(+1)={gp:.6f}, g(-1)={gm:.6f}.")

    B = -gm  # positive

    # We want h1 < 0 < h2, and also h1 < t0 < h2 for the f(t0)=0 derivation
    # We'll search h1 in (-1,0) and compute h2 analytically.
    best = None
    best_info = None

    h1_grid = np.linspace(h1_min, h1_max, h1_grid_num)
    for h1 in h1_grid:
        # ensure single-point-only for neuron1: inactive at -1, active at +1
        # with hinge h1, neuron1 pre1(t)=w1(t-h1), so inactive at -1 iff -1<=h1
        if h1 < -1.0 + 1e-12:
            continue

        # C = gp * (t0-h1)/(1-h1)
        denom1 = (1.0 - h1)
        if denom1 <= 1e-12:
            continue
        C = gp * (t0 - h1) / denom1

        # Need B - C > 0 to make h2 finite and typically positive
        if (B - C) <= 1e-12:
            continue

        h2 = (B * t0 + C) / (B - C)

        # enforce hinge location for neuron2 to get b2>0 and inactive at +1
        # want 0<h2<1 and also t0<h2 (for both active at t0)
        if not (h2 > 1e-6 and h2 < 1.0 - 1e-6):
            continue
        if not (h1 < t0 < h2):
            continue

        # Build slopes from endpoint constraints
        w1 = gp / (1.0 - h1)                # >0
        b1 = -w1 * h1                       # >0 because h1<0
        w2 = -B / (1.0 + h2)                # <0
        b2 = -w2 * h2                       # >0 because -w2>0 and h2>0

        # strict activity checks
        pre1_m1 = w1 * (-1.0) + b1
        pre1_p1 = w1 * (+1.0) + b1
        pre2_m1 = w2 * (-1.0) + b2
        pre2_p1 = w2 * (+1.0) + b2

        if not (pre1_p1 > act_eps and pre1_m1 <= act_eps):
            continue
        if not (pre2_m1 > act_eps and pre2_p1 <= act_eps):
            continue

        # Verify constraints numerically
        params_col = NetworkParams(
            w=np.array([[w1], [w2]], dtype=float),
            b=np.array([b1, b2], dtype=float),
            v=np.array([+1.0, -1.0], dtype=float),
        )

        f_p1 = float(network_forward(params_col, np.array([+1.0]).reshape(-1, 1))[0])
        f_m1 = float(network_forward(params_col, np.array([-1.0]).reshape(-1, 1))[0])
        f_t0 = float(network_forward(params_col, np.array([t0]).reshape(-1, 1))[0])

        err = abs(f_p1 - gp) + abs(f_m1 - gm) + abs(f_t0 - 0.0)

        best = err
        best_info = {
            "t0_teacher": float(t0),
            "g_plus1": float(gp),
            "g_minus1": float(gm),
            "h1": float(h1),
            "h2": float(h2),
            "w1": float(w1),
            "b1": float(b1),
            "w2": float(w2),
            "b2": float(b2),
            "checks": {
                "f(+1)": f_p1,
                "f(-1)": f_m1,
                "f(t0)": f_t0,
                "pre1(-1)": float(pre1_m1),
                "pre1(+1)": float(pre1_p1),
                "pre2(-1)": float(pre2_m1),
                "pre2(+1)": float(pre2_p1),
            },
            "constraint_error": float(err),
        }
        return params_col, best_info

    raise RuntimeError(
        "Failed to find hinges (h1<0, h2>0) that satisfy all constraints. "
        "Try widening search ranges or increasing h1_grid_num."
    )

def collapse_to_2relu_1d_with_margin_match(
    params_rich: NetworkParams,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    distill_iters: int = 20000,
    distill_lr: float = 0.01,
):
    """
    Collapse/Distill a rich 1D k-ReLU network into:
        f_simple(x) = ReLU(w_pos x + b_pos) - ReLU(w_neg x + b_neg)
    by MSE distillation on (x_fit), and then OPTIONAL margin-match rescaling:

        m_rich = min_i y_i f_rich(x_i)
        m_col  = min_i y_i f_col(x_i)

    If m_col > 0, rescale BOTH (w,b) of both ReLUs by alpha = m_rich/m_col
    so that margin matches exactly under the same margin definition.

    Returns:
        params_collapse_1d, info_dict
    """

    f_target = network_forward(params_rich, x_fit).astype(float)
    invN = 1.0 / float(x_fit.shape[0])

    # warm-start: average by sign of v (fallback to overall mean)
    if np.any(params_rich.v > 0):
        w_pos = float(np.mean(params_rich.w[params_rich.v > 0]))
        b_pos = float(np.mean(params_rich.b[params_rich.v > 0]))
    else:
        w_pos = float(np.mean(params_rich.w))
        b_pos = float(np.mean(params_rich.b))

    if np.any(params_rich.v < 0):
        w_neg = float(np.mean(params_rich.w[params_rich.v < 0]))
        b_neg = float(np.mean(params_rich.b[params_rich.v < 0]))
    else:
        w_neg = float(np.mean(params_rich.w))
        b_neg = float(np.mean(params_rich.b))

    best = (w_pos, b_pos, w_neg, b_neg)
    best_mse = float("inf")

    for _ in range(distill_iters):
        z1 = w_pos * x_fit + b_pos
        z2 = w_neg * x_fit + b_neg
        a1 = (z1 > 0).astype(float)
        a2 = (z2 > 0).astype(float)

        f_simple = np.maximum(0.0, z1) - np.maximum(0.0, z2)
        r = f_simple - f_target

        grad_w1 = invN * np.sum(r * (a1 * x_fit))
        grad_b1 = invN * np.sum(r * a1)
        grad_w2 = invN * np.sum(r * (-a2 * x_fit))
        grad_b2 = invN * np.sum(r * (-a2))

        w_pos -= distill_lr * grad_w1
        b_pos -= distill_lr * grad_b1
        w_neg -= distill_lr * grad_w2
        b_neg -= distill_lr * grad_b2

        mse = 0.5 * invN * float(np.sum(r ** 2))
        if mse < best_mse:
            best_mse = mse
            best = (w_pos, b_pos, w_neg, b_neg)

    w_pos, b_pos, w_neg, b_neg = best

    # build params
    params_collapse_1d = NetworkParams(
        w=np.array([w_pos, w_neg], dtype=float),
        b=np.array([b_pos, b_neg], dtype=float),
        v=np.array([1.0, -1.0], dtype=float),
    )

    # margin match (under min y*f) — works for the "easy case" and later for n=1000 too
    f_rich = network_forward(params_rich, x_fit).astype(float)
    f_col = network_forward(params_collapse_1d, x_fit).astype(float)

    m_rich = float(np.min(y_fit * f_rich))
    m_col_before = float(np.min(y_fit * f_col))

    alpha = 1.0
    eps = 1e-12
    did_rescale = False
    if m_col_before > eps:
        alpha = m_rich / m_col_before
        # keep alpha positive; if m_rich is also positive, alpha will be positive
        params_collapse_1d = NetworkParams(
            w=params_collapse_1d.w * alpha,
            b=params_collapse_1d.b * alpha,
            v=params_collapse_1d.v.copy(),
        )
        did_rescale = True

    f_col_after = network_forward(params_collapse_1d, x_fit).astype(float)
    m_col_after = float(np.min(y_fit * f_col_after))

    info = {
        "best_mse": float(best_mse),
        "m_rich": float(m_rich),
        "m_col_before": float(m_col_before),
        "alpha": float(alpha),
        "did_rescale": bool(did_rescale),
        "m_col_after": float(m_col_after),
        "w_pos": float(params_collapse_1d.w[0]),
        "b_pos": float(params_collapse_1d.b[0]),
        "w_neg": float(params_collapse_1d.w[1]),
        "b_neg": float(params_collapse_1d.b[1]),
    }
    return params_collapse_1d, info

# def collapse_2relu_1d_with_3_constraints_enum(
#     params_rich: NetworkParams,
#     x_fit: np.ndarray,
#     y_fit: np.ndarray,
#     wpos_grid: np.ndarray = None,
#     x0_search_min: float = -2.0,
#     x0_search_max: float = 2.0,
#     x0_search_num: int = 5000,
#     act_eps: float = 1e-9,
#     d: Optional[int] = None,
# ):
#     """
#     Collapse to a 2-ReLU *1D* student model with EXACT constraints:

#       1) same axis crossing: f_col(t0)=0 where t0 is a root of the teacher 1D function
#       2) f_col(+1)=teacher(+1)
#       3) f_col(-1)=teacher(-1)

#     Here the teacher 1D function is defined as:
#       - If params_rich is truly 1D (w has shape (k,1) or (k,)): teacher(t)=f_rich(t)
#       - If params_rich is d-dimensional (w has shape (k,d) with d>1): teacher(t)=f_rich((t,0,...,0)),
#         i.e. restriction to the y-axis. In this case you must pass d (input dimension).

#     Remaining DOF: choose w_pos from a grid, solve a linear system for (b_pos, w_neg, b_neg)
#     under enumerated activation patterns on {t0, +1, -1}. Pick the solution with minimal MSE on x_fit.

#     Returns:
#       (params_collapse_1d, info_dict)
#     """

#     # ---- define the teacher 1D function ----
#     def _teacher_forward(t_1d: np.ndarray) -> np.ndarray:
#         t_1d = np.asarray(t_1d, dtype=float).reshape(-1)

#         # Case A: params_rich is already 1D but stored as (k,1)
#         if getattr(params_rich, "w", None) is not None and np.ndim(params_rich.w) == 2 and params_rich.w.shape[1] == 1:
#             X = t_1d.reshape(-1, 1)
#             return network_forward(params_rich, X).astype(float).reshape(-1)

#         # Case B: params_rich is dD (k,d), d>1 -> restrict to y-axis
#         if getattr(params_rich, "w", None) is not None and np.ndim(params_rich.w) == 2 and params_rich.w.shape[1] > 1:
#             if d is None:
#                 raise TypeError("collapse_2relu_1d_with_3_constraints_enum: params_rich looks d-dimensional, but d was not provided.")
#             X_axis = points_on_y_axis(t_1d, d)
#             return network_forward(params_rich, X_axis).astype(float).reshape(-1)

#         # Case C: legacy 1D params where w is (k,)
#         # (kept for backward compatibility if you still have other 1D experiments)
#         return network_forward(params_rich, t_1d).astype(float).reshape(-1)

#     # student forward (always 1D-in-dim-1: input shape (n,1))
#     def _student_forward(params_1d: NetworkParams, t_1d: np.ndarray) -> np.ndarray:
#         t_1d = np.asarray(t_1d, dtype=float).reshape(-1)
#         X = t_1d.reshape(-1, 1)
#         return network_forward(params_1d, X).astype(float).reshape(-1)

#     # ---- teacher targets on fit points ----
#     x_fit = np.asarray(x_fit, dtype=float).reshape(-1)
#     f_target = _teacher_forward(x_fit)
#     t_p1 = float(_teacher_forward(np.array([1.0]))[0])
#     t_m1 = float(_teacher_forward(np.array([-1.0]))[0])

#     # ---- find t0: a root of teacher(t) in range, choose closest to 0 ----
#     xg = np.linspace(x0_search_min, x0_search_max, x0_search_num)
#     fg = _teacher_forward(xg)
#     s = np.sign(fg)
#     s[s == 0] = 1.0
#     idx = np.where(s[:-1] * s[1:] < 0)[0]
#     if len(idx) == 0:
#         raise RuntimeError("No root found in grid range. Try widening [x0_search_min,x0_search_max].")

#     roots = []
#     for i in idx:
#         x1, x2 = float(xg[i]), float(xg[i + 1])
#         f1, f2 = float(fg[i]), float(fg[i + 1])
#         xr = x1 - f1 * (x2 - x1) / (f2 - f1 + 1e-18)
#         roots.append(xr)
#     roots = np.array(roots, dtype=float)
#     t0 = float(roots[np.argmin(np.abs(roots))])

#     # ---- grid for w_pos ----
#     if wpos_grid is None:
#         # positive slopes grid (same spirit as before)
#         wpos_grid = np.concatenate([
#             np.linspace(0.1, 5.0, 200),
#             np.linspace(5.0, 50.0, 200),
#         ]).astype(float)

#     # ---- enumerate activation patterns on {t0, +1, -1} ----
#     # For each ReLU, we need to know whether it is active at each constraint point.
#     # We'll enumerate patterns for (+) and (-) units.
#     pts = np.array([t0, 1.0, -1.0], dtype=float)  # constraint points
#     teacher_vals = np.array([0.0, t_p1, t_m1], dtype=float)

#     best = None
#     best_info = None

#     # Activation state vectors in {0,1}^3 for each unit
#     patterns = np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).T.reshape(-1, 3)

#     # Student model: f_col(t) = ReLU(wp t + bp) - ReLU(wn t + bn)
#     v_col = np.array([+1.0, -1.0], dtype=float)

#     for wp in wpos_grid:
#         for ap in patterns:
#             for an in patterns:
#                 # Unknowns: bp, wn, bn
#                 # Constraints at pts: ReLU(wp*pt + bp) - ReLU(wn*pt + bn) = teacher_vals
#                 # Under fixed activation patterns, each ReLU is either linear or 0 at each pt.
#                 # Build linear system A u = rhs where u=[bp, wn, bn].

#                 A = np.zeros((3, 3), dtype=float)
#                 rhs = teacher_vals.copy()

#                 for r, t in enumerate(pts):
#                     # positive unit contribution: ap[r]*(wp*t + bp)
#                     # -> coefficient on bp is ap[r]
#                     A[r, 0] = ap[r]

#                     # negative unit contribution: - an[r]*(wn*t + bn)
#                     # -> coefficient on wn is -an[r]*t, on bn is -an[r]
#                     A[r, 1] = -an[r] * t
#                     A[r, 2] = -an[r]

#                     # move known term ap[r]*(wp*t) to rhs
#                     rhs[r] -= ap[r] * (wp * t)

#                 # Solve if well-conditioned
#                 try:
#                     u = np.linalg.solve(A, rhs)
#                 except np.linalg.LinAlgError:
#                     continue

#                 bp, wn, bn = float(u[0]), float(u[1]), float(u[2])

#                 # Check pattern consistency (activation depends on sign of pre-activation)
#                 pre_p = wp * pts + bp
#                 pre_n = wn * pts + bn
#                 ok_p = np.all((pre_p > act_eps).astype(int) == ap)
#                 ok_n = np.all((pre_n > act_eps).astype(int) == an)
#                 if not (ok_p and ok_n):
#                     continue

#                 # Build student params in dim-1 form: w shape (2,1)
#                 w_col = np.array([[wp], [wn]], dtype=float)   # (2,1)
#                 b_col = np.array([bp, bn], dtype=float)       # (2,)
#                 params_col = NetworkParams(w=w_col, b=b_col, v=v_col)

#                 # Fit quality on x_fit (1D)
#                 pred = _student_forward(params_col, x_fit)
#                 mse = float(np.mean((pred - f_target) ** 2))

#                 if (best is None) or (mse < best):
#                     best = mse
#                     best_info = {
#                         "t0": t0,
#                         "w_pos": wp,
#                         "b_pos": bp,
#                         "w_neg": wn,
#                         "b_neg": bn,
#                         "mse_fit": mse,
#                         "pattern_pos": ap.copy(),
#                         "pattern_neg": an.copy(),
#                         "teacher_t_p1": t_p1,
#                         "teacher_t_m1": t_m1,
#                     }
#                     best_params = params_col

#     if best_info is None:
#         raise RuntimeError("No feasible 2-ReLU collapse solution found (patterns/grid too restrictive).")

#     return best_params, best_info

# def signed_train_margin(params: "NetworkParams", x_train: np.ndarray, y_train: np.ndarray) -> float:
#     """
#     Signed margin on a finite dataset:
#         m = min_i y_i f(x_i)
#     This is the SAME definition used by your margin-match rescale in collapse_to_2relu_1d_with_margin_match.
#     """
#     x_train = np.asarray(x_train, dtype=float).reshape(-1)
#     y_train = np.asarray(y_train, dtype=float).reshape(-1)
#     f = network_forward(params, x_train).astype(float)
#     return float(np.min(y_train * f))


# def count_single_point_only_neurons(
#     params: "NetworkParams",
#     x_1d: np.ndarray,
#     act_eps: float = 1e-9,
# ):
#     """
#     For exactly 2 training points x_1d = [-1, +1] (or any two scalars),
#     count how many neurons are:
#       - dead: active on 0 points
#       - single: active on exactly 1 point
#       - both: active on both points
#     where "active" means (w*x + b) > act_eps.
#     """
#     x_1d = np.asarray(x_1d, dtype=float).reshape(-1)
#     if x_1d.shape[0] != 2:
#         raise ValueError("This helper expects exactly 2 training points.")

#     z = x_1d[:, None] * params.w[None, :] + params.b[None, :]  # (2,k)
#     a = (z > act_eps).astype(int)                               # (2,k)
#     s = a.sum(axis=0)                                           # (k,) in {0,1,2}

#     num_dead = int(np.sum(s == 0))
#     num_single = int(np.sum(s == 1))
#     num_both = int(np.sum(s == 2))
#     return num_dead, num_single, num_both


def collapse_2relu_1d_with_3_constraints_enum(
    params_rich: "NetworkParams",
    x_fit: np.ndarray,              # expected [-1, +1]
    y_fit: np.ndarray,              # expected [-1, +1] (used only for sanity)
    x0_search_min: float = -2.0,
    x0_search_max: float =  2.0,
    x0_search_num: int = 4001,
    act_eps: float = 1e-9,
    d: int = 10,
) -> Tuple["NetworkParams", Dict]:
    """
    Collapse a rich dD network to a 1D 2-ReLU network on the y-axis.

    We construct:
        f(t) = ReLU(w1 t + b1) - ReLU(w2 t + b2)
    with v=[+1,-1], such that:
      - f(+1) = g(+1)
      - f(-1) = g(-1)
      - f(t0) = 0   (t0 is a root of g(t))
      - neuron 0 is active only at +1 (inactive at -1)
      - neuron 1 is active only at -1 (inactive at +1)

    Implementation:
      Choose both hinges at t0:
        w1 t0 + b1 = 0,  w2 t0 + b2 = 0
      Then solve w1,b1 from f(+1)=g(+1), and w2,b2 from f(-1)=g(-1).
    """
    x_fit = np.asarray(x_fit, dtype=float).reshape(-1)
    y_fit = np.asarray(y_fit, dtype=float).reshape(-1)
    if x_fit.shape[0] != 2:
        raise ValueError("This collapse expects exactly 2 fit points (typically [-1,+1]).")

    # Teacher on y-axis: g(t)=f_rich((t,0,...,0))
    def g(tt: np.ndarray) -> np.ndarray:
        tt = np.asarray(tt, dtype=float).reshape(-1)
        X_axis = np.zeros((tt.shape[0], d), dtype=float)
        X_axis[:, 0] = tt
        return network_forward(params_rich, X_axis).astype(float).reshape(-1)

    # Find teacher root t0 in [x0_search_min, x0_search_max]
    t0 = find_root_on_grid_1d_fn(g, x_min=x0_search_min, x_max=x0_search_max, num=x0_search_num)

    # Values at +/-1
    gp = float(g(np.array([+1.0]))[0])  # g(+1)
    gm = float(g(np.array([-1.0]))[0])  # g(-1)

    # We need gp>0 and gm<0 to match the desired classification orientation.
    # If signs are flipped, we can still build but the network would be "inverted".
    # Here we enforce the standard orientation; otherwise we fail loudly (better than silent wrong collapse).
    if not (gp > 0 and gm < 0):
        raise RuntimeError(
            f"Collapse expects g(+1)>0 and g(-1)<0. Got g(+1)={gp:.6f}, g(-1)={gm:.6f}."
        )

    # Enforce a small positive lower bound so the active neuron is strictly active (>act_eps)
    a1 = max(gp, 10.0 * act_eps)      # desired ReLU output at +1 for neuron 0
    a2 = max(-gm, 10.0 * act_eps)     # desired ReLU output at -1 for neuron 1 (before the minus sign)

    # Require t0 strictly between (-1,+1) for strict single-point-only on both sides.
    # If t0 is too close to an endpoint, strict inactivity might break numerically.
    # if not (-1.0 + 1e-6 < t0 < 1.0 - 1e-6):
    #     # You can relax this if you want, but single-point-only becomes numerically fragile.
    #     raise RuntimeError(f"Root t0={t0:.6f} is not strictly inside (-1,1). Cannot guarantee single-point-only.")

    # Neuron 0: hinge at t0, active at +1 only
    # pre1(t) = w1 (t - t0)
    # pre1(+1)=w1(1-t0)=a1 => w1=a1/(1-t0), b1=-w1*t0
    w1 = a1 / (1.0 - t0)
    b1 = -w1 * t0

    # Neuron 1: hinge at t0, active at -1 only
    # pre2(t) = w2 (t - t0)
    # need ReLU(pre2(-1))=a2, and w2 must be negative:
    # pre2(-1)=w2(-1-t0)=a2 => w2=a2/(-1-t0) (denominator negative) => w2<0, b2=-w2*t0
    w2 = a2 / (-1.0 - t0)
    b2 = -w2 * t0

    # Sanity: check activity pattern at training points
    pre1_m1, pre1_p1 = w1 * (-1.0) + b1, w1 * (+1.0) + b1
    pre2_m1, pre2_p1 = w2 * (-1.0) + b2, w2 * (+1.0) + b2

    if not (pre1_p1 > act_eps and pre1_m1 <= act_eps):
        raise RuntimeError("Neuron 0 is not single-point-only on (+1).")
    if not (pre2_m1 > act_eps and pre2_p1 <= act_eps):
        raise RuntimeError("Neuron 1 is not single-point-only on (-1).")

    # Build collapsed params in the *d-dim core format* for a 1D model:
    # We represent t as a 1D input with shape (n,1), so w must be (k,1).
    W = np.array([[w1], [w2]], dtype=float)     # (2,1)
    B = np.array([b1, b2], dtype=float)         # (2,)
    V = np.array([+1.0, -1.0], dtype=float)     # (2,)

    params_col = NetworkParams(w=W, b=B, v=V)

    info = {
        "t0_teacher": float(t0),
        "g_plus1": float(gp),
        "g_minus1": float(gm),
        "constructed": {
            "w1": float(w1), "b1": float(b1),
            "w2": float(w2), "b2": float(b2),
        },
        "activity_checks": {
            "pre1(-1)": float(pre1_m1), "pre1(+1)": float(pre1_p1),
            "pre2(-1)": float(pre2_m1), "pre2(+1)": float(pre2_p1),
        },
    }
    return params_col, info

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def signed_train_margin(params: "NetworkParams", x_train: np.ndarray, y_train: np.ndarray) -> float:
    """
    Signed margin on a finite dataset:
        m = min_i y_i f(x_i)
    This is the SAME definition used in collapse_to_2relu_1d_with_margin_match.
    """
    x_train = np.asarray(x_train, dtype=float).reshape(-1)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    f = network_forward(params, x_train).astype(float)
    return float(np.min(y_train * f))


def count_single_point_only_neurons(
    params: "NetworkParams",
    x_1d: np.ndarray,
    act_eps: float = 1e-9,
):
    """
    For exactly 2 training points x_1d (shape (2,)), count how many neurons are:
      - dead:   active on 0 points
      - single: active on exactly 1 point
      - both:   active on both points
    where "active" means (w*x + b) > act_eps.
    """
    x_1d = np.asarray(x_1d, dtype=float).reshape(-1)
    if x_1d.shape[0] != 2:
        raise ValueError("This helper expects exactly 2 training points.")

    z = x_1d[:, None] * params.w[None, :] + params.b[None, :]  # (2,k)
    a = (z > act_eps).astype(int)                               # (2,k)
    s = a.sum(axis=0)                                           # (k,) in {0,1,2}

    num_dead = int(np.sum(s == 0))
    num_single = int(np.sum(s == 1))
    num_both = int(np.sum(s == 2))
    return num_dead, num_single, num_both


def overlay_plot_rich_vs_collapse_1d(
    params_rich: "NetworkParams",
    params_collapse_1d: "NetworkParams",
    t_collapse: int,
    save_path: str,
):
    """
    Overlay plot on a 1D grid: rich vs collapsed.
    """
    x_plot = np.linspace(-2.0, 2.0, 800)
    f_rich = network_forward(params_rich, x_plot).astype(float).reshape(-1)
    f_col = network_forward(params_collapse_1d, x_plot).astype(float).reshape(-1)

    f_r_m1 = float(network_forward(params_rich, np.array([-1.0]))[0])
    f_r_p1 = float(network_forward(params_rich, np.array([+1.0]))[0])
    f_c_m1 = float(network_forward(params_collapse_1d, np.array([-1.0]))[0])
    f_c_p1 = float(network_forward(params_collapse_1d, np.array([+1.0]))[0])

    print("=== OVERLAY DIAGNOSTICS (1D, on y-axis coordinate) ===")
    print(f"t_collapse={t_collapse}")
    print(f"rich:     f(-1)={f_r_m1:.6f}, f(+1)={f_r_p1:.6f}")
    print(f"collapse: f(-1)={f_c_m1:.6f}, f(+1)={f_c_p1:.6f}")
    print(f"max|rich-collapse| on grid = {float(np.max(np.abs(f_rich - f_col))):.6e}")
    print("======================================================\n")

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, f_rich, label="Rich (trained on y)")
    plt.plot(x_plot, f_col, "--", label="Collapse (1D 2-ReLU)")
    plt.axhline(0.0, linestyle="--")
    plt.scatter([-1.0, +1.0], [f_r_m1, f_r_p1], s=80, marker="X", label="train points (rich)")
    plt.scatter([-1.0, +1.0], [f_c_m1, f_c_p1], s=80, marker="o", label="train points (collapse)")
    plt.xlabel("t (y coordinate)")
    plt.ylabel("f")
    plt.title(f"Overlay: rich + 1D collapse, t={t_collapse}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print("Saved overlay plot to:", save_path)


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

# def projection_margin(params: "NetworkParams") -> float:
#     x_grid = np.linspace(-2.0, 2.0, 1000)
#     y_grid = np.sign(x_grid)
#     y_grid[y_grid == 0] = 1.0
#     f = network_forward(params, x_grid)
#     return float(np.min(y_grid * f))

def projection_margin(params: "NetworkParams", x_min=-2.0, x_max=2.0, num=2000, gap=0.05) -> float:
    """
    Projection margin on a 1D grid:
        m_proj = min_{x in grid} sign(x) * f(x)
    Excludes [-gap, +gap] so the min isn't dominated near the decision boundary.
    """
    x_left = np.linspace(x_min, -gap, num // 2, endpoint=True)
    x_right = np.linspace(gap, x_max, num // 2, endpoint=True)
    x_grid = np.concatenate([x_left, x_right], axis=0)

    y_grid = np.sign(x_grid)
    y_grid[y_grid == 0] = 1.0

    f = network_forward(params, x_grid).astype(float).reshape(-1)
    return float(np.min(y_grid * f))


def find_root_on_grid_1d(
    params: "NetworkParams",
    x_min: float = -2.0,
    x_max: float = 2.0,
    num: int = 4001,
) -> float:
    """
    Finds an approximate root x0 where f(x0)=0 by scanning a grid and linear interpolation.
    Picks the root closest to 0 if multiple exist.
    Raises if no sign change found.
    """
    xg = np.linspace(x_min, x_max, num)
    fg = network_forward(params, xg).astype(float).reshape(-1)

    s = np.sign(fg)
    s[s == 0] = 1.0
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if len(idx) == 0:
        raise RuntimeError("No root found in grid range. Try widening [x_min,x_max].")

    roots = []
    for i in idx:
        x1, x2 = float(xg[i]), float(xg[i + 1])
        f1, f2 = float(fg[i]), float(fg[i + 1])
        xr = x1 - f1 * (x2 - x1) / (f2 - f1 + 1e-18)
        roots.append(xr)

    roots = np.array(roots, dtype=float)
    x0 = float(roots[np.argmin(np.abs(roots))])  # closest to 0
    return x0



def geometric_margin_from_root(x0: float) -> float:
    """
    margin = min(|x0+1|, |x0-1|)
    """
    return float(min(abs(x0 + 1.0), abs(x0 - 1.0)))



# def experiment_6e_overparam_cluster_then_collapse_compare_margins(
#     k: int = 20,
#     d: int = 10,
#     n: int = 1000,
#     radius: float = 0.01,
#     learning_rate: float = 0.01,
#     # max_pre_collapse_iters: int = 1_000_000,
#     max_pre_collapse_iters: int = 500_000,
#     post_collapse_iters: int = 1_000_000,
#     collapse_loss_threshold: float = 1e-12,
#     seed: int = 42,
#     # track_every: int = 100_000,
#     track_every: int = 1000,
#     act_eps: float = 1e-9,
#     root_x_min: float = -2.0,
#     root_x_max: float = 2.0,
#     root_grid_num: int = 4001,
# ):
#     """
#     What this does:

#     1) Train a rich 1D k-ReLU network on two points x={-1,+1}, y={-1,+1}
#        until BOTH:
#          - every neuron is active on exactly one of the two points (single-point-only),
#          - exponential loss is tiny (loss < collapse_loss_threshold),
#        OR until max_pre_collapse_iters.

#     2) Collapse (distill) the rich network to a 2-ReLU 1D network:
#          f(x) = ReLU(w1 x + b1) - ReLU(w2 x + b2)   (v=[+1,-1])
#        using your existing collapse_to_2relu_1d_with_margin_match(...).

#     3) Save overlay plot at collapse time:
#          debug_projection_with_collapse_overlay.png

#     4) Continue training BOTH networks in parallel for post_collapse_iters,
#        and plot a GEOMETRIC projection margin defined by you:
#          - find x0 where f(x0)=0 (x-axis crossing on 1D projection),
#          - margin = min(|x0+1|, |x0-1|)   (distance to nearest of ±1).

#        IMPORTANT: The rich margin curve is shown from iteration 0 (init),
#        and the collapsed curve starts at t_collapse.
#        A vertical RED line is drawn at t=t_collapse.

#        Saved to:
#          debug_geom_margin_root_distance.png
#     """

#     rng = np.random.default_rng(seed)

#     # -----------------------------
#     # local helpers
#     # -----------------------------
#     def print_params(title: str, params: "NetworkParams", max_print: int = 10):
#         print(f"\n=== {title} ===")
#         print(f"k={params.k}")
#         m = min(max_print, params.k)
#         for j in range(m):
#             print(
#                 f"j={j:02d}: v={int(params.v[j]):+d}, "
#                 f"w={float(params.w[j]): .6f}, b={float(params.b[j]): .6f}"
#             )
#         if m < params.k:
#             print(f"... (printed first {m} of {params.k})")
#         print("=========================\n")

#     def gd_step_wb_only(params: "NetworkParams", x: np.ndarray, y: np.ndarray, lr: float):
#         grads = compute_gradients(params, x, y)
#         new_params = NetworkParams(
#             w=params.w - lr * grads.w,
#             b=params.b - lr * grads.b,
#             v=params.v.copy(),
#         )
#         loss = float(exponential_loss(y, network_forward(new_params, x)))
#         return new_params, loss

#     def overlay_plot_rich_vs_collapse_1d(
#         params_rich: "NetworkParams",
#         params_collapse_1d: "NetworkParams",
#         t_collapse: int,
#         save_path: str,
#     ):
#         x_plot = np.linspace(-2.0, 2.0, 800)
#         f_rich = network_forward(params_rich, x_plot).astype(float)
#         f_col = network_forward(params_collapse_1d, x_plot).astype(float)

#         f_r_m1 = float(network_forward(params_rich, np.array([-1.0]))[0])
#         f_r_p1 = float(network_forward(params_rich, np.array([+1.0]))[0])
#         f_c_m1 = float(network_forward(params_collapse_1d, np.array([-1.0]))[0])
#         f_c_p1 = float(network_forward(params_collapse_1d, np.array([+1.0]))[0])

#         print("=== OVERLAY DIAGNOSTICS (1D) ===")
#         print(f"t_collapse={t_collapse}")
#         print(f"rich:     f(-1)={f_r_m1:.6f}, f(+1)={f_r_p1:.6f}")
#         print(f"collapse: f(-1)={f_c_m1:.6f}, f(+1)={f_c_p1:.6f}")
#         print(f"max|rich-collapse| on grid = {float(np.max(np.abs(f_rich - f_col))):.6e}")
#         print("===============================\n")

#         plt.figure(figsize=(10, 6))
#         plt.plot(x_plot, f_rich, label="Projection (rich f(x) at collapse)")
#         plt.plot(x_plot, f_col, "--", label="Collapse (1D 2-ReLU)")
#         plt.axhline(0.0, linestyle="--")
#         plt.scatter([-1.0, +1.0], [f_r_m1, f_r_p1], s=80, marker="X", label="train points (rich)")
#         plt.scatter([-1.0, +1.0], [f_c_m1, f_c_p1], s=80, marker="o", label="train points (collapse)")
#         plt.xlabel("x")
#         plt.ylabel("f")
#         plt.title(f"Overlay: rich projection + 1D collapse, t={t_collapse}")
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=200)
#         plt.close()
#         print("Saved overlay plot to:", save_path)

#     # -----------------------------
#     # STEP 1: dataset (two points)
#     # -----------------------------
#     X_full = np.zeros((2, d), dtype=float)
#     y = np.array([-1.0, +1.0], dtype=float)
#     X_full[0, 0] = -1.0
#     X_full[1, 0] = +1.0
#     x_1d = X_full[:, 0].astype(float)  # [-1, +1]

#     # -----------------------------
#     # STEP 2: init rich network
#     # -----------------------------
#     while True:
#         v = rng.choice([-1.0, 1.0], size=k).astype(float)
#         if np.any(v > 0) and np.any(v < 0):
#             break

#     w = rng.normal(0.0, np.sqrt(2.0), size=k).astype(float)
#     b = np.zeros(k, dtype=float)
#     params_rich = NetworkParams(w=w, b=b, v=v)

#     print_params("INIT rich network (1D He init)", params_rich, max_print=10)

#     f0 = network_forward(params_rich, x_1d)
#     L0 = float(exponential_loss(y, f0))
#     print("At init:")
#     print("  f(x_1d) =", f0)
#     print("  y*f     =", y * f0)
#     print("  loss    =", f"{L0:.6e}\n")

#     # -----------------------------
#     # Track geom margin for RICH from the very beginning
#     # (global iteration axis)
#     # -----------------------------
#     global_iters = [0]
#     try:
#         x0_init = find_root_on_grid_1d(params_rich, x_min=root_x_min, x_max=root_x_max, num=root_grid_num)
#         m_init = geometric_margin_from_root(x0_init)
#     except RuntimeError:
#         x0_init, m_init = np.nan, np.nan
#     rich_geom_hist = [m_init]

#     # -----------------------------
#     # STEP 3: pre-collapse training
#     # -----------------------------
#     t_collapse = None
#     loss_gate = collapse_loss_threshold

#     for t in range(1, max_pre_collapse_iters + 1):
#         params_rich, loss = gd_step_wb_only(params_rich, x_1d, y, learning_rate)

#         num_dead, num_single, num_both = count_single_point_only_neurons(params_rich, x_1d, act_eps=act_eps)

#         if t % track_every == 0 or t == 1:
#             # log + record rich geom margin along the way
#             f = network_forward(params_rich, x_1d)
#             if t%100_000 == 0: 
#                 print(f"[pre t={t}] loss={loss:.3e} | f={f} | y*f={y*f} | "
#                     f"neurons(single={num_single}, both={num_both}, dead={num_dead})")

#             try:
#                 x0 = find_root_on_grid_1d(params_rich, x_min=root_x_min, x_max=root_x_max, num=root_grid_num)
#                 m = geometric_margin_from_root(x0)
#             except RuntimeError:
#                 m = np.nan
#             global_iters.append(t)
#             rich_geom_hist.append(m)

#         # stop criterion
#         if (num_single == params_rich.k) and (loss < loss_gate):
#             t_collapse = t
#             print(f"\n*** READY TO COLLAPSE at t={t} | loss={loss:.3e} ***\n")
#             break

#     if t_collapse is None:
#         t_collapse = max_pre_collapse_iters
#         print("WARNING: did not reach (single-point-only AND tiny-loss) within max_pre_collapse_iters.")
#         print(f"Proceeding to collapse at t={t_collapse}.\n")

#     # Make sure we recorded a point exactly at t_collapse (useful for the vertical line)
#     if global_iters[-1] != t_collapse:
#         try:
#             x0 = find_root_on_grid_1d(params_rich, x_min=root_x_min, x_max=root_x_max, num=root_grid_num)
#             m = geometric_margin_from_root(x0)
#         except RuntimeError:
#             m = np.nan
#         global_iters.append(t_collapse)
#         rich_geom_hist.append(m)

#     print(f"\n*** COLLAPSE at t={t_collapse} ***\n")
#     print_params("rich params at collapse", params_rich, max_print=10)

#     # -----------------------------
#     # STEP 4: collapse to 2-ReLU
#     # -----------------------------
#     # params_collapse_1d, collapse_info = collapse_to_2relu_1d_with_margin_match(
#     #     params_rich=params_rich,
#     #     x_fit=x_1d,
#     #     y_fit=y,
#     #     distill_iters=20000,
#     #     distill_lr=0.01,
#     # )
#     params_collapse_1d, collapse_info = collapse_2relu_1d_with_3_constraints_enum(
#         params_rich=params_rich,
#         x_fit=x_1d,
#         y_fit=y,
#         x0_search_min=root_x_min,
#         x0_search_max=root_x_max,
#         x0_search_num=root_grid_num,
# )
#     # ===== GEOMETRIC MARGIN AT COLLAPSE =====

#     # rich
#     try:
#         x0_rich = find_root_on_grid_1d(
#             params_rich,
#             x_min=root_x_min,
#             x_max=root_x_max,
#             num=root_grid_num,
#         )
#         margin_rich_geom = geometric_margin_from_root(x0_rich)
#     except RuntimeError:
#         x0_rich = np.nan
#         margin_rich_geom = np.nan

#     # collapse
#     try:
#         x0_col = find_root_on_grid_1d(
#             params_collapse_1d,
#             x_min=root_x_min,
#             x_max=root_x_max,
#             num=root_grid_num,
#         )
#         margin_col_geom = geometric_margin_from_root(x0_col)
#     except RuntimeError:
#         x0_col = np.nan
#         margin_col_geom = np.nan

#     print("\n=== GEOMETRIC MARGINS AT COLLAPSE ===")
#     print(f"rich:     x0={x0_rich:.6f}  margin={margin_rich_geom:.6f}")
#     print(f"collapse: x0={x0_col:.6f}  margin={margin_col_geom:.6f}")
#     print(f"|diff| = {abs(margin_rich_geom - margin_col_geom):.6e}")
#     print("=====================================\n")

#     print("Collapsed v (should be [1, -1]):", params_collapse_1d.v)
#     print("Collapsed params:")
#     print(f"  w1={float(params_collapse_1d.w[0]):.6f}, b1={float(params_collapse_1d.b[0]):.6f}, v1=+1")
#     print(f"  w2={float(params_collapse_1d.w[1]):.6f}, b2={float(params_collapse_1d.b[1]):.6f}, v2=-1")
#     print("===============================================================\n")

#     # -----------------------------
#     # STEP 5: overlay at collapse time (your "good" plot)
#     # -----------------------------
#     save_overlay_path = "debug_projection_with_collapse_overlay.png"
#     overlay_plot_rich_vs_collapse_1d(
#         params_rich=params_rich,
#         params_collapse_1d=params_collapse_1d,
#         t_collapse=t_collapse,
#         save_path=save_overlay_path,
#     )

#     # -----------------------------
#     # STEP 6: post-collapse training in parallel
#     # Track geom margin for COLLAPSED starting at t_collapse
#     # -----------------------------
#     col_iters = [t_collapse]
#     try:
#         x0c = find_root_on_grid_1d(params_collapse_1d, x_min=root_x_min, x_max=root_x_max, num=root_grid_num)
#         mc0 = geometric_margin_from_root(x0c)
#     except RuntimeError:
#         mc0 = np.nan
#     col_geom_hist = [mc0]

#     for t_post in range(1, post_collapse_iters + 1):
#         params_rich, loss_r = gd_step_wb_only(params_rich, x_1d, y, learning_rate)
#         params_collapse_1d, loss_c = gd_step_wb_only(params_collapse_1d, x_1d, y, learning_rate)

#         if t_post % 10_000 == 0:
#             w1 = float(params_collapse_1d.w[0])
#             b1 = float(params_collapse_1d.b[0])
#             w2 = float(params_collapse_1d.w[1])
#             b2 = float(params_collapse_1d.b[1])

#             # optional: גם להדפיס את ה-root והמרג׳ין הגיאומטרי
#             try:
#                 x0 = find_root_on_grid_1d(
#                     params_collapse_1d,
#                     x_min=root_x_min,
#                     x_max=root_x_max,
#                     num=root_grid_num,
#                 )
#                 margin_geom = geometric_margin_from_root(x0)
#             except RuntimeError:
#                 x0 = np.nan
#                 margin_geom = np.nan

#             print(f"[post t={t_post}] "
#                 f"w1={w1:.6f}, b1={b1:.6f} | "
#                 f"w2={w2:.6f}, b2={b2:.6f} | "
#                 f"x0={x0:.6f}, geom_margin={margin_geom:.6f}")

#         if t_post % track_every == 0 or t_post == 1:
#             t_global = t_collapse + t_post

#             # rich
#             try:
#                 x0r = find_root_on_grid_1d(params_rich, x_min=root_x_min, x_max=root_x_max, num=root_grid_num)
#                 mr = geometric_margin_from_root(x0r)
#             except RuntimeError:
#                 mr = np.nan
#             global_iters.append(t_global)
#             rich_geom_hist.append(mr)

#             # collapse
#             try:
#                 x0c = find_root_on_grid_1d(params_collapse_1d, x_min=root_x_min, x_max=root_x_max, num=root_grid_num)
#                 mc = geometric_margin_from_root(x0c)
#             except RuntimeError:
#                 mc = np.nan
#             col_iters.append(t_global)
#             col_geom_hist.append(mc)

#             if t_post%100_000 == 0:
#                 print(f"[post t={t_post}] (global={t_global}) loss_r={loss_r:.3e} loss_c={loss_c:.3e} | "
#                   f"m_rich={mr:.6f} m_col={mc:.6f}")

#     # -----------------------------
#     # STEP 7: plot geom margin from init + vertical red line at collapse join
#     # -----------------------------
#     save_margin_path = "debug_geom_margin_root_distance.png"
#     plt.figure(figsize=(10, 6))

#     plt.plot(global_iters, rich_geom_hist, label="Geom margin (rich projection)")
#     plt.plot(col_iters, col_geom_hist, label="Geom margin (collapsed)")

#     plt.axvline(x=t_collapse, color="red", linestyle="--", linewidth=2, label="collapse join")

#     plt.ylim(bottom=0.0)
#     plt.xlabel("Training iterations (global)")
#     plt.ylabel("margin = min(|x0+1|, |x0-1|)")
#     plt.title("Geometric margin via root-distance: rich from init + collapse join")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_margin_path, dpi=200)
#     plt.close()
#     print("Saved geometric margin plot to:", save_margin_path)

#     return {
#         "t_collapse": int(t_collapse),
#         "params_rich_final": params_rich,
#         "params_collapse_final": params_collapse_1d,
#         "collapse_info": collapse_info,
#         "overlay_plot_path": save_overlay_path,
#         "geom_margin_plot_path": save_margin_path,
#         "geom_margin_history": {
#             "iters_rich": np.array(global_iters, dtype=int),
#             "m_rich": np.array(rich_geom_hist, dtype=float),
#             "iters_collapse": np.array(col_iters, dtype=int),
#             "m_collapse": np.array(col_geom_hist, dtype=float),
#             "t_collapse": int(t_collapse),
#         },
#     }



def _sample_unit_directions(rng: np.random.Generator, d: int, num: int) -> np.ndarray:
    """
    Sample 'num' random unit directions in R^d.
    """
    U = rng.normal(0.0, 1.0, size=(num, d))
    U /= (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    return U


def _forward_on_line_ddim(params_rich: "NetworkParams", t_grid: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Evaluate the rich d-dim network on the line x(t)=t*u.
    """
    t_grid = np.asarray(t_grid, dtype=float).reshape(-1)
    X_line = t_grid[:, None] * u[None, :]  # (n,d)
    return network_forward(params_rich, X_line).astype(float).reshape(-1)


def _forward_on_line(params_rich: "NetworkParams", t_grid: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Evaluate the rich network on the line x(t)=t*u.

    Supports two cases:
      - 1D rich network: params_rich.w is shape (k,), and network_forward expects (n,) scalars.
        In this case, the function effectively depends only on the scalar t (we ignore u).
      - dD rich network: params_rich.w is shape (d,k), and network_forward expects X shape (n,d).
    """
    t_grid = np.asarray(t_grid, dtype=float).reshape(-1)

    w = params_rich.w
    if np.ndim(w) == 1:
        # 1D network: f(t)
        return network_forward(params_rich, t_grid).astype(float).reshape(-1)

    # dD network: f(t*u)
    X_line = t_grid[:, None] * u[None, :]  # (n,d)
    return network_forward(params_rich, X_line).astype(float).reshape(-1)

def find_root_on_grid_1d_fn(fn_1d, x_min: float, x_max: float, num: int = 4001) -> float:
    """
    Find an approximate root x0 where fn_1d(x0)=0 by scanning a 1D grid and linear interpolation.
    Picks the root closest to 0 if multiple exist.
    Raises if no sign change found.
    """
    xg = np.linspace(x_min, x_max, num)
    fg = np.asarray(fn_1d(xg), dtype=float).reshape(-1)

    s = np.sign(fg)
    s[s == 0] = 1.0
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if len(idx) == 0:
        raise RuntimeError("No root found in grid range. Try widening [x_min,x_max].")

    roots = []
    for i in idx:
        x1, x2 = float(xg[i]), float(xg[i + 1])
        f1, f2 = float(fg[i]), float(fg[i + 1])
        xr = x1 - f1 * (x2 - x1) / (f2 - f1 + 1e-18)
        roots.append(xr)

    roots = np.array(roots, dtype=float)
    return float(roots[np.argmin(np.abs(roots))])

def worst_projection_geometric_margin_rich(
    params_rich: "NetworkParams",
    X_train: np.ndarray,
    num_directions: int = 200,
    seed: int = 0,
    root_grid_num: int = 4001,
) -> float:
    """
    Approximate the minimal geometric margin over random 1D directions u.

    For each random unit direction u:
      - Project training points: s_i = <u, X_i>
      - Restrict classifier to line x(t)=t*u: g_u(t)=f_rich(t*u)
      - Find root t0 where g_u(t0)=0 (closest to 0)
      - Define margin_u = min_i |t0 - s_i|

    Returns:
      min_u margin_u over sampled directions.
    """
    X_train = np.asarray(X_train, dtype=float)
    n, d = X_train.shape
    if n != 2:
        raise ValueError("This helper expects exactly 2 training points (shape (2,d)).")

    rng = np.random.default_rng(seed)
    U = _sample_unit_directions(rng, d=d, num=num_directions)

    # Conservative search radius for roots
    R = float(np.sqrt(d) + 3.0)
    x_min, x_max = -R, +R

    best = np.inf
    for u in U:
        s = X_train @ u  # (2,)
        try:
            t0 = find_root_on_grid_1d_fn(
                fn_1d=lambda tt, uu=u: _forward_on_line_ddim(params_rich, tt, uu),
                x_min=x_min,
                x_max=x_max,
                num=root_grid_num,
            )
        except RuntimeError:
            continue

        margin_u = float(np.min(np.abs(t0 - s)))
        if margin_u < best:
            best = margin_u

    if not np.isfinite(best):
        return np.nan
    return float(best)


# ------------------------------------------------------------
# External sampler (data generation)
# ------------------------------------------------------------
# def sample_points_y_times_sphere(
#     d: int,
#     n_points: int = 2,
#     seed: Optional[int] = None,
#     ensure_opposite: bool = True,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Sample training points in R^d as follows:
#       1) Sample x in R^{d-1} uniformly on the sphere of radius sqrt(d).
#       2) Sample y in {+1,-1}.
#       3) Form the point: (y, x_1, ..., x_{d-1}).

#     Returns:
#       X: shape (n_points, d)
#       y: shape (n_points,)
#     """
#     if d < 2:
#         raise ValueError("d must be >= 2 (first coordinate is y, plus d-1 sphere coordinates).")

#     rng = np.random.default_rng(seed)

#     if ensure_opposite and n_points == 2:
#         y = np.array([-1.0, +1.0], dtype=float)
#         rng.shuffle(y)
#     else:
#         y = rng.choice([-1.0, +1.0], size=n_points).astype(float)

#     z = rng.normal(0.0, 1.0, size=(n_points, d - 1))
#     norms = np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
#     x = z / norms
#     x *= np.sqrt(float(d))

#     X = np.zeros((n_points, d), dtype=float)
#     X[:, 0] = y
#     X[:, 1:] = x
#     return X, y


# def sample_points_y_times_sphere(
#     d: int,
#     n_points: int = 2,
#     seed: Optional[int] = None,
#     ensure_opposite: bool = True,
#     fixed_second_coord: Optional[float] = None,  # 👈 NEW
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Sample training points in R^d.

#     If fixed_second_coord is None:
#         1) Sample x in R^{d-1} uniformly on sphere radius sqrt(d).
#         2) Form (y, x_1, ..., x_{d-1})

#     If fixed_second_coord is not None:
#         Return points of the form:
#             (y, fixed_second_coord, 0, ..., 0)
#     """

#     if d < 2:
#         raise ValueError("d must be >= 2.")

#     rng = np.random.default_rng(seed)

#     if ensure_opposite and n_points == 2:
#         y = np.array([-1.0, +1.0], dtype=float)
#         rng.shuffle(y)
#     else:
#         y = rng.choice([-1.0, +1.0], size=n_points).astype(float)

#     X = np.zeros((n_points, d), dtype=float)
#     X[:, 0] = y

#     # -----------------------------------------
#     # CASE 1: fixed second coordinate
#     # -----------------------------------------
#     if fixed_second_coord is not None:
#         X[:, 1] = float(fixed_second_coord)
#         # all other coords already zero
#         return X, y

#     # -----------------------------------------
#     # CASE 2: original sphere sampling
#     # -----------------------------------------
#     z = rng.normal(0.0, 1.0, size=(n_points, d - 1))
#     norms = np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
#     x = z / norms
#     x *= np.sqrt(float(d))

#     X[:, 1:] = x
#     return X, y

def sample_points_y_times_sphere(
    d: int,
    n_points: int = 2,
    seed: Optional[int] = None,
    ensure_opposite: bool = True,
    fixed_second_coord: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    if d < 3:
        raise ValueError("d must be >= 3 (need y + fixed coord + sphere coords).")

    rng = np.random.default_rng(seed)

    if ensure_opposite and n_points == 2:
        y = np.array([-1.0, +1.0], dtype=float)
        rng.shuffle(y)
    else:
        y = rng.choice([-1.0, +1.0], size=n_points).astype(float)

    X = np.zeros((n_points, d), dtype=float)
    X[:, 0] = y

    if fixed_second_coord is not None:
        X[:, 1] = float(fixed_second_coord)

        # sample the remaining coords on a sphere in R^{d-2} with radius sqrt(d)
        z = rng.normal(0.0, 1.0, size=(n_points, d - 2))
        norms = np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
        x = z / norms
        x *= np.sqrt(float(d))
        X[:, 2:] = x

        return X, y

    # original: sphere in R^{d-1}
    z = rng.normal(0.0, 1.0, size=(n_points, d - 1))
    norms = np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    x = z / norms
    x *= np.sqrt(float(d))
    X[:, 1:] = x

    return X, y

# ------------------------------------------------------------
# Helpers for "collapse is on y"
# ------------------------------------------------------------
def points_on_y_axis(t_1d: np.ndarray, d: int) -> np.ndarray:
    """
    Embed 1D scalars t into R^d along the y-axis:
        (t, 0, 0, ..., 0)
    """
    t_1d = np.asarray(t_1d, dtype=float).reshape(-1)
    X = np.zeros((t_1d.shape[0], d), dtype=float)
    X[:, 0] = t_1d
    return X


def forward_rich_on_y_axis(params_rich: "NetworkParams", t_1d: np.ndarray, d: int) -> np.ndarray:
    """
    Define the 1D function used for collapse:
        g(t) = f_rich((t, 0, ..., 0))
    """
    X_axis = points_on_y_axis(t_1d, d)
    return network_forward(params_rich, X_axis).astype(float).reshape(-1)



# def signed_train_margin(params: "NetworkParams", X_train: np.ndarray, y_train: np.ndarray) -> float:
#     """
#     Signed margin on a finite dataset:
#         m = min_i y_i f(x_i)
#     """
#     X_train = np.asarray(X_train, dtype=float)
#     y_train = np.asarray(y_train, dtype=float).reshape(-1)
#     f = network_forward(params, X_train).astype(float).reshape(-1)
#     return float(np.min(y_train * f))
# ------------------------------------------------------------
# Helpers (1D network, collapse/projection is on y)
# ------------------------------------------------------------
def signed_train_margin(params: "NetworkParams", x_train: np.ndarray, y_train: np.ndarray) -> float:
    """
    Signed margin on a finite dataset:
        m = min_i y_i f(x_i)
    """
    x_train = np.asarray(x_train, dtype=float).reshape(-1)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    f = network_forward(params, x_train).astype(float).reshape(-1)
    return float(np.min(y_train * f))


# def count_single_point_only_neurons_2pts(
#     params: "NetworkParams",
#     X_2pts: np.ndarray,
#     act_eps: float = 1e-9,
# ):
#     """
#     For exactly 2 training points X_2pts (shape (2,d)), count neurons that are:
#       - dead:   active on 0 points
#       - single: active on exactly 1 point
#       - both:   active on both points

#     "active" means pre-activation (w·x + b) > act_eps.

#     IMPORTANT:
#     This assumes your network is of the form sum_j v_j ReLU(<w_j,x> + b_j),
#     and that params.w is shaped (d,k) (or anything compatible with X @ w).
#     """
#     X_2pts = np.asarray(X_2pts, dtype=float)
#     if X_2pts.shape[0] != 2:
#         raise ValueError("This helper expects exactly 2 training points.")

#     # pre-activations: (2,k)
#     z = X_2pts @ params.w + params.b  # expects params.w shape (d,k) and params.b shape (k,)
#     a = (z > act_eps).astype(int)
#     s = a.sum(axis=0)  # (k,) in {0,1,2}

#     num_dead = int(np.sum(s == 0))
#     num_single = int(np.sum(s == 1))
#     num_both = int(np.sum(s == 2))
#     return num_dead, num_single, num_both

def count_single_point_only_neurons(
    params: "NetworkParams",
    x_1d: np.ndarray,
    act_eps: float = 1e-9,
):
    """
    For exactly 2 training points x_1d (shape (2,)), count how many neurons are:
      - dead:   active on 0 points
      - single: active on exactly 1 point
      - both:   active on both points
    where "active" means (w*x + b) > act_eps.
    """
    x_1d = np.asarray(x_1d, dtype=float).reshape(-1)
    if x_1d.shape[0] != 2:
        raise ValueError("This helper expects exactly 2 training points.")

    z = x_1d[:, None] * params.w[None, :] + params.b[None, :]  # (2,k)
    a = (z > act_eps).astype(int)                               # (2,k)
    s = a.sum(axis=0)                                           # (k,) in {0,1,2}

    num_dead = int(np.sum(s == 0))
    num_single = int(np.sum(s == 1))
    num_both = int(np.sum(s == 2))
    return num_dead, num_single, num_both



def projection_margin_on_y_axis(params_rich: "NetworkParams", d: int,
                               t_min=-2.0, t_max=2.0, num=2000, gap=0.05) -> float:
    """
    Projection margin on the y-axis (excluding a small gap around 0):
        m_proj = min_{t in grid} sign(t) * g(t)
    where g(t)=f((t,0,...,0)).
    """
    t_left = np.linspace(t_min, -gap, num // 2, endpoint=True)
    t_right = np.linspace(gap, t_max, num // 2, endpoint=True)
    t_grid = np.concatenate([t_left, t_right], axis=0)

    y_grid = np.sign(t_grid)
    y_grid[y_grid == 0] = 1.0

    g = forward_rich_on_y_axis(params_rich, t_grid, d).astype(float).reshape(-1)
    return float(np.min(y_grid * g))





def geometric_margin_from_root(x0: float) -> float:
    """
    margin = min(|x0+1|, |x0-1|)
    """
    return float(min(abs(x0 + 1.0), abs(x0 - 1.0)))


def overlay_plot_rich_vs_collapse_on_y_axis(
    params_rich: "NetworkParams",
    params_collapse_1d: "NetworkParams",
    t_collapse: int,
    d: int,
    save_path: str,
):
    """
    Overlay plot on the y-axis: rich g(t)=f((t,0,...,0)) vs collapsed 1D network.
    """
    t_plot = np.linspace(-2.0, 2.0, 800)
    g_rich = forward_rich_on_y_axis(params_rich, t_plot, d).astype(float).reshape(-1)
    f_col = network_forward(params_collapse_1d, t_plot).astype(float).reshape(-1)

    g_r_m1 = float(forward_rich_on_y_axis(params_rich, np.array([-1.0]), d)[0])
    g_r_p1 = float(forward_rich_on_y_axis(params_rich, np.array([+1.0]), d)[0])
    f_c_m1 = float(network_forward(params_collapse_1d, np.array([-1.0]))[0])
    f_c_p1 = float(network_forward(params_collapse_1d, np.array([+1.0]))[0])

    print("=== OVERLAY DIAGNOSTICS (y-axis) ===")
    print(f"t_collapse={t_collapse}")
    print(f"rich g(t): f(-1)={g_r_m1:.6f}, f(+1)={g_r_p1:.6f}")
    print(f"collapse:  f(-1)={f_c_m1:.6f}, f(+1)={f_c_p1:.6f}")
    print(f"max|rich-collapse| on grid = {float(np.max(np.abs(g_rich - f_col))):.6e}")
    print("===================================\n")

    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, g_rich, label="Rich g(t)=f((t,0,...,0))")
    plt.plot(t_plot, f_col, "--", label="Collapsed (1D 2-ReLU)")
    plt.axhline(0.0, linestyle="--")
    plt.scatter([-1.0, +1.0], [g_r_m1, g_r_p1], s=80, marker="X", label="axis points (rich)")
    plt.scatter([-1.0, +1.0], [f_c_m1, f_c_p1], s=80, marker="o", label="axis points (collapse)")
    plt.xlabel("t (y-axis coordinate)")
    plt.ylabel("value")
    plt.title(f"Overlay on y-axis: rich projection + 1D collapse, t={t_collapse}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print("Saved overlay plot to:", save_path)


# ============================================================
# Neuron activity counts (d-dim, exactly 2 points)
# ============================================================
def count_single_point_only_neurons_2pts_ddim(
    params: "NetworkParams",
    X_2pts: np.ndarray,
    act_eps: float = 1e-9,
):
    """
    For exactly 2 training points X_2pts (shape (2,d)), count how many neurons are:
      - dead:   active on 0 points
      - single: active on exactly 1 point
      - both:   active on both points

    "active" means pre-activation > act_eps.

    Assumes d-dim network with:
      params.w shape (k,d), params.b shape (k,)
      pre = w @ X^T + b
    """
    X_2pts = np.asarray(X_2pts, dtype=float)
    if X_2pts.shape[0] != 2:
        raise ValueError("This helper expects exactly 2 training points.")

    pre = params.w @ X_2pts.T + params.b[:, None]  # (k,2)
    a = (pre > act_eps).astype(int)                # (k,2)
    s = a.sum(axis=1)                              # (k,) in {0,1,2}

    num_dead = int(np.sum(s == 0))
    num_single = int(np.sum(s == 1))
    num_both = int(np.sum(s == 2))
    return num_dead, num_single, num_both


def restrict_rich_to_y_axis_params(params_rich: "NetworkParams") -> "NetworkParams":
    """
    Build an equivalent 1D network along the y-axis:
        g(t) = f_rich((t,0,...,0)) = sum_j v_j ReLU(w_j0 * t + b_j)
    """
    w_axis = params_rich.w[:, 0].copy()   # (k,)
    b_axis = params_rich.b.copy()         # (k,)
    v_axis = params_rich.v.copy()         # (k,)
    return NetworkParams(w=w_axis, b=b_axis, v=v_axis)


def _format_w_entry(w_row) -> str:
    """
    Format a neuron's weight entry robustly:
      - if scalar -> print scalar
      - if shape (1,) or (1,1) -> print scalar
      - else -> print full vector
    """
    arr = np.asarray(w_row)
    if arr.ndim == 0:
        return f"{float(arr): .6f}"
    flat = arr.reshape(-1)
    if flat.size == 1:
        return f"{float(flat[0]): .6f}"
    return np.array2string(flat, precision=6, suppress_small=False)

# ------------------------------------------------------------
# Main experiment (FULL function)
# ------------------------------------------------------------
def experiment_6e_overparam_cluster_then_collapse_compare_margins(
    k: int = 20,
    d: int = 10,
    learning_rate: float = 0.001,
    max_pre_collapse_iters: int = 1_000_000,
    post_collapse_iters: int = 1_000_000,
    collapse_loss_threshold: float = 1e-12,
    seed: int = 42,
    track_every: int = 1000,
    act_eps: float = 1e-9,
    root_grid_num: int = 4001,
    root_t_min: float = -2.0,
    root_t_max: float = 2.0,
    num_directions: int = 200,
    worst_proj_seed: int = 123,
):
    """
    d-dimensional rich training + 1D collapse on the y-axis.

    Data:
      - Sample x in R^{d-1} uniformly on sphere radius sqrt(d)
      - Sample y in {±1}
      - Point is (y, x)

    Rich model (trained on full X in R^d):
        f(x) = sum_j v_j ReLU(<w_j, x> + b_j)

    Collapse model (1D in dim-1):
        g(t) = f_rich((t,0,...,0))
        f_col(t) = ReLU(w1 t + b1) - ReLU(w2 t + b2)

    Tracked margins (global iterations):
      1) Geom margin on y-axis for rich via root-distance on g(t)
      2) Worst-projection geom margin for rich (Monte-Carlo over directions)
      3) Geom margin for collapsed model via root-distance

    Saves:
      - debug_projection_with_collapse_overlay.png
      - debug_geom_margin_root_distance.png
      - debug_geom_margin_root_distance_with_worst_projection.png
    """

    rng = np.random.default_rng(seed)

    # -----------------------------
    # local GD step: update w,b only (keep v fixed)
    # -----------------------------
    def gd_step_wb_only(params: "NetworkParams", X: np.ndarray, y: np.ndarray, lr: float):
        grads = compute_gradients(params, X, y)
        new_params = NetworkParams(
            w=params.w - lr * grads.w,
            b=params.b - lr * grads.b,
            v=params.v.copy(),
        )
        loss = float(exponential_loss(y, network_forward(new_params, X)))
        return new_params, loss

    # -----------------------------
    # STEP 1: sample two points in R^d
    # -----------------------------
    X_full, y = sample_points_y_times_sphere(
        d=d,
        n_points=2,
        seed=seed + 12345,
        ensure_opposite=True,
        fixed_second_coord=10
    )

    print("Train points X_full (shape (2,d)):")
    print(X_full)
    print("Train labels y:", y)

    # Collapse training points on axis
    t_axis = np.array([-1.0, +1.0], dtype=float)
    y_axis = np.array([-1.0, +1.0], dtype=float)

    # -----------------------------
    # STEP 2: init rich network (k,d)
    # -----------------------------
    while True:
        v = rng.choice([-1.0, 1.0], size=k).astype(float)
        if np.any(v > 0) and np.any(v < 0):
            break

    # Scaled He-like init to reduce blow-ups in dD
    w = rng.normal(0.0, np.sqrt(2.0 / float(d)), size=(k, d)).astype(float)
    b = np.zeros(k, dtype=float)
    params_rich = NetworkParams(w=w, b=b, v=v)

    f0 = network_forward(params_rich, X_full).astype(float).reshape(-1)
    L0 = float(exponential_loss(y, f0))
    print("At init (rich on full points):")
    print("  f(X_full) =", f0)
    print("  y*f       =", y * f0)
    print("  loss      =", f"{L0:.6e}\n")

    # -----------------------------
    # Track histories
    # -----------------------------
    global_iters = [0]

    # rich y-axis margin at init
    try:
        t0_init = find_root_on_grid_1d_fn(
            fn_1d=lambda tt: forward_rich_on_y_axis(params_rich, tt, d),
            x_min=root_t_min,
            x_max=root_t_max,
            num=root_grid_num,
        )
        m_rich_y_init = geometric_margin_from_root(t0_init)
    except RuntimeError:
        m_rich_y_init = np.nan

    # rich worst-projection margin at init
    m_rich_worst_init = worst_projection_geometric_margin_rich(
        params_rich=params_rich,
        X_train=X_full,
        num_directions=num_directions,
        seed=worst_proj_seed,
        root_grid_num=root_grid_num,
    )

    rich_y_geom_hist = [m_rich_y_init]
    rich_worstproj_hist = [m_rich_worst_init]

    # -----------------------------
    # STEP 3: pre-collapse training (rich trains on full X_full)
    # -----------------------------
    t_collapse = None
    for t in range(1, max_pre_collapse_iters + 1):
        params_rich, loss = gd_step_wb_only(params_rich, X_full, y, learning_rate)

        num_dead, num_single, num_both = count_single_point_only_neurons_2pts_ddim(
            params_rich, X_full, act_eps=act_eps
        )

        if t % track_every == 0 or t == 1:
            if t % 100_000 == 0:
                f_train = network_forward(params_rich, X_full).astype(float).reshape(-1)
                print(
                    f"[pre t={t}] loss={loss:.3e} | f_train={f_train} | y*f={y*f_train} | "
                    f"neurons(single={num_single}, both={num_both}, dead={num_dead})"
                )

            # rich y-axis margin
            try:
                t0 = find_root_on_grid_1d_fn(
                    fn_1d=lambda tt: forward_rich_on_y_axis(params_rich, tt, d),
                    x_min=root_t_min,
                    x_max=root_t_max,
                    num=root_grid_num,
                )
                m_y = geometric_margin_from_root(t0)
            except RuntimeError:
                m_y = np.nan

            # rich worst-projection margin
            m_worst = worst_projection_geometric_margin_rich(
                params_rich=params_rich,
                X_train=X_full,
                num_directions=num_directions,
                seed=worst_proj_seed,
                root_grid_num=root_grid_num,
            )

            global_iters.append(t)
            rich_y_geom_hist.append(m_y)
            rich_worstproj_hist.append(m_worst)

        if (num_single == params_rich.k) and (loss < collapse_loss_threshold):
            t_collapse = t
            print(f"\n*** READY TO COLLAPSE at t={t} | loss={loss:.3e} ***\n")
            break

    if t_collapse is None:
        t_collapse = max_pre_collapse_iters
        print("WARNING: did not reach (single-point-only AND tiny-loss) within max_pre_collapse_iters.")
        print(f"Proceeding to collapse at t={t_collapse}.\n")

    # Ensure recorded at t_collapse
    if global_iters[-1] != t_collapse:
        try:
            t0 = find_root_on_grid_1d_fn(
                fn_1d=lambda tt: forward_rich_on_y_axis(params_rich, tt, d),
                x_min=root_t_min,
                x_max=root_t_max,
                num=root_grid_num,
            )
            m_y = geometric_margin_from_root(t0)
        except RuntimeError:
            m_y = np.nan

        m_worst = worst_projection_geometric_margin_rich(
            params_rich=params_rich,
            X_train=X_full,
            num_directions=num_directions,
            seed=worst_proj_seed,
            root_grid_num=root_grid_num,
        )

        global_iters.append(t_collapse)
        rich_y_geom_hist.append(m_y)
        rich_worstproj_hist.append(m_worst)

    print(f"\n*** COLLAPSE at t={t_collapse} ***\n")
    print(f"Parameters at collapse:\n{params_rich}")
    gp = float(forward_rich_on_y_axis(params_rich, np.array([+1.0]), d)[0])
    gm = float(forward_rich_on_y_axis(params_rich, np.array([-1.0]), d)[0])

    print(f"[collapse check] g(+1)={gp:.6f}, g(-1)={gm:.6f}")

    # Require opposite signs (strict)
    if gp * gm >= 0.0:
        print(
            "\n[STOP] Rich network is not separable on the y-axis at ±1.\n"
            f"g(+1)={gp:.6f}, g(-1)={gm:.6f}\n"
            "Skipping collapse and exiting experiment early.\n"
        )
        return {
            "status": "stopped_before_collapse",
            "reason": "not_separable_on_y_axis",
            "t_collapse": int(t_collapse),
            "g_plus1": gp,
            "g_minus1": gm,
            "params_rich_at_stop": params_rich,
            "train_data": {"X_full": X_full, "y": y},
        }

    # -----------------------------
    # STEP 4: collapse to 2-ReLU (1D) on y-axis
    # -----------------------------
    params_collapse_1d, collapse_info = collapse_2relu_1d_with_3_constraints_enum(
        params_rich=params_rich,
        x_fit=t_axis,
        y_fit=y_axis,
        x0_search_min=root_t_min,
        x0_search_max=root_t_max,
        x0_search_num=root_grid_num,
        act_eps=act_eps,
        d=d,
    )

    print("\n=== COLLAPSED 2-ReLU NETWORK (at collapse) ===")
    print(f"t_collapse = {t_collapse}")
    print(f"w.shape = {np.asarray(params_collapse_1d.w).shape}")
    print(f"b.shape = {np.asarray(params_collapse_1d.b).shape}")
    print(f"v.shape = {np.asarray(params_collapse_1d.v).shape}")
    print(f"v = {params_collapse_1d.v}")

    for j in range(params_collapse_1d.k):
        wj_str = _format_w_entry(params_collapse_1d.w[j])
        bj = float(np.asarray(params_collapse_1d.b[j]).reshape(-1)[0])
        vj = float(np.asarray(params_collapse_1d.v[j]).reshape(-1)[0])
        print(f"neuron {j}: w = {wj_str}, b = {bj: .6f}, v = {vj: .1f}")

    print("===============================================\n")

    # -----------------------------
    # STEP 5: overlay at collapse time on y-axis
    # -----------------------------
    save_overlay_path = f"debug_projection_with_collapse_overlay_{seed}.png"
    overlay_plot_rich_vs_collapse_on_y_axis(
        params_rich=params_rich,
        params_collapse_1d=params_collapse_1d,
        t_collapse=t_collapse,
        d=d,
        save_path=save_overlay_path,
    )

    # -----------------------------
    # STEP 6: post-collapse training
    # -----------------------------
    col_iters = [t_collapse]

    # collapsed margin at join
    try:
        t0c = find_root_on_grid_1d_fn(
            fn_1d=lambda tt: network_forward(params_collapse_1d, np.asarray(tt, dtype=float).reshape(-1, 1)),
            x_min=root_t_min,
            x_max=root_t_max,
            num=root_grid_num,
        )
        mc0 = geometric_margin_from_root(t0c)
    except RuntimeError:
        mc0 = np.nan
    col_geom_hist = [mc0]

    for t_post in range(1, post_collapse_iters + 1):
        params_rich, loss_r = gd_step_wb_only(params_rich, X_full, y, learning_rate)
        params_collapse_1d, loss_c = gd_step_wb_only(params_collapse_1d, t_axis.reshape(-1, 1), y_axis, learning_rate)

        if t_post % track_every == 0 or t_post == 1:
            t_global = t_collapse + t_post

            # rich y-axis margin
            try:
                t0 = find_root_on_grid_1d_fn(
                    fn_1d=lambda tt: forward_rich_on_y_axis(params_rich, tt, d),
                    x_min=root_t_min,
                    x_max=root_t_max,
                    num=root_grid_num,
                )
                m_y = geometric_margin_from_root(t0)
            except RuntimeError:
                m_y = np.nan

            # rich worst-projection margin
            m_worst = worst_projection_geometric_margin_rich(
                params_rich=params_rich,
                X_train=X_full,
                num_directions=num_directions,
                seed=worst_proj_seed,
                root_grid_num=root_grid_num,
            )

            global_iters.append(t_global)
            rich_y_geom_hist.append(m_y)
            rich_worstproj_hist.append(m_worst)

            # collapsed margin
            try:
                t0c = find_root_on_grid_1d_fn(
                    fn_1d=lambda tt: network_forward(params_collapse_1d, np.asarray(tt, dtype=float).reshape(-1, 1)),
                    x_min=root_t_min,
                    x_max=root_t_max,
                    num=root_grid_num,
                )
                mc = geometric_margin_from_root(t0c)
            except RuntimeError:
                mc = np.nan

            col_iters.append(t_global)
            col_geom_hist.append(mc)

            if t_post % 100_000 == 0:
                print(
                    f"[post t={t_post}] (global={t_global}) loss_r={loss_r:.3e} loss_c={loss_c:.3e} | "
                    f"m_rich_y={m_y:.6f} m_worstproj={m_worst:.6f} m_col={mc:.6f}"
                )

    # -----------------------------
    # STEP 7: plots
    # -----------------------------
    save_margin_path = f"debug_geom_margin_root_distance_{seed}.png"
    plt.figure(figsize=(10, 6))
    plt.plot(global_iters, rich_y_geom_hist, label="Geom margin (rich on y-axis)")
    plt.plot(col_iters, col_geom_hist, label="Geom margin (collapsed)")
    plt.axvline(x=t_collapse, color="red", linestyle="--", linewidth=2, label="collapse join")
    plt.ylim(bottom=0.0)
    plt.xlabel("Training iterations (global)")
    plt.ylabel("margin")
    plt.title("Geometric margin: rich (y-axis) + collapse join")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_margin_path, dpi=200)
    plt.close()
    print("Saved geometric margin plot to:", save_margin_path)

    save_margin_path2 = f"debug_geom_margin_root_distance_with_worst_projection_{seed}.png"
    plt.figure(figsize=(10, 6))
    plt.plot(global_iters, rich_y_geom_hist, label="Geom margin (rich on y-axis)")
    plt.plot(global_iters, rich_worstproj_hist, label="Geom margin (rich, worst over projections)")
    plt.plot(col_iters, col_geom_hist, label="Geom margin (collapsed)")
    plt.axvline(x=t_collapse, color="red", linestyle="--", linewidth=2, label="collapse join")
    plt.ylim(bottom=0.0)
    plt.xlabel("Training iterations (global)")
    plt.ylabel("margin")
    plt.title("Geometric margins: y-axis vs worst-projection (rich) + collapse")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_margin_path2, dpi=200)
    plt.close()
    print("Saved combined margin plot to:", save_margin_path2)

    return {
        "t_collapse": int(t_collapse),
        "params_rich_final": params_rich,
        "params_collapse_final": params_collapse_1d,
        "collapse_info": collapse_info,
        "overlay_plot_path": save_overlay_path,
        "geom_margin_plot_path": save_margin_path,
        "geom_margin_plot_path_worstproj": save_margin_path2,
        "geom_margin_history": {
            "iters_rich": np.array(global_iters, dtype=int),
            "m_rich_y": np.array(rich_y_geom_hist, dtype=float),
            "m_rich_worstproj": np.array(rich_worstproj_hist, dtype=float),
            "iters_collapse": np.array(col_iters, dtype=int),
            "m_collapse": np.array(col_geom_hist, dtype=float),
            "t_collapse": int(t_collapse),
        },
        "train_data": {
            "X_full": X_full,
            "y": y,
            "t_axis": t_axis,
            "y_axis": y_axis,
        },
    }



import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

# assumes these already exist in your package:
# - network_forward, compute_gradients, exponential_loss, NetworkParams
# - sample_points_y_times_sphere (we'll call it with n_points=1000)
# - find_root_on_grid_1d_fn (you already have it)


def _unit_u_diag_d(d: int) -> np.ndarray:
    """u = (1,1,0,...,0)/||.|| in R^d."""
    u = np.zeros(d, dtype=float)
    u[0] = 1.0
    u[1] = 1.0
    u /= (np.linalg.norm(u) + 1e-12)
    return u


def _forward_rich_on_line(params_rich: "NetworkParams", s_grid: np.ndarray, u: np.ndarray) -> np.ndarray:
    """g_big(s) = f_big(s*u)."""
    s_grid = np.asarray(s_grid, dtype=float).reshape(-1)
    X_line = s_grid[:, None] * u[None, :]  # (n,d)
    return network_forward(params_rich, X_line).astype(float).reshape(-1)


def _forward_small2d_on_line(params_small2d: "NetworkParams", s_grid: np.ndarray) -> np.ndarray:
    """g_small(s) = f_small( s*(1,1)/sqrt(2) ) in R^2."""
    s_grid = np.asarray(s_grid, dtype=float).reshape(-1)
    u2 = np.array([1.0, 1.0], dtype=float)
    u2 /= (np.linalg.norm(u2) + 1e-12)
    X_line = s_grid[:, None] * u2[None, :]  # (n,2)
    return network_forward(params_small2d, X_line).astype(float).reshape(-1)


def _signed_margin(params: "NetworkParams", X: np.ndarray, y: np.ndarray) -> float:
    """m = min_i y_i f(X_i)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    f = network_forward(params, X).astype(float).reshape(-1)
    return float(np.min(y * f))


def _gd_step_wb_only(params: "NetworkParams", X: np.ndarray, y: np.ndarray, lr: float):
    """One GD step updating only w,b (v frozen)."""
    grads = compute_gradients(params, X, y)
    new_params = NetworkParams(
        w=params.w - lr * grads.w,
        b=params.b - lr * grads.b,
        v=params.v.copy(),
    )
    loss = float(exponential_loss(y, network_forward(new_params, X)))
    return new_params, loss


def distill_2relu_student_to_target(
    target_fn,          # function: s_grid -> target values
    s_fit: np.ndarray,  # 1D grid
    distill_iters: int = 20000,
    distill_lr: float = 0.01,
) -> Tuple["NetworkParams", Dict]:
    """
    Distill a 2-ReLU 1D student:
        f(s) = ReLU(w1 s + b1) - ReLU(w2 s + b2)
    to match target_fn(s) over s_fit by MSE.
    Returns params with w shape (2,1) so it works with network_forward(..., s.reshape(-1,1)).
    """
    s_fit = np.asarray(s_fit, dtype=float).reshape(-1)
    f_target = target_fn(s_fit).astype(float).reshape(-1)
    invN = 1.0 / float(s_fit.size)

    # simple init
    w1, b1 = 1.0, 0.0
    w2, b2 = -1.0, 0.0

    best = (w1, b1, w2, b2)
    best_mse = float("inf")

    for _ in range(distill_iters):
        z1 = w1 * s_fit + b1
        z2 = w2 * s_fit + b2
        a1 = (z1 > 0).astype(float)
        a2 = (z2 > 0).astype(float)

        f = np.maximum(0.0, z1) - np.maximum(0.0, z2)
        r = f - f_target

        grad_w1 = invN * np.sum(r * (a1 * s_fit))
        grad_b1 = invN * np.sum(r * a1)
        grad_w2 = invN * np.sum(r * (-a2 * s_fit))
        grad_b2 = invN * np.sum(r * (-a2))

        w1 -= distill_lr * grad_w1
        b1 -= distill_lr * grad_b1
        w2 -= distill_lr * grad_w2
        b2 -= distill_lr * grad_b2

        mse = 0.5 * invN * float(np.sum(r ** 2))
        if mse < best_mse:
            best_mse = mse
            best = (w1, b1, w2, b2)

    w1, b1, w2, b2 = best
    params_1d = NetworkParams(
        w=np.array([[w1], [w2]], dtype=float),   # (2,1)
        b=np.array([b1, b2], dtype=float),       # (2,)
        v=np.array([+1.0, -1.0], dtype=float),   # (2,)
    )
    return params_1d, {"best_mse": float(best_mse), "w1": float(w1), "b1": float(b1), "w2": float(w2), "b2": float(b2)}


def embed_1d_student_to_2d(params_1d: "NetworkParams") -> "NetworkParams":
    """
    Make a 2D network that acts on s = <u2, x>, u2=(1,1)/sqrt(2):
        f_2d(x) = f_1d(<u2, x>)
    """
    u2 = np.array([1.0, 1.0], dtype=float)
    u2 /= (np.linalg.norm(u2) + 1e-12)

    W2 = np.zeros((2, 2), dtype=float)
    W2[0, :] = float(params_1d.w[0, 0]) * u2
    W2[1, :] = float(params_1d.w[1, 0]) * u2

    return NetworkParams(
        w=W2,                        # (2,2)
        b=params_1d.b.copy(),        # (2,)
        v=params_1d.v.copy(),        # (2,)
    )


def project_rich_to_x_axis(params_rich: NetworkParams) -> NetworkParams:
    """
    Returns a new NetworkParams where all weights except index 0 are zero.
    This creates a true 1D network depending only on x_1.
    """
    W_proj = params_rich.w.copy()
    W_proj[:, 1:] = 0.0  # zero out coordinates 2..d

    return NetworkParams(
        w=W_proj,
        b=params_rich.b.copy(),
        v=params_rich.v.copy(),
    )


def distill_2relu_with_fixed_root(
    g: Callable[[np.ndarray], np.ndarray],
    t0: float,
    t_grid: np.ndarray,
    *,
    lr: float = 1e-2,
    iters: int = 20000,
    sign_weight: float = 10.0,
    l2_weight: float = 1e-6,
    w1_init: float = 1.0,
    w2_init: float = -1.0,
    eps: float = 1e-12,
    return_column_w: bool = True,
    v: Tuple[float, float] = (1.0, -1.0),
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Student:
        h(t) = ReLU(w1*(t - t0)) - ReLU(w2*(t - t0))

    Constraints:
        1. h(t0) = 0        (hard via parameterization)
        2. h(1) > 0         (soft hinge penalty)
        3. h(-1) < 0        (soft hinge penalty)

    Objective:
        minimize (h(1)-g(1))^2 + (h(-1)-g(-1))^2
    """

    g1 = float(g(np.array([1.0]))[0])
    gm1 = float(g(np.array([-1.0]))[0])

    def relu(z):
        return np.maximum(0.0, z)

    def h_val(w1, w2, t):
        z1 = w1 * (t - t0)
        z2 = w2 * (t - t0)
        return relu(z1) - relu(z2), z1, z2

    w1 = float(w1_init)
    w2 = float(w2_init)

    loss_hist = []

    for _ in range(iters):

        # forward at endpoints
        h1, z11, z21 = h_val(w1, w2, 1.0)
        hm1, z1m, z2m = h_val(w1, w2, -1.0)

        # endpoint fitting loss
        L_fit = (h1 - g1) ** 2 + (hm1 - gm1) ** 2

        # sign hinge penalties
        pen_pos = max(0.0, -h1+0.5)
        pen_neg = max(0.0, hm1+0.5)
        L_sign = sign_weight * (pen_pos ** 2 + pen_neg ** 2)

        # L2 regularization
        L_reg = 0.5 * l2_weight * (w1 ** 2 + w2 ** 2)

        L = L_fit + L_sign + L_reg
        loss_hist.append(L)

        # gradients
        a11 = 1.0 if z11 > 0 else 0.0
        a21 = 1.0 if z21 > 0 else 0.0
        a1m = 1.0 if z1m > 0 else 0.0
        a2m = 1.0 if z2m > 0 else 0.0

        dh1_dw1 = a11 * (1.0 - t0)
        dh1_dw2 = -a21 * (1.0 - t0)
        dhm1_dw1 = a1m * (-1.0 - t0)
        dhm1_dw2 = -a2m * (-1.0 - t0)

        grad_w1 = 2 * (h1 - g1) * dh1_dw1 + 2 * (hm1 - gm1) * dhm1_dw1
        grad_w2 = 2 * (h1 - g1) * dh1_dw2 + 2 * (hm1 - gm1) * dhm1_dw2

        # sign gradients
        if pen_pos > 0:
            grad_w1 += sign_weight * 2 * pen_pos * (-dh1_dw1)
            grad_w2 += sign_weight * 2 * pen_pos * (-dh1_dw2)

        if pen_neg > 0:
            grad_w1 += sign_weight * 2 * pen_neg * dhm1_dw1
            grad_w2 += sign_weight * 2 * pen_neg * dhm1_dw2

        # regularization gradient
        grad_w1 += l2_weight * w1
        grad_w2 += l2_weight * w2

        # update
        w1 -= lr * grad_w1
        w2 -= lr * grad_w2

    # construct final params
    b1 = -w1 * t0
    b2 = -w2 * t0

    w_arr = np.array([[w1], [w2]]) if return_column_w else np.array([w1, w2])
    b_arr = np.array([b1, b2])
    v_arr = np.array([v[0], v[1]])

    params_dict = {"w": w_arr, "b": b_arr, "v": v_arr}

    info = {
        "w1": w1,
        "w2": w2,
        "b1": b1,
        "b2": b2,
        "h(1)": float(h_val(w1, w2, 1.0)[0]),
        "h(-1)": float(h_val(w1, w2, -1.0)[0]),
        "g(1)": g1,
        "g(-1)": gm1,
        "loss_final": loss_hist[-1],
    }

    return params_dict, info


def find_root_on_grid_1d_fn(fn, x_min, x_max, num=4001):
    xg = np.linspace(x_min, x_max, num)
    fg = np.asarray(fn(xg), dtype=float).reshape(-1)

    s = np.sign(fg)
    s[s == 0] = 1.0
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if len(idx) == 0:
        raise RuntimeError(f"No root found in [{x_min},{x_max}]")

    roots = []
    for i in idx:
        x1, x2 = float(xg[i]), float(xg[i+1])
        f1, f2 = float(fg[i]), float(fg[i+1])
        xr = x1 - f1 * (x2 - x1) / (f2 - f1 + 1e-18)  # linear interp
        roots.append(xr)

    roots = np.array(roots, dtype=float)
    return float(roots[np.argmin(np.abs(roots))])  # closest to 0


def experiment_7_big_train_project_distill_small_parallel(
    k: int = 20,
    d: int = 10,
    n_points: int = 1000,
    # seed: int = 43,
    seed: int = 49,
    lr_big: float = 1e-3,
    lr_small: float = 1e-3,
    max_pretrain_iters: int = 10_000,
    pretrain_loss_threshold: float = 1e-6,
    post_iters: int = 200_000,
    track_every: int = 1000,
    root_grid_num: int = 4001,
    distill_s_grid_num: int = 2000,
    distill_iters: int = 30_000,
    distill_lr: float = 0.01,
    save_prefix: str = "experiment_7",
) -> Dict:
    """
    Your requested pipeline:

    1) Train BIG network (k=20,d=10) on n_points (default 1000) until good loss.
    2) Project BIG to 1D line u=(1,1,0,...)/||.|| : g_big(s)=f_big(su) and plot it.
    3) Find SMALL 2-neuron net in R^2 that is very close to g_big(s) (via distillation on s-grid),
       and train it only on two points (1,1) with label +1 and (-1,-1) with label -1.
       Plot g_big and g_small together.
    4) Continue training BIG on its full dataset and SMALL on its 2 points in parallel.
       Track and plot three margins over global iterations:
         - signed margin of BIG on full data: min_i y_i f_big(X_i)
         - geometric projection margin of BIG on u-line: min_i |t0 - <u,X_i>| where g_big(t0)=0
         - geometric margin of SMALL on its line: min(|t0 - sqrt(2)|, |t0 + sqrt(2)|) where g_small(t0)=0
    """

    rng = np.random.default_rng(seed)

    # -----------------------------
    # DATA: big dataset (cluster on sphere radius sqrt(d) in last d-1 coords)
    # -----------------------------
    X_full, y_full = sample_points_y_times_sphere(
        d=d,
        n_points=n_points,
        seed=seed,
        ensure_opposite=False,  # ignored for n_points != 2 in your original; ok if you updated it
        fixed_second_coord=1.4
    )
    X_full = np.asarray(X_full, dtype=float)
    y_full = np.asarray(y_full, dtype=float).reshape(-1)

    print(X_full[:10])
    # -----------------------------
    # INIT BIG network
    # -----------------------------
    # ensure both signs in v
    while True:
        v_big = rng.choice([-1.0, 1.0], size=k).astype(float)
        if np.any(v_big > 0) and np.any(v_big < 0):
            break

    W_big = rng.normal(0.0, np.sqrt(2.0 / float(d)), size=(k, d)).astype(float)
    b_big = np.zeros(k, dtype=float)
    params_big = NetworkParams(w=W_big, b=b_big, v=v_big)

    # -----------------------------
    # PRETRAIN BIG
    # -----------------------------
    t_join = 0
    loss_big = float(exponential_loss(y_full, network_forward(params_big, X_full)))

    for t in range(1, max_pretrain_iters + 1):
        params_big, loss_big = _gd_step_wb_only(params_big, X_full, y_full, lr_big)
        if (t % 10_000) == 0:
            print(f"[pretrain big] t={t} loss={loss_big:.3e}")
        if loss_big < pretrain_loss_threshold:
            t_join = t
            break

    if t_join == 0:
        t_join = max_pretrain_iters
        print(f"[pretrain big] did not reach loss<{pretrain_loss_threshold}; join at t={t_join} (loss={loss_big:.3e})")
    else:
        print(f"[pretrain big] reached loss<{pretrain_loss_threshold} at t={t_join} (loss={loss_big:.3e})")

    # -----------------------------
    # TRUE PROJECTION TO x-AXIS (zero weights from index 1 onward)
    # -----------------------------
    params_big_proj = project_rich_to_x_axis(params_big)

    # training projections are just first coordinate
    s_train = X_full[:, 0].astype(float).reshape(-1)

    # plotting/distillation range
    S = float(np.max(np.abs(s_train)) + 1.0)

    # 1D projected function
    def g_big(ss: np.ndarray) -> np.ndarray:
        ss = np.asarray(ss, dtype=float).reshape(-1)
        X_axis = np.zeros((ss.size, d), dtype=float)
        X_axis[:, 0] = ss
        return network_forward(params_big_proj, X_axis).reshape(-1)
    # -----------------------------

    print("\n--- Active neurons (|w| > 1e-6) ---")

    for j in range(params_big_proj.w.shape[0]):
        w1 = params_big_proj.w[j, 0]
        if abs(w1) > 1e-6:
            b  = params_big_proj.b[j]
            v  = params_big_proj.v[j]
            print(f"Neuron {j}:  v={v:+.4f}   w={w1:+.6f}   b={b:+.6f}")


    # ---------------------------------
    # Plot projected 1D function g_big
    # ---------------------------------
    t_plot = np.linspace(-S, S, 2000)
    g_vals = g_big(t_plot)

    plt.figure(figsize=(8, 5))
    plt.plot(t_plot, g_vals, label="Projected big network (x-axis)")
    plt.axhline(0.0, linestyle="--")
    plt.scatter(s_train, np.zeros_like(s_train), s=10, alpha=0.3, label="train x_1 values")
    plt.xlabel("t = x_1")
    plt.ylabel("g(t)")
    plt.title("Big network projected to x-axis")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("big_vs_2relu.png", dpi=300)
    plt.close()

    # -----------------------------
    # DISTILL SMALL 1D (2-ReLU) to g_big over s-grid, then embed to 2D
    # -----------------------------
    # s_fit = np.linspace(-S, S, distill_s_grid_num)
    # params_small_1d, distill_info = distill_2relu_student_to_target(
    #     target_fn=g_big,
    #     s_fit=s_fit,
    #     distill_iters=distill_iters,
    #     distill_lr=distill_lr,
    # )
    # params_small_2d = embed_1d_student_to_2d(params_small_1d)

    t_grid = np.linspace(-2, 2, 2001)

    t0 = find_root_on_grid_1d_fn(g_big, x_min=-S, x_max=+S, num=8001)
    print("t0 =", t0, "g(t0) =", float(g_big(np.array([t0]))[0]))

    params_small_dict, info = distill_2relu_with_fixed_root(
        g=g_big,
        t0=t0,
        t_grid=t_grid,
        lr=1e-2,
        iters=30000,
        sign_weight=200.0,
        l2_weight=1e-4,
        w1_init=5.0,
        w2_init=-5.0,
        return_column_w=True,
    )

    print(info)
    params_small = NetworkParams(
        w=params_small_dict["w"],
        b=params_small_dict["b"],
        v=params_small_dict["v"],
    )

    
    # -----------------------------
    # SMALL training set: (1,1)->+1, (-1,-1)->-1
    # -----------------------------
    X_small = np.array([[1.0, 1.0], [-1.0, -1.0]], dtype=float)
    y_small = np.array([+1.0, -1.0], dtype=float)

    # direction of the small problem line: (1,1)/sqrt(2)
    u2 = np.array([1.0, 1.0], dtype=float)
    u2 = u2 / np.linalg.norm(u2)

    # unpack 1D student params
    w1 = float(params_small_dict["w"][0, 0])   # because shape (2,1)
    w2 = float(params_small_dict["w"][1, 0])
    b  = params_small_dict["b"].astype(float)  # shape (2,)
    v  = params_small_dict["v"].astype(float)  # shape (2,)

    # lift to 2D: each row is a neuron weight vector in R^2
    W2d = np.vstack([w1 * u2, w2 * u2])        # shape (2,2)

    params_small_2d = NetworkParams(w=W2d, b=b, v=v)

    # -----------------------------
    # PLOT OVERLAY at join time
    # -----------------------------
    s_plot = np.linspace(-S, S, 1200)
    gb = g_big(s_plot)
    gs = _forward_small2d_on_line(params_small_2d, s_plot)

    plt.figure(figsize=(10, 6))
    plt.plot(s_plot, gb, label="g_big(s)")
    plt.plot(s_plot, gs, "--", label="g_small")
    plt.axhline(0.0, linestyle="--")
    plt.scatter([np.sqrt(2), -np.sqrt(2)], [0, 0], s=80, marker="x", label="small train s=±sqrt(2)")
    plt.title(f"Overlay at join (t={t_join})")
    plt.xlabel("s")
    plt.ylabel("value")
    plt.grid(True)
    plt.legend()
    overlay_path = f"{save_prefix}_overlay_join_seed{seed}.png"
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=200)
    plt.close()
    print("Saved:", overlay_path)

    # -----------------------------
    # HELPERS (using your existing functions exactly)
    # -----------------------------
    def _geom_proj_margin_big(params_big_curr) -> float:
        # 1) project current big to x-axis
        params_big_proj_curr = project_rich_to_x_axis(params_big_curr)

        # 2) define g_big on x-axis (same convention you already use)
        def g_big(ss: np.ndarray) -> np.ndarray:
            ss = np.asarray(ss, dtype=float).reshape(-1)
            X_axis = np.zeros((ss.size, d), dtype=float)
            X_axis[:, 0] = ss
            return network_forward(params_big_proj_curr, X_axis).reshape(-1)

        # 3) find root t0 of the projected function
        try:
            t0 = find_root_on_grid_1d_fn(g_big, x_min=-S, x_max=+S, num=root_grid_num)
        except RuntimeError:
            return np.nan

        # 4) geometric margin to nearest projected train point (x1 values)
        return float(np.minimum(np.abs(t0 - 1.0), np.abs(t0 + 1.0)))


    def _geom_margin_small(params_small_2d_curr) -> float:
        # root of g_small(s) on the same s-axis you plot on
        try:
            t0 = find_root_on_grid_1d_fn(
                lambda ss: _forward_small2d_on_line(params_small_2d_curr, ss),
                x_min=-S, x_max=+S, num=root_grid_num
            )
        except RuntimeError:
            return np.nan

        return float(min(abs(t0 - 1), abs(t0 + 1)))
    # -----------------------------
    # POST training in parallel + track margins
    # -----------------------------
    iters = [t_join]

    m_big_proj   = [_geom_proj_margin_big(params_big)]
    m_small_geom = [_geom_margin_small(params_small_2d)]

    for t_post in range(1, post_iters + 1):
        # BIG trains on full dataset only
        params_big, _ = _gd_step_wb_only(params_big, X_full, y_full, lr_big)

        # SMALL trains on its 2 points only
        params_small_2d, _ = _gd_step_wb_only(params_small_2d, X_small, y_small, lr_small)

        if (t_post % track_every) == 0 or t_post == 1:
            t_global = t_join + t_post
            iters.append(t_global)

            m_big_proj.append(_geom_proj_margin_big(params_big))          # ✅ re-project every time
            m_small_geom.append(_geom_margin_small(params_small_2d))      # ✅ just compute

        if (t_post % 100_000) == 0:
            print(f"params_small_2d: {params_small_2d}")
            print(f"[post] t_post={t_post} (global={t_join+t_post})")


    # -----------------------------
    # PLOT MARGINS
    # -----------------------------
    plt.figure(figsize=(10, 6))
    # plt.plot(iters, m_big_signed, label="BIG signed margin on full data: min y f(X)")
    plt.plot(iters, m_big_proj,   label="BIG projection geom margin on x-axis")
    plt.plot(iters, m_small_geom, label="SMALL geom margin on (1,1)/(-1,-1)")
    plt.axvline(x=t_join, color="red", linestyle="--", linewidth=2, label="join (distill time)")
    plt.xlabel("Global iterations")
    plt.ylabel("margin")
    plt.title("Margins: big(full), big(x-axis projection), small(2pts)")
    plt.grid(True)
    plt.ylim(bottom=0, top=1.1)
    plt.legend()
    margins_path = f"{save_prefix}_margins_seed{seed}.png"
    plt.tight_layout()
    plt.savefig(margins_path, dpi=200)
    plt.close()
    print("Saved:", margins_path)

    return {
        "t_join": int(t_join),
        "paths": {"overlay": overlay_path, "margins": margins_path},
        "params_big_final": params_big,
        "params_small_final": params_small_2d,
        "distill_info": info,  # (make sure you named it distill_info above)
        "history": {
            "iters": np.array(iters, dtype=int),
            "m_big_proj": np.array(m_big_proj, dtype=float),
            "m_small_geom": np.array(m_small_geom, dtype=float),
        },
        "data": {
            "X_full": X_full,
            "y_full": y_full,
            "X_small": X_small,
            "y_small": y_small,
            "s_train": s_train,  # projected coords (=x1)
        },
    }