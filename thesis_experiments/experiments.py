import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
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


# def experiment_6e_overparam_cluster_then_collapse_compare_margins(
#     k: int = 20,
#     d: int = 50,
#     n: int = 1000,
#     radius: float = 0.5,
#     learning_rate: float = 0.01,
#     max_pre_collapse_iters: int = 1_000_000,
#     post_collapse_iters: int = 1_000_000,
#     activation_frac: float = 0.99,
#     seed: int = 42,
#     track_every: int = 100,
#     save_margins_path: str = "experiment_6e_margins.png",
#     save_neurons_path: str = "experiment_6e_neurons.png",
# ):
#     """
#     Updated Spec (with new dataset generation):

#     1) Init network with k=20 neurons.
#        v_j ∈ {-1,1}, ensure at least one +1 and at least one -1.
#        w_j,b_j ~ N(0,2).

#     2) Data (n points in R^d):
#        For each i:
#          - sample x_raw ~ Unif(B(0, radius)) in R^d
#          - sample y_i ∈ {-1,+1} uniformly
#          - set x_i = x_raw; x_i[0] += y_i
#        NOTE: the current network code is 1D (uses scalar x), so we feed only x_i[0] (projection).

#     3) Train (ONLY w,b; v fixed) until each neuron is active on the correct cluster points
#        (measured by fraction >= activation_frac for each neuron), where:
#          if v_j=+1: active on y=+1 points, inactive on y=-1 points
#          if v_j=-1: active on y=-1 points, inactive on y=+1 points

#     4) Collapse:
#        - group neurons by sign(v): v=+1 group and v=-1 group
#        - average w and b within each group -> 2-neuron network with v=[+1,-1]
#        - collapse data to two instances: (x,y) = (1,1) and (-1,-1)

#     5) Continue training both (rich vs collapsed) and plot margins on same chart.
#        Also saves a chart of (w_j,b_j) for all neurons + collapsed ones and their lines.
#     """

#     rng = np.random.default_rng(seed)

#     # ----------------------------
#     # Helper: sample uniformly from a d-dim ball of radius R
#     # ----------------------------
#     def sample_uniform_ball(rng: np.random.Generator, d: int, R: float) -> np.ndarray:
#         u = rng.normal(size=d)
#         u /= (np.linalg.norm(u) + 1e-12)
#         r = R * (rng.uniform() ** (1.0 / d))
#         return r * u

#     # ----------------------------
#     # Helper: GD step updating only w,b (keep v fixed)
#     # ----------------------------
#     def gd_step_wb_only(params: NetworkParams, x: np.ndarray, y: np.ndarray, lr: float):
#         grads = compute_gradients(params, x, y)
#         new_params = NetworkParams(
#             w=params.w - lr * grads.w,
#             b=params.b - lr * grads.b,
#             v=params.v.copy(),  # freeze v
#         )
#         loss = float(exponential_loss(y, network_forward(new_params, x)))
#         return new_params, loss

#     # ----------------------------
#     # 1) Initialization
#     # ----------------------------
#     while True:
#         v = rng.choice([-1.0, 1.0], size=k)
#         if np.any(v == 1.0) and np.any(v == -1.0):
#             break

#     std = float(np.sqrt(2.0))  # N(0,2) => std=sqrt(2)
#     w = rng.normal(0.0, std, size=k).astype(float)
#     b = rng.normal(0.0, std, size=k).astype(float)

#     params_rich = NetworkParams(w=w, b=b, v=v.astype(float))

#     print("\n=== Initialization (rich network) ===")
#     print(f"k={k}, n={n}, d={d}, radius={radius}, lr={learning_rate}")
#     print(f"#(v=+1)={int((params_rich.v > 0).sum())}, #(v=-1)={int((params_rich.v < 0).sum())}")
#     print("v:", np.array2string(params_rich.v, precision=0))
#     print("w (first 10):", np.array2string(params_rich.w[:10], precision=4))
#     print("b (first 10):", np.array2string(params_rich.b[:10], precision=4))
#     print("====================================\n")

#     # ----------------------------
#     # 2) Data generation (new method), then project to 1D (first coordinate)
#     # ----------------------------
#     X_full = np.zeros((n, d), dtype=float)
#     y_rich = rng.choice([-1.0, 1.0], size=n).astype(float)

#     for i in range(n):
#         x_raw = sample_uniform_ball(rng, d=d, R=radius)
#         x = x_raw.copy()
#         x[0] += y_rich[i]
#         X_full[i] = x

#     # Feed only x1 into the existing 1D network
#     x_rich = X_full[:, 0].astype(float)

#     # ----------------------------
#     # DEBUG: print sampled data points
#     # ----------------------------
#     print("\n=== DEBUG: Sampled data points ===")
#     print(f"n={n}, d={d}, radius={radius}")
#     print("Generation: x_raw ~ Unif(B(0,radius)), y ∈ {-1,+1}, then x[0]+=y")
#     print()

#     print_points_per_cluster = 10

#     neg_idxs = np.where(y_rich < 0)[0]
#     pos_idxs = np.where(y_rich > 0)[0]

#     def show_point(vec: np.ndarray) -> str:
#         head = ", ".join(f"{vec[j]: .4f}" for j in range(min(6, d)))
#         if d > 6:
#             return "[" + head + ", ...]"
#         return "[" + head + "]"

#     print("Negative cluster samples:")
#     for i in range(min(print_points_per_cluster, len(neg_idxs))):
#         idx = int(neg_idxs[i])
#         print(f"  y=-1  x={show_point(X_full[idx])}")

#     print()

#     print("Positive cluster samples:")
#     for i in range(min(print_points_per_cluster, len(pos_idxs))):
#         idx = int(pos_idxs[i])
#         print(f"  y=+1  x={show_point(X_full[idx])}")

#     print("====================================\n")

#     pos_mask = (y_rich > 0)
#     neg_mask = (y_rich < 0)

#     # ----------------------------
#     # 3) Pre-collapse training until each neuron matches activation pattern
#     # ----------------------------
#     def neuron_correct_fraction(params: NetworkParams) -> np.ndarray:
#         pre = params.w[:, None] * x_rich[None, :] + params.b[:, None]  # (k,n)
#         act = (pre > 0)

#         frac = np.zeros(params.k, dtype=float)
#         for j in range(params.k):
#             if params.v[j] > 0:
#                 good_pos = act[j, pos_mask]
#                 good_neg = ~act[j, neg_mask]
#             else:
#                 good_pos = ~act[j, pos_mask]
#                 good_neg = act[j, neg_mask]
#             good = np.concatenate([good_pos, good_neg])
#             frac[j] = float(np.mean(good))
#         return frac

#     t_collapse = None
#     for t in range(max_pre_collapse_iters + 1):
#         frac = neuron_correct_fraction(params_rich)
#         if np.all(frac >= activation_frac):
#             t_collapse = t
#             break

#         params_rich, loss_t = gd_step_wb_only(params_rich, x_rich, y_rich, learning_rate)

#         if t % 10000 == 0:
#             grads_now = compute_gradients(params_rich, x_rich, y_rich)
#             dw_norm = float(np.linalg.norm(grads_now.w))
#             db_norm = float(np.linalg.norm(grads_now.b))
#             print(
#                 f"[pre-collapse] t={t}, loss={loss_t:.6f}, "
#                 f"min_correct_frac={frac.min():.3f}, ||dw||={dw_norm:.3e}, ||db||={db_norm:.3e}"
#             )

#     if t_collapse is None:
#         print("Did not reach per-neuron activation condition within max_pre_collapse_iters.")
#         print("Proceeding to collapse anyway at final params (may be noisy).")
#         t_collapse = max_pre_collapse_iters

#     print(f"Collapse time: t_collapse={t_collapse}")

#     # ----------------------------
#     # 4) Collapse network + data
#     # ----------------------------
#     pos_group = np.where(params_rich.v > 0)[0]
#     neg_group = np.where(params_rich.v < 0)[0]
#     assert len(pos_group) > 0 and len(neg_group) > 0

#     w_pos = float(np.mean(params_rich.w[pos_group]))
#     b_pos = float(np.mean(params_rich.b[pos_group]))
#     w_neg = float(np.mean(params_rich.w[neg_group]))
#     b_neg = float(np.mean(params_rich.b[neg_group]))

#     params_simple = NetworkParams(
#         w=np.array([w_pos, w_neg], dtype=float),
#         b=np.array([b_pos, b_neg], dtype=float),
#         v=np.array([1.0, -1.0], dtype=float),
#     )

#     # Collapsed dataset as two instances: (x,y) = (-1,-1), (1,1)
#     x_simple = np.array([-1.0, 1.0], dtype=float)
#     y_simple = np.array([-1.0, 1.0], dtype=float)

#     # ----------------------------
#     # Save chart: all (w_j, b_j) + collapsed (w,b) and also lines wx+b vs x
#     # ----------------------------
#     x_plot = np.linspace(-1.5, 1.5, 400)

#     plt.figure(figsize=(12, 5))

#     plt.subplot(1, 2, 1)
#     pos_idx = params_rich.v > 0
#     neg_idx = params_rich.v < 0
#     plt.scatter(params_rich.w[pos_idx], params_rich.b[pos_idx], alpha=0.6, label="v=+1 neurons")
#     plt.scatter(params_rich.w[neg_idx], params_rich.b[neg_idx], alpha=0.6, label="v=-1 neurons")
#     plt.scatter([params_simple.w[0]], [params_simple.b[0]], marker="X", s=120, label="collapsed v=+1")
#     plt.scatter([params_simple.w[1]], [params_simple.b[1]], marker="X", s=120, label="collapsed v=-1")
#     plt.title("(w_j, b_j) for all neurons + collapsed")
#     plt.xlabel("w")
#     plt.ylabel("b")
#     plt.grid(True)
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     for j in range(params_rich.k):
#         y_line = params_rich.w[j] * x_plot + params_rich.b[j]
#         plt.plot(x_plot, y_line, alpha=0.25)
#     for j in range(2):
#         y_line = params_simple.w[j] * x_plot + params_simple.b[j]
#         plt.plot(x_plot, y_line, linewidth=3)
#     plt.axhline(0, linestyle="--")
#     plt.title("Lines w x + b (light=20 neurons, bold=collapsed)")
#     plt.xlabel("x")
#     plt.ylabel("w x + b")
#     plt.grid(True)

#     plt.tight_layout()
#     plt.savefig(save_neurons_path, dpi=200)
#     plt.close()
#     print(f"Saved neuron chart to {save_neurons_path}")

#     # ----------------------------
#     # 5) Continue training both, track margins
#     # ----------------------------
#     def safe_margin(params: NetworkParams, x: np.ndarray, y: np.ndarray) -> float:
#         return float(compute_margin(params, x, y))

#     iters = []
#     margins_rich = []
#     margins_simple = []

#     for s in range(post_collapse_iters + 1):
#         if s % track_every == 0:
#             iters.append(s)
#             margins_rich.append(safe_margin(params_rich, x_rich, y_rich))
#             margins_simple.append(safe_margin(params_simple, x_simple, y_simple))

#         if s == post_collapse_iters:
#             break

#         params_rich, _ = gd_step_wb_only(params_rich, x_rich, y_rich, learning_rate)
#         params_simple, _ = gd_step_wb_only(params_simple, x_simple, y_simple, learning_rate)

#     plt.figure(figsize=(10, 6))
#     plt.plot(iters, margins_rich, label="Rich (k=20, full clustered data; x=x1)")
#     plt.plot(iters, margins_simple, label="Collapsed (k=2, data: (-1,-1),(1,1))")
#     plt.xlabel("Post-collapse iteration")
#     plt.ylabel("Margin")
#     plt.title("Margin comparison: rich vs collapsed")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_margins_path, dpi=200)
#     plt.close()
#     print(f"Saved margin chart to {save_margins_path}")

#     return {
#         "t_collapse": int(t_collapse),
#         "iters": np.array(iters, dtype=int),
#         "margins_rich": np.array(margins_rich, dtype=float),
#         "margins_simple": np.array(margins_simple, dtype=float),
#         "neuron_plot": save_neurons_path,
#         "margin_plot": save_margins_path,
#     }

# def experiment_6e_overparam_cluster_then_collapse_compare_margins(
#     d: int = 50,
#     n: int = 1000,
#     radius: float = 0.5,
#     learning_rate: float = 0.01,
#     max_pre_collapse_iters: int = 1_000_000,
#     post_collapse_iters: int = 1_000_000,
#     activation_frac: float = 0.99,
#     seed: int = 42,
#     track_every: int = 100,
#     save_margins_path: str = "experiment_6e_margins.png",
#     save_neurons_path: str = "experiment_6e_neurons.png",
# ):
#     """
#     DEBUG ONLY (updated data generation):

#     Data generation:
#       - sample x_raw ~ Unif(B(0, radius)) in R^d
#       - sample y ∈ {-1,+1} uniform
#       - set x = x_raw; x[0] += y
#       - repeat n times

#     Print:
#       - initial neuron params (w_j,b_j,v_j)
#       - sample data points (full 50D display)
#       - immediate collapse by v sign
#       - collapsed params and collapsed data centers

#     Returns:
#       params_rich, params_simple, X_full, y_full, x_rich, y_rich, x_simple, y_simple
#     """

#     import numpy as np
#     from .core import NetworkParams

#     print_neurons = 20
#     print_points_per_cluster = 10
#     k = 20

#     rng = np.random.default_rng(seed)

#     # ----------------------------
#     # Helper: sample uniformly from a d-dim ball of radius R
#     # ----------------------------
#     def sample_uniform_ball(rng: np.random.Generator, d: int, R: float) -> np.ndarray:
#         # direction uniform on sphere
#         u = rng.normal(size=d)
#         u /= (np.linalg.norm(u) + 1e-12)
#         # radius with correct distribution
#         r = R * (rng.uniform() ** (1.0 / d))
#         return r * u

#     # ----------------------------
#     # 1) Init rich network
#     # ----------------------------
#     while True:
#         v = rng.choice([-1.0, 1.0], size=k)
#         if np.any(v == 1.0) and np.any(v == -1.0):
#             break

#     std = float(np.sqrt(2.0))
#     w = rng.normal(0.0, std, size=k).astype(float)
#     b = rng.normal(0.0, std, size=k).astype(float)

#     params_rich = NetworkParams(w=w, b=b, v=v.astype(float))

#     # Print neuron params
#     print("\n=== DEBUG: Initial neuron parameters (rich) ===")
#     print(f"k={k}, seed={seed}")
#     print(f"#(v=+1)={int((params_rich.v > 0).sum())}, #(v=-1)={int((params_rich.v < 0).sum())}")
#     m = min(print_neurons, k)
#     for j in range(m):
#         print(f"j={j:02d}: v={int(params_rich.v[j]):+d}, w={params_rich.w[j]: .6f}, b={params_rich.b[j]: .6f}")
#     if m < k:
#         print(f"... (printed first {m} of {k})")
#     print("=============================================\n")

#     # ----------------------------
#     # 2) Generate data using your new method
#     # ----------------------------
#     X_full = np.zeros((n, d), dtype=float)
#     y_full = rng.choice([-1.0, 1.0], size=n).astype(float)

#     for i in range(n):
#         x_raw = sample_uniform_ball(rng, d=d, R=radius)   # Unif ball
#         x = x_raw.copy()
#         x[0] += y_full[i]                                # shift first coord by ±1
#         X_full[i] = x

#     # For the existing 1D network: take first coordinate only
#     x_rich = X_full[:, 0].astype(float)
#     y_rich = y_full.astype(float)

#     # Print sample data points
#     print("\n=== DEBUG: Sample data points (full 50D display) ===")
#     print(f"n={n}, d={d}, radius={radius}")
#     print("Generation: x_raw ~ Unif(B(0,radius)), y ∈ {-1,+1}, then x[0]+=y")
#     print(f"Showing up to {print_points_per_cluster} points from each label:\n")

#     neg_idxs = np.where(y_rich < 0)[0]
#     pos_idxs = np.where(y_rich > 0)[0]

#     def show_point(vec: np.ndarray) -> str:
#         # show first few coords + indicate zeros beyond
#         # (not guaranteed zeros now, because x_raw has all coords)
#         head = ", ".join(f"{vec[j]: .4f}" for j in range(min(6, d)))
#         if d > 6:
#             return "[" + head + ", ...]"
#         return "[" + head + "]"

#     for i in range(min(print_points_per_cluster, len(neg_idxs))):
#         idx = int(neg_idxs[i])
#         print(f"neg[{i:02d}] y=-1  x={show_point(X_full[idx])}")

#     print()

#     for i in range(min(print_points_per_cluster, len(pos_idxs))):
#         idx = int(pos_idxs[i])
#         print(f"pos[{i:02d}] y=+1  x={show_point(X_full[idx])}")

#     print("=============================================\n")

#     # ----------------------------
#     # 3) Immediate collapse (by sign of v)
#     # ----------------------------
#     pos_group = np.where(params_rich.v > 0)[0]
#     neg_group = np.where(params_rich.v < 0)[0]
#     assert len(pos_group) > 0 and len(neg_group) > 0

#     w_pos = float(np.mean(params_rich.w[pos_group]))
#     b_pos = float(np.mean(params_rich.b[pos_group]))
#     w_neg = float(np.mean(params_rich.w[neg_group]))
#     b_neg = float(np.mean(params_rich.b[neg_group]))

#     params_simple = NetworkParams(
#         w=np.array([w_pos, w_neg], dtype=float),
#         b=np.array([b_pos, b_neg], dtype=float),
#         v=np.array([1.0, -1.0], dtype=float),
#     )

#     # Collapsed data to centers
#     x_simple = np.array([-1.0, 1.0], dtype=float)
#     y_simple = np.array([-1.0, 1.0], dtype=float)

#     # Print collapsed params + collapsed data
#     print("\n=== DEBUG: Collapsed network parameters (simple) ===")
#     print("Neuron 0 (v=+1):")
#     print(f"  w={params_simple.w[0]: .6f}, b={params_simple.b[0]: .6f}")
#     print("Neuron 1 (v=-1):")
#     print(f"  w={params_simple.w[1]: .6f}, b={params_simple.b[1]: .6f}")
#     print("v:", params_simple.v)
#     print("===============================================\n")

#     print("\n=== DEBUG: Collapsed data (centers, 50D meaning) ===")
#     print("neg center y=-1: [-1, 0, 0, ..., 0]")
#     print("pos center y=+1: [ 1, 0, 0, ..., 0]")
#     print("===============================================\n")

#     return params_rich, params_simple, X_full, y_full, x_rich, y_rich, x_simple, y_simple

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
):
    """
    Experiment 6e (rewritten):

    Data generation (R^d):
      For each point:
        x_raw ~ Unif(B(0, radius)) in R^d
        y ∈ {-1,+1} uniform
        x = x_raw; x[0] += y
      Train the 1D network on x1 = x[0].

    Pre-collapse:
      Train only (w,b); keep v fixed.
      Stop when loss < collapse_loss_threshold (or after max_pre_collapse_iters).

    Collapse (functional grouping):
      Compute m_j^+ = mean(w_j x + b_j | y=+1), m_j^- analog.
      group+ if m_j^+ > m_j^- else group-.

      We PRINT collapsed (w,b) in two ways:
        (A) using ALL neurons (including dead)
        (B) using ONLY alive neurons (ignoring dead), where "dead" means never active on dataset.

      Then we USE (B) as params_simple.

      Collapsed dataset: (-1,-1) and (1,1).

    Tracking:
      Track rich margin pre+post; collapsed margin is NaN pre, tracked post.
      Plot margins with a red vertical line at collapse time.

    Extras:
      - Print neurons at init and at collapse
      - Print sampled points
      - Plot f_rich(x) vs f_collapsed(x) at collapse
      - Plot f_rich(x) at collapse without collapsing neurons + f(-1), f(1), zero crossings
    """

    import numpy as np
    import matplotlib.pyplot as plt

    from .core import NetworkParams, network_forward, exponential_loss, compute_gradients
    from .metrics import compute_margin

    rng = np.random.default_rng(seed)

    # -------------------------------------------------
    # Helper: sample uniformly from d-ball radius R
    # -------------------------------------------------
    def sample_uniform_ball(d_, R_):
        u = rng.normal(size=d_)
        u /= (np.linalg.norm(u) + 1e-12)
        r = R_ * (rng.uniform() ** (1.0 / d_))
        return r * u

    # -------------------------------------------------
    # Helper: GD step updating only w,b (freeze v)
    # -------------------------------------------------
    def gd_step_wb_only(params: NetworkParams, x: np.ndarray, y: np.ndarray, lr: float):
        grads = compute_gradients(params, x, y)
        new_params = NetworkParams(
            w=params.w - lr * grads.w,
            b=params.b - lr * grads.b,
            v=params.v.copy(),
        )
        loss = float(exponential_loss(y, network_forward(new_params, x)))
        return new_params, loss

    # -------------------------------------------------
    # Pretty printing helpers
    # -------------------------------------------------
    def print_neuron_params(title: str, params: NetworkParams, max_print: int):
        print(f"\n=== {title} ===")
        print(f"k={params.k}")
        m = min(max_print, params.k)
        for j in range(m):
            print(
                f"j={j:02d}: v={int(params.v[j]):+d}, "
                f"w={float(params.w[j]): .6f}, b={float(params.b[j]): .6f}"
            )
        if m < params.k:
            print(f"... (printed first {m} of {params.k})")
        print("=========================\n")

    def show_point(vec: np.ndarray) -> str:
        head = ", ".join(f"{vec[j]: .4f}" for j in range(min(6, vec.shape[0])))
        if vec.shape[0] > 6:
            return "[" + head + ", ...]"
        return "[" + head + "]"

    def print_sampled_points(X_full: np.ndarray, y: np.ndarray, title: str):
        print(f"\n=== {title} ===")
        print(f"n={X_full.shape[0]}, d={X_full.shape[1]}, radius={radius}")
        print("Generation: x_raw ~ Unif(B(0,radius)), y ∈ {-1,+1}, then x[0]+=y\n")

        neg_idxs = np.where(y < 0)[0]
        pos_idxs = np.where(y > 0)[0]

        print("Negative label samples (y=-1):")
        for i in range(min(print_points_per_cluster, len(neg_idxs))):
            idx = int(neg_idxs[i])
            print(f"  {i:02d}: x={show_point(X_full[idx])}")

        print("\nPositive label samples (y=+1):")
        for i in range(min(print_points_per_cluster, len(pos_idxs))):
            idx = int(pos_idxs[i])
            print(f"  {i:02d}: x={show_point(X_full[idx])}")

        print("=========================\n")

    # -------------------------------------------------
    # Rich-only diagnostic plot + center predictions + zero crossings
    # -------------------------------------------------
    def count_zero_crossings(params: NetworkParams, x_min=-2.0, x_max=2.0, num=5000) -> int:
        xg = np.linspace(x_min, x_max, num)
        fg = network_forward(params, xg)
        s = np.sign(fg)
        s[s == 0] = 1
        return int(np.sum(s[:-1] * s[1:] < 0))

    def plot_rich_only_1d(params: NetworkParams, save_path: str):
        xg = np.linspace(-2.0, 2.0, 800)
        fg = network_forward(params, xg)

        f_m1 = float(network_forward(params, np.array([-1.0]))[0])
        f_p1 = float(network_forward(params, np.array([1.0]))[0])
        crossings = count_zero_crossings(params)

        plt.figure(figsize=(10, 6))
        plt.plot(xg, fg, label="f_rich(x) (k=20)")
        plt.axhline(0.0, linestyle="--")
        plt.scatter([-1.0, 1.0], [f_m1, f_p1], s=80, marker="X", label="centers x=-1, x=1")
        plt.xlabel("x (1D projection: first coordinate)")
        plt.ylabel("f(x)")
        plt.title(f"Rich network (no neuron collapse) at collapse time | zero-crossings={crossings}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"Saved rich-only plot: {save_path}")
        print(f"f_rich(-1)={f_m1:.6f}, f_rich(1)={f_p1:.6f}, zero_crossings={crossings}")

    # -------------------------------------------------
    # 1) Initialization (STANDARD He init for ReLU, 1D)
    #   - w ~ N(0, 2/fan_in) with fan_in=1 => std=sqrt(2)
    #   - b = 0 (standard practice)
    # -------------------------------------------------
    while True:
        v = rng.choice([-1.0, 1.0], size=k)
        if np.any(v == 1.0) and np.any(v == -1.0):
            break

    fan_in = 1
    w_std = float(np.sqrt(2.0 / fan_in))
    w = rng.normal(0.0, w_std, size=k).astype(float)
    b = np.zeros(k, dtype=float)

    params_rich = NetworkParams(w=w, b=b, v=v.astype(float))

    print_neuron_params("Initialization (rich network) [He init]", params_rich, print_neurons)

    # -------------------------------------------------
    # 2) Data generation (new method) + print sampled points
    # -------------------------------------------------
    X_full = np.zeros((n, d), dtype=float)
    y_rich = rng.choice([-1.0, 1.0], size=n).astype(float)

    for i in range(n):
        x_raw = sample_uniform_ball(d, radius)
        x = x_raw.copy()
        x[0] += y_rich[i]
        X_full[i] = x

    print_sampled_points(X_full, y_rich, "Sampled data points (full 50D)")

    # Train using projection x1 only (current architecture is 1D)
    x_rich = X_full[:, 0].astype(float)

    # -------------------------------------------------
    # 3) Pre-collapse: train until loss small
    # -------------------------------------------------
    t_all = [0]
    margin_rich_all = [float(compute_margin(params_rich, x_rich, y_rich))]
    margin_simple_all = [float("nan")]

    t_collapse = None
    loss_at_collapse = None

    for t in range(1, max_pre_collapse_iters + 1):
        params_rich, loss = gd_step_wb_only(params_rich, x_rich, y_rich, learning_rate)

        if t % track_every == 0:
            t_all.append(t)
            margin_rich_all.append(float(compute_margin(params_rich, x_rich, y_rich)))
            margin_simple_all.append(float("nan"))

        if t % 10000 == 0:
            print(f"[pre-collapse] t={t}, loss={loss:.6e}")

        if loss < collapse_loss_threshold:
            t_collapse = t
            loss_at_collapse = loss
            break

    if t_collapse is None:
        t_collapse = max_pre_collapse_iters
        loss_at_collapse = float(exponential_loss(y_rich, network_forward(params_rich, x_rich)))
        print("WARNING: Did not reach collapse_loss_threshold within max_pre_collapse_iters.")

    print(f"\n*** COLLAPSE at t={t_collapse} (loss={loss_at_collapse:.6e}) ***\n")

    # Print rich params at collapse
    print_neuron_params("Rich network parameters at collapse", params_rich, print_neurons)

    # Rich-only plot (no neuron collapse) at collapse time
    plot_rich_only_1d(params_rich, save_rich_1d_path)

    # -------------------------------------------------
    # 4) Functional collapse (compare with/without dead removal)
    # -------------------------------------------------
    pos_mask = (y_rich > 0)
    neg_mask = (y_rich < 0)

    pre = params_rich.w[:, None] * x_rich[None, :] + params_rich.b[:, None]  # (k,n)

    # Dead neuron = never active on dataset (pre <= 0 everywhere)
    alive_mask = np.any(pre > 0.0, axis=1)
    alive_idxs = np.where(alive_mask)[0]
    dead_idxs = np.where(~alive_mask)[0]

    print(f"Dead neurons: {len(dead_idxs)} / {params_rich.k} (alive={len(alive_idxs)})")

    # --------------------------
    # (A) Collapse using ALL neurons
    # --------------------------
    m_plus_all = pre[:, pos_mask].mean(axis=1)
    m_minus_all = pre[:, neg_mask].mean(axis=1)

    group_plus_all = np.where(m_plus_all > m_minus_all)[0]
    group_minus_all = np.where(m_plus_all <= m_minus_all)[0]

    if len(group_plus_all) == 0 or len(group_minus_all) == 0:
        print("Degenerate grouping (all) — fallback split by sign(v).")
        group_plus_all = np.where(params_rich.v > 0)[0]
        group_minus_all = np.where(params_rich.v < 0)[0]

    w_pos_all = float(params_rich.w[group_plus_all].mean())
    b_pos_all = float(params_rich.b[group_plus_all].mean())
    w_neg_all = float(params_rich.w[group_minus_all].mean())
    b_neg_all = float(params_rich.b[group_minus_all].mean())

    print("\n=== Collapse WITHOUT removing dead neurons ===")
    print(f"group_plus_all size  = {len(group_plus_all)}")
    print(f"group_minus_all size = {len(group_minus_all)}")
    print(f"(v=+1) w={w_pos_all:.6f}, b={b_pos_all:.6f}")
    print(f"(v=-1) w={w_neg_all:.6f}, b={b_neg_all:.6f}")

    # --------------------------
    # (B) Collapse using ONLY alive neurons (ignore dead)
    # --------------------------
    if len(alive_idxs) == 0:
        raise RuntimeError("All neurons are dead on the dataset — cannot collapse.")

    pre_alive = pre[alive_idxs, :]
    m_plus_alive = pre_alive[:, pos_mask].mean(axis=1)
    m_minus_alive = pre_alive[:, neg_mask].mean(axis=1)

    group_plus_alive = alive_idxs[np.where(m_plus_alive > m_minus_alive)[0]]
    group_minus_alive = alive_idxs[np.where(m_plus_alive <= m_minus_alive)[0]]

    if len(group_plus_alive) == 0 or len(group_minus_alive) == 0:
        print("Degenerate grouping (alive) — fallback split by sign(v) among alive.")
        group_plus_alive = alive_idxs[np.where(params_rich.v[alive_idxs] > 0)[0]]
        group_minus_alive = alive_idxs[np.where(params_rich.v[alive_idxs] < 0)[0]]

    if len(group_plus_alive) == 0 or len(group_minus_alive) == 0:
        print("Still degenerate (alive) — splitting alive set in half as last resort.")
        mid = len(alive_idxs) // 2
        group_plus_alive = alive_idxs[:mid]
        group_minus_alive = alive_idxs[mid:]

    if len(group_plus_alive) == 0 or len(group_minus_alive) == 0:
        raise RuntimeError("Collapse failed: empty group after fallbacks.")

    w_pos_alive = float(params_rich.w[group_plus_alive].mean())
    b_pos_alive = float(params_rich.b[group_plus_alive].mean())
    w_neg_alive = float(params_rich.w[group_minus_alive].mean())
    b_neg_alive = float(params_rich.b[group_minus_alive].mean())

    print("\n=== Collapse WITH removing dead neurons ===")
    print(f"group_plus_alive size  = {len(group_plus_alive)}")
    print(f"group_minus_alive size = {len(group_minus_alive)}")
    print(f"(v=+1) w={w_pos_alive:.6f}, b={b_pos_alive:.6f}")
    print(f"(v=-1) w={w_neg_alive:.6f}, b={b_neg_alive:.6f}")

    # Use alive-only collapse as the actual collapsed model
    params_simple = NetworkParams(
        w=np.array([w_pos_alive, w_neg_alive], dtype=float),
        b=np.array([b_pos_alive, b_neg_alive], dtype=float),
        v=np.array([1.0, -1.0], dtype=float),
    )

    print_neuron_params("Collapsed (simple) network parameters (alive-only)", params_simple, max_print=2)

    # Collapsed dataset as pairs (-1,-1), (1,1)
    x_simple = np.array([-1.0, 1.0], dtype=float)
    y_simple = np.array([-1.0, 1.0], dtype=float)

    print("\n=== Collapsed dataset (as pairs) ===")
    print("(-1,-1)")
    print("( 1, 1)")
    print("===================================\n")

    # -------------------------------------------------
    # 5) Plot network functions at collapse: rich vs collapsed
    # -------------------------------------------------
    x_plot = np.linspace(-2.0, 2.0, 600)
    f_rich = network_forward(params_rich, x_plot)
    f_simple = network_forward(params_simple, x_plot)

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, f_rich, label="Rich f(x) at collapse (k=20)")
    plt.plot(x_plot, f_simple, label="Collapsed f(x) at collapse (k=2)")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("x (1D projection: first coordinate)")
    plt.ylabel("f(x)")
    plt.title("Network function at collapse: rich vs collapsed")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_function_at_collapse_path, dpi=200)
    plt.close()
    print(f"Saved function-at-collapse plot to {save_function_at_collapse_path}")

    # Ensure we have an entry at t_collapse and fill collapsed margin there
    if t_all[-1] != t_collapse:
        t_all.append(t_collapse)
        margin_rich_all.append(float(compute_margin(params_rich, x_rich, y_rich)))
        margin_simple_all.append(float(compute_margin(params_simple, x_simple, y_simple)))
    else:
        margin_simple_all[-1] = float(compute_margin(params_simple, x_simple, y_simple))

    # -------------------------------------------------
    # 6) Post-collapse training + margin tracking
    # -------------------------------------------------
    for s in range(1, post_collapse_iters + 1):
        params_rich, _ = gd_step_wb_only(params_rich, x_rich, y_rich, learning_rate)
        params_simple, _ = gd_step_wb_only(params_simple, x_simple, y_simple, learning_rate)

        if s % track_every == 0:
            t_global = t_collapse + s
            t_all.append(t_global)
            margin_rich_all.append(float(compute_margin(params_rich, x_rich, y_rich)))
            margin_simple_all.append(float(compute_margin(params_simple, x_simple, y_simple)))

    # -------------------------------------------------
    # Plot margins pre+post with red vertical line at collapse time
    # -------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(t_all, margin_rich_all, label="Rich margin (k=20)")
    plt.plot(t_all, margin_simple_all, label="Collapsed margin (k=2)")
    plt.axvline(t_collapse, color="red", linewidth=2, label="Collapse time")
    plt.xlabel("Iteration")
    plt.ylabel("Margin")
    plt.title("Margins: pre+post collapse (red line = collapse time)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_margins_path, dpi=200)
    plt.close()
    print(f"Saved margins plot to {save_margins_path}")

    return {
        "t_collapse": int(t_collapse),
        "loss_at_collapse": float(loss_at_collapse),
        "t_all": np.array(t_all, dtype=int),
        "margin_rich_all": np.array(margin_rich_all, dtype=float),
        "margin_simple_all": np.array(margin_simple_all, dtype=float),
        "save_margins_path": save_margins_path,
        "save_function_at_collapse_path": save_function_at_collapse_path,
        "save_rich_1d_path": save_rich_1d_path,
        "group_plus_size": int(len(group_plus_alive)),
        "group_minus_size": int(len(group_minus_alive)),
        "dead_neurons": int(len(dead_idxs)),
        "alive_neurons": int(len(alive_idxs)),
        "collapse_all": {
            "w_pos": float(w_pos_all),
            "b_pos": float(b_pos_all),
            "w_neg": float(w_neg_all),
            "b_neg": float(b_neg_all),
            "group_plus_size": int(len(group_plus_all)),
            "group_minus_size": int(len(group_minus_all)),
        },
        "collapse_alive": {
            "w_pos": float(w_pos_alive),
            "b_pos": float(b_pos_alive),
            "w_neg": float(w_neg_alive),
            "b_neg": float(b_neg_alive),
            "group_plus_size": int(len(group_plus_alive)),
            "group_minus_size": int(len(group_minus_alive)),
        },
    }