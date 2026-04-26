import numpy as np
import matplotlib.pyplot as plt

import csv

from .core import network_forward, exponential_loss, gradient_descent_step
from .datasets import create_dataset
from .init_utils import initialize_network

def experiment_5f_hit_linear_condition_with_low_loss(
    num_runs: int = 10000,
    max_iterations: int = 10_000_000,
    learning_rate: float = 0.05,
    optimizer_name: str = "gd",
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
            optimizer_state = {} if optimizer_name.upper() == "ADAM" else None

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
                    params,
                    x,
                    y,
                    learning_rate=learning_rate,
                    optimizer_name=optimizer_name,
                    optimizer_state=optimizer_state,
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
        f.write(f"optimizer={optimizer_name.upper()}\n")
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
