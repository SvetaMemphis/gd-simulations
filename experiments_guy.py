import argparse
import csv
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import torch


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TwoLayerFrozenSecondLayer(torch.nn.Module):
    """
    2D -> 2 hidden ReLU units -> 1 output.
    Only first layer is trainable.
    """

    def __init__(self, first_layer_weight: torch.Tensor, first_layer_bias: torch.Tensor, device: torch.device):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 2, bias=True, device=device)
        with torch.no_grad():
            self.fc1.weight.copy_(first_layer_weight)
            self.fc1.bias.copy_(first_layer_bias)
        self.fc2 = torch.nn.Linear(2, 1, bias=False, device=device)
        with torch.no_grad():
            self.fc2.weight.copy_(torch.tensor([[1.0, -1.0]], dtype=torch.float32, device=device))
        self.fc2.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        return self.fc2(h).squeeze(-1)


def exponential_loss(y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return torch.exp(-y * logits).mean()


@dataclass
class RunResult:
    stop_reason: str
    t_last: int
    t_hit: int
    w1_0_eff: float
    b1_0: float
    w2_0_eff: float
    b2_0: float
    w1_T_eff: float
    b1_T: float
    w2_T_eff: float
    b2_T: float
    loss_last: float
    expr_last: float
    metric_min: str


def run_experiment(
    num_runs: int = 10_000,
    max_iterations: int = 10_000_000,
    loss_abort_iteration: int = 10_000,
    learning_rate: float = 0.01,
    optimizer_name: str = "GD",
    tol: float = 1e-3,
    loss_threshold: float = 0.5,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> None:
    device = device or get_device("auto")
    torch.manual_seed(seed)

    # Same two-point structure but in 2D.
    x = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float32, device=device)
    y = torch.tensor([-1.0, 1.0], dtype=torch.float32, device=device)

    hit_times = torch.full((num_runs,), -1, dtype=torch.int64, device=device)
    metric_values: list[float] = []
    results: list[RunResult] = []

    count_hit = 0
    count_loss_abort = 0
    count_max_iterations = 0

    for r in range(num_runs):
        print(f"Run: {r}")

        # Initialize "same way": Gaussian with variance 2 and zero biases.
        # We initialize effective scalar weights and replicate on two input dimensions.
        w1_0_eff = float(torch.normal(mean=0.0, std=torch.sqrt(torch.tensor(2.0, device=device))).item())
        w2_0_eff = float(torch.normal(mean=0.0, std=torch.sqrt(torch.tensor(2.0, device=device))).item())
        b1_0 = 0.0
        b2_0 = 0.0

        first_w = torch.tensor(
            [[w1_0_eff / 2.0, w1_0_eff / 2.0], [w2_0_eff / 2.0, w2_0_eff / 2.0]],
            dtype=torch.float32,
            device=device,
        )
        first_b = torch.tensor([b1_0, b2_0], dtype=torch.float32, device=device)

        model = TwoLayerFrozenSecondLayer(first_w, first_b, device=device)

        if optimizer_name in ("GD", "SGD"):
            optimizer = torch.optim.SGD(model.fc1.parameters(), lr=learning_rate)
        elif optimizer_name == "ADAM":
            optimizer = torch.optim.Adam(model.fc1.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        stop_reason = "max-iterations"
        loss_t = float("nan")
        expr_t = float("nan")
        t = 0

        while t <= max_iterations:
            with torch.no_grad():
                logits = model(x)
                loss_t = float(exponential_loss(y, logits).item())
                w_eff = model.fc1.weight.sum(dim=1)
                expr_t = float(torch.abs(w_eff[0] + model.fc1.bias[0] + w_eff[1] - model.fc1.bias[1]).item())

            if t == loss_abort_iteration and not (loss_t < loss_threshold):
                stop_reason = "loss-abort"
                count_loss_abort += 1
                break

            if loss_t < loss_threshold and expr_t < tol:
                hit_times[r] = t
                stop_reason = "hit condition"
                count_hit += 1
                metric_min = min(
                    abs(float(model.fc1.bias[1].item()) - float(model.fc1.bias[0].item())),
                    abs(w1_0_eff + w2_0_eff) / 2.0,
                )
                metric_values.append(metric_min)
                break

            if t == max_iterations:
                stop_reason = "max-iterations"
                count_max_iterations += 1
                break

            optimizer.zero_grad()
            if optimizer_name == "SGD":
                idx = torch.randint(0, x.size(0), (1,), device=device)
                x_batch = x[idx]
                y_batch = y[idx]
            else:
                x_batch = x
                y_batch = y
            train_logits = model(x_batch)
            train_loss = exponential_loss(y_batch, train_logits)
            train_loss.backward()
            optimizer.step()
            t += 1

        with torch.no_grad():
            w_eff_T = model.fc1.weight.sum(dim=1)
            w1_T_eff = float(w_eff_T[0].item())
            w2_T_eff = float(w_eff_T[1].item())
            b1_T = float(model.fc1.bias[0].item())
            b2_T = float(model.fc1.bias[1].item())

        metric_val = ""
        if stop_reason == "hit condition":
            metric_val = f"{metric_values[-1]}"

        results.append(
            RunResult(
                stop_reason=stop_reason,
                t_last=t,
                t_hit=int(hit_times[r].item()),
                w1_0_eff=w1_0_eff,
                b1_0=b1_0,
                w2_0_eff=w2_0_eff,
                b2_0=b2_0,
                w1_T_eff=w1_T_eff,
                b1_T=b1_T,
                w2_T_eff=w2_T_eff,
                b2_T=b2_T,
                loss_last=loss_t,
                expr_last=expr_t,
                metric_min=metric_val,
            )
        )

    assert count_hit + count_loss_abort + count_max_iterations == num_runs

    runs_csv_path = "experiment_guy_runs.csv"
    summary_txt_path = "experiment_guy_summary.txt"
    hist1_csv_path = "experiment_guy_hit_time_hist.csv"
    hist2_csv_path = "experiment_guy_metric_hist.csv"

    with open(runs_csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "run",
                "stop_reason",
                "t_last",
                "t_hit",
                "w1_0_eff",
                "b1_0",
                "w2_0_eff",
                "b2_0",
                "w1_T_eff",
                "b1_T",
                "w2_T_eff",
                "b2_T",
                "loss_last",
                "expr_last",
                "metric_min",
            ],
        )
        writer.writeheader()
        for i, row in enumerate(results):
            writer.writerow({"run": i, **row.__dict__})

    successful_hits = hit_times[hit_times >= 0].detach().cpu()
    hist1_counts = None
    hist1_edges = None
    if successful_hits.numel() > 0:
        hist1_counts, hist1_edges = torch.histogram(successful_hits.to(torch.float32), bins=40)
        plt.figure(figsize=(8, 5))
        plt.hist(successful_hits.numpy(), bins=40)
        plt.title("Hit time histogram (experiments_guy)")
        plt.xlabel("Iterations")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("experiment_guy_hit_time_hist.png", dpi=200)
        plt.close()

        with open(hist1_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bin_left", "bin_right", "count"])
            for i in range(len(hist1_counts)):
                writer.writerow([float(hist1_edges[i]), float(hist1_edges[i + 1]), int(hist1_counts[i])])

    metric_arr = torch.tensor(metric_values, dtype=torch.float32)
    hist2_counts = None
    hist2_edges = None
    if metric_arr.numel() > 0:
        hist2_counts, hist2_edges = torch.histogram(metric_arr, bins=40)
        plt.figure(figsize=(8, 5))
        plt.hist(metric_arr.numpy(), bins=40)
        plt.title("Histogram of min(|b2-b1|, |w1^0+w2^0|/2) (experiments_guy)")
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("experiment_guy_metric_hist.png", dpi=200)
        plt.close()

        with open(hist2_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bin_left", "bin_right", "count"])
            for i in range(len(hist2_counts)):
                writer.writerow([float(hist2_edges[i]), float(hist2_edges[i + 1]), int(hist2_counts[i])])

    with open(summary_txt_path, "w") as f:
        f.write("=== Experiment guy summary ===\n")
        f.write(f"num_runs={num_runs}\n")
        f.write(f"max_iterations={max_iterations}\n")
        f.write(f"loss_abort_iteration={loss_abort_iteration}\n")
        f.write(f"learning_rate={learning_rate}\n")
        f.write(f"optimizer={optimizer_name}\n")
        f.write(f"tol={tol}, loss_threshold={loss_threshold}\n")
        f.write(f"device={device}\n\n")
        f.write(f"Hit condition: {count_hit}/{num_runs}\n")
        f.write(f"Failed (loss-abort): {count_loss_abort}/{num_runs}\n")
        f.write(f"Failed (max-iterations): {count_max_iterations}/{num_runs}\n")

    print("\n=== Experiment guy summary ===")
    print(f"Device: {device}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Hit condition: {count_hit}/{num_runs}")
    print(f"Failed (loss-abort): {count_loss_abort}/{num_runs}")
    print(f"Failed (max-iterations): {count_max_iterations}/{num_runs}")
    print("Files written:")
    print(" - experiment_guy_runs.csv")
    print(" - experiment_guy_summary.txt")
    if hist1_counts is not None:
        print(" - experiment_guy_hit_time_hist.png")
        print(" - experiment_guy_hit_time_hist.csv")
    if hist2_counts is not None:
        print(" - experiment_guy_metric_hist.png")
        print(" - experiment_guy_metric_hist.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyTorch-compatible 2D two-layer experiment with frozen second layer."
    )
    parser.add_argument("--iterations", "-i", type=int, help="Number of GD iterations (unused)")
    parser.add_argument("--lr", "--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--runs", type=int, help="Number of runs")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--max-iterations", type=int, help="Max iterations")
    parser.add_argument(
        "--loss-abort-iteration",
        type=int,
        default=10_000,
        help="Abort a run at this iteration if loss is still above threshold",
    )
    parser.add_argument(
        "--optimizer",
        choices=["GD", "SGD", "ADAM", "gd", "sgd", "adam"],
        default="GD",
        help="Optimizer type: GD, SGD, or ADAM",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device (default: auto, prefers cuda when available)",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")
    optimizer_name = args.optimizer.upper()

    run_experiment(
        num_runs=args.runs if args.runs is not None else 10_000,
        max_iterations=args.max_iterations if args.max_iterations is not None else 10_000_000,
        loss_abort_iteration=args.loss_abort_iteration,
        learning_rate=args.lr if args.lr is not None else 0.01,
        optimizer_name=optimizer_name,
        seed=args.seed if args.seed else 42,
        device=device,
    )


if __name__ == "__main__":
    main()
