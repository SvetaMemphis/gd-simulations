import argparse
import sys
from typing import List, Optional, Union


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Thesis Experiments: GD Convergence and Adversarial Robustness",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )


    # Common parameters
    parser.add_argument("--iterations", "-i", type=int, help="Number of GD iterations")
    parser.add_argument("--lr", "--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--runs", type=int, help="Number of runs")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--max-iterations", type=int, help="Max iterations")
    parser.add_argument(
        "--optimizer",
        choices=["gd", "adam", "GD", "ADAM"],
        default="gd",
        help="Optimizer to use: gd or adam",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Adam beta1 (momentum decay)",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Adam beta2 (RMS decay)",
    )
  
    args = parser.parse_args(argv)
    print(args)

    # Delayed import so `--help`/`--list-experiments` are fast.
    from .experiments import experiment_5f_hit_linear_condition_with_low_loss

    print("=" * 60)
    print("Thesis Experiments: GD Convergence and Adversarial Robustness")
    print("=" * 60)

   
    experiment_5f_hit_linear_condition_with_low_loss(
        num_runs=args.runs if args.runs is not None else 10_000,
        max_iterations=args.max_iterations if args.max_iterations is not None else 10_000_000,
        learning_rate=args.lr if args.lr is not None else 0.01,
        optimizer_name=args.optimizer,
        seed=args.seed if args.seed else 42,
        beta1=args.beta1,
        beta2=args.beta2,
    )


    print("\n" + "=" * 60)
    print("All requested experiments completed!")
    print("=" * 60)
