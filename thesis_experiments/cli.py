import argparse
import sys
from typing import List, Optional

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Thesis Experiments (PyTorch): Exp 1 (k=2), 5f, 6e",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  python main.py --experiment 1 --k 2 --iterations 2000 --lr 0.01
  python main.py --experiment 5f --runs 2000 --lr 0.05
  python main.py --experiment 6e --lr 0.01 --pre-iters 200000 --post-iters 200000
        """,
    )

    parser.add_argument(
        "--experiment",
        "-e",
        nargs="+",
        choices=["1", "5f", "6e", "all"],
        default=["all"],
        help="Experiment(s) to run: 1, 5f, 6e, or all (default)",
    )
    parser.add_argument("--list-experiments", "-l", action="store_true", help="List experiments and exit")

    # common
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")

    # exp 1
    parser.add_argument("--k", type=int, default=2, help="Number of neurons (exp 1 supports only k=2 here)")
    parser.add_argument("--iterations", type=int, default=2000, help="GD iterations (exp 1)")
    parser.add_argument("--w1", type=float, default=1.0, help="Init w1 for exp 1 (k=2)")
    parser.add_argument("--b1", type=float, default=1.0, help="Init b1 for exp 1 (k=2)")
    parser.add_argument("--w2", type=float, default=-10.0, help="Init w2 for exp 1 (k=2)")
    parser.add_argument("--b2", type=float, default=10.0, help="Init b2 for exp 1 (k=2)")

    # exp 5f
    parser.add_argument("--runs", type=int, default=10000, help="Number of runs (exp 5f)")
    parser.add_argument("--max-iterations", type=int, default=10_000_000, help="Max iterations per run (exp 5f)")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance for |w1+b1+w2-b2| (exp 5f)")
    parser.add_argument("--loss-threshold", type=float, default=0.5, help="Loss threshold (exp 5f)")

    # exp 6e
    parser.add_argument("--d", type=int, default=50, help="Dimension d (exp 6e)")
    parser.add_argument("--n", type=int, default=1000, help="Number of points n (exp 6e)")
    parser.add_argument("--radius", type=float, default=0.5, help="Ball radius (exp 6e)")
    parser.add_argument("--pre-iters", type=int, default=200_000, help="Max pre-collapse iterations (exp 6e)")
    parser.add_argument("--post-iters", type=int, default=200_000, help="Post-collapse iterations (exp 6e)")
    parser.add_argument("--collapse-loss-threshold", type=float, default=1e-5, help="Pre-collapse stop loss (exp 6e)")
    parser.add_argument("--track-every", type=int, default=100, help="Track margin every N iters (exp 6e)")

    args = parser.parse_args(argv)

    if args.list_experiments:
        print("\nAvailable Experiments:")
        print("=" * 60)
        print("1  - Experiment 1 (ONLY k=2 here)")
        print("5f - Hit linear condition with low loss (many random runs)")
        print("6e - Overparam cluster then collapse (margin comparison)")
        print("all - Run 1, 5f, 6e")
        print("=" * 60)
        sys.exit(0)

    from .experiments import (
        experiment_1_k2,
        experiment_5f_hit_linear_condition_with_low_loss,
        experiment_6e_overparam_cluster_then_collapse_compare_margins,
    )

    experiments_to_run = args.experiment if "all" not in args.experiment else ["1", "5f", "6e"]

    print("=" * 60)
    print("Thesis Experiments (PyTorch)")
    print("=" * 60)

    if "1" in experiments_to_run:
        if args.k != 2:
            raise ValueError("This PyTorch bundle supports Experiment 1 only with k=2.")
        experiment_1_k2(
            num_iterations=args.iterations,
            learning_rate=args.lr,
            seed=args.seed,
            w1_init=args.w1,
            b1_init=args.b1,
            w2_init=args.w2,
            b2_init=args.b2,
        )

    if "5f" in experiments_to_run:
        experiment_5f_hit_linear_condition_with_low_loss(
            num_runs=args.runs,
            max_iterations=args.max_iterations,
            learning_rate=args.lr,
            tol=args.tol,
            loss_threshold=args.loss_threshold,
            seed=args.seed,
        )

    if "6e" in experiments_to_run:
        experiment_6e_overparam_cluster_then_collapse_compare_margins(
            k=20,
            d=args.d,
            n=args.n,
            radius=args.radius,
            learning_rate=args.lr,
            max_pre_collapse_iters=args.pre_iters,
            post_collapse_iters=args.post_iters,
            collapse_loss_threshold=args.collapse_loss_threshold,
            seed=args.seed,
            track_every=args.track_every,
        )

    print("\n" + "=" * 60)
    print("All requested experiments completed!")
    print("=" * 60)