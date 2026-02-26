import argparse
import sys
from typing import List, Optional, Union


def parse_list_arg(arg_str: Optional[str]) -> Optional[List[Union[int, float]]]:
    """Parse comma-separated numeric list argument."""
    if arg_str is None:
        return None
    out: List[Union[int, float]] = []
    for token in arg_str.split(","):
        t = token.strip()
        if t == "":
            continue
        out.append(float(t) if "." in t else int(t))
    return out


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Thesis Experiments: GD Convergence and Adversarial Robustness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all default experiments
  python main.py

  # Run specific experiment
  python main.py --experiment 1

  # Run multiple experiments
  python main.py --experiment 1 2 3

  # Run with custom parameters
  python main.py --experiment 1 --k 10 --iterations 1000 --lr 0.005

  # Run experiment 5a with custom k values
  python main.py --experiment 5a --k-values 2,5,10,20,50 --runs 10

  # Run experiment 5c with custom thresholds
  python main.py --experiment 5c --runs 100 --delta 0.01 --epsilon 0.01

  # List all available experiments
  python main.py --list-experiments
        """,
    )

    parser.add_argument(
        "--experiment",
        "-e",
        nargs="+",
        choices=["1", "2", "3", "4", "5a", "5b", "5c", "5d", "5f", "6e", "7", "all", "init"],
        default=["all"],
        help="Experiment(s) to run: 1, 2, 3, 4, 5a, 5b, 5c, 5d, 5f, 6e, 7, all, or init",
    )
    parser.add_argument("--list-experiments", "-l", action="store_true", help="List all available experiments and exit")

    # Common parameters
    parser.add_argument("--k", type=int, help="Number of neurons")
    parser.add_argument("--iterations", "-i", type=int, help="Number of GD iterations")
    parser.add_argument("--lr", "--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--init-type", choices=["random", "thesis", "binary", "symmetric"], help="Initialization type")

    # Experiment-specific parameters
    parser.add_argument("--k-values", type=str, help="Comma-separated k values (for exp 5a)")
    parser.add_argument("--d-values", type=str, help="Comma-separated d values (for exp 5b)")
    parser.add_argument("--shifts", type=str, help="Comma-separated shifts (for exp 3)")
    parser.add_argument("--runs", type=int, help="Number of runs (for exp 5a, 5b, 5c, 5d)")
    parser.add_argument("--delta", type=float, help="Margin gap threshold (for exp 5c)")
    parser.add_argument("--epsilon", type=float, help="Loss threshold (for exp 5c)")
    parser.add_argument("--max-iterations", type=int, help="Max iterations (for exp 5c)")
    parser.add_argument("--w1", type=float, help="Initial w1 (for exp 2, 3)")
    parser.add_argument("--b1", type=float, help="Initial b1 (for exp 2, 3)")
    parser.add_argument("--w2", type=float, help="Second neuron weight (k=2 thesis init)")
    parser.add_argument("--b2", type=float, help="Second neuron bias (k=2 thesis init)")
    parser.add_argument("--M", type=float, help="Parameter M (for exp 2, 3)")
    parser.add_argument("--skip-init-example", action="store_true", help="Skip initialization examples output")

    args = parser.parse_args(argv)

    if args.list_experiments:
        print("\nAvailable Experiments:")
        print("=" * 60)
        print("1  - Arbitrary Neurons and Initialization")
        print("2  - Decision Boundary Count")
        print("3  - Robust Case (shifted dataset)")
        print("4  - Non-Symmetric Data")
        print("5a - Over-parameterized Regime")
        print("5b - High-dimensional Clustered Data")
        print("5c - Two-neuron Margin Convergence")
        print("5d - Mixture of All Settings")
        print("5f - Hit linear condition with small loss")
        print("init - Show initialization options")
        print("all - Run experiments 1-4 (default)")
        print("\nUse --help for detailed usage and examples.")
        sys.exit(0)

    # Delayed import so `--help`/`--list-experiments` are fast.
    from .experiments import (
        example_initialization_options,
        experiment_1_arbitrary_neurons,
        experiment_2_boundary_count,
        experiment_3_robust_case,
        experiment_4_non_symmetric,
        experiment_5_overparameterized,
        experiment_5b_highdimensional_clustered,
        experiment_5c_margin_convergence_rate,
        experiment_5f_hit_linear_condition_with_low_loss,
        experiment_5d_mixture,
        experiment_6e_overparam_cluster_then_collapse_compare_margins,
        experiment_7_big_train_project_distill_small_parallel
    )

    print("=" * 60)
    print("Thesis Experiments: GD Convergence and Adversarial Robustness")
    print("=" * 60)

    if "init" in args.experiment or (not args.skip_init_example and "all" in args.experiment):
        example_initialization_options()

    experiments_to_run = args.experiment if "all" not in args.experiment else ["1", "2", "3", "4"]

    lr = args.lr if args.lr is not None else 0.01

    if "1" in experiments_to_run:
        experiment_1_arbitrary_neurons(
            k=args.k if args.k is not None else 5,
            num_iterations=args.iterations if args.iterations is not None else 500,
            learning_rate=lr,
            init_type=args.init_type if args.init_type is not None else "random",
            seed=args.seed,
            w1_init=args.w1 if args.w1 is not None else 1.0,
            b1_init=args.b1 if args.b1 is not None else 1.0,
            w2_init=args.w2 if args.w2 is not None else -10.0,
            b2_init=args.b2 if args.b2 is not None else 10.0,
        )

    if "2" in experiments_to_run:
        experiment_2_boundary_count(
            k=args.k if args.k is not None else 2,
            num_iterations=args.iterations if args.iterations is not None else 1000,
            learning_rate=lr,
            init_type=args.init_type if args.init_type is not None else "thesis",
            w1_init=args.w1_init if args.w1_init is not None else 1.0,
            b1_init=args.b1_init if args.b1_init is not None else 1.0,
            M=args.M if args.M is not None else 10.0,
        )

    if "3" in experiments_to_run:
        shifts = parse_list_arg(args.shifts) if args.shifts else [0.0, -0.3, -0.5, -0.7]
        experiment_3_robust_case(
            k=args.k if args.k is not None else 2,
            num_iterations=args.iterations if args.iterations is not None else 1000,
            learning_rate=lr,
            shifts=[float(s) for s in shifts],  # type: ignore[arg-type]
            init_type=args.init_type if args.init_type is not None else "thesis",
            w1_init=args.w1_init if args.w1_init is not None else 1.0,
            b1_init=args.b1_init if args.b1_init is not None else 1.0,
            M=args.M if args.M is not None else 10.0,
        )

    if "4" in experiments_to_run:
        experiment_4_non_symmetric(
            k=args.k if args.k is not None else 2,
            num_iterations=args.iterations if args.iterations is not None else 1000,
            learning_rate=lr,
            init_type=args.init_type if args.init_type is not None else "random",
            seed=args.seed,
        )

    if "5a" in experiments_to_run:
        k_values = parse_list_arg(args.k_values) if args.k_values else [2, 5, 10, 20]
        experiment_5_overparameterized(
            k_values=[int(k) for k in k_values],  # type: ignore[arg-type]
            num_iterations=args.iterations if args.iterations is not None else 2000,
            learning_rate=lr,
            num_runs=args.runs if args.runs is not None else 5,
            seed=args.seed,
        )

    if "5b" in experiments_to_run:
        d_values = parse_list_arg(args.d_values) if args.d_values else [1, 2, 5]
        experiment_5b_highdimensional_clustered(
            d_values=[int(d) for d in d_values],  # type: ignore[arg-type]
            num_iterations=args.iterations if args.iterations is not None else 2000,
            learning_rate=lr,
            k=args.k if args.k is not None else 10,
            seed=args.seed,
        )

    if "5c" in experiments_to_run:
        experiment_5c_margin_convergence_rate(
            k=args.k if args.k is not None else 2,
            num_runs=args.runs if args.runs is not None else 50,
            max_iterations=args.max_iterations if args.max_iterations is not None else 10000,
            learning_rate=lr,
            delta=args.delta if args.delta is not None else 0.01,
            epsilon=args.epsilon if args.epsilon is not None else 0.01,
            seed=args.seed,
        )

    if "5d" in experiments_to_run:
        experiment_5d_mixture(
            k=args.k if args.k is not None else 20,
            num_iterations=args.iterations if args.iterations is not None else 2000,
            learning_rate=lr,
            num_runs=args.runs if args.runs is not None else 5,
            seed=args.seed,
        )

    if "5f" in experiments_to_run:
        experiment_5f_hit_linear_condition_with_low_loss(
        num_runs=args.runs if args.runs is not None else 10000,
        max_iterations=args.max_iterations if args.max_iterations is not None else 10_000_000,
        learning_rate=lr,
        seed=args.seed,
    )

    if "7" in experiments_to_run:
        experiment_7_big_train_project_distill_small_parallel(
        k=20, d=10, n_points=1000,
        lr_big=1e-3, lr_small=1e-3,
        max_pretrain_iters=args.iterations if args.iterations else 200_000,
        post_iters=200_000,
        track_every=1000,
        seed= args.seed if args.seed else 42,
    )

    print("\n" + "=" * 60)
    print("All requested experiments completed!")
    print("=" * 60)

    if "6e" in experiments_to_run:
        experiment_6e_overparam_cluster_then_collapse_compare_margins(
            learning_rate=lr,
            post_collapse_iters=args.iterations if args.iterations else 1000000,
            seed=args.seed if args.seed else 42,
        )