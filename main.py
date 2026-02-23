"""
Thin entrypoint.

Run:
  python main.py --help
  python main.py --experiment 1 --k 2
  python main.py --experiment 5f
  python main.py --experiment 6e
"""

def _run_cli() -> None:
    from thesis_experiments.cli import main as cli_main
    cli_main()

if __name__ == "__main__":
    _run_cli()
else:
    from thesis_experiments.api import *  # noqa: F401,F403