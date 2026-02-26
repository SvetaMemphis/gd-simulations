
# Thesis Experiments — Gradient Descent Dynamics

This repository contains code for running a controlled gradient descent experiment on a 1D two-neuron ReLU network.

Currently, the CLI runs **Experiment 5f** only.

---

# Model

We study a 1D ReLU network with two neurons:

[
f(x) = v_1 \mathrm{ReLU}(w_1 x + b_1) + v_2 \mathrm{ReLU}(w_2 x + b_2)
]

For this experiment:

* Number of neurons: `k = 2`
* Output weights: `v = [1, -1]` (fixed during training)
* Initialization:

  * ( w_1, w_2 \sim \mathcal{N}(0,2) )
  * ( b_1 = b_2 = 0 )

Training is performed using full-batch gradient descent with **exponential loss**:

[
L = \frac{1}{n} \sum_i \exp(-y_i f(x_i))
]

---

# Experiment — Description

Each run:

1. Initializes the network randomly.
2. Runs gradient descent.
3. Stops when one of the following occurs:

### Hit Condition

Both:

* ( |w_1 + b_1 + w_2 - b_2| < \text{tol} )
* Loss < `loss_threshold`

### Loss Abort

If after 10,000 iterations the loss is still above the threshold.

### Max Iterations

If `max_iterations` is reached.

The experiment records:

* Hit times
* Final parameters
* Loss values
* A geometric metric:
  [
  \min(|b_2 - b_1|,\ |w_1^{(0)} + w_2^{(0)}|/2)
  ]

---

# How to Run

From the project root directory:

```bash
python main.py
```

By default this runs:

* `num_runs = 10_000`
* `learning_rate = 0.01`
* `max_iterations = 10_000_000`
* `seed = 42`

---

# CLI Arguments

You can override parameters:

```bash
python main.py \
  --runs 10000 \
  --lr 0.05 \
  --max-iterations 1000000 \
  --seed 123
```

### Available Arguments

| Argument                    | Description                                  |
| --------------------------- | -------------------------------------------- |
| `--runs`                    | Number of independent random initializations |
| `--lr` or `--learning-rate` | Learning rate                                |
| `--max-iterations`          | Maximum GD iterations                        |
| `--seed` or `-s`            | Random seed                                  |

---

# Output Files

Running the experiment produces:

* `experiment_5f_runs.csv` — detailed results per run
* `experiment_5f_summary.txt` — overall statistics
* `experiment_5f_hit_time_hist.png`
* `experiment_5f_hit_time_hist.csv`
* `experiment_5f_metric_hist.png`
* `experiment_5f_metric_hist.csv`

---

# Project Structure

```
core.py         # Network, gradients, GD
experiments.py  # Experiment 5f implementation
datasets.py     # Dataset generation
init_utils.py   # Parameter initialization
cli.py          # Command-line interface
main.py         # Entry point
```

---

# Notes

* Output weights `v` are fixed.
