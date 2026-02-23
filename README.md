# Thesis Experiments: GD Convergence and Adversarial Robustness

This repository contains experiments for analyzing gradient descent (GD) convergence and adversarial robustness in neural networks.

## Quick Start (Terminal)

```bash
# Install dependencies
pip install -r requirements.txt

# Run all default experiments
python main.py

# Run specific experiment
python main.py --experiment 1

# Run multiple experiments
python main.py --experiment 1 2 3

# List all available experiments
python main.py --list-experiments

# Show help with all options
python main.py --help
```

## Code Structure (for code review)

The implementation is split into small modules under `thesis_experiments/`:

- `thesis_experiments/core.py`: network definition + GD + training loop
- `thesis_experiments/metrics.py`: decision boundaries, margin, margin gap
- `thesis_experiments/init_utils.py`: initializations + Theta utilities
- `thesis_experiments/datasets.py`: dataset helpers
- `thesis_experiments/experiments.py`: experiment runners + plotting
- `thesis_experiments/cli.py`: command-line interface

`main.py` is a **thin wrapper**: running it executes the CLI; importing from it re-exports the public API for backwards compatibility (`from main import ...`).

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Experiments

The code implements multiple experiments for analyzing GD convergence and margin maximization:

### 1. Arbitrary Neurons and Initialization
Simulates GD on neural networks with an arbitrary number of neurons and various initialization schemes.

### 2. Decision Boundary Count
Computes the number of x-axis intersections (decision boundaries) as a function of iteration t.

### 3. Robust Case
Tests the robust case where the training set is moved closer to the origin.

### 4. Non-Symmetric Data
Allows simulations over non-symmetric datasets.

### 5. Margin Convergence Analysis

#### 5a. Over-parameterized Regime
Compares margin convergence rates for different numbers of neurons (k = 2, 5, 10, 20, 50). Analyzes how over-parameterization affects convergence to optimal margin.

#### 5b. High-dimensional Clustered Data
Tests margin convergence on high-dimensional clustered datasets (simulated using 1D projections). Analyzes how data dimensionality affects margin convergence.

#### 5c. Two-neuron Margin Convergence
Runs multiple random initializations and tracks the number of iterations needed to achieve:
- Margin gap < δ (delta)
- Training loss < ε (epsilon)
- Both conditions simultaneously

Aggregates statistics across runs (mean, median, std, min, max) and creates histograms.

#### 5d. Mixture of All Settings
Combines over-parameterization, non-symmetric data, and clustered data to test the most general setting.

## How to Run Experiments

### Quick Reference

| Experiment | Default Status | Runtime | Key Parameters |
|------------|---------------|---------|----------------|
| Experiment 1 | ✅ Enabled | ~5-10s | `k`, `num_iterations`, `init_type` |
| Experiment 2 | ✅ Enabled | ~10-15s | `k`, `num_iterations`, `init_type` |
| Experiment 3 | ✅ Enabled | ~40-60s | `shifts`, `k`, `num_iterations` |
| Experiment 4 | ✅ Enabled | ~30-40s | `k`, `num_iterations`, `init_type` |
| Experiment 5a | ⚠️ Commented | ~2-5min | `k_values`, `num_runs` |
| Experiment 5b | ⚠️ Commented | ~1-3min | `d_values`, `k` |
| Experiment 5c | ⚠️ Commented | ~5-15min | `num_runs`, `delta`, `epsilon` |
| Experiment 5d | ⚠️ Commented | ~1-2min | `k`, `num_runs` |

### Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run all default experiments**:
   ```bash
   python main.py
   ```

   This will run Experiments 1-4 and generate plots in the current directory.

### Running from Terminal (Command Line)

The code supports command-line arguments for easy terminal usage:

#### Basic Commands

```bash
# Run all default experiments (1-4)
python main.py

# Run all default experiments without initialization examples
python main.py --skip-init-example

# List all available experiments
python main.py --list-experiments
# or
python main.py -l
```

#### Run Specific Experiments

```bash
# Run a single experiment
python main.py --experiment 1
python main.py -e 2

# Run multiple experiments
python main.py --experiment 1 2 3
python main.py -e 1 4

# Run experiment 5a (over-parameterized)
python main.py --experiment 5a

# Run experiment 5c (margin convergence)
python main.py --experiment 5c

# Run all experiments including 5a-5d
python main.py --experiment all
```

#### Customize Parameters

```bash
# Experiment 1: Custom neurons and iterations
python main.py --experiment 1 --k 10 --iterations 1000 --lr 0.005

# Experiment 2: Custom initialization
python main.py --experiment 2 --k 2 --iterations 2000 --w1-init 1.5 --b1-init 1.5 --M 15.0

# Experiment 3: Custom shifts
python main.py --experiment 3 --shifts "0.0,-0.2,-0.4,-0.6,-0.8"

# Experiment 4: Custom parameters
python main.py --experiment 4 --k 3 --iterations 2000 --init-type random --seed 123

# Experiment 5a: Custom k values and runs
python main.py --experiment 5a --k-values "2,5,10,20,50" --runs 10 --iterations 3000

# Experiment 5b: Custom dimensions
python main.py --experiment 5b --d-values "1,2,3,5,10" --k 15

# Experiment 5c: Custom thresholds and runs
python main.py --experiment 5c --runs 100 --delta 0.005 --epsilon 0.005 --max-iterations 20000

# Experiment 5d: Custom parameters
python main.py --experiment 5d --k 30 --runs 10 --iterations 3000
```

#### Common Parameters

| Parameter | Short | Description | Example |
|-----------|-------|-------------|---------|
| `--experiment` | `-e` | Experiment(s) to run | `-e 1 2 3` |
| `--k` | | Number of neurons | `--k 10` |
| `--iterations` | `-i` | GD iterations | `-i 2000` |
| `--lr` | | Learning rate | `--lr 0.005` |
| `--seed` | `-s` | Random seed | `-s 42` |
| `--init-type` | | Initialization type | `--init-type binary` |
| `--runs` | | Number of runs (exp 5a-5d) | `--runs 10` |
| `--k-values` | | Comma-separated k values | `--k-values "2,5,10"` |
| `--shifts` | | Comma-separated shifts | `--shifts "0.0,-0.3,-0.5"` |
| `--delta` | | Margin gap threshold | `--delta 0.01` |
| `--epsilon` | | Loss threshold | `--epsilon 0.01` |

#### Help and Examples

```bash
# Show help message with all options
python main.py --help

# Show help with examples
python main.py -h
```

#### Practical Examples

```bash
# Quick test run (fewer iterations)
python main.py --experiment 1 --iterations 100

# Reproducible run with seed
python main.py --experiment 1 2 3 --seed 42

# High-resolution experiment 5a
python main.py --experiment 5a --k-values "2,5,10,20,50" --runs 20 --iterations 5000

# Test robustness with many shifts
python main.py --experiment 3 --shifts "0.0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8"

# Run margin convergence with strict thresholds
python main.py --experiment 5c --runs 200 --delta 0.001 --epsilon 0.001
```

### Running All Experiments

**Using Terminal (Recommended)**:
```bash
# Run all default experiments (1-4)
python main.py

# Run all experiments including 5a-5d
python main.py --experiment all
```

**Using Python Code** (Alternative):
By default, `main.py` runs Experiments 1-4. To run experiments 5a-5d, use the terminal:
```bash
python main.py --experiment 5a 5b 5c 5d
```

Or modify the code directly:
1. Open `main.py`
2. Find the experiment calls (around line 1506-1520)
3. Uncomment the experiments you want to run
4. Run: `python main.py`

### Running Individual Experiments

#### Method 1: Python Script

Create a new Python file (e.g., `run_experiment.py`):

```python
from main import *

# Run a specific experiment
experiment_1_arbitrary_neurons(k=5, num_iterations=500, learning_rate=0.01, 
                               init_type="random", seed=42)
```

#### Method 2: Python Interactive Session

```python
python
>>> from main import *
>>> experiment_2_boundary_count(k=2, num_iterations=1000, learning_rate=0.01,
...                            init_type="thesis", w1_init=1.0, b1_init=1.0, M=10.0)
```

#### Method 3: Command Line (using Python -c)

```bash
python -c "from main import *; experiment_3_robust_case(k=2, num_iterations=1000, learning_rate=0.01, shifts=[0.0, -0.3, -0.5, -0.7])"
```

### Experiment Examples

#### Experiment 1: Arbitrary Neurons
```python
from main import experiment_1_arbitrary_neurons

experiment_1_arbitrary_neurons(
    k=5,                    # Number of neurons
    num_iterations=500,     # GD iterations
    learning_rate=0.01,     # Learning rate
    init_type="random",     # Initialization type
    seed=42                 # Random seed
)
```

#### Experiment 2: Boundary Count
```python
from main import experiment_2_boundary_count

experiment_2_boundary_count(
    k=2,                    # Number of neurons (must be 2 for thesis init)
    num_iterations=1000,
    learning_rate=0.01,
    init_type="thesis",     # Thesis-specific initialization
    w1_init=1.0,           # Initial w1 value
    b1_init=1.0,           # Initial b1 value
    M=10.0                  # Parameter M for thesis initialization
)
```

#### Experiment 3: Robust Case
```python
from main import experiment_3_robust_case

experiment_3_robust_case(
    k=2,
    num_iterations=1000,
    learning_rate=0.01,
    shifts=[0.0, -0.3, -0.5, -0.7],  # Negative = move closer to origin
    init_type="thesis",
    w1_init=1.0,
    b1_init=1.0,
    M=10.0
)
```

#### Experiment 4: Non-Symmetric Data
```python
from main import experiment_4_non_symmetric

experiment_4_non_symmetric(
    k=2,
    num_iterations=1000,
    learning_rate=0.01,
    init_type="random",
    seed=42
)
```

#### Experiment 5a: Over-parameterized Regime
```python
from main import experiment_5_overparameterized

experiment_5_overparameterized(
    k_values=[2, 5, 10, 20],  # Different neuron counts to test
    num_iterations=2000,
    learning_rate=0.01,
    num_runs=5,               # Number of random initializations per k
    seed=42
)
```

#### Experiment 5b: High-dimensional Clustered Data
```python
from main import experiment_5b_highdimensional_clustered

experiment_5b_highdimensional_clustered(
    d_values=[1, 2, 5],      # Effective dimensions to test
    num_clusters=2,
    points_per_cluster=10,
    num_iterations=2000,
    learning_rate=0.01,
    k=10,                    # Number of neurons
    seed=42
)
```

#### Experiment 5c: Margin Convergence Statistics
```python
from main import experiment_5c_margin_convergence_rate

experiment_5c_margin_convergence_rate(
    k=2,                     # Number of neurons
    num_runs=50,             # Number of random initializations
    max_iterations=10000,    # Maximum iterations per run
    learning_rate=0.01,
    delta=0.01,              # Margin gap threshold
    epsilon=0.01,            # Loss threshold
    seed=42
)
```

#### Experiment 5d: Mixture of All Settings
```python
from main import experiment_5d_mixture

experiment_5d_mixture(
    k=20,                    # Over-parameterized
    num_iterations=2000,
    learning_rate=0.01,
    num_runs=5,              # Number of random initializations
    seed=42
)
```

### Understanding the Output

When you run experiments, you'll see:

1. **Console Output**:
   - Initial parameters (w, b, v, Theta vector)
   - Training progress and final statistics
   - Convergence information (loss, boundaries, margins)

2. **Generated Plot Files**:
   - `experiment_1_k{k}_{init_type}.png` - Loss, boundaries, network output, parameter evolution
   - `experiment_2_boundary_count.png` - Boundary count over iterations
   - `experiment_3_robust_case.png` - Comparison across different shifts
   - `experiment_4_non_symmetric.png` - Symmetric vs non-symmetric comparison
   - `experiment_5a_overparameterized.png` - Margin convergence for different k
   - `experiment_5b_highdimensional.png` - High-dimensional analysis
   - `experiment_5c_margin_convergence.png` - Histograms of convergence times
   - `experiment_5d_mixture.png` - Combined settings analysis

### Customizing Parameters

All experiments accept customizable parameters. Common parameters:

- `k`: Number of neurons
- `num_iterations`: Number of GD iterations
- `learning_rate`: Learning rate (eta)
- `init_type`: Initialization type (`"random"`, `"thesis"`, `"binary"`, `"symmetric"`)
- `seed`: Random seed for reproducibility
- `track_margin`: Whether to track margin during training (default: False)
- `optimal_margin`: Optimal margin value (default: 1.0 for symmetric 2-point dataset)

### Running Example Scripts

The repository includes example scripts:

```bash
# Run initialization examples
python example_init.py
```

This demonstrates different initialization methods and shows how to control initial weights.

### Expected Runtime

Approximate runtime for default experiments:

- **Experiment 1**: ~5-10 seconds (500 iterations, k=5)
- **Experiment 2**: ~10-15 seconds (1000 iterations, k=2)
- **Experiment 3**: ~40-60 seconds (4 shifts × 1000 iterations)
- **Experiment 4**: ~30-40 seconds (4 configurations × 1000 iterations)
- **Experiment 5a**: ~2-5 minutes (multiple k values, 5 runs each)
- **Experiment 5b**: ~1-3 minutes (multiple dimensions, 3 runs each)
- **Experiment 5c**: ~5-15 minutes (50 runs, up to 10000 iterations each)
- **Experiment 5d**: ~1-2 minutes (5 runs, 2000 iterations each)

**Total runtime for all experiments**: ~10-25 minutes (depending on hardware)

### Troubleshooting

#### Common Issues

1. **Import errors**: Make sure you're in the repository directory:
   ```bash
   cd /path/to/thesis-experiments
   python main.py
   ```

2. **Missing dependencies**: Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. **Plots not saving**: Check write permissions in the current directory

4. **Experiments taking too long**: 
   - Reduce `num_iterations` for faster testing
   - Reduce `num_runs` for experiments 5a-5d
   - Reduce `resolution` in margin computation (default: 10000)

5. **Memory issues with large k**:
   - Reduce `num_iterations` or disable parameter history tracking
   - Use smaller `k_values` in Experiment 5a

6. **Numerical overflow warnings**:
   - Normal for extreme parameter values
   - Consider reducing learning rate if persistent

### Tips

- **Start small**: Test with `num_iterations=100` first to verify everything works
- **Use seeds**: Set `seed=42` for reproducibility
- **Monitor progress**: Experiments print progress every 10 runs (Experiment 5c)
- **Check plots**: Generated PNG files show detailed visualizations
- **Customize**: All parameters can be adjusted to fit your needs

## Controlling Initial Weights Theta

You have full control over the initial parameter vector Theta = [w_1, ..., w_k, b_1, ..., b_k, v_1, ..., v_k]:

### 1. Binary initialization (w_j = +1/-1)
```python
from main import initialize_network, print_initial_params, get_theta_vector

# Initialize w_j randomly to +1/-1
params = initialize_network(k=5, init_type="binary", v_binary=True)
print_initial_params(params)  # Shows w, b, v, and Theta vector
```

### 2. Explicit w_binary list
```python
# Explicitly set w_j = [1, -1, 1, -1, 1]
params = initialize_network(k=5, w_binary=[1, -1, 1, -1, 1], v_binary=True)
```

### 3. Fully custom initialization
```python
import numpy as np

# Specify all parameters explicitly
w_custom = np.array([1.0, -1.0, 0.5, -0.5])
b_custom = np.array([0.1, -0.1, 0.2, -0.2])
v_custom = np.array([1.0, -1.0, 1.0, -1.0])
params = initialize_network(k=4, w_init=w_custom, b_init=b_custom, v_init=v_custom)
```

### 4. Get Theta vector
```python
# Get the parameter vector Theta
theta = get_theta_vector(params)
print(f"Theta = {theta}")
print(f"||Theta|| = {np.linalg.norm(theta)}")
```

### 5. Available initialization types
- `"random"`: Random normal initialization (default)
- `"thesis"`: Thesis-specific initialization (k=2 only)
- `"symmetric"`: Small symmetric initialization
- `"binary"`: Initialize w_j to +1/-1 randomly
- `"custom"`: Use explicitly provided arrays

See `example_init.py` for complete examples:
```bash
python example_init.py
```

## Network Architecture

The neural network has the form:
```
Phi(x) = sum_j v_j * ReLU(w_j * x + b_j)
```

where:
- `k` is the number of neurons
- `w_j` are input weights
- `b_j` are biases
- `v_j` are output weights

## Loss Function

The exponential loss function is used:
```
L(theta) = (1/n) * sum_i exp(-y_i * Phi(theta; x_i))
```

## Gradient Descent

Parameters are updated via:
```
theta^(t+1) = theta^(t) - eta * grad_theta L(theta^(t))
```

## Margin Computation

The code implements margin computation based on the thesis definition:

- **Margin**: The smallest change (in Euclidean distance) one must make to a data point to change its prediction.
- **Margin Gap**: Δ(t) = optimal_margin - margin(t), measuring distance from optimal margin.

The `train_gd()` function can track margins during training:
```python
params_history, losses, boundary_counts, margins, margin_gaps = train_gd(
    params, x, y, learning_rate, num_iterations,
    track_margin=True, optimal_margin=1.0
)
```

You can also compute margin directly:
```python
from main import compute_margin, compute_margin_gap

margin = compute_margin(params, x, y)
margin_gap = compute_margin_gap(params, x, y, optimal_margin=1.0)
```

## Quick Reference

### Key Functions

- `initialize_network()`: Initialize network parameters with various schemes
- `train_gd()`: Train network with GD, optionally tracking margins
- `compute_margin()`: Compute margin of trained network
- `compute_margin_gap()`: Compute margin gap from optimal
- `get_theta_vector()`: Get flattened parameter vector Theta
- `print_initial_params()`: Print parameters in readable format

### Experiment Functions

- `experiment_1_arbitrary_neurons()`: Test arbitrary neuron counts
- `experiment_2_boundary_count()`: Track decision boundaries over time
- `experiment_3_robust_case()`: Test robustness with shifted data
- `experiment_4_non_symmetric()`: Test non-symmetric datasets
- `experiment_5_overparameterized()`: Over-parameterization analysis
- `experiment_5b_highdimensional_clustered()`: High-dimensional data analysis
- `experiment_5c_margin_convergence_rate()`: Margin convergence statistics
- `experiment_5d_mixture()`: Combined settings analysis

| Experiment | What changes             | What stays fixed | What question                          | CLI Command                                              |
| ---------- | ------------------------ | ---------------- | -------------------------------------- | -------------------------------------------------------- |
| Exp 1      | Initialization / width k | Dataset          | Does GD converge?                      | `python main.py --experiment 1 --k 5 --init-type random` |
| Exp 2      | None                     | Dataset          | How do decision boundaries evolve?     | `python main.py --experiment 2`                          |
| Exp 3      | Data margin (shift)      | Model            | Sensitivity to robustness              | `python main.py --experiment 3`                          |
| Exp 5a     | Width k (2,5,10,20,50)   | Dataset          | Does overparameterization help margin? | `python main.py --experiment 5a`                         |
| Exp 5b     | Cluster geometry (d)     | Width k          | Does data structure affect margin?     | `python main.py --experiment 5b --k 10`                  |
| Exp 5c     | Random seed              | Data, k          | How stable is convergence?             | `python main.py --experiment 5c`                         |
