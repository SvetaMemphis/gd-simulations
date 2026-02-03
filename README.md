# Thesis Experiments: GD Convergence and Adversarial Robustness

This repository contains experiments for analyzing gradient descent (GD) convergence and adversarial robustness in neural networks.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Experiments

The code implements four main experiments:

### 1. Arbitrary Neurons and Initialization
Simulates GD on neural networks with an arbitrary number of neurons and various initialization schemes.

### 2. Decision Boundary Count
Computes the number of x-axis intersections (decision boundaries) as a function of iteration t.

### 3. Robust Case
Tests the robust case where the training set is moved closer to the origin.

### 4. Non-Symmetric Data
Allows simulations over non-symmetric datasets.

## Usage

Run all experiments:
```bash
python main.py
```

Or import and run individual experiments:
```python
from main import *

# Experiment 1: Arbitrary neurons
experiment_1_arbitrary_neurons(k=5, num_iterations=500, learning_rate=0.01)

# Experiment 2: Boundary count
experiment_2_boundary_count(k=2, num_iterations=1000, learning_rate=0.01)

# Experiment 3: Robust case
experiment_3_robust_case(k=2, num_iterations=1000, shifts=[0.0, -0.3, -0.5, -0.7])

# Experiment 4: Non-symmetric data
experiment_4_non_symmetric(k=2, num_iterations=1000, learning_rate=0.01)
```

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
