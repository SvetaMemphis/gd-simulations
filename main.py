"""
Experiments for thesis on GD convergence and adversarial robustness.

This module implements:
1. GD simulation on neural networks with arbitrary number of neurons
2. Computing x-axis intersections (decision boundaries) as a function of iteration
3. Testing robust case (training set moved closer to origin)
4. Supporting non-symmetric data
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass
import json


@dataclass
class NetworkParams:
    """Parameters for a neural network with k neurons."""
    w: np.ndarray  # shape (k,): weights for each neuron
    b: np.ndarray  # shape (k,): biases for each neuron
    v: np.ndarray  # shape (k,): output weights for each neuron
    
    def __post_init__(self):
        assert len(self.w) == len(self.b) == len(self.v)
    
    @property
    def k(self):
        """Number of neurons."""
        return len(self.w)
    
    def copy(self):
        """Create a deep copy."""
        return NetworkParams(
            w=self.w.copy(),
            b=self.b.copy(),
            v=self.v.copy()
        )


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, x)


def network_forward(params: NetworkParams, x: np.ndarray) -> np.ndarray:
    """
    Forward pass through the network.
    
    Args:
        params: Network parameters
        x: Input values (scalar or array)
    
    Returns:
        Network outputs
    """
    x = np.asarray(x)
    # For each neuron: v_j * ReLU(w_j * x + b_j)
    activations = np.zeros((params.k, len(x)))
    for j in range(params.k):
        pre_activation = params.w[j] * x + params.b[j]
        activations[j] = params.v[j] * relu(pre_activation)
    
    return np.sum(activations, axis=0)


def exponential_loss(y: np.ndarray, predictions: np.ndarray) -> float:
    """
    Compute exponential loss: exp(-y * prediction).
    
    Args:
        y: True labels
        predictions: Network predictions
    
    Returns:
        Average loss
    """
    return np.mean(np.exp(-y * predictions))


def compute_gradients(
    params: NetworkParams,
    x: np.ndarray,
    y: np.ndarray,
    loss_fn: Callable = exponential_loss
) -> NetworkParams:
    """
    Compute gradients of the loss with respect to network parameters.
    
    Args:
        params: Current network parameters
        x: Training inputs
        y: Training labels
        loss_fn: Loss function
    
    Returns:
        Gradients for w, b, v
    """
    k = params.k
    n = len(x)
    
    # Forward pass
    pre_activations = np.zeros((k, n))
    activations = np.zeros((k, n))
    outputs = np.zeros(n)
    
    for j in range(k):
        pre_activations[j] = params.w[j] * x + params.b[j]
        activations[j] = params.v[j] * relu(pre_activations[j])
        outputs += activations[j]
    
    # Loss: L = (1/n) * sum_i exp(-y_i * output_i)
    # Gradient w.r.t. output: dL/doutput_i = -(1/n) * y_i * exp(-y_i * output_i)
    loss_grad_output = -(1.0 / n) * y * np.exp(-y * outputs)
    
    # Initialize gradients
    grad_w = np.zeros(k)
    grad_b = np.zeros(k)
    grad_v = np.zeros(k)
    
    # Backward pass
    for j in range(k):
        # Indicator: pre_activation > 0
        indicator = (pre_activations[j] > 0).astype(float)
        
        # Gradient w.r.t. v_j
        grad_v[j] = np.sum(loss_grad_output * relu(pre_activations[j]))
        
        # Gradient w.r.t. w_j and b_j (only where pre_activation > 0)
        grad_w[j] = np.sum(loss_grad_output * params.v[j] * indicator * x)
        grad_b[j] = np.sum(loss_grad_output * params.v[j] * indicator)
    
    return NetworkParams(w=grad_w, b=grad_b, v=grad_v)


def gradient_descent_step(
    params: NetworkParams,
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    loss_fn: Callable = exponential_loss
) -> Tuple[NetworkParams, float]:
    """
    Perform one step of gradient descent.
    
    Args:
        params: Current parameters
        x: Training inputs
        y: Training labels
        learning_rate: Learning rate eta
        loss_fn: Loss function
    
    Returns:
        Updated parameters and current loss
    """
    gradients = compute_gradients(params, x, y, loss_fn)
    
    # Update: theta = theta - eta * grad
    new_params = NetworkParams(
        w=params.w - learning_rate * gradients.w,
        b=params.b - learning_rate * gradients.b,
        v=params.v - learning_rate * gradients.v
    )
    
    predictions = network_forward(new_params, x)
    loss = loss_fn(y, predictions)
    
    return new_params, loss


def find_decision_boundaries(params: NetworkParams, x_range: Tuple[float, float] = (-10, 10), 
                            resolution: int = 10000) -> List[float]:
    """
    Find all x-axis intersections (decision boundaries) of the network.
    
    The decision boundaries are points where Phi(x) = 0.
    
    Args:
        params: Network parameters
        x_range: Range of x values to search
        resolution: Number of points to evaluate
    
    Returns:
        List of x values where the network output crosses zero
    """
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    outputs = network_forward(params, x_vals)
    
    # Find sign changes
    boundaries = []
    for i in range(len(outputs) - 1):
        if outputs[i] * outputs[i + 1] <= 0:  # Sign change or zero
            # Linear interpolation for more accurate boundary
            if outputs[i] == 0:
                boundaries.append(x_vals[i])
            elif outputs[i + 1] == 0:
                boundaries.append(x_vals[i + 1])
            else:
                # Linear interpolation
                t = -outputs[i] / (outputs[i + 1] - outputs[i])
                boundary = x_vals[i] + t * (x_vals[i + 1] - x_vals[i])
                boundaries.append(boundary)
    
    return boundaries


def train_gd(
    initial_params: NetworkParams,
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    num_iterations: int,
    loss_fn: Callable = exponential_loss,
    track_boundaries: bool = True,
    x_range: Tuple[float, float] = (-10, 10)
) -> Tuple[List[NetworkParams], List[float], List[List[float]]]:
    """
    Train network using gradient descent.
    
    Args:
        initial_params: Initial network parameters
        x: Training inputs
        y: Training labels
        learning_rate: Learning rate
        num_iterations: Number of GD iterations
        loss_fn: Loss function
        track_boundaries: Whether to track decision boundaries at each iteration
        x_range: Range for finding decision boundaries
    
    Returns:
        List of parameters at each iteration, list of losses, list of boundary counts
    """
    params = initial_params.copy()
    params_history = [params.copy()]
    losses = []
    boundary_counts = []
    
    for t in range(num_iterations):
        params, loss = gradient_descent_step(params, x, y, learning_rate, loss_fn)
        params_history.append(params.copy())
        losses.append(loss)
        
        if track_boundaries:
            boundaries = find_decision_boundaries(params, x_range)
            boundary_counts.append(len(boundaries))
    
    return params_history, losses, boundary_counts


def get_theta_vector(params: NetworkParams) -> np.ndarray:
    """
    Get the parameter vector Theta = [w_1, ..., w_k, b_1, ..., b_k, v_1, ..., v_k].
    
    Args:
        params: Network parameters
    
    Returns:
        Flattened parameter vector Theta
    """
    return np.concatenate([params.w, params.b, params.v])


def print_initial_params(params: NetworkParams, title: str = "Initial Parameters"):
    """
    Print initial parameters in a readable format.
    
    Args:
        params: Network parameters
        title: Title for the output
    """
    print(f"\n{title}:")
    print(f"  Number of neurons (k): {params.k}")
    print(f"  w = {params.w}")
    print(f"  b = {params.b}")
    print(f"  v = {params.v}")
    theta = get_theta_vector(params)
    print(f"  Theta (parameter vector) = {theta}")
    print(f"  ||Theta|| = {np.linalg.norm(theta):.6f}")


def initialize_network(
    k: int,
    init_type: str = "random",
    w1_init: Optional[float] = None,
    b1_init: Optional[float] = None,
    M: Optional[float] = None,
    seed: Optional[int] = None,
    w_init: Optional[np.ndarray] = None,
    b_init: Optional[np.ndarray] = None,
    v_init: Optional[np.ndarray] = None,
    w_binary: Optional[List[int]] = None,
    b_scale: float = 0.1,
    v_binary: bool = False
) -> NetworkParams:
    """
    Initialize network parameters.
    
    Args:
        k: Number of neurons
        init_type: Type of initialization:
            - "random": Random normal initialization
            - "thesis": Thesis-specific initialization (k=2 only)
            - "symmetric": Small symmetric initialization
            - "binary": Initialize w_j to +1/-1
            - "custom": Use explicitly provided w_init, b_init, v_init
        w1_init: Initial value for w_1 (for thesis initialization)
        b1_init: Initial value for b_1 (for thesis initialization)
        M: Parameter M for thesis initialization
        seed: Random seed
        w_init: Explicit initial weights array (shape: (k,)) - overrides init_type
        b_init: Explicit initial biases array (shape: (k,)) - overrides init_type
        v_init: Explicit initial output weights array (shape: (k,)) - overrides init_type
        w_binary: List of +1/-1 values for w_j initialization (length k)
        b_scale: Scale for bias initialization (used with binary w)
        v_binary: If True, initialize v_j to +1/-1 randomly
    
    Returns:
        Initialized network parameters
    """
    if seed is not None:
        np.random.seed(seed)
    
    # If explicit arrays provided, use them (highest priority)
    if w_init is not None or b_init is not None or v_init is not None:
        if w_init is None:
            w_init = np.random.randn(k) * 0.5
        if b_init is None:
            b_init = np.random.randn(k) * 0.5
        if v_init is None:
            v_init = np.random.choice([-1, 1], size=k) * np.random.rand(k)
        
        assert len(w_init) == k and len(b_init) == k and len(v_init) == k, \
            "Initialization arrays must have length k"
        
        return NetworkParams(
            w=np.array(w_init),
            b=np.array(b_init),
            v=np.array(v_init)
        )
    
    # Binary w initialization (w_j = +1/-1)
    if w_binary is not None:
        assert len(w_binary) == k, "w_binary must have length k"
        w = np.array(w_binary, dtype=float)
        b = np.random.randn(k) * b_scale
        if v_binary:
            v = np.random.choice([-1, 1], size=k)
        else:
            v = np.random.choice([-1, 1], size=k) * np.random.rand(k)
        return NetworkParams(w=w, b=b, v=v)
    
    if init_type == "thesis" and k == 2:
        # Thesis initialization: v1=1, w1=b1>0.5, v2=-1, w2=-b2=-M
        if w1_init is None:
            w1_init = 1.0
        if b1_init is None:
            b1_init = 1.0
        if M is None:
            M = 10.0
        
        return NetworkParams(
            w=np.array([w1_init, -M]),
            b=np.array([b1_init, M]),
            v=np.array([1.0, -1.0])
        )
    
    elif init_type == "binary":
        # Initialize w_j to +1/-1 randomly
        w = np.random.choice([-1, 1], size=k)
        b = np.random.randn(k) * b_scale
        if v_binary:
            v = np.random.choice([-1, 1], size=k)
        else:
            v = np.random.choice([-1, 1], size=k) * np.random.rand(k)
        return NetworkParams(w=w, b=b, v=v)
    
    elif init_type == "symmetric":
        # Symmetric initialization
        return NetworkParams(
            w=np.random.randn(k) * 0.1,
            b=np.random.randn(k) * 0.1,
            v=np.random.choice([-1, 1], size=k)
        )
    
    else:  # random
        return NetworkParams(
            w=np.random.randn(k) * 0.5,
            b=np.random.randn(k) * 0.5,
            v=np.random.choice([-1, 1], size=k) * np.random.rand(k)
        )


def create_dataset(
    symmetric: bool = True,
    shift: float = 0.0,
    scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training dataset.
    
    Args:
        symmetric: If True, use symmetric dataset (-1, 1), else use non-symmetric
        shift: Shift dataset (for robust case, move closer to origin)
        scale: Scale dataset
    
    Returns:
        x, y arrays
    """
    if symmetric:
        x = np.array([-1.0, 1.0]) * scale + shift
        y = np.array([-1.0, 1.0])
    else:
        # Non-symmetric: can be customized
        x = np.array([-0.5, 1.5]) * scale + shift
        y = np.array([-1.0, 1.0])
    
    return x, y


def experiment_1_arbitrary_neurons(
    k: int = 5,
    num_iterations: int = 1000,
    learning_rate: float = 0.01,
    init_type: str = "random",
    seed: int = 42
):
    """
    Experiment 1: Simulate GD on arbitrary number of neurons with arbitrary initialization.
    
    Args:
        k: Number of neurons
        num_iterations: Number of GD iterations
        learning_rate: Learning rate
        init_type: Initialization type
        seed: Random seed
    """
    print(f"\n=== Experiment 1: {k} neurons, {init_type} initialization ===")
    
    # Initialize network
    params = initialize_network(k, init_type=init_type, seed=seed)
    print_initial_params(params, "Initial Parameters")
    
    # Create dataset
    x, y = create_dataset(symmetric=True)
    print(f"Dataset: x={x}, y={y}")
    
    # Train
    params_history, losses, boundary_counts = train_gd(
        params, x, y, learning_rate, num_iterations,
        track_boundaries=True
    )
    
    print(f"\nTraining completed:")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Final boundaries: {boundary_counts[-1]}")
    print(f"  Final parameters:")
    print(f"    w: {params_history[-1].w}")
    print(f"    b: {params_history[-1].b}")
    print(f"    v: {params_history[-1].v}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss over time
    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss vs Iteration')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True)
    
    # Boundary count over time
    axes[0, 1].plot(boundary_counts)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Number of Decision Boundaries')
    axes[0, 1].set_title('Decision Boundaries vs Iteration')
    axes[0, 1].grid(True)
    
    # Final network output
    x_plot = np.linspace(-3, 3, 1000)
    y_plot = network_forward(params_history[-1], x_plot)
    axes[1, 0].plot(x_plot, y_plot, label='Network output')
    axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].scatter(x, [0]*len(x), c=['red' if yi < 0 else 'blue' for yi in y], 
                       s=100, zorder=5, label='Data points')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Phi(x)')
    axes[1, 0].set_title('Final Network Output')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Parameter evolution (show first few parameters)
    for j in range(min(3, k)):
        w_vals = [p.w[j] for p in params_history]
        axes[1, 1].plot(w_vals, label=f'w_{j+1}')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Parameter Value')
    axes[1, 1].set_title('Parameter Evolution (first 3)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'experiment_1_k{k}_{init_type}.png', dpi=150)
    print(f"\nPlot saved to experiment_1_k{k}_{init_type}.png")
    
    return params_history, losses, boundary_counts


def experiment_2_boundary_count(
    k: int = 2,
    num_iterations: int = 1000,
    learning_rate: float = 0.01,
    init_type: str = "thesis",
    w1_init: float = 1.0,
    b1_init: float = 1.0,
    M: float = 10.0
):
    """
    Experiment 2: Compute number of x-axis intersections as a function of iteration.
    
    Args:
        k: Number of neurons
        num_iterations: Number of GD iterations
        learning_rate: Learning rate
        init_type: Initialization type
        w1_init, b1_init, M: Parameters for thesis initialization
    """
    print(f"\n=== Experiment 2: Boundary count over time ===")
    
    # Initialize network
    params = initialize_network(k, init_type=init_type, 
                               w1_init=w1_init, b1_init=b1_init, M=M)
    
    # Create dataset
    x, y = create_dataset(symmetric=True)
    
    # Train and track boundaries
    params_history, losses, boundary_counts = train_gd(
        params, x, y, learning_rate, num_iterations,
        track_boundaries=True, x_range=(-10, 10)
    )
    
    # Plot boundary count
    plt.figure(figsize=(10, 6))
    plt.plot(boundary_counts, linewidth=2)
    plt.xlabel('Iteration t', fontsize=12)
    plt.ylabel('Number of Decision Boundaries', fontsize=12)
    plt.title(f'Number of x-axis Intersections vs Iteration (k={k})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('experiment_2_boundary_count.png', dpi=150)
    print(f"Plot saved to experiment_2_boundary_count.png")
    print(f"Initial boundaries: {boundary_counts[0]}")
    print(f"Final boundaries: {boundary_counts[-1]}")
    print(f"Max boundaries: {max(boundary_counts)}")
    
    return boundary_counts


def experiment_3_robust_case(
    k: int = 2,
    num_iterations: int = 1000,
    learning_rate: float = 0.01,
    shifts: List[float] = [0.0, -0.3, -0.5, -0.7],
    init_type: str = "thesis",
    w1_init: float = 1.0,
    b1_init: float = 1.0,
    M: float = 10.0
):
    """
    Experiment 3: Test robust case (training set moved closer to origin).
    
    Args:
        k: Number of neurons
        num_iterations: Number of GD iterations
        learning_rate: Learning rate
        shifts: List of shifts to apply (negative = move closer to origin)
        init_type: Initialization type
        w1_init, b1_init, M: Parameters for thesis initialization
    """
    print(f"\n=== Experiment 3: Robust case (shifted dataset) ===")
    
    results = {}
    
    for shift in shifts:
        print(f"\nTesting shift = {shift}")
        
        # Initialize network
        params = initialize_network(k, init_type=init_type,
                                   w1_init=w1_init, b1_init=b1_init, M=M)
        
        # Create shifted dataset (closer to origin)
        x, y = create_dataset(symmetric=True, shift=shift)
        print(f"  Dataset: x={x}, y={y}")
        
        # Train
        params_history, losses, boundary_counts = train_gd(
            params, x, y, learning_rate, num_iterations,
            track_boundaries=True
        )
        
        results[shift] = {
            'losses': losses,
            'boundary_counts': boundary_counts,
            'final_params': params_history[-1],
            'x': x
        }
        
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Final boundaries: {boundary_counts[-1]}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss comparison
    for shift in shifts:
        axes[0, 0].plot(results[shift]['losses'], label=f'shift={shift}')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss vs Iteration (Different Shifts)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Boundary count comparison
    for shift in shifts:
        axes[0, 1].plot(results[shift]['boundary_counts'], label=f'shift={shift}')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Number of Decision Boundaries')
    axes[0, 1].set_title('Boundaries vs Iteration (Different Shifts)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Final outputs comparison
    x_plot = np.linspace(-2, 2, 1000)
    for shift in shifts:
        y_plot = network_forward(results[shift]['final_params'], x_plot)
        axes[1, 0].plot(x_plot, y_plot, label=f'shift={shift}')
    axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Phi(x)')
    axes[1, 0].set_title('Final Network Outputs')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Data points visualization
    for shift in shifts:
        x_data = results[shift]['x']
        axes[1, 1].scatter(x_data, [shift]*len(x_data), 
                          label=f'shift={shift}', s=100, alpha=0.7)
    axes[1, 1].axvline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Shift')
    axes[1, 1].set_title('Data Point Positions')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('experiment_3_robust_case.png', dpi=150)
    print(f"\nPlot saved to experiment_3_robust_case.png")
    
    return results


def experiment_4_non_symmetric(
    k: int = 2,
    num_iterations: int = 1000,
    learning_rate: float = 0.01,
    init_type: str = "random",
    seed: int = 42
):
    """
    Experiment 4: Simulations over non-symmetric data.
    
    Args:
        k: Number of neurons
        num_iterations: Number of GD iterations
        learning_rate: Learning rate
        init_type: Initialization type
        seed: Random seed
    """
    print(f"\n=== Experiment 4: Non-symmetric data ===")
    
    # Test multiple non-symmetric configurations
    configs = [
        {"x": np.array([-0.5, 1.5]), "y": np.array([-1.0, 1.0]), "name": "asymmetric_1"},
        {"x": np.array([-1.5, 0.5]), "y": np.array([-1.0, 1.0]), "name": "asymmetric_2"},
        {"x": np.array([-0.8, 1.2]), "y": np.array([-1.0, 1.0]), "name": "asymmetric_3"},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting {config['name']}: x={config['x']}, y={config['y']}")
        
        # Initialize network
        params = initialize_network(k, init_type=init_type, seed=seed)
        
        # Train
        params_history, losses, boundary_counts = train_gd(
            params, config['x'], config['y'], learning_rate, num_iterations,
            track_boundaries=True
        )
        
        results[config['name']] = {
            'losses': losses,
            'boundary_counts': boundary_counts,
            'final_params': params_history[-1],
            'x': config['x'],
            'y': config['y']
        }
        
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Final boundaries: {boundary_counts[-1]}")
    
    # Compare with symmetric case
    print(f"\nTesting symmetric case for comparison")
    params_sym = initialize_network(k, init_type=init_type, seed=seed)
    x_sym, y_sym = create_dataset(symmetric=True)
    params_history_sym, losses_sym, boundary_counts_sym = train_gd(
        params_sym, x_sym, y_sym, learning_rate, num_iterations,
        track_boundaries=True
    )
    results['symmetric'] = {
        'losses': losses_sym,
        'boundary_counts': boundary_counts_sym,
        'final_params': params_history_sym[-1],
        'x': x_sym,
        'y': y_sym
    }
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss comparison
    for name in results:
        axes[0, 0].plot(results[name]['losses'], label=name)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss vs Iteration (Symmetric vs Non-symmetric)')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Boundary count comparison
    for name in results:
        axes[0, 1].plot(results[name]['boundary_counts'], label=name)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Number of Decision Boundaries')
    axes[0, 1].set_title('Boundaries vs Iteration')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Final outputs comparison
    x_plot = np.linspace(-2, 2, 1000)
    for name in results:
        y_plot = network_forward(results[name]['final_params'], x_plot)
        axes[1, 0].plot(x_plot, y_plot, label=name, alpha=0.7)
    axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Phi(x)')
    axes[1, 0].set_title('Final Network Outputs')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Data points visualization
    for name in results:
        x_data = results[name]['x']
        y_data = results[name]['y']
        colors = ['red' if yi < 0 else 'blue' for yi in y_data]
        axes[1, 1].scatter(x_data, [0]*len(x_data), c=colors, 
                          label=name, s=100, alpha=0.7)
    axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y (dummy)')
    axes[1, 1].set_title('Data Point Positions')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('experiment_4_non_symmetric.png', dpi=150)
    print(f"\nPlot saved to experiment_4_non_symmetric.png")
    
    return results


def example_initialization_options():
    """
    Example demonstrating different initialization options, especially for controlling w_j.
    """
    print("\n" + "=" * 60)
    print("Example: Initialization Options")
    print("=" * 60)
    
    k = 4
    
    # Example 1: Binary initialization (w_j = +1/-1)
    print("\n1. Binary initialization (w_j = +1/-1):")
    params1 = initialize_network(k=k, init_type="binary", v_binary=True, seed=42)
    print_initial_params(params1, "Binary w_j initialization")
    
    # Example 2: Explicit w_binary list
    print("\n2. Explicit w_binary list:")
    params2 = initialize_network(k=k, w_binary=[1, -1, 1, -1], v_binary=True, seed=42)
    print_initial_params(params2, "Explicit w_binary=[1, -1, 1, -1]")
    
    # Example 3: Custom explicit arrays
    print("\n3. Custom explicit arrays:")
    w_custom = np.array([1.0, -1.0, 0.5, -0.5])
    b_custom = np.array([0.1, -0.1, 0.2, -0.2])
    v_custom = np.array([1.0, -1.0, 1.0, -1.0])
    params3 = initialize_network(k=k, w_init=w_custom, b_init=b_custom, v_init=v_custom)
    print_initial_params(params3, "Custom explicit arrays")
    
    # Example 4: Thesis initialization (k=2)
    print("\n4. Thesis initialization (k=2):")
    params4 = initialize_network(k=2, init_type="thesis", w1_init=1.0, b1_init=1.0, M=10.0)
    print_initial_params(params4, "Thesis initialization")
    
    # Example 5: Random initialization
    print("\n5. Random initialization:")
    params5 = initialize_network(k=k, init_type="random", seed=42)
    print_initial_params(params5, "Random initialization")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run all experiments
    
    print("=" * 60)
    print("Thesis Experiments: GD Convergence and Adversarial Robustness")
    print("=" * 60)
    
    # Show initialization options
    example_initialization_options()
    
    # Experiment 1: Arbitrary neurons and initialization
    experiment_1_arbitrary_neurons(k=5, num_iterations=500, learning_rate=0.01, 
                                   init_type="random", seed=42)
    
    # Experiment 2: Boundary count over time
    experiment_2_boundary_count(k=2, num_iterations=1000, learning_rate=0.01,
                                init_type="thesis", w1_init=1.0, b1_init=1.0, M=10.0)
    
    # Experiment 3: Robust case
    experiment_3_robust_case(k=2, num_iterations=1000, learning_rate=0.01,
                             shifts=[0.0, -0.3, -0.5, -0.7],
                             init_type="thesis", w1_init=1.0, b1_init=1.0, M=10.0)
    
    # Experiment 4: Non-symmetric data
    experiment_4_non_symmetric(k=2, num_iterations=1000, learning_rate=0.01,
                               init_type="random", seed=42)
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)
