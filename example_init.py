"""
Example script demonstrating how to control initial weights, including w_j = +1/-1.
"""

from main import (initialize_network, print_initial_params, get_theta_vector, 
                  train_gd, create_dataset, compute_margin, compute_margin_gap)
import numpy as np

print("=" * 70)
print("Examples: Controlling Initial Weights Theta")
print("=" * 70)

# Example 1: Initialize w_j to +1/-1 randomly
print("\n" + "-" * 70)
print("Example 1: Binary initialization (w_j = +1/-1 randomly)")
print("-" * 70)
params1 = initialize_network(k=5, init_type="binary", v_binary=True, seed=42)
print_initial_params(params1)

# Example 2: Explicitly specify w_j = +1/-1
print("\n" + "-" * 70)
print("Example 2: Explicitly set w_j = [1, -1, 1, -1, 1]")
print("-" * 70)
params2 = initialize_network(k=5, w_binary=[1, -1, 1, -1, 1], v_binary=True, seed=42)
print_initial_params(params2)

# Example 3: Fully custom initialization
print("\n" + "-" * 70)
print("Example 3: Fully custom initialization (all parameters specified)")
print("-" * 70)
w_custom = np.array([1.0, -1.0, 0.5, -0.5, 1.0])
b_custom = np.array([0.1, -0.1, 0.2, -0.2, 0.0])
v_custom = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
params3 = initialize_network(k=5, w_init=w_custom, b_init=b_custom, v_init=v_custom)
print_initial_params(params3)

# Example 4: Thesis initialization (k=2)
print("\n" + "-" * 70)
print("Example 4: Thesis initialization (k=2, as in your paper)")
print("-" * 70)
params4 = initialize_network(k=2, init_type="thesis", w1_init=1.0, b1_init=1.0, M=10.0)
print_initial_params(params4)

# Example 5: Run training with binary w_j initialization
print("\n" + "-" * 70)
print("Example 5: Training with binary w_j initialization")
print("-" * 70)
params5 = initialize_network(k=3, w_binary=[1, -1, 1], v_binary=True, seed=42)
print_initial_params(params5, "Before training")

x, y = create_dataset(symmetric=True)
params_history, losses, boundary_counts, margins, margin_gaps = train_gd(
    params5, x, y, learning_rate=0.01, num_iterations=100, 
    track_boundaries=True, track_margin=True, optimal_margin=1.0
)

print(f"\nAfter {len(losses)} iterations:")
print(f"  Final loss: {losses[-1]:.6f}")
print(f"  Final boundaries: {boundary_counts[-1]}")
if margins:
    print(f"  Final margin: {margins[-1]:.6f}")
    print(f"  Final margin gap: {margin_gaps[-1]:.6f}")
print_initial_params(params_history[-1], "After training")

# Show Theta evolution
print(f"\nTheta norm evolution:")
print(f"  Initial ||Theta|| = {np.linalg.norm(get_theta_vector(params5)):.6f}")
print(f"  Final ||Theta|| = {np.linalg.norm(get_theta_vector(params_history[-1])):.6f}")

# Example 6: Margin computation
print("\n" + "-" * 70)
print("Example 6: Computing margin directly")
print("-" * 70)
margin = compute_margin(params_history[-1], x, y)
margin_gap = compute_margin_gap(params_history[-1], x, y, optimal_margin=1.0)
print(f"  Margin: {margin:.6f}")
print(f"  Margin gap: {margin_gap:.6f}")
print(f"  Optimal margin: 1.0")

print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)
print("""
To control initial weights Theta:

1. Binary w_j (+1/-1):
   params = initialize_network(k=5, init_type="binary", v_binary=True)

2. Explicit w_binary list:
   params = initialize_network(k=5, w_binary=[1, -1, 1, -1, 1])

3. Fully custom:
   params = initialize_network(k=5, w_init=w_array, b_init=b_array, v_init=v_array)

4. Get Theta vector:
   theta = get_theta_vector(params)  # Returns [w_1,...,w_k, b_1,...,b_k, v_1,...,v_k]

5. Print parameters:
   print_initial_params(params)

6. Track margin during training:
   params_history, losses, _, margins, margin_gaps = train_gd(
       params, x, y, learning_rate, num_iterations,
       track_margin=True, optimal_margin=1.0
   )

7. Compute margin directly:
   margin = compute_margin(params, x, y)
   margin_gap = compute_margin_gap(params, x, y, optimal_margin=1.0)
""")
