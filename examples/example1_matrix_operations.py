"""
Example 1: Matrix Operations for Industrial Engineering
========================================================

This example demonstrates basic matrix operations commonly used
in Industrial Engineering applications.

Author: MATH_LAB Course
"""

import numpy as np
import matplotlib.pyplot as plt

def demonstrate_matrix_basics():
    """Demonstrate basic matrix operations."""
    print("=" * 60)
    print("MATRIX OPERATIONS FOR INDUSTRIAL ENGINEERING")
    print("=" * 60)
    
    # Example 1: Resource Allocation Matrix
    print("\n1. Resource Allocation in Manufacturing")
    print("-" * 40)
    
    # Products x Resources matrix
    # Rows: Products (P1, P2, P3)
    # Columns: Resources (Labor, Material, Machine)
    resources = np.array([
        [2, 3, 1],  # Product 1
        [1, 2, 3],  # Product 2
        [3, 1, 2]   # Product 3
    ])
    
    print("Resource Requirements Matrix:")
    print(resources)
    print("\nRows: Products, Columns: [Labor, Material, Machine]")
    
    # Production quantities
    production = np.array([10, 15, 8])
    
    # Total resource usage
    total_usage = resources.T @ production
    
    print(f"\nProduction plan: {production} units")
    print(f"Total resource usage:")
    print(f"  Labor: {total_usage[0]} hours")
    print(f"  Material: {total_usage[1]} kg")
    print(f"  Machine: {total_usage[2]} hours")
    
    # Example 2: Cost Calculation
    print("\n2. Multi-Product Cost Analysis")
    print("-" * 40)
    
    # Cost per unit of resource
    costs = np.array([20, 15, 30])  # Cost per unit [Labor, Material, Machine]
    
    # Total cost calculation
    unit_costs = resources @ costs
    total_cost = production @ unit_costs
    
    print(f"Cost per resource: {costs}")
    print(f"Cost per product unit: {unit_costs}")
    print(f"Total production cost: ${total_cost:.2f}")
    
    # Example 3: Inverse and System Solving
    print("\n3. Solving Production Requirements")
    print("-" * 40)
    
    # If we want specific resource usage, find required production
    desired_usage = np.array([50, 60, 55])
    
    if np.linalg.det(resources) != 0:
        required_production = np.linalg.solve(resources.T, desired_usage)
        print(f"To achieve usage {desired_usage}:")
        print(f"Required production: {required_production}")
        
        # Verify
        actual_usage = resources.T @ required_production
        print(f"Verification: {actual_usage}")
    else:
        print("System has no unique solution (singular matrix)")

def demonstrate_eigenvalues():
    """Demonstrate eigenvalue applications in quality control."""
    print("\n" + "=" * 60)
    print("EIGENVALUE ANALYSIS IN QUALITY CONTROL")
    print("=" * 60)
    
    # Covariance matrix of quality measurements
    cov_matrix = np.array([
        [4.0, 2.0, 0.5],
        [2.0, 3.0, 1.0],
        [0.5, 1.0, 2.0]
    ])
    
    print("\nCovariance Matrix of Quality Measurements:")
    print(cov_matrix)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalue magnitude
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print("\nEigenvalues (sorted):")
    for i, val in enumerate(eigenvalues):
        print(f"  λ{i+1} = {val:.4f}")
    
    # Calculate variance explained
    total_variance = np.sum(eigenvalues)
    variance_explained = eigenvalues / total_variance * 100
    
    print("\nVariance Explained by Each Principal Component:")
    cumulative = 0
    for i, var in enumerate(variance_explained):
        cumulative += var
        print(f"  PC{i+1}: {var:.2f}% (Cumulative: {cumulative:.2f}%)")
    
    print(f"\nFirst 2 components explain {cumulative:.2f}% of variance")
    print("This suggests we could reduce from 3 to 2 quality measurements")

def demonstrate_markov_chain():
    """Demonstrate Markov chain for machine state analysis."""
    print("\n" + "=" * 60)
    print("MARKOV CHAIN: MACHINE STATE ANALYSIS")
    print("=" * 60)
    
    # Transition matrix for machine states
    # States: [Working, Maintenance, Failed]
    P = np.array([
        [0.7, 0.2, 0.1],  # From Working
        [0.5, 0.3, 0.2],  # From Maintenance
        [0.0, 0.6, 0.4]   # From Failed
    ])
    
    print("\nTransition Probability Matrix:")
    print(P)
    print("States: [Working, Maintenance, Failed]")
    
    # Find steady-state distribution (eigenvector for eigenvalue 1)
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    
    # Find eigenvector for eigenvalue 1
    idx = np.argmin(np.abs(eigenvalues - 1))
    steady_state = np.real(eigenvectors[:, idx])
    steady_state = steady_state / np.sum(steady_state)
    
    print("\nSteady-State Probabilities:")
    states = ['Working', 'Maintenance', 'Failed']
    for state, prob in zip(states, steady_state):
        print(f"  {state}: {prob:.4f} ({prob*100:.2f}%)")
    
    # Simulate transitions
    print("\nState Evolution Starting from Working State:")
    current = np.array([1.0, 0.0, 0.0])  # Start in Working state
    
    print(f"Initial:     {current}")
    for step in range(1, 6):
        current = current @ P
        print(f"After {step} step(s): {current}")
    
    print(f"\nConverging to steady state: {steady_state}")

def visualize_matrix_properties():
    """Visualize matrix transformations."""
    print("\n" + "=" * 60)
    print("VISUALIZING LINEAR TRANSFORMATIONS")
    print("=" * 60)
    
    # Create a transformation matrix (scaling and rotation)
    theta = np.pi / 6  # 30 degrees
    scale = 1.5
    
    transform = np.array([
        [scale * np.cos(theta), -scale * np.sin(theta)],
        [scale * np.sin(theta), scale * np.cos(theta)]
    ])
    
    print("\nTransformation Matrix:")
    print(transform)
    
    # Original unit vectors
    original = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]]).T
    
    # Apply transformation
    transformed = transform @ original
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original
    ax1.plot(original[0], original[1], 'b-o', linewidth=2, markersize=8)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.set_title('Original Shape', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Transformed
    ax2.plot(transformed[0], transformed[1], 'r-o', linewidth=2, markersize=8)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    ax2.set_title('After Transformation', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig('matrix_transformation.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'matrix_transformation.png'")
    
    # Also show eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(transform)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")

if __name__ == "__main__":
    demonstrate_matrix_basics()
    demonstrate_eigenvalues()
    demonstrate_markov_chain()
    visualize_matrix_properties()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
