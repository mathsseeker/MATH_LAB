"""
Example 2: Optimization Techniques for Industrial Engineering
==============================================================

This example demonstrates various optimization techniques used
in Industrial Engineering applications.

Author: MATH_LAB Course
"""

import numpy as np
from scipy.optimize import linprog, minimize
import matplotlib.pyplot as plt

def linear_programming_example():
    """Demonstrate linear programming for production planning."""
    print("=" * 60)
    print("LINEAR PROGRAMMING: PRODUCTION PLANNING")
    print("=" * 60)
    
    print("\nProblem: Maximize profit from two products")
    print("Product A: $50 profit, requires 2 hours labor, 1 kg material")
    print("Product B: $60 profit, requires 3 hours labor, 2 kg material")
    print("Constraints: 100 hours labor, 80 kg material available")
    
    # Objective function (minimize negative profit = maximize profit)
    c = [-50, -60]
    
    # Inequality constraints: Ax <= b
    A_ub = [
        [2, 3],   # Labor constraint
        [1, 2]    # Material constraint
    ]
    b_ub = [100, 80]
    
    # Bounds
    bounds = [(0, None), (0, None)]
    
    # Solve
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    print(f"\nOptimal Solution:")
    print(f"  Product A: {result.x[0]:.2f} units")
    print(f"  Product B: {result.x[1]:.2f} units")
    print(f"  Maximum Profit: ${-result.fun:.2f}")
    
    # Check constraint utilization
    labor_used = 2*result.x[0] + 3*result.x[1]
    material_used = 1*result.x[0] + 2*result.x[1]
    
    print(f"\nResource Utilization:")
    print(f"  Labor: {labor_used:.2f}/100 hours ({labor_used/100*100:.1f}%)")
    print(f"  Material: {material_used:.2f}/80 kg ({material_used/80*100:.1f}%)")
    
    # Visualize feasible region
    visualize_linear_program(result)

def visualize_linear_program(result):
    """Visualize the feasible region and optimal point."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create grid
    x = np.linspace(0, 60, 300)
    
    # Constraint lines
    y1 = (100 - 2*x) / 3  # Labor constraint
    y2 = (80 - x) / 2      # Material constraint
    
    # Plot constraints
    ax.plot(x, y1, 'r-', linewidth=2, label='Labor: 2A + 3B ≤ 100')
    ax.plot(x, y2, 'b-', linewidth=2, label='Material: A + 2B ≤ 80')
    
    # Fill feasible region
    y_feasible = np.minimum(y1, y2)
    y_feasible = np.maximum(y_feasible, 0)
    ax.fill_between(x, 0, y_feasible, where=(y_feasible>=0), alpha=0.3, color='green', label='Feasible Region')
    
    # Plot optimal point
    ax.plot(result.x[0], result.x[1], 'r*', markersize=20, label=f'Optimal: ({result.x[0]:.1f}, {result.x[1]:.1f})')
    
    # Iso-profit lines
    for profit in [1000, 1500, 2000, -result.fun]:
        y_profit = (profit - 50*x) / 60
        if profit == -result.fun:
            ax.plot(x, y_profit, 'g--', linewidth=2, alpha=0.7, label=f'Max Profit: ${profit:.0f}')
        else:
            ax.plot(x, y_profit, 'gray', linewidth=1, alpha=0.5)
    
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 50)
    ax.set_xlabel('Product A (units)', fontsize=12)
    ax.set_ylabel('Product B (units)', fontsize=12)
    ax.set_title('Linear Programming: Production Planning', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('linear_programming.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'linear_programming.png'")

def eoq_optimization():
    """Economic Order Quantity optimization."""
    print("\n" + "=" * 60)
    print("ECONOMIC ORDER QUANTITY (EOQ) OPTIMIZATION")
    print("=" * 60)
    
    # Parameters
    D = 10000  # Annual demand
    S = 50     # Ordering cost per order
    H = 2      # Holding cost per unit per year
    P = 10     # Purchase cost per unit
    
    print(f"\nParameters:")
    print(f"  Annual Demand (D): {D} units")
    print(f"  Ordering Cost (S): ${S} per order")
    print(f"  Holding Cost (H): ${H} per unit per year")
    print(f"  Purchase Cost (P): ${P} per unit")
    
    # Define cost function
    def total_cost(Q):
        if Q <= 0:
            return float('inf')
        ordering_cost = (D / Q) * S
        holding_cost = (Q / 2) * H
        purchase_cost = D * P
        return ordering_cost + holding_cost + purchase_cost
    
    # Analytical solution
    EOQ_analytical = np.sqrt((2 * D * S) / H)
    
    # Numerical optimization
    result = minimize(total_cost, x0=100, bounds=[(1, None)], method='L-BFGS-B')
    EOQ_numerical = result.x[0]
    
    print(f"\nOptimal Order Quantity:")
    print(f"  Analytical (EOQ formula): {EOQ_analytical:.2f} units")
    print(f"  Numerical optimization: {EOQ_numerical:.2f} units")
    
    # Calculate costs at optimal Q
    min_cost = total_cost(EOQ_analytical)
    ordering_cost = (D / EOQ_analytical) * S
    holding_cost = (EOQ_analytical / 2) * H
    purchase_cost = D * P
    
    print(f"\nCost Breakdown at Optimal Q:")
    print(f"  Ordering Cost: ${ordering_cost:.2f}")
    print(f"  Holding Cost: ${holding_cost:.2f}")
    print(f"  Purchase Cost: ${purchase_cost:.2f}")
    print(f"  Total Annual Cost: ${min_cost:.2f}")
    
    print(f"\nOperational Metrics:")
    print(f"  Number of Orders per Year: {D/EOQ_analytical:.2f}")
    print(f"  Time Between Orders: {365/(D/EOQ_analytical):.2f} days")
    
    # Visualize cost function
    visualize_eoq(total_cost, EOQ_analytical)

def visualize_eoq(cost_func, optimal_Q):
    """Visualize EOQ cost function."""
    Q_range = np.linspace(50, 1000, 500)
    costs = [cost_func(q) for q in Q_range]
    
    plt.figure(figsize=(10, 6))
    plt.plot(Q_range, costs, 'b-', linewidth=2, label='Total Cost')
    plt.axvline(optimal_Q, color='r', linestyle='--', linewidth=2, label=f'Optimal Q = {optimal_Q:.0f}')
    plt.plot(optimal_Q, cost_func(optimal_Q), 'r*', markersize=15)
    
    plt.xlabel('Order Quantity (Q)', fontsize=12)
    plt.ylabel('Total Annual Cost ($)', fontsize=12)
    plt.title('Economic Order Quantity (EOQ) Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('eoq_optimization.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'eoq_optimization.png'")

def constrained_optimization():
    """Nonlinear optimization with constraints."""
    print("\n" + "=" * 60)
    print("CONSTRAINED NONLINEAR OPTIMIZATION")
    print("=" * 60)
    
    print("\nProblem: Minimize production cost with quality constraints")
    print("Cost function: C(x,y) = x² + 2y² - 2xy + 10x + 5y")
    print("Constraints: x + y ≥ 10, x ≥ 0, y ≥ 0")
    
    # Objective function
    def objective(vars):
        x, y = vars
        return x**2 + 2*y**2 - 2*x*y + 10*x + 5*y
    
    # Constraint function (must be >= 0)
    def constraint(vars):
        x, y = vars
        return x + y - 10
    
    # Constraint dictionary
    constraints = {'type': 'ineq', 'fun': constraint}
    
    # Bounds
    bounds = [(0, None), (0, None)]
    
    # Initial guess
    x0 = [5, 5]
    
    # Solve
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    print(f"\nOptimal Solution:")
    print(f"  x = {result.x[0]:.4f}")
    print(f"  y = {result.x[1]:.4f}")
    print(f"  Minimum Cost: ${result.fun:.4f}")
    print(f"  Constraint satisfaction: x + y = {result.x[0] + result.x[1]:.4f} ≥ 10")
    
    # Check if constraint is binding
    if abs(result.x[0] + result.x[1] - 10) < 1e-4:
        print("\nConstraint is BINDING (active at optimum)")
    else:
        print("\nConstraint is NOT binding")

def sensitivity_analysis():
    """Perform sensitivity analysis on optimization problem."""
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    print("\nAnalyzing how optimal solution changes with parameters")
    
    # Base case: profit coefficients
    base_profits = np.array([50, 60])
    
    # Vary Product A profit from 40 to 70
    profit_A_range = np.linspace(40, 70, 20)
    optimal_A = []
    optimal_B = []
    max_profits = []
    
    for profit_A in profit_A_range:
        c = [-profit_A, -60]
        A_ub = [[2, 3], [1, 2]]
        b_ub = [100, 80]
        bounds = [(0, None), (0, None)]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        optimal_A.append(result.x[0])
        optimal_B.append(result.x[1])
        max_profits.append(-result.fun)
    
    # Visualize sensitivity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(profit_A_range, optimal_A, 'b-', linewidth=2, marker='o', label='Product A')
    ax1.plot(profit_A_range, optimal_B, 'r-', linewidth=2, marker='s', label='Product B')
    ax1.axvline(50, color='gray', linestyle='--', alpha=0.5, label='Base Case')
    ax1.set_xlabel('Profit of Product A ($)', fontsize=11)
    ax1.set_ylabel('Optimal Production (units)', fontsize=11)
    ax1.set_title('Sensitivity: Production Quantities', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(profit_A_range, max_profits, 'g-', linewidth=2, marker='o')
    ax2.axvline(50, color='gray', linestyle='--', alpha=0.5, label='Base Case')
    ax2.set_xlabel('Profit of Product A ($)', fontsize=11)
    ax2.set_ylabel('Maximum Total Profit ($)', fontsize=11)
    ax2.set_title('Sensitivity: Total Profit', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSensitivity analysis saved as 'sensitivity_analysis.png'")
    
    print(f"\nKey Insights:")
    print(f"  When Product A profit < $55: Produce more B")
    print(f"  When Product A profit > $55: Produce more A")
    print(f"  Total profit increases linearly with Product A profit")

if __name__ == "__main__":
    linear_programming_example()
    eoq_optimization()
    constrained_optimization()
    sensitivity_analysis()
    
    print("\n" + "=" * 60)
    print("All optimization examples completed successfully!")
    print("=" * 60)
