# Lab 3: Optimization Techniques for Industrial Engineering

## Objectives
- Implement linear programming solutions
- Apply optimization to resource allocation problems
- Use SciPy optimization tools
- Solve constrained and unconstrained optimization problems

## Background

Optimization is central to Industrial Engineering for:
- Production planning and scheduling
- Inventory management
- Transportation and logistics
- Workforce allocation
- Cost minimization and profit maximization

## Exercises

### Exercise 1: Linear Programming - Production Optimization (25 points)

A factory produces two products: A and B.

**Constraints:**
- Product A requires 2 hours of labor, Product B requires 3 hours
- Total available labor: 100 hours
- Product A requires 1 kg of raw material, Product B requires 2 kg
- Total available raw material: 80 kg
- At least 10 units of Product A must be produced
- Production quantities must be non-negative

**Objective:** Maximize profit where Product A gives $50 profit and Product B gives $60 profit per unit.

**Tasks:**
1. Formulate the linear programming problem
2. Solve using SciPy's linprog
3. Interpret the optimal solution
4. Identify binding and non-binding constraints
5. Calculate shadow prices

### Exercise 2: Transportation Problem (25 points)

A company has 3 factories (F1, F2, F3) and 4 warehouses (W1, W2, W3, W4).

**Supply at factories:** F1=50, F2=60, F3=40 units  
**Demand at warehouses:** W1=30, W2=40, W3=35, W4=45 units

**Transportation costs (per unit):**
```
      W1   W2   W3   W4
F1    8    6    10   9
F2    9    12   13   7
F3    14   9    16   5
```

**Tasks:**
1. Formulate as a linear programming problem
2. Find the optimal transportation plan
3. Calculate minimum total transportation cost
4. Identify which routes are used

### Exercise 3: Nonlinear Optimization - Inventory Management (25 points)

Economic Order Quantity (EOQ) problem with the following parameters:
- Annual demand (D): 10,000 units
- Ordering cost (S): $50 per order
- Holding cost (H): $2 per unit per year
- Purchase cost (P): $10 per unit

**Cost function:**
```
Total Cost = (D/Q)*S + (Q/2)*H + D*P
```

Where Q is the order quantity.

**Tasks:**
1. Define the total cost function
2. Find the optimal order quantity using optimization
3. Calculate the minimum total cost
4. Determine the optimal number of orders per year
5. Plot the cost function around the optimal point

### Exercise 4: Multi-Objective Optimization (25 points)

A production schedule must consider two objectives:
1. Minimize production time: T(x,y) = 2x² + 3y² - 4xy
2. Minimize cost: C(x,y) = 50x + 40y

Subject to constraints:
- x + y ≥ 10 (minimum production)
- x ≤ 20, y ≤ 15 (capacity limits)
- x, y ≥ 0

**Tasks:**
1. Solve for each objective separately
2. Find the Pareto frontier (trade-off curve)
3. Use weighted sum method: minimize αT + (1-α)C for α = 0.5
4. Visualize the solutions

## Starter Code

```python
import numpy as np
from scipy.optimize import linprog, minimize, LinearConstraint
import matplotlib.pyplot as plt

print("=== Lab 3: Optimization Techniques ===\n")

# Exercise 1: Linear Programming
print("Exercise 1: Production Optimization")

# Objective function coefficients (for minimization, so negate for max)
c = [-50, -60]  # Negative because linprog minimizes

# Inequality constraints: Ax <= b
A_ub = [
    [2, 3],   # Labor constraint
    [1, 2],   # Material constraint
    [-1, 0]   # Minimum production of A (rewritten as -x <= -10)
]
b_ub = [100, 80, -10]

# Bounds for variables (x >= 0, y >= 0)
bounds = [(0, None), (0, None)]

# Solve
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

print(f"Optimal solution: A = {result.x[0]:.2f}, B = {result.x[1]:.2f}")
print(f"Maximum profit: ${-result.fun:.2f}")  # Negate back to get max

# Your additional analysis here

# Exercise 2: Transportation Problem
print("\nExercise 2: Transportation Problem")
# Your code here

# Exercise 3: EOQ Problem
print("\nExercise 3: Inventory Optimization")

D = 10000  # Annual demand
S = 50     # Ordering cost
H = 2      # Holding cost
P = 10     # Purchase cost

def total_cost(Q):
    if Q <= 0:
        return float('inf')
    return (D/Q)*S + (Q/2)*H + D*P

# Your code here

# Exercise 4: Multi-objective Optimization
print("\nExercise 4: Multi-objective Optimization")

def time_objective(x):
    return 2*x[0]**2 + 3*x[1]**2 - 4*x[0]*x[1]

def cost_objective(x):
    return 50*x[0] + 40*x[1]

# Your code here
```

## Visualization Requirements

Create the following plots:

1. **Exercise 1**: Feasible region and optimal point
2. **Exercise 3**: Total cost vs. order quantity curve
3. **Exercise 4**: Pareto frontier showing trade-offs

## Submission Guidelines

1. Submit `lab3_yourname.py` with complete solutions
2. Include all required plots saved as PNG files
3. Create a report (PDF/markdown) containing:
   - Problem formulation for each exercise
   - Solution methodology
   - Optimal solutions with interpretation
   - Sensitivity analysis where applicable
   - Discussion of practical implications

## Expected Output

```
=== Exercise 1: Production Optimization ===
Optimal production: A = X units, B = Y units
Maximum profit: $Z
Binding constraints: [list]
Shadow prices: [values]

=== Exercise 2: Transportation Problem ===
Optimal transportation plan: [matrix]
Minimum total cost: $X

=== Exercise 3: Inventory Optimization ===
Optimal order quantity: X units
Minimum total annual cost: $Y
Number of orders per year: Z

=== Exercise 4: Multi-objective Optimization ===
Time-optimal solution: (x, y) with time = T
Cost-optimal solution: (x, y) with cost = C
Compromise solution (α=0.5): (x, y)
```

## Grading Rubric

- Correct problem formulation: 25%
- Correct solutions: 40%
- Code quality and comments: 15%
- Visualizations: 10%
- Report and interpretation: 10%

## Bonus Challenges (15 points)

1. **Sensitivity Analysis**: For Exercise 1, analyze how the optimal solution changes if:
   - Labor increases to 120 hours
   - Profit from Product A increases by 10%

2. **Genetic Algorithm**: Implement a simple genetic algorithm to solve Exercise 4

3. **Integer Programming**: Modify Exercise 1 to require integer production quantities

## Resources

- SciPy Optimization: https://docs.scipy.org/doc/scipy/reference/optimize.html
- Linear Programming: Hillier & Lieberman textbook
- Operations Research applications in IE

## Due Date

One week from lab session date
