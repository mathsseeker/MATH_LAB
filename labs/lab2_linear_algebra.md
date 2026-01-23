# Lab 2: Linear Algebra Applications in Industrial Engineering

## Objectives
- Solve systems of linear equations
- Apply matrix methods to real engineering problems
- Understand linear transformations
- Work with eigenvalues and eigenvectors

## Background

Linear algebra is fundamental in Industrial Engineering for:
- Production planning and scheduling
- Supply chain optimization
- Quality control systems
- Network flow problems

## Exercises

### Exercise 1: Production Planning (25 points)

A manufacturing company produces three products (P1, P2, P3) using three resources (R1, R2, R3).

Resource requirements per unit:
- P1: 2 units R1, 3 units R2, 1 unit R3
- P2: 1 unit R1, 2 units R2, 3 units R3
- P3: 3 units R1, 1 unit R2, 2 units R3

Available resources: 100 units R1, 120 units R2, 90 units R3

**Task**: Set up and solve the system of linear equations to find if there's a production plan that uses exactly all available resources.

```python
# Resource constraint matrix A and available resources b
# Ax = b, where x = [P1, P2, P3]
```

### Exercise 2: Network Flow Analysis (25 points)

A logistics network has 4 nodes (warehouses). The flow balance equations are:

```
Node 1:  x1 - x2 - x3 = 10
Node 2:  x2 - x4 = -5
Node 3:  x3 + x4 - x5 = 5
Node 4:  x5 = 0
```

Where x1, x2, x3, x4, x5 represent flows on different routes.

**Task**: 
1. Write the system in matrix form Ax = b
2. Solve for the flow values
3. Verify your solution satisfies all constraints

### Exercise 3: Quality Control - Principal Component Analysis (25 points)

A quality control system measures 3 dimensions of manufactured parts. The covariance matrix is:

```
C = [[4.0, 2.0, 0.5],
     [2.0, 3.0, 1.0],
     [0.5, 1.0, 2.0]]
```

**Task**:
1. Calculate eigenvalues and eigenvectors
2. Identify the principal components (eigenvectors with largest eigenvalues)
3. Determine which dimensions contribute most to variation
4. Calculate the percentage of variance explained by each principal component

### Exercise 4: Markov Chain - Machine States (25 points)

A manufacturing machine has three states: Working (W), Maintenance (M), Failed (F).

Transition probability matrix:
```
     W    M    F
W [0.7  0.2  0.1]
M [0.5  0.3  0.2]
F [0.0  0.6  0.4]
```

**Task**:
1. Find the steady-state probabilities (eigenvector for eigenvalue 1)
2. If the machine starts in Working state, what's the probability distribution after 5 transitions?
3. Calculate the long-term percentage of time in each state

## Starter Code

```python
import numpy as np
from numpy.linalg import solve, inv, eig, matrix_power

print("=== Lab 2: Linear Algebra Applications ===\n")

# Exercise 1: Production Planning
print("Exercise 1: Production Planning")
# Resource requirement matrix
A1 = np.array([[2, 1, 3],
               [3, 2, 1],
               [1, 3, 2]])

# Available resources
b1 = np.array([100, 120, 90])

# Solve Ax = b
# Your code here

# Exercise 2: Network Flow
print("\nExercise 2: Network Flow Analysis")
# Your code here

# Exercise 3: PCA
print("\nExercise 3: Quality Control - PCA")
C = np.array([[4.0, 2.0, 0.5],
              [2.0, 3.0, 1.0],
              [0.5, 1.0, 2.0]])

# Your code here

# Exercise 4: Markov Chain
print("\nExercise 4: Machine State Analysis")
P = np.array([[0.7, 0.2, 0.1],
              [0.5, 0.3, 0.2],
              [0.0, 0.6, 0.4]])

# Your code here
```

## Submission Guidelines

1. Submit `lab2_yourname.py` with complete solutions
2. Include detailed comments explaining each step
3. Provide interpretation of results for each exercise
4. Create a separate report (PDF or markdown) with:
   - Problem analysis
   - Solution methodology
   - Results interpretation
   - Conclusions

## Expected Output Format

```
=== Exercise 1: Production Planning ===
Production quantities: P1 = X, P2 = Y, P3 = Z
Resource utilization: [verification]

=== Exercise 2: Network Flow ===
Flow values: x1 = A, x2 = B, ...
Verification: [check flow balance]

=== Exercise 3: PCA ===
Eigenvalues: [λ1, λ2, λ3]
Principal components: [eigenvectors]
Variance explained: [percentages]

=== Exercise 4: Markov Chain ===
Steady-state probabilities: [pw, pm, pf]
5-step transition from W: [probabilities]
Long-term state distribution: [analysis]
```

## Grading Rubric

- Correct solutions: 60%
- Code quality: 15%
- Interpretation and analysis: 20%
- Report quality: 5%

## Bonus Challenge (10 points)

Implement a function to solve any n×n system using:
1. Gaussian elimination (from scratch)
2. Compare results with NumPy's solve function
3. Measure computational time for both methods

## Resources

- Linear Algebra concepts: Gilbert Strang's textbook
- NumPy linear algebra: https://numpy.org/doc/stable/reference/routines.linalg.html
- Markov Chains in IE: Industrial modeling references

## Due Date

One week from lab session date
