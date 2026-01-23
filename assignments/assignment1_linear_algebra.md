# Assignment 1: Matrix Operations and Linear Systems

## Due Date
Two weeks from assignment date

## Total Points: 100

## Instructions

1. Complete all problems showing your work
2. Submit Python code files with clear comments
3. Include a report (PDF or Markdown) with:
   - Problem analysis
   - Solution approach
   - Results and interpretation
   - Answers to discussion questions

## Problems

### Problem 1: Supply Chain Network (25 points)

A supply chain has the following structure:
- 3 Suppliers (S1, S2, S3)
- 2 Manufacturers (M1, M2)
- 4 Distribution Centers (D1, D2, D3, D4)

**Flow from Suppliers to Manufacturers (units per day):**
```
     M1   M2
S1   20   30
S2   25   15
S3   30   20
```

**Flow from Manufacturers to Distribution Centers:**
```
     D1   D2   D3   D4
M1   15   20   25   10
M2   10   15   20   15
```

**Tasks:**
1. Represent these flows as matrices F1 (Suppliers→Manufacturers) and F2 (Manufacturers→Distributors)
2. Calculate total flow from each supplier to each distribution center using matrix multiplication
3. If each unit costs $5 from suppliers to manufacturers and $8 from manufacturers to distributors, calculate the total daily transportation cost
4. Which distribution center receives the most units? From which supplier do these units originate?

### Problem 2: Production Scheduling (25 points)

A factory produces 4 products using 3 machines. The time (in hours) required on each machine for each product is:

```
        P1   P2   P3   P4
M1      2    3    1    4
M2      1    2    2    3
M3      3    1    4    2
```

Each machine is available 40 hours per week.

**Tasks:**
1. Write the system of equations representing the machine constraints if x1, x2, x3, x4 are the weekly production quantities
2. Is it possible to produce 8 units of P1, 10 units of P2, 6 units of P3, and 7 units of P4 per week? Verify by solving the system.
3. If the target is to use exactly all available machine hours, find one feasible production plan
4. Discuss whether the solution is unique or if multiple production plans exist

### Problem 3: Quality Control Analysis (30 points)

A manufacturing process has quality measurements on 3 dimensions for a sample of 100 products. The covariance matrix of these measurements is:

```python
Cov = [[9.0, 4.5, 2.0],
       [4.5, 6.0, 3.0],
       [2.0, 3.0, 5.0]]
```

**Tasks:**
1. Calculate the eigenvalues and eigenvectors of the covariance matrix
2. Determine the principal components (ranked by eigenvalue magnitude)
3. Calculate the percentage of total variance explained by each principal component
4. How many principal components are needed to explain at least 90% of the variance?
5. If we want to reduce quality inspections from 3 measurements to 2, which measurement would you eliminate? Justify your answer using the eigenvector components.
6. Create a visualization showing:
   - Eigenvalues (bar chart)
   - Cumulative variance explained
   - Eigenvector components (heatmap)

### Problem 4: Process Control - Markov Chain (20 points)

A production line can be in one of three states each hour:
- **Normal (N)**: Operating normally
- **Warning (W)**: Minor issues detected  
- **Down (D)**: Stopped for repairs

The transition probabilities per hour are:

```
        N     W     D
N     0.85  0.10  0.05
W     0.40  0.40  0.20
D     0.00  0.50  0.50
```

**Tasks:**
1. Calculate the steady-state probabilities for each state
2. If the line starts in Normal state, what is the probability distribution after:
   - 2 hours?
   - 5 hours?
   - 10 hours?
3. Calculate the expected long-term percentage of time the line is:
   - Operating normally
   - Down for repairs
4. If downtime costs $500/hour and warning state costs $100/hour, what is the expected cost per hour in the long run?
5. A process improvement is proposed that changes P(N→N) from 0.85 to 0.90 by reducing P(N→W) to 0.05. Calculate the new steady-state distribution and expected hourly cost. Is the improvement worthwhile?

## Coding Requirements

Your Python code should:
- Be well-organized with functions for each problem
- Include docstrings explaining each function
- Use NumPy for all matrix operations
- Include error checking (e.g., matrix dimensions, singular matrices)
- Generate all required visualizations
- Print results in a clear, formatted manner

## Report Requirements

Your report should include:

### For Each Problem:
1. **Problem Statement**: Briefly restate the problem in your own words
2. **Approach**: Explain your solution methodology
3. **Mathematical Formulation**: Show relevant equations and matrix representations
4. **Results**: Present numerical results with proper units and formatting
5. **Interpretation**: Explain what the results mean in the context of the problem
6. **Validation**: Show how you verified your solution is correct

### Overall:
- Introduction explaining the importance of these techniques in IE
- Conclusion summarizing key learnings
- References (if any external sources used)

## Submission Checklist

- [ ] Python code file(s) with all solutions
- [ ] Code runs without errors
- [ ] All visualizations generated and saved
- [ ] Report in PDF or Markdown format
- [ ] All problems attempted
- [ ] Code is well-commented
- [ ] Results are clearly presented

## Grading Rubric

| Category | Points |
|----------|--------|
| Correct solutions | 60 |
| Code quality and documentation | 15 |
| Report clarity and completeness | 15 |
| Visualizations | 10 |
| **Total** | **100** |

## Bonus Opportunities (+15 points max)

1. **Optimization Extension** (+5 points): For Problem 2, formulate and solve an optimization problem to maximize profit if products P1, P2, P3, P4 have profits of $20, $30, $25, and $35 respectively.

2. **Sensitivity Analysis** (+5 points): For Problem 4, analyze how the steady-state distribution changes as P(N→N) varies from 0.75 to 0.95. Create a visualization.

3. **3D Visualization** (+5 points): For Problem 3, create a 3D visualization of the principal components showing the direction of maximum variance.

## Tips

- Start early and test your code incrementally
- Use NumPy's linear algebra functions (`np.linalg.solve`, `np.linalg.eig`, etc.)
- Verify your results using alternative methods when possible
- Check dimensions of matrices before operations
- Use meaningful variable names
- Test edge cases

## Getting Help

- Review lab materials and examples
- Consult NumPy and SciPy documentation
- Discuss concepts (not solutions) with classmates
- Create an issue in the repository for questions

## Academic Integrity

This is an individual assignment. You may discuss concepts with others but must write your own code and report. Copying code from any source without citation is academic dishonesty.

---

Good luck!
