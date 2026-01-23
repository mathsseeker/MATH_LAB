# Mathematics for Industrial Engineering - Quick Reference Guide

## Linear Algebra

### Matrix Operations

#### Matrix Multiplication
```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = A @ B  # or np.dot(A, B)
```

#### Solving Linear Systems
```python
# Solve Ax = b
A = np.array([[2, 1], [1, 3]])
b = np.array([5, 7])

x = np.linalg.solve(A, b)
```

#### Eigenvalues and Eigenvectors
```python
# Get eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
```

#### Matrix Properties
```python
# Determinant
det_A = np.linalg.det(A)

# Inverse
A_inv = np.linalg.inv(A)

# Transpose
A_T = A.T

# Rank
rank_A = np.linalg.matrix_rank(A)
```

---

## Optimization

### Linear Programming
```python
from scipy.optimize import linprog

# Minimize: c^T x
# Subject to: A_ub @ x <= b_ub
#             A_eq @ x == b_eq
#             bounds

result = linprog(c, A_ub=A_ub, b_ub=b_ub, 
                 A_eq=A_eq, b_eq=b_eq,
                 bounds=bounds, method='highs')

print(f"Optimal x: {result.x}")
print(f"Optimal value: {result.fun}")
```

### Nonlinear Optimization
```python
from scipy.optimize import minimize

def objective(x):
    return x[0]**2 + x[1]**2

def constraint(x):
    return x[0] + x[1] - 1

constraints = {'type': 'eq', 'fun': constraint}
x0 = [0.5, 0.5]

result = minimize(objective, x0, constraints=constraints)
```

### Economic Order Quantity (EOQ)
```python
import numpy as np

D = 10000  # Annual demand
S = 50     # Ordering cost
H = 2      # Holding cost

# Optimal order quantity
EOQ = np.sqrt((2 * D * S) / H)

# Total cost
TC = (D/EOQ)*S + (EOQ/2)*H + D*P
```

---

## Statistics and Probability

### Descriptive Statistics
```python
import numpy as np

data = np.array([...])

mean = np.mean(data)
median = np.median(data)
std_dev = np.std(data)
variance = np.var(data)
```

### Probability Distributions
```python
from scipy import stats

# Normal distribution
mu, sigma = 0, 1
x = np.linspace(-3, 3, 100)
pdf = stats.norm.pdf(x, mu, sigma)
cdf = stats.norm.cdf(x, mu, sigma)

# Generate random samples
samples = stats.norm.rvs(mu, sigma, size=1000)
```

### Hypothesis Testing
```python
from scipy import stats

# t-test
t_stat, p_value = stats.ttest_ind(sample1, sample2)

# Chi-square test
chi2, p_value = stats.chisquare(observed, expected)
```

### Correlation and Regression
```python
import numpy as np
from scipy import stats

# Correlation
corr_matrix = np.corrcoef(X, Y)

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
```

---

## Numerical Methods

### Root Finding
```python
from scipy.optimize import fsolve, brentq

# Define equation
def equation(x):
    return x**3 - 2*x - 5

# Find root
root = fsolve(equation, x0=2)

# Or using Brent's method (requires interval)
root = brentq(equation, a=1, b=3)
```

### Integration
```python
from scipy.integrate import quad, simpson

# Numerical integration
def integrand(x):
    return x**2

result, error = quad(integrand, 0, 1)

# Simpson's rule
x = np.linspace(0, 1, 101)
y = x**2
area = simpson(y, x=x)
```

### Solving ODEs
```python
from scipy.integrate import odeint

def deriv(y, t):
    return -2*y + 3

t = np.linspace(0, 5, 100)
y0 = 1

solution = odeint(deriv, y0, t)
```

---

## Data Analysis

### Reading Data
```python
import pandas as pd

# From CSV
df = pd.read_csv('data.csv')

# From Excel
df = pd.read_excel('data.xlsx')

# Basic info
df.head()
df.describe()
df.info()
```

### Data Manipulation
```python
# Filtering
filtered = df[df['column'] > 10]

# Grouping
grouped = df.groupby('category').mean()

# Sorting
sorted_df = df.sort_values('column', ascending=False)
```

### Visualization
```python
import matplotlib.pyplot as plt

# Line plot
plt.plot(x, y, label='Data')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Title')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot
plt.scatter(x, y)

# Histogram
plt.hist(data, bins=20)

# Bar chart
plt.bar(categories, values)
```

---

## Simulation

### Monte Carlo Simulation
```python
import numpy as np

# Simulate random process
n_simulations = 10000
results = []

for i in range(n_simulations):
    # Generate random variables
    x = np.random.normal(100, 15)
    y = np.random.exponential(20)
    
    # Calculate outcome
    outcome = x + y
    results.append(outcome)

# Analyze results
mean_result = np.mean(results)
confidence_interval = np.percentile(results, [2.5, 97.5])
```

### Discrete Event Simulation
```python
import numpy as np

class Queue:
    def __init__(self):
        self.customers = []
        self.waiting_times = []
    
    def arrive(self, time):
        self.customers.append(time)
    
    def serve(self, time, service_time):
        if self.customers:
            arrival = self.customers.pop(0)
            wait = max(0, time - arrival)
            self.waiting_times.append(wait)
            return wait + service_time
        return 0

# Simulate arrivals
arrival_rate = 0.5  # customers per minute
service_rate = 0.4  # customers per minute

# Run simulation...
```

---

## Common Formulas

### Industrial Engineering

**Productivity**
```
Productivity = Output / Input
```

**Utilization**
```
Utilization = Actual Output / Maximum Possible Output
```

**Overall Equipment Effectiveness (OEE)**
```
OEE = Availability × Performance × Quality
```

**Takt Time**
```
Takt Time = Available Production Time / Customer Demand
```

**Little's Law**
```
L = λ × W
where:
  L = average number in system
  λ = arrival rate
  W = average time in system
```

---

## Tips and Best Practices

1. **Always validate inputs**
   - Check matrix dimensions before operations
   - Verify data types
   - Handle edge cases

2. **Numerical stability**
   - Use `np.linalg.lstsq` for overdetermined systems
   - Check condition number: `np.linalg.cond(A)`
   - Avoid matrix inversion when possible

3. **Optimization**
   - Provide good initial guesses
   - Check convergence status
   - Verify constraints are satisfied

4. **Visualization**
   - Label axes clearly
   - Include legends
   - Use appropriate scales
   - Add grid for readability

5. **Documentation**
   - Comment your code
   - Use descriptive variable names
   - Explain assumptions
   - Document units

---

## Additional Resources

### Documentation
- NumPy: https://numpy.org/doc/
- SciPy: https://docs.scipy.org/
- Matplotlib: https://matplotlib.org/
- Pandas: https://pandas.pydata.org/docs/

### Textbooks
- "Introduction to Linear Algebra" - Gilbert Strang
- "Numerical Methods for Engineers" - Chapra & Canale
- "Introduction to Operations Research" - Hillier & Lieberman
- "Probability and Statistics for Engineers" - Montgomery & Runger

### Online Courses
- MIT OpenCourseWare: Linear Algebra
- Coursera: Operations Research
- Khan Academy: Statistics and Probability

---

**Last Updated**: January 2026
