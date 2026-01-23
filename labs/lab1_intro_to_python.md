# Lab 1: Introduction to Python for Mathematical Computing

## Objectives
- Set up Python environment for mathematical computing
- Learn basic NumPy operations
- Create visualizations with Matplotlib
- Work with arrays and matrices

## Prerequisites
- Python 3.8+
- NumPy, Matplotlib installed

## Exercises

### Exercise 1: NumPy Basics (20 points)

Write Python code to:

1. Create a 1D array with values from 0 to 20 with step 2
2. Create a 3x3 matrix with random integers between 1 and 10
3. Create an identity matrix of size 5x5
4. Create a matrix of zeros (4x3) and ones (3x4)

### Exercise 2: Matrix Operations (30 points)

Given two matrices:
```
A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

B = [[9, 8, 7],
     [6, 5, 4],
     [3, 2, 1]]
```

Perform the following operations:
1. Element-wise addition and subtraction
2. Matrix multiplication
3. Transpose of A
4. Determinant of A and B
5. Inverse of B (if it exists)

### Exercise 3: Array Indexing and Slicing (20 points)

Create a 5x5 matrix of random integers (0-50) and:
1. Extract the second row
2. Extract the third column
3. Extract a 2x2 sub-matrix from the center
4. Replace all values greater than 25 with 25

### Exercise 4: Mathematical Functions (30 points)

Create arrays and apply mathematical functions:

1. Create an array of 50 equally spaced values between 0 and 2π
2. Calculate sin(x), cos(x), and tan(x) for these values
3. Plot all three functions on the same graph with:
   - Proper labels for x and y axes
   - Legend identifying each function
   - Grid lines
   - Title

## Starter Code

```python
import numpy as np
import matplotlib.pyplot as plt

# Exercise 1
print("=== Exercise 1: NumPy Basics ===")
# Your code here

# Exercise 2
print("\n=== Exercise 2: Matrix Operations ===")
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])

# Your code here

# Exercise 3
print("\n=== Exercise 3: Array Indexing ===")
# Your code here

# Exercise 4
print("\n=== Exercise 4: Mathematical Functions ===")
# Your code here

plt.show()
```

## Submission Guidelines

1. Create a Python script named `lab1_yourname.py`
2. Include comments explaining your code
3. Ensure all outputs are clearly labeled
4. Save any plots as PNG files
5. Submit via GitHub repository

## Expected Output

Your program should display:
- All matrix results with appropriate labels
- A plot showing sin, cos, and tan functions
- Clear formatting for easy reading

## Grading Rubric

- Code correctness: 60%
- Code quality and comments: 20%
- Output formatting: 10%
- Plot quality: 10%

## Additional Challenges (Bonus: 10 points)

1. Create a 3D surface plot of function z = sin(x) * cos(y)
2. Calculate the mean, median, and standard deviation of a random matrix
3. Find eigenvalues and eigenvectors of matrix A

## Due Date

One week from lab session date

## Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Tutorial](https://matplotlib.org/stable/tutorials/index.html)
