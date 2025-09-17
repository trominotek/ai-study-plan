# Linear Algebra Exercises

## Exercise Set 1: Vector Operations

### Problem 1: Basic Vector Arithmetic
Given vectors:
- **a** = [3, -2, 1]
- **b** = [1, 4, -2]
- **c** = [-2, 1, 3]

Calculate:
1. **a** + **b**
2. 2**a** - **b**
3. **a** · **b** (dot product)
4. ||**a**|| (magnitude of **a**)

**Solutions:**
1. [4, 2, -1]
2. [5, -8, 4]
3. 3×1 + (-2)×4 + 1×(-2) = 3 - 8 - 2 = -7
4. √(3² + (-2)² + 1²) = √(9 + 4 + 1) = √14

### Problem 2: Vector Properties
1. Find a unit vector in the direction of **v** = [6, -8]
2. Determine if vectors **u** = [2, 3] and **w** = [-3, 2] are orthogonal
3. Find the angle between vectors **p** = [1, 1] and **q** = [1, 0]

**Solutions:**
1. ||**v**|| = √(36 + 64) = 10, unit vector = [0.6, -0.8]
2. **u** · **w** = 2(-3) + 3(2) = -6 + 6 = 0, so they are orthogonal
3. cos(θ) = (**p** · **q**)/(||**p**|| ||**q**||) = 1/(√2 × 1) = 1/√2, so θ = 45°

## Exercise Set 2: Matrix Operations

### Problem 3: Matrix Arithmetic
Given matrices:
```
A = [2  1]    B = [1  3]    C = [5]
    [3 -1]        [2 -1]        [1]
```

Calculate:
1. A + B
2. A × B
3. A × C
4. A^T (transpose of A)

**Solutions:**
```
1. A + B = [3  4]
           [5 -2]

2. A × B = [2×1+1×2  2×3+1×(-1)] = [4  5]
           [3×1-1×2  3×3-1×(-1)]   [1 10]

3. A × C = [2×5+1×1] = [11]
           [3×5-1×1]   [14]

4. A^T = [2  3]
         [1 -1]
```

### Problem 4: Matrix Properties
1. Find the determinant of matrix D = [[4, 2], [1, 3]]
2. Check if matrix E = [[1, 2], [2, 4]] is invertible
3. Find the inverse of matrix F = [[2, 1], [1, 1]] if it exists

**Solutions:**
1. det(D) = 4×3 - 2×1 = 12 - 2 = 10
2. det(E) = 1×4 - 2×2 = 0, so E is not invertible
3. det(F) = 2×1 - 1×1 = 1, F⁻¹ = [[1, -1], [-1, 2]]

## Exercise Set 3: Eigenvalues and Eigenvectors

### Problem 5: Finding Eigenvalues
Find the eigenvalues of matrix G = [[3, 1], [0, 2]]

**Solution:**
Characteristic equation: det(G - λI) = 0
```
det([3-λ  1  ]) = (3-λ)(2-λ) - 1×0 = (3-λ)(2-λ) = 0
   ([0   2-λ])
```
Eigenvalues: λ₁ = 3, λ₂ = 2

### Problem 6: Finding Eigenvectors
For the matrix G from Problem 5, find the eigenvectors corresponding to each eigenvalue.

**Solution:**
For λ₁ = 3:
```
(G - 3I)v = 0 → [0  1][v₁] = [0] → v₁ = [1]
                 [0 -1][v₂]   [0]        [0]
```

For λ₂ = 2:
```
(G - 2I)v = 0 → [1  1][v₁] = [0] → v₂ = [1]
                 [0  0][v₂]   [0]        [-1]
```

## Exercise Set 4: Applied Problems

### Problem 7: Principal Component Analysis Setup
You have a dataset with two features:
- Data matrix X = [[2, 1], [1, 2], [3, 1], [1, 3]]

1. Center the data (subtract the mean)
2. Compute the covariance matrix
3. What would be the first step to find principal components?

**Solution:**
```
1. Mean = [1.75, 1.75]
   Centered data = [[0.25, -0.75], [-0.75, 0.25], [1.25, -0.75], [-0.75, 1.25]]

2. Covariance matrix = [[0.92, -0.58], [-0.58, 0.92]]

3. Find eigenvalues and eigenvectors of the covariance matrix
```

### Problem 8: Neural Network Weight Update
In a simple neural network, you have weight matrix W = [[0.5, -0.2], [0.3, 0.1]] and gradient G = [[0.1, 0.05], [-0.02, 0.03]].

If learning rate α = 0.1, what is the new weight matrix after one gradient descent step?

**Solution:**
```
W_new = W - α × G
W_new = [[0.5, -0.2], [0.3, 0.1]] - 0.1 × [[0.1, 0.05], [-0.02, 0.03]]
W_new = [[0.5, -0.2], [0.3, 0.1]] - [[0.01, 0.005], [-0.002, 0.003]]
W_new = [[0.49, -0.205], [0.302, 0.097]]
```

## Programming Exercises

### Exercise 9: NumPy Implementation
Write Python code to:
1. Create vectors and perform dot products
2. Multiply matrices
3. Find eigenvalues using numpy.linalg.eig()

```python
import numpy as np

# 1. Vector operations
a = np.array([3, -2, 1])
b = np.array([1, 4, -2])
dot_product = np.dot(a, b)
print(f"Dot product: {dot_product}")

# 2. Matrix multiplication
A = np.array([[2, 1], [3, -1]])
B = np.array([[1, 3], [2, -1]])
result = np.matmul(A, B)
print(f"Matrix product:\n{result}")

# 3. Eigenvalues and eigenvectors
G = np.array([[3, 1], [0, 2]])
eigenvals, eigenvecs = np.linalg.eig(G)
print(f"Eigenvalues: {eigenvals}")
print(f"Eigenvectors:\n{eigenvecs}")
```

### Exercise 10: Gradient Descent Simulation
Implement a simple 2D gradient descent to minimize f(x, y) = x² + y²

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_f(x, y):
    return np.array([2*x, 2*y])

def gradient_descent_2d():
    # Starting point
    point = np.array([3.0, 2.0])
    learning_rate = 0.1
    history = [point.copy()]

    for i in range(50):
        grad = gradient_f(point[0], point[1])
        point = point - learning_rate * grad
        history.append(point.copy())

    return np.array(history)

# Run and visualize
path = gradient_descent_2d()
print(f"Final point: {path[-1]}")
```

## Challenge Problems

### Problem 11: Matrix Decomposition
Given matrix H = [[4, 2], [2, 1]], perform eigenvalue decomposition and verify that H = QΛQ^T where Q contains eigenvectors and Λ contains eigenvalues.

### Problem 12: Optimization Problem
Find the minimum of f(x, y) = 3x² + 2xy + 3y² subject to x + y = 1 using matrix methods.

**Hint:** Use the method of Lagrange multipliers and represent the problem in matrix form.

## Self-Assessment Questions

1. Can you explain the geometric interpretation of matrix multiplication?
2. Why are eigenvalues important for dimensionality reduction?
3. How does the gradient relate to the direction of steepest ascent?
4. What's the difference between linear dependence and independence?
5. How would you check if a set of vectors spans a particular space?

## Practical Applications

### Application 1: Image Compression
Research how Singular Value Decomposition (SVD) is used for image compression. Try implementing a simple version.

### Application 2: Recommendation Systems
Understand how matrix factorization is used in collaborative filtering for recommendation systems.

### Application 3: Graph Analysis
Explore how the adjacency matrix of a graph can be analyzed using eigenvalue decomposition to find communities.