# Linear Algebra Quiz

**Time Limit: 45 minutes**
**Instructions: Choose the best answer for each question. Show your work for calculation problems.**

---

## Section A: Multiple Choice (2 points each)

### Question 1
Given vectors **a** = [3, -1, 2] and **b** = [1, 4, -1], what is **a** · **b**?

A) 1
B) -1
C) 3
D) 5

<details>
<summary>Answer</summary>
**B) -1**

**a** · **b** = 3(1) + (-1)(4) + 2(-1) = 3 - 4 - 2 = -1
</details>

### Question 2
What is the magnitude of vector **v** = [4, -3]?

A) 1
B) 5
C) 7
D) 25

<details>
<summary>Answer</summary>
**B) 5**

||**v**|| = √(4² + (-3)²) = √(16 + 9) = √25 = 5
</details>

### Question 3
If matrix A = [[2, 1], [3, -1]] and B = [[1, 2], [0, 1]], what is A × B?

A) [[2, 5], [3, 5]]
B) [[2, 3], [3, -1]]
C) [[3, 6], [3, 5]]
D) [[1, 4], [0, -1]]

<details>
<summary>Answer</summary>
**A) [[2, 5], [3, 5]]**

A × B = [[2×1+1×0, 2×2+1×1], [3×1+(-1)×0, 3×2+(-1)×1]]
      = [[2, 5], [3, 5]]
</details>

### Question 4
Which of the following is true about the transpose of a matrix?

A) (A^T)^T = A
B) (AB)^T = A^T B^T
C) (A + B)^T = A^T - B^T
D) The transpose changes the determinant sign

<details>
<summary>Answer</summary>
**A) (A^T)^T = A**

The transpose of a transpose returns the original matrix.
</details>

### Question 5
Two vectors are orthogonal if:

A) They have the same magnitude
B) Their dot product is zero
C) They point in opposite directions
D) They are parallel

<details>
<summary>Answer</summary>
**B) Their dot product is zero**

Orthogonal vectors are perpendicular, which means their dot product equals zero.
</details>

---

## Section B: Short Answer (5 points each)

### Question 6
Find the determinant of matrix C = [[3, 2], [1, 4]].

<details>
<summary>Answer</summary>
det(C) = 3×4 - 2×1 = 12 - 2 = **10**
</details>

### Question 7
Given matrix D = [[1, 2], [3, 4]], find D^T.

<details>
<summary>Answer</summary>
D^T = [[1, 3], [2, 4]]

The transpose flips the matrix along its diagonal.
</details>

### Question 8
Calculate the unit vector in the direction of **u** = [6, 8].

<details>
<summary>Answer</summary>
First find magnitude: ||**u**|| = √(6² + 8²) = √(36 + 64) = √100 = 10

Unit vector = **u**/||**u**|| = [6, 8]/10 = **[0.6, 0.8]**
</details>

### Question 9
If **p** = [2, 3] and **q** = [-1, 2], find **p** + 2**q**.

<details>
<summary>Answer</summary>
2**q** = 2[-1, 2] = [-2, 4]
**p** + 2**q** = [2, 3] + [-2, 4] = **[0, 7]**
</details>

### Question 10
For matrix E = [[2, 0], [0, 3]], what are the eigenvalues?

<details>
<summary>Answer</summary>
For a diagonal matrix, the eigenvalues are the diagonal elements.
**Eigenvalues: λ₁ = 2, λ₂ = 3**
</details>

---

## Section C: Problem Solving (10 points each)

### Question 11
Given the system of equations:
- 2x + y = 5
- x - y = 1

Write this system in matrix form **Ax** = **b** and solve for **x** using matrix methods.

<details>
<summary>Answer</summary>
Matrix form:
```
A = [[2, 1], [1, -1]]
x = [x, y]
b = [5, 1]
```

Using elimination or inverse:
From second equation: x = y + 1
Substitute: 2(y + 1) + y = 5
2y + 2 + y = 5
3y = 3
y = 1, x = 2

**Solution: x = [2, 1]**
</details>

### Question 12
Find the eigenvalues of matrix F = [[4, 2], [1, 3]].

<details>
<summary>Answer</summary>
Characteristic equation: det(F - λI) = 0

det([[4-λ, 2], [1, 3-λ]]) = (4-λ)(3-λ) - 2×1 = 0
(4-λ)(3-λ) - 2 = 0
12 - 4λ - 3λ + λ² - 2 = 0
λ² - 7λ + 10 = 0
(λ - 5)(λ - 2) = 0

**Eigenvalues: λ₁ = 5, λ₂ = 2**
</details>

### Question 13
In machine learning, we often need to compute **X^T X** where **X** is a data matrix. If **X** = [[1, 2], [3, 1], [2, 3]], compute **X^T X**.

<details>
<summary>Answer</summary>
First find X^T:
X^T = [[1, 3, 2], [2, 1, 3]]

Now compute X^T X:
X^T X = [[1, 3, 2], [2, 1, 3]] × [[1, 2], [3, 1], [2, 3]]

= [[1×1+3×3+2×2, 1×2+3×1+2×3], [2×1+1×3+3×2, 2×2+1×1+3×3]]
= [[1+9+4, 2+3+6], [2+3+6, 4+1+9]]
= **[[14, 11], [11, 14]]**
</details>

---

## Section D: Applied Problems (15 points each)

### Question 14
A dataset has two features: height (in inches) and weight (in pounds). The data matrix is:

**X** = [[65, 120], [70, 150], [68, 140], [72, 160]]

To standardize this data, we need to compute the mean of each column and subtract it from each element. Find the mean vector and the centered data matrix.

<details>
<summary>Answer</summary>
**Mean calculation:**
- Height mean: (65 + 70 + 68 + 72)/4 = 275/4 = 68.75
- Weight mean: (120 + 150 + 140 + 160)/4 = 570/4 = 142.5

**Mean vector: [68.75, 142.5]**

**Centered data matrix:**
X_centered = [[65-68.75, 120-142.5], [70-68.75, 150-142.5], [68-68.75, 140-142.5], [72-68.75, 160-142.5]]
= **[[-3.75, -22.5], [1.25, 7.5], [-0.75, -2.5], [3.25, 17.5]]**
</details>

### Question 15
In Principal Component Analysis (PCA), we find the eigenvectors of the covariance matrix. Given a 2×2 covariance matrix **C** = [[4, 1], [1, 2]], find its eigenvalues and explain what they represent in the context of PCA.

<details>
<summary>Answer</summary>
**Finding eigenvalues:**
det(C - λI) = det([[4-λ, 1], [1, 2-λ]]) = 0
(4-λ)(2-λ) - 1 = 0
8 - 4λ - 2λ + λ² - 1 = 0
λ² - 6λ + 7 = 0

Using quadratic formula: λ = (6 ± √(36-28))/2 = (6 ± √8)/2 = (6 ± 2√2)/2 = 3 ± √2

**Eigenvalues: λ₁ = 3 + √2 ≈ 4.41, λ₂ = 3 - √2 ≈ 1.59**

**Interpretation in PCA:**
- The eigenvalues represent the variance explained by each principal component
- λ₁ ≈ 4.41 is the variance along the first principal component (captures most variation)
- λ₂ ≈ 1.59 is the variance along the second principal component
- Total variance = 4.41 + 1.59 = 6 (trace of covariance matrix)
- First PC explains 4.41/6 ≈ 73.5% of the variance
</details>

---

## Bonus Question (5 points)

### Question 16
In neural networks, we often perform operations like **W**^T**x** where **W** is a weight matrix and **x** is an input vector. If **W** = [[0.5, 0.3], [0.2, 0.8], [0.1, 0.9]] and **x** = [2, 3], what is the dimension of the result and compute **W**^T**x**.

<details>
<summary>Answer</summary>
**W** is 3×2, so **W**^T is 2×3
**x** is 2×1

**W**^T**x** will be 2×1 (2D vector)

W^T = [[0.5, 0.2, 0.1], [0.3, 0.8, 0.9]]

W^T x = [[0.5, 0.2, 0.1], [0.3, 0.8, 0.9]] × [2, 3]
      = [0.5×2 + 0.2×3 + 0.1×3, 0.3×2 + 0.8×3 + 0.9×3]
      = [1 + 0.6 + 0.3, 0.6 + 2.4 + 2.7]
      = **[1.9, 5.7]**

**Result dimension: 2×1 (2D vector)**
</details>

---

## Scoring Guide

- **Section A (Multiple Choice):** 10 points total
- **Section B (Short Answer):** 25 points total
- **Section C (Problem Solving):** 30 points total
- **Section D (Applied Problems):** 30 points total
- **Bonus Question:** 5 points extra credit

**Total: 95 points (+ 5 bonus)**

### Grading Scale:
- A: 85-95+ points
- B: 75-84 points
- C: 65-74 points
- D: 55-64 points
- F: Below 55 points

### Study Tips for Next Time:
- Practice matrix multiplication daily
- Memorize basic derivative rules
- Understand geometric interpretations
- Connect concepts to ML applications
- Review eigenvector/eigenvalue calculations