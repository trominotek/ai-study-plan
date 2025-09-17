# Linear Algebra Basics for AI/ML

## Vectors

### What is a Vector?
A vector is a mathematical object that has both magnitude and direction. In AI/ML, we often work with vectors in n-dimensional space.

**Notation**: Vector `v` = [v₁, v₂, v₃, ..., vₙ]

### Vector Operations

#### Addition
```
v₁ = [2, 3]
v₂ = [1, 4]
v₁ + v₂ = [2+1, 3+4] = [3, 7]
```

#### Scalar Multiplication
```
v = [2, 3]
3v = [3×2, 3×3] = [6, 9]
```

#### Dot Product
```
v₁ · v₂ = v₁₁×v₂₁ + v₁₂×v₂₂ + ... + v₁ₙ×v₂ₙ
v₁ = [2, 3]
v₂ = [1, 4]
v₁ · v₂ = 2×1 + 3×4 = 2 + 12 = 14
```

### Vector Properties
- **Magnitude (Length)**: ||v|| = √(v₁² + v₂² + ... + vₙ²)
- **Unit Vector**: A vector with magnitude 1
- **Orthogonal Vectors**: Two vectors are orthogonal if their dot product is 0

## Matrices

### What is a Matrix?
A matrix is a rectangular array of numbers arranged in rows and columns.

**Notation**: Matrix A with m rows and n columns is written as A_{m×n}

### Matrix Operations

#### Addition
```
A = [1, 2]    B = [5, 6]
    [3, 4]        [7, 8]

A + B = [1+5, 2+6] = [6,  8]
        [3+7, 4+8]   [10, 12]
```

#### Multiplication
```
A = [1, 2]    B = [5, 6]
    [3, 4]        [7, 8]

A × B = [1×5+2×7, 1×6+2×8] = [19, 22]
        [3×5+4×7, 3×6+4×8]   [43, 50]
```

#### Transpose
```
A = [1, 2, 3]    A^T = [1, 4]
    [4, 5, 6]          [2, 5]
                       [3, 6]
```

### Special Matrices
- **Identity Matrix (I)**: Square matrix with 1s on diagonal, 0s elsewhere
- **Inverse Matrix (A⁻¹)**: A × A⁻¹ = I
- **Symmetric Matrix**: A = A^T

## Eigenvalues and Eigenvectors

### Definition
For a square matrix A, an eigenvector v and eigenvalue λ satisfy:
```
Av = λv
```

### Intuition
- Eigenvectors are directions that don't change when the matrix transformation is applied
- Eigenvalues tell us how much the eigenvector is scaled

### Applications in ML
- **Principal Component Analysis (PCA)**: Uses eigenvectors to find directions of maximum variance
- **Neural Networks**: Understanding how data flows through layers
- **Dimensionality Reduction**: Keeping the most important features

## Key Concepts for AI/ML

### Why Linear Algebra Matters
1. **Data Representation**: Features are vectors, datasets are matrices
2. **Transformations**: Neural networks perform matrix multiplications
3. **Optimization**: Gradient descent uses vector derivatives
4. **Dimensionality**: Understanding high-dimensional spaces

### Common Applications
- **Image Processing**: Images as matrices, filters as convolutions
- **Natural Language Processing**: Words as vectors (word embeddings)
- **Recommendation Systems**: User-item interactions as matrices
- **Deep Learning**: All computations are matrix operations

## Study Tips
1. Practice with small examples by hand first
2. Use visualization tools for 2D/3D vectors
3. Implement basic operations in Python (NumPy)
4. Connect concepts to ML applications early
5. Focus on geometric intuition, not just computation

## Next Steps
- Practice vector operations with NumPy
- Implement matrix multiplication from scratch
- Explore PCA as an application of eigenvalue decomposition
- Learn about matrix decomposition techniques (SVD, LU)