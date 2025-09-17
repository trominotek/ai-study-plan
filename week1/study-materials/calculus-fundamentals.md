# Calculus Fundamentals for AI/ML

## Derivatives

### What is a Derivative?
A derivative measures the rate of change of a function at a specific point.

**Geometric Interpretation**: Slope of the tangent line
**Physical Interpretation**: Instantaneous rate of change

### Basic Derivative Rules

#### Power Rule
```
d/dx [x^n] = n × x^(n-1)

Examples:
d/dx [x²] = 2x
d/dx [x³] = 3x²
d/dx [√x] = d/dx [x^(1/2)] = (1/2)x^(-1/2) = 1/(2√x)
```

#### Constant Rule
```
d/dx [c] = 0
d/dx [5] = 0
```

#### Sum Rule
```
d/dx [f(x) + g(x)] = f'(x) + g'(x)
d/dx [x² + 3x] = 2x + 3
```

#### Product Rule
```
d/dx [f(x)g(x)] = f'(x)g(x) + f(x)g'(x)
d/dx [x²sin(x)] = 2x×sin(x) + x²×cos(x)
```

#### Chain Rule
```
d/dx [f(g(x))] = f'(g(x)) × g'(x)
d/dx [sin(x²)] = cos(x²) × 2x
```

### Common Functions

#### Exponential Functions
```
d/dx [e^x] = e^x
d/dx [a^x] = a^x × ln(a)
```

#### Logarithmic Functions
```
d/dx [ln(x)] = 1/x
d/dx [log_a(x)] = 1/(x × ln(a))
```

#### Trigonometric Functions
```
d/dx [sin(x)] = cos(x)
d/dx [cos(x)] = -sin(x)
d/dx [tan(x)] = sec²(x)
```

## Partial Derivatives

### Definition
When a function depends on multiple variables, partial derivatives measure the rate of change with respect to one variable while keeping others constant.

### Notation
```
f(x, y) = x² + 3xy + y²

∂f/∂x = 2x + 3y    (treat y as constant)
∂f/∂y = 3x + 2y    (treat x as constant)
```

### Chain Rule for Multiple Variables
```
If z = f(x, y) where x = g(t) and y = h(t):
dz/dt = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt)
```

## Gradients

### Definition
The gradient is a vector of all partial derivatives.

```
For f(x, y) = x² + 3xy + y²:
∇f = [∂f/∂x, ∂f/∂y] = [2x + 3y, 3x + 2y]
```

### Geometric Interpretation
- **Direction**: Points in direction of steepest increase
- **Magnitude**: Rate of steepest increase

### Properties
- **Perpendicular to level curves**
- **Points toward maximum increase**
- **Zero at critical points (maxima, minima, saddle points)**

## Applications in Machine Learning

### Loss Functions and Optimization

#### Mean Squared Error
```
Loss = (1/2n) Σ(yᵢ - ŷᵢ)²

∂Loss/∂w = (1/n) Σ(ŷᵢ - yᵢ) × xᵢ
```

#### Gradient Descent Algorithm
```
1. Start with initial parameters w₀
2. Calculate gradient: ∇L(w)
3. Update: w_new = w_old - α × ∇L(w)
4. Repeat until convergence
```

### Activation Functions

#### Sigmoid Function
```
σ(x) = 1/(1 + e^(-x))
σ'(x) = σ(x)(1 - σ(x))
```

#### ReLU Function
```
ReLU(x) = max(0, x)
ReLU'(x) = {1 if x > 0, 0 if x ≤ 0}
```

#### Tanh Function
```
tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
tanh'(x) = 1 - tanh²(x)
```

### Backpropagation

#### Chain Rule in Neural Networks
For a neural network: Input → Hidden → Output

```
∂Loss/∂w₁ = ∂Loss/∂output × ∂output/∂hidden × ∂hidden/∂w₁
```

#### Example: Simple Network
```
Network: x → w₁ → sigmoid → w₂ → output
Loss: (output - target)²

∂Loss/∂w₂ = 2(output - target) × sigmoid_output
∂Loss/∂w₁ = ∂Loss/∂w₂ × w₂ × sigmoid'(w₁x) × x
```

## Optimization Concepts

### Critical Points
Points where gradient is zero: ∇f = 0

### Types of Critical Points

#### Local Minimum
- **Gradient**: ∇f = 0
- **Hessian**: Positive definite (all eigenvalues > 0)
- **Intuition**: Bowl shape

#### Local Maximum
- **Gradient**: ∇f = 0
- **Hessian**: Negative definite (all eigenvalues < 0)
- **Intuition**: Inverted bowl

#### Saddle Point
- **Gradient**: ∇f = 0
- **Hessian**: Indefinite (mixed sign eigenvalues)
- **Intuition**: Horse saddle shape

### Global vs Local Optima
- **Global Minimum**: Lowest point overall
- **Local Minimum**: Lowest point in neighborhood
- **ML Challenge**: Finding global optimum in high-dimensional space

## Advanced Topics

### Taylor Series
Approximating functions using polynomials:
```
f(x) ≈ f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ...
```

**ML Application**: Second-order optimization methods

### Lagrange Multipliers
Optimizing constrained functions:
```
Maximize f(x, y) subject to g(x, y) = 0
∇f = λ∇g
```

**ML Application**: Support Vector Machines, regularization

### Directional Derivatives
Rate of change in a specific direction:
```
D_u f = ∇f · u    (where u is unit vector)
```

## Practical Implementation Tips

### Numerical Derivatives
```python
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)
```

### Gradient Checking
Verify analytical gradients against numerical gradients:
```python
analytical_grad = compute_gradient(params)
numerical_grad = numerical_gradient(loss_function, params)
diff = abs(analytical_grad - numerical_grad)
assert diff < 1e-7  # Should be very small
```

## Common Calculus Patterns in ML

### Sigmoid Derivative Trick
```
σ'(x) = σ(x)(1 - σ(x))
# Can compute using only forward pass result!
```

### Log-Sum-Exp Trick
```
# Numerically stable way to compute log(Σe^xᵢ)
log_sum_exp(x) = max(x) + log(Σe^(xᵢ - max(x)))
```

### Chain Rule Pattern
```
# Forward pass: compute and store intermediate values
# Backward pass: multiply gradients following chain rule
```

## Study Tips

1. **Practice by Hand**: Start with simple functions
2. **Visualization**: Graph functions and their derivatives
3. **Connect to ML**: Always relate to optimization problems
4. **Computational Tools**: Use SymPy for symbolic math
5. **Numerical Methods**: Implement gradient descent from scratch

## Common Mistakes

1. **Forgetting chain rule** in composite functions
2. **Sign errors** in derivatives
3. **Confusing partial derivatives** with total derivatives
4. **Not checking dimensions** in multivariable calculus
5. **Numerical instability** in implementations

## Next Steps
- Implement gradient descent algorithm
- Practice backpropagation calculations
- Explore automatic differentiation tools
- Study second-order optimization methods
- Work with multivariable optimization problems