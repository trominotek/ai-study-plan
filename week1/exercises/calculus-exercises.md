# Calculus Exercises for Machine Learning

## Exercise Set 1: Basic Derivatives

### Problem 1: Power Rule Practice
Find the derivatives of the following functions:

1. f(x) = x⁴ + 3x² - 5x + 2
2. g(x) = √x + 1/x²
3. h(x) = (2x³ - x)²
4. k(x) = x²e^x

**Solutions:**
1. f'(x) = 4x³ + 6x - 5
2. g'(x) = 1/(2√x) - 2/x³ = x^(-1/2)/2 - 2x^(-3)
3. Using chain rule: h'(x) = 2(2x³ - x)(6x² - 1) = 2(6x² - 1)(2x³ - x)
4. Using product rule: k'(x) = 2x·e^x + x²·e^x = xe^x(2 + x)

### Problem 2: Chain Rule Applications
Find the derivatives:

1. f(x) = sin(3x² + 1)
2. g(x) = e^(2x + 5)
3. h(x) = ln(x² + 4x)
4. k(x) = (x² + 1)^(3/2)

**Solutions:**
1. f'(x) = cos(3x² + 1) · 6x = 6x cos(3x² + 1)
2. g'(x) = e^(2x + 5) · 2 = 2e^(2x + 5)
3. h'(x) = (2x + 4)/(x² + 4x)
4. k'(x) = (3/2)(x² + 1)^(1/2) · 2x = 3x√(x² + 1)

## Exercise Set 2: Partial Derivatives

### Problem 3: Basic Partial Derivatives
For f(x, y) = x³y² + 2xy - y³, find:

1. ∂f/∂x
2. ∂f/∂y
3. ∂²f/∂x²
4. ∂²f/∂x∂y

**Solutions:**
1. ∂f/∂x = 3x²y² + 2y
2. ∂f/∂y = 2x³y + 2x - 3y²
3. ∂²f/∂x² = 6xy²
4. ∂²f/∂x∂y = 6x²y + 2

### Problem 4: Multivariable Chain Rule
If z = x² + y², x = cos(t), y = sin(t), find dz/dt.

**Solution:**
```
dz/dt = (∂z/∂x)(dx/dt) + (∂z/∂y)(dy/dt)
∂z/∂x = 2x, ∂z/∂y = 2y
dx/dt = -sin(t), dy/dt = cos(t)

dz/dt = 2x(-sin(t)) + 2y(cos(t))
      = 2cos(t)(-sin(t)) + 2sin(t)(cos(t))
      = -2cos(t)sin(t) + 2sin(t)cos(t) = 0
```

## Exercise Set 3: Gradients and Optimization

### Problem 5: Computing Gradients
Find the gradient of:

1. f(x, y) = x² + 4y² - 2xy + 3
2. g(x, y) = xe^y + y ln(x)
3. h(x, y, z) = x²y + yz² - 3z

**Solutions:**
1. ∇f = [2x - 2y, 8y - 2x]
2. ∇g = [e^y + y/x, xe^y + ln(x)]
3. ∇h = [2xy, x² + z², 2yz - 3]

### Problem 6: Critical Points
Find the critical points of f(x, y) = x³ - 3xy + y³

**Solution:**
```
∇f = [3x² - 3y, -3x + 3y²] = [0, 0]

From 3x² - 3y = 0: x² = y
From -3x + 3y² = 0: x = y²

Substituting: x = y² and y = x²
So y = (y²)² = y⁴
y⁴ - y = 0, y(y³ - 1) = 0
y = 0 or y = 1

Critical points: (0, 0) and (1, 1)
```

## Exercise Set 4: Applications to Machine Learning

### Problem 7: Linear Regression Cost Function
The mean squared error for linear regression is:
J(w₀, w₁) = (1/2m) Σᵢ₌₁ᵐ (w₀ + w₁xᵢ - yᵢ)²

Find the partial derivatives ∂J/∂w₀ and ∂J/∂w₁.

**Solution:**
```
∂J/∂w₀ = (1/m) Σᵢ₌₁ᵐ (w₀ + w₁xᵢ - yᵢ)

∂J/∂w₁ = (1/m) Σᵢ₌₁ᵐ (w₀ + w₁xᵢ - yᵢ)xᵢ
```

### Problem 8: Gradient Descent Update
Given the cost function J(w) = w² - 4w + 5, implement gradient descent to find the minimum.

**Solution:**
```
J'(w) = 2w - 4
Gradient descent update: w_new = w_old - α(2w_old - 4)

With α = 0.1 and starting point w₀ = 0:
w₁ = 0 - 0.1(2·0 - 4) = 0.4
w₂ = 0.4 - 0.1(2·0.4 - 4) = 0.4 - 0.1(-3.2) = 0.72
w₃ = 0.72 - 0.1(2·0.72 - 4) = 0.72 - 0.1(-2.56) = 0.976

Converges to w = 2 (analytical minimum: J'(w) = 0 → w = 2)
```

### Problem 9: Logistic Function Derivative
The logistic (sigmoid) function is σ(x) = 1/(1 + e^(-x)).
Show that σ'(x) = σ(x)(1 - σ(x)).

**Solution:**
```
σ(x) = (1 + e^(-x))^(-1)
σ'(x) = -(1 + e^(-x))^(-2) · (-e^(-x)) = e^(-x)/(1 + e^(-x))²

Rewrite as:
σ'(x) = e^(-x)/(1 + e^(-x))² = [1/(1 + e^(-x))] · [e^(-x)/(1 + e^(-x))]
      = σ(x) · [e^(-x)/(1 + e^(-x))]
      = σ(x) · [(1 + e^(-x) - 1)/(1 + e^(-x))]
      = σ(x) · [1 - 1/(1 + e^(-x))]
      = σ(x)(1 - σ(x))
```

## Exercise Set 5: Advanced Applications

### Problem 10: Neural Network Backpropagation
Consider a simple network: input x → w₁ → sigmoid → w₂ → output ŷ
Loss: L = (1/2)(ŷ - y)²

Derive the gradients ∂L/∂w₁ and ∂L/∂w₂ using the chain rule.

**Solution:**
```
Forward pass:
z₁ = w₁x
a₁ = σ(z₁) = 1/(1 + e^(-z₁))
ŷ = w₂a₁
L = (1/2)(ŷ - y)²

Backward pass:
∂L/∂w₂ = ∂L/∂ŷ · ∂ŷ/∂w₂ = (ŷ - y) · a₁

∂L/∂w₁ = ∂L/∂ŷ · ∂ŷ/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂w₁
       = (ŷ - y) · w₂ · σ(z₁)(1-σ(z₁)) · x
       = (ŷ - y) · w₂ · a₁(1-a₁) · x
```

### Problem 11: Regularization
Add L2 regularization to the linear regression cost function:
J(w) = (1/2m) Σᵢ₌₁ᵐ (wᵀxᵢ - yᵢ)² + (λ/2) ||w||²

Find the gradient ∇J(w).

**Solution:**
```
J(w) = (1/2m) Σᵢ₌₁ᵐ (wᵀxᵢ - yᵢ)² + (λ/2) wᵀw

∇J(w) = (1/m) Σᵢ₌₁ᵐ (wᵀxᵢ - yᵢ)xᵢ + λw
      = (1/m) Xᵀ(Xw - y) + λw

where X is the feature matrix and y is the target vector.
```

## Programming Exercises

### Exercise 12: Numerical Gradient Checking

```python
import numpy as np

def numerical_gradient(f, x, h=1e-5):
    """Compute numerical gradient using central difference"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

def analytical_gradient(x):
    """Analytical gradient of f(x) = x[0]² + 2*x[1]² + 3*x[0]*x[1]"""
    return np.array([2*x[0] + 3*x[1], 4*x[1] + 3*x[0]])

def f(x):
    """Test function: f(x) = x₁² + 2x₂² + 3x₁x₂"""
    return x[0]**2 + 2*x[1]**2 + 3*x[0]*x[1]

# Test gradient computation
x = np.array([2.0, 3.0])
num_grad = numerical_gradient(f, x)
ana_grad = analytical_gradient(x)

print(f"Numerical gradient: {num_grad}")
print(f"Analytical gradient: {ana_grad}")
print(f"Difference: {np.abs(num_grad - ana_grad)}")
```

### Exercise 13: Implement Gradient Descent

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent_2d(f, grad_f, start_point, learning_rate=0.01, max_iters=1000, tol=1e-6):
    """2D gradient descent implementation"""
    point = np.array(start_point, dtype=float)
    history = [point.copy()]

    for i in range(max_iters):
        grad = grad_f(point)
        new_point = point - learning_rate * grad

        if np.linalg.norm(new_point - point) < tol:
            print(f"Converged after {i+1} iterations")
            break

        point = new_point
        history.append(point.copy())

    return np.array(history)

# Example: minimize f(x,y) = (x-1)² + 2(y+2)²
def f(p):
    return (p[0] - 1)**2 + 2*(p[1] + 2)**2

def grad_f(p):
    return np.array([2*(p[0] - 1), 4*(p[1] + 2)])

# Run gradient descent
path = gradient_descent_2d(f, grad_f, [5, 3], learning_rate=0.1)
print(f"Final point: {path[-1]}")
print(f"True minimum: [1, -2]")
```

### Exercise 14: Automatic Differentiation (Basic)

```python
class Variable:
    """Simple automatic differentiation for single variable"""
    def __init__(self, value, grad=0.0):
        self.value = value
        self.grad = grad

    def __add__(self, other):
        if isinstance(other, Variable):
            return Variable(self.value + other.value, self.grad + other.grad)
        else:
            return Variable(self.value + other, self.grad)

    def __mul__(self, other):
        if isinstance(other, Variable):
            return Variable(self.value * other.value,
                          self.grad * other.value + self.value * other.grad)
        else:
            return Variable(self.value * other, self.grad * other)

    def __pow__(self, n):
        return Variable(self.value ** n, n * (self.value ** (n-1)) * self.grad)

    def exp(self):
        exp_val = np.exp(self.value)
        return Variable(exp_val, self.grad * exp_val)

# Example usage
x = Variable(2.0, 1.0)  # Value=2, gradient=1 (seed for dx/dx = 1)
f = x * x + x * 3 + Variable(5.0)  # f(x) = x² + 3x + 5
print(f"f(2) = {f.value}")
print(f"f'(2) = {f.grad}")  # Should be 2*2 + 3 = 7
```

## Advanced Challenge Problems

### Problem 15: Constrained Optimization
Find the minimum of f(x, y) = x² + y² subject to the constraint x + y = 1 using Lagrange multipliers.

**Solution Setup:**
```
L(x, y, λ) = x² + y² + λ(x + y - 1)
∇L = [2x + λ, 2y + λ, x + y - 1] = [0, 0, 0]
```

### Problem 16: Hessian Matrix
For f(x, y) = x⁴ + y⁴ - 4xy, compute the Hessian matrix and classify the critical points.

### Problem 17: Optimization with Momentum
Implement gradient descent with momentum and compare convergence to standard gradient descent.

```python
def gradient_descent_momentum(grad_f, start_point, learning_rate=0.01,
                            momentum=0.9, max_iters=1000):
    """Gradient descent with momentum"""
    point = np.array(start_point, dtype=float)
    velocity = np.zeros_like(point)
    history = [point.copy()]

    for i in range(max_iters):
        grad = grad_f(point)
        velocity = momentum * velocity - learning_rate * grad
        point = point + velocity
        history.append(point.copy())

        if np.linalg.norm(grad) < 1e-6:
            break

    return np.array(history)
```

## Applications to Deep Learning

### Problem 18: Softmax Derivative
The softmax function for a vector z is: softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)
Derive the derivative ∂softmax(z)ᵢ/∂zⱼ.

### Problem 19: Cross-Entropy Loss
For the cross-entropy loss L = -Σᵢ yᵢ log(ŷᵢ) where ŷ is the softmax output, show that:
∂L/∂zᵢ = ŷᵢ - yᵢ

### Problem 20: Batch Normalization
The batch normalization operation is:
BN(x) = γ((x - μ)/σ) + β
where μ and σ are batch statistics. Find the gradients with respect to γ, β, and x.

## Self-Assessment Questions

1. Why is the chain rule crucial for training neural networks?
2. How does the learning rate affect gradient descent convergence?
3. What's the difference between a local minimum and a saddle point?
4. When might numerical gradients be useful despite being computationally expensive?
5. How do second-order methods (using Hessian) potentially improve optimization?

## Practical Tips

1. **Always verify gradients** using numerical approximation during development
2. **Understand the geometry** - visualize 2D functions to build intuition
3. **Start simple** - implement gradient descent for quadratic functions first
4. **Monitor convergence** - plot loss curves to understand optimization behavior
5. **Learn automatic differentiation** - understand how modern ML frameworks work