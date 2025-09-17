# Calculus Quiz

**Time Limit: 50 minutes**
**Instructions: Show all work for full credit. Partial credit will be given for correct methods.**

---

## Section A: Basic Derivatives (3 points each)

### Question 1
Find the derivative of f(x) = 3x⁴ - 2x³ + 5x - 7

<details>
<summary>Answer</summary>
f'(x) = **12x³ - 6x² + 5**

Using power rule: d/dx[xⁿ] = nxⁿ⁻¹
</details>

### Question 2
Find dy/dx for y = e^(2x+1)

<details>
<summary>Answer</summary>
dy/dx = **2e^(2x+1)**

Using chain rule: d/dx[e^u] = e^u × du/dx, where u = 2x+1, so du/dx = 2
</details>

### Question 3
Find the derivative of g(x) = ln(x² + 3x)

<details>
<summary>Answer</summary>
g'(x) = **(2x + 3)/(x² + 3x)**

Using chain rule: d/dx[ln(u)] = (1/u) × du/dx, where u = x² + 3x, so du/dx = 2x + 3
</details>

### Question 4
Find f'(x) if f(x) = x²sin(x)

<details>
<summary>Answer</summary>
f'(x) = **2x sin(x) + x² cos(x)**

Using product rule: d/dx[uv] = u'v + uv'
where u = x², u' = 2x, v = sin(x), v' = cos(x)
</details>

### Question 5
Find the derivative of h(x) = (3x + 1)⁵

<details>
<summary>Answer</summary>
h'(x) = **15(3x + 1)⁴**

Using chain rule: d/dx[uⁿ] = nuⁿ⁻¹ × du/dx
where u = 3x + 1, du/dx = 3, n = 5
So: 5(3x + 1)⁴ × 3 = 15(3x + 1)⁴
</details>

---

## Section B: Partial Derivatives (4 points each)

### Question 6
For f(x, y) = x³y² - 2xy + y³, find ∂f/∂x and ∂f/∂y

<details>
<summary>Answer</summary>
**∂f/∂x = 3x²y² - 2y** (treat y as constant)

**∂f/∂y = 2x³y - 2x + 3y²** (treat x as constant)
</details>

### Question 7
Find the gradient ∇f for f(x, y) = e^(xy) + x² - y²

<details>
<summary>Answer</summary>
∇f = [∂f/∂x, ∂f/∂y]

∂f/∂x = ye^(xy) + 2x
∂f/∂y = xe^(xy) - 2y

**∇f = [ye^(xy) + 2x, xe^(xy) - 2y]**
</details>

### Question 8
For g(x, y, z) = xyz + x²z - 3y, find ∂g/∂z

<details>
<summary>Answer</summary>
**∂g/∂z = xy + x²**

Treating x and y as constants when differentiating with respect to z.
</details>

---

## Section C: Chain Rule Applications (6 points each)

### Question 9
If z = x² + y², x = 3t, and y = 4t, find dz/dt using the chain rule.

<details>
<summary>Answer</summary>
**Method 1 (Chain rule):**
dz/dt = (∂z/∂x)(dx/dt) + (∂z/∂y)(dy/dt)

∂z/∂x = 2x, ∂z/∂y = 2y
dx/dt = 3, dy/dt = 4

dz/dt = 2x(3) + 2y(4) = 6x + 8y
Substituting x = 3t and y = 4t:
dz/dt = 6(3t) + 8(4t) = 18t + 32t = **50t**

**Method 2 (Direct substitution verification):**
z = (3t)² + (4t)² = 9t² + 16t² = 25t²
dz/dt = 50t ✓
</details>

### Question 10
A spherical balloon is being inflated. If the radius is increasing at a rate of 2 cm/min, find the rate at which the volume is increasing when r = 6 cm.

<details>
<summary>Answer</summary>
**Given:**
- dr/dt = 2 cm/min
- r = 6 cm when we want dV/dt
- V = (4/3)πr³

**Solution:**
dV/dt = dV/dr × dr/dt

dV/dr = 4πr²

dV/dt = 4πr² × dr/dt = 4π(6)² × 2 = 4π(36) × 2 = **288π cm³/min**
</details>

---

## Section D: Optimization (8 points each)

### Question 11
Find the critical points of f(x, y) = x² + y² - 4x + 6y + 5 and classify them.

<details>
<summary>Answer</summary>
**Step 1: Find critical points**
∇f = [∂f/∂x, ∂f/∂y] = [2x - 4, 2y + 6] = [0, 0]

2x - 4 = 0 → x = 2
2y + 6 = 0 → y = -3

**Critical point: (2, -3)**

**Step 2: Second derivative test**
∂²f/∂x² = 2, ∂²f/∂y² = 2, ∂²f/∂x∂y = 0

Discriminant D = (∂²f/∂x²)(∂²f/∂y²) - (∂²f/∂x∂y)² = (2)(2) - (0)² = 4 > 0

Since D > 0 and ∂²f/∂x² = 2 > 0, the critical point **(2, -3) is a local minimum**.

**Minimum value:** f(2, -3) = 4 + 9 - 8 - 18 + 5 = **-8**
</details>

### Question 12
Use gradient descent to minimize f(x) = x² - 4x + 7. Start with x₀ = 0, use learning rate α = 0.1, and perform 3 iterations.

<details>
<summary>Answer</summary>
**Gradient:** f'(x) = 2x - 4

**Gradient descent update:** x_{n+1} = x_n - α × f'(x_n)

**Iteration 0:** x₀ = 0
f'(0) = 2(0) - 4 = -4
x₁ = 0 - 0.1(-4) = 0 + 0.4 = **0.4**

**Iteration 1:** x₁ = 0.4
f'(0.4) = 2(0.4) - 4 = -3.2
x₂ = 0.4 - 0.1(-3.2) = 0.4 + 0.32 = **0.72**

**Iteration 2:** x₂ = 0.72
f'(0.72) = 2(0.72) - 4 = -2.56
x₃ = 0.72 - 0.1(-2.56) = 0.72 + 0.256 = **0.976**

**Results:** x₀ = 0, x₁ = 0.4, x₂ = 0.72, x₃ = 0.976

**Note:** Analytical minimum is at x = 2 (where f'(x) = 0)
</details>

---

## Section E: Applications to Machine Learning (10 points each)

### Question 13
For logistic regression, the sigmoid function is σ(z) = 1/(1 + e^(-z)).

a) Show that σ'(z) = σ(z)(1 - σ(z))
b) If z = w₀ + w₁x, find ∂σ/∂w₁

<details>
<summary>Answer</summary>
**a) Derivative of sigmoid:**

σ(z) = (1 + e^(-z))^(-1)

Using chain rule:
σ'(z) = -(1 + e^(-z))^(-2) × (-e^(-z)) = e^(-z)/(1 + e^(-z))²

Rewrite the result:
σ'(z) = e^(-z)/(1 + e^(-z))² = [1/(1 + e^(-z))] × [e^(-z)/(1 + e^(-z))]

= σ(z) × [e^(-z)/(1 + e^(-z))]

= σ(z) × [(1 + e^(-z) - 1)/(1 + e^(-z))]

= σ(z) × [1 - 1/(1 + e^(-z))]

= **σ(z)(1 - σ(z))** ✓

**b) Chain rule application:**
∂σ/∂w₁ = σ'(z) × ∂z/∂w₁ = σ(z)(1 - σ(z)) × x = **xσ(z)(1 - σ(z))**
</details>

### Question 14
In a neural network, the mean squared error loss for one training example is:
L = (1/2)(ŷ - y)²

where ŷ = w₂σ(w₁x + b₁) + b₂, σ is the sigmoid function, and y is the target.

Find ∂L/∂w₁ using the chain rule (you may use σ'(z) = σ(z)(1-σ(z))).

<details>
<summary>Answer</summary>
**Chain rule breakdown:**
∂L/∂w₁ = ∂L/∂ŷ × ∂ŷ/∂σ × ∂σ/∂z × ∂z/∂w₁

where z = w₁x + b₁

**Step-by-step:**

1) **∂L/∂ŷ = (ŷ - y)**

2) **∂ŷ/∂σ = w₂** (since ŷ = w₂σ + b₂)

3) **∂σ/∂z = σ(z)(1 - σ(z))** (given)

4) **∂z/∂w₁ = x** (since z = w₁x + b₁)

**Final result:**
∂L/∂w₁ = (ŷ - y) × w₂ × σ(z)(1 - σ(z)) × x

= **x(ŷ - y)w₂σ(z)(1 - σ(z))**

where z = w₁x + b₁ and σ(z) = 1/(1 + e^(-z))
</details>

---

## Section F: Advanced Problems (12 points each)

### Question 15
A rectangular box with no top is to be constructed from 1200 cm² of cardboard. Find the dimensions that maximize the volume.

<details>
<summary>Answer</summary>
**Setup:**
Let the base be x × y and height be h.
- Surface area: xy + 2xh + 2yh = 1200
- Volume: V = xyh (to maximize)

**Express h in terms of x and y:**
xy + 2h(x + y) = 1200
h = (1200 - xy)/(2(x + y))

**Substitute into volume:**
V(x, y) = xy × (1200 - xy)/(2(x + y)) = xy(1200 - xy)/(2(x + y))

**Find critical points:**
This is complex, so let's use symmetry. For maximum volume, optimal design is often when x = y.

Let x = y, then:
h = (1200 - x²)/(4x)
V(x) = x² × (1200 - x²)/(4x) = x(1200 - x²)/4 = (1200x - x³)/4

**Maximize V(x):**
dV/dx = (1200 - 3x²)/4 = 0
1200 - 3x² = 0
x² = 400
x = 20 cm

**Therefore:**
- x = y = **20 cm**
- h = (1200 - 400)/(4 × 20) = 800/80 = **10 cm**

**Maximum volume:** V = 20 × 20 × 10 = **4000 cm³**
</details>

### Question 16
In machine learning, regularized linear regression uses the cost function:
J(w) = (1/2m)∑ᵢ₌₁ᵐ(wᵀxᵢ - yᵢ)² + (λ/2)∑ⱼ₌₁ⁿwⱼ²

Find the gradient ∇J(w) and derive the gradient descent update rule.

<details>
<summary>Answer</summary>
**Cost function breakdown:**
J(w) = MSE term + Regularization term

**Gradient of MSE term:**
∂/∂wⱼ [(1/2m)∑ᵢ₌₁ᵐ(wᵀxᵢ - yᵢ)²] = (1/m)∑ᵢ₌₁ᵐ(wᵀxᵢ - yᵢ)xᵢⱼ

In matrix form: (1/m)Xᵀ(Xw - y)

**Gradient of regularization term:**
∂/∂wⱼ [(λ/2)∑ⱼ₌₁ⁿwⱼ²] = λwⱼ

In matrix form: λw

**Complete gradient:**
∇J(w) = (1/m)Xᵀ(Xw - y) + λw

**Gradient descent update:**
w_{new} = w_{old} - α∇J(w)
w_{new} = w_{old} - α[(1/m)Xᵀ(Xw - y) + λw]
w_{new} = w_{old} - α(1/m)Xᵀ(Xw - y) - αλw
**w_{new} = w_{old}(1 - αλ) - α(1/m)Xᵀ(Xw - y)**

**Note:** The term (1 - αλ) represents weight decay - the regularization shrinks weights toward zero.
</details>

---

## Bonus Questions (5 points each)

### Question 17
Prove that for any differentiable function f(x), if f has a local extremum at x = c, then f'(c) = 0.

<details>
<summary>Answer</summary>
**Proof by contradiction:**

Assume f has a local extremum at x = c but f'(c) ≠ 0.

**Case 1:** f'(c) > 0
- By definition of derivative: f'(c) = lim[h→0] (f(c+h) - f(c))/h > 0
- This means for sufficiently small h > 0: (f(c+h) - f(c))/h > 0
- Therefore: f(c+h) > f(c) for small positive h
- And for small negative h: f(c+h) < f(c)
- This contradicts f having a local extremum at c

**Case 2:** f'(c) < 0
- Similar argument shows f(c+h) < f(c) for small positive h
- And f(c+h) > f(c) for small negative h
- Again contradicts local extremum

**Conclusion:** Since both cases lead to contradiction, we must have **f'(c) = 0**.

**Note:** This is Fermat's theorem. The converse is not true - f'(c) = 0 doesn't guarantee an extremum (consider f(x) = x³ at x = 0).
</details>

### Question 18
Explain the geometric interpretation of the gradient vector ∇f(x, y) and how it relates to gradient descent optimization.

<details>
<summary>Answer</summary>
**Geometric Interpretation:**

1. **Direction:** ∇f(x, y) points in the direction of **steepest increase** of f at point (x, y)

2. **Magnitude:** ||∇f(x, y)|| represents the **rate of steepest increase**

3. **Perpendicular to level curves:** The gradient is always perpendicular to the contour lines (level curves) of f

4. **Zero gradient:** ∇f = 0 at critical points (local maxima, minima, or saddle points)

**Relationship to Gradient Descent:**

1. **Opposite direction:** Gradient descent moves in the direction **-∇f** (negative gradient) to find minima

2. **Update rule:** w_{new} = w_{old} - α∇f(w_{old})

3. **Step size:** The learning rate α controls how far we move in the gradient direction

4. **Convergence:** At the minimum, ∇f = 0, so updates become w_{new} = w_{old} (convergence)

5. **Geometric intuition:** Like a ball rolling downhill - it naturally follows the steepest descent path

**Practical implications:**
- Large gradient → steep slope → larger steps
- Small gradient → gentle slope → smaller steps
- Zero gradient → flat surface → stop (minimum found)
</details>

---

## Scoring Guide

**Total Points: 100 + 10 bonus**

### Section Breakdown:
- **Section A (Basic Derivatives):** 15 points
- **Section B (Partial Derivatives):** 12 points
- **Section C (Chain Rule):** 12 points
- **Section D (Optimization):** 16 points
- **Section E (ML Applications):** 20 points
- **Section F (Advanced Problems):** 24 points
- **Bonus Questions:** 10 points extra credit

### Grading Scale:
- A: 90-100+ points
- B: 80-89 points
- C: 70-79 points
- D: 60-69 points
- F: Below 60 points

### Key Concepts Tested:
✓ Basic differentiation rules
✓ Chain rule applications
✓ Partial derivatives and gradients
✓ Optimization and critical points
✓ Gradient descent algorithm
✓ ML-specific applications (sigmoid, backpropagation)
✓ Regularization and constraint optimization

### Study Recommendations:
- Master basic derivative rules first
- Practice chain rule extensively
- Understand gradient geometric interpretation
- Connect optimization to ML algorithms
- Work through backpropagation step-by-step
- Practice with real optimization problems