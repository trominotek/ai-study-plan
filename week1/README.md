# Week 1: Core Mathematical Concepts

**Duration:** Week 1-2 of your AI Engineer Study Plan
**Focus:** Linear algebra basics, statistics & probability, calculus fundamentals

## Learning Objectives

By the end of this week, you should be able to:

- [ ] Perform vector and matrix operations by hand and with NumPy
- [ ] Calculate basic statistics and understand probability distributions
- [ ] Compute derivatives and gradients for optimization problems
- [ ] Apply these mathematical concepts to simple machine learning scenarios
- [ ] Explain the geometric intuition behind linear algebra and calculus operations

## Study Structure

### Daily Schedule (1.5-2.5 hours)
- **Theory (30-45 min):** Read study materials, watch videos
- **Practice (45-75 min):** Work through exercises, implement in Python
- **Review (15 min):** Take notes, create mental connections

### Weekly Breakdown

#### Days 1-3: Linear Algebra
- **Day 1:** Vectors, vector operations, dot products
- **Day 2:** Matrices, matrix operations, transpose, inverse
- **Day 3:** Eigenvalues, eigenvectors, applications to PCA

#### Days 4-6: Statistics & Probability
- **Day 4:** Descriptive statistics, probability rules
- **Day 5:** Probability distributions, Bayes' theorem
- **Day 6:** Statistical inference, hypothesis testing

#### Days 7-9: Calculus
- **Day 7:** Basic derivatives, chain rule
- **Day 8:** Partial derivatives, gradients
- **Day 9:** Optimization, gradient descent

#### Days 10-12: Integration & Practice
- **Day 10:** Review and integration exercises
- **Day 11:** Take quizzes, identify weak areas
- **Day 12:** Python implementations, prepare for next week

## Resources by Topic

### Linear Algebra
**Study Materials:** [`linear-algebra-basics.md`](study-materials/linear-algebra-basics.md)
**Exercises:** [`linear-algebra-exercises.md`](exercises/linear-algebra-exercises.md)
**Quiz:** [`linear-algebra-quiz.md`](quizzes/linear-algebra-quiz.md)

**Key Topics:**
- Vector operations (addition, scalar multiplication, dot product)
- Matrix operations (multiplication, transpose, inverse)
- Eigenvalues and eigenvectors
- Applications to data representation and PCA

**Python Libraries:** NumPy, matplotlib for visualization

### Statistics & Probability
**Study Materials:** [`statistics-probability.md`](study-materials/statistics-probability.md)
**Exercises:** [`statistics-exercises.md`](exercises/statistics-exercises.md)
**Quiz:** [`statistics-quiz.md`](quizzes/statistics-quiz.md)

**Key Topics:**
- Descriptive statistics (mean, variance, standard deviation)
- Probability rules and conditional probability
- Bayes' theorem and applications
- Common probability distributions
- Statistical inference and hypothesis testing

**Python Libraries:** NumPy, SciPy, matplotlib, seaborn

### Calculus
**Study Materials:** [`calculus-fundamentals.md`](study-materials/calculus-fundamentals.md)
**Exercises:** [`calculus-exercises.md`](exercises/calculus-exercises.md)
**Quiz:** [`calculus-quiz.md`](quizzes/calculus-quiz.md)

**Key Topics:**
- Basic derivatives and derivative rules
- Partial derivatives and gradients
- Chain rule applications
- Optimization and critical points
- Gradient descent algorithm

**Python Libraries:** NumPy, matplotlib, SymPy for symbolic math

## Practical Applications

### Mini-Projects to Complete

1. **Linear Algebra in Action** (Days 1-3)
   - Implement basic vector/matrix operations from scratch
   - Apply PCA to a simple 2D dataset
   - Visualize data transformation using matrices

2. **Statistical Analysis** (Days 4-6)
   - Analyze a real dataset using descriptive statistics
   - Implement Bayes' theorem for a classification problem
   - Conduct hypothesis testing on sample data

3. **Optimization Fundamentals** (Days 7-9)
   - Implement gradient descent from scratch
   - Visualize optimization landscapes for simple functions
   - Apply chain rule to compute gradients in a simple neural network

## Assessment Strategy

### Self-Assessment Checklist

**Linear Algebra:**
- [ ] Can multiply matrices by hand and explain the geometric meaning
- [ ] Understand the relationship between eigenvectors and data variance
- [ ] Can implement basic linear algebra operations in Python

**Statistics & Probability:**
- [ ] Can calculate probabilities using Bayes' theorem
- [ ] Understand the difference between population and sample statistics
- [ ] Can interpret confidence intervals and p-values

**Calculus:**
- [ ] Can compute derivatives using chain rule
- [ ] Understand the geometric interpretation of gradients
- [ ] Can implement and explain gradient descent algorithm

### Quiz Performance Targets
- **Linear Algebra Quiz:** 80%+ (Focus on matrix operations and eigenvalue concepts)
- **Statistics Quiz:** 85%+ (Emphasize probability and inference)
- **Calculus Quiz:** 80%+ (Stress optimization and gradient computation)

## Study Tips

### Effective Learning Strategies
1. **Start with intuition:** Understand the geometric or physical meaning before diving into formulas
2. **Practice by hand first:** Work through small examples manually before using software
3. **Connect to ML:** Always ask "How is this used in machine learning?"
4. **Visualize:** Use plots and diagrams to understand abstract concepts
5. **Implement from scratch:** Build understanding by coding algorithms yourself

### Common Pitfalls to Avoid
- Memorizing formulas without understanding concepts
- Skipping the geometric intuition behind mathematical operations
- Not practicing enough computational problems
- Failing to connect math concepts to ML applications
- Rushing through probability concepts (they're crucial for ML)

## Python Implementation Guide

### Essential Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from scipy import linalg
import sympy as sp  # For symbolic math
```

### Key Functions to Master
- **NumPy:** `np.dot()`, `np.linalg.eig()`, `np.linalg.inv()`, `np.transpose()`
- **SciPy:** `stats.norm()`, `stats.ttest_1samp()`, `stats.pearsonr()`
- **Matplotlib:** Basic plotting for visualization of functions and data

## Next Week Preparation

### Transition to Machine Learning Basics
- Review any mathematical concepts that felt challenging
- Ensure you're comfortable with Python's scientific computing stack
- Prepare for supervised learning concepts by understanding how math applies to pattern recognition
- Start thinking about how optimization applies to model training

## Additional Resources

### Recommended Videos
- 3Blue1Brown's "Essence of Linear Algebra" series
- Khan Academy's Statistics and Probability courses
- 3Blue1Brown's "Essence of Calculus" series

### Reference Books
- "Linear Algebra and Its Applications" by David Lay
- "Think Stats" by Allen B. Downey (free online)
- "Calculus: Early Transcendentals" by James Stewart

### Online Practice
- Khan Academy for additional exercises
- Kaggle Learn modules for applied statistics
- Jupyter notebooks for hands-on practice

---

**Remember:** Mathematics is the language of machine learning. Invest time in building strong foundations now, and the advanced concepts will be much easier to grasp later. Don't rush through these fundamentals!