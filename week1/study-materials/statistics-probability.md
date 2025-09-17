# Statistics & Probability for AI/ML

## Descriptive Statistics

### Measures of Central Tendency

#### Mean (Average)
```
Mean = (x₁ + x₂ + ... + xₙ) / n
Example: [2, 4, 6, 8, 10] → Mean = 30/5 = 6
```

#### Median
The middle value when data is sorted
```
Example: [1, 3, 5, 7, 9] → Median = 5
Example: [1, 3, 5, 7] → Median = (3+5)/2 = 4
```

#### Mode
The most frequently occurring value
```
Example: [1, 2, 2, 3, 4] → Mode = 2
```

### Measures of Spread

#### Variance
Average of squared differences from the mean
```
Variance = Σ(xᵢ - mean)² / n
```

#### Standard Deviation
Square root of variance
```
Standard Deviation = √(Variance)
```

#### Range
Difference between maximum and minimum values

## Probability Fundamentals

### Basic Probability Rules

#### Probability of an Event
```
P(A) = Number of favorable outcomes / Total number of outcomes
```

#### Addition Rule
```
P(A or B) = P(A) + P(B) - P(A and B)
```

#### Multiplication Rule
```
P(A and B) = P(A) × P(B|A)
```

### Conditional Probability
```
P(A|B) = P(A and B) / P(B)
```
**Interpretation**: Probability of A given that B has occurred

## Bayes' Theorem

### Formula
```
P(A|B) = P(B|A) × P(A) / P(B)
```

### Components
- **P(A|B)**: Posterior probability
- **P(B|A)**: Likelihood
- **P(A)**: Prior probability
- **P(B)**: Evidence

### Example: Medical Diagnosis
```
Disease prevalence: 1% (Prior)
Test accuracy: 95% (Likelihood)
Positive test result (Evidence)

P(Disease|Positive) = P(Positive|Disease) × P(Disease) / P(Positive)
```

### ML Applications
- **Naive Bayes Classifier**
- **Bayesian Optimization**
- **Uncertainty Quantification**
- **A/B Testing**

## Probability Distributions

### Discrete Distributions

#### Bernoulli Distribution
Single trial with two outcomes (success/failure)
```
P(X = 1) = p
P(X = 0) = 1 - p
```

#### Binomial Distribution
n independent Bernoulli trials
```
P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
```

#### Poisson Distribution
Number of events in fixed interval
```
P(X = k) = (λ^k × e^(-λ)) / k!
```

### Continuous Distributions

#### Normal (Gaussian) Distribution
```
f(x) = (1/√(2πσ²)) × e^(-(x-μ)²/(2σ²))
```
- **μ**: Mean
- **σ**: Standard deviation
- **68-95-99.7 Rule**: 68% within 1σ, 95% within 2σ, 99.7% within 3σ

#### Uniform Distribution
All values equally likely in [a, b]
```
f(x) = 1/(b-a) for a ≤ x ≤ b
```

#### Exponential Distribution
Time between events in Poisson process
```
f(x) = λe^(-λx) for x ≥ 0
```

## Statistical Inference

### Hypothesis Testing

#### Steps
1. **Null Hypothesis (H₀)**: No effect/difference
2. **Alternative Hypothesis (H₁)**: Effect exists
3. **Test Statistic**: Measure of evidence
4. **P-value**: Probability of observed result under H₀
5. **Decision**: Reject or fail to reject H₀

#### Common Tests
- **t-test**: Compare means
- **Chi-square test**: Test independence
- **ANOVA**: Compare multiple groups

### Confidence Intervals
Range of plausible values for parameter
```
95% CI for mean: x̄ ± 1.96 × (σ/√n)
```

## Key Concepts for ML

### Central Limit Theorem
Sample means approach normal distribution as sample size increases
- **Importance**: Justifies many statistical methods
- **ML Application**: Bootstrap sampling, confidence intervals

### Law of Large Numbers
Sample average converges to population mean
- **ML Application**: Training with more data improves estimates

### Maximum Likelihood Estimation
Find parameters that make observed data most likely
- **ML Application**: Training neural networks, logistic regression

### Correlation vs Causation
- **Correlation**: Linear relationship between variables
- **Causation**: One variable causes another
- **ML Importance**: Understanding what models can and cannot tell us

## Applications in AI/ML

### Feature Engineering
- **Normalization**: Use mean and standard deviation
- **Outlier Detection**: Use standard deviations from mean
- **Feature Selection**: Use correlation analysis

### Model Evaluation
- **Cross-validation**: Statistical sampling technique
- **Confidence Intervals**: Uncertainty in model performance
- **Hypothesis Testing**: Comparing model performance

### Uncertainty Quantification
- **Bayesian Networks**: Model uncertainty explicitly
- **Ensemble Methods**: Use distribution of predictions
- **Confidence Scores**: Probabilistic outputs

## Study Tips

1. **Visual Learning**: Use histograms, box plots, scatter plots
2. **Real Data**: Practice with actual datasets
3. **Simulation**: Generate data from known distributions
4. **Connect to ML**: Always ask "How is this used in machine learning?"
5. **Software Tools**: Learn NumPy, SciPy, matplotlib

## Common Pitfalls

1. **Confusing correlation with causation**
2. **Misinterpreting p-values**
3. **Ignoring assumptions of statistical tests**
4. **Over-interpreting confidence intervals**
5. **Not checking for outliers**

## Next Steps
- Implement statistical functions in Python
- Work with real datasets to calculate statistics
- Practice Bayesian reasoning problems
- Explore statistical libraries (scipy.stats)
- Apply statistical concepts to ML problems