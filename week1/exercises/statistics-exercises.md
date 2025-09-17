# Statistics & Probability Exercises

## Exercise Set 1: Descriptive Statistics

### Problem 1: Basic Statistics
Given the dataset: [12, 15, 18, 20, 22, 25, 28, 30, 32, 35]

Calculate:
1. Mean
2. Median
3. Mode (if any)
4. Range
5. Variance
6. Standard deviation

**Solutions:**
1. Mean = (12+15+18+20+22+25+28+30+32+35)/10 = 23.7
2. Median = (22+25)/2 = 23.5
3. Mode = None (no repeated values)
4. Range = 35-12 = 23
5. Variance = Σ(xi - mean)²/n = 64.81
6. Standard deviation = √64.81 ≈ 8.05

### Problem 2: Real-World Statistics
A dataset of test scores: [78, 85, 92, 76, 88, 94, 82, 89, 91, 77, 86, 90]

1. Calculate the 25th, 50th, and 75th percentiles
2. Identify any potential outliers using the IQR method
3. Interpret what these statistics tell us about student performance

**Solutions:**
1. Sorted: [76, 77, 78, 82, 85, 86, 88, 89, 90, 91, 92, 94]
   - Q1 (25th) = 79, Q2 (50th) = 87, Q3 (75th) = 90.75
2. IQR = 90.75 - 79 = 11.75
   - Lower fence: 79 - 1.5×11.75 = 61.375
   - Upper fence: 90.75 + 1.5×11.75 = 108.375
   - No outliers (all values within fences)
3. Interpretation: Most students performed well (median=87), with relatively tight distribution around high scores

## Exercise Set 2: Probability Fundamentals

### Problem 3: Basic Probability
A bag contains 5 red balls, 3 blue balls, and 2 green balls.

1. What's the probability of drawing a red ball?
2. What's the probability of drawing a blue or green ball?
3. If you draw two balls without replacement, what's the probability both are red?

**Solutions:**
1. P(Red) = 5/10 = 0.5
2. P(Blue or Green) = P(Blue) + P(Green) = 3/10 + 2/10 = 0.5
3. P(Red, Red) = P(First Red) × P(Second Red | First Red) = (5/10) × (4/9) = 20/90 = 2/9

### Problem 4: Conditional Probability
In a company:
- 60% of employees are engineers
- 40% of engineers have advanced degrees
- 20% of non-engineers have advanced degrees

1. What's the probability an employee has an advanced degree?
2. If an employee has an advanced degree, what's the probability they're an engineer?

**Solutions:**
1. P(Advanced) = P(Advanced|Engineer) × P(Engineer) + P(Advanced|Non-Engineer) × P(Non-Engineer)
   = 0.4 × 0.6 + 0.2 × 0.4 = 0.24 + 0.08 = 0.32

2. Using Bayes' theorem:
   P(Engineer|Advanced) = P(Advanced|Engineer) × P(Engineer) / P(Advanced)
   = (0.4 × 0.6) / 0.32 = 0.24 / 0.32 = 0.75

## Exercise Set 3: Probability Distributions

### Problem 5: Binomial Distribution
A student takes a 10-question multiple choice test where each question has 4 options.

1. If the student guesses randomly, what's the probability of getting exactly 3 questions correct?
2. What's the probability of passing (getting at least 6 correct)?
3. What's the expected number of correct answers?

**Solutions:**
1. n=10, p=0.25, X~Binomial(10, 0.25)
   P(X = 3) = C(10,3) × (0.25)³ × (0.75)⁷ ≈ 0.250

2. P(X ≥ 6) = 1 - P(X ≤ 5) = 1 - 0.980 ≈ 0.020

3. E[X] = np = 10 × 0.25 = 2.5

### Problem 6: Normal Distribution
Heights of adult males are normally distributed with mean μ = 70 inches and standard deviation σ = 3 inches.

1. What percentage of men are taller than 76 inches?
2. What height represents the 90th percentile?
3. What's the probability a randomly selected man is between 67 and 73 inches tall?

**Solutions:**
1. Z = (76-70)/3 = 2, P(Z > 2) = 1 - 0.9772 = 0.0228 = 2.28%

2. For 90th percentile: Z ≈ 1.28
   Height = μ + Z×σ = 70 + 1.28×3 = 73.84 inches

3. Z₁ = (67-70)/3 = -1, Z₂ = (73-70)/3 = 1
   P(-1 < Z < 1) = 0.8413 - 0.1587 = 0.6826 = 68.26%

## Exercise Set 4: Statistical Inference

### Problem 7: Hypothesis Testing
A manufacturer claims their light bulbs last an average of 1000 hours. You test 25 bulbs and find a sample mean of 950 hours with a sample standard deviation of 100 hours.

Test at α = 0.05 level whether the manufacturer's claim is correct.

**Solution:**
```
H₀: μ = 1000 (manufacturer's claim is correct)
H₁: μ ≠ 1000 (two-tailed test)

Test statistic: t = (x̄ - μ₀)/(s/√n) = (950 - 1000)/(100/√25) = -50/20 = -2.5

Degrees of freedom: df = n-1 = 24
Critical values: ±t₀.₀₂₅,₂₄ = ±2.064

Since |t| = 2.5 > 2.064, reject H₀
Conclusion: Evidence suggests the true mean is significantly different from 1000 hours
```

### Problem 8: Confidence Intervals
From the same sample (n=25, x̄=950, s=100), construct a 95% confidence interval for the true mean.

**Solution:**
```
95% CI: x̄ ± t₀.₀₂₅,₂₄ × (s/√n)
CI: 950 ± 2.064 × (100/√25)
CI: 950 ± 2.064 × 20
CI: 950 ± 41.28
CI: [908.72, 991.28]

Interpretation: We are 95% confident the true mean lifetime is between 908.72 and 991.28 hours
```

## Exercise Set 5: Bayes' Theorem Applications

### Problem 9: Medical Diagnosis
A disease affects 1% of the population. A diagnostic test has:
- Sensitivity (true positive rate): 95%
- Specificity (true negative rate): 98%

If a person tests positive, what's the probability they have the disease?

**Solution:**
```
Let D = has disease, T = tests positive

P(D) = 0.01 (prior)
P(T|D) = 0.95 (sensitivity)
P(T|¬D) = 1 - 0.98 = 0.02 (false positive rate)

P(T) = P(T|D)P(D) + P(T|¬D)P(¬D)
     = 0.95×0.01 + 0.02×0.99 = 0.0095 + 0.0198 = 0.0293

P(D|T) = P(T|D)P(D)/P(T) = (0.95×0.01)/0.0293 ≈ 0.324

Only about 32.4% chance the person actually has the disease!
```

### Problem 10: Spam Email Classification
An email filter has the following probabilities:
- P(Spam) = 0.3
- P("Free" | Spam) = 0.8
- P("Free" | Not Spam) = 0.1

If an email contains the word "Free", what's the probability it's spam?

**Solution:**
```
P(Spam | "Free") = P("Free" | Spam) × P(Spam) / P("Free")

P("Free") = P("Free" | Spam) × P(Spam) + P("Free" | Not Spam) × P(Not Spam)
          = 0.8 × 0.3 + 0.1 × 0.7 = 0.24 + 0.07 = 0.31

P(Spam | "Free") = (0.8 × 0.3) / 0.31 = 0.24 / 0.31 ≈ 0.774

About 77.4% probability it's spam
```

## Programming Exercises

### Exercise 11: Statistical Analysis with Python

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
data = np.random.normal(100, 15, 1000)  # Mean=100, SD=15, n=1000

# Calculate descriptive statistics
mean = np.mean(data)
median = np.median(data)
std_dev = np.std(data, ddof=1)
percentiles = np.percentile(data, [25, 50, 75])

print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Percentiles (25, 50, 75): {percentiles}")

# Hypothesis test
t_stat, p_value = stats.ttest_1samp(data, 100)
print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")

# Confidence interval
confidence_interval = stats.t.interval(0.95, len(data)-1,
                                     loc=mean,
                                     scale=stats.sem(data))
print(f"95% CI: {confidence_interval}")
```

### Exercise 12: Probability Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_coin_flips(n_flips, n_simulations):
    """Simulate multiple sequences of coin flips"""
    results = []
    for _ in range(n_simulations):
        flips = np.random.binomial(1, 0.5, n_flips)
        proportion_heads = np.mean(flips)
        results.append(proportion_heads)
    return np.array(results)

# Law of Large Numbers demonstration
n_sims = 1000
for n_flips in [10, 100, 1000, 10000]:
    proportions = simulate_coin_flips(n_flips, n_sims)
    print(f"n={n_flips}: Mean={np.mean(proportions):.3f}, "
          f"Std={np.std(proportions):.3f}")

# Central Limit Theorem visualization
sample_means = []
for _ in range(1000):
    sample = np.random.exponential(2, 30)  # Non-normal population
    sample_means.append(np.mean(sample))

plt.hist(sample_means, bins=50, density=True, alpha=0.7)
plt.title("Distribution of Sample Means (CLT)")
plt.show()
```

## Advanced Applications

### Exercise 13: A/B Testing
An e-commerce site tests two versions of a checkout page:
- Version A: 120 conversions out of 2000 visitors (6%)
- Version B: 140 conversions out of 2000 visitors (7%)

Is the difference statistically significant at α = 0.05?

```python
import scipy.stats as stats

# Data
n_a, x_a = 2000, 120  # Version A
n_b, x_b = 2000, 140  # Version B

# Proportions
p_a = x_a / n_a
p_b = x_b / n_b

# Two-proportion z-test
z_stat, p_value = stats.proportions_ztest([x_a, x_b], [n_a, n_b])

print(f"Version A conversion rate: {p_a:.3f}")
print(f"Version B conversion rate: {p_b:.3f}")
print(f"Z-statistic: {z_stat:.3f}")
print(f"P-value: {p_value:.3f}")
print(f"Significant at α=0.05: {p_value < 0.05}")
```

### Exercise 14: Bootstrap Confidence Intervals
Estimate the confidence interval for the median using bootstrap resampling.

```python
def bootstrap_median(data, n_bootstrap=10000):
    """Calculate bootstrap confidence interval for median"""
    bootstrap_medians = []
    n = len(data)

    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_medians.append(np.median(bootstrap_sample))

    # Calculate confidence interval
    ci_lower = np.percentile(bootstrap_medians, 2.5)
    ci_upper = np.percentile(bootstrap_medians, 97.5)

    return ci_lower, ci_upper, bootstrap_medians

# Example usage
data = [23, 34, 12, 56, 78, 45, 67, 89, 23, 45, 67, 89, 34, 56]
ci_lower, ci_upper, boot_medians = bootstrap_median(data)

print(f"Original median: {np.median(data)}")
print(f"Bootstrap 95% CI for median: [{ci_lower:.2f}, {ci_upper:.2f}]")
```

## Challenge Problems

### Problem 15: Monte Carlo Integration
Use random sampling to estimate π by simulating points in a unit circle.

### Problem 16: Bayesian Update
Implement a Bayesian updating process for estimating a coin's bias after observing sequences of heads and tails.

### Problem 17: Maximum Likelihood Estimation
Given a sample from an exponential distribution, find the maximum likelihood estimator for the parameter λ.

## Self-Assessment Questions

1. When would you use a t-test versus a z-test?
2. Explain the difference between Type I and Type II errors.
3. What does it mean for two events to be independent?
4. How does sample size affect the width of confidence intervals?
5. What assumptions are required for linear regression?

## Real-World Applications

### Application 1: Quality Control
Design a statistical process control chart for monitoring manufacturing defects.

### Application 2: Survey Analysis
Analyze survey data with potential response bias and calculate appropriate confidence intervals.

### Application 3: Clinical Trial Design
Determine sample sizes needed to detect clinically significant differences with adequate statistical power.