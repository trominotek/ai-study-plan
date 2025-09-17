# Statistics & Probability Quiz

**Time Limit: 60 minutes**
**Instructions: Choose the best answer for multiple choice questions. Show your work for calculations.**

---

## Section A: Multiple Choice (3 points each)

### Question 1
A dataset has values [10, 15, 20, 25, 30]. What is the variance?

A) 20
B) 50
C) 62.5
D) 250

<details>
<summary>Answer</summary>
**B) 50**

Mean = (10+15+20+25+30)/5 = 100/5 = 20

Variance = Σ(xi - mean)²/n
= [(10-20)² + (15-20)² + (20-20)² + (25-20)² + (30-20)²]/5
= [100 + 25 + 0 + 25 + 100]/5 = 250/5 = 50
</details>

### Question 2
If P(A) = 0.6 and P(B) = 0.4, and A and B are independent events, what is P(A and B)?

A) 0.24
B) 0.4
C) 0.6
D) 1.0

<details>
<summary>Answer</summary>
**A) 0.24**

For independent events: P(A and B) = P(A) × P(B) = 0.6 × 0.4 = 0.24
</details>

### Question 3
In a normal distribution, approximately what percentage of data falls within 2 standard deviations of the mean?

A) 68%
B) 95%
C) 99.7%
D) 50%

<details>
<summary>Answer</summary>
**B) 95%**

This is the empirical rule (68-95-99.7 rule). About 95% of data falls within 2 standard deviations of the mean.
</details>

### Question 4
Which of the following best describes a Type I error?

A) Failing to reject a true null hypothesis
B) Rejecting a true null hypothesis
C) Failing to reject a false null hypothesis
D) Rejecting a false null hypothesis

<details>
<summary>Answer</summary>
**B) Rejecting a true null hypothesis**

Type I error occurs when we reject H₀ when it's actually true (false positive).
</details>

### Question 5
A coin is flipped 10 times. What type of probability distribution best models the number of heads?

A) Normal
B) Poisson
C) Binomial
D) Exponential

<details>
<summary>Answer</summary>
**C) Binomial**

Fixed number of trials (10), two outcomes (heads/tails), constant probability (0.5) - classic binomial distribution.
</details>

### Question 6
In Bayes' theorem P(A|B) = P(B|A)P(A)/P(B), what is P(A) called?

A) Posterior probability
B) Prior probability
C) Likelihood
D) Evidence

<details>
<summary>Answer</summary>
**B) Prior probability**

P(A) is the prior probability - our belief about A before seeing evidence B.
</details>

---

## Section B: Short Calculations (5 points each)

### Question 7
Calculate the median of the dataset: [7, 2, 9, 4, 5, 8, 1, 6]

<details>
<summary>Answer</summary>
First sort: [1, 2, 4, 5, 6, 7, 8, 9]
n = 8 (even), so median = (4th + 5th values)/2 = (5 + 6)/2 = **5.5**
</details>

### Question 8
If the mean is 50 and standard deviation is 10 in a normal distribution, what is the z-score for a value of 65?

<details>
<summary>Answer</summary>
z = (x - μ)/σ = (65 - 50)/10 = 15/10 = **1.5**
</details>

### Question 9
Two dice are rolled. What is the probability of getting a sum of 7?

<details>
<summary>Answer</summary>
Favorable outcomes for sum = 7: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1) = 6 outcomes
Total possible outcomes = 6 × 6 = 36
Probability = 6/36 = **1/6** or **0.167**
</details>

### Question 10
A bag has 8 red balls and 4 blue balls. What's the probability of drawing 2 red balls without replacement?

<details>
<summary>Answer</summary>
P(1st red) = 8/12 = 2/3
P(2nd red | 1st red) = 7/11

P(2 red balls) = (8/12) × (7/11) = 56/132 = **14/33** ≈ **0.424**
</details>

---

## Section C: Problem Solving (10 points each)

### Question 11
A manufacturing process produces items with a defect rate of 3%. In a batch of 200 items:

a) What is the expected number of defective items?
b) What is the standard deviation of defective items?
c) What probability distribution models this scenario?

<details>
<summary>Answer</summary>
This follows a **binomial distribution** with n = 200, p = 0.03

a) **Expected value:** E[X] = np = 200 × 0.03 = **6 items**

b) **Standard deviation:** σ = √(np(1-p)) = √(200 × 0.03 × 0.97) = √(5.82) ≈ **2.41 items**

c) **Distribution:** Binomial(n=200, p=0.03)
</details>

### Question 12
A test has a mean score of 75 with a standard deviation of 12. Assuming normal distribution:

a) What percentage of students score above 87?
b) What score represents the 90th percentile?

<details>
<summary>Answer</summary>
a) **Score above 87:**
z = (87 - 75)/12 = 12/12 = 1.0
P(Z > 1.0) = 1 - 0.8413 = 0.1587 = **15.87%**

b) **90th percentile:**
For 90th percentile, z ≈ 1.28
Score = μ + z×σ = 75 + 1.28×12 = 75 + 15.36 = **90.36**
</details>

### Question 13
In a clinical trial:
- Disease prevalence: 2%
- Test sensitivity (true positive rate): 90%
- Test specificity (true negative rate): 95%

If someone tests positive, what's the probability they actually have the disease?

<details>
<summary>Answer</summary>
Using Bayes' theorem:
- P(Disease) = 0.02
- P(Positive | Disease) = 0.90
- P(Positive | No Disease) = 1 - 0.95 = 0.05

**Calculate P(Positive):**
P(Positive) = P(Pos|Disease)×P(Disease) + P(Pos|No Disease)×P(No Disease)
= 0.90×0.02 + 0.05×0.98 = 0.018 + 0.049 = 0.067

**Calculate P(Disease | Positive):**
P(Disease|Positive) = P(Pos|Disease)×P(Disease) / P(Positive)
= (0.90 × 0.02) / 0.067 = 0.018 / 0.067 ≈ **0.269** or **26.9%**
</details>

---

## Section D: Hypothesis Testing (15 points each)

### Question 14
A company claims their light bulbs last an average of 1200 hours. A sample of 16 bulbs has a mean lifetime of 1150 hours with a standard deviation of 80 hours. Test at α = 0.05 if the company's claim is valid.

<details>
<summary>Answer</summary>
**Setup:**
- H₀: μ = 1200 (company's claim is true)
- H₁: μ ≠ 1200 (two-tailed test)
- α = 0.05, n = 16, x̄ = 1150, s = 80

**Test statistic:**
t = (x̄ - μ₀)/(s/√n) = (1150 - 1200)/(80/√16) = -50/(80/4) = -50/20 = **-2.5**

**Critical values:**
df = n - 1 = 15
t₀.₀₂₅,₁₅ = ±2.131

**Decision:**
|t| = 2.5 > 2.131, so **reject H₀**

**Conclusion:** There is sufficient evidence at α = 0.05 to reject the company's claim. The true mean lifetime appears to be significantly different from 1200 hours.
</details>

### Question 15
An online retailer wants to test if a new website design increases conversion rates. The old design had a 5% conversion rate. After implementing the new design, 180 out of 3000 visitors converted.

a) Set up the hypothesis test
b) Calculate the test statistic
c) Make a decision at α = 0.01

<details>
<summary>Answer</summary>
**a) Hypothesis setup:**
- H₀: p = 0.05 (no improvement)
- H₁: p > 0.05 (one-tailed test, new design is better)

**b) Test statistic:**
Sample proportion: p̂ = 180/3000 = 0.06
Standard error: SE = √(p₀(1-p₀)/n) = √(0.05×0.95/3000) = √(0.0000158) = 0.00398

z = (p̂ - p₀)/SE = (0.06 - 0.05)/0.00398 = 0.01/0.00398 = **2.51**

**c) Decision:**
Critical value for α = 0.01 (one-tailed): z₀.₀₁ = 2.33
Since z = 2.51 > 2.33, **reject H₀**

**Conclusion:** At α = 0.01, there is sufficient evidence that the new design increases conversion rates.
</details>

---

## Section E: Applied Statistics (20 points)

### Question 16
A machine learning engineer is evaluating model performance. She collected the following accuracy scores from 10-fold cross-validation:

[0.82, 0.85, 0.79, 0.88, 0.84, 0.86, 0.81, 0.87, 0.83, 0.80]

a) Calculate the mean and standard deviation of accuracy
b) Construct a 95% confidence interval for the true mean accuracy
c) If the baseline model has 80% accuracy, is this new model significantly better? (Use α = 0.05)
d) Interpret your results in the context of machine learning model evaluation

<details>
<summary>Answer</summary>
**a) Descriptive statistics:**
Mean: x̄ = (0.82+0.85+0.79+0.88+0.84+0.86+0.81+0.87+0.83+0.80)/10 = 8.35/10 = **0.835**

For standard deviation:
s² = Σ(xi - x̄)²/(n-1) = 0.000905/(10-1) = 0.0001006
s = √0.0001006 = **0.0317**

**b) 95% Confidence Interval:**
df = 9, t₀.₀₂₅,₉ = 2.262
SE = s/√n = 0.0317/√10 = 0.01003
CI = x̄ ± t×SE = 0.835 ± 2.262×0.01003 = 0.835 ± 0.0227
**CI: [0.812, 0.858]**

**c) Hypothesis test against baseline:**
H₀: μ = 0.80 (no improvement over baseline)
H₁: μ > 0.80 (new model is better)

t = (x̄ - μ₀)/(s/√n) = (0.835 - 0.80)/(0.0317/√10) = 0.035/0.01003 = **3.49**

Critical value: t₀.₀₅,₉ = 1.833 (one-tailed)
Since t = 3.49 > 1.833, **reject H₀**

**d) Interpretation:**
- The new model achieves a mean accuracy of 83.5% ± 3.17%
- We're 95% confident the true accuracy is between 81.2% and 85.8%
- The model performs significantly better than the 80% baseline (p < 0.05)
- The improvement is both statistically significant and practically meaningful
- Cross-validation provides reliable estimates with proper uncertainty quantification
</details>

---

## Bonus Section (5 points each)

### Question 17
In A/B testing, if the control group has 1000 visitors with 50 conversions, and the test group has 1000 visitors with 65 conversions, calculate the 95% confidence interval for the difference in conversion rates.

<details>
<summary>Answer</summary>
**Control group:** p₁ = 50/1000 = 0.05
**Test group:** p₂ = 65/1000 = 0.065
**Difference:** p₂ - p₁ = 0.015

**Standard error for difference:**
SE = √[(p₁(1-p₁)/n₁) + (p₂(1-p₂)/n₂)]
SE = √[(0.05×0.95/1000) + (0.065×0.935/1000)]
SE = √[0.0000475 + 0.0000608] = √0.0001083 = 0.0104

**95% CI for difference:**
(p₂ - p₁) ± 1.96×SE = 0.015 ± 1.96×0.0104 = 0.015 ± 0.0204
**CI: [-0.0054, 0.0354]**

Since the CI includes 0, the difference is not statistically significant at α = 0.05.
</details>

### Question 18
Explain the difference between correlation and causation using a specific example, and describe one method that could help establish causation.

<details>
<summary>Answer</summary>
**Example:** Ice cream sales and drowning deaths both increase in summer months, showing a strong positive correlation. However, ice cream consumption doesn't cause drowning deaths.

**Correlation:** Statistical relationship where two variables change together. High correlation (r close to ±1) indicates strong linear relationship.

**Causation:** One variable directly influences or causes changes in another variable.

**Why they're different:** Both ice cream sales and drowning are caused by a third variable - hot weather. People buy more ice cream when it's hot, and more people swim (and unfortunately sometimes drown) when it's hot.

**Method to establish causation:**
**Randomized Controlled Trial (RCT):** Randomly assign subjects to treatment and control groups, then compare outcomes. Random assignment helps control for confounding variables and establish causal relationships.

**Other methods:** Natural experiments, instrumental variables, regression discontinuity, difference-in-differences analysis.
</details>

---

## Scoring Guide

**Total Points: 100 + 10 bonus**

### Section Breakdown:
- **Section A (Multiple Choice):** 18 points
- **Section B (Short Calculations):** 20 points
- **Section C (Problem Solving):** 30 points
- **Section D (Hypothesis Testing):** 30 points
- **Section E (Applied Statistics):** 20 points
- **Bonus Section:** 10 points extra credit

### Grading Scale:
- A: 90-100+ points
- B: 80-89 points
- C: 70-79 points
- D: 60-69 points
- F: Below 60 points

### Key Concepts Tested:
✓ Descriptive statistics (mean, median, variance, standard deviation)
✓ Probability rules and conditional probability
✓ Bayes' theorem applications
✓ Probability distributions (binomial, normal)
✓ Hypothesis testing (t-tests, z-tests)
✓ Confidence intervals
✓ Statistical significance and p-values
✓ Applied statistics in ML context

### Study Recommendations:
- Review probability rules and Bayes' theorem
- Practice hypothesis testing procedures
- Understand confidence interval interpretation
- Connect statistical concepts to ML applications
- Work on real datasets for applied problems