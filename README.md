# Statistical Illusions  
**When Correct Calculations Produce Misleading Conclusions**

---

## Overview

This project investigates **statistical illusions** — situations where numerical results are mathematically correct but structurally misleading.

Rather than focusing on prediction or real-world datasets, the goal is to analyze how statistical summaries can distort interpretation under specific structural conditions.

All experiments are conducted using explicitly defined **synthetic data-generating processes (DGPs)**, ensuring that the ground truth is known.

The central question is:

> Under what structural conditions does statistical inference fail to represent underlying reality?

---

## Core Thesis

Statistical calculations are internally correct.  
Misinterpretation arises when structural assumptions are ignored.

Illusion does not originate from arithmetic error,  
but from misunderstanding:

- Data structure  
- Distributional properties  
- Selection mechanisms  
- Hidden variables  
- Estimation variance  

---

## Illusions Examined

The project analyzes five distinct mechanisms of statistical distortion:

### 1. Simpson’s Paradox  
Aggregation can reverse relationships across groups.  
**Mechanism:** Aggregation distortion

---

### 2. Mean vs Median Illusion  
The mean may not represent the typical outcome under skewed distributions.  
**Mechanism:** Distributional skewness

---

### 3. Survivorship Bias  
Observed samples may exclude failed entities, inflating performance metrics.  
**Mechanism:** Selection distortion

---

### 4. Spurious Correlation  
Two variables may appear correlated due to a hidden common factor.  
**Mechanism:** Omitted variable bias

---

### 5. Instability under Small Samples  
Strong statistical signals may arise purely from sampling noise.  
**Mechanism:** Estimation variance

---

## Structural Classification

Statistical illusions arise from one of three levels:

### I. Distribution Level  
Skewness, heavy tails, non-representative moments.

### II. Structural Level  
Aggregation, hidden variables, selection mechanisms.

### III. Estimation Level  
Finite-sample variability of estimators.

Understanding which level generates distortion is essential for correct interpretation.

---

## Methodology

- No external datasets required in early phases.
- All results derived from controlled probabilistic models.
- Each illusion analyzed through:
  - Explicit DGP definition
  - Simulation
  - Visualization
  - Structural interpretation

The emphasis is on **mechanism**, not prediction.

---

## Repository Structure

```
statistical-illusions/
│
├── notebooks/
│ ├── 00_introduction.ipynb
│ ├── 01_simpsons_paradox.ipynb
│ ├── 02_mean_vs_median.ipynb
│ ├── 03_survivorship_bias.ipynb
│ ├── 04_spurious_correlation.ipynb
│ ├── 05_instability_under_small_samples.ipynb
│ └── 06_synthesis.ipynb
│
├── src/
│ ├── __init__.py
│ ├── dgp.py
│ ├── simulations.py
│ ├── metrics.py
│ └── visualization.py
│
├── requirements.txt
└── README.md
```


---

## Intended Audience

- Data scientists  
- Quantitative analysts  
- Students of probability and statistics  
- Researchers concerned with inference reliability  

---

## Key Insight

Correct computation does not guarantee correct interpretation.

Statistical inference always depends on:

- Structural assumptions  
- Sample representativeness  
- Distributional properties  
- Estimator stability  

Recognizing statistical illusions is essential to avoid overconfidence in quantitative reasoning.