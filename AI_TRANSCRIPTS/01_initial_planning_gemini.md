# Session 1: Initial ML Service Planning

**Date**: 2026-01-06
**Tool**: Google Gemini
**Duration**: ~30 minutes

## Context Provided to AI

Working on Ferocia take-home: Build tiny but production-ready ML service for term deposit prediction.

## My Prompt

"Develop a machine learning model in Python that predicts, its need to be tiny but production-ready ML service that predicts whether a customer will subscribe to a term deposit using the provided edm dataset. list potential pros and cons for each model you selected. for each pro and cons verify using factual reasoning and provide the final selection. We also need to simulate like real data. set 20% control baseline and 80% for training. we traffic to deterministic so it not all random split every run so construct a pseudo-ID. we need to answer business kpi like expected uplift, false positive cost, conversion rate, lift vs baseline ROC-AUC for diagnostic. don't worry about code plan out the idea first"

## AI Response

"This is a sophisticated request that moves beyond a simple script and into the realm of Decision Support Systems. Since we are aiming for a "tiny but production-ready" service, the focus will be on high-impact, low-latency models that provide clear business interpretability.

Below is the strategic plan for developing this ML service, focusing on deterministic data splitting, business-centric KPIs, and model selection.

1. Data Architecture & Preprocessing
   To ensure the system is production-ready and deterministic, we will implement the following:

Deterministic Splitting via Pseudo-ID
Standard random splits can lead to "data leakage" across different runs if not handled carefully.

The Logic: We will create a pseudo_id by hashing a combination of stable features (e.g., age, job, marital, education).

The Split: We will use a modulo operation on the hash to assign 20% to the Control (Baseline) and 80% to the Training/Validation set. This ensures that the same customer always falls into the same bucket every time the code runs.

Feature Engineering
Categorical Encoding: One-hot encoding for job, education, and contact.

Temporal Features: Converting month into a cyclical numeric representation (sine/cosine) to capture seasonal trends in banking.

Binary Conversion: Mapping "yes/no" fields (housing, loan, default) to 1 and 0.

2. Model Selection: Pros & Cons
   We will evaluate two "tiny" yet powerful candidates suitable for a lightweight service.

Model Pros Cons Factual Reasoning
Logistic Regression High Interpretability: coefficients directly show how features impact conversion. Fast: Millisecond inference. Linearity Constraint: Struggles with complex interactions (e.g., how age and balance interact). Best for "production-ready" baselines where we must explain why a customer was targeted to stakeholders.
LightGBM (Small) Handles Non-linearity: Excellent at finding complex patterns in financial data. Efficient: Very small memory footprint. Black Box: Harder to explain than regression without SHAP values. Banking datasets often have non-linear thresholds (e.g., balance >$5,000 might trigger a sharp increase in conversion).
Final Selection: LightGBM (Gradient Boosting) While Logistic Regression is simpler, banking behavior is rarely linear. A "tiny" Gradient Boosting model (with limited depth) offers significantly better ROC-AUC while remaining fast enough for a microservice environment.

3. Business KPIs & Diagnostics
   We will not just measure "Accuracy"; we will measure Value.

Conversion & Uplift
Baseline Conversion Rate (CR
base

): The conversion rate observed in the 20% Control set.

Expected Uplift: The percentage increase in conversions in the Training/Test set compared to the Control set.

Uplift=(
CR
base

CR
model

−CR
base

)×100

Financial Impact (Cost-Benefit)
False Positive (FP) Cost: The cost of calling/marketing to a customer who says "No" (wasted labor/SMS costs).

False Negative (FN) Cost: The "Opportunity Cost" of missing a customer who would have subscribed (lost lifetime value of the deposit).

Technical Diagnostic
ROC-AUC: This will tell us how much better our model is at ranking customers compared to random guessing (0.5). For this type of campaign, we aim for >0.75.

4. Simulation & Production Plan
   The service will be structured as a simple Python class:

DataIngestion: Loads the data and generates the deterministic pseudo_id.

FeaturePipeline: Transforms raw dictionary inputs into numeric vectors.

Predictor: The LightGBM model that outputs a probability score.

BusinessLayer: Converts the probability into a "Call" or "Don't Call" decision based on the FP Cost vs. Potential Deposit Profit."

## Key Decisions Made

- ✅ Selected LightGBM over Logistic Regression (handles non-linearity)
- ✅ Chose hash-based deterministic splitting for reproducibility
- ✅ Defined business KPIs: uplift, ROC-AUC, cost-benefit analysis
- ⚠️ Identified need for drift monitoring in Part B

## What I Accepted

- Deterministic splitting via SHA-256 hash
- LightGBM for production readiness
- Business-first KPI framework

## What I Modified/Rejected

- Simplified the cyclical temporal features for the MVP to keep it "tiny".
