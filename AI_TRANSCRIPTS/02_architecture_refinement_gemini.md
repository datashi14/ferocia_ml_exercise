# Session 2: Architecture Refinement

**Date**: 2026-01-06
**Tool**: Google Gemini
**Duration**: ~15 minutes

## Context Provided to AI

"This updated strategy aligns your plan with the Ferocia Mid-Senior ML Engineer take-home requirements. The goal is a "tiny but production-ready" service, focusing on reproducibility, portability, and clean software engineering over Kaggle-level performance."

## AI Response

"1. Refined Data Architecture: Reproducible & Portable
To meet the requirement for reusable and portable code, we will structure the preprocessing as a pipeline that can be shared between training and serving.

Deterministic Splitting (The "Pseudo-ID"): Instead of standard random seeds, we will hash customer attributes (e.g., age, job, education) using hashlib.sha256. We then take the hash modulo 100; if the result is <20, the record goes to the 20% Control/Baseline. This ensures consistent results across different environments without requiring a fixed database state.

The Pipeline: We will use scikit-learn's Pipeline and ColumnTransformer. This packages the scaling of balance and duration, and the encoding of categorical variables (like job or education) into a single artifact.

2. Model Selection & Rationale
   For an MVP, we need a model that is easy to package and serves predictions with low latency.

Model Pros Cons Factual Reasoning
LightGBM

Portable & Fast: The model file is tiny (kilobytes) and inference is extremely quick.

Complexity: Slightly harder to explain to business stakeholders than linear models.

Gradient boosting handles the non-linear relationship between duration (contact time) and conversion better than regression.

Logistic Regression

Simple MVP: Very easy to implement as a first pass.

Underfitting: May miss complex patterns in financial data, leading to poor uplift.

Good for a baseline, but LightGBM is more "production-ready" for modern ML stacks.

Final Selection: LightGBM. It provides superior performance for banking datasets while remaining "lightweight" enough for a tiny API.

3. Business KPIs & Decision Logic
   The exercise asks for an evaluation of whether a customer will subscribe. We will use the following diagnostic framework:

ROC-AUC: Used to verify the model's ability to distinguish between subscribers and non-subscribers regardless of the threshold.

Conversion Rate & Uplift: We will compare the conversion rate in our 80% "Treatment" group (model-led) against the 20% "Control" group (baseline).

Cost-Benefit Analysis: We will simulate the cost of a False Positive (marketing spend on a non-converter) vs. the value of a True Positive (the profit from a new term deposit).

4. Implementation Strategy (Part A & B)
   Following the 40/60 time split guidance:

Part A: Model Training (40%): Focus on the Git history. Each step—data cleaning, hash-based splitting, and training—will be its own commit to show "the evolution of the solution". We will use joblib or pickle to package the model and preprocessing pipeline as a single artifact.

Part B: Model Hosting (60%): We will use FastAPI to create the /predict endpoint. It will accept raw JSON, run it through the saved pipeline, and return a subscription probability.

5. Production & Transparency

Tool Stack: In the README.md, we will document the use of Gemini/ChatGPT to comply with transparency requirements.

Git Strategy: We will avoid "squashing" commits. Instead, we will show the iterative process: e.g., "Initial model train" → "Refactor preprocessing into pipeline"."

## Decisions

- Adopted the 40/60 split strategy.
- Confirmed LightGBM + FastAPI stack.
- Committed to atomic git commits ("evolution of solution").
