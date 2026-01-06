# Ferocia Term Deposit Prediction Service (Tiny & Production-Ready)

## Overview

A lightweight, production-grade machine learning service designed to predict whether a customer will subscribe to a term deposit.
Built with a focus on **CheckOps**, **Reproducibility**, and **Business Impact** rather than raw academic accuracy.

## Architecture

This service is split into two distinct parts:

1.  **Part A: The Training Pipeline** (`src/train.py`)

    - **Deterministic Splitting**: Uses SHA-256 hashing of customer attributes (`utils.py`) to create a stable "Pseudo-ID". This ensures the same customer always ends up in the same split (Control vs. Train) across different runs and environments.
    - **Leakage Protection**: Explicitly drops the `duration` feature (call length), which is a proxy for the target variable and unavailable pre-call.
    - **Model**: A `LightGBM` classifier optimized for the 88:12 class imbalance.

2.  **Part B: The Serving Layer** (`src/serve.py`)
    - **FastAPI**: A high-performance async API for real-time inference.
    - **Drift Monitoring**: Implements a real-time `Population Stability Index (PSI)` check on predictions.
    - **Data Integrity**: Auto-validates incoming JSON for null values in critical features (Balance, Housing).

## Key Components

- **Pseudo-ID**: `hash(age + job + education + balance)` % 100.
  - Allows "Control Group" persistence without a database.
- **DriftMonitor**: Calculates the shift between training probabilities and live traffic.
- **Business Logic**:
  - Optimized Threshold: `0.5` is NOT assumed. usage of `src/train.py` calculates the profit-maximizing threshold based on Cost per Call vs. Profit per Deposit.

## Tool Stack

- **Core**: Python 3.10+
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`, `lightgbm`, `joblib`
- **API & Serving**: `fastapi`, `uvicorn`, `pydantic`
- **Testing**: `pytest`, `requests` (for integration checks)
- **AI/Dev Tools**: Agentic IDE (Cursor-like), LLMs (Gemini/GPT-4) for boilerplate & docs.

## Usage

### 1. Setup

```bash
# Create a virtual environment (Recommended)
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# IMPORTANT: Ensure dataset is in the correct folder
# Place 'dataset.csv' into the /data folder
# Expected path: ./data/dataset.csv
```

### 2. Train the Model (Part A)

Executes the pipeline, splits data (80/20), trains LightGBM, and saves the artifact locally.

```bash
python src/train.py
```

_Outputs:_

- `models/term_deposit_model.joblib`: The serialized pipeline.
- `models/threshold.txt`: The optimal decision threshold.

### 3. Run the API (Part B)

Starts the production server.

```bash
uvicorn src.serve:app --reload
```

**Test the endpoint:**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"age": 30, "job": "admin", "marital": "single", "education": "secondary", "default": "no", "balance": 100, "housing": "yes", "loan": "no", "contact": "cellular", "day": 5, "month": "may", "campaign": 1, "pdays": -1, "previous": 0, "poutcome": "unknown"}'
```

## Business KPIs & Simulation Results

| Metric                | Value (Simulated) | Business Interpretation                                                          |
| :-------------------- | :---------------- | :------------------------------------------------------------------------------- |
| **ROC-AUC**           | ~0.75             | The model is 75% effective at ranking a subscriber higher than a non-subscriber. |
| **Uplift**            | +25%              | Expected increase in conversion rate compared to random calling (Control Group). |
| **Optimal Threshold** | Variable          | Adjusted dynamically to maximize net profit (Profit - Call Cost).                |

## Pros & Cons (Model Selection)

**Why LightGBM?**

- **Pros**: Handles non-linear relationships (e.g., age vs. balance) significantly better than Logistic Regression. Tiny memory footprint (<500KB).
- **Cons**: Less interpretable than a linear equation (requires SHAP for explanation).

**Why Deterministic Hashing?**

- **Pros**: Zero-state reproducibility. No need for a "User Table" in the dev environment.
- **Cons**: Potential for collision if input features have low entropy (mitigated by including `balance`).

## Transparency

- **AI Assistants**: This codebase was developed with the assistance of LLMs (Gemini/ChatGPT) for boilerplate generation and docstring formatting, adhering to the "Human in the Loop" engineering standard.
- **Usage Log**: See `/AI_TRANSCRIPTS/ai_usage_log.md` for a summary of AI sessions.
- **Data Sources**: Uses the `edm` dataset provided for the exercise.

## Future Roadmap (Drift)

- [x] Null Rate Monitoring
- [x] PSI (Population Stability)
- [ ] Automated Retraining Trigger (Airflow)
