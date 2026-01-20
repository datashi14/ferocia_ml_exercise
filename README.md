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

### 4. Demo: Verify End-to-End

For a quick validation of the service, open a new terminal and run:

**Check Health:**

```bash
curl http://127.0.0.1:8000/health
# Returns: {"status": "healthy", "threshold": 0.44}
```

**Run Prediction (Subscriber Profile):**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"age": 30, "job": "admin", "marital": "single", "education": "secondary", "default": "no", "balance": 100, "housing": "yes", "loan": "no", "contact": "cellular", "day": 5, "month": "may", "campaign": 1, "pdays": -1, "previous": 0, "poutcome": "unknown"}'
```

_Expected Response:_ `{"prediction": "NO_CALL", "probability": 0.04...}` (Low probability due to 'unknown' outcome and low balance).

## Testing Strategy

We employ a "CheckOps" approach to ensure reliability:

### Unit Tests (Logic Verification)

Validates the critical deterministic components.

```bash
pytest tests/test_pipeline.py
```

- **What it checks**:
  - Does `generate_pseudo_id` always return the same Hash for the same input?
  - Does the 20/80 split correctly buckets IDs based on the hash?

### Integration Tests (API)

Validates the serving infrastructure.

- **Manual**: Using the `curl` commands above.
- **Automated**: The `src/serve.py` contains data integrity warnings that trigger if you send nulls for critical features.

## Business KPIs (Framework)

This service is designed to measure:

### Model Performance

- **ROC-AUC**: Ranking power (currently printed during training)
- **Precision-Recall Curve**: Class imbalance handling
- **F1-Score**: Balance between false positives and false negatives

### Business Impact (Requires A/B Testing)

- **Expected Uplift**: Conversion rate improvement vs. control group
  - _Measurement_: Compare 80% model-scored group vs. 20% random control
  - _Timeline_: 30-day window for term deposit confirmations
- **Cost-Benefit**: ROI calculation based on call cost vs. deposit value
  - _Future_: Implement in `src/monitor.py` once cost parameters defined

### Production Health

- **PSI (Population Stability Index)**: Drift detection (implemented in `serve.py`)
- **Null Rate Monitoring**: Data quality checks (implemented)
- **Prediction Distribution**: Real-time monitoring endpoint

**Note**: Actual uplift and ROI metrics require production deployment with control group.
Current implementation provides the _framework_ for these measurements.

## Pros & Cons (Model Selection)

**Why LightGBM?**

- **Pros**: Handles non-linear relationships (e.g., age vs. balance) significantly better than Logistic Regression. Tiny memory footprint (<500KB).
- **Cons**: Less interpretable than a linear equation (requires SHAP for explanation).

**Why Deterministic Hashing?**

- **Pros**: Zero-state reproducibility. No need for a "User Table" in the dev environment.
- **Cons**: Potential for collision if input features have low entropy (mitigated by including `balance`).

## Time Allocation (3-Hour Constraint)

Following Ferocia's 40/60 split recommendation:

### Part A: Training Pipeline (1h 20min)

- 0:00-0:20 → EDA, identify duration leakage and class imbalance
- 0:20-0:50 → Implement deterministic splitting with hash collision analysis
- 0:50-1:20 → LightGBM training with imbalance handling + threshold optimization

### Part B: Serving Layer (1h 40min)

- 1:20-2:00 → FastAPI endpoint with Pydantic validation
- 2:00-2:30 → PSI drift monitoring implementation
- 2:30-2:50 → Integration testing and documentation
- 2:50-3:00 → AI transcript organization and README finalization

**Scope Trade-offs Made**:

- ✅ Comprehensive drift monitoring
- ✅ Business-optimized threshold
- ⚠️ Limited unit test coverage (1 example test per Ferocia guidance)
- ⚠️ No hyperparameter tuning (not required per exercise)

## AI Tool Usage & Transparency

### Tools Used

- **Google Gemini**: Strategic planning, architecture design, gap analysis
- **GitHub Copilot**: Inline code suggestions (disabled for critical logic)

### AI-Assisted Components

See detailed transcripts in `/AI_TRANSCRIPTS/` folder.

#### Strategic Planning (Gemini)

- Initial model selection rationale (LightGBM vs LogReg)
- Business KPI framework design
- Production gap analysis (duration leakage, class imbalance)

#### Code Generation (Mix of AI + Manual)

- **AI-Generated**: Boilerplate (FastAPI endpoint structure, pytest templates)
- **Human-Written**: Core ML logic (deterministic splitting, PSI calculation)
- **AI-Suggested, Human-Modified**: Drift monitoring implementation

### Process Documentation

All major AI interactions are documented in chronological order:

1. `01_initial_planning_gemini.md` - Model selection & architecture
2. `02_architecture_refinement_gemini.md` - Ferocia-specific adjustments
3. `03_production_gaps_analysis_gemini.md` - Critical error identification

### What I Changed From AI Suggestions

- AI suggested SMOTE for imbalance → I used LightGBM's native weighting (simpler)
- AI proposed separate drift script → I integrated into `serve.py` (cohesive)
- AI generated generic tests → I wrote domain-specific edge case tests

### Human-in-the-Loop Approach

Every AI suggestion was:

1. Critically evaluated against Ferocia's requirements
2. Validated against my production experience (Wesfarmers/Tabcorp)
3. Modified to fit 3-hour time constraint
4. Committed with clear attribution in git messages

## Future Roadmap (Drift)

- [x] Null Rate Monitoring
- [x] PSI (Population Stability)
- [ ] Automated Retraining Trigger (Airflow)
