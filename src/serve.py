from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import time
import logging
from monitor import check_null_rates, DriftMonitor

app = FastAPI(title="Ferocia Term Deposit Prediction Service")

# Global State
model = None
threshold = 0.5
monitor = DriftMonitor()
logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str

@app.on_event("startup")
def load_artifacts():
    global model, threshold
    model_path = os.path.join("models", "term_deposit_model.joblib")
    threshold_path = os.path.join("models", "threshold.txt")
    
    try:
        model = joblib.load(model_path)
        with open(threshold_path, "r") as f:
            threshold = float(f.read().strip())
        logger.info(f"Model loaded. Decision Threshold: {threshold}")
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        raise RuntimeError("Model loading failed")

@app.get("/health")
def health_check():
    return {"status": "healthy", "threshold": threshold}

@app.post("/predict")
def predict(data: CustomerData, request: Request):
    if not model:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    start_time = time.time()
    data_dict = data.dict()
    
    # 1. Data Integrity Check
    critical_features = ["balance", "campaign", "housing"] # Example criticals
    warnings = check_null_rates(data_dict, critical_features)
    if warnings:
        logger.warning(f"Data Quality Warning: {warnings}")

    # 2. Preprocess (Convert to DataFrame)
    # The pipeline expects a DataFrame with specific columns
    df = pd.DataFrame([data_dict])
    
    # 3. Predict
    try:
        prob = model.predict_proba(df)[0][1]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction logic failed")
    
    # 4. Decision Logic
    decision = "CALL" if prob >= threshold else "NO_CALL"
    
    # 5. Monitoring
    monitor.log_prediction(prob)
    psi = monitor.check_psi()
    
    latency = (time.time() - start_time) * 1000
    
    logger.info(f"Pred: {prob:.4f} | Dec: {decision} | Latency: {latency:.2f}ms | PSI: {psi:.4f}")
    
    return {
        "prediction": decision,
        "probability": round(prob, 4),
        "decision_threshold": threshold,
        "warnings": warnings,
        "monitoring": {
            "psi_alert": psi > 0.25,
            "latency_ms": round(latency, 2)
        }
    }
