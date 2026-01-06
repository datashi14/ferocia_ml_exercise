import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
import lightgbm as lgb
import joblib
import os
from utils import generate_pseudo_id, assign_split

def load_and_prep_data(filepath: str):
    """
    Loads data, cleans it, and assigns deterministic splits.
    """
    # The dataset uses semi-colons as separators
    df = pd.read_csv(filepath, sep=';')
    
    # Generate Pseudo-ID
    print("Generating Pseudo-IDs...")
    df['pseudo_id'] = df.apply(generate_pseudo_id, axis=1)
    
    # Assign deterministic split
    print("Assigning Train/Control splits...")
    df['split_group'] = df['pseudo_id'].apply(lambda x: assign_split(x, control_pct=20))
    
    # Convert 'y' to binary 0/1 (yes/no)
    df['target'] = df['y'].map({'yes': 1, 'no': 0})
    
    return df

def train_model():
    # Load Data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dataset.csv')
    if not os.path.exists(data_path):
         # Fallback for manual placement or different CWD
         data_path = 'dataset.csv'
         
    df = load_and_prep_data(data_path)
    
    # Split into sets
    train_df = df[df['split_group'] == 'train'].copy()
    control_df = df[df['split_group'] == 'control'].copy()
    
    print(f"Training set size: {len(train_df)}")
    print(f"Control set size: {len(control_df)}")
    
    # Define features
    # CRITICAL: EXCLUDE 'duration' to prevent data leakage
    numeric_features = ['age', 'balance', 'campaign', 'pdays', 'previous']
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    
    # Preprocessing Pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Model Pipeline
    # using is_unbalance=True to handle the low conversion rate (~10%)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(
            random_state=42, 
            n_estimators=100,
            is_unbalance=True,
            n_jobs=-1
        ))
    ])
    
    # Train
    X_train = train_df[numeric_features + categorical_features]
    y_train = train_df['target']
    
    print("Training LightGBM model...")
    model_pipeline.fit(X_train, y_train)
    
    # Evaluation on the full train set (or a hold-out validation set if we preferred)
    # Here we simulate the lift evaluation on the Training vs Control 
    # (In a real campaign, 'Control' is hold-out from treatment, but here we check model stats on the 'train' group 
    # to see how well it fits, usually we would have a validation split INSIDE the train group).
    
    # Let's split the 'train' group into internal train/val for metric reporting to be honest
    X_internal_train, X_val, y_internal_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Provide validation metrics
    val_probs = model_pipeline.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, val_probs)
    precision, recall, _ = precision_recall_curve(y_val, val_probs)
    pr_auc = auc(recall, precision)
    
    print(f"\n--- Model Performance (Validation Split) ---\n")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    # Business KPI Simulation using the Control Group as a Baseline
    # Baseline Conversion Rate
    baseline_cr = control_df['target'].mean()
    print(f"\n--- Business Value Estimation ---\n")
    print(f"Baseline Conversion Rate (Control Group): {baseline_cr * 100:.2f}%")
    
    # Uplift Calculation
    # If we apply the model to the validation set, select top 50% sorted by probability
    # and compare their CR to baseline
    
    # Get predictions for validation set
    validation_results = pd.DataFrame({'target': y_val, 'prob': val_probs})
    
    # Threshold Optimization: Maximize Profit
    # Assumptions: 
    #   profit_per_deposit = $100
    #   cost_per_call = $10
    
    profit_deposit = 100
    cost_call = 10
    
    best_threshold = 0.5
    max_profit = -float('inf')
    
    thresholds = np.linspace(0.1, 0.9, 9)
    for t in thresholds:
        calls = validation_results[validation_results['prob'] >= t]
        if len(calls) == 0: continue
        
        true_positives = calls['target'].sum()
        false_positives = len(calls) - true_positives
        
        profit = (true_positives * profit_deposit) - (len(calls) * cost_call)
        
        if profit > max_profit:
            max_profit = profit
            best_threshold = t
            
    print(f"Optimal Threshold for Profit: {best_threshold:.2f}")
    print(f"Estimated Profit on Validation Set (at optimal threshold): ${max_profit}")
    
    # Save Artifacts
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'term_deposit_model.joblib')
    
    joblib.dump(model_pipeline, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Also save the threshold for the service to use
    with open(os.path.join(model_dir, 'threshold.txt'), 'w') as f:
        f.write(str(best_threshold))

if __name__ == "__main__":
    train_model()
