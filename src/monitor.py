from collections import deque
import numpy as np
from utils import calculate_psi

class DriftMonitor:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.prediction_history = deque(maxlen=window_size)
        self.training_distribution = None # Should be loaded from a reference file
        
    def log_prediction(self, probability: float):
        self.prediction_history.append(probability)
        
    def check_psi(self) -> float:
        if len(self.prediction_history) < 100 or self.training_distribution is None:
            return 0.0
            
        return calculate_psi(
            self.training_distribution, 
            np.array(self.prediction_history)
        )

def check_null_rates(row_dict: dict, critical_features: list) -> list:
    """
    Checks if critical features are missing or null.
    Returns a list of warnings.
    """
    warnings = []
    for feature in critical_features:
        value = row_dict.get(feature)
        if value is None or value == "" or value == "unknown":
            # Note: 'unknown' is specific to this dataset
            warnings.append(f"Missing critical feature: {feature}")
    return warnings
