import hashlib
import pandas as pd
import numpy as np

def generate_pseudo_id(row: pd.Series) -> str:
    """
    Generates a deterministic pseudo-ID based on stable customer features.
    
    Args:
        row: A pandas Series containing customer attributes.
        
    Returns:
        A SHA-256 hash string representing the unique customer ID.
    """
    # Combine features into a single string. 
    # ensuring high cardinality to avoid collisions (balance, day)
    # and stable features (age, job, education)
    
    # Handle potential non-string types for concatenation
    features = [
        str(row.get('age', '')),
        str(row.get('job', '')),
        str(row.get('education', '')),
        str(row.get('balance', '')),
        str(row.get('day', '')),
        str(row.get('month', ''))
    ]
    
    raw_id = "_".join(features)
    return hashlib.sha256(raw_id.encode('utf-8')).hexdigest()

def assign_split(pseudo_id: str, control_pct: int = 20) -> str:
    """
    Deterministically assigns a record to 'control' or 'train' based on pseudo-ID.
    
    Args:
        pseudo_id: The customer's unique hash ID.
        control_pct: The percentage of traffic to allocate to control (0-100).
        
    Returns:
        'control' or 'train'
    """
    # Use the last 2 characters of the hash to create an integer 0-255
    # Then modulo 100 to get 0-99
    hash_int = int(pseudo_id[-2:], 16)
    bucket = hash_int % 100
    
    if bucket < control_pct:
        return 'control'
    return 'train'

def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    """
    Calculate the Population Stability Index (PSI) to compare two distributions.
    
    Args:
        expected: Numpy array of expected values (e.g., training probability scores)
        actual: Numpy array of actual values (e.g., production probability scores)
        buckets: Number of quantiles to use
        
    Returns:
        PSI value
    """
    def scale_range (input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

    if buckettype == 'bins':
        breakpoints = np.percentile(expected, breakpoints)
    
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    def sub_psi(e_perc, a_perc):
        if a_perc == 0: a_perc = 0.0001
        if e_perc == 0: e_perc = 0.0001
        
        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        return value

    psi_value = np.sum([sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents))])

    return psi_value
