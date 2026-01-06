import pytest
import pandas as pd
import numpy as np
from src.utils import generate_pseudo_id, assign_split

# Sample data for testing
@pytest.fixture
def sample_row():
    return pd.Series({
        'age': 30,
        'job': 'technician',
        'education': 'tertiary',
        'balance': 1000,
        'day': 5,
        'month': 'may'
    })

def test_deterministic_hashing(sample_row):
    """Test that the same input always produces the same ID and split"""
    id1 = generate_pseudo_id(sample_row)
    id2 = generate_pseudo_id(sample_row)
    
    assert id1 == id2
    assert len(id2) == 64 # SHA-256 length

def test_split_assignment():
    """Test that split assignment is stable for a known hash"""
    # Create a fake hash where the last 2 chars are '00' (0) -> Control (<20)
    fake_hash_control = "abc..." + "00"
    assert assign_split(fake_hash_control, control_pct=20) == 'control'
    
    # Create a fake hash where the last 2 chars are '63' (99) -> Train (>=20)
    fake_hash_train = "abc..." + "63"
    assert assign_split(fake_hash_train, control_pct=20) == 'train'

def test_hashing_stability():
    """Ensure small changes in input produce different hashes"""
    row1 = pd.Series({'age': 30, 'job': 'admin', 'balance': 100})
    row2 = pd.Series({'age': 31, 'job': 'admin', 'balance': 100})
    
    assert generate_pseudo_id(row1) != generate_pseudo_id(row2)
