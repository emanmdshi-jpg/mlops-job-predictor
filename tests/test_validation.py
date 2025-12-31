"""
Unit Tests for Data Validation using Great Expectations.
"""
import pytest
import pandas as pd
from src.data_validation import validate_training_data

def test_validate_training_data_valid():
    """Test with valid data."""
    df = pd.DataFrame({
        "skills": ["Python", "Java"],
        "qualification": ["B.Sc", "M.Sc"],
        "experience_level": ["Entry", "Mid"],
        "job_role": ["Dev", "Dev"]
    })
    assert validate_training_data(df) is True

def test_validate_training_data_missing_col():
    """Test with missing required column."""
    df = pd.DataFrame({
        "skills": ["Python"],
        # Missing qualification
        "experience_level": ["Entry"],
        "job_role": ["Dev"]
    })
    with pytest.raises(ValueError):
        validate_training_data(df)

def test_validate_training_data_nulls():
    """Test with null values in critical column."""
    df = pd.DataFrame({
        "skills": [None, "Java"],
        "qualification": ["B.Sc", "M.Sc"],
        "experience_level": ["Entry", "Mid"],
        "job_role": ["Dev", "Dev"]
    })
    # GX default behavior for expect_column_values_to_not_be_null might fail
    # We expect it to raise ValueError because our wrapper raises it on failure
    with pytest.raises(ValueError):
        validate_training_data(df)
