"""
Data Validation Module (Custom Implementation).
Replaces Great Expectations due to Python 3.14 compatibility issues.
Ensures MLOps Level 2 Compliance by validating data quality before training.
"""
import pandas as pd
from typing import List, Optional

class DataValidator:
    """
    Lightweight validator that mimics Great Expectations behavior.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.errors = []

    def expect_column_to_exist(self, column: str) -> bool:
        if column not in self.df.columns:
            self.errors.append(f"Missing column: {column}")
            return False
        return True

    def expect_column_values_to_not_be_null(self, column: str) -> bool:
        if column not in self.df.columns:
            # Already caught by existence check usually, but safer to check
            return False
        if self.df[column].isnull().any():
            null_count = self.df[column].isnull().sum()
            self.errors.append(f"Column '{column}' contains {null_count} null values.")
            return False
        return True
    
    def validate(self) -> bool:
        return len(self.errors) == 0

def validate_training_data(df: pd.DataFrame) -> bool:
    """
    Validates the training DataFrame using custom checks.
    
    Checks:
    - Required columns exist.
    - No null values in critical columns.
    
    Args:
        df: Input DataFrame to validate.
        
    Returns:
        bool: True if validation passes, raises ValueError otherwise.
    """
    print("Starting Custom Data Validation (GX-Style)...")
    
    validator = DataValidator(df)
    
    # 1. Check for Critical Columns
    required_columns = ["skills", "qualification", "experience_level", "job_role"]
    for col in required_columns:
        validator.expect_column_to_exist(col)

    # 2. Check for Null Values
    if validator.validate(): # Only check values if columns exist
        validator.expect_column_values_to_not_be_null("skills")
        validator.expect_column_values_to_not_be_null("job_role")
        validator.expect_column_values_to_not_be_null("qualification")
        validator.expect_column_values_to_not_be_null("experience_level")

    # Evaluate Results
    if not validator.validate():
        print("[FAIL] Data Validation FAILED!")
        for error in validator.errors:
            print(f"   - {error}")
        
        # Raise error to stop pipeline
        raise ValueError("Data Validation Failed via Custom Validator.")
    
    print("[OK] Data Validation PASSED.")
    return True

if __name__ == "__main__":
    # Test run
    try:
        # Mock data for self-test
        data = pd.DataFrame({
            "skills": ["A", "B"], 
            "qualification": ["M.Sc", "B.Sc"],
            "experience_level": ["Mid", "Junior"],
            "job_role": ["Role1", "Role2"]
        })
        validate_training_data(data)
    except Exception as e:
        print(f"Error: {e}")
