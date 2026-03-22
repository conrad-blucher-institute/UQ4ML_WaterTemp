#!/usr/bin/env python3
"""
Test the deletingMissingValues function
"""

import pandas as pd
import numpy as np
from src.helper.utils_mse_crps import deletingMissingValues

# Create test data with -999 values
test_data = pd.DataFrame({
    'date': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'],
    'temp1': [1.0, -999.0, 3.0, 4.0],
    'temp2': [5.0, 6.0, -999.0, 8.0],
    'temp3': [9.0, 10.0, 11.0, 12.0]
})

print("Original test data:")
print(test_data)
print(f"Shape: {test_data.shape}")
print(f"Rows with -999: {(test_data == -999).any(axis=1).sum()}")

print("\n" + "="*60)
print("Running deletingMissingValues...")
print("="*60 + "\n")

result = deletingMissingValues(test_data)

print("\nResult:")
print(result)
print(f"Shape: {result.shape}")
print(f"Rows with -999: {(result == -999).any(axis=1).sum()}")

expected_rows = 2  # Only rows 0 and 3 should remain (rows 1 and 2 have -999)
if len(result) == expected_rows and (result == -999).any(axis=1).sum() == 0:
    print("\n✓ Test PASSED: Function correctly removed rows with -999")
else:
    print(f"\n✗ Test FAILED: Expected {expected_rows} rows with no -999, got {len(result)} rows with {(result == -999).any(axis=1).sum()} -999 values")
