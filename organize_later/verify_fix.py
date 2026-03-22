#!/usr/bin/env python3
"""
POST-FIX VERIFICATION SCRIPT
Verify that -999 values have been properly removed from training data
"""

import pandas as pd
import numpy as np

print("="*80)
print("POST-FIX VERIFICATION: Checking -999 Removal")
print("="*80)

def check_file(filename, data_type):
    """Check a cleaned CSV file for -999 values"""
    print(f"\n{data_type.upper()} DATA:")
    print("-" * 80)
    
    try:
        df = pd.read_csv(filename, index_col=0)
        print(f"✓ File loaded: {filename}")
        print(f"  Shape: {df.shape}")
        
        # Check for -999
        has_999 = (df == -999).any().any()
        count_999 = (df == -999).sum().sum()
        rows_with_999 = (df == -999).any(axis=1).sum()
        
        print(f"  Has -999 values: {has_999}")
        if has_999:
            print(f"  ⚠️  PROBLEM: Found {count_999} instances of -999 in {rows_with_999} rows!")
            return False
        else:
            print(f"  ✓ No -999 values found")
        
        # Check for NaN
        has_nan = df.isna().any().any()
        count_nan = df.isna().sum().sum()
        print(f"  Has NaN values: {has_nan}")
        if has_nan:
            print(f"  ⚠️  WARNING: Found {count_nan} NaN values")
        else:
            print(f"  ✓ No NaN values found")
        
        # Show temperature statistics
        # Assuming last column is the target (water temperature forecast)
        target_col = df.columns[-1]
        if 'temperature' in target_col.lower() or 'water' in target_col.lower():
            target = df[target_col]
            print(f"\n  Target column: {target_col}")
            print(f"    Min: {target.min():.2f}°C")
            print(f"    Max: {target.max():.2f}°C")
            print(f"    Mean: {target.mean():.2f}°C")
            print(f"    Std: {target.std():.2f}°C")
            
            # Check for extreme values
            extreme_cold = target[target < -10]
            if len(extreme_cold) > 0:
                print(f"    ⚠️  Found {len(extreme_cold)} extreme cold values (< -10°C)")
                print(f"      Min extreme: {extreme_cold.min():.2f}°C")
            else:
                print(f"    ✓ No extreme cold values (< -10°C)")
        
        return True
        
    except FileNotFoundError:
        print(f"✗ File not found: {filename}")
        print("  This file should exist after running preparingData with the fixed code.")
        return False

# Check all cleaned files
results = {}
results['Training'] = check_file('debug_training_data_CLEANED.csv', 'training')
results['Testing'] = check_file('debug_testing_data_CLEANED.csv', 'testing')
results['Validation'] = check_file('debug_validation_data_CLEANED.csv', 'validation')

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if all(results.values()):
    print("✓ ALL CHECKS PASSED!")
    print("\nThe -999 sentinel values have been successfully removed.")
    print("Your model should no longer predict extreme temperatures.")
else:
    print("✗ SOME CHECKS FAILED")
    print("\nThere may still be issues with -999 handling.")
    print("Review the detailed output above for more information.")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
1. Run your training pipeline with the fixed code
2. Compare results between cycle and independent testing
3. Check that predictions are within normal temperature ranges
4. If issues persist, check the debug output from preparingData()
   Look for the "*** BEFORE DELETION ***" and "*** AFTER DELETION ***" sections
""")
