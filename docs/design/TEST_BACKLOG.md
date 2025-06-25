# Test Backlog - Cases to Discuss

## 1. test_find_parent_transformers_module
**Issue**: Specifically tests 'transformers' module functionality, violating CARDINAL RULE #1
**Location**: tests/test_param_mapping.py:152
**Action**: Skip - requires hardcoded transformers logic
**Alternative**: Use universal _find_parent_module method

## 2. Slow test timeouts
**Issue**: Tests taking >60s to load models
**Action**: Consider using smaller/mock models for faster testing


