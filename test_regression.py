#!/usr/bin/env python3
"""
Simple test script for git bisect to identify the regression.
Returns 0 if the test passes, 1 if it fails.
"""

import sys
import os

def test_bert_export():
    """Test BERT model export functionality."""
    try:
        # Add the project root to the path
        sys.path.insert(0, os.getcwd())
        
        # Try both the new and old API
        try:
            from modelexport.core.model_input_generator import generate_dummy_inputs
            # New API
            inputs = generate_dummy_inputs(
                model_name_or_path="prajjwal1/bert-tiny",
                exporter="onnx"
            )
        except ImportError:
            # Old API
            from modelexport.core.model_input_generator import generate_dummy_inputs_from_model_path
            inputs = generate_dummy_inputs_from_model_path(
                model_name_or_path="prajjwal1/bert-tiny",
                exporter="onnx"
            )
        
        # Verify we got the expected inputs
        expected_keys = {"input_ids", "attention_mask"}
        if not expected_keys.issubset(inputs.keys()):
            print(f"❌ Missing expected keys. Got: {list(inputs.keys())}")
            return 1
            
        print(f"✅ SUCCESS: Generated inputs: {list(inputs.keys())}")
        return 0
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return 1

if __name__ == "__main__":
    exit_code = test_bert_export()
    sys.exit(exit_code)