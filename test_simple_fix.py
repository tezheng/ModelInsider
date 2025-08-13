#!/usr/bin/env python3
"""
Test the fixed model_input_generator directly.
"""

from modelexport.core.model_input_generator import generate_dummy_inputs

def test_fixed_generator():
    print("Testing fixed generator...")
    
    try:
        inputs = generate_dummy_inputs(
            model_name_or_path="prajjwal1/bert-tiny",
            exporter="onnx"
        )
        print(f"✅ Success! Generated inputs: {list(inputs.keys())}")
        for name, tensor in inputs.items():
            print(f"  {name}: {tensor.shape}, {tensor.dtype}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_generator()