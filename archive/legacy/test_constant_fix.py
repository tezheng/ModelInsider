#!/usr/bin/env python3
"""
Test script to verify constant handling fix
Simulates the constant extraction issue and fix
"""

import json
from pathlib import Path


def analyze_onnx_constants():
    """Analyze constant handling in latest test results"""
    
    # Check if test results exist
    test_dir = Path("temp/bert_self_attention_test")
    if not test_dir.exists():
        print("❌ Test directory not found. Run BERT test first.")
        return
    
    # Load comparison results
    try:
        with open(test_dir / "detailed_comparison/detailed_comparison_report.json") as f:
            comparison = json.load(f)
        
        print("=== CONSTANT ANALYSIS ===")
        print("This analysis shows the difference in constant handling:")
        print()
        
        # Extract constant counts
        standalone_constants = comparison["graph_comparison"]["node_type_differences"]["Constant"]["standalone"]
        extracted_constants = comparison["graph_comparison"]["node_type_differences"]["Constant"]["extracted"]
        difference = comparison["graph_comparison"]["node_type_differences"]["Constant"]["difference"]
        
        print(f"Standalone model constants: {standalone_constants}")
        print(f"Extracted model constants:  {extracted_constants}")  
        print(f"Difference:                 {difference}")
        print()
        
        if extracted_constants == 0 and standalone_constants > 0:
            print("⚠️  ANALYSIS: Pre-fix state detected")
            print("   - Standalone: Has explicit Constant nodes (normal)")
            print("   - Extracted:  Constants as external inputs (needs fix)")
            print("   - Action:     Re-run test after applying constant handling fix")
        elif extracted_constants == standalone_constants:
            print("✅ ANALYSIS: Constants correctly fixed!")
            print("   - Both models have same number of Constant nodes")
            print("   - Fix successful: Constant nodes now included in extracted model")
            print("   - Result: Models have identical constant structure")
        else:
            print("⚠️  ANALYSIS: Unexpected constant difference")
            print("   - May need further investigation")
            
        print()
        print("=== CONTEXT ===")
        print("Why this matters:")
        print("- Standalone ONNX: Constants appear as explicit nodes with embedded values")
        print("- Before fix: Extracted ONNX had constants as external inputs (wrong)")
        print("- After fix: Extracted ONNX should match standalone structure")
        
    except FileNotFoundError as e:
        print(f"❌ Could not load comparison results: {e}")
        print("Run the BERT self-attention test first to generate comparison data.")
    except KeyError as e:
        print(f"❌ Unexpected comparison format: {e}")

if __name__ == "__main__":
    analyze_onnx_constants()