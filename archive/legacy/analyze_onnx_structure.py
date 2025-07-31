#!/usr/bin/env python3
"""
Analyze ONNX model structure to understand constants
"""

import json
from pathlib import Path


def analyze_model_structure():
    """Analyze the structure of both models"""
    
    test_dir = Path("temp/bert_self_attention_test")
    
    # Check if models exist
    standalone_path = test_dir / "bert_self_attention_standalone.onnx"
    extracted_path = test_dir / "bert_self_attention_extracted.onnx"
    
    if not standalone_path.exists():
        print(f"❌ Standalone model not found: {standalone_path}")
        return
        
    if not extracted_path.exists():
        print(f"❌ Extracted model not found: {extracted_path}")
        return
    
    print("=== ONNX MODEL STRUCTURE ANALYSIS ===")
    print()
    
    # Try to read with simple analysis
    try:
        # Read first few lines to look for differences
        print("=== STANDALONE MODEL (first 50 lines) ===")
        with open(standalone_path, 'rb') as f:
            content = f.read(2000)  # Read first 2KB
            # Look for text patterns that might indicate structure
            if b'Constant' in content:
                print("✅ Contains 'Constant' nodes")
            if b'initializer' in content:
                print("✅ Contains 'initializer' section")
        
        print("\n=== EXTRACTED MODEL (first 50 lines) ===") 
        with open(extracted_path, 'rb') as f:
            content = f.read(2000)  # Read first 2KB
            if b'Constant' in content:
                print("✅ Contains 'Constant' nodes")
            else:
                print("❌ No 'Constant' nodes found in first 2KB")
            if b'initializer' in content:
                print("✅ Contains 'initializer' section")
            else:
                print("❌ No 'initializer' section found in first 2KB")
                
    except Exception as e:
        print(f"❌ Error reading models: {e}")
        return
    
    print("\n=== COMPARISON SUMMARY ===")
    
    # Load detailed comparison if available
    comparison_file = test_dir / "detailed_comparison/detailed_comparison_report.json"
    if comparison_file.exists():
        try:
            with open(comparison_file) as f:
                comp = json.load(f)
            
            # Show key differences
            node_diff = comp["graph_comparison"]["node_count"]
            print(f"Node count - Standalone: {node_diff['standalone']}, Extracted: {node_diff['extracted']}")
            
            type_diffs = comp["graph_comparison"]["node_type_differences"]
            for op_type, diff_data in type_diffs.items():
                if diff_data["difference"] != 0:
                    print(f"{op_type} - Standalone: {diff_data['standalone']}, Extracted: {diff_data['extracted']} (Δ{diff_data['difference']})")
            
        except Exception as e:
            print(f"Could not load detailed comparison: {e}")
    
    print("\n=== WHAT THIS MEANS ===")
    print("If you see constants as INPUTS in extracted model but WEIGHTS in standalone:")
    print("- Standalone: Constant nodes with embedded values (correct)")
    print("- Extracted: Constants become external inputs (incorrect)")
    print("- Our fix: Should add Constant nodes to extracted model to match standalone")
    print("\nTo fully verify, open both models in Netron and compare:")
    print(f"1. {standalone_path}")
    print(f"2. {extracted_path}")

if __name__ == "__main__":
    analyze_model_structure()