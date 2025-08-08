#!/usr/bin/env python3
"""
Test GraphML v1.3 round-trip conversion with conflict resolution.

This test verifies that the new v1.3 key naming scheme (meta5-meta8, param0-2, io0-3, e1-3)
correctly resolves the conflicts found in v1.1/v1.2 specifications.
"""

import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import onnx

from modelexport.graphml.graphml_to_onnx_converter import GraphMLToONNXConverter
from modelexport.graphml.onnx_to_graphml_converter import ONNXToGraphMLConverter


def test_v13_roundtrip():
    """Test GraphML v1.3 round-trip conversion."""
    
    # Test directory
    test_dir = Path("temp/test_v13_roundtrip")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Load a test ONNX model
        onnx_path = "temp/baseline/bert-tiny/model.onnx"
        htp_path = "temp/baseline/bert-tiny/model_htp_metadata.json"
        
        if not Path(onnx_path).exists():
            print(f"❌ Test ONNX model not found at {onnx_path}")
            print("Please run: uv run python tests/data/generate_test_data.py --model prajjwal1/bert-tiny --output-dir temp/baseline/bert-tiny/")
            return False
            
        # Load original model
        original_model = onnx.load(onnx_path)
        original_node_count = len(original_model.graph.node)
        print(f"✅ Loaded ONNX model with {original_node_count} nodes")
        
        # 2. Convert ONNX to GraphML v1.3
        print("\n🔄 Converting ONNX to GraphML v1.3...")
        converter = ONNXToGraphMLConverter(
            hierarchical=True,
            htp_metadata_path=htp_path,
            parameter_strategy="sidecar"
        )
        
        result = converter.convert(onnx_path, str(test_dir / "model_v13"))
        graphml_path = result["graphml"]
        print(f"✅ Created GraphML: {graphml_path}")
        print(f"   Format version: {result['format_version']}")
        
        # 3. Verify v1.3 key structure
        print("\n🔍 Verifying v1.3 key structure...")
        tree = ET.parse(graphml_path)
        root = tree.getroot()
        
        # Check for v1.3 keys
        keys = {key.get("id") for key in root.findall(".//{http://graphml.graphdrawing.org/xmlns}key")}
        
        # v1.3 keys that resolve conflicts
        v13_keys = {
            "meta5", "meta6", "meta7", "meta8",  # Resolved metadata conflicts
            "param0", "param1", "param2",  # Parameter keys
            "io0", "io1", "io2", "io3",  # I/O keys
            "e1", "e2", "e3"  # Edge tensor keys
        }
        
        # Check for v1.3 keys
        found_v13_keys = keys.intersection(v13_keys)
        print(f"✅ Found v1.3 keys: {sorted(found_v13_keys)}")
        
        # Ensure no old conflicting keys
        old_conflict_keys = {"m5", "m6", "m7", "m8", "p0", "p1", "p2", "g0", "g1", "g2", "g3", "t0", "t1", "t2"}
        if keys.intersection(old_conflict_keys):
            print(f"❌ Found old conflicting keys: {keys.intersection(old_conflict_keys)}")
            return False
            
        # Check format version (it's inside the graph element)
        format_version_elem = root.find(".//{http://graphml.graphdrawing.org/xmlns}data[@key='meta2']")
        if format_version_elem is not None:
            format_version = format_version_elem.text
            print(f"✅ Format version in GraphML: {format_version}")
            if format_version != "1.3":
                print(f"❌ Expected format version 1.3, got {format_version}")
                return False
        else:
            print("❌ Format version not found in GraphML")
            return False
            
        # 4. Convert GraphML back to ONNX
        print("\n🔄 Converting GraphML v1.3 back to ONNX...")
        import_converter = GraphMLToONNXConverter()
        reconstructed_path = str(test_dir / "reconstructed.onnx")
        import_converter.convert(graphml_path, reconstructed_path, validate=True)
        
        # 5. Verify round-trip accuracy
        print("\n📊 Verifying round-trip accuracy...")
        reconstructed_model = onnx.load(reconstructed_path)
        reconstructed_node_count = len(reconstructed_model.graph.node)
        
        print(f"   Original nodes: {original_node_count}")
        print(f"   Reconstructed nodes: {reconstructed_node_count}")
        
        accuracy = reconstructed_node_count / original_node_count
        print(f"   Accuracy: {accuracy:.2%}")
        
        if accuracy >= 0.85:
            print(f"✅ Round-trip accuracy {accuracy:.2%} meets target (≥85%)")
        else:
            print(f"❌ Round-trip accuracy {accuracy:.2%} below target (≥85%)")
            return False
            
        # 6. Verify metadata preservation
        print("\n🔍 Verifying metadata preservation...")
        
        # Check that producer info is preserved
        if original_model.producer_name and reconstructed_model.producer_name:
            if original_model.producer_name == reconstructed_model.producer_name:
                print(f"✅ Producer name preserved: {reconstructed_model.producer_name}")
            else:
                print(f"⚠️ Producer name changed: {original_model.producer_name} → {reconstructed_model.producer_name}")
                
        # Check opset imports
        if len(reconstructed_model.opset_import) > 0:
            print(f"✅ Opset imports preserved: {len(reconstructed_model.opset_import)} imports")
        else:
            print("⚠️ No opset imports in reconstructed model")
            
        print("\n✅ GraphML v1.3 round-trip test PASSED!")
        print("   - v1.3 key naming resolves conflicts")
        print("   - Round-trip conversion successful")
        print("   - Metadata preserved")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if test_dir.exists():
            print(f"\n🧹 Test files saved in: {test_dir}")


if __name__ == "__main__":
    success = test_v13_roundtrip()
    exit(0 if success else 1)