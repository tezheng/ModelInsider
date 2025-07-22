#!/usr/bin/env python3
"""
Test iteration 3 - Compare metadata and report outputs with baseline
"""

import sys
import json
import difflib
from pathlib import Path
from datetime import datetime

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.export_monitor.export_monitor_v2 import HTPExportMonitor, HTPExportStep

def test_metadata_generation():
    """Test metadata generation with v2 monitor."""
    print("🧪 Testing Metadata Generation")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("/home/zhengte/modelexport_allmodels/experiments/export_monitor/iterations/iteration_003")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create monitor
    monitor = HTPExportMonitor(
        output_path=str(output_dir / "model.onnx"),
        model_name="prajjwal1/bert-tiny",
        verbose=False,  # Quiet for metadata test
        enable_report=True
    )
    
    # Simulate export with full data
    with monitor:
        # Step 1: Model preparation
        monitor.update(HTPExportStep.MODEL_PREP,
            model_class="BertModel",
            total_modules=48,
            total_parameters=4385920,
            embed_hierarchy_attributes=True
        )
        
        # Step 2: Input generation
        monitor.update(HTPExportStep.INPUT_GEN,
            model_type="bert",
            task="feature-extraction",
            method="auto_generated",
            inputs={
                "input_ids": {"shape": [2, 16], "dtype": "torch.int64"},
                "attention_mask": {"shape": [2, 16], "dtype": "torch.int64"},
                "token_type_ids": {"shape": [2, 16], "dtype": "torch.int64"}
            }
        )
        
        # Step 3: Hierarchy building
        hierarchy = {
            "": {
                "class_name": "BertModel",
                "traced_tag": "/BertModel",
                "module_type": "huggingface",
                "execution_order": 0
            },
            "embeddings": {
                "class_name": "BertEmbeddings",
                "traced_tag": "/BertModel/BertEmbeddings",
                "module_type": "huggingface",
                "execution_order": 1
            },
            "encoder": {
                "class_name": "BertEncoder",
                "traced_tag": "/BertModel/BertEncoder",
                "module_type": "huggingface",
                "execution_order": 2
            },
            "pooler": {
                "class_name": "BertPooler",
                "traced_tag": "/BertModel/BertPooler",
                "module_type": "huggingface",
                "execution_order": 17
            }
        }
        
        # Add all encoder layers
        for i in range(2):
            layer_prefix = f"encoder.layer.{i}"
            hierarchy.update({
                layer_prefix: {
                    "class_name": "BertLayer",
                    "traced_tag": f"/BertModel/BertEncoder/BertLayer.{i}",
                    "module_type": "huggingface",
                    "execution_order": 3 + i*7
                },
                f"{layer_prefix}.attention": {
                    "class_name": "BertAttention",
                    "traced_tag": f"/BertModel/BertEncoder/BertLayer.{i}/BertAttention",
                    "module_type": "huggingface",
                    "execution_order": 4 + i*7
                },
                f"{layer_prefix}.attention.self": {
                    "class_name": "BertSdpaSelfAttention",
                    "traced_tag": f"/BertModel/BertEncoder/BertLayer.{i}/BertAttention/BertSdpaSelfAttention",
                    "module_type": "huggingface",
                    "execution_order": 5 + i*7
                },
                f"{layer_prefix}.attention.output": {
                    "class_name": "BertSelfOutput",
                    "traced_tag": f"/BertModel/BertEncoder/BertLayer.{i}/BertAttention/BertSelfOutput",
                    "module_type": "huggingface",
                    "execution_order": 6 + i*7
                },
                f"{layer_prefix}.intermediate": {
                    "class_name": "BertIntermediate",
                    "traced_tag": f"/BertModel/BertEncoder/BertLayer.{i}/BertIntermediate",
                    "module_type": "huggingface",
                    "execution_order": 7 + i*7
                },
                f"{layer_prefix}.intermediate.intermediate_act_fn": {
                    "class_name": "GELUActivation",
                    "traced_tag": f"/BertModel/BertEncoder/BertLayer.{i}/BertIntermediate/GELUActivation",
                    "module_type": "huggingface",
                    "execution_order": 8 + i*7
                },
                f"{layer_prefix}.output": {
                    "class_name": "BertOutput",
                    "traced_tag": f"/BertModel/BertEncoder/BertLayer.{i}/BertOutput",
                    "module_type": "huggingface",
                    "execution_order": 9 + i*7
                }
            })
        
        monitor.update(HTPExportStep.HIERARCHY,
            hierarchy=hierarchy,
            execution_steps=36
        )
        
        # Step 4-5: ONNX export and tagger
        monitor.update(HTPExportStep.ONNX_EXPORT,
            opset_version=17,
            do_constant_folding=True
        )
        
        monitor.update(HTPExportStep.TAGGER_CREATION,
            enable_operation_fallback=False
        )
        
        # Step 6: Node tagging with full node list
        tagged_nodes = {}
        
        # Add all 136 nodes from baseline
        # Root nodes (19)
        root_nodes = [
            "/Cast", "/Cast_1", "/Cast_2", "/Constant", "/ConstantOfShape",
            "/Constant_1", "/Constant_2", "/Constant_3", "/Constant_4",
            "/Constant_5", "/Constant_6", "/Equal", "/Expand", "/Mul",
            "/Sub", "/Unsqueeze", "/Unsqueeze_1", "/Where", "/Where_1"
        ]
        for node in root_nodes:
            tagged_nodes[node] = "/BertModel"
        
        # Embeddings nodes (8)
        embeddings_nodes = [
            "/embeddings/Add", "/embeddings/Add_1", "/embeddings/Constant",
            "/embeddings/Constant_1", "/embeddings/LayerNorm/LayerNormalization",
            "/embeddings/position_embeddings/Gather", "/embeddings/token_type_embeddings/Gather",
            "/embeddings/word_embeddings/Gather"
        ]
        for node in embeddings_nodes:
            tagged_nodes[node] = "/BertModel/BertEmbeddings"
        
        # Add layer-specific nodes
        for i in range(2):
            # Self-attention nodes (35 each)
            base_tag = f"/BertModel/BertEncoder/BertLayer.{i}/BertAttention/BertSdpaSelfAttention"
            attention_nodes = [
                f"/encoder/layer.{i}/attention/self/Add",
                f"/encoder/layer.{i}/attention/self/Cast",
                f"/encoder/layer.{i}/attention/self/Cast_1",
                f"/encoder/layer.{i}/attention/self/Div",
                f"/encoder/layer.{i}/attention/self/MatMul",
                f"/encoder/layer.{i}/attention/self/MatMul_1",
                f"/encoder/layer.{i}/attention/self/Mul",
                f"/encoder/layer.{i}/attention/self/Mul_1",
                f"/encoder/layer.{i}/attention/self/Reshape",
                f"/encoder/layer.{i}/attention/self/Reshape_1",
                f"/encoder/layer.{i}/attention/self/Reshape_2",
                f"/encoder/layer.{i}/attention/self/Reshape_3",
                f"/encoder/layer.{i}/attention/self/Shape",
                f"/encoder/layer.{i}/attention/self/Slice",
                f"/encoder/layer.{i}/attention/self/Softmax",
                f"/encoder/layer.{i}/attention/self/Sqrt",
                f"/encoder/layer.{i}/attention/self/Sqrt_1",
                f"/encoder/layer.{i}/attention/self/Sqrt_2",
                f"/encoder/layer.{i}/attention/self/Transpose",
                f"/encoder/layer.{i}/attention/self/Transpose_1",
                f"/encoder/layer.{i}/attention/self/Transpose_2",
                f"/encoder/layer.{i}/attention/self/Transpose_3",
                f"/encoder/layer.{i}/attention/self/key/Add",
                f"/encoder/layer.{i}/attention/self/key/MatMul",
                f"/encoder/layer.{i}/attention/self/query/Add",
                f"/encoder/layer.{i}/attention/self/query/MatMul",
                f"/encoder/layer.{i}/attention/self/value/Add",
                f"/encoder/layer.{i}/attention/self/value/MatMul"
            ]
            # Add constants
            for j in range(7):
                attention_nodes.append(f"/encoder/layer.{i}/attention/self/Constant_{j}")
            
            for node in attention_nodes:
                tagged_nodes[node] = base_tag
            
            # Self output nodes (4 each)
            output_tag = f"/BertModel/BertEncoder/BertLayer.{i}/BertAttention/BertSelfOutput"
            output_nodes = [
                f"/encoder/layer.{i}/attention/output/Add",
                f"/encoder/layer.{i}/attention/output/LayerNorm/LayerNormalization",
                f"/encoder/layer.{i}/attention/output/dense/Add",
                f"/encoder/layer.{i}/attention/output/dense/MatMul"
            ]
            for node in output_nodes:
                tagged_nodes[node] = output_tag
            
            # Intermediate nodes (2 each)
            intermediate_tag = f"/BertModel/BertEncoder/BertLayer.{i}/BertIntermediate"
            intermediate_nodes = [
                f"/encoder/layer.{i}/intermediate/dense/Add",
                f"/encoder/layer.{i}/intermediate/dense/MatMul"
            ]
            for node in intermediate_nodes:
                tagged_nodes[node] = intermediate_tag
            
            # GELU nodes (8 each)
            gelu_tag = f"/BertModel/BertEncoder/BertLayer.{i}/BertIntermediate/GELUActivation"
            gelu_nodes = [
                f"/encoder/layer.{i}/intermediate/intermediate_act_fn/Add",
                f"/encoder/layer.{i}/intermediate/intermediate_act_fn/Constant",
                f"/encoder/layer.{i}/intermediate/intermediate_act_fn/Constant_1",
                f"/encoder/layer.{i}/intermediate/intermediate_act_fn/Constant_2",
                f"/encoder/layer.{i}/intermediate/intermediate_act_fn/Div",
                f"/encoder/layer.{i}/intermediate/intermediate_act_fn/Erf",
                f"/encoder/layer.{i}/intermediate/intermediate_act_fn/Mul",
                f"/encoder/layer.{i}/intermediate/intermediate_act_fn/Mul_1"
            ]
            for node in gelu_nodes:
                tagged_nodes[node] = gelu_tag
            
            # Output nodes (4 each)
            bert_output_tag = f"/BertModel/BertEncoder/BertLayer.{i}/BertOutput"
            bert_output_nodes = [
                f"/encoder/layer.{i}/output/Add",
                f"/encoder/layer.{i}/output/LayerNorm/LayerNormalization",
                f"/encoder/layer.{i}/output/dense/Add",
                f"/encoder/layer.{i}/output/dense/MatMul"
            ]
            for node in bert_output_nodes:
                tagged_nodes[node] = bert_output_tag
        
        # Pooler nodes (3)
        pooler_nodes = [
            "/pooler/Gather",
            "/pooler/activation/Tanh",
            "/pooler/dense/Gemm"
        ]
        for node in pooler_nodes:
            tagged_nodes[node] = "/BertModel/BertPooler"
        
        monitor.update(HTPExportStep.NODE_TAGGING,
            total_nodes=136,
            tagged_nodes=tagged_nodes,
            statistics={
                "direct_matches": 83,
                "parent_matches": 34,
                "operation_matches": 0,
                "root_fallbacks": 19,
                "empty_tags": 0
            }
        )
        
        # Step 7-8: Tag injection and metadata
        monitor.update(HTPExportStep.TAG_INJECTION,
            tags_injected=True
        )
        
        # Create dummy ONNX file for size calculation
        onnx_path = output_dir / "model.onnx"
        with open(onnx_path, 'wb') as f:
            f.write(b'dummy' * 3515927)  # ~16.76MB
        
        monitor.update(HTPExportStep.METADATA_GEN)
    
    return output_dir

def compare_metadata():
    """Compare v2 metadata with baseline."""
    print("\n📊 Comparing Metadata Files")
    print("=" * 60)
    
    # Read baseline metadata
    baseline_path = Path("/home/zhengte/modelexport_allmodels/temp/baseline/model_htp_metadata.json")
    with open(baseline_path) as f:
        baseline = json.load(f)
    
    # Read v2 metadata
    v2_path = Path("/home/zhengte/modelexport_allmodels/experiments/export_monitor/iterations/iteration_003/model_htp_metadata.json")
    with open(v2_path) as f:
        v2 = json.load(f)
    
    # Compare structure
    print("\nStructure comparison:")
    print(f"Baseline keys: {sorted(baseline.keys())}")
    print(f"V2 keys: {sorted(v2.keys())}")
    
    # Check each section
    differences = []
    
    for key in baseline.keys():
        if key not in v2:
            differences.append(f"Missing key: {key}")
        elif key == "export_info":
            # Skip timestamp comparison
            for subkey in baseline[key]:
                if subkey != "timestamp" and baseline[key][subkey] != v2[key].get(subkey):
                    differences.append(f"{key}.{subkey}: {baseline[key][subkey]} != {v2[key].get(subkey)}")
        elif key == "file_info":
            # Skip paths comparison
            if baseline[key].get("onnx_size_mb") and v2[key].get("onnx_size_mb"):
                size_diff = abs(baseline[key]["onnx_size_mb"] - v2[key]["onnx_size_mb"])
                if size_diff > 0.1:
                    differences.append(f"ONNX size difference: {size_diff:.2f}MB")
    
    # Check node count
    baseline_nodes = len(baseline.get("nodes", {}))
    v2_nodes = len(v2.get("nodes", {}))
    
    print(f"\nNode count: Baseline={baseline_nodes}, V2={v2_nodes}")
    
    if differences:
        print("\n⚠️ Differences found:")
        for diff in differences[:10]:
            print(f"  - {diff}")
    else:
        print("\n✅ Metadata structure matches baseline!")
    
    # Save formatted metadata for inspection
    with open(v2_path.with_suffix('.pretty.json'), 'w') as f:
        json.dump(v2, f, indent=2)
    
    return len(differences) == 0

def compare_reports():
    """Compare v2 report with baseline."""
    print("\n📄 Comparing Report Files")
    print("=" * 60)
    
    # Read baseline report
    baseline_path = Path("/home/zhengte/modelexport_allmodels/temp/baseline/model_full_report.txt")
    with open(baseline_path) as f:
        baseline_lines = f.readlines()
    
    # Read v2 report
    v2_path = Path("/home/zhengte/modelexport_allmodels/experiments/export_monitor/iterations/iteration_003/model_full_report.txt")
    if not v2_path.exists():
        print("❌ V2 report not found!")
        return False
        
    with open(v2_path) as f:
        v2_lines = f.readlines()
    
    print(f"Baseline lines: {len(baseline_lines)}")
    print(f"V2 lines: {len(v2_lines)}")
    
    # Compare structure
    diff = list(difflib.unified_diff(
        baseline_lines[:50],  # Compare headers
        v2_lines[:50],
        fromfile='baseline',
        tofile='v2',
        lineterm=''
    ))
    
    if diff:
        print("\nHeader differences:")
        for line in diff[:20]:
            print(line)
    
    # Check key sections
    sections = [
        "HTP EXPORT FULL REPORT",
        "MODEL INFORMATION",
        "INPUT GENERATION",
        "COMPLETE MODULE HIERARCHY",
        "NODE TAGGING STATISTICS",
        "COMPLETE NODE MAPPINGS",
        "EXPORT SUMMARY"
    ]
    
    for section in sections:
        baseline_has = any(section in line for line in baseline_lines)
        v2_has = any(section in line for line in v2_lines)
        
        if baseline_has and v2_has:
            print(f"✓ {section}")
        else:
            print(f"✗ {section} - Baseline: {baseline_has}, V2: {v2_has}")
    
    return len(diff) < 10  # Allow small differences

def main():
    """Run iteration 3 tests."""
    print("🔧 ITERATION 3 - Metadata and Report Testing")
    print("=" * 60)
    
    # Test metadata generation
    output_dir = test_metadata_generation()
    print(f"\n✅ Generated files in: {output_dir}")
    
    # Compare outputs
    metadata_ok = compare_metadata()
    report_ok = compare_reports()
    
    print("\n📊 ITERATION 3 RESULTS:")
    print(f"  - Metadata comparison: {'✅ PASS' if metadata_ok else '❌ FAIL'}")
    print(f"  - Report comparison: {'✅ PASS' if report_ok else '❌ FAIL'}")
    
    if metadata_ok and report_ok:
        print("\n🎉 Iteration 3 completed successfully!")
    else:
        print("\n⚠️ Iteration 3 needs fixes")

if __name__ == "__main__":
    main()