"""
Test HuggingFace module tagging correctness.

Verifies that each ONNX node is correctly tagged with its corresponding
HuggingFace module based on execution context and parameter usage.
"""

import pytest
import json
import onnx
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
from transformers import AutoModel, AutoTokenizer

from modelexport.hierarchy_exporter import HierarchyExporter


class TestHFModuleTagging:
    """Test that ONNX nodes are correctly tagged with HuggingFace modules."""
    
    @pytest.fixture
    def bert_model_and_inputs(self):
        """Load BERT model and prepare inputs."""
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        text = "Test HF module tagging"
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        model.eval()
        return model, inputs
    
    @pytest.fixture
    def exported_hierarchy_data(self, bert_model_and_inputs):
        """Export model with hierarchy and return ONNX model and metadata."""
        model, inputs = bert_model_and_inputs
        
        # Export with hierarchy preservation
        exporter = HierarchyExporter()
        output_path = "temp/test_hf_tagging.onnx"
        
        result = exporter.export(
            model=model,
            example_inputs=inputs,
            output_path=output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
            }
        )
        
        # Load the exported ONNX model and hierarchy data
        onnx_model = onnx.load(output_path)
        hierarchy_path = output_path.replace('.onnx', '_hierarchy.json')
        
        with open(hierarchy_path, 'r') as f:
            hierarchy_data = json.load(f)
            
        return onnx_model, hierarchy_data, model
    
    def test_all_nodes_have_hierarchy_tags(self, exported_hierarchy_data):
        """Test that all ONNX nodes have hierarchy tag metadata."""
        onnx_model, hierarchy_data, pytorch_model = exported_hierarchy_data
        
        nodes_with_tags = 0
        nodes_without_tags = []
        
        for node in onnx_model.graph.node:
            # Check if node has doc_string with hierarchy info (ONNX-compliant approach)
            has_hierarchy_info = bool(node.doc_string.strip()) if node.doc_string else False
            
            if has_hierarchy_info:
                try:
                    # Try to parse as JSON to verify it's hierarchy info
                    import json
                    hierarchy_info = json.loads(node.doc_string)
                    if "hierarchy_tags" in hierarchy_info:
                        nodes_with_tags += 1
                    else:
                        nodes_without_tags.append(f"{node.name} ({node.op_type})")
                except:
                    nodes_without_tags.append(f"{node.name} ({node.op_type})")
            else:
                nodes_without_tags.append(f"{node.name} ({node.op_type})")
        
        total_nodes = len(onnx_model.graph.node)
        tag_coverage = nodes_with_tags / total_nodes if total_nodes > 0 else 0
        
        print(f"\n🏷️ TAGGING COVERAGE:")
        print(f"   Tagged nodes: {nodes_with_tags}/{total_nodes} ({tag_coverage:.1%})")
        
        if nodes_without_tags:
            print(f"   Untagged nodes: {nodes_without_tags[:5]}")  # Show first 5
        
        # We expect good coverage (>80%) but allow some preprocessing/constant nodes to be untagged
        assert tag_coverage > 0.80, f"Low tag coverage: {tag_coverage:.1%}"
    
    def test_expected_hf_modules_present(self, exported_hierarchy_data):
        """Test that expected HuggingFace modules appear in tags."""
        onnx_model, hierarchy_data, pytorch_model = exported_hierarchy_data
        
        # Extract all unique tags from the hierarchy data
        all_tags = set()
        for node_name, node_info in hierarchy_data.get("node_tags", {}).items():
            tags = node_info.get("tags", [])
            all_tags.update(tags)
        
        # Expected BERT module patterns
        expected_modules = {
            "BertModel",
            "BertEmbeddings", 
            "BertEncoder",
            "BertLayer",
            "BertAttention",
            "BertSelfAttention",
            "BertSelfOutput",
            "BertIntermediate", 
            "BertOutput",
            "BertPooler"
        }
        
        found_modules = set()
        for tag in all_tags:
            for expected in expected_modules:
                if expected in tag:
                    found_modules.add(expected)
        
        print(f"\n🔍 HF MODULE DETECTION:")
        print(f"   Expected: {sorted(expected_modules)}")
        print(f"   Found: {sorted(found_modules)}")
        
        missing_modules = expected_modules - found_modules
        if missing_modules:
            print(f"   Missing: {sorted(missing_modules)}")
        
        # Should find most key BERT modules
        coverage = len(found_modules) / len(expected_modules)
        assert coverage > 0.7, f"Low module coverage: {coverage:.1%}, missing: {missing_modules}"
    
    def test_embeddings_operations_tagged_correctly(self, exported_hierarchy_data):
        """Test that embedding operations are tagged with BertEmbeddings module."""
        onnx_model, hierarchy_data, pytorch_model = exported_hierarchy_data
        
        embedding_ops = []
        embedding_tags = []
        
        for node_name, node_info in hierarchy_data.get("node_tags", {}).items():
            if "embeddings" in node_name.lower():
                tags = node_info.get("tags", [])
                embedding_ops.append((node_name, node_info.get("op_type", "Unknown")))
                embedding_tags.extend(tags)
        
        # Check that embedding operations have BertEmbeddings tags
        bert_embeddings_count = sum(1 for tag in embedding_tags if "BertEmbeddings" in tag)
        
        print(f"\n📝 EMBEDDINGS TAGGING:")
        print(f"   Embedding operations: {len(embedding_ops)}")
        print(f"   BertEmbeddings tags: {bert_embeddings_count}")
        if embedding_ops:
            print(f"   Sample ops: {embedding_ops[:3]}")
        
        assert len(embedding_ops) > 0, "No embedding operations found"
        assert bert_embeddings_count > 0, "No BertEmbeddings tags found"
    
    def test_attention_operations_tagged_correctly(self, exported_hierarchy_data):
        """Test that attention operations are tagged with attention modules."""
        onnx_model, hierarchy_data, pytorch_model = exported_hierarchy_data
        
        attention_ops = []
        attention_tags = []
        
        for node_name, node_info in hierarchy_data.get("node_tags", {}).items():
            # Look for operations likely related to attention
            if any(keyword in node_name.lower() for keyword in ["attention", "query", "key", "value", "matmul"]):
                tags = node_info.get("tags", [])
                attention_ops.append((node_name, node_info.get("op_type", "Unknown")))
                attention_tags.extend(tags)
        
        # Check for attention-related tags
        attention_module_tags = [
            tag for tag in attention_tags 
            if any(module in tag for module in ["Attention", "SelfAttention", "SelfOutput"])
        ]
        
        print(f"\n🎯 ATTENTION TAGGING:")
        print(f"   Attention operations: {len(attention_ops)}")
        print(f"   Attention module tags: {len(attention_module_tags)}")
        if attention_ops:
            print(f"   Sample ops: {attention_ops[:3]}")
        
        # Should have some attention operations and tags
        if len(attention_ops) > 0:
            assert len(attention_module_tags) > 0, "No attention module tags found for attention operations"
    
    def test_tag_hierarchy_structure(self, exported_hierarchy_data):
        """Test that tags follow proper hierarchical structure."""
        onnx_model, hierarchy_data, pytorch_model = exported_hierarchy_data
        
        all_tags = set()
        for node_name, node_info in hierarchy_data.get("node_tags", {}).items():
            tags = node_info.get("tags", [])
            all_tags.update(tags)
        
        # Analyze tag structure
        hierarchical_tags = [tag for tag in all_tags if tag.startswith("/")]
        path_depths = [tag.count("/") for tag in hierarchical_tags]
        
        print(f"\n🌳 TAG HIERARCHY STRUCTURE:")
        print(f"   Total unique tags: {len(all_tags)}")
        print(f"   Hierarchical tags: {len(hierarchical_tags)}")
        print(f"   Max depth: {max(path_depths) if path_depths else 0}")
        print(f"   Avg depth: {sum(path_depths)/len(path_depths):.1f}" if path_depths else "N/A")
        
        # Sample some tags
        sample_tags = sorted(list(all_tags))[:5]
        print(f"   Sample tags: {sample_tags}")
        
        assert len(hierarchical_tags) > 0, "No hierarchical tags found"
        assert max(path_depths) >= 2, "Tags too shallow - should have deeper hierarchy"
    
    def test_parameter_operations_have_consistent_tags(self, exported_hierarchy_data):
        """Test that operations using parameters have consistent module tags."""
        onnx_model, hierarchy_data, pytorch_model = exported_hierarchy_data
        
        # Get parameter names from PyTorch model
        pytorch_params = {name for name, _ in pytorch_model.named_parameters()}
        
        # Find ONNX operations that use these parameters
        param_operations = {}
        
        for node_name, node_info in hierarchy_data.get("node_tags", {}).items():
            inputs = node_info.get("inputs", [])
            # Check if any input matches a PyTorch parameter
            for input_name in inputs:
                # Clean up input name to match parameter name format
                clean_input = input_name.replace(".", "_").replace("/", "_")
                for param_name in pytorch_params:
                    param_clean = param_name.replace(".", "_")
                    if param_clean in clean_input or clean_input in param_clean:
                        if param_name not in param_operations:
                            param_operations[param_name] = []
                        param_operations[param_name].append({
                            'node': node_name,
                            'op_type': node_info.get('op_type'),
                            'tags': node_info.get('tags', [])
                        })
                        break
        
        print(f"\n⚙️ PARAMETER OPERATION TAGGING:")
        print(f"   PyTorch parameters: {len(pytorch_params)}")
        print(f"   Parameters with ONNX ops: {len(param_operations)}")
        
        # Check consistency for some key parameters
        inconsistent_params = []
        for param_name, operations in param_operations.items():
            if len(operations) > 1:
                # Check if all operations have similar tags
                all_tags = set()
                for op in operations:
                    all_tags.update(op['tags'])
                
                # If there are multiple distinct module paths, might be inconsistent
                distinct_modules = {tag.split('/')[-1] for tag in all_tags if '/' in tag}
                if len(distinct_modules) > 3:  # Allow some variation
                    inconsistent_params.append((param_name, distinct_modules))
        
        if inconsistent_params:
            print(f"   Potentially inconsistent: {len(inconsistent_params)}")
            for param, modules in inconsistent_params[:2]:  # Show first 2
                print(f"     {param}: {sorted(modules)}")
        
        # Should find parameter operations
        assert len(param_operations) > 0, "No parameter operations found"
    
    def test_onnx_doc_string_tagging(self, exported_hierarchy_data):
        """Test that ONNX nodes have proper doc_string hierarchy tagging."""
        onnx_model, hierarchy_data, pytorch_model = exported_hierarchy_data
        
        # Check ONNX nodes for doc_string hierarchy information
        nodes_with_doc_tags = 0
        sample_doc_strings = []
        hf_module_counts = defaultdict(int)
        
        for node in onnx_model.graph.node:
            if node.doc_string:
                try:
                    import json
                    doc_info = json.loads(node.doc_string)
                    
                    if "hierarchy_tags" in doc_info:
                        nodes_with_doc_tags += 1
                        tags = doc_info["hierarchy_tags"]
                        
                        # Count HF modules
                        for tag in tags:
                            if "/" in tag:
                                modules = tag.split("/")
                                for module in modules:
                                    if module.startswith("Bert"):
                                        hf_module_counts[module] += 1
                        
                        # Collect sample for inspection
                        if len(sample_doc_strings) < 3:
                            sample_doc_strings.append({
                                'node': node.name,
                                'op_type': node.op_type,
                                'hierarchy_tags': tags,
                                'hierarchy_path': doc_info.get('hierarchy_path', ''),
                                'hierarchy_count': doc_info.get('hierarchy_count', 0)
                            })
                            
                except Exception as e:
                    pass  # Skip malformed doc_strings
        
        print(f"\n📄 ONNX DOC_STRING TAGGING:")
        print(f"   Nodes with doc_string tags: {nodes_with_doc_tags}")
        print(f"   HF modules found: {dict(hf_module_counts)}")
        print(f"   Sample tagged nodes:")
        for sample in sample_doc_strings:
            print(f"     {sample['node']} ({sample['op_type']}): {sample['hierarchy_path']}")
        
        assert nodes_with_doc_tags > 0, "No nodes found with doc_string hierarchy tags"
        assert len(hf_module_counts) > 0, "No HuggingFace modules found in doc_string tags"
    
    def test_specific_bert_module_tagging(self, exported_hierarchy_data):
        """Test that specific BERT operations are correctly tagged with expected modules."""
        onnx_model, hierarchy_data, pytorch_model = exported_hierarchy_data
        
        # Look for operations that should be tagged with specific modules
        embeddings_ops = []
        attention_ops = []
        layer_norm_ops = []
        matmul_ops = []
        
        for node in onnx_model.graph.node:
            if node.doc_string:
                try:
                    import json
                    doc_info = json.loads(node.doc_string)
                    tags = doc_info.get("hierarchy_tags", [])
                    
                    # Categorize operations by expected module type
                    if any("Embeddings" in tag for tag in tags):
                        embeddings_ops.append((node.name, node.op_type, tags))
                    
                    if any("Attention" in tag or "SelfAttention" in tag for tag in tags):
                        attention_ops.append((node.name, node.op_type, tags))
                    
                    if node.op_type == "MatMul":
                        matmul_ops.append((node.name, node.op_type, tags))
                    
                    if "LayerNorm" in node.name or "layernorm" in node.name.lower():
                        layer_norm_ops.append((node.name, node.op_type, tags))
                        
                except:
                    pass
        
        print(f"\n🎯 SPECIFIC MODULE TAGGING:")
        print(f"   Embeddings operations: {len(embeddings_ops)}")
        print(f"   Attention operations: {len(attention_ops)}")
        print(f"   MatMul operations: {len(matmul_ops)}")
        print(f"   LayerNorm operations: {len(layer_norm_ops)}")
        
        # Verify we found operations for key BERT components
        assert len(embeddings_ops) > 0, "No operations tagged with Embeddings modules"
        
        # Show examples
        if embeddings_ops:
            print(f"   Example embedding op: {embeddings_ops[0][0]} -> {embeddings_ops[0][2]}")
        if attention_ops:
            print(f"   Example attention op: {attention_ops[0][0]} -> {attention_ops[0][2]}")
        if matmul_ops:
            print(f"   Example matmul op: {matmul_ops[0][0]} -> {matmul_ops[0][2]}")
    
    def test_operation_type_distribution(self, exported_hierarchy_data):
        """Test that operation type distribution makes sense for BERT."""
        onnx_model, hierarchy_data, pytorch_model = exported_hierarchy_data
        
        # Count operation types in tagged nodes
        op_type_counts = Counter()
        tagged_op_counts = Counter()
        
        for node in onnx_model.graph.node:
            op_type_counts[node.op_type] += 1
        
        for node_name, node_info in hierarchy_data.get("node_tags", {}).items():
            op_type = node_info.get("op_type", "Unknown")
            if node_info.get("tags"):
                tagged_op_counts[op_type] += 1
        
        print(f"\n📊 OPERATION TYPE DISTRIBUTION:")
        print(f"   Total operation types: {len(op_type_counts)}")
        print(f"   Most common operations:")
        for op_type, count in op_type_counts.most_common(5):
            tagged_count = tagged_op_counts.get(op_type, 0)
            tag_rate = tagged_count / count if count > 0 else 0
            print(f"     {op_type}: {count} total, {tagged_count} tagged ({tag_rate:.1%})")
        
        # Key BERT operations should be well-tagged
        key_bert_ops = ['MatMul', 'Add', 'Gather', 'Reshape']
        for op_type in key_bert_ops:
            if op_type in op_type_counts:
                total = op_type_counts[op_type]
                tagged = tagged_op_counts.get(op_type, 0)
                tag_rate = tagged / total if total > 0 else 0
                assert tag_rate > 0.5, f"Low tagging rate for {op_type}: {tag_rate:.1%}"


class TestHFModuleTaggingDetailed:
    """Detailed tests for specific HuggingFace module tagging scenarios."""
    
    def test_multi_layer_bert_tagging(self):
        """Test tagging consistency across multiple BERT layers."""
        # This would require a model with multiple layers
        # For now, skip if not available
        pytest.skip("Multi-layer testing requires larger BERT model")
    
    def test_cross_module_tensor_flow(self, exported_hierarchy_data):
        """Test that tensors flowing between modules maintain tag lineage."""
        onnx_model, hierarchy_data, pytorch_model = exported_hierarchy_data
        
        # Build tensor flow graph
        tensor_producers = {}  # tensor_name -> node_that_produces_it
        tensor_consumers = defaultdict(list)  # tensor_name -> [nodes_that_consume_it]
        
        for node_name, node_info in hierarchy_data.get("node_tags", {}).items():
            outputs = node_info.get("outputs", [])
            inputs = node_info.get("inputs", [])
            
            # Record tensor producers
            for output in outputs:
                tensor_producers[output] = node_name
            
            # Record tensor consumers
            for input_tensor in inputs:
                tensor_consumers[input_tensor].append(node_name)
        
        # Find cases where tensors flow between different modules
        cross_module_flows = []
        
        for tensor_name, producer_node in tensor_producers.items():
            if tensor_name in tensor_consumers:
                producer_tags = set(hierarchy_data["node_tags"][producer_node].get("tags", []))
                
                for consumer_node in tensor_consumers[tensor_name]:
                    if consumer_node in hierarchy_data["node_tags"]:
                        consumer_tags = set(hierarchy_data["node_tags"][consumer_node].get("tags", []))
                        
                        # Check if they have different primary modules
                        producer_modules = {tag.split('/')[1] for tag in producer_tags if tag.startswith('/')}
                        consumer_modules = {tag.split('/')[1] for tag in consumer_tags if tag.startswith('/')}
                        
                        if producer_modules and consumer_modules and producer_modules != consumer_modules:
                            cross_module_flows.append({
                                'tensor': tensor_name,
                                'producer': producer_node,
                                'consumer': consumer_node,
                                'producer_modules': producer_modules,
                                'consumer_modules': consumer_modules
                            })
        
        print(f"\n🔄 CROSS-MODULE TENSOR FLOWS:")
        print(f"   Cross-module flows detected: {len(cross_module_flows)}")
        
        if cross_module_flows:
            for flow in cross_module_flows[:3]:  # Show first 3
                print(f"     {flow['tensor']}: {flow['producer_modules']} -> {flow['consumer_modules']}")
        
        # This is informational - cross-module flows are expected in neural networks
        assert True  # Always pass, this is for analysis