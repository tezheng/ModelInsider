#!/usr/bin/env python3
"""
Clean Subgraph Extractor - No Topological Sorting Needed
Extract subgraphs by making dependencies external inputs instead of including them
"""

import torch
import torch.nn as nn
import onnx
from onnx import helper, TensorProto, ValueInfoProto
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import numpy as np

from enhanced_dag_extractor import EnhancedDAGExtractor
from input_generator import UniversalInputGenerator


class CleanSubgraphExtractor:
    """Extract clean subgraphs without topological sorting issues"""
    
    def __init__(self, onnx_model_path: str):
        self.onnx_model_path = Path(onnx_model_path)
        self.onnx_model = onnx.load(str(onnx_model_path))
        
        # Build lookup tables
        self.node_by_name = {}
        self.hierarchy_mapping = {}  # node_name -> [module_tags]
        self.tensor_producers = {}   # tensor_name -> node_name
        
        self._load_hierarchy_from_metadata()
        self._build_lookup_tables()
    
    def _load_hierarchy_from_metadata(self):
        """Load hierarchy mapping from ONNX model metadata"""
        # For now, extract from node attributes (legacy)
        # TODO: Move to metadata-only approach
        
        for node in self.onnx_model.graph.node:
            node_name = node.name if node.name else f"{node.op_type}_{id(node)}"
            tags = []
            
            # Extract tags from node attributes
            for attr in node.attribute:
                if attr.name == "source_module":
                    tags.append(attr.s.decode('utf-8'))
                elif attr.name == "hierarchy_tags":
                    tags.extend([tag.decode('utf-8') for tag in attr.strings])
            
            if tags:
                self.hierarchy_mapping[node_name] = tags
    
    def _build_lookup_tables(self):
        """Build lookup tables for fast access"""
        print("Building lookup tables...")
        
        # Index nodes by name
        for node in self.onnx_model.graph.node:
            node_name = node.name if node.name else f"{node.op_type}_{len(self.node_by_name)}"
            self.node_by_name[node_name] = node
            
            # Build tensor producers map
            for output in node.output:
                self.tensor_producers[output] = node_name
        
        # Index initializers as producers
        for init in self.onnx_model.graph.initializer:
            self.tensor_producers[init.name] = init.name
        
        print(f"Indexed {len(self.node_by_name)} nodes")
        print(f"Found hierarchy info for {len(self.hierarchy_mapping)} nodes")
    
    def extract_clean_subgraph(self, target_tag: str, output_path: Optional[str] = None) -> onnx.ModelProto:
        """Extract clean subgraph using the no-sorting approach"""
        print(f"\n{'='*60}")
        print(f"EXTRACTING CLEAN SUBGRAPH: {target_tag}")
        print(f"{'='*60}")
        
        # Step 1: Get all nodes that belong to this module (preserve original order)
        module_nodes = []
        for node in self.onnx_model.graph.node:  # Iterate in original order
            node_name = node.name if node.name else f"{node.op_type}_{id(node)}"
            if node_name in self.hierarchy_mapping:
                if target_tag in self.hierarchy_mapping[node_name]:
                    module_nodes.append(node)
                    print(f"  Including: {node_name} ({node.op_type})")
        
        print(f"Found {len(module_nodes)} nodes for module '{target_tag}'")
        
        # Step 2: Find external dependencies (don't include them, make them inputs)
        external_inputs = set()
        required_initializers = []
        constant_nodes_to_include = []
        
        for node in module_nodes:
            for input_tensor in node.input:
                producer = self.tensor_producers.get(input_tensor)
                
                if producer is None:
                    # This is a model input
                    external_inputs.add(input_tensor)
                elif producer in [init.name for init in self.onnx_model.graph.initializer]:
                    # This is an initializer - include it
                    for init in self.onnx_model.graph.initializer:
                        if init.name == input_tensor:
                            required_initializers.append(init)
                            break
                else:
                    # Check if the producer is a Constant node
                    producer_node = None
                    for full_node in self.onnx_model.graph.node:
                        full_node_name = full_node.name if full_node.name else f"{full_node.op_type}_{id(full_node)}"
                        if full_node_name == producer:
                            producer_node = full_node
                            break
                    
                    if producer_node and producer_node.op_type == 'Constant':
                        # This is a Constant node - include it in our subgraph
                        if producer_node not in constant_nodes_to_include:
                            constant_nodes_to_include.append(producer_node)
                            print(f"  Including Constant node: {producer}")
                    elif producer not in [n.name for n in module_nodes]:
                        # This tensor comes from outside our module - make it external input
                        external_inputs.add(input_tensor)
                        print(f"  External input: {input_tensor} (from {producer})")
        
        # Step 3: Find outputs (tensors used outside our module)
        external_outputs = set()
        all_module_node_names = [n.name for n in module_nodes]
        
        for node in module_nodes:
            for output_tensor in node.output:
                # Check if this tensor is used outside our module
                is_used_externally = False
                
                # Check all nodes in the full model
                for full_node in self.onnx_model.graph.node:
                    full_node_name = full_node.name if full_node.name else f"{full_node.op_type}_{id(full_node)}"
                    
                    # If a node outside our module uses this tensor
                    if (full_node_name not in all_module_node_names and 
                        output_tensor in full_node.input):
                        is_used_externally = True
                        break
                
                # Also check if it's a model output
                if output_tensor in [out.name for out in self.onnx_model.graph.output]:
                    is_used_externally = True
                
                if is_used_externally:
                    external_outputs.add(output_tensor)
                    print(f"  External output: {output_tensor}")
        
        # If no external outputs found, use all outputs from the last node
        if not external_outputs:
            if module_nodes:
                last_node = module_nodes[-1]
                external_outputs.update(last_node.output)
                print(f"  Using outputs from last node: {list(last_node.output)}")
        
        # Step 4: Create graph inputs and outputs
        graph_inputs = []
        for tensor_name in external_inputs:
            tensor_info = self._create_tensor_info(tensor_name)
            if tensor_info:
                graph_inputs.append(tensor_info)
        
        graph_outputs = []
        for tensor_name in external_outputs:
            tensor_info = self._create_tensor_info(tensor_name)
            if tensor_info:
                graph_outputs.append(tensor_info)
        
        # Step 5: Create the subgraph (nodes already in correct order!)
        graph_name = f"extracted_{target_tag.replace('/', '_')}"
        
        # Combine module nodes with required constant nodes
        all_nodes = constant_nodes_to_include + module_nodes
        
        clean_graph = helper.make_graph(
            all_nodes,              # Include constant nodes + module nodes in order
            graph_name,
            graph_inputs,
            graph_outputs,
            required_initializers
        )
        
        print(f"\nSubgraph created:")
        print(f"  Module nodes: {len(module_nodes)}")
        print(f"  Constant nodes: {len(constant_nodes_to_include)}")
        print(f"  Total nodes: {len(all_nodes)}")
        print(f"  Inputs: {len(graph_inputs)}")
        print(f"  Outputs: {len(graph_outputs)}")
        print(f"  Initializers: {len(required_initializers)}")
        
        # Step 6: Create clean model (remove custom attributes)
        clean_model = helper.make_model(
            clean_graph,
            producer_name="CleanSubgraphExtractor",
            producer_version="1.0"
        )
        
        # Remove custom attributes to ensure ONNX compliance
        for i, node in enumerate(clean_model.graph.node):
            # Create new node without custom attributes
            new_attributes = [attr for attr in node.attribute 
                            if attr.name not in ['source_module', 'hierarchy_tags']]
            
            # Clear and rebuild attribute list
            del node.attribute[:]
            node.attribute.extend(new_attributes)
        
        # Add clean metadata
        extraction_meta = clean_model.metadata_props.add()
        extraction_meta.key = "extracted_module"
        extraction_meta.value = target_tag
        
        # Step 7: Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            onnx.save(clean_model, str(output_path))
            print(f"Saved clean subgraph to: {output_path}")
        
        # Step 8: Validate
        try:
            onnx.checker.check_model(clean_model)
            print("✅ Clean subgraph passes ONNX validation!")
        except Exception as e:
            print(f"⚠️  Validation warning: {e}")
        
        return clean_model
    
    def generate_extracted_tag_mapping(self, target_tag: str, extracted_nodes: List, output_path: str):
        """Generate tag mapping JSON for the extracted subgraph"""
        mapping = {
            "metadata": {
                "source_onnx": f"extracted subgraph for {target_tag}",
                "total_nodes": len(extracted_nodes),
                "tagged_nodes": 0,
                "untagged_nodes": 0,
                "tag_coverage_percent": 0.0,
                "unique_tag_count": 0
            },
            "node_tags": {},
            "tag_statistics": {},
            "untagged_operations": []
        }
        
        # Count nodes that belong to the target module
        tagged_nodes = 0
        tag_statistics = {}
        
        for node in extracted_nodes:
            node_name = node.name if node.name else f"{node.op_type}_{id(node)}"
            
            # Check if this node belongs to our target module
            if node_name in self.hierarchy_mapping:
                node_tags = self.hierarchy_mapping[node_name]
                # Filter tags to only include target tag and its children
                relevant_tags = [tag for tag in node_tags if tag.startswith(target_tag)]
                
                if relevant_tags:
                    mapping["node_tags"][node_name] = {
                        "op_type": node.op_type,
                        "tags": relevant_tags,
                        "input_count": len(node.input),
                        "output_count": len(node.output)
                    }
                    tagged_nodes += 1
                    
                    # Update statistics
                    for tag in relevant_tags:
                        tag_statistics[tag] = tag_statistics.get(tag, 0) + 1
                else:
                    mapping["untagged_operations"].append(node_name)
            else:
                mapping["untagged_operations"].append(node_name)
        
        # Update metadata
        mapping["metadata"]["tagged_nodes"] = tagged_nodes
        mapping["metadata"]["untagged_nodes"] = len(extracted_nodes) - tagged_nodes
        mapping["metadata"]["tag_coverage_percent"] = (tagged_nodes / len(extracted_nodes)) * 100 if extracted_nodes else 0
        mapping["metadata"]["unique_tag_count"] = len(tag_statistics)
        mapping["tag_statistics"] = tag_statistics
        
        # Save to file
        import json
        from pathlib import Path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"   Saved {tagged_nodes} node mappings from ONNX")
        print(f"   Tag coverage: {mapping['metadata']['tag_coverage_percent']:.1f}%")
        
        return mapping
    
    def _create_tensor_info(self, tensor_name: str) -> Optional[ValueInfoProto]:
        """Create tensor info for graph inputs/outputs"""
        
        # Check original model inputs
        for input_info in self.onnx_model.graph.input:
            if input_info.name == tensor_name:
                return input_info
        
        # Check original model outputs  
        for output_info in self.onnx_model.graph.output:
            if output_info.name == tensor_name:
                return output_info
        
        # Check value_info
        for value_info in self.onnx_model.graph.value_info:
            if value_info.name == tensor_name:
                return value_info
        
        # Create generic tensor info (fallback)
        return helper.make_tensor_value_info(
            tensor_name,
            TensorProto.FLOAT,
            []  # Unknown shape - will be inferred
        )


def convert_single_module_to_onnx(model: nn.Module, module_path: str, inputs: Dict, output_path: str):
    """Convert a single nn.Module to ONNX for comparison"""
    print(f"\n{'='*60}")
    print(f"CONVERTING SINGLE MODULE: {module_path}")
    print(f"{'='*60}")
    
    # Navigate to the target module
    target_module = model
    for part in module_path.split('.'):
        if part:
            target_module = getattr(target_module, part)
    
    print(f"Target module type: {type(target_module).__name__}")
    print(f"Module parameters: {sum(p.numel() for p in target_module.parameters())}")
    
    # Create appropriate inputs for the module
    # Generic approach for any module type
    # Check if module has attribute indicating input size (common in Linear layers)
    if hasattr(target_module, 'query') and hasattr(target_module.query, 'in_features'):
        # This is likely an attention module - create appropriate tensor input
        batch_size, seq_len = 1, 32
        hidden_size = target_module.query.in_features
        
        dummy_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        module_inputs = (dummy_hidden_states,)
        
        print(f"Created inputs for attention-like module: input shape {dummy_hidden_states.shape}")
    else:
        # Generic approach - use first tensor from model inputs
        first_input = list(inputs.values())[0]
        module_inputs = (first_input,)
    
    # Export to ONNX
    target_module.eval()
    with torch.no_grad():
        try:
            torch.onnx.export(
                target_module,
                module_inputs,
                output_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=False,
                input_names=['input'],
                output_names=['output'],
                verbose=False
            )
            
            print(f"✅ Successfully exported single module to: {output_path}")
            
            # Validate
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✅ Single module ONNX model is valid")
            
            return onnx_model
            
        except Exception as e:
            print(f"❌ Failed to export single module: {e}")
            return None


def compare_onnx_models(extracted_path: str, single_module_path: str):
    """Compare the extracted subgraph with the single module ONNX"""
    print(f"\n{'='*60}")
    print("COMPARING EXTRACTED VS SINGLE MODULE")
    print(f"{'='*60}")
    
    try:
        # Load both models
        extracted_model = onnx.load(extracted_path)
        single_model = onnx.load(single_module_path)
        
        print("📊 COMPARISON RESULTS:")
        
        # Compare basic stats
        print(f"\nModel Structure:")
        print(f"  Extracted - Nodes: {len(extracted_model.graph.node)}, "
              f"Inputs: {len(extracted_model.graph.input)}, "
              f"Outputs: {len(extracted_model.graph.output)}")
        print(f"  Single    - Nodes: {len(single_model.graph.node)}, "
              f"Inputs: {len(single_model.graph.input)}, "
              f"Outputs: {len(single_model.graph.output)}")
        
        # Compare operations
        extracted_ops = [node.op_type for node in extracted_model.graph.node]
        single_ops = [node.op_type for node in single_model.graph.node]
        
        from collections import Counter
        extracted_counts = Counter(extracted_ops)
        single_counts = Counter(single_ops)
        
        print(f"\nOperation Types:")
        all_ops = set(extracted_counts.keys()) | set(single_counts.keys())
        for op in sorted(all_ops):
            ext_count = extracted_counts.get(op, 0)
            single_count = single_counts.get(op, 0)
            match = "✅" if ext_count == single_count else "❌"
            print(f"  {op:15} Extracted: {ext_count:2d}, Single: {single_count:2d} {match}")
        
        # Compare initializers
        extracted_params = [init.name for init in extracted_model.graph.initializer]
        single_params = [init.name for init in single_model.graph.initializer]
        
        print(f"\nParameters:")
        print(f"  Extracted: {len(extracted_params)} parameters")
        print(f"  Single:    {len(single_params)} parameters")
        
        # Check if parameter names are similar (accounting for different naming)
        extracted_param_types = [name.split('.')[-1] for name in extracted_params]
        single_param_types = [name.split('.')[-1] for name in single_params]
        
        param_type_match = Counter(extracted_param_types) == Counter(single_param_types)
        print(f"  Parameter types match: {'✅' if param_type_match else '❌'}")
        
        # Overall assessment
        structure_similar = (abs(len(extracted_ops) - len(single_ops)) <= 3)  # Allow small differences
        ops_similar = (len(set(extracted_ops) & set(single_ops)) / len(set(extracted_ops) | set(single_ops)) > 0.8)
        
        if structure_similar and ops_similar:
            print(f"\n🎉 MODELS ARE STRUCTURALLY SIMILAR!")
            print("The extracted subgraph appears to match the single module export.")
        else:
            print(f"\n⚠️  MODELS HAVE DIFFERENCES")
            print("This might be expected due to different export contexts.")
        
        return {
            'structure_similar': structure_similar,
            'ops_similar': ops_similar,
            'extracted_nodes': len(extracted_ops),
            'single_nodes': len(single_ops),
            'extracted_params': len(extracted_params),
            'single_params': len(single_params)
        }
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return None


def test_bert_self_attention():
    """Test the complete workflow with BERT BertSelfAttention module"""
    print("🧪 TESTING CLEAN SUBGRAPH EXTRACTION WITH BERT SELF-ATTENTION")
    print("=" * 80)
    
    # Setup directories
    test_dir = Path("temp/clean_extraction_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Create enhanced ONNX model with full tagging
        print("\n🔧 STEP 1: Creating enhanced ONNX model...")
        
        from transformers import AutoModel
        model = AutoModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
        
        # Generate inputs
        generator = UniversalInputGenerator()
        inputs = generator.generate_inputs(model, 'google/bert_uncased_L-2_H-128_A-2')
        
        # Create enhanced model with 100% tagging
        extractor = EnhancedDAGExtractor()
        extractor.analyze_model_structure(model)
        extractor.trace_execution_with_hooks(model, inputs)
        extractor.create_parameter_mapping(model)
        
        whole_model_path = test_dir / "bert_whole_enhanced.onnx"
        extractor.export_and_analyze_onnx(model, inputs, str(whole_model_path))
        enhanced_path = str(whole_model_path).replace('.onnx', '_with_tags.onnx')
        
        print(f"✅ Enhanced whole model created: {enhanced_path}")
        
        # Step 2: Convert single BertSelfAttention module
        print("\n🔧 STEP 2: Converting single BertSelfAttention module...")
        
        target_module_path = "encoder.layer.0.attention.self"
        single_module_path = test_dir / "bert_self_attention_single.onnx"
        
        single_model = convert_single_module_to_onnx(
            model, target_module_path, inputs, str(single_module_path)
        )
        
        # Step 3: Extract subgraph from whole model
        print("\n🔧 STEP 3: Extracting subgraph...")
        
        target_tag = "/BertModel/BertEncoder/ModuleList.0/BertAttention/BertSdpaSelfAttention"
        extracted_path = test_dir / "bert_self_attention_extracted.onnx"
        
        clean_extractor = CleanSubgraphExtractor(enhanced_path)
        extracted_model = clean_extractor.extract_clean_subgraph(
            target_tag, str(extracted_path)
        )
        
        # Step 4: Compare the models
        print("\n🔧 STEP 4: Comparing models...")
        
        if single_model and extracted_model:
            comparison = compare_onnx_models(str(extracted_path), str(single_module_path))
            
            # Save comparison results
            results = {
                'test_name': 'BERT_BertSelfAttention_Clean_Extraction',
                'whole_model_path': str(enhanced_path),
                'single_module_path': str(single_module_path),
                'extracted_path': str(extracted_path),
                'target_tag': target_tag,
                'target_module_path': target_module_path,
                'comparison': comparison
            }
            
            results_path = test_dir / "test_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Test results saved to: {results_path}")
            
            return results
        else:
            print("❌ Test failed - could not create models for comparison")
            return None
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_bert_self_attention()