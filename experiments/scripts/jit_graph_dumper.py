#!/usr/bin/env python3
"""
JIT Graph Access and Context Preservation Module

This module implements functionality to access TorchScript graph information
before ONNX conversion, where module hierarchy is still available.
"""

import json
import re
from pathlib import Path
from typing import Any

import torch


class JITGraphDumper:
    """
    Dumps TorchScript graph information before ONNX conversion.
    
    This class can extract module hierarchy information that gets lost
    during the ONNX conversion process.
    """
    
    def __init__(self):
        self.graph_info = {}
        self.scope_hierarchy = set()
        self.node_scope_mapping = {}
    
    def extract_jit_graph_info(
        self, 
        traced_model: torch.jit.ScriptModule,
        output_path: str | None = None
    ) -> dict[str, Any]:
        """
        Extract comprehensive information from TorchScript graph.
        
        Args:
            traced_model: The traced TorchScript model
            output_path: Optional path to save the extracted information
            
        Returns:
            Dictionary containing graph information and scope mappings
        """
        
        graph = traced_model.graph
        nodes = list(graph.nodes())
        
        print(f"🔍 Analyzing TorchScript graph with {len(nodes)} nodes...")
        
        # Extract basic graph information
        graph_info = {
            "total_nodes": len(nodes),
            "graph_string_length": len(str(graph)),
            "graph_type": str(type(graph)),
            "extraction_timestamp": torch.jit._get_jit_operator_name_for_aten_op.__name__ if hasattr(torch.jit, '_get_jit_operator_name_for_aten_op') else "unknown"
        }
        
        # Method 1: Try to access individual node scope information
        node_scopes = self._extract_node_scopes(nodes)
        
        # Method 2: Parse the graph string representation for scope information
        graph_string_scopes = self._parse_graph_string_scopes(str(graph))
        
        # Method 3: Try to access inlined graph if available
        inlined_scopes = self._extract_inlined_graph_scopes(traced_model)
        
        # Combine all scope information
        all_scopes = {
            "node_level_scopes": node_scopes,
            "graph_string_scopes": graph_string_scopes, 
            "inlined_graph_scopes": inlined_scopes
        }
        
        # Create unified scope hierarchy
        unified_scopes = self._create_unified_scope_mapping(all_scopes)
        
        result = {
            "graph_info": graph_info,
            "scope_extraction_methods": all_scopes,
            "unified_scope_hierarchy": unified_scopes,
            "scope_statistics": self._compute_scope_statistics(unified_scopes)
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✅ JIT graph information saved to: {output_path}")
        
        return result
    
    def _extract_node_scopes(self, nodes: list) -> dict[str, Any]:
        """Extract scope information from individual nodes."""
        
        node_scopes = {}
        successful_extractions = 0
        
        for i, node in enumerate(nodes):
            node_info = {
                "index": i,
                "kind": str(node.kind()),
                "scope_methods": {}
            }
            
            # Try different scope extraction methods
            scope_methods = {
                "scopeName": lambda n: str(n.scopeName()) if hasattr(n, 'scopeName') else None,
                "sourceRange": lambda n: str(n.sourceRange()) if hasattr(n, 'sourceRange') else None,
                "debugName": lambda n: str(n.debugName()) if hasattr(n, 'debugName') else None,
            }
            
            for method_name, method_func in scope_methods.items():
                try:
                    result = method_func(node)
                    node_info["scope_methods"][method_name] = result
                    if result and result.strip() and result != "":
                        successful_extractions += 1
                        self.scope_hierarchy.add(result)
                except Exception as e:
                    node_info["scope_methods"][method_name] = f"Error: {e}"
            
            node_scopes[f"node_{i}"] = node_info
        
        return {
            "total_nodes": len(nodes),
            "successful_extractions": successful_extractions,
            "node_details": node_scopes
        }
    
    def _parse_graph_string_scopes(self, graph_str: str) -> dict[str, Any]:
        """Parse scope information from graph string representation."""
        
        # Look for scope patterns in the graph string
        scope_pattern = r'scope:\s*([^#\n]+)'
        scope_matches = re.findall(scope_pattern, graph_str)
        
        # Also look for module patterns
        module_pattern = r'%[^:]*:\s*([^=]+)='
        module_matches = re.findall(module_pattern, graph_str)
        
        # Extract unique scopes
        unique_scopes = set()
        for scope in scope_matches:
            cleaned_scope = scope.strip()
            if cleaned_scope:
                unique_scopes.add(cleaned_scope)
                self.scope_hierarchy.add(cleaned_scope)
        
        return {
            "graph_contains_scope_info": "scope:" in graph_str,
            "total_scope_matches": len(scope_matches),
            "unique_scopes": sorted(unique_scopes),
            "unique_scope_count": len(unique_scopes),
            "module_pattern_matches": len(module_matches),
            "sample_scopes": sorted(unique_scopes)[:10]  # First 10 for preview
        }
    
    def _extract_inlined_graph_scopes(self, traced_model: torch.jit.ScriptModule) -> dict[str, Any]:
        """Try to extract scope information from inlined graph."""
        
        inlined_info = {
            "has_inlined_graph": hasattr(traced_model, 'inlined_graph'),
            "inlined_scopes": []
        }
        
        try:
            if hasattr(traced_model, 'inlined_graph'):
                inlined_graph = traced_model.inlined_graph
                inlined_nodes = list(inlined_graph.nodes())
                
                inlined_info["inlined_node_count"] = len(inlined_nodes)
                
                for i, node in enumerate(inlined_nodes[:50]):  # Limit to first 50
                    scope_name = str(node.scopeName()) if hasattr(node, 'scopeName') else None
                    if scope_name and scope_name.strip():
                        inlined_info["inlined_scopes"].append({
                            "node_index": i,
                            "kind": str(node.kind()),
                            "scope": scope_name.strip()
                        })
                        self.scope_hierarchy.add(scope_name.strip())
            
        except Exception as e:
            inlined_info["error"] = str(e)
        
        return inlined_info
    
    def _create_unified_scope_mapping(self, all_scopes: dict[str, Any]) -> dict[str, Any]:
        """Create a unified scope hierarchy from all extraction methods."""
        
        # Collect all unique scopes
        all_unique_scopes = set()
        
        # From graph string parsing
        if "unique_scopes" in all_scopes["graph_string_scopes"]:
            all_unique_scopes.update(all_scopes["graph_string_scopes"]["unique_scopes"])
        
        # From inlined graph
        for scope_info in all_scopes["inlined_graph_scopes"]["inlined_scopes"]:
            all_unique_scopes.add(scope_info["scope"])
        
        # Add from class-level scope hierarchy
        all_unique_scopes.update(self.scope_hierarchy)
        
        # Analyze scope structure
        scope_analysis = self._analyze_scope_structure(all_unique_scopes)
        
        return {
            "total_unique_scopes": len(all_unique_scopes),
            "scope_list": sorted(all_unique_scopes),
            "scope_analysis": scope_analysis
        }
    
    def _analyze_scope_structure(self, scopes: set) -> dict[str, Any]:
        """Analyze the structure of extracted scopes."""
        
        analysis = {
            "depth_distribution": {},
            "module_types": set(),
            "common_prefixes": {},
            "transformers_specific": {
                "bert_modules": [],
                "attention_modules": [],
                "layer_modules": []
            }
        }
        
        for scope in scopes:
            # Analyze depth (count of :: or / separators)
            depth = scope.count('::') + scope.count('/')
            analysis["depth_distribution"][depth] = analysis["depth_distribution"].get(depth, 0) + 1
            
            # Extract module types
            if '::' in scope:
                parts = scope.split('::')
                for part in parts:
                    if part and '.' not in part:  # Likely a class name
                        analysis["module_types"].add(part.split('/')[-1])
            
            # Find common prefixes
            if '::' in scope:
                prefix = scope.split('::')[0]
                analysis["common_prefixes"][prefix] = analysis["common_prefixes"].get(prefix, 0) + 1
            
            # Transformers-specific analysis
            scope_lower = scope.lower()
            if 'bert' in scope_lower:
                analysis["transformers_specific"]["bert_modules"].append(scope)
            if 'attention' in scope_lower:
                analysis["transformers_specific"]["attention_modules"].append(scope)
            if 'layer' in scope_lower:
                analysis["transformers_specific"]["layer_modules"].append(scope)
        
        # Convert sets to lists for JSON serialization
        analysis["module_types"] = sorted(analysis["module_types"])
        
        return analysis
    
    def _compute_scope_statistics(self, unified_scopes: dict[str, Any]) -> dict[str, Any]:
        """Compute statistics about the extracted scope information."""
        
        stats = {
            "total_scopes": unified_scopes["total_unique_scopes"],
            "extraction_success": unified_scopes["total_unique_scopes"] > 0,
            "depth_stats": {},
            "coverage_analysis": {}
        }
        
        if "scope_analysis" in unified_scopes:
            analysis = unified_scopes["scope_analysis"]
            
            # Depth statistics
            if "depth_distribution" in analysis:
                depths = list(analysis["depth_distribution"].keys())
                if depths:
                    stats["depth_stats"] = {
                        "min_depth": min(depths),
                        "max_depth": max(depths),
                        "avg_depth": sum(d * c for d, c in analysis["depth_distribution"].items()) / sum(analysis["depth_distribution"].values())
                    }
            
            # Coverage analysis
            stats["coverage_analysis"] = {
                "has_bert_modules": len(analysis["transformers_specific"]["bert_modules"]) > 0,
                "has_attention_modules": len(analysis["transformers_specific"]["attention_modules"]) > 0,
                "has_layer_modules": len(analysis["transformers_specific"]["layer_modules"]) > 0,
                "unique_module_types": len(analysis["module_types"])
            }
        
        return stats


def dump_jit_graph_before_onnx_export(
    model: torch.nn.Module,
    example_inputs: Any,
    output_dir: str = "temp"
) -> tuple[torch.jit.ScriptModule, dict[str, Any]]:
    """
    Convenience function to trace model and dump graph info before ONNX export.
    
    Args:
        model: PyTorch model to trace
        example_inputs: Example inputs for tracing
        output_dir: Directory to save the graph information
        
    Returns:
        Tuple of (traced_model, graph_info_dict)
    """
    
    # Ensure output directory exists
    Path(output_dir).mkdir(exist_ok=True)
    
    # Trace the model
    print("🔄 Tracing model...")
    with torch.no_grad():
        if isinstance(example_inputs, dict):
            # Handle dict inputs (like from tokenizer) - convert to positional args
            # Use only the main inputs to avoid conflicts
            main_inputs = []
            if 'input_ids' in example_inputs:
                main_inputs.append(example_inputs['input_ids'])
            if 'attention_mask' in example_inputs:
                main_inputs.append(example_inputs['attention_mask'])
            
            traced_model = torch.jit.trace(
                model, 
                tuple(main_inputs),
                strict=False,
                check_trace=False
            )
        else:
            # Handle tuple/tensor inputs
            traced_model = torch.jit.trace(model, example_inputs, strict=False, check_trace=False)
    
    print("✅ Model traced successfully")
    
    # Extract graph information
    dumper = JITGraphDumper()
    output_path = f"{output_dir}/jit_graph_info.json"
    
    graph_info = dumper.extract_jit_graph_info(traced_model, output_path)
    
    return traced_model, graph_info


if __name__ == "__main__":
    # Example usage
    from transformers import AutoModel, AutoTokenizer
    
    # Load test model
    model_name = "prajjwal1/bert-tiny"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare inputs
    inputs = tokenizer("Hello world", return_tensors="pt")
    
    # Dump graph info
    traced_model, graph_info = dump_jit_graph_before_onnx_export(model, inputs)
    
    print("\n🎯 EXTRACTION SUMMARY:")
    print(f"Total scopes found: {graph_info['unified_scope_hierarchy']['total_unique_scopes']}")
    print(f"Extraction successful: {graph_info['scope_statistics']['extraction_success']}")
    
    if graph_info['scope_statistics']['extraction_success']:
        print("✅ Successfully extracted module hierarchy from TorchScript!")
    else:
        print("❌ No scope information could be extracted")