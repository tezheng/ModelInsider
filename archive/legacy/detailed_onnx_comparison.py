#!/usr/bin/env python3
"""
Detailed ONNX Model Comparison
Generate nodes/edges JSON and operator descriptions, then compare excluding inputs/outputs
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx


class ONNXModelAnalyzer:
    """Analyze ONNX model and generate detailed JSON representations"""
    
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.model = onnx.load(model_path)
        
    def generate_nodes_edges_json(self) -> dict[str, Any]:
        """Generate nodes and edges representation as JSON"""
        nodes = []
        edges = []
        node_outputs = {}  # Track which node produces each output
        
        # Process all nodes (excluding inputs/outputs from the graph level)
        for i, node in enumerate(self.model.graph.node):
            node_id = f"node_{i}"
            node_name = node.name if node.name else f"{node.op_type}_{i}"
            
            # Create node representation
            node_data = {
                "id": node_id,
                "name": node_name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": {}
            }
            
            # Extract attributes
            for attr in node.attribute:
                attr_value = self._extract_attribute_value(attr)
                node_data["attributes"][attr.name] = attr_value
            
            nodes.append(node_data)
            
            # Track outputs for edge creation
            for output in node.output:
                node_outputs[output] = node_id
        
        # Create edges based on data flow
        for node in nodes:
            for input_tensor in node["inputs"]:
                # Skip model inputs (they don't come from internal nodes)
                if input_tensor in node_outputs:
                    source_node = node_outputs[input_tensor]
                    target_node = node["id"]
                    
                    edge = {
                        "source": source_node,
                        "target": target_node,
                        "tensor": input_tensor
                    }
                    edges.append(edge)
        
        return {
            "model_name": self.model_name,
            "nodes": nodes,
            "edges": edges,
            "statistics": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "node_types": dict(Counter(node["op_type"] for node in nodes))
            }
        }
    
    def generate_operator_descriptions(self) -> dict[str, Any]:
        """Generate detailed operator descriptions"""
        operators = defaultdict(list)
        
        for i, node in enumerate(self.model.graph.node):
            op_type = node.op_type
            node_name = node.name if node.name else f"{op_type}_{i}"
            
            # Create operator instance description
            op_instance = {
                "node_index": i,
                "node_name": node_name,
                "input_count": len(node.input),
                "output_count": len(node.output),
                "attributes": {},
                "input_shapes": [],  # Would need shape inference for this
                "output_shapes": []  # Would need shape inference for this
            }
            
            # Extract attributes
            for attr in node.attribute:
                attr_value = self._extract_attribute_value(attr)
                op_instance["attributes"][attr.name] = attr_value
            
            operators[op_type].append(op_instance)
        
        # Create summary statistics
        operator_summary = {}
        for op_type, instances in operators.items():
            operator_summary[op_type] = {
                "count": len(instances),
                "instances": instances,
                "common_attributes": self._find_common_attributes(instances)
            }
        
        return {
            "model_name": self.model_name,
            "operators": operator_summary,
            "statistics": {
                "total_operators": len(operators),
                "total_instances": sum(len(instances) for instances in operators.values()),
                "operator_counts": {op_type: len(instances) for op_type, instances in operators.items()}
            }
        }
    
    def _extract_attribute_value(self, attr):
        """Extract attribute value based on its type"""
        if attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode('utf-8')
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr.type == onnx.AttributeProto.STRINGS:
            return [s.decode('utf-8') for s in attr.strings]
        else:
            return f"<{attr.type}>"
    
    def _find_common_attributes(self, instances: list[dict]) -> dict[str, Any]:
        """Find attributes that are common across all instances"""
        if not instances:
            return {}
        
        common_attrs = {}
        first_attrs = instances[0]["attributes"]
        
        for attr_name, attr_value in first_attrs.items():
            # Check if all instances have this attribute with the same value
            if all(inst["attributes"].get(attr_name) == attr_value for inst in instances):
                common_attrs[attr_name] = attr_value
        
        return common_attrs


class ONNXModelComparator:
    """Compare two ONNX models in detail"""
    
    def __init__(self, standalone_path: str, extracted_path: str):
        self.standalone_analyzer = ONNXModelAnalyzer(standalone_path, "standalone")
        self.extracted_analyzer = ONNXModelAnalyzer(extracted_path, "extracted")
    
    def compare_models(self, output_dir: Path) -> dict[str, Any]:
        """Perform detailed comparison and save results"""
        print("🔍 DETAILED ONNX MODEL COMPARISON")
        print("=" * 60)
        
        # Generate detailed representations
        print("📊 Generating nodes/edges representations...")
        standalone_graph = self.standalone_analyzer.generate_nodes_edges_json()
        extracted_graph = self.extracted_analyzer.generate_nodes_edges_json()
        
        print("🔧 Generating operator descriptions...")
        standalone_ops = self.standalone_analyzer.generate_operator_descriptions()
        extracted_ops = self.extracted_analyzer.generate_operator_descriptions()
        
        # Save detailed JSONs
        output_dir.mkdir(parents=True, exist_ok=True)
        
        standalone_graph_path = output_dir / "standalone_nodes_edges.json"
        extracted_graph_path = output_dir / "extracted_nodes_edges.json"
        standalone_ops_path = output_dir / "standalone_operators.json"
        extracted_ops_path = output_dir / "extracted_operators.json"
        
        with open(standalone_graph_path, 'w') as f:
            json.dump(standalone_graph, f, indent=2)
        
        with open(extracted_graph_path, 'w') as f:
            json.dump(extracted_graph, f, indent=2)
        
        with open(standalone_ops_path, 'w') as f:
            json.dump(standalone_ops, f, indent=2)
        
        with open(extracted_ops_path, 'w') as f:
            json.dump(extracted_ops, f, indent=2)
        
        print(f"📁 Saved detailed representations to {output_dir}")
        
        # Perform comparisons
        graph_comparison = self._compare_graphs(standalone_graph, extracted_graph)
        operator_comparison = self._compare_operators(standalone_ops, extracted_ops)
        
        # Generate comprehensive comparison report
        comparison_report = {
            "comparison_type": "detailed_onnx_comparison",
            "models": {
                "standalone": self.standalone_analyzer.model_path,
                "extracted": self.extracted_analyzer.model_path
            },
            "files_generated": {
                "standalone_graph": str(standalone_graph_path),
                "extracted_graph": str(extracted_graph_path),
                "standalone_operators": str(standalone_ops_path),
                "extracted_operators": str(extracted_ops_path)
            },
            "graph_comparison": graph_comparison,
            "operator_comparison": operator_comparison
        }
        
        # Save comparison report
        comparison_path = output_dir / "detailed_comparison_report.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison_report, f, indent=2)
        
        print(f"📋 Saved comparison report to {comparison_path}")
        
        # Print summary
        self._print_comparison_summary(graph_comparison, operator_comparison)
        
        return comparison_report
    
    def _compare_graphs(self, standalone_graph: dict, extracted_graph: dict) -> dict[str, Any]:
        """Compare graph structures (nodes and edges)"""
        standalone_nodes = standalone_graph["nodes"]
        extracted_nodes = extracted_graph["nodes"]
        standalone_edges = standalone_graph["edges"]
        extracted_edges = extracted_graph["edges"]
        
        # Compare node types and counts
        standalone_node_types = Counter(node["op_type"] for node in standalone_nodes)
        extracted_node_types = Counter(node["op_type"] for node in extracted_nodes)
        
        # Find differences in node types
        all_node_types = set(standalone_node_types.keys()) | set(extracted_node_types.keys())
        node_type_differences = {}
        
        for node_type in all_node_types:
            standalone_count = standalone_node_types.get(node_type, 0)
            extracted_count = extracted_node_types.get(node_type, 0)
            
            if standalone_count != extracted_count:
                node_type_differences[node_type] = {
                    "standalone": standalone_count,
                    "extracted": extracted_count,
                    "difference": extracted_count - standalone_count
                }
        
        # Compare graph connectivity
        standalone_connectivity = len(standalone_edges)
        extracted_connectivity = len(extracted_edges)
        
        return {
            "node_count": {
                "standalone": len(standalone_nodes),
                "extracted": len(extracted_nodes),
                "difference": len(extracted_nodes) - len(standalone_nodes)
            },
            "edge_count": {
                "standalone": standalone_connectivity,
                "extracted": extracted_connectivity,
                "difference": extracted_connectivity - standalone_connectivity
            },
            "node_type_differences": node_type_differences,
            "node_types_match": len(node_type_differences) == 0
        }
    
    def _compare_operators(self, standalone_ops: dict, extracted_ops: dict) -> dict[str, Any]:
        """Compare operator descriptions in detail"""
        standalone_op_counts = standalone_ops["statistics"]["operator_counts"]
        extracted_op_counts = extracted_ops["statistics"]["operator_counts"]
        
        # Compare operator instances
        all_op_types = set(standalone_op_counts.keys()) | set(extracted_op_counts.keys())
        operator_differences = {}
        detailed_operator_analysis = {}
        
        for op_type in all_op_types:
            standalone_count = standalone_op_counts.get(op_type, 0)
            extracted_count = extracted_op_counts.get(op_type, 0)
            
            if standalone_count != extracted_count:
                operator_differences[op_type] = {
                    "standalone": standalone_count,
                    "extracted": extracted_count,
                    "difference": extracted_count - standalone_count
                }
            
            # Detailed analysis for operators present in both
            if op_type in standalone_ops["operators"] and op_type in extracted_ops["operators"]:
                standalone_instances = standalone_ops["operators"][op_type]["instances"]
                extracted_instances = extracted_ops["operators"][op_type]["instances"]
                
                detailed_operator_analysis[op_type] = {
                    "instance_count_match": len(standalone_instances) == len(extracted_instances),
                    "standalone_instances": len(standalone_instances),
                    "extracted_instances": len(extracted_instances),
                    "attribute_analysis": self._compare_operator_attributes(
                        standalone_instances, extracted_instances
                    )
                }
        
        return {
            "operator_count_differences": operator_differences,
            "operators_match": len(operator_differences) == 0,
            "detailed_operator_analysis": detailed_operator_analysis,
            "total_operators": {
                "standalone": len(standalone_op_counts),
                "extracted": len(extracted_op_counts)
            }
        }
    
    def _compare_operator_attributes(self, standalone_instances: list, extracted_instances: list) -> dict[str, Any]:
        """Compare attributes across operator instances"""
        if not standalone_instances or not extracted_instances:
            return {"status": "no_instances_to_compare"}
        
        # Compare common attributes
        standalone_attrs = set()
        extracted_attrs = set()
        
        for instance in standalone_instances:
            standalone_attrs.update(instance["attributes"].keys())
        
        for instance in extracted_instances:
            extracted_attrs.update(instance["attributes"].keys())
        
        common_attrs = standalone_attrs & extracted_attrs
        standalone_only = standalone_attrs - extracted_attrs
        extracted_only = extracted_attrs - standalone_attrs
        
        return {
            "common_attributes": list(common_attrs),
            "standalone_only_attributes": list(standalone_only),
            "extracted_only_attributes": list(extracted_only),
            "attribute_sets_match": len(standalone_only) == 0 and len(extracted_only) == 0
        }
    
    def _print_comparison_summary(self, graph_comparison: dict, operator_comparison: dict):
        """Print a human-readable comparison summary"""
        print("\n📈 GRAPH STRUCTURE COMPARISON:")
        print(f"   Nodes: Standalone {graph_comparison['node_count']['standalone']}, "
              f"Extracted {graph_comparison['node_count']['extracted']} "
              f"(Δ {graph_comparison['node_count']['difference']:+d})")
        
        print(f"   Edges: Standalone {graph_comparison['edge_count']['standalone']}, "
              f"Extracted {graph_comparison['edge_count']['extracted']} "
              f"(Δ {graph_comparison['edge_count']['difference']:+d})")
        
        if graph_comparison["node_type_differences"]:
            print("\n🔧 NODE TYPE DIFFERENCES:")
            for node_type, diff in graph_comparison["node_type_differences"].items():
                print(f"   {node_type:15} Standalone: {diff['standalone']:2d}, "
                      f"Extracted: {diff['extracted']:2d} (Δ {diff['difference']:+d})")
        else:
            print("   ✅ All node types match perfectly!")
        
        print("\n⚙️  OPERATOR COMPARISON:")
        if operator_comparison["operators_match"]:
            print("   ✅ All operator counts match perfectly!")
        else:
            print("   ❌ Operator count differences found:")
            for op_type, diff in operator_comparison["operator_count_differences"].items():
                print(f"   {op_type:15} Standalone: {diff['standalone']:2d}, "
                      f"Extracted: {diff['extracted']:2d} (Δ {diff['difference']:+d})")
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        structure_match = graph_comparison["node_types_match"]
        operator_match = operator_comparison["operators_match"]
        
        if structure_match and operator_match:
            print("   ✅ Models are structurally identical!")
        elif structure_match:
            print("   ⚠️  Node types match, but operator counts differ")
        elif operator_match:
            print("   ⚠️  Operator counts match, but node types differ")
        else:
            print("   ❌ Significant structural differences found")


def main():
    """Main comparison function"""
    # Define paths
    test_dir = Path("temp/bert_self_attention_test")
    standalone_path = test_dir / "bert_self_attention_standalone.onnx"
    extracted_path = test_dir / "bert_self_attention_extracted.onnx"
    output_dir = test_dir / "detailed_comparison"
    
    # Check if files exist
    if not standalone_path.exists():
        print(f"❌ Standalone model not found: {standalone_path}")
        return
    
    if not extracted_path.exists():
        print(f"❌ Extracted model not found: {extracted_path}")
        return
    
    # Perform detailed comparison
    comparator = ONNXModelComparator(str(standalone_path), str(extracted_path))
    comparison_report = comparator.compare_models(output_dir)
    
    print(f"\n💾 Detailed comparison complete!")
    print(f"📁 All files saved to: {output_dir}")
    
    return comparison_report


if __name__ == "__main__":
    main()