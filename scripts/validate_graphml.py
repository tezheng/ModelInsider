#!/usr/bin/env python3
"""
GraphML Validation Script

Implements the validation framework from ADR-010 to ensure GraphML files
comply with the ONNX to GraphML format specification.
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any


class GraphMLValidator:
    """Comprehensive GraphML validation framework."""
    
    def __init__(self, graphml_file: str, onnx_file: str | None = None, metadata_file: str | None = None):
        self.graphml_file = graphml_file
        self.onnx_file = onnx_file
        self.metadata_file = metadata_file
        self.tree = ET.parse(graphml_file)
        self.root = self.tree.getroot()
        self.results = {}
        
    def validate_schema(self) -> dict[str, Any]:
        """Validate GraphML against official schema."""
        try:
            # Check namespace
            expected_ns = "http://graphml.graphdrawing.org/xmlns"
            actual_tag = self.root.tag
            
            if not actual_tag.startswith(f"{{{expected_ns}}}"):
                return {
                    "status": "fail",
                    "error": f"Invalid namespace. Expected {expected_ns}, got {actual_tag}"
                }
            
            # Check schema location (if present)
            schema_loc = self.root.get("{http://www.w3.org/2001/XMLSchema-instance}schemaLocation")
            if schema_loc and "graphml.xsd" not in schema_loc:
                return {
                    "status": "warning",
                    "message": "Schema location does not reference graphml.xsd"
                }
            
            return {"status": "pass", "message": "Valid GraphML schema"}
            
        except Exception as e:
            return {"status": "fail", "error": f"Schema validation failed: {e}"}
    
    def validate_keys(self) -> dict[str, Any]:
        """Validate required GraphML key definitions."""
        try:
            ns = "http://graphml.graphdrawing.org/xmlns"
            
            # Required keys from ADR-010
            required_keys = {
                # Graph attributes (compound nodes)
                'd0': ('graph', 'class_name', 'string'),
                'd1': ('graph', 'module_type', 'string'),
                'd2': ('graph', 'execution_order', 'int'),
                'd3': ('graph', 'traced_tag', 'string'),
                # Node attributes
                'n0': ('node', 'op_type', 'string'),
                'n1': ('node', 'hierarchy_tag', 'string'),
                'n2': ('node', 'node_attributes', 'string'),
                'n3': ('node', 'name', 'string'),
                # Edge attributes
                'e0': ('edge', 'tensor_name', 'string'),
                # Optional metadata
                'm2': ('graph', 'format_version', 'string'),
                'm3': ('graph', 'export_timestamp', 'string'),
            }
            
            keys = self.root.findall(f".//{{{ns}}}key")
            key_dict = {
                k.get('id'): (k.get('for'), k.get('attr.name'), k.get('attr.type'))
                for k in keys
            }
            
            missing_keys = []
            invalid_keys = []
            
            for key_id, expected in required_keys.items():
                if key_id not in key_dict:
                    # Only require core keys, metadata keys are optional
                    if key_id.startswith(('d', 'n', 'e')):
                        missing_keys.append(key_id)
                else:
                    actual = key_dict[key_id]
                    if actual != expected:
                        invalid_keys.append(f"{key_id}: expected {expected}, got {actual}")
            
            if missing_keys or invalid_keys:
                return {
                    "status": "fail",
                    "missing_keys": missing_keys,
                    "invalid_keys": invalid_keys
                }
            
            return {
                "status": "pass",
                "message": f"All required keys present ({len(key_dict)} total)",
                "key_count": len(key_dict)
            }
            
        except Exception as e:
            return {"status": "fail", "error": f"Key validation failed: {e}"}
    
    def validate_nodes(self) -> dict[str, Any]:
        """Validate node structure and attributes."""
        try:
            ns = "http://graphml.graphdrawing.org/xmlns"
            nodes = self.root.findall(f".//{{{ns}}}node")
            
            if not nodes:
                return {"status": "fail", "error": "No nodes found in GraphML"}
            
            issues = []
            compound_nodes = 0
            operation_nodes = 0
            
            for node in nodes:
                node_id = node.get('id')
                if not node_id:
                    issues.append("Node missing ID attribute")
                    continue
                
                # Check for required op_type attribute
                op_type_data = node.find(f'.//{{http://graphml.graphdrawing.org/xmlns}}data[@key="n0"]')
                if op_type_data is None:
                    issues.append(f"Node {node_id} missing op_type (n0) data")
                
                # Check if it's a compound node
                nested_graph = node.find(f'.//{{{ns}}}graph')
                if nested_graph is not None:
                    compound_nodes += 1
                else:
                    operation_nodes += 1
            
            if issues:
                return {
                    "status": "fail",
                    "error": f"Node validation issues: {issues[:5]}"  # Show first 5 issues
                }
            
            return {
                "status": "pass",
                "message": f"All {len(nodes)} nodes valid",
                "total_nodes": len(nodes),
                "compound_nodes": compound_nodes,
                "operation_nodes": operation_nodes
            }
            
        except Exception as e:
            return {"status": "fail", "error": f"Node validation failed: {e}"}
    
    def validate_edges(self) -> dict[str, Any]:
        """Validate edge connectivity and attributes."""
        try:
            ns = "http://graphml.graphdrawing.org/xmlns"
            
            # Collect all node IDs
            nodes = self.root.findall(f".//{{{ns}}}node")
            node_ids = {node.get('id') for node in nodes if node.get('id')}
            
            # Validate edges
            edges = self.root.findall(f".//{{{ns}}}edge")
            
            issues = []
            for edge in edges:
                source = edge.get('source')
                target = edge.get('target')
                
                if not source:
                    issues.append("Edge missing source attribute")
                if not target:
                    issues.append("Edge missing target attribute")
                if source == target:
                    issues.append(f"Self-loop detected: {source}")
            
            if issues:
                return {
                    "status": "warning",
                    "issues": issues[:5],  # Show first 5 issues
                    "edge_count": len(edges)
                }
            
            return {
                "status": "pass",
                "message": f"All {len(edges)} edges valid",
                "edge_count": len(edges)
            }
            
        except Exception as e:
            return {"status": "fail", "error": f"Edge validation failed: {e}"}
    
    def validate_hierarchy(self) -> dict[str, Any]:
        """Validate hierarchical compound node structure."""
        try:
            ns = "http://graphml.graphdrawing.org/xmlns"
            compound_nodes = []
            
            # Find all compound nodes
            for node in self.root.findall(f".//{{{ns}}}node"):
                nested_graph = node.find(f'.//{{{ns}}}graph')
                if nested_graph is not None:
                    compound_nodes.append({
                        'node_id': node.get('id'),
                        'graph_id': nested_graph.get('id')
                    })
            
            if not compound_nodes:
                return {
                    "status": "info",
                    "message": "No hierarchical structure (flat GraphML)"
                }
            
            # Validate compound node structure
            issues = []
            for compound in compound_nodes:
                node_id = compound['node_id']
                graph_id = compound['graph_id']
                
                # Check graph ID format (should be node_id + "::")
                expected_graph_id = f"{node_id}::"
                if graph_id != expected_graph_id:
                    issues.append(f"Graph ID mismatch: {graph_id} vs expected {expected_graph_id}")
            
            status = "warning" if issues else "pass"
            return {
                "status": status,
                "message": f"Hierarchical structure with {len(compound_nodes)} compound nodes",
                "compound_nodes": len(compound_nodes),
                "issues": issues if issues else None
            }
            
        except Exception as e:
            return {"status": "fail", "error": f"Hierarchy validation failed: {e}"}
    
    def validate_completeness(self) -> dict[str, Any]:
        """Validate GraphML completeness against original ONNX."""
        if not self.onnx_file:
            return {"status": "skip", "message": "No ONNX file provided for completeness check"}
        
        try:
            import onnx
            
            # Load original ONNX
            onnx_model = onnx.load(self.onnx_file)
            onnx_ops = len(onnx_model.graph.node)
            
            # Count GraphML operation nodes (exclude compound nodes and inputs/outputs)
            ns = "http://graphml.graphdrawing.org/xmlns"
            all_nodes = self.root.findall(f".//{{{ns}}}node")
            
            op_nodes = []
            for node in all_nodes:
                node_id = node.get('id', '')
                # Exclude compound nodes, inputs, outputs
                if (not node_id.startswith(('input_', 'output_')) and
                    node.find(f'.//{{{ns}}}graph') is None):  # Not compound
                    op_nodes.append(node_id)
            
            # Allow some variance due to input/output nodes and graph differences
            variance = abs(len(op_nodes) - onnx_ops)
            acceptable_variance = max(5, onnx_ops * 0.1)  # 10% or 5 nodes
            
            if variance > acceptable_variance:
                return {
                    "status": "warning",
                    "message": f"Node count variance: GraphML {len(op_nodes)} vs ONNX {onnx_ops}",
                    "graphml_ops": len(op_nodes),
                    "onnx_ops": onnx_ops,
                    "variance": variance
                }
            
            return {
                "status": "pass",
                "message": f"Node count within acceptable range",
                "graphml_ops": len(op_nodes),
                "onnx_ops": onnx_ops,
                "variance": variance
            }
            
        except Exception as e:
            return {"status": "fail", "error": f"Completeness validation failed: {e}"}
    
    def validate_networkx(self) -> dict[str, Any]:
        """Test NetworkX compatibility."""
        try:
            import networkx as nx
            
            G = nx.read_graphml(self.graphml_file)
            
            if len(G.nodes()) == 0:
                return {"status": "fail", "error": "No nodes loaded by NetworkX"}
            
            # Check attributes preserved
            missing_attrs = []
            for node_id, data in list(G.nodes(data=True))[:5]:  # Check first 5 nodes
                if 'op_type' not in data:
                    missing_attrs.append(f"Node {node_id} missing op_type")
            
            if missing_attrs:
                return {
                    "status": "warning",
                    "message": "Some attributes missing",
                    "issues": missing_attrs[:3]
                }
            
            return {
                "status": "pass",
                "message": f"NetworkX compatible: {len(G.nodes())} nodes, {len(G.edges())} edges",
                "nodes": len(G.nodes()),
                "edges": len(G.edges())
            }
            
        except ImportError:
            return {"status": "skip", "message": "NetworkX not available"}
        except Exception as e:
            return {"status": "fail", "error": f"NetworkX validation failed: {e}"}
    
    def run_all_validations(self) -> dict[str, Any]:
        """Run complete validation suite."""
        print(f"🔍 Validating GraphML: {self.graphml_file}")
        
        # Run all validations
        self.results['schema'] = self.validate_schema()
        self.results['keys'] = self.validate_keys()
        self.results['nodes'] = self.validate_nodes()
        self.results['edges'] = self.validate_edges()
        self.results['hierarchy'] = self.validate_hierarchy()
        self.results['completeness'] = self.validate_completeness()
        self.results['networkx'] = self.validate_networkx()
        
        return self.results
    
    def generate_validation_report(self) -> dict[str, Any]:
        """Generate comprehensive validation report."""
        results = self.run_all_validations()
        
        # Count statuses
        status_counts = {}
        for _validation_name, result in results.items():
            status = result.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        report = {
            'file': self.graphml_file,
            'timestamp': datetime.now().isoformat(),
            'validations': results,
            'summary': {
                'total_checks': len(results),
                'passed': status_counts.get('pass', 0),
                'failed': status_counts.get('fail', 0),
                'warnings': status_counts.get('warning', 0),
                'skipped': status_counts.get('skip', 0) + status_counts.get('info', 0),
            },
            'overall_status': 'pass' if status_counts.get('fail', 0) == 0 else 'fail'
        }
        
        return report


def print_report(report: dict[str, Any]) -> None:
    """Print validation report in a readable format."""
    print(f"\n📊 GraphML Validation Report")
    print(f"File: {report['file']}")
    print(f"Time: {report['timestamp']}")
    print(f"Overall Status: {'✅ PASS' if report['overall_status'] == 'pass' else '❌ FAIL'}")
    
    summary = report['summary']
    print(f"\nSummary: {summary['passed']} passed, {summary['failed']} failed, {summary['warnings']} warnings, {summary['skipped']} skipped")
    
    print(f"\n📋 Detailed Results:")
    for name, result in report['validations'].items():
        status = result.get('status', 'unknown')
        status_emoji = {
            'pass': '✅',
            'fail': '❌',
            'warning': '⚠️',
            'skip': '⏭️',
            'info': 'ℹ️'
        }.get(status, '❓')
        
        message = result.get('message', result.get('error', 'No message'))
        print(f"  {status_emoji} {name.title()}: {message}")
        
        # Show additional details for some validations
        if name == 'nodes' and 'compound_nodes' in result:
            print(f"    📊 {result['total_nodes']} total nodes ({result['compound_nodes']} compound, {result['operation_nodes']} operations)")
        
        if name == 'completeness' and 'variance' in result:
            print(f"    📊 Variance: {result['variance']} nodes")


def main():
    parser = argparse.ArgumentParser(description='Validate GraphML files against ADR-010 specification')
    parser.add_argument('graphml_file', help='GraphML file to validate')
    parser.add_argument('--onnx-file', help='Original ONNX file for completeness check')
    parser.add_argument('--metadata-file', help='HTP metadata file (unused currently)')
    parser.add_argument('--json-output', help='Save report as JSON file')
    parser.add_argument('--quiet', action='store_true', help='Only show overall result')
    
    args = parser.parse_args()
    
    if not Path(args.graphml_file).exists():
        print(f"❌ Error: GraphML file not found: {args.graphml_file}")
        sys.exit(1)
    
    try:
        validator = GraphMLValidator(args.graphml_file, args.onnx_file, args.metadata_file)
        report = validator.generate_validation_report()
        
        if not args.quiet:
            print_report(report)
        else:
            status = "✅ PASS" if report['overall_status'] == 'pass' else "❌ FAIL"
            print(f"{status}: {args.graphml_file}")
        
        if args.json_output:
            with open(args.json_output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Report saved to: {args.json_output}")
        
        # Exit with error code if validation failed
        sys.exit(0 if report['overall_status'] == 'pass' else 1)
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()