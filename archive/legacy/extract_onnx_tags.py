#!/usr/bin/env python3
"""
Universal ONNX Tag Extractor
Extract all hierarchy tags from any ONNX model without hardcoded assumptions
"""

import onnx
import json
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict


class ONNXTagExtractor:
    """Universal extractor for hierarchy tags from ONNX models"""
    
    def __init__(self, onnx_path: str):
        """Initialize with ONNX model path"""
        self.onnx_path = Path(onnx_path)
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load ONNX model"""
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        
        try:
            self.model = onnx.load(str(self.onnx_path))
            print(f"✅ Loaded ONNX model: {self.onnx_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def extract_all_tags(self) -> Dict:
        """Extract all hierarchy tags from the ONNX model"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        results = {
            "model_info": {
                "path": str(self.onnx_path),
                "producer_name": self.model.producer_name,
                "producer_version": self.model.producer_version,
                "ir_version": self.model.ir_version,
                "model_version": self.model.model_version,
                "total_nodes": len(self.model.graph.node),
                "total_inputs": len(self.model.graph.input),
                "total_outputs": len(self.model.graph.output),
                "total_initializers": len(self.model.graph.initializer)
            },
            "node_tags": {},
            "tag_statistics": defaultdict(int),
            "unique_tags": set(),
            "nodes_without_tags": [],
            "tag_hierarchy": {}
        }
        
        # Extract tags from each node
        for i, node in enumerate(self.model.graph.node):
            node_name = node.name if node.name else f"{node.op_type}_{i}"
            node_info = {
                "index": i,
                "name": node_name,
                "op_type": node.op_type,
                "input_count": len(node.input),
                "output_count": len(node.output),
                "tags": []
            }
            
            # Look for hierarchy tags in node attributes
            hierarchy_tags = self._extract_tags_from_node(node)
            if hierarchy_tags:
                node_info["tags"] = hierarchy_tags
                results["node_tags"][node_name] = node_info
                
                # Update statistics
                for tag in hierarchy_tags:
                    results["tag_statistics"][tag] += 1
                    results["unique_tags"].add(tag)
                    
                    # Build hierarchy structure
                    self._build_tag_hierarchy(tag, results["tag_hierarchy"])
            else:
                results["nodes_without_tags"].append(node_info)
        
        # Convert sets to lists for JSON serialization
        results["unique_tags"] = sorted(list(results["unique_tags"]))
        results["tag_statistics"] = dict(results["tag_statistics"])
        
        return results
    
    def _extract_tags_from_node(self, node) -> List[str]:
        """Extract hierarchy tags from a single node"""
        tags = []
        
        # Check node attributes for hierarchy information
        for attr in node.attribute:
            if attr.name in ["hierarchy_tag", "module_tag", "source_module", "module_hierarchy"]:
                if attr.type == onnx.AttributeProto.STRING:
                    tag = attr.s.decode('utf-8')
                    if tag:
                        tags.append(tag)
                elif attr.type == onnx.AttributeProto.STRINGS:
                    for tag_bytes in attr.strings:
                        tag = tag_bytes.decode('utf-8')
                        if tag:
                            tags.append(tag)
        
        # Check metadata_props for hierarchy information
        for prop in getattr(node, 'metadata_props', []):
            if 'hierarchy' in prop.key.lower() or 'module' in prop.key.lower():
                tags.append(prop.value)
        
        return tags
    
    def _build_tag_hierarchy(self, tag: str, hierarchy: Dict):
        """Build hierarchical structure from tag path"""
        parts = [part for part in tag.split('/') if part]
        current = hierarchy
        
        for i, part in enumerate(parts):
            if part not in current:
                current[part] = {
                    "children": {},
                    "full_path": '/' + '/'.join(parts[:i+1]),
                    "depth": i,
                    "node_count": 0
                }
            current[part]["node_count"] += 1
            current = current[part]["children"]
    
    def get_tags_by_pattern(self, pattern: str) -> List[str]:
        """Get all tags matching a pattern"""
        results = self.extract_all_tags()
        matching_tags = []
        
        pattern_lower = pattern.lower()
        for tag in results["unique_tags"]:
            if pattern_lower in tag.lower():
                matching_tags.append(tag)
        
        return matching_tags
    
    def get_nodes_by_tag(self, target_tag: str) -> List[Dict]:
        """Get all nodes with a specific tag"""
        results = self.extract_all_tags()
        matching_nodes = []
        
        for node_name, node_info in results["node_tags"].items():
            if target_tag in node_info["tags"]:
                matching_nodes.append(node_info)
        
        return matching_nodes
    
    def analyze_tag_coverage(self) -> Dict:
        """Analyze how many nodes have tags vs no tags"""
        results = self.extract_all_tags()
        
        total_nodes = results["model_info"]["total_nodes"]
        tagged_nodes = len(results["node_tags"])
        untagged_nodes = len(results["nodes_without_tags"])
        
        return {
            "total_nodes": total_nodes,
            "tagged_nodes": tagged_nodes,
            "untagged_nodes": untagged_nodes,
            "tag_coverage_percent": (tagged_nodes / total_nodes * 100) if total_nodes > 0 else 0,
            "unique_tag_count": len(results["unique_tags"])
        }
    
    def save_results(self, output_path: str, include_details: bool = True):
        """Save extraction results to JSON file"""
        results = self.extract_all_tags()
        
        if not include_details:
            # Simplified output for overview
            simplified = {
                "model_info": results["model_info"],
                "tag_statistics": results["tag_statistics"], 
                "unique_tags": results["unique_tags"],
                "coverage": self.analyze_tag_coverage()
            }
            results = simplified
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to: {output_path}")


def main():
    """CLI interface for ONNX tag extraction"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract hierarchy tags from ONNX models")
    parser.add_argument("onnx_path", help="Path to ONNX model file")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--pattern", "-p", help="Filter tags by pattern")
    parser.add_argument("--tag", "-t", help="Show nodes with specific tag")
    parser.add_argument("--summary", "-s", action="store_true", help="Show summary only")
    parser.add_argument("--detailed", "-d", action="store_true", help="Include detailed node information")
    
    args = parser.parse_args()
    
    try:
        extractor = ONNXTagExtractor(args.onnx_path)
        
        if args.pattern:
            # Filter by pattern
            matching_tags = extractor.get_tags_by_pattern(args.pattern)
            print(f"\n🔍 Tags matching pattern '{args.pattern}':")
            for tag in matching_tags:
                print(f"  {tag}")
        
        elif args.tag:
            # Show nodes with specific tag
            matching_nodes = extractor.get_nodes_by_tag(args.tag)
            print(f"\n🎯 Nodes with tag '{args.tag}':")
            for node in matching_nodes:
                print(f"  {node['name']} ({node['op_type']})")
        
        elif args.summary:
            # Show summary
            coverage = extractor.analyze_tag_coverage()
            print("\n📊 ONNX Tag Summary:")
            print(f"  Total nodes: {coverage['total_nodes']}")
            print(f"  Tagged nodes: {coverage['tagged_nodes']}")
            print(f"  Untagged nodes: {coverage['untagged_nodes']}")
            print(f"  Tag coverage: {coverage['tag_coverage_percent']:.1f}%")
            print(f"  Unique tags: {coverage['unique_tag_count']}")
        
        else:
            # Full extraction
            results = extractor.extract_all_tags()
            print("\n📋 All Unique Tags:")
            for tag in results["unique_tags"]:
                count = results["tag_statistics"][tag]
                print(f"  {tag} ({count} nodes)")
        
        # Save to file if requested
        if args.output:
            extractor.save_results(args.output, include_details=args.detailed)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())