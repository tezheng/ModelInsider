#!/usr/bin/env python3
"""Validate GraphML structure and hierarchy preservation."""

import xml.etree.ElementTree as ET
from pathlib import Path


def validate_graphml_structure(graphml_path: str) -> dict:
    """Validate GraphML structure and return detailed analysis."""
    
    # Register namespace
    ET.register_namespace('', 'http://graphml.graphdrawing.org/xmlns')
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
    
    with open(graphml_path) as f:
        content = f.read()
    
    root = ET.fromstring(content)
    
    # Validate basic structure
    validation = {
        'valid_xml': True,
        'has_graphml_root': root.tag.endswith('graphml'),
        'key_definitions': [],
        'main_graph': None,
        'compound_nodes': [],
        'hierarchy_validation': [],
        'metadata_preservation': [],
        'edge_validation': []
    }
    
    # Check key definitions
    keys = root.findall('./g:key', ns)
    for key in keys:
        validation['key_definitions'].append({
            'id': key.get('id'),
            'for': key.get('for'),
            'name': key.get('attr.name'),
            'type': key.get('attr.type')
        })
    
    # Find main graph
    main_graphs = root.findall('./g:graph', ns)
    if main_graphs:
        main_graph = main_graphs[0]
        validation['main_graph'] = {
            'id': main_graph.get('id'),
            'edgedefault': main_graph.get('edgedefault')
        }
        
        # Analyze compound nodes
        compound_nodes = main_graph.findall('./g:node', ns)
        for node in compound_nodes:
            node_id = node.get('id')
            
            # Check if this is a compound node (has nested graph)
            nested_graphs = node.findall('./g:graph', ns)
            if nested_graphs:
                nested_graph = nested_graphs[0]
                
                # Count child nodes and edges in nested graph
                child_nodes = nested_graph.findall('./g:node', ns)
                child_edges = nested_graph.findall('./g:edge', ns)
                
                # Extract metadata
                data_elements = node.findall('./g:data', ns)
                metadata = {}
                for data in data_elements:
                    key = data.get('key')
                    value = data.text or ''
                    metadata[key] = value
                
                compound_info = {
                    'id': node_id,
                    'nested_graph_id': nested_graph.get('id'),
                    'child_nodes': len(child_nodes),
                    'child_edges': len(child_edges),
                    'metadata': metadata,
                    'has_hierarchy_tag': 'n1' in metadata,
                    'has_json_attributes': 'n2' in metadata,
                    'hierarchy_tag': metadata.get('n1', ''),
                    'op_type': metadata.get('n0', ''),
                    'name': metadata.get('n3', '')
                }
                
                validation['compound_nodes'].append(compound_info)
    
    return validation

def analyze_hierarchy_preservation(validation: dict) -> list:
    """Analyze how well hierarchy is preserved in the GraphML."""
    
    issues = []
    compound_nodes = validation['compound_nodes']
    
    # Check for expected embedding modules
    embedding_nodes = [n for n in compound_nodes if 'embedding' in n['id']]
    if len(embedding_nodes) < 3:
        issues.append(f"Expected at least 3 embedding nodes, found {len(embedding_nodes)}")
    
    # Validate unique hierarchy tags
    hierarchy_tags = [n['hierarchy_tag'] for n in compound_nodes if n['hierarchy_tag']]
    if len(hierarchy_tags) != len(set(hierarchy_tags)):
        duplicates = [tag for tag in set(hierarchy_tags) if hierarchy_tags.count(tag) > 1]
        issues.append(f"Duplicate hierarchy tags found: {duplicates}")
    
    # Check metadata completeness
    missing_metadata = [n['id'] for n in compound_nodes if not n['has_hierarchy_tag']]
    if missing_metadata:
        issues.append(f"Nodes missing hierarchy tags: {missing_metadata}")
    
    missing_json = [n['id'] for n in compound_nodes if not n['has_json_attributes']]
    if missing_json:
        issues.append(f"Nodes missing JSON attributes: {missing_json}")
    
    return issues

def main():
    """Run GraphML structure validation."""
    
    print("🔍 GraphML Structure Validation")
    print("=" * 50)
    
    # Test files to validate
    test_files = [
        "temp/bert-tiny-complete/model.graphml",
        "temp/benchmark_prajjwal1_bert-tiny/model.graphml"
    ]
    
    for graphml_path in test_files:
        if not Path(graphml_path).exists():
            print(f"❌ File not found: {graphml_path}")
            continue
            
        print(f"\n📁 Validating: {graphml_path}")
        
        try:
            validation = validate_graphml_structure(graphml_path)
            
            # Basic structure validation
            print(f"✅ Valid XML: {validation['valid_xml']}")
            print(f"✅ GraphML root: {validation['has_graphml_root']}")
            
            # Key definitions
            print(f"\n🔑 Key Definitions ({len(validation['key_definitions'])}):")
            for key in validation['key_definitions']:
                print(f"  {key['id']}: {key['name']} ({key['for']}, {key['type']})")
            
            # Main graph
            if validation['main_graph']:
                print(f"\n📊 Main Graph: {validation['main_graph']['id']}")
            
            # Compound nodes
            compound_count = len(validation['compound_nodes'])
            print(f"\n🏗️ Compound Nodes: {compound_count}")
            
            # Show first few compound nodes with details
            print("\n📋 Sample Compound Nodes:")
            for i, node in enumerate(validation['compound_nodes'][:5]):
                print(f"  {i+1}. {node['id']}:")
                print(f"     Hierarchy: {node['hierarchy_tag']}")
                print(f"     Op Type: {node['op_type']}")
                print(f"     Child nodes: {node['child_nodes']}")
                print(f"     Child edges: {node['child_edges']}")
            
            if compound_count > 5:
                print(f"     ... and {compound_count - 5} more")
            
            # Hierarchy preservation analysis
            issues = analyze_hierarchy_preservation(validation)
            
            print(f"\n🧪 Hierarchy Preservation Analysis:")
            if not issues:
                print("  ✅ All validation checks passed!")
            else:
                print(f"  ⚠️ Found {len(issues)} issues:")
                for issue in issues:
                    print(f"    - {issue}")
            
            # Embedding validation
            embedding_nodes = [n for n in validation['compound_nodes'] if 'embedding' in n['id']]
            print(f"\n🔤 Embedding Modules ({len(embedding_nodes)}):")
            for emb in embedding_nodes:
                print(f"  - {emb['id']}: {emb['hierarchy_tag']}")
            
            # Statistics
            nodes_with_metadata = sum(1 for n in validation['compound_nodes'] if n['has_hierarchy_tag'])
            coverage = (nodes_with_metadata / compound_count * 100) if compound_count > 0 else 0
            print(f"\n📈 Statistics:")
            print(f"  Compound nodes: {compound_count}")
            print(f"  With hierarchy tags: {nodes_with_metadata} ({coverage:.1f}%)")
            print(f"  With JSON attributes: {sum(1 for n in validation['compound_nodes'] if n['has_json_attributes'])}")
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()