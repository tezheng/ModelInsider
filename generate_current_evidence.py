#!/usr/bin/env python3
"""Generate current evidence with enhanced converter."""

import xml.etree.ElementTree as ET
from pathlib import Path

from modelexport.graphml.hierarchical_converter import EnhancedHierarchicalConverter
from modelexport.strategies.htp.htp_exporter import HTPExporter


def generate_enhanced_graphml():
    """Generate GraphML using our enhanced converter with all fixes."""
    
    output_dir = "temp/current_evidence/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    model_name = "prajjwal1/bert-tiny"
    onnx_path = f"{output_dir}model.onnx"
    metadata_path = f"{output_dir}model_htp_metadata.json"
    graphml_path = f"{output_dir}model.graphml"
    
    print("🔧 Step 1: Export with enhanced HTP (all torch.nn modules)")
    
    # Export with our enhanced settings - include ALL torch.nn modules
    exporter = HTPExporter(
        verbose=False,
        enable_reporting=False,
        embed_hierarchy_attributes=True,
        torch_module="all"  # Include ALL torch.nn modules to match baseline
    )
    
    stats = exporter.export(
        model_name_or_path=model_name,
        output_path=onnx_path
    )
    
    print(f"  Hierarchy modules: {stats['hierarchy_modules']}")
    print(f"  Tagged nodes: {stats['tagged_nodes']}")
    print(f"  Coverage: {stats['coverage_percentage']:.1f}%")
    
    print("\n🔧 Step 2: Generate GraphML with enhanced converter")
    
    # Use our enhanced converter with hybrid hierarchy (structural + traced)
    converter = EnhancedHierarchicalConverter(
        htp_metadata_path=metadata_path,
        use_hybrid_hierarchy=True  # Enable structural discovery to reach 44 compound nodes
    )
    graphml_content = converter.convert(onnx_path)
    
    with open(graphml_path, 'w', encoding='utf-8') as f:
        f.write(graphml_content)
    
    print(f"  GraphML saved to: {graphml_path}")
    
    # Analyze the generated GraphML
    print("\n📊 Step 3: Analyze generated GraphML")
    
    ET.register_namespace('', 'http://graphml.graphdrawing.org/xmlns')
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
    
    root = ET.fromstring(graphml_content)
    
    # Count compound nodes
    all_nodes = root.findall(".//g:node", ns)
    compound_count = 0
    compound_nodes = []
    
    for node in all_nodes:
        nested_graphs = node.findall("./g:graph", ns)
        if nested_graphs:
            compound_count += 1
            node_id = node.get('id')
            
            # Get hierarchy tag
            data_elements = node.findall("./g:data", ns)
            hierarchy_tag = ""
            op_type = ""
            for data in data_elements:
                if data.get('key') == 'n1':
                    hierarchy_tag = data.text or ''
                elif data.get('key') == 'n0':
                    op_type = data.text or ''
            
            compound_nodes.append({
                'id': node_id,
                'hierarchy_tag': hierarchy_tag,
                'op_type': op_type
            })
    
    print(f"  Total compound nodes: {compound_count}")
    
    # Group by type
    from collections import defaultdict
    by_type = defaultdict(list)
    for node in compound_nodes:
        by_type[node['op_type']].append(node)
    
    print(f"\n📋 Compound nodes by type:")
    for op_type, nodes in sorted(by_type.items()):
        print(f"  {op_type}: {len(nodes)} nodes")
        for node in nodes[:3]:  # Show first 3
            print(f"    - {node['id']}: {node['hierarchy_tag']}")
        if len(nodes) > 3:
            print(f"    ... and {len(nodes) - 3} more")
    
    # Check for embeddings specifically
    embedding_nodes = [n for n in compound_nodes if 'embedding' in n['id'].lower() or n['op_type'] == 'Embedding']
    print(f"\n🔤 Embedding-related nodes: {len(embedding_nodes)}")
    for emb in embedding_nodes:
        print(f"  - {emb['id']} ({emb['op_type']}): {emb['hierarchy_tag']}")
    
    return {
        'compound_count': compound_count,
        'graphml_path': graphml_path,
        'metadata_path': metadata_path,
        'stats': stats,
        'compound_nodes': compound_nodes
    }

def compare_with_baseline(current_result):
    """Compare current result with baseline."""
    
    baseline_path = '/mnt/d/BYOM/modelexport/experiments/model_architecture_visualization/temp/bert-tiny-compound-nodes.graphml'
    
    print(f"\n📊 Step 4: Compare with baseline")
    
    if not Path(baseline_path).exists():
        print(f"❌ Baseline file not found: {baseline_path}")
        return
    
    # Load baseline
    ET.register_namespace('', 'http://graphml.graphdrawing.org/xmlns')
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
    
    with open(baseline_path) as f:
        baseline_content = f.read()
    
    baseline_root = ET.fromstring(baseline_content)
    
    # Count baseline compound nodes
    baseline_nodes = baseline_root.findall(".//g:node", ns)
    baseline_compound_count = 0
    baseline_compounds = []
    
    for node in baseline_nodes:
        nested_graphs = node.findall("./g:graph", ns)
        if nested_graphs:
            baseline_compound_count += 1
            baseline_compounds.append(node.get('id'))
    
    print(f"  Baseline compound nodes: {baseline_compound_count}")
    print(f"  Current compound nodes: {current_result['compound_count']}")
    print(f"  Difference: {current_result['compound_count'] - baseline_compound_count}")
    
    # Check overlap
    current_ids = {n['id'] for n in current_result['compound_nodes']}
    baseline_ids = set(baseline_compounds)
    
    overlap = current_ids & baseline_ids
    current_only = current_ids - baseline_ids
    baseline_only = baseline_ids - current_ids
    
    print(f"\n📈 Overlap analysis:")
    print(f"  Common nodes: {len(overlap)}")
    print(f"  Current only: {len(current_only)}")
    print(f"  Baseline only: {len(baseline_only)}")
    
    if current_only:
        print(f"\n➕ Current only (first 10):")
        for node_id in sorted(list(current_only)[:10]):
            print(f"    - {node_id}")
    
    if baseline_only:
        print(f"\n➖ Baseline only (first 10):")
        for node_id in sorted(list(baseline_only)[:10]):
            print(f"    - {node_id}")

if __name__ == "__main__":
    print("🚀 Generating Current Evidence with Enhanced Converter")
    print("=" * 60)
    
    result = generate_enhanced_graphml()
    compare_with_baseline(result)
    
    print(f"\n✅ Evidence generation complete!")
    print(f"📁 Files generated:")
    print(f"  - GraphML: {result['graphml_path']}")
    print(f"  - Metadata: {result['metadata_path']}")
    print(f"  - Compound nodes: {result['compound_count']}")