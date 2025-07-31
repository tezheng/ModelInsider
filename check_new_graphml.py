#!/usr/bin/env python3
"""Check the newly generated GraphML file."""

import xml.etree.ElementTree as ET
from pathlib import Path

# Check if the new file exists
graphml_path = "temp/iteration7/model_hierarchical_graph.graphml"

if Path(graphml_path).exists():
    print(f"✅ Found GraphML file: {graphml_path}")
    
    # Register namespace
    ET.register_namespace('', 'http://graphml.graphdrawing.org/xmlns')
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
    
    with open(graphml_path) as f:
        content = f.read()
    
    root = ET.fromstring(content)
    
    # Find all compound nodes
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
            for data in data_elements:
                if data.get('key') == 'n1':
                    hierarchy_tag = data.text or ''
                    break
            
            compound_nodes.append({
                'id': node_id,
                'hierarchy_tag': hierarchy_tag
            })
    
    print(f"\nTotal compound nodes: {compound_count}")
    print("\nAll compound nodes:")
    for i, node in enumerate(compound_nodes, 1):
        print(f"  {i}. {node['id']}: {node['hierarchy_tag']}")
    
    # Check for embedding-related nodes
    embedding_nodes = [n for n in compound_nodes if 'embedding' in n['id'].lower()]
    print(f"\nEmbedding compound nodes: {len(embedding_nodes)}")
    for emb in embedding_nodes:
        print(f"  - {emb['id']}: {emb['hierarchy_tag']}")
        
else:
    print(f"❌ GraphML file not found: {graphml_path}")
    
    # List files in directory
    dir_path = Path("temp/iteration7/")
    if dir_path.exists():
        print(f"\nFiles in {dir_path}:")
        for file in dir_path.iterdir():
            print(f"  - {file.name}")
    else:
        print("Directory doesn't exist")