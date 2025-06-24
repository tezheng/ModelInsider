#!/usr/bin/env python3
"""
Compare old and new approaches for layer tagging.
"""

import torch
import torch.nn as nn
from modelexport.hierarchy_exporter import HierarchyExporter
import json

print("Comparing old vs new approach for layer tagging...")

class LayeredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_0 = nn.Linear(4, 8)
        self.layer_1 = nn.Linear(8, 2)
    
    def forward(self, x):
        x = self.layer_0(x)
        x = torch.relu(x)
        x = self.layer_1(x)
        return x

# Test the model
model = LayeredModel()
inputs = torch.randn(1, 4)

print("\n=== OLD APPROACH (Regular HTP) ===")
exporter_old = HierarchyExporter(strategy='htp')
exporter_old._use_builtin_module_tracking = False  # Force old approach
result_old = exporter_old.export(model, inputs, 'temp/old_approach_test.onnx')

with open('temp/old_approach_test_hierarchy.json', 'r') as f:
    hierarchy_old = json.load(f)

print(f"Strategy: {hierarchy_old['exporter']['strategy']}")
print("Tag statistics:")
for tag, count in hierarchy_old['tag_statistics'].items():
    print(f"  {tag}: {count}")

print("Node tags:")
for node_name, node_info in hierarchy_old['node_tags'].items():
    tags = node_info.get('tags', [])
    if tags:
        print(f"  {node_name}: {tags}")

print("\n=== NEW APPROACH (Built-in Tracking) ===")
exporter_new = HierarchyExporter(strategy='htp')
exporter_new._use_builtin_module_tracking = True  # Force new approach
result_new = exporter_new.export(model, inputs, 'temp/new_approach_test.onnx')

with open('temp/new_approach_test_hierarchy.json', 'r') as f:
    hierarchy_new = json.load(f)

print(f"Strategy: {hierarchy_new['exporter']['strategy']}")
print(f"Built-in tracking: {hierarchy_new['summary'].get('builtin_tracking_enabled', False)}")
print("Tag statistics:")
for tag, count in hierarchy_new['tag_statistics'].items():
    print(f"  {tag}: {count}")

print("Node tags:")
for node_name, node_info in hierarchy_new['node_tags'].items():
    tags = node_info.get('tags', [])
    if tags:
        print(f"  {node_name}: {tags}")

print("\n=== COMPARISON ===")
print("Cross-layer contamination check:")

def check_cross_layer_contamination(hierarchy, approach_name):
    """Check if layer operations are tagged with wrong layer contexts."""
    contamination_found = False
    for node_name, node_info in hierarchy['node_tags'].items():
        tags = node_info.get('tags', [])
        if len(tags) > 1:
            # Check if tags contain references to different layers
            layer_contexts = set()
            for tag in tags:
                if 'layer_0' in node_name.lower() or 'Layer0' in tag:
                    layer_contexts.add('layer_0')
                if 'layer_1' in node_name.lower() or 'Layer1' in tag:
                    layer_contexts.add('layer_1')
            
            if len(layer_contexts) > 1:
                print(f"  {approach_name}: CONTAMINATION in {node_name}: {tags}")
                contamination_found = True
    
    if not contamination_found:
        print(f"  {approach_name}: No cross-layer contamination detected ✅")

check_cross_layer_contamination(hierarchy_old, "Old approach")
check_cross_layer_contamination(hierarchy_new, "New approach")

print(f"\nOld approach tagged operations: {result_old['tagged_operations']}")
print(f"New approach tagged operations: {result_new['tagged_operations']}")
print(f"Old approach trace length: {result_old['operation_trace_length']}")
print(f"New approach trace length: {result_new['operation_trace_length']}")