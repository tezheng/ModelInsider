#!/usr/bin/env python3
"""
Test the built-in tracking approach with a simple model.
"""

import torch
import torch.nn as nn

from modelexport.hierarchy_exporter import HierarchyExporter

print("Testing built-in tracking with simple model...")

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

print("Testing built-in tracking approach...")
exporter = HierarchyExporter(strategy='htp')
print(f"Built-in tracking enabled: {exporter._use_builtin_module_tracking}")

result = exporter.export(model, inputs, 'temp/builtin_test.onnx')
print("Export completed!")

# Check the hierarchy file
hierarchy_file = 'temp/builtin_test_hierarchy.json'
import json

with open(hierarchy_file) as f:
    hierarchy = json.load(f)

print(f"\nExporter strategy: {hierarchy['exporter']['strategy']}")
print(f"Built-in tracking: {hierarchy['summary'].get('builtin_tracking_enabled', False)}")

print("\nTag statistics:")
for tag, count in hierarchy['tag_statistics'].items():
    print(f"  {tag}: {count}")

print("\nChecking for proper layer differentiation...")
for node_name, node_info in hierarchy['node_tags'].items():
    tags = node_info.get('tags', [])
    if tags:
        print(f"{node_name}: {tags}")