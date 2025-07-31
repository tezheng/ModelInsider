#!/usr/bin/env python3
"""
Test built-in tracking with BERT-like model to validate cross-layer fix.
"""

import json

import torch
import torch.nn as nn

from modelexport.hierarchy_exporter import HierarchyExporter

print("Testing built-in tracking with BERT-like model...")

class BertLayer(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.attention = nn.Linear(hidden_size, hidden_size)  # Simplified attention
        self.intermediate = nn.Linear(hidden_size, hidden_size*2)
        self.output = nn.Linear(hidden_size*2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        # Simplified BERT layer
        attn = self.attention(x)
        x = x + attn  # Residual
        x = self.layer_norm(x)
        
        # Feed forward
        intermediate = torch.relu(self.intermediate(x))
        output = self.output(intermediate)
        x = x + output  # Residual
        x = self.layer_norm(x)
        return x

class BertLikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(1000, 128)
        # Use similar naming to real BERT
        self.encoder = nn.ModuleDict({
            'layer': nn.ModuleDict({
                '0': BertLayer(),
                '1': BertLayer()
            })
        })
        self.pooler = nn.Linear(128, 128)
    
    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        
        # Pass through layers (similar to BERT structure)
        x = self.encoder['layer']['0'](x)
        x = self.encoder['layer']['1'](x)
        
        # Simple pooling
        x = x.mean(dim=1)
        x = self.pooler(x)
        return x

# Test the model
model = BertLikeModel()
input_ids = torch.randint(0, 1000, (1, 10))

print("\n=== OLD APPROACH (Regular HTP) ===")
exporter_old = HierarchyExporter(strategy='htp')
exporter_old._use_builtin_module_tracking = False  # Force old approach
result_old = exporter_old.export(model, input_ids, 'temp/bert_like_old.onnx')

with open('temp/bert_like_old_hierarchy.json') as f:
    hierarchy_old = json.load(f)

print(f"Strategy: {hierarchy_old['exporter']['strategy']}")
print("Tag statistics:")
for tag, count in hierarchy_old['tag_statistics'].items():
    print(f"  {tag}: {count}")

print("\nLooking for layer-specific operations:")
layer_0_ops = []
layer_1_ops = []
for node_name, node_info in hierarchy_old['node_tags'].items():
    if 'layer.0' in node_name or 'layer/0' in node_name:
        layer_0_ops.append((node_name, node_info.get('tags', [])))
    elif 'layer.1' in node_name or 'layer/1' in node_name:
        layer_1_ops.append((node_name, node_info.get('tags', [])))

print(f"Layer 0 operations: {len(layer_0_ops)}")
for name, tags in layer_0_ops[:3]:  # Show first 3
    print(f"  {name}: {tags}")

print(f"Layer 1 operations: {len(layer_1_ops)}")
for name, tags in layer_1_ops[:3]:  # Show first 3
    print(f"  {name}: {tags}")

print("\n=== NEW APPROACH (Built-in Tracking) ===")
exporter_new = HierarchyExporter(strategy='htp')
exporter_new._use_builtin_module_tracking = True  # Force new approach
result_new = exporter_new.export(model, input_ids, 'temp/bert_like_new.onnx')

with open('temp/bert_like_new_hierarchy.json') as f:
    hierarchy_new = json.load(f)

print(f"Strategy: {hierarchy_new['exporter']['strategy']}")
print(f"Built-in tracking: {hierarchy_new['summary'].get('builtin_tracking_enabled', False)}")
print("Tag statistics:")
for tag, count in hierarchy_new['tag_statistics'].items():
    print(f"  {tag}: {count}")

print("\nLooking for layer-specific operations:")
layer_0_ops_new = []
layer_1_ops_new = []
for node_name, node_info in hierarchy_new['node_tags'].items():
    tags = node_info.get('tags', [])
    if 'layer.0' in node_name or 'layer/0' in node_name:
        layer_0_ops_new.append((node_name, tags))
    elif 'layer.1' in node_name or 'layer/1' in node_name:
        layer_1_ops_new.append((node_name, tags))

print(f"Layer 0 operations: {len(layer_0_ops_new)}")
for name, tags in layer_0_ops_new[:3]:  # Show first 3
    print(f"  {name}: {tags}")

print(f"Layer 1 operations: {len(layer_1_ops_new)}")  
for name, tags in layer_1_ops_new[:3]:  # Show first 3
    print(f"  {name}: {tags}")

print("\n=== CROSS-LAYER CONTAMINATION CHECK ===")

def check_bert_contamination(hierarchy, approach_name):
    """Check for BERT-style cross-layer contamination."""
    contamination_found = False
    
    for node_name, node_info in hierarchy['node_tags'].items():
        tags = node_info.get('tags', [])
        
        # Check if layer.1 operations have layer.0 tags or vice versa
        if 'layer.1' in node_name:
            for tag in tags:
                if 'layer.0' in tag or 'Layer.0' in tag:
                    print(f"  {approach_name}: CONTAMINATION - {node_name} has layer.0 tag: {tag}")
                    contamination_found = True
        elif 'layer.0' in node_name:
            for tag in tags:
                if 'layer.1' in tag or 'Layer.1' in tag:
                    print(f"  {approach_name}: CONTAMINATION - {node_name} has layer.1 tag: {tag}")
                    contamination_found = True
    
    if not contamination_found:
        print(f"  {approach_name}: No cross-layer contamination detected ✅")

check_bert_contamination(hierarchy_old, "Old approach")
check_bert_contamination(hierarchy_new, "New approach")

print(f"\nOld approach trace length: {result_old['operation_trace_length']}")
print(f"New approach trace length: {result_new['operation_trace_length']}")
print(f"Old approach tagged: {result_old['tagged_operations']}")
print(f"New approach tagged: {result_new['tagged_operations']}")