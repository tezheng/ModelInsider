#!/usr/bin/env python3
"""
Quick test to load BERT and verify export works.
"""

import torch
import torch.nn as nn
from modelexport.hierarchy_exporter import HierarchyExporter

print("Testing with a simple model that mimics BERT structure...")

class SimpleBertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(128, 2, batch_first=True)
        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(256, 128)
        self.norm = nn.LayerNorm(128)
    
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = self.norm(x)
        
        ff_out = self.linear2(torch.relu(self.linear1(x)))
        x = x + ff_out
        x = self.norm(x)
        return x

class SimpleBertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 128)
        self.layer_0 = SimpleBertLayer()
        self.layer_1 = SimpleBertLayer()
        self.pooler = nn.Linear(128, 128)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.layer_0(x)
        x = self.layer_1(x)
        # Simple pooling to avoid ONNX slice issues
        x = x.mean(dim=1)  # Average pooling instead of first token
        return self.pooler(x)

# Test the model
model = SimpleBertModel()
input_ids = torch.randint(0, 1000, (1, 10))

print("Testing HTP export...")
exporter = HierarchyExporter(strategy='htp')
result = exporter.export(model, input_ids, 'temp/simple_bert_test.onnx')
print("Export completed!")

# Check for layer tagging issues
hierarchy_file = 'temp/simple_bert_test_hierarchy.json'
import json
with open(hierarchy_file, 'r') as f:
    hierarchy = json.load(f)

print("\nTag statistics:")
for tag, count in hierarchy['tag_statistics'].items():
    print(f"  {tag}: {count}")

print("\nLooking for cross-layer contamination...")
for node_name, node_info in hierarchy['node_tags'].items():
    tags = node_info.get('tags', [])
    if len(tags) > 1:
        # Check if tags contain references to different layers
        layer_0_tags = [tag for tag in tags if 'layer_0' in tag.lower()]
        layer_1_tags = [tag for tag in tags if 'layer_1' in tag.lower()]
        
        if layer_0_tags and layer_1_tags:
            print(f"CROSS-LAYER CONTAMINATION: {node_name}")
            print(f"  Layer 0 tags: {layer_0_tags}")
            print(f"  Layer 1 tags: {layer_1_tags}")