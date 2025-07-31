#!/usr/bin/env python3
"""
Test the new built-in tracking approach with real BERT model.
"""

import json
import time

import torch
import torch.nn as nn

from modelexport.hierarchy_exporter import HierarchyExporter


def test_with_cached_model():
    """Test using a pre-cached model to avoid loading timeouts."""
    print("Testing built-in tracking with cached simple BERT-style model...")
    
    # Create a more realistic BERT-like model with proper layer structure
    class BertSelfAttention(nn.Module):
        def __init__(self, hidden_size=128, num_heads=2):
            super().__init__()
            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)
            self.dropout = nn.Dropout(0.1)
        
        def forward(self, hidden_states):
            query = self.query(hidden_states)
            key = self.key(hidden_states)
            value = self.value(hidden_states)
            
            # Simplified attention computation
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
            attention_probs = torch.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            context = torch.matmul(attention_probs, value)
            return context

    class BertSelfOutput(nn.Module):
        def __init__(self, hidden_size=128):
            super().__init__()
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.LayerNorm = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(0.1)
        
        def forward(self, hidden_states, input_tensor):
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            return hidden_states

    class BertAttention(nn.Module):
        def __init__(self, hidden_size=128):
            super().__init__()
            self.self = BertSelfAttention(hidden_size)
            self.output = BertSelfOutput(hidden_size)
        
        def forward(self, hidden_states):
            self_output = self.self(hidden_states)
            attention_output = self.output(self_output, hidden_states)
            return attention_output

    class BertIntermediate(nn.Module):
        def __init__(self, hidden_size=128, intermediate_size=256):
            super().__init__()
            self.dense = nn.Linear(hidden_size, intermediate_size)
        
        def forward(self, hidden_states):
            return torch.relu(self.dense(hidden_states))

    class BertOutput(nn.Module):
        def __init__(self, hidden_size=128, intermediate_size=256):
            super().__init__()
            self.dense = nn.Linear(intermediate_size, hidden_size)
            self.LayerNorm = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(0.1)
        
        def forward(self, hidden_states, input_tensor):
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            return hidden_states

    class BertLayer(nn.Module):
        def __init__(self, hidden_size=128):
            super().__init__()
            self.attention = BertAttention(hidden_size)
            self.intermediate = BertIntermediate(hidden_size)
            self.output = BertOutput(hidden_size)
        
        def forward(self, hidden_states):
            attention_output = self.attention(hidden_states)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output

    class BertEncoder(nn.Module):
        def __init__(self, hidden_size=128, num_layers=2):
            super().__init__()
            self.layer = nn.ModuleList([BertLayer(hidden_size) for _ in range(num_layers)])
        
        def forward(self, hidden_states):
            for layer_module in self.layer:
                hidden_states = layer_module(hidden_states)
            return hidden_states

    class BertEmbeddings(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=128, max_position=512):
            super().__init__()
            self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
            self.position_embeddings = nn.Embedding(max_position, hidden_size)
            self.LayerNorm = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(0.1)
        
        def forward(self, input_ids):
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
            
            words_embeddings = self.word_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            
            embeddings = words_embeddings + position_embeddings
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings

    class BertModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = BertEmbeddings()
            self.encoder = BertEncoder()
        
        def forward(self, input_ids):
            embedding_output = self.embeddings(input_ids)
            encoder_output = self.encoder(embedding_output)
            return encoder_output

    return BertModel()

def run_comparison_test():
    """Run the comparison test between old and new approaches."""
    print("=== Real BERT-Style Model Comparison ===")
    
    model = test_with_cached_model()
    input_ids = torch.randint(0, 1000, (1, 20))  # Longer sequence
    
    print(f"Model structure preview:")
    print(f"- BertModel")
    print(f"  - embeddings: BertEmbeddings")
    print(f"  - encoder: BertEncoder")
    print(f"    - layer.0: BertLayer")
    print(f"      - attention: BertAttention (self + output)")
    print(f"      - intermediate: BertIntermediate") 
    print(f"      - output: BertOutput")
    print(f"    - layer.1: BertLayer (same structure)")
    
    results = {}
    
    # Test old approach
    print(f"\n=== OLD APPROACH (Regular HTP) ===")
    try:
        exporter_old = HierarchyExporter(strategy='htp')
        exporter_old._use_builtin_module_tracking = False
        
        start_time = time.time()
        result_old = exporter_old.export(model, input_ids, 'temp/real_bert_old.onnx')
        old_time = time.time() - start_time
        
        with open('temp/real_bert_old_hierarchy.json') as f:
            hierarchy_old = json.load(f)
        
        results['old'] = {
            'result': result_old,
            'hierarchy': hierarchy_old,
            'export_time': old_time
        }
        
        print(f"Export time: {old_time:.2f}s")
        print(f"Tagged operations: {result_old['tagged_operations']}")
        print(f"Trace length: {result_old['operation_trace_length']}")
        
    except Exception as e:
        print(f"Old approach failed: {e}")
        results['old'] = None
    
    # Test new approach  
    print(f"\n=== NEW APPROACH (Built-in Tracking) ===")
    try:
        exporter_new = HierarchyExporter(strategy='htp')
        exporter_new._use_builtin_module_tracking = True
        
        start_time = time.time()
        result_new = exporter_new.export(model, input_ids, 'temp/real_bert_new.onnx')
        new_time = time.time() - start_time
        
        with open('temp/real_bert_new_hierarchy.json') as f:
            hierarchy_new = json.load(f)
        
        results['new'] = {
            'result': result_new,
            'hierarchy': hierarchy_new,
            'export_time': new_time
        }
        
        print(f"Export time: {new_time:.2f}s")
        print(f"Tagged operations: {result_new['tagged_operations']}")
        print(f"Trace length: {result_new['operation_trace_length']}")
        
    except Exception as e:
        print(f"New approach failed: {e}")
        results['new'] = None
    
    return results

def analyze_cross_layer_contamination(results):
    """Analyze cross-layer contamination in both approaches."""
    print(f"\n=== CROSS-LAYER CONTAMINATION ANALYSIS ===")
    
    for approach_name, data in results.items():
        if data is None:
            print(f"{approach_name.upper()} approach: Failed to export")
            continue
            
        hierarchy = data['hierarchy']
        print(f"\n{approach_name.upper()} approach results:")
        
        # Show tag statistics
        print("Tag statistics:")
        tag_stats = hierarchy.get('tag_statistics', {})
        for tag, count in sorted(tag_stats.items()):
            print(f"  {tag}: {count}")
        
        # Look for contamination
        contamination_cases = []
        node_tags = hierarchy.get('node_tags', {})
        
        for node_name, node_info in node_tags.items():
            tags = node_info.get('tags', [])
            
            # Check if layer.0 operations have layer.1 tags or vice versa
            if any('layer.0' in part or '/0/' in part for part in node_name.split('/')):
                for tag in tags:
                    if any('layer.1' in part or 'Layer.1' in part or '/1/' in part for part in tag.split('/')):
                        contamination_cases.append(f"Layer 0 op '{node_name}' has Layer 1 tag: {tag}")
            
            elif any('layer.1' in part or '/1/' in part for part in node_name.split('/')):
                for tag in tags:
                    if any('layer.0' in part or 'Layer.0' in part or '/0/' in part for part in tag.split('/')):
                        contamination_cases.append(f"Layer 1 op '{node_name}' has Layer 0 tag: {tag}")
        
        if contamination_cases:
            print(f"Cross-layer contamination found ({len(contamination_cases)} cases):")
            for case in contamination_cases[:5]:  # Show first 5
                print(f"  ❌ {case}")
            if len(contamination_cases) > 5:
                print(f"  ... and {len(contamination_cases) - 5} more cases")
        else:
            print("✅ No cross-layer contamination detected!")
    
    # Compare results
    if results.get('old') and results.get('new'):
        print(f"\n=== COMPARISON SUMMARY ===")
        old_tagged = results['old']['result']['tagged_operations']
        new_tagged = results['new']['result']['tagged_operations'] 
        old_trace = results['old']['result']['operation_trace_length']
        new_trace = results['new']['result']['operation_trace_length']
        
        print(f"Tagged operations: {old_tagged} (old) → {new_tagged} (new)")
        print(f"Trace length: {old_trace} (old) → {new_trace} (new)")
        print(f"Export time: {results['old']['export_time']:.2f}s (old) → {results['new']['export_time']:.2f}s (new)")

if __name__ == "__main__":
    try:
        results = run_comparison_test()
        analyze_cross_layer_contamination(results)
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()