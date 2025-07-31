"""
Test to replicate the BERT attention layer slice context issue.
This test creates a model structure similar to BERT attention layers
to understand why slice operations get root-level tags instead of 
specific submodule context tags.
"""

import json
import tempfile

import torch
import torch.nn as nn

from modelexport.hierarchy_exporter import HierarchyExporter


class BertSdpaSelfAttention(nn.Module):
    """Mimics the BertSdpaSelfAttention class that has slice operations."""
    
    def __init__(self, hidden_size=768, num_attention_heads=12):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
    
    def forward(self, hidden_states):
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Create Q, K, V
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = query_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        key_layer = key_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        value_layer = value_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        
        # THIS IS THE CRITICAL SLICE OPERATION THAT SHOULD BE TAGGED WITH THIS MODULE
        # In real BERT, this might be slicing for attention patterns
        query_slice = query_layer[:, 1:-1, :, :]  # Remove first and last tokens
        key_slice = key_layer[:, 1:-1, :, :]
        value_slice = value_layer[:, 1:-1, :, :]
        
        # Simple attention computation (not full BERT implementation)
        attention_scores = torch.matmul(query_slice, key_slice.transpose(-1, -2))
        attention_output = torch.matmul(attention_scores, value_slice)
        
        # Reshape back to match expected dimensions
        new_shape = attention_output.size()[:-2] + (self.all_head_size,)
        attention_output = attention_output.view(new_shape)
        
        # Pad back to original sequence length to match residual connection
        batch_size, reduced_seq_len = attention_output.shape[:2]
        original_seq_len = hidden_states.shape[1]
        
        # Create padding for the sliced tokens
        padding = torch.zeros(batch_size, original_seq_len - reduced_seq_len, self.all_head_size, 
                            dtype=attention_output.dtype, device=attention_output.device)
        attention_output = torch.cat([padding[:, :1], attention_output, padding[:, 1:]], dim=1)
        
        return attention_output


class BertAttention(nn.Module):
    """Container for BertSdpaSelfAttention."""
    
    def __init__(self, hidden_size=768, num_attention_heads=12):
        super().__init__()
        self.self = BertSdpaSelfAttention(hidden_size, num_attention_heads)
        self.output_dense = nn.Linear(hidden_size, hidden_size)
        self.output_layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_states):
        self_attention_output = self.self(hidden_states)
        attention_output = self.output_dense(self_attention_output)
        attention_output = self.output_layer_norm(attention_output + hidden_states)
        return attention_output


class BertLayer(nn.Module):
    """Single BERT layer containing attention."""
    
    def __init__(self, layer_id, hidden_size=768, num_attention_heads=12):
        super().__init__()
        self.layer_id = layer_id
        self.attention = BertAttention(hidden_size, num_attention_heads)
        self.intermediate = nn.Linear(hidden_size, hidden_size * 4)
        self.output = nn.Linear(hidden_size * 4, hidden_size)
        self.output_layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = torch.relu(intermediate_output)
        
        layer_output = self.output(intermediate_output)
        layer_output = self.output_layer_norm(layer_output + attention_output)
        
        return layer_output


class BertEncoder(nn.Module):
    """BERT encoder with multiple layers."""
    
    def __init__(self, num_layers=2, hidden_size=768, num_attention_heads=12):
        super().__init__()
        self.layer = nn.ModuleList([
            BertLayer(i, hidden_size, num_attention_heads) 
            for i in range(num_layers)
        ])
    
    def forward(self, hidden_states):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class BertPooler(nn.Module):
    """BERT pooler for classification."""
    
    def __init__(self, hidden_size=768):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden_states):
        # Take [CLS] token (first token)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = torch.tanh(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """Simplified BERT model structure."""
    
    def __init__(self, vocab_size=1000, hidden_size=768, num_attention_heads=12, num_layers=2):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.encoder = BertEncoder(num_layers, hidden_size, num_attention_heads)
        self.pooler = BertPooler(hidden_size)
    
    def forward(self, input_ids):
        hidden_states = self.embeddings(input_ids)
        sequence_output = self.encoder(hidden_states)
        pooled_output = self.pooler(sequence_output)
        return pooled_output


def test_bert_slice_context_issue():
    """Test the specific BERT attention slice context issue."""
    
    print("=== Testing BERT Attention Slice Context Issue ===")
    
    # Create BERT-like model
    model = BertModel(vocab_size=1000, hidden_size=768, num_attention_heads=12, num_layers=2)
    model.eval()
    
    # Create input (batch_size=2, seq_length=10)
    inputs = torch.randint(0, 1000, (2, 10))
    
    print(f"Model structure:")
    for name, module in model.named_modules():
        if name:  # Skip root
            print(f"  {name}: {module.__class__.__name__}")
    
    print(f"\nInput shape: {inputs.shape}")
    
    # Test with HTP strategy 
    exporter = HierarchyExporter(strategy="htp")
    
    with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
        print("\n--- Exporting BERT model ---")
        result = exporter.export(
            model=model,
            example_inputs=inputs,
            output_path=tmp.name
        )
        
        print(f"Export result: {result}")
        
        # Load and analyze ONNX model
        import onnx
        onnx_model = onnx.load(tmp.name)
        
        # Find all slice operations
        slice_nodes = [node for node in onnx_model.graph.node if node.op_type == 'Slice']
        print(f"\nONNX Slice nodes found: {len(slice_nodes)}")
        
        # Get tag mapping
        tag_mapping = exporter.get_tag_mapping()
        
        # Analyze slice operation tags
        slice_tagging_results = []
        for node_name, node_info in tag_mapping.items():
            if node_info.get('op_type') == 'Slice':
                tags = node_info.get('tags', [])
                slice_tagging_results.append({
                    'node_name': node_name,
                    'tags': tags,
                    'expected_context': 'BertSdpaSelfAttention',
                    'actual_context': 'root-level' if any(tag.strip('/') == 'BertModel' for tag in tags) else 'submodule'
                })
        
        print(f"\n=== SLICE OPERATION ANALYSIS ===")
        for result in slice_tagging_results:
            print(f"Node: {result['node_name']}")
            print(f"  Tags: {result['tags']}")
            
            # Check if tagged with root level
            is_root_level = any(tag == '/BertModel' or '/BertModel/BertPooler' in tag for tag in result['tags'])
            is_attention_level = any('BertSdpaSelfAttention' in tag for tag in result['tags'])
            
            if is_root_level and not is_attention_level:
                print(f"  ❌ ISSUE: Tagged with root-level context instead of attention submodule")
                print(f"     Expected: /BertModel/BertEncoder/BertLayer.X/BertAttention/BertSdpaSelfAttention")
                print(f"     Actual: {result['tags']}")
            elif is_attention_level:
                print(f"  ✅ CORRECT: Tagged with attention submodule context")
            else:
                print(f"  ? Unknown tagging pattern")
        
        # Load hierarchy metadata
        hierarchy_file = tmp.name.replace('.onnx', '_hierarchy.json')
        try:
            with open(hierarchy_file) as f:
                hierarchy_data = json.load(f)
            
            print(f"\n=== HIERARCHY METADATA ANALYSIS ===")
            htp_metadata = hierarchy_data.get('htp_metadata', {})
            slice_ops = htp_metadata.get('slice_operations', [])
            
            print(f"Slice operations tracked in metadata: {len(slice_ops)}")
            for i, slice_op in enumerate(slice_ops):
                print(f"  {i}: context='{slice_op.get('context', 'None')}', key='{slice_op.get('key', 'None')}'")
                
                # Check if context shows proper module hierarchy
                context = slice_op.get('context', '')
                if 'BertSdpaSelfAttention' in context:
                    print(f"    ✅ Slice tracked with correct context")
                else:
                    print(f"    ❌ Slice tracked with incorrect context: {context}")
        
        except FileNotFoundError:
            print("No hierarchy JSON file found")
        
        # Summary
        print(f"\n=== SUMMARY ===")
        print(f"Total slice nodes: {len(slice_tagging_results)}")
        
        correctly_tagged = sum(1 for r in slice_tagging_results 
                              if any('BertSdpaSelfAttention' in tag for tag in r['tags']))
        incorrectly_tagged = len(slice_tagging_results) - correctly_tagged
        
        print(f"Correctly tagged with attention context: {correctly_tagged}")
        print(f"Incorrectly tagged with root context: {incorrectly_tagged}")
        
        if incorrectly_tagged > 0:
            print("❌ CONFIRMED ISSUE: Slice operations are getting root-level tags")
        else:
            print("✅ Issue appears to be resolved")


def test_simple_attention_slice():
    """Test with just the attention module to isolate the issue."""
    
    print("\n\n=== Testing Isolated Attention Slice ===")
    
    attention = BertSdpaSelfAttention(hidden_size=768, num_attention_heads=12)
    attention.eval()
    
    # Input: (batch_size=2, seq_length=5, hidden_size=768)
    inputs = torch.randn(2, 5, 768)
    
    print(f"Isolated attention module test")
    print(f"Input shape: {inputs.shape}")
    
    exporter = HierarchyExporter(strategy="htp")
    
    with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
        result = exporter.export(
            model=attention,
            example_inputs=inputs,
            output_path=tmp.name
        )
        
        # Analyze slice operations
        tag_mapping = exporter.get_tag_mapping()
        slice_nodes = {name: info for name, info in tag_mapping.items() 
                      if info.get('op_type') == 'Slice'}
        
        print(f"Slice nodes in isolated attention: {len(slice_nodes)}")
        for name, info in slice_nodes.items():
            tags = info.get('tags', [])
            print(f"  {name}: {tags}")
            
            if any('BertSdpaSelfAttention' in tag for tag in tags):
                print(f"    ✅ Correctly tagged with attention context")
            else:
                print(f"    ❌ Missing attention context in tags")


if __name__ == "__main__":
    test_bert_slice_context_issue()
    test_simple_attention_slice()