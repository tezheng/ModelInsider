"""
Test to understand the timing issue with slice operations in BERT attention.
This test attempts to reproduce the exact sequence that causes incorrect tagging.
"""

import tempfile

import torch
import torch.nn as nn

from modelexport.hierarchy_exporter import HierarchyExporter


def create_context_debug_exporter():
    """Create an exporter with extensive debugging for context tracking."""
    
    exporter = HierarchyExporter(strategy="htp")
    
    # Store original methods
    original_patch_getitem = exporter._patch_tensor_getitem
    original_get_current_tag = exporter.get_current_tag
    
    # Counter for tracking execution order
    execution_counter = {'count': 0}
    
    def debug_get_current_tag():
        tag = original_get_current_tag()
        execution_counter['count'] += 1
        print(f"[{execution_counter['count']:03d}] get_current_tag() -> '{tag}' (stack: {exporter._tag_stack})")
        return tag
    
    def debug_patch_tensor_getitem():
        """Enhanced __getitem__ patch with detailed debugging."""
        if exporter._original_getitem is None:
            exporter._original_getitem = torch.Tensor.__getitem__
            
            def debug_context_aware_getitem(tensor_self, key):
                is_slice = exporter._is_slice_operation(key)
                
                if is_slice:
                    execution_counter['count'] += 1
                    print(f"\n[{execution_counter['count']:03d}] === SLICE OPERATION ===")
                    print(f"  Tensor shape: {tensor_self.shape}")
                    print(f"  Key: {key}")
                    print(f"  Current stack: {exporter._tag_stack}")
                    
                    # Get current context
                    current_tag = debug_get_current_tag()
                    print(f"  Captured context: '{current_tag}'")
                    
                    # Record slice operation
                    if current_tag:
                        slice_info = {
                            'tensor_id': id(tensor_self),
                            'key': str(key),
                            'context': current_tag,
                            'order': len(exporter._slice_operations),
                            'type': 'slice',
                            'execution_order': execution_counter['count']
                        }
                        exporter._slice_operations.append(slice_info)
                        print(f"  Recorded: order={slice_info['order']}, exec_order={slice_info['execution_order']}")
                    
                    print(f"[{execution_counter['count']:03d}] === END SLICE ===\n")
                
                return exporter._original_getitem(tensor_self, key)
            
            torch.Tensor.__getitem__ = debug_context_aware_getitem
    
    # Replace methods
    exporter.get_current_tag = debug_get_current_tag
    exporter._patch_tensor_getitem = debug_patch_tensor_getitem
    
    return exporter


class MockBertSelfAttention(nn.Module):
    """Mock BERT self-attention that reproduces the problematic slice pattern."""
    
    def __init__(self, hidden_size=768, num_attention_heads=12):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
    def forward(self, hidden_states):
        print(f"[MODEL] Entering MockBertSelfAttention.forward")
        
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Generate Q, K, V
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        # Reshape for attention heads
        query_layer = query_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        key_layer = key_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        value_layer = value_layer.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        
        # Transpose for attention computation
        query_layer = query_layer.transpose(1, 2)  # [batch, heads, seq, head_dim]
        key_layer = key_layer.transpose(1, 2)
        value_layer = value_layer.transpose(1, 2)
        
        print(f"[MODEL] About to perform attention slicing...")
        
        # CRITICAL SLICE OPERATIONS - these should be tagged with this module
        # Simulate what happens in scaled_dot_product_attention or similar
        query_slice = query_layer[:, :, 1:-1, :]  # Remove first and last tokens
        key_slice = key_layer[:, :, 1:-1, :]
        value_slice = value_layer[:, :, 1:-1, :]
        
        print(f"[MODEL] Slice operations completed")
        
        # Simple attention computation
        attention_scores = torch.matmul(query_slice, key_slice.transpose(-1, -2))
        attention_probs = torch.softmax(attention_scores, dim=-1)
        context_layer = torch.matmul(attention_probs, value_slice)
        
        # Reshape back
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        attention_output = context_layer.view(new_shape)
        
        print(f"[MODEL] Exiting MockBertSelfAttention.forward")
        return attention_output


class MockBertAttention(nn.Module):
    """Mock BERT attention wrapper."""
    
    def __init__(self, hidden_size=768, num_attention_heads=12):
        super().__init__()
        self.self = MockBertSelfAttention(hidden_size, num_attention_heads)
        self.output = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states):
        print(f"[MODEL] Entering MockBertAttention.forward")
        self_attention_output = self.self(hidden_states)
        attention_output = self.output(self_attention_output)
        print(f"[MODEL] Exiting MockBertAttention.forward")
        return attention_output


class MockBertLayer(nn.Module):
    """Mock BERT layer."""
    
    def __init__(self, layer_id, hidden_size=768, num_attention_heads=12):
        super().__init__()
        self.layer_id = layer_id
        self.attention = MockBertAttention(hidden_size, num_attention_heads)
        
    def forward(self, hidden_states):
        print(f"[MODEL] Entering MockBertLayer.{self.layer_id}.forward")
        attention_output = self.attention(hidden_states)
        print(f"[MODEL] Exiting MockBertLayer.{self.layer_id}.forward")
        return attention_output


class MockBertEncoder(nn.Module):
    """Mock BERT encoder."""
    
    def __init__(self, num_layers=2, hidden_size=768, num_attention_heads=12):
        super().__init__()
        self.layer = nn.ModuleList([
            MockBertLayer(i, hidden_size, num_attention_heads) 
            for i in range(num_layers)
        ])
        
    def forward(self, hidden_states):
        print(f"[MODEL] Entering MockBertEncoder.forward")
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        print(f"[MODEL] Exiting MockBertEncoder.forward")
        return hidden_states


class MockBertModel(nn.Module):
    """Mock BERT model that reproduces the slice tagging issue."""
    
    def __init__(self, vocab_size=1000, hidden_size=768, num_attention_heads=12, num_layers=2):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.encoder = MockBertEncoder(num_layers, hidden_size, num_attention_heads)
        # Add pooler to see if it affects slice tagging
        self.pooler = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input_ids):
        print(f"[MODEL] Entering MockBertModel.forward")
        
        hidden_states = self.embeddings(input_ids)
        print(f"[MODEL] Embeddings computed")
        
        sequence_output = self.encoder(hidden_states)
        print(f"[MODEL] Encoder completed")
        
        # Pooler operation (might interfere with attention slicing)
        pooled_output = self.pooler(sequence_output[:, 0, :])
        print(f"[MODEL] Pooler completed")
        
        print(f"[MODEL] Exiting MockBertModel.forward")
        return pooled_output


def test_bert_slice_timing_issue():
    """Test to understand the timing of slice operations in BERT-like models."""
    
    print("=== Testing BERT Slice Timing Issue ===")
    
    model = MockBertModel(vocab_size=1000, hidden_size=768, num_attention_heads=12, num_layers=2)
    model.eval()
    
    # Small input for easier debugging
    inputs = torch.randint(0, 1000, (1, 6))  # batch=1, seq_len=6
    
    print(f"Model structure:")
    for name, module in model.named_modules():
        if name and name.count('.') <= 4:  # Limit depth
            print(f"  {name}: {module.__class__.__name__}")
    
    print(f"\nInput shape: {inputs.shape}")
    
    # Create debug exporter
    exporter = create_context_debug_exporter()
    
    with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
        print(f"\n--- Starting export with timing debugging ---")
        
        result = exporter.export(
            model=model,
            example_inputs=inputs,
            output_path=tmp.name
        )
        
        print(f"\n--- Export completed ---")
        print(f"Export result: {result}")
        
        # Analyze captured slice operations
        print(f"\n=== SLICE OPERATION ANALYSIS ===")
        print(f"Total slice operations captured: {len(exporter._slice_operations)}")
        
        for i, slice_op in enumerate(exporter._slice_operations):
            context = slice_op['context']
            key = slice_op['key']
            exec_order = slice_op.get('execution_order', 'unknown')
            
            print(f"  {i}: exec_order={exec_order}, context='{context}'")
            print(f"     key='{key}'")
            
            # Check if it's an attention slice
            if 'MockBertSelfAttention' in context:
                print(f"     ✅ Attention slice correctly captured")
            elif context.strip('/') == 'MockBertModel':
                print(f"     ❌ Root-level context (likely pooler slice)")
            else:
                print(f"     ? Other context: {context}")
        
        # Check ONNX slice node tagging
        tag_mapping = exporter.get_tag_mapping()
        slice_nodes = {name: info for name, info in tag_mapping.items() 
                      if info.get('op_type') == 'Slice'}
        
        print(f"\n=== ONNX SLICE NODE TAGGING ===")
        for node_name, node_info in slice_nodes.items():
            tags = node_info.get('tags', [])
            print(f"  {node_name}: {tags}")
            
            # Check if correctly tagged
            if any('MockBertSelfAttention' in tag for tag in tags):
                print(f"    ✅ Correctly tagged with attention context")
            elif any(tag.strip('/') in ['MockBertModel'] for tag in tags):
                print(f"    ❌ Incorrectly tagged with root context")
            else:
                print(f"    ? Other tagging pattern")
        
        # Load hierarchy metadata for additional analysis
        hierarchy_file = tmp.name.replace('.onnx', '_hierarchy.json')
        try:
            import json
            with open(hierarchy_file) as f:
                hierarchy_data = json.load(f)
            
            print(f"\n=== METADATA VALIDATION ===")
            htp_metadata = hierarchy_data.get('htp_metadata', {})
            metadata_slices = htp_metadata.get('slice_operations', [])
            
            print(f"Slice operations in metadata: {len(metadata_slices)}")
            print(f"Slice operations captured during export: {len(exporter._slice_operations)}")
            
            if len(metadata_slices) != len(exporter._slice_operations):
                print(f"❌ Mismatch between captured and stored slice operations!")
            else:
                print(f"✅ Slice operation count matches")
        
        except FileNotFoundError:
            print("No hierarchy JSON file found")


if __name__ == "__main__":
    test_bert_slice_timing_issue()