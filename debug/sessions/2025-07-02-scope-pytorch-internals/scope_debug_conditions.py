#!/usr/bin/env python3
"""
Debug specific conditions that might affect scopeName() results.

The user is getting rich scope info like:
'transformers.models.bert.modeling_bert.BertModel::/transformers.models.bert.modeling_bert.BertEmbeddings::embeddings'

But my analysis shows empty scopes. Let's investigate potential differences.
"""

import torch
import torch.jit
from transformers import AutoModel
import os


def test_pytorch_version_effects():
    """Test if PyTorch version affects scope information."""
    
    print("=" * 80)
    print("PyTorch Environment Analysis")
    print("=" * 80)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"JIT available: {hasattr(torch, 'jit')}")
    
    # Check JIT compilation settings
    if hasattr(torch.jit, 'get_jit_operator_bailout_heuristic'):
        print(f"JIT bailout heuristic: {torch.jit.get_jit_operator_bailout_heuristic()}")
    
    # Check debug settings
    debug_vars = ['PYTORCH_JIT_LOG_LEVEL', 'TORCH_JIT_DISABLE', 'PYTORCH_DISABLE_JIT']
    for var in debug_vars:
        value = os.environ.get(var)
        if value:
            print(f"Environment {var}: {value}")


def test_manual_forward_hooks():
    """Test if using forward hooks during tracing affects scope information."""
    
    print(f"\n" + "=" * 80)
    print("Testing Forward Hooks During Tracing")
    print("=" * 80)
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    dummy_input = torch.randint(0, 1000, (1, 4))
    
    # Add forward hooks to capture module context
    hook_data = []
    
    def forward_hook(module, input, output):
        hook_data.append({
            'module': module.__class__.__name__,
            'name': str(module)[:50]
        })
    
    # Register hooks on all modules
    handles = []
    for name, module in model.named_modules():
        handle = module.register_forward_hook(forward_hook)
        handles.append(handle)
    
    print(f"Registered {len(handles)} forward hooks")
    
    try:
        # Trace with hooks active
        traced = torch.jit.trace(model, dummy_input, strict=False)
        nodes = list(traced.graph.nodes())
        
        print(f"Forward hook activations: {len(hook_data)}")
        print(f"Traced nodes: {len(nodes)}")
        
        # Check scopes
        scopes_found = 0
        for i, node in enumerate(nodes):
            scope = node.scopeName()
            if scope:
                scopes_found += 1
                print(f"  Node[{i}] {node.kind()}: '{scope}'")
        
        print(f"Nodes with scopes: {scopes_found}")
        
    finally:
        # Clean up hooks
        for handle in handles:
            handle.remove()


def test_trace_with_concrete_inputs():
    """Test tracing with different input types and shapes."""
    
    print(f"\n" + "=" * 80)
    print("Testing Different Input Configurations")
    print("=" * 80)
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    
    input_configs = [
        ("Random integers (1,4)", torch.randint(0, 1000, (1, 4))),
        ("Random integers (2,8)", torch.randint(0, 1000, (2, 8))),
        ("Sequential integers", torch.tensor([[1, 2, 3, 4]])),
        ("Zeros", torch.zeros((1, 4), dtype=torch.long)),
        ("Ones", torch.ones((1, 4), dtype=torch.long)),
    ]
    
    for config_name, input_tensor in input_configs:
        print(f"\n🔬 {config_name}:")
        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Input dtype: {input_tensor.dtype}")
        
        try:
            traced = torch.jit.trace(model, input_tensor, strict=False)
            nodes = list(traced.graph.nodes())
            
            scopes_found = 0
            for node in nodes:
                scope = node.scopeName()
                if scope:
                    scopes_found += 1
            
            print(f"  Nodes with scopes: {scopes_found}/{len(nodes)}")
            
            if scopes_found > 0:
                print(f"  ✅ FOUND SCOPES WITH THIS CONFIG!")
                
        except Exception as e:
            print(f"  ERROR: {e}")


def test_model_modifications():
    """Test if model modifications affect scope tracking."""
    
    print(f"\n" + "=" * 80)
    print("Testing Model Modifications")
    print("=" * 80)
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    dummy_input = torch.randint(0, 1000, (1, 4))
    
    modifications = [
        ("Original model", lambda m: m),
        ("Model.eval()", lambda m: m.eval()),
        ("Model.train()", lambda m: m.train()),
        ("torch.no_grad() context", lambda m: m),  # Will wrap in no_grad
    ]
    
    for mod_name, mod_func in modifications:
        print(f"\n🔬 {mod_name}:")
        
        try:
            # Apply modification
            if mod_name == "torch.no_grad() context":
                with torch.no_grad():
                    test_model = mod_func(model)
                    traced = torch.jit.trace(test_model, dummy_input, strict=False)
            else:
                test_model = mod_func(model)
                traced = torch.jit.trace(test_model, dummy_input, strict=False)
            
            nodes = list(traced.graph.nodes())
            
            scopes_found = 0
            example_scopes = []
            for node in nodes:
                scope = node.scopeName()
                if scope:
                    scopes_found += 1
                    if len(example_scopes) < 3:
                        example_scopes.append((node.kind(), scope))
            
            print(f"  Nodes with scopes: {scopes_found}/{len(nodes)}")
            
            for kind, scope in example_scopes:
                print(f"    {kind}: '{scope[:60]}{'...' if len(scope) > 60 else ''}'")
                
        except Exception as e:
            print(f"  ERROR: {e}")


def test_torch_jit_settings():
    """Test different torch.jit settings that might affect scope tracking."""
    
    print(f"\n" + "=" * 80)
    print("Testing torch.jit Settings")
    print("=" * 80)
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    dummy_input = torch.randint(0, 1000, (1, 4))
    
    # Test with different JIT settings
    settings = [
        ("Default settings", {}),
        ("Check inputs disabled", {'check_inputs': False}),
        ("Optimize disabled", {'optimize': False}),
    ]
    
    for setting_name, kwargs in settings:
        print(f"\n🔬 {setting_name}:")
        print(f"  Settings: {kwargs}")
        
        try:
            if kwargs:
                # Apply settings if torch.jit.trace supports them
                traced = torch.jit.trace(model, dummy_input, strict=False, **kwargs)
            else:
                traced = torch.jit.trace(model, dummy_input, strict=False)
            
            nodes = list(traced.graph.nodes())
            
            scopes_found = 0
            for node in nodes:
                scope = node.scopeName()
                if scope:
                    scopes_found += 1
                    if scopes_found <= 2:  # Show first 2 examples
                        print(f"    {node.kind()}: '{scope}'")
            
            print(f"  Total scopes found: {scopes_found}/{len(nodes)}")
            
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                print(f"  Setting not supported: {e}")
            else:
                print(f"  ERROR: {e}")
        except Exception as e:
            print(f"  ERROR: {e}")


def replicate_user_scenario():
    """Try to replicate the exact scenario where user got scope information."""
    
    print(f"\n" + "=" * 80)
    print("Attempting to Replicate User's Scenario")
    print("=" * 80)
    
    # The user's example suggests they got:
    # 'transformers.models.bert.modeling_bert.BertModel::/transformers.models.bert.modeling_bert.BertEmbeddings::embeddings'
    
    print("User reported scope format:")
    print("'transformers.models.bert.modeling_bert.BertModel::/transformers.models.bert.modeling_bert.BertEmbeddings::embeddings'")
    
    # This suggests the scope includes:
    # 1. Full class path of parent module
    # 2. :: separator  
    # 3. Full class path of current module
    # 4. :: separator
    # 5. Module name
    
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    dummy_input = torch.randint(0, 1000, (1, 4))
    
    print(f"\nModel hierarchy preview:")
    for name, module in list(model.named_modules())[:5]:
        print(f"  {name}: {module.__class__}")
    
    # Try different approaches to get this information
    approaches = [
        ("Direct trace", lambda: torch.jit.trace(model, dummy_input, strict=False)),
        ("Trace after forward", lambda: trace_after_forward(model, dummy_input)),
        ("Script then trace", lambda: script_then_trace(model, dummy_input)),
    ]
    
    for approach_name, approach_func in approaches:
        print(f"\n🔬 {approach_name}:")
        
        try:
            traced = approach_func()
            if traced is None:
                continue
                
            nodes = list(traced.graph.nodes())
            
            scope_count = 0
            for i, node in enumerate(nodes):
                scope = node.scopeName()
                if scope:
                    scope_count += 1
                    print(f"  [{i:2d}] {node.kind()}: '{scope}'")
                    
                    # Check if this matches user's pattern
                    if '::' in scope and 'transformers.models.bert' in scope:
                        print(f"       ✅ MATCHES USER PATTERN!")
            
            print(f"  Total scopes: {scope_count}")
            
        except Exception as e:
            print(f"  ERROR: {e}")


def trace_after_forward(model, dummy_input):
    """Try tracing after a forward pass."""
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)  # Do a forward pass first
    return torch.jit.trace(model, dummy_input, strict=False)


def script_then_trace(model, dummy_input):
    """Try scripting then tracing."""
    try:
        scripted = torch.jit.script(model)
        return scripted
    except:
        return None


if __name__ == "__main__":
    try:
        test_pytorch_version_effects()
        test_manual_forward_hooks()
        test_trace_with_concrete_inputs()
        test_model_modifications()
        test_torch_jit_settings()
        replicate_user_scenario()
        
        print(f"\n" + "=" * 80)
        print("INVESTIGATION SUMMARY")
        print("=" * 80)
        print("Attempted to identify conditions where scopeName() returns rich information.")
        print("If still no scopes found, this suggests environment or version differences.")
        print("User's example shows PyTorch can provide detailed scope information.")
        
    except Exception as e:
        print(f"Investigation failed: {e}")
        import traceback
        traceback.print_exc()