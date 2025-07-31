#!/usr/bin/env python3
"""
Investigate scope mapping issues:
1. How to map to HF modules (not torch.nn modules)
2. Why some nodes have empty/minimal scope names
"""

from pathlib import Path

import onnx
import torch
from transformers import AutoModel, AutoTokenizer


def investigate_hf_vs_torch_modules():
    """Investigate the difference between HF modules and torch.nn modules."""
    
    print("🔍 Investigating HF vs torch.nn Module Mapping")
    print("="*60)
    
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    
    print("\n📊 HuggingFace Module Hierarchy:")
    hf_modules = []
    torch_modules = []
    
    for name, module in model.named_modules():
        if name:  # Skip root
            class_name = module.__class__.__name__
            module_path = module.__class__.__module__
            
            if 'transformers' in module_path:
                hf_modules.append((name, class_name, module_path))
            elif 'torch.nn' in module_path:
                torch_modules.append((name, class_name, module_path))
    
    print(f"\n🎯 HuggingFace-specific modules ({len(hf_modules)} found):")
    for name, class_name, module_path in hf_modules[:10]:
        print(f"  {name} → {class_name} ({module_path})")
    
    print(f"\n⚙️ torch.nn modules ({len(torch_modules)} found):")
    for name, class_name, module_path in torch_modules[:10]:
        print(f"  {name} → {class_name} ({module_path})")
    
    print(f"\n💡 Key Insight:")
    print(f"  • HF models use torch.nn modules as building blocks")
    print(f"  • HF-specific classes (like BertAttention) contain torch.nn modules")
    print(f"  • We need to map to HF semantic units, not low-level torch.nn components")
    
    return model


def investigate_empty_scope_nodes():
    """Investigate nodes with empty or minimal scope information."""
    
    print("\n🔍 Investigating Empty/Minimal Scope Nodes")
    print("="*60)
    
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    
    # Export to ONNX to see the actual node names
    inputs = tokenizer(["Test"], return_tensors="pt", max_length=8, padding=True, truncation=True)
    
    output_dir = Path("temp/scope_investigation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model, inputs['input_ids'], 
        output_dir / "investigate.onnx",
        verbose=False
    )
    
    onnx_model = onnx.load(str(output_dir / "investigate.onnx"))
    
    # Categorize nodes by scope patterns
    scoped_nodes = []
    minimal_scope_nodes = []
    empty_scope_nodes = []
    
    for node in onnx_model.graph.node:
        node_name = node.name
        
        if not node_name or node_name.startswith('/'):
            if node_name.count('/') >= 3:  # Has meaningful scope
                scoped_nodes.append(node)
            elif node_name.count('/') <= 2:  # Minimal scope
                minimal_scope_nodes.append(node)
        else:
            empty_scope_nodes.append(node)  # No leading slash
    
    print(f"\n📊 Node categorization:")
    print(f"  • Well-scoped nodes: {len(scoped_nodes)}")
    print(f"  • Minimal scope nodes: {len(minimal_scope_nodes)}")
    print(f"  • Empty scope nodes: {len(empty_scope_nodes)}")
    
    print(f"\n🎯 Examples of minimal/empty scope nodes:")
    for node in minimal_scope_nodes[:5] + empty_scope_nodes[:5]:
        print(f"  {node.name} ({node.op_type})")
    
    print(f"\n🎯 Examples of well-scoped nodes:")
    for node in scoped_nodes[:5]:
        print(f"  {node.name} ({node.op_type})")
    
    return onnx_model


def analyze_scope_name_patterns():
    """Analyze the patterns in ONNX scope names to understand the mapping."""
    
    print("\n🔍 Analyzing Scope Name Patterns")
    print("="*60)
    
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    
    # Export with verbose to see scope information
    inputs = tokenizer(["Analyze scope patterns"], return_tensors="pt", max_length=8, padding=True, truncation=True)
    
    output_dir = Path("temp/scope_investigation")
    onnx_path = output_dir / "scope_analysis.onnx"
    
    print("\n🚀 Exporting with scope analysis...")
    torch.onnx.export(
        model, inputs['input_ids'], onnx_path,
        verbose=True  # This will show us the scope information during export
    )
    
    # Load and analyze
    onnx_model = onnx.load(str(onnx_path))
    
    # Group nodes by scope patterns
    scope_patterns = {}
    
    for node in onnx_model.graph.node:
        if '/' in node.name:
            # Extract scope path (everything except the last component)
            parts = node.name.strip('/').split('/')
            if len(parts) >= 2:
                scope_path = '/'.join(parts[:-1])
                operation = parts[-1]
                
                if scope_path not in scope_patterns:
                    scope_patterns[scope_path] = []
                scope_patterns[scope_path].append((operation, node.op_type))
    
    print(f"\n📊 Scope Pattern Analysis:")
    print(f"  Total unique scope paths: {len(scope_patterns)}")
    
    # Show patterns
    for scope_path, operations in list(scope_patterns.items())[:10]:
        print(f"\n  Scope: {scope_path}")
        op_summary = {}
        for _op_name, op_type in operations:
            op_summary[op_type] = op_summary.get(op_type, 0) + 1
        print(f"    Operations: {dict(op_summary)}")
    
    # Check if we can map scope paths to HF modules
    print(f"\n🔍 Mapping scope paths to HF modules:")
    
    hf_module_names = {name for name, _ in model.named_modules() if name}
    
    mapped_count = 0
    for scope_path in list(scope_patterns.keys())[:10]:
        # Try to convert scope path to HF module name
        # "/bert/encoder/layer.0/attention/self" -> "encoder.layer.0.attention.self"
        potential_hf_name = scope_path.split('/', 1)[-1].replace('/', '.')
        
        if potential_hf_name in hf_module_names:
            print(f"  ✅ {scope_path} → {potential_hf_name}")
            mapped_count += 1
        else:
            print(f"  ❌ {scope_path} → {potential_hf_name} (not found)")
    
    print(f"\n📈 Mapping success rate: {mapped_count}/{min(10, len(scope_patterns))}")


def investigate_node_generation_process():
    """Investigate how different types of nodes are generated during ONNX export."""
    
    print("\n🔍 Investigating Node Generation Process")
    print("="*60)
    
    # Create a simple model to understand the generation process
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)
            
        def forward(self, x):
            # Explicit operations that might generate nodes with different scopes
            y = self.linear(x)  # Should have scope
            z = y + 1  # Might not have scope (constant)
            w = torch.gather(z, 1, torch.tensor([[0], [1]]))  # Might be like /Gather_3
            return w
    
    simple_model = SimpleModel()
    simple_input = torch.randn(2, 4)
    
    print("\n🧪 Exporting simple model to understand node generation...")
    
    output_dir = Path("temp/scope_investigation")
    simple_onnx_path = output_dir / "simple_model.onnx"
    
    torch.onnx.export(
        simple_model, simple_input, simple_onnx_path,
        verbose=True
    )
    
    # Analyze the simple model
    simple_onnx = onnx.load(str(simple_onnx_path))
    
    print(f"\n📊 Simple model analysis:")
    for node in simple_onnx.graph.node:
        print(f"  {node.name} ({node.op_type})")
    
    print(f"\n💡 Insights:")
    print(f"  • Nodes from nn.Module operations get proper scope")
    print(f"  • Implicit operations (constants, reshapes) may get minimal scope")
    print(f"  • Operations on intermediate tensors may not inherit scope")


def propose_solutions():
    """Propose solutions for the identified issues."""
    
    print("\n🚀 Proposed Solutions")
    print("="*60)
    
    print("\n📍 Issue 1: Mapping to HF modules (not torch.nn)")
    print("Solution:")
    print("  1. Build HF semantic hierarchy mapping")
    print("  2. Map torch.nn modules to their containing HF modules")
    print("  3. Provide HF-level semantic labels")
    
    print("\n📍 Issue 2: Nodes with empty/minimal scope")
    print("Solutions:")
    print("  1. Fallback mapping: Use parent scope or operation context")
    print("  2. Operation classification: Categorize by operation type")
    print("  3. Data flow analysis: Trace tensor origins")
    print("  4. Pattern-based inference: Use naming patterns as fallback")
    
    print("\n💡 Combined Approach:")
    print("  1. Primary: Use scope-based mapping for well-scoped nodes")
    print("  2. Secondary: Use context inference for minimal-scope nodes")
    print("  3. Tertiary: Use operation classification for empty-scope nodes")
    print("  4. Always provide HF-level semantic information")


if __name__ == "__main__":
    model = investigate_hf_vs_torch_modules()
    onnx_model = investigate_empty_scope_nodes()
    analyze_scope_name_patterns()
    investigate_node_generation_process()
    propose_solutions()
    
    print(f"\n🎯 Investigation Complete!")
    print(f"Key findings saved in temp/scope_investigation/")