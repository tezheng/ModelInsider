#!/usr/bin/env python3
"""
Detailed BERT-tiny Analysis: Expected Hierarchy + ONNX Node Tagging
==================================================================

This script provides:
1. Complete expected hierarchy structure
2. Actual ONNX nodes that should be tagged with hierarchy tags
3. Reference implementation for verification
"""

import json
import tempfile
from pathlib import Path

import onnx
import torch
from transformers import AutoModel, AutoTokenizer


def analyze_complete_hierarchy():
    """Analyze complete module hierarchy with expected tags."""
    
    print("1️⃣ EXPECTED HIERARCHY STRUCTURE")
    print("=" * 50)
    
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    
    expected_tags = {}
    torch_nn_filtered = []
    
    for name, module in model.named_modules():
        module_class = type(module).__name__
        module_path = type(module).__module__
        
        # Apply MUST-002 filtering
        if module_path.startswith('torch.nn'):
            # Whitelist from requirements: LayerNorm, Embedding, BatchNorm, etc.
            whitelist = {'LayerNorm', 'Embedding', 'BatchNorm1d', 'BatchNorm2d', 'GroupNorm', 'InstanceNorm'}
            if module_class not in whitelist:
                torch_nn_filtered.append((name, module_class))
                expected_tags[name] = ""  # Empty tag for filtered modules
                continue
        
        # Generate expected hierarchy tag
        if not name:  # Root module
            expected_tags[name] = "/BertModel"
        else:
            # Convert module path to hierarchy tag
            tag = convert_to_hierarchy_tag(name, module_class)
            expected_tags[name] = tag
    
    # Print expected hierarchy
    print("🏗️ Complete Expected Hierarchy:")
    print("   (showing non-empty tags only)")
    print()
    
    non_empty_tags = {k: v for k, v in expected_tags.items() if v}
    for module_name, tag in sorted(non_empty_tags.items(), key=lambda x: x[1]):
        module_class = type(dict(model.named_modules())[module_name]).__name__
        print(f"   {module_name:35s} → {tag:50s} ({module_class})")
    
    print(f"\n📊 Summary:")
    print(f"   Total modules: {len(expected_tags)}")
    print(f"   Non-empty tags: {len(non_empty_tags)}")
    print(f"   Filtered torch.nn modules: {len(torch_nn_filtered)}")
    
    return expected_tags


def convert_to_hierarchy_tag(module_name: str, module_class: str) -> str:
    """Convert PyTorch module name to expected hierarchy tag."""
    
    if not module_name:
        return "/BertModel"
    
    parts = module_name.split('.')
    tag_parts = []
    
    i = 0
    while i < len(parts):
        part = parts[i]
        
        if part == 'bert':
            tag_parts.append('BertModel')
        elif part == 'embeddings':
            tag_parts.append('BertEmbeddings')
        elif part == 'encoder':
            tag_parts.append('BertEncoder')
        elif part == 'layer':
            # Next part should be a number
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_num = parts[i + 1]
                tag_parts.append(f'BertLayer.{layer_num}')
                i += 1  # Skip the number
        elif part == 'attention':
            tag_parts.append('BertAttention')
        elif part == 'self':
            tag_parts.append('BertSelfAttention')
        elif part == 'output':
            tag_parts.append('BertOutput')
        elif part == 'intermediate':
            tag_parts.append('BertIntermediate')
        elif part == 'pooler':
            tag_parts.append('BertPooler')
        elif part == 'dense':
            tag_parts.append('Dense')
        elif part == 'query':
            tag_parts.append('Query')
        elif part == 'key':
            tag_parts.append('Key')
        elif part == 'value':
            tag_parts.append('Value')
        elif part == 'word_embeddings':
            tag_parts.append('WordEmbeddings')
        elif part == 'position_embeddings':
            tag_parts.append('PositionEmbeddings')
        elif part == 'token_type_embeddings':
            tag_parts.append('TokenTypeEmbeddings')
        elif part == 'LayerNorm':
            tag_parts.append('LayerNorm')
        else:
            # Convert to PascalCase
            tag_parts.append(part.title())
        
        i += 1
    
    return '/' + '/'.join(tag_parts)


def analyze_onnx_nodes_with_tagging():
    """Create ONNX export and analyze which nodes should be tagged."""
    
    print("\n\n2️⃣ ONNX NODES THAT SHOULD BE TAGGED")
    print("=" * 50)
    
    # Load model
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    
    # Prepare inputs
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Export to ONNX (baseline - no hierarchy tags)
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
        temp_path = tmp.name
    
    print("🚀 Creating baseline ONNX export...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        temp_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'}
        },
        opset_version=17,
        verbose=False
    )
    
    # Load and analyze ONNX model
    onnx_model = onnx.load(temp_path)
    
    print(f"✅ ONNX export complete. Analyzing {len(onnx_model.graph.node)} nodes...")
    
    # Categorize nodes by operation type
    nodes_by_type = {}
    for node in onnx_model.graph.node:
        op_type = node.op_type
        if op_type not in nodes_by_type:
            nodes_by_type[op_type] = []
        nodes_by_type[op_type].append(node)
    
    print("\n📊 ONNX Operation Distribution:")
    for op_type, nodes in sorted(nodes_by_type.items()):
        print(f"   {op_type:15s}: {len(nodes):3d} nodes")
    
    print(f"\n   Total nodes: {len(onnx_model.graph.node)}")
    
    # Analyze which nodes should be tagged
    tagging_analysis = analyze_node_tagging_requirements(nodes_by_type)
    
    print("\n🏷️ TAGGING REQUIREMENTS BY OPERATION TYPE:")
    print()
    
    for category, info in tagging_analysis.items():
        print(f"   {category.upper().replace('_', ' ')}:")
        for op_type, requirement in info.items():
            count = len(nodes_by_type.get(op_type, []))
            print(f"      {op_type:15s} ({count:2d} nodes): {requirement}")
        print()
    
    # Show specific examples of nodes that should be tagged
    show_node_tagging_examples(nodes_by_type)
    
    # Clean up
    Path(temp_path).unlink()
    
    return nodes_by_type, tagging_analysis


def analyze_node_tagging_requirements(nodes_by_type: dict[str, list]) -> dict[str, dict[str, str]]:
    """Analyze tagging requirements for different operation types."""
    
    return {
        'critical_operations': {
            'MatMul': 'MUST be tagged with source module (attention, dense layers)',
            'Add': 'MUST be tagged (residual connections, bias addition)',
            'Softmax': 'MUST be tagged (attention probabilities)',
            'Mul': 'MUST be tagged (attention masking, scaling)',
            'Div': 'MUST be tagged (attention scaling)',
        },
        'semantic_operations': {
            'Gather': 'SHOULD be tagged (embedding lookups)',
            'Sub': 'SHOULD be tagged (attention masking)',
            'Erf': 'SHOULD be tagged (GELU activation)',
            'Pow': 'SHOULD be tagged (layer normalization)',
            'Sqrt': 'SHOULD be tagged (layer normalization)',
            'ReduceSum': 'SHOULD be tagged (aggregation operations)',
        },
        'structural_operations': {
            'Reshape': 'MAY be tagged (context-dependent)',
            'Transpose': 'MAY be tagged (attention head reshaping)',
            'Concat': 'MAY be tagged (multi-head concatenation)',
            'Slice': 'MAY be tagged (positional embeddings, attention)',
            'Unsqueeze': 'MAY be tagged (dimension expansion)',
        },
        'support_operations': {
            'Shape': 'Empty tags acceptable (infrastructure)',
            'Cast': 'Empty tags acceptable (type conversion)',
            'Constant': 'Empty tags acceptable (unless parameter)',
            'Equal': 'Empty tags acceptable (mask generation)',
            'Where': 'Empty tags acceptable (conditional logic)',
        }
    }


def show_node_tagging_examples(nodes_by_type: dict[str, list]):
    """Show specific examples of how nodes should be tagged."""
    
    print("🎯 SPECIFIC NODE TAGGING EXAMPLES:")
    print()
    
    examples = [
        {
            'op_type': 'MatMul',
            'example_context': 'attention query-key multiplication',
            'expected_tag': '/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention',
            'rationale': 'Core attention computation - MUST be tagged'
        },
        {
            'op_type': 'Add', 
            'example_context': 'residual connection',
            'expected_tag': '/BertModel/BertEncoder/BertLayer.0',
            'rationale': 'Residual connections are architecturally significant'
        },
        {
            'op_type': 'Softmax',
            'example_context': 'attention probabilities',
            'expected_tag': '/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSelfAttention',
            'rationale': 'Core attention mechanism - MUST be tagged'
        },
        {
            'op_type': 'Gather',
            'example_context': 'word embedding lookup',
            'expected_tag': '/BertModel/BertEmbeddings/WordEmbeddings',
            'rationale': 'Semantic embedding operation'
        },
        {
            'op_type': 'Slice',
            'example_context': 'positional embedding selection',
            'expected_tag': '/BertModel/BertEmbeddings/PositionEmbeddings',
            'rationale': 'Position-aware operation - should be tagged'
        },
        {
            'op_type': 'Shape',
            'example_context': 'dynamic shape calculation',
            'expected_tag': '(empty)',
            'rationale': 'Infrastructure operation - empty tag acceptable'
        }
    ]
    
    for example in examples:
        print(f"   {example['op_type']:10s} | {example['example_context']:30s}")
        print(f"              Expected tag: {example['expected_tag']}")
        print(f"              Rationale: {example['rationale']}")
        print()


def main():
    """Generate complete BERT-tiny hierarchy and tagging analysis."""
    
    print("🎯 BERT-TINY DETAILED ANALYSIS")
    print("=" * 80)
    print("Providing definitive answers to:")
    print("1. Expected hierarchy structure")
    print("2. ONNX nodes that should be tagged")
    print("=" * 80)
    
    # 1. Analyze expected hierarchy
    expected_tags = analyze_complete_hierarchy()
    
    # 2. Analyze ONNX node tagging
    nodes_by_type, tagging_analysis = analyze_onnx_nodes_with_tagging()
    
    # Save results
    output_dir = Path("temp")
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'expected_hierarchy': expected_tags,
        'onnx_nodes_by_type': {op: len(nodes) for op, nodes in nodes_by_type.items()},
        'tagging_analysis': tagging_analysis,
        'summary': {
            'total_modules': len(expected_tags),
            'non_empty_tags': len([tag for tag in expected_tags.values() if tag]),
            'total_onnx_nodes': sum(len(nodes) for nodes in nodes_by_type.values()),
            'critical_operations': len(tagging_analysis['critical_operations']),
        }
    }
    
    with open(output_dir / "bert_tiny_detailed_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print(f"• {results['summary']['non_empty_tags']} modules should have hierarchy tags")
    print(f"• {results['summary']['total_onnx_nodes']} ONNX nodes in total")
    print(f"• {len(tagging_analysis['critical_operations'])} critical operation types MUST be tagged")
    print(f"• Results saved to: {output_dir / 'bert_tiny_detailed_analysis.json'}")


if __name__ == "__main__":
    main()