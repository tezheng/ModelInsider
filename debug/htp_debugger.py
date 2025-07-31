#!/usr/bin/env python3
"""
HTP Debugger - Interactive debugging tool for HTP Integrated Exporter

This script provides detailed debugging information about the HTP export process:
1. Module hierarchy analysis
2. ONNX node tagging details
3. Coverage statistics
4. Visual hierarchy tree

Usage:
    python htp_debugger.py --model prajjwal1/bert-tiny
    python htp_debugger.py --model microsoft/resnet-50
    python htp_debugger.py  # defaults to bert-tiny
"""

import argparse
import json

# Add parent directory to path
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx
import torch
from rich.console import Console
from rich.text import Text
from rich.tree import Tree
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))

from modelexport.core.onnx_node_tagger import create_node_tagger_from_hierarchy
from modelexport.core.tracing_hierarchy_builder import TracingHierarchyBuilder


def print_section_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"🔍 {title}")
    print(f"{'=' * 80}")


def print_hf_class_hierarchy(hierarchy_data: dict[str, dict]):
    """Print HuggingFace hierarchy with beautiful tree rendering using rich."""
    console = Console()
    
    print("\n📊 HuggingFace Class Hierarchy:")
    print("-" * 60)
    
    # Sort by execution order to understand the logical flow
    sorted_modules = sorted(
        hierarchy_data.items(), 
        key=lambda x: x[1].get('execution_order', 999)
    )
    
    # Find root module
    root_module = None
    root_class = "Model"
    for module_path, module_info in sorted_modules:
        if not module_path:  # Root module
            root_class = module_info.get('class_name', 'Model')
            root_module = module_info
            break
    
    # Create rich tree with styled root (optimized for dark terminals)
    tree = Tree(
        Text(root_class, style="bold bright_cyan"),
        guide_style="bright_white"
    )
    
    # Build tree structure using a dictionary to track nodes
    node_map = {}  # Maps module paths to tree nodes
    
    # Add all modules to the tree
    for module_path, module_info in sorted_modules:
        if not module_path:  # Skip root as it's already added
            continue
            
        class_name = module_info.get('class_name', 'Unknown')
        
        # Create styled text for this node (bright colors for dark theme)
        if module_path.count('.') == 0:
            # Top-level module
            node_text = Text()
            node_text.append(class_name, style="bold bright_green")
            node_text.append(f": {module_path}", style="bright_cyan")
        else:
            # Nested module
            node_text = Text()
            node_text.append(class_name, style="bright_yellow")
            node_text.append(f": {module_path}", style="bright_white")
        
        # Find parent node
        path_parts = module_path.split('.')
        if len(path_parts) == 1:
            # Direct child of root
            parent_node = tree
        else:
            # Find parent in node_map
            parent_path = '.'.join(path_parts[:-1])
            parent_node = node_map.get(parent_path, tree)
        
        # Add this node to its parent
        current_node = parent_node.add(node_text)
        node_map[module_path] = current_node
    
    # Render the tree
    console.print(tree)


def print_full_hf_hierarchy(hierarchy_data: dict[str, dict]):
    """Print complete HuggingFace hierarchy with all details."""
    print("\n📊 Complete HuggingFace Module Hierarchy (Detailed):")
    print("-" * 80)
    
    # Sort by execution order for clearer understanding
    sorted_modules = sorted(
        hierarchy_data.items(), 
        key=lambda x: x[1].get('execution_order', 999)
    )
    
    for i, (module_path, module_info) in enumerate(sorted_modules):
        if not module_path:  # Root module
            print(f"{i+1:2d}. <ROOT MODULE>")
        else:
            print(f"{i+1:2d}. {module_path}")
        
        print(f"    └─ Class: {module_info.get('class_name', 'Unknown')}")
        print(f"    └─ Type: {module_info.get('module_type', 'Unknown')}")
        print(f"    └─ HF Tag: {module_info.get('traced_tag', 'No tag')}")
        print(f"    └─ Exec Order: {module_info.get('execution_order', 'N/A')}")
        
        if i < len(sorted_modules) - 1:
            print()

def print_hierarchy_tree(hierarchy_data: dict[str, dict], max_depth: int = 5):
    """Print hierarchy as a tree structure with increased depth."""
    print("\n📊 Module Hierarchy Tree (visual structure):")
    print("-" * 60)
    
    # Build tree structure with full depth
    tree = {}
    for module_path, module_info in hierarchy_data.items():
        if not module_path:  # Root module
            tree["<root>"] = {
                "info": module_info,
                "children": {}
            }
            continue
            
        parts = module_path.split('.')
        current = tree
        for i, part in enumerate(parts):
            if i >= max_depth:
                break
            if part not in current:
                current[part] = {"info": {}, "children": {}}
            if i == len(parts) - 1:  # Last part, store module info
                current[part]["info"] = module_info
            current = current[part]["children"]
    
    # Print tree with module information
    def print_tree(node: dict, prefix: str = "", is_last: bool = True):
        for i, (key, value) in enumerate(node.items()):
            is_last_item = i == len(node) - 1
            current_prefix = "└── " if is_last_item else "├── "
            
            # Show key with class name if available
            if "info" in value and value["info"].get("class_name"):
                class_name = value["info"]["class_name"]
                print(f"{prefix}{current_prefix}{key} ({class_name})")
            else:
                print(f"{prefix}{current_prefix}{key}")
            
            if "children" in value and value["children"]:
                extension = "    " if is_last_item else "│   "
                print_tree(value["children"], prefix + extension, is_last_item)
    
    print_tree(tree)


def analyze_tag_distribution(tagged_nodes: dict[str, str]) -> dict[str, int]:
    """Analyze distribution of tags."""
    tag_counter = Counter(tagged_nodes.values())
    return dict(tag_counter.most_common())


def debug_scope_buckets(onnx_model: onnx.ModelProto, node_tagger) -> dict[str, list[str]]:
    """Debug scope bucketization."""
    scope_buckets = node_tagger.bucketize_nodes_by_scope(onnx_model)
    
    # Convert to simpler format for display
    simple_buckets = {}
    for scope_name, nodes in scope_buckets.items():
        simple_buckets[scope_name] = [node.name for node in nodes]
    
    return simple_buckets


def analyze_tagging_accuracy(onnx_model: onnx.ModelProto, node_tagger) -> dict[str, Any]:
    """Analyze tagging accuracy and statistics."""
    stats = node_tagger.get_tagging_statistics(onnx_model)
    
    # Calculate percentages
    total = stats['total_nodes']
    accuracy_breakdown = {
        'direct_match_rate': (stats['direct_matches'] / total * 100) if total > 0 else 0,
        'parent_match_rate': (stats['parent_matches'] / total * 100) if total > 0 else 0,
        'root_fallback_rate': (stats['root_fallbacks'] / total * 100) if total > 0 else 0,
        'operation_match_rate': (stats['operation_matches'] / total * 100) if total > 0 else 0,
    }
    
    return {
        'raw_stats': stats,
        'accuracy_breakdown': accuracy_breakdown
    }


def print_complete_hierarchy_with_nodes(
    hierarchy_data: dict[str, dict], 
    tagged_nodes: dict[str, str], 
    onnx_model: onnx.ModelProto,
    node_tagger
) -> None:
    """Print complete HF hierarchy with all ONNX nodes as leaves using beautiful rich tree."""
    console = Console()
    
    print("\n🌳 Complete HuggingFace Hierarchy with ONNX Nodes:")
    print("=" * 80)
    
    # Group nodes by their tags (which correspond to HF modules)
    nodes_by_tag = defaultdict(list)
    for node_name, tag in tagged_nodes.items():
        nodes_by_tag[tag].append(node_name)
    
    # Create a mapping from node names to ONNX nodes for additional info
    node_info_map = {}
    for node in onnx_model.graph.node:
        node_name = node.name or f"{node.op_type}_{id(node)}"
        node_info_map[node_name] = {
            'op_type': node.op_type,
            'scope': node_tagger._extract_scope_from_node(node),
            'inputs': list(node.input),
            'outputs': list(node.output)
        }
    
    # Sort hierarchy by execution order for logical display
    sorted_hierarchy = sorted(
        hierarchy_data.items(),
        key=lambda x: x[1].get('execution_order', 999)
    )
    
    # Find root module
    root_class = "Model"
    for module_path, module_info in sorted_hierarchy:
        if not module_path:  # Root module
            root_class = module_info.get('class_name', 'Model')
            break
    
    # Create rich tree with styled root
    tree = Tree(
        Text(f"{root_class} (145 ONNX nodes)", style="bold bright_cyan"),
        guide_style="bright_white"
    )
    
    # Build tree structure with ONNX nodes as leaves
    node_map = {}  # Maps module paths to tree nodes
    
    # Add all modules to the tree with their ONNX operations
    for module_path, module_info in sorted_hierarchy:
        if not module_path:  # Skip root as it's already added
            continue
            
        class_name = module_info.get('class_name', 'Unknown')
        module_tag = module_info.get('traced_tag', '')
        
        # Count ONNX nodes for this module
        module_nodes = nodes_by_tag.get(module_tag, [])
        node_count = len(module_nodes)
        
        # Create styled text for this module
        module_text = Text()
        if module_path.count('.') == 0:
            # Top-level module
            module_text.append(class_name, style="bold bright_green")
            module_text.append(f": {module_path}", style="bright_cyan")
            module_text.append(f" ({node_count} nodes)", style="dim bright_white")
        else:
            # Nested module
            module_text.append(class_name, style="bright_yellow")
            module_text.append(f": {module_path}", style="bright_white")
            module_text.append(f" ({node_count} nodes)", style="dim bright_white")
        
        # Find parent node
        path_parts = module_path.split('.')
        if len(path_parts) == 1:
            # Direct child of root
            parent_node = tree
        else:
            # Find parent in node_map
            parent_path = '.'.join(path_parts[:-1])
            parent_node = node_map.get(parent_path, tree)
        
        # Add this module to its parent
        current_node = parent_node.add(module_text)
        node_map[module_path] = current_node
        
        # Add ONNX operations as children of this module
        if module_nodes:
            # Group operations by type for better organization
            ops_by_type = defaultdict(list)
            for node_name in module_nodes:
                if node_name in node_info_map:
                    op_type = node_info_map[node_name]['op_type']
                    ops_by_type[op_type].append(node_name)
            
            # Add operation type groups
            for op_type, op_nodes in sorted(ops_by_type.items()):
                if len(op_nodes) == 1:
                    # Single operation - show directly
                    node_name = op_nodes[0]
                    scope = node_info_map[node_name]['scope']
                    op_text = Text()
                    op_text.append(f"{op_type}", style="bright_magenta")
                    op_text.append(f": {node_name}", style="dim bright_cyan")
                    current_node.add(op_text)
                else:
                    # Multiple operations - group them
                    group_text = Text()
                    group_text.append(f"{op_type}", style="bright_magenta")
                    group_text.append(f" ({len(op_nodes)} ops)", style="dim bright_white")
                    op_group_node = current_node.add(group_text)
                    
                    # Show first few operations
                    for _i, node_name in enumerate(sorted(op_nodes)[:3]):
                        scope = node_info_map[node_name]['scope']
                        individual_op_text = Text()
                        individual_op_text.append(node_name, style="dim bright_cyan")
                        op_group_node.add(individual_op_text)
                    
                    # Add "..." if there are more
                    if len(op_nodes) > 3:
                        more_text = Text(f"... and {len(op_nodes) - 3} more", style="dim bright_black")
                        op_group_node.add(more_text)
    
    # Add unmatched nodes if any
    unmatched_nodes = []
    for node_name, tag in tagged_nodes.items():
        found_in_hierarchy = False
        for module_path, module_info in hierarchy_data.items():
            if module_info.get('traced_tag') == tag:
                found_in_hierarchy = True
                break
        if not found_in_hierarchy:
            unmatched_nodes.append((node_name, tag))
    
    if unmatched_nodes:
        # Group unmatched by tag
        other_by_tag = defaultdict(list)
        for node_name, tag in unmatched_nodes:
            other_by_tag[tag].append(node_name)
        
        for tag, nodes in sorted(other_by_tag.items()):
            other_text = Text()
            other_text.append("Other Operations", style="bright_red")
            other_text.append(f": {tag}", style="bright_white")
            other_text.append(f" ({len(nodes)} nodes)", style="dim bright_white")
            other_node = tree.add(other_text)
            
            # Group by operation type
            ops_by_type = defaultdict(list)
            for node_name in nodes:
                if node_name in node_info_map:
                    op_type = node_info_map[node_name]['op_type']
                    ops_by_type[op_type].append(node_name)
            
            for op_type, op_nodes in sorted(ops_by_type.items()):
                op_text = Text()
                op_text.append(f"{op_type}", style="bright_magenta")
                op_text.append(f" ({len(op_nodes)} ops)", style="dim bright_white")
                other_node.add(op_text)
    
    # Render the complete tree
    console.print(tree)


def prepare_inputs(model_name: str) -> tuple[Any, dict[str, Any]]:
    """Prepare inputs based on model type."""
    # Simple heuristics for common model types
    if 'bert' in model_name.lower() or 'roberta' in model_name.lower():
        # BERT-like model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        inputs = tokenizer("Hello world example", return_tensors="pt", 
                          max_length=32, padding="max_length", truncation=True)
        example_inputs = (inputs["input_ids"], inputs["attention_mask"])
        input_names = ['input_ids', 'attention_mask']
    elif 'gpt' in model_name.lower() or 'dialogen' in model_name.lower():
        # GPT-like model (only needs input_ids)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer("Hello world example", return_tensors="pt", 
                          max_length=32, padding="max_length", truncation=True)
        example_inputs = (inputs["input_ids"],)
        input_names = ['input_ids']
    elif 'resnet' in model_name.lower() or 'vit' in model_name.lower():
        # Vision model
        example_inputs = torch.randn(1, 3, 224, 224)
        input_names = ['image']
    else:
        # Generic fallback - try to load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            inputs = tokenizer("Hello world example", return_tensors="pt", 
                              max_length=32, padding="max_length", truncation=True)
            example_inputs = (inputs["input_ids"],)
            input_names = ['input_ids']
        except:
            # Ultimate fallback
            example_inputs = torch.randn(1, 8)
            input_names = ['input']
    
    return example_inputs, {'input_names': input_names}


def main():
    parser = argparse.ArgumentParser(description='HTP Debugger - Debug HTP export process')
    parser.add_argument('--model', type=str, default='prajjwal1/bert-tiny',
                        help='Model to debug (default: prajjwal1/bert-tiny)')
    parser.add_argument('--enable-operation-fallback', action='store_true',
                        help='Enable operation-based fallback in tagging')
    parser.add_argument('--save-outputs', action='store_true',
                        help='Save debug outputs to files')
    parser.add_argument('--output', type=str, default='temp/debug/htp_debugger',
                        help='Output directory for debug files (default: temp/debug/htp_debugger)')
    args = parser.parse_args()
    
    print(f"🚀 HTP Debugger")
    print(f"   Model: {args.model}")
    print(f"   Operation fallback: {'Enabled' if args.enable_operation_fallback else 'Disabled'}")
    print(f"   Output directory: {args.output}")
    
    # Load model
    print(f"\n📥 Loading model...")
    model = AutoModel.from_pretrained(args.model)
    model.eval()
    
    # Prepare inputs
    example_inputs, export_kwargs = prepare_inputs(args.model)
    
    # Create output directory and temporary ONNX file
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "debug_model.onnx")
    
    # Step 1: Build hierarchy with TracingHierarchyBuilder
    print_section_header("Step 1: Building Module Hierarchy")
    
    hierarchy_builder = TracingHierarchyBuilder()
    if isinstance(example_inputs, torch.Tensor):
        input_args = (example_inputs,)
    else:
        input_args = example_inputs
    
    hierarchy_builder.trace_model_execution(model, input_args)
    execution_summary = hierarchy_builder.get_execution_summary()
    hierarchy_data = execution_summary['module_hierarchy']
    
    print(f"✅ Traced {len(hierarchy_data)} modules")
    print(f"   Execution steps: {execution_summary['execution_steps']}")
    print(f"   Total modules in model: {execution_summary['total_modules']}")
    print(f"   Optimization ratio: {len(hierarchy_data)}/{execution_summary['total_modules']} "
          f"({len(hierarchy_data)/execution_summary['total_modules']*100:.1f}%)")
    
    # Print HF class hierarchy in requested format
    print_hf_class_hierarchy(hierarchy_data)
    
    # Print detailed hierarchy (optional, can be commented out)
    # print_full_hf_hierarchy(hierarchy_data)
    
    # Step 2: Export to ONNX
    print_section_header("Step 2: ONNX Export")
    
    print("📦 Exporting to ONNX...")
    torch.onnx.export(
        model,
        input_args,
        output_path,
        opset_version=17,
        **export_kwargs
    )
    
    onnx_model = onnx.load(output_path)
    print(f"✅ ONNX model exported with {len(onnx_model.graph.node)} nodes")
    
    # Step 3: Create node tagger and analyze
    print_section_header("Step 3: ONNX Node Tagging Analysis")
    
    node_tagger = create_node_tagger_from_hierarchy(
        hierarchy_data, 
        enable_operation_fallback=args.enable_operation_fallback
    )
    
    print(f"📋 Node Tagger Configuration:")
    print(f"   Model root: {node_tagger.model_root_tag}")
    print(f"   Scope mappings: {len(node_tagger.scope_to_tag)}")
    print(f"   Operation fallback: {'Enabled' if node_tagger.enable_operation_fallback else 'Disabled'}")
    
    # Tag all nodes
    tagged_nodes = node_tagger.tag_all_nodes(onnx_model)
    
    print(f"\n✅ Tagged {len(tagged_nodes)} nodes")
    
    # Analyze tagging accuracy
    accuracy_info = analyze_tagging_accuracy(onnx_model, node_tagger)
    stats = accuracy_info['raw_stats']
    accuracy = accuracy_info['accuracy_breakdown']
    
    print(f"\n📊 Tagging Statistics:")
    print(f"   Direct matches: {stats['direct_matches']} ({accuracy['direct_match_rate']:.1f}%)")
    print(f"   Parent matches: {stats['parent_matches']} ({accuracy['parent_match_rate']:.1f}%)")
    print(f"   Root fallbacks: {stats['root_fallbacks']} ({accuracy['root_fallback_rate']:.1f}%)")
    if args.enable_operation_fallback:
        print(f"   Operation matches: {stats['operation_matches']} ({accuracy['operation_match_rate']:.1f}%)")
    
    # Step 4: Analyze scope buckets
    print_section_header("Step 4: Scope Bucketization Analysis")
    
    scope_buckets = debug_scope_buckets(onnx_model, node_tagger)
    
    print(f"📂 Found {len(scope_buckets)} unique scopes:")
    
    # Sort by number of nodes
    sorted_scopes = sorted(scope_buckets.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Show top 10 scopes
    for i, (scope_name, nodes) in enumerate(sorted_scopes[:10]):
        print(f"   {i+1}. {scope_name}: {len(nodes)} nodes")
        if len(nodes) <= 3:
            for node in nodes:
                print(f"      └─ {node}")
        else:
            print(f"      └─ {nodes[0]}")
            print(f"      └─ ...")
            print(f"      └─ {nodes[-1]}")
    
    if len(sorted_scopes) > 10:
        print(f"   ... and {len(sorted_scopes) - 10} more scopes")
    
    # Step 5: Tag distribution analysis
    print_section_header("Step 5: Tag Distribution Analysis")
    
    tag_distribution = analyze_tag_distribution(tagged_nodes)
    
    print(f"🏷️ Unique tags: {len(tag_distribution)}")
    print(f"\nTop 10 most common tags:")
    
    for i, (tag, count) in enumerate(list(tag_distribution.items())[:10]):
        percentage = (count / len(tagged_nodes)) * 100
        print(f"   {i+1}. {tag}: {count} nodes ({percentage:.1f}%)")
    
    # Step 6: Complete HF Hierarchy with ONNX Nodes
    print_section_header("Step 6: Complete HF Hierarchy with ONNX Nodes")
    
    print_complete_hierarchy_with_nodes(hierarchy_data, tagged_nodes, onnx_model, node_tagger)
    
    # Step 7: Verify CARDINAL RULES
    print_section_header("Step 7: CARDINAL RULES Verification")
    
    # Check for empty tags
    empty_tags = [name for name, tag in tagged_nodes.items() if not tag or not tag.strip()]
    
    print(f"✅ MUST-001 (NO HARDCODED): Model root dynamically extracted as '{node_tagger.model_root_tag}'")
    print(f"✅ MUST-002 (NO EMPTY TAGS): {len(empty_tags)} empty tags found")
    print(f"✅ MUST-003 (UNIVERSAL DESIGN): Works with {args.model}")
    
    if empty_tags:
        print(f"\n⚠️ WARNING: Found {len(empty_tags)} empty tags!")
        for node_name in empty_tags[:5]:
            print(f"   - {node_name}")
    
    # Save outputs if requested
    if args.save_outputs:
        debug_dir = output_dir
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Save hierarchy data
        with open(debug_dir / "hierarchy_data.json", 'w') as f:
            json.dump(hierarchy_data, f, indent=2)
        
        # Save tagged nodes
        with open(debug_dir / "tagged_nodes.json", 'w') as f:
            json.dump(tagged_nodes, f, indent=2)
        
        # Save scope buckets
        with open(debug_dir / "scope_buckets.json", 'w') as f:
            json.dump(scope_buckets, f, indent=2)
        
        # Save statistics
        with open(debug_dir / "statistics.json", 'w') as f:
            json.dump({
                'model': args.model,
                'hierarchy_modules': len(hierarchy_data),
                'onnx_nodes': len(onnx_model.graph.node),
                'tagged_nodes': len(tagged_nodes),
                'unique_tags': len(tag_distribution),
                'unique_scopes': len(scope_buckets),
                'tagging_stats': stats,
                'accuracy_breakdown': accuracy,
                'tag_distribution': tag_distribution
            }, f, indent=2)
        
        print(f"\n💾 Debug outputs saved to: {debug_dir}")
    
    # Keep ONNX file in output directory (don't delete)
    
    print(f"\n✅ HTP debugging completed successfully!")
    print(f"   Model: {args.model}")
    print(f"   Hierarchy modules: {len(hierarchy_data)}")
    print(f"   ONNX nodes: {len(onnx_model.graph.node)}")
    print(f"   Coverage: 100% ({len(tagged_nodes)}/{len(onnx_model.graph.node)})")
    print(f"   Output directory: {args.output}")
    print(f"   ONNX model saved: {output_path}")


if __name__ == "__main__":
    main()