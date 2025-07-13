"""
Command Line Interface for modelexport.

This module provides a simple CLI for HTP (Hierarchical Trace-and-Project) 
ONNX export with hierarchy preservation.
"""

import json
import sys
from pathlib import Path

import click
import torch

from .core import tag_utils
from .strategies.htp.htp_hierarchy_exporter import HierarchyExporter


@click.group()
@click.version_option()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """Universal hierarchy-preserving ONNX export for PyTorch models."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


@cli.command()
@click.argument('model_name_or_path')
@click.argument('output_path')
@click.option('--input-specs', type=click.Path(exists=True), help='JSON file with input specifications (optional, auto-generates if not provided)')
@click.option('--opset-version', default=14, type=int,
              help='ONNX opset version to use')
@click.option('--config', type=click.Path(exists=True),
              help='Export configuration file (JSON)')
@click.option('--temp-dir', type=click.Path(),
              help='Directory for temporary files (default: system temp)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def export(ctx, model_name_or_path, output_path, input_specs, opset_version, config, temp_dir, verbose):
    """
    Export a PyTorch model to ONNX with hierarchy preservation.
    
    MODEL_NAME_OR_PATH: HuggingFace model name or local path to model
    OUTPUT_PATH: Path where to save the ONNX model
    """
    try:
        # Load model
        if verbose:
            click.echo(f"🔄 Loading model: {model_name_or_path}")
        
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model.eval()
        
        # Generate inputs
        if input_specs:
            with open(input_specs) as f:
                input_specs_dict = json.load(f)
            inputs = {name: torch.tensor(data) for name, data in input_specs_dict.items()}
        else:
            # Auto-generate inputs
            test_text = "This is a test sequence for ONNX export"
            tokenized = tokenizer(test_text, return_tensors='pt', padding=True, truncation=True)
            # Convert BatchEncoding to dict of tensors
            inputs = {key: value for key, value in tokenized.items()}
        
        if verbose:
            click.echo(f"✅ Generated inputs: {list(inputs.keys())}")
        
        # Export with HTP strategy
        if verbose:
            click.echo("🧠 Using HTP (Hierarchical Trace-and-Project) strategy")
        
        exporter = HierarchyExporter(strategy="htp")
        result = exporter.export(
            model=model,
            example_inputs=inputs,
            output_path=output_path,
            opset_version=opset_version
        )
        
        # Output results
        click.echo(f"✅ Export completed successfully!")
        click.echo(f"   ONNX Output: {output_path}")
        click.echo(f"   Sidecar: {output_path.replace('.onnx', '_hierarchy.json')}")
        if 'total_operations' in result:
            click.echo(f"   Total operations: {result['total_operations']}")
        if 'tagged_operations' in result:
            click.echo(f"   Tagged operations: {result['tagged_operations']}")
        click.echo(f"   Strategy: {result['strategy']}")
        
        if verbose:
            # Show tag statistics
            try:
                stats = tag_utils.get_tag_statistics(output_path)
                click.echo(f"\nTag Distribution:")
                for tag, count in sorted(stats.items(), key=lambda x: x[1], reverse=True)[:5]:
                    click.echo(f"   {tag}: {count} operations")
            except Exception as e:
                click.echo(f"Warning: Tag statistics unavailable: {e}", err=True)
        
    except Exception as e:
        import sys
        click.echo(f"Error during export: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


@cli.command()
@click.argument('onnx_path')
@click.option('--output-format', default='json',
              type=click.Choice(['json', 'csv', 'summary']),
              help='Output format for analysis')
@click.option('--filter', type=str, help='Filter tags containing this string')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def analyze(onnx_path, output_format, filter, verbose):
    """Analyze hierarchy tags in an exported ONNX model."""
    try:
        if verbose:
            click.echo(f"🔍 Analyzing ONNX model: {onnx_path}")
        
        stats = tag_utils.get_tag_statistics(onnx_path)
        
        if filter:
            stats = {tag: count for tag, count in stats.items() if filter in tag}
        
        if output_format == 'summary':
            click.echo(f"Total unique tags: {len(stats)}")
            click.echo(f"Total tagged operations: {sum(stats.values())}")
            click.echo(f"Top 5 tags:")
            for tag, count in sorted(stats.items(), key=lambda x: x[1], reverse=True)[:5]:
                click.echo(f"  {tag}: {count}")
        elif output_format == 'json':
            click.echo(json.dumps(stats, indent=2))
        elif output_format == 'csv':
            click.echo("tag,count")
            for tag, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                click.echo(f'"{tag}",{count}')
                
    except Exception as e:
        click.echo(f"Error during analysis: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('onnx_path')
@click.option('--repair', is_flag=True, help='Attempt to repair tag inconsistencies')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def validate(onnx_path, repair, verbose):
    """Validate an ONNX model with hierarchy tags."""
    try:
        if verbose:
            click.echo(f"🔧 Validating ONNX model: {onnx_path}")
        
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        
        click.echo("✅ ONNX model is valid")
        
        # Check for hierarchy tags
        hierarchy_nodes = 0
        for node in model.graph.node:
            if node.doc_string:
                try:
                    hierarchy_info = json.loads(node.doc_string)
                    if isinstance(hierarchy_info, dict) and "hierarchy_tags" in hierarchy_info:
                        hierarchy_nodes += 1
                except json.JSONDecodeError:
                    pass
        
        click.echo(f"Found {hierarchy_nodes} nodes with hierarchy tags")
        
    except Exception as e:
        click.echo(f"Error during validation: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('model1_path')
@click.argument('model2_path')
@click.option('--output', type=click.Path(), help='Save comparison to file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def compare(model1_path, model2_path, output, verbose):
    """Compare hierarchy tags between two ONNX models."""
    try:
        if verbose:
            click.echo(f"🔄 Comparing {model1_path} vs {model2_path}")
        
        stats1 = tag_utils.get_tag_statistics(model1_path)
        stats2 = tag_utils.get_tag_statistics(model2_path)
        
        # Find differences
        all_tags = set(stats1.keys()) | set(stats2.keys())
        differences = []
        
        for tag in all_tags:
            count1 = stats1.get(tag, 0)
            count2 = stats2.get(tag, 0)
            if count1 != count2:
                differences.append({
                    'tag': tag,
                    'model1_count': count1,
                    'model2_count': count2,
                    'difference': count2 - count1
                })
        
        click.echo(f"Found {len(differences)} tag differences")
        
        if output:
            with open(output, 'w') as f:
                json.dump(differences, f, indent=2)
            click.echo(f"Comparison saved to {output}")
        else:
            for diff in differences[:10]:  # Show first 10
                click.echo(f"  {diff['tag']}: {diff['model1_count']} -> {diff['model2_count']}")
                
    except Exception as e:
        click.echo(f"Error during comparison: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()