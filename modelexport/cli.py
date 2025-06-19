"""
Command Line Interface for modelexport.

This module provides a structured CLI with subcommands for ONNX export,
validation, and analysis. All commands are designed to be:
1. Reusable with extensible arguments
2. Testable with pytest
3. User-friendly with clear help text
"""

import click
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import shutil

from .hierarchy_exporter import HierarchyExporter
from . import tag_utils


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
@click.option('--input-text', default='Hello world test input for model tracing and ONNX export validation', 
              help='Input text for model tracing (for language models). Default provides reasonable coverage.')
@click.option('--input-shape', help='Input shape as comma-separated values (e.g., 1,3,224,224)')
@click.option('--strategy', default='usage_based', 
              type=click.Choice(['usage_based']), 
              help='Tagging strategy to use')
@click.option('--opset-version', default=14, type=int,
              help='ONNX opset version to use')
@click.option('--temp-dir', type=click.Path(),
              help='Directory for temporary files (default: system temp)')
@click.pass_context
def export(ctx, model_name_or_path, output_path, input_text, input_shape, strategy, opset_version, temp_dir):
    """
    Export a PyTorch model to ONNX with hierarchy preservation.
    
    MODEL_NAME_OR_PATH: HuggingFace model name or local path to model
    OUTPUT_PATH: Path where to save the ONNX model
    
    Examples:
    \b
        # Export BERT model
        modelexport export prajjwal1/bert-tiny bert_tiny.onnx
        
        # Export with custom input text
        modelexport export prajjwal1/bert-tiny bert.onnx --input-text "Custom test input"
        
        # Export with specific opset version
        modelexport export prajjwal1/bert-tiny bert.onnx --opset-version 16
    """
    verbose = ctx.obj['verbose']
    
    try:
        # Set up temp directory
        if temp_dir:
            temp_dir = Path(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            click.echo(f"Loading model: {model_name_or_path}")
        
        # Dynamic import to avoid heavy dependencies if not needed
        try:
            from transformers import AutoModel, AutoTokenizer
            
            # Load model and tokenizer
            model = AutoModel.from_pretrained(model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            
            # Prepare inputs
            inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
            
        except ImportError:
            click.echo("Error: transformers library required for HuggingFace models", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error loading model: {e}", err=True)
            sys.exit(1)
        
        if verbose:
            click.echo(f"Exporting to: {output_path}")
        
        # Export with hierarchy preservation
        exporter = HierarchyExporter(strategy=strategy)
        result = exporter.export(
            model=model,
            example_inputs=inputs,
            output_path=output_path,
            opset_version=opset_version
        )
        
        # Output results
        click.echo(f"✅ Export completed successfully!")
        click.echo(f"   Output: {output_path}")
        click.echo(f"   Sidecar: {output_path.replace('.onnx', '_hierarchy.json')}")
        click.echo(f"   Total operations: {result['total_operations']}")
        click.echo(f"   Tagged operations: {result['tagged_operations']}")
        click.echo(f"   Strategy: {result['strategy']}")
        
        if verbose:
            # Show tag statistics
            try:
                stats = tag_utils.get_tag_statistics(output_path)
                click.echo("\nTag distribution:")
                for tag, count in stats.items():
                    click.echo(f"   {tag}: {count}")
            except Exception as e:
                click.echo(f"Warning: Could not load tag statistics: {e}", err=True)
        
    except Exception as e:
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
@click.option('--output-file', type=click.Path(),
              help='Save analysis to file instead of stdout')
@click.option('--filter-tag', help='Filter operations by tag pattern')
@click.pass_context  
def analyze(ctx, onnx_path, output_format, output_file, filter_tag):
    """
    Analyze hierarchy tags in an exported ONNX model.
    
    ONNX_PATH: Path to the ONNX model to analyze
    
    Examples:
    \b
        # Show tag summary
        modelexport analyze bert_tiny.onnx
        
        # Export detailed analysis to CSV
        modelexport analyze bert_tiny.onnx --output-format csv --output-file analysis.csv
        
        # Filter operations by tag
        modelexport analyze bert_tiny.onnx --filter-tag BertAttention
    """
    verbose = ctx.obj['verbose']
    
    try:
        if not Path(onnx_path).exists():
            click.echo(f"Error: ONNX file not found: {onnx_path}", err=True)
            sys.exit(1)
        
        if output_format == 'summary':
            # Show summary statistics
            stats = tag_utils.get_tag_statistics(onnx_path)
            
            output_data = {
                "model_path": onnx_path,
                "tag_statistics": stats,
                "total_unique_tags": len(stats),
                "total_tagged_operations": sum(stats.values())
            }
            
            if filter_tag:
                filtered_ops = tag_utils.query_operations_by_tag(onnx_path, filter_tag)
                output_data["filtered_operations"] = len(filtered_ops)
                if verbose:
                    output_data["filtered_details"] = filtered_ops[:10]  # Show first 10
            
        elif output_format == 'json':
            # Load full sidecar data
            try:
                sidecar_data = tag_utils.load_tags_from_sidecar(onnx_path)
                output_data = sidecar_data
                
                if filter_tag:
                    filtered_ops = tag_utils.query_operations_by_tag(onnx_path, filter_tag)
                    output_data["filtered_operations"] = filtered_ops
                    
            except FileNotFoundError:
                # Fallback to ONNX attributes
                onnx_tags = tag_utils.load_tags_from_onnx(onnx_path)
                output_data = {"node_tags": onnx_tags}
        
        elif output_format == 'csv':
            if not output_file:
                output_file = onnx_path.replace('.onnx', '_analysis.csv')
            
            tag_utils.export_tags_to_csv(onnx_path, output_file)
            click.echo(f"✅ Analysis exported to: {output_file}")
            return
        
        # Output results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            click.echo(f"✅ Analysis saved to: {output_file}")
        else:
            if output_format == 'summary':
                click.echo(f"📊 Analysis Summary for {onnx_path}")
                click.echo(f"   Total unique tags: {output_data['total_unique_tags']}")
                click.echo(f"   Total tagged operations: {output_data['total_tagged_operations']}")
                
                if filter_tag:
                    click.echo(f"   Operations matching '{filter_tag}': {output_data.get('filtered_operations', 0)}")
                
                click.echo("\nTag distribution:")
                for tag, count in output_data['tag_statistics'].items():
                    click.echo(f"   {tag}: {count}")
            else:
                click.echo(json.dumps(output_data, indent=2))
        
    except Exception as e:
        click.echo(f"Error during analysis: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


@cli.command()
@click.argument('onnx_path')
@click.option('--check-consistency', is_flag=True,
              help='Validate consistency between ONNX attributes and sidecar')
@click.option('--repair', is_flag=True,
              help='Attempt to repair inconsistencies')
@click.pass_context
def validate(ctx, onnx_path, check_consistency, repair):
    """
    Validate hierarchy tags in an ONNX model.
    
    ONNX_PATH: Path to the ONNX model to validate
    
    Examples:
    \b
        # Basic validation
        modelexport validate bert_tiny.onnx
        
        # Check consistency between ONNX and sidecar
        modelexport validate bert_tiny.onnx --check-consistency
        
        # Attempt to repair inconsistencies
        modelexport validate bert_tiny.onnx --check-consistency --repair
    """
    verbose = ctx.obj['verbose']
    
    try:
        if not Path(onnx_path).exists():
            click.echo(f"Error: ONNX file not found: {onnx_path}", err=True)
            sys.exit(1)
        
        # Basic validation
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            click.echo(f"✅ ONNX model is valid: {onnx_path}")
        except Exception as e:
            click.echo(f"❌ ONNX model is invalid: {e}", err=True)
            sys.exit(1)
        
        # Check for hierarchy tags
        onnx_tags = tag_utils.load_tags_from_onnx(onnx_path)
        if onnx_tags:
            click.echo(f"✅ Found {len(onnx_tags)} operations with hierarchy tags in ONNX")
        else:
            click.echo("⚠️  No hierarchy tags found in ONNX attributes")
        
        # Check sidecar file
        sidecar_path = onnx_path.replace('.onnx', '_hierarchy.json')
        if Path(sidecar_path).exists():
            try:
                sidecar_data = tag_utils.load_tags_from_sidecar(onnx_path)
                node_tags = sidecar_data.get('node_tags', {})
                click.echo(f"✅ Found sidecar file with {len(node_tags)} tagged operations")
            except Exception as e:
                click.echo(f"❌ Invalid sidecar file: {e}", err=True)
                sys.exit(1)
        else:
            click.echo("⚠️  No sidecar file found")
        
        # Consistency check
        if check_consistency:
            validation_report = tag_utils.validate_tag_consistency(onnx_path)
            
            if validation_report['consistent']:
                click.echo("✅ Tags are consistent between ONNX and sidecar")
            else:
                click.echo("❌ Tag inconsistencies found:")
                
                if 'error' in validation_report:
                    click.echo(f"   Error: {validation_report['error']}")
                else:
                    click.echo(f"   ONNX nodes: {validation_report['total_onnx_nodes']}")
                    click.echo(f"   Sidecar nodes: {validation_report['total_sidecar_nodes']}")
                    
                    if validation_report.get('tag_mismatches'):
                        click.echo(f"   Tag mismatches: {len(validation_report['tag_mismatches'])}")
                        if verbose:
                            for mismatch in validation_report['tag_mismatches'][:5]:
                                click.echo(f"     {mismatch['node']}: {mismatch['onnx_tags']} vs {mismatch['sidecar_tags']}")
                    
                    if validation_report.get('onnx_only_nodes'):
                        click.echo(f"   ONNX-only nodes: {len(validation_report['onnx_only_nodes'])}")
                    
                    if validation_report.get('sidecar_only_nodes'):
                        click.echo(f"   Sidecar-only nodes: {len(validation_report['sidecar_only_nodes'])}")
                
                if repair:
                    click.echo("🔧 Repair functionality not yet implemented")
                    # TODO: Implement repair logic
        
    except Exception as e:
        click.echo(f"Error during validation: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


@cli.command()
@click.argument('onnx_path1')
@click.argument('onnx_path2')
@click.option('--output-file', type=click.Path(),
              help='Save comparison to file')
@click.pass_context
def compare(ctx, onnx_path1, onnx_path2, output_file):
    """
    Compare hierarchy tags between two ONNX models.
    
    ONNX_PATH1: Path to first ONNX model
    ONNX_PATH2: Path to second ONNX model
    
    Examples:
    \b
        # Compare two models
        modelexport compare model1.onnx model2.onnx
        
        # Save comparison to file
        modelexport compare model1.onnx model2.onnx --output-file comparison.json
    """
    verbose = ctx.obj['verbose']
    
    try:
        for path in [onnx_path1, onnx_path2]:
            if not Path(path).exists():
                click.echo(f"Error: ONNX file not found: {path}", err=True)
                sys.exit(1)
        
        comparison = tag_utils.compare_tag_distributions(onnx_path1, onnx_path2)
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            click.echo(f"✅ Comparison saved to: {output_file}")
        else:
            click.echo(f"📊 Tag Distribution Comparison")
            click.echo(f"   Model 1: {onnx_path1}")
            click.echo(f"   Model 2: {onnx_path2}")
            
            if comparison['model1_only_tags']:
                click.echo(f"\n🔵 Tags only in Model 1 ({len(comparison['model1_only_tags'])}):")
                for tag in comparison['model1_only_tags'][:5]:
                    click.echo(f"   {tag}")
                if len(comparison['model1_only_tags']) > 5:
                    click.echo(f"   ... and {len(comparison['model1_only_tags']) - 5} more")
            
            if comparison['model2_only_tags']:
                click.echo(f"\n🔴 Tags only in Model 2 ({len(comparison['model2_only_tags'])}):")
                for tag in comparison['model2_only_tags'][:5]:
                    click.echo(f"   {tag}")
                if len(comparison['model2_only_tags']) > 5:
                    click.echo(f"   ... and {len(comparison['model2_only_tags']) - 5} more")
            
            if comparison['tag_differences']:
                click.echo(f"\n⚖️  Count Differences ({len(comparison['tag_differences'])}):")
                for diff in comparison['tag_differences'][:5]:
                    click.echo(f"   {diff['tag']}: {diff['model1_count']} vs {diff['model2_count']} (Δ{diff['difference']:+d})")
                if len(comparison['tag_differences']) > 5:
                    click.echo(f"   ... and {len(comparison['tag_differences']) - 5} more")
        
    except Exception as e:
        click.echo(f"Error during comparison: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()