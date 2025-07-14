"""
Command Line Interface for modelexport.

This module provides a structured CLI with subcommands for ONNX export,
validation, and analysis. All commands are designed to be:
1. Reusable with extensible arguments
2. Testable with pytest
3. User-friendly with clear help text
"""

import json
import sys
from pathlib import Path

import click
import torch

from .core import tag_utils
from .strategies.htp.htp_hierarchy_exporter import HierarchyExporter
from .strategies.htp.htp_exporter import export_with_htp_reporting


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
@click.option('--jit-graph', is_flag=True,
              help='Dump TorchScript graph information before ONNX export (preserves context)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def export(ctx, model_name_or_path, output_path, input_specs, opset_version, config, temp_dir, jit_graph, verbose):
    """
    Export a PyTorch model to ONNX with hierarchy preservation.
    
    MODEL_NAME_OR_PATH: HuggingFace model name or local path to model
    OUTPUT_PATH: Path where to save the ONNX model
    
    Examples:
    \b
        # Export BERT model with auto-generated inputs (default, recommended)
        modelexport export prajjwal1/bert-tiny bert.onnx
        
        # Export with custom input specifications
        modelexport export prajjwal1/bert-tiny bert.onnx --input-specs input_specs.json
        
        # Export with verbose output showing optimization details
        modelexport export prajjwal1/bert-tiny bert.onnx --verbose
        
        # Export with export configuration (for ONNX export settings)
        modelexport export prajjwal1/bert-tiny bert.onnx --config export_config.json
        
        # Export with Enhanced Semantic mapping (alternative strategy)
        modelexport export prajjwal1/bert-tiny bert.onnx --strategy enhanced_semantic
        
        # Export with legacy HTP strategy
        modelexport export prajjwal1/bert-tiny bert.onnx --strategy htp --jit-graph
        
        # Export with FX graph alternative (for analysis)
        modelexport export prajjwal1/bert-tiny bert.onnx --fx-graph both
        
        # Full debug export with all features
        modelexport export prajjwal1/bert-tiny bert.onnx --config config.json --jit-graph --fx-graph both --verbose
    """
    
    try:
        # Set up temp directory
        if temp_dir:
            temp_dir = Path(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            click.echo(f"Loading model: {model_name_or_path}")
        
        # Dynamic import to avoid heavy dependencies if not needed
        try:
            from transformers import AutoModel
            
            # Load model
            model = AutoModel.from_pretrained(model_name_or_path)
            
            # Load input specs if provided
            input_specs_dict = None
            if input_specs:
                with open(input_specs) as f:
                    input_specs_dict = json.load(f)
                if verbose:
                    click.echo(f"Loaded input specs from: {input_specs}")
            
        except ImportError:
            click.echo("Error: transformers library required for HuggingFace models", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error loading model: {e}", err=True)
            sys.exit(1)
        
        if verbose:
            click.echo(f"Exporting to: {output_path}")
        
        # Set up export parameters
        if config:
            # Load config file
            if verbose:
                click.echo(f"Loading export config: {config}")
            with open(config) as f:
                export_kwargs = json.load(f)
            
            # Extract input_specs from config if available and not provided via CLI
            if 'input_specs' in export_kwargs and not input_specs_dict:
                input_specs_dict = export_kwargs.pop('input_specs')
                if verbose:
                    click.echo(f"Using input_specs from config: {list(input_specs_dict.keys())}")
            
            # Convert dynamic_axes string keys to integers (JSON limitation workaround)
            if 'dynamic_axes' in export_kwargs:
                fixed_dynamic_axes = {}
                for input_name, axes in export_kwargs['dynamic_axes'].items():
                    fixed_dynamic_axes[input_name] = {int(k): v for k, v in axes.items()}
                export_kwargs['dynamic_axes'] = fixed_dynamic_axes
        else:
            # Use command line arguments
            export_kwargs = {
                'opset_version': opset_version
            }
        
        # Determine output directory for additional exports
        output_base = Path(output_path).parent
        output_name = Path(output_path).stem
        
        # JIT Graph dumping (before ONNX export)
        jit_info = None
        if jit_graph:
            if verbose:
                click.echo("🔍 Dumping TorchScript graph information...")
            
            try:
                # Import the JIT graph dumper
                import sys
                sys.path.append(str(Path(__file__).parent.parent))
                from jit_graph_dumper import dump_jit_graph_before_onnx_export
                
                jit_output_dir = output_base / f"{output_name}_jit_debug"
                jit_output_dir.mkdir(exist_ok=True)
                
                traced_model, jit_info = dump_jit_graph_before_onnx_export(
                    model, inputs, str(jit_output_dir)
                )
                
                if verbose:
                    scopes_found = jit_info['unified_scope_hierarchy']['total_unique_scopes']
                    click.echo(f"✅ JIT graph analysis complete: {scopes_found} scopes found")
                    click.echo(f"   Debug info saved to: {jit_output_dir}")
                
            except Exception as e:
                click.echo(f"Warning: JIT graph dumping failed: {e}", err=True)
                if verbose:
                    import traceback
                    click.echo(traceback.format_exc(), err=True)
        
        # FX Graph support removed
                if verbose:
                    import traceback
                    click.echo(traceback.format_exc(), err=True)
        
        # Generate inputs for strategies that don't have built-in generation
        inputs = None
        if strategy != 'htp_integrated':
            from .core.model_input_generator import generate_dummy_inputs
            
            inputs = generate_dummy_inputs(
                model_name_or_path=model_name_or_path,
                input_specs=input_specs_dict,
                exporter="onnx",
                **export_kwargs.get("input_generation_kwargs", {})
            )
            
            if verbose:
                click.echo(f"✅ Generated inputs: {list(inputs.keys())}")
        
        # Export with hierarchy preservation using appropriate strategy
        # HTP strategy - the only supported strategy
        if verbose:
            click.echo("🧠 Using HTP (Hierarchical Trace-and-Project) strategy")
        exporter = HierarchyExporter(strategy="htp")
        result = exporter.export(
            model=model,
            example_inputs=inputs,
            output_path=output_path,
            **export_kwargs
        )
        
        # Output results
        click.echo(f"✅ Export completed successfully!")
        
        # HTP strategy output
        click.echo(f"   ONNX Output: {output_path}")
        click.echo(f"   Sidecar: {output_path.replace('.onnx', '_hierarchy.json')}")
        if 'total_operations' in result:
            click.echo(f"   Total operations: {result['total_operations']}")
        if 'tagged_operations' in result:
            click.echo(f"   Tagged operations: {result['tagged_operations']}")
        click.echo(f"   Strategy: {result['strategy']}")
        
        # Report on additional exports
            # Enhanced Semantic strategy specific output
            click.echo(f"   ONNX Output: {output_path}")
            sidecar_path = output_path.replace('.onnx', '_enhanced_semantic_metadata.json')
            click.echo(f"   Enhanced Metadata: {sidecar_path}")
            click.echo(f"   Total ONNX nodes: {result['total_onnx_nodes']}")
            click.echo(f"   HF module mappings: {result['hf_module_mappings']}")
            click.echo(f"   Coverage: {(result['hf_module_mappings'] + result['operation_inferences'] + result['pattern_fallbacks'])/result['total_onnx_nodes']*100:.1f}%")
            click.echo(f"   Export time: {result['export_time']:.2f}s")
            if verbose:
                click.echo(f"   Confidence levels:")
                for conf, count in result['confidence_levels'].items():
                    click.echo(f"     {conf}: {count} nodes")
        elif strategy == 'usage_based':
            # Usage-based strategy output
            click.echo(f"   ONNX Output: {output_path}")
            click.echo(f"   Sidecar: {result['sidecar_path']}")
            click.echo(f"   Hierarchy nodes: {result['hierarchy_nodes']}")
            click.echo(f"   Unique modules: {result['unique_modules']}")
            click.echo(f"   Strategy: {result['strategy']}")
        else:
            # HTP strategy output
            click.echo(f"   ONNX Output: {output_path}")
            click.echo(f"   Sidecar: {output_path.replace('.onnx', '_hierarchy.json')}")
            if 'total_operations' in result:
                click.echo(f"   Total operations: {result['total_operations']}")
            if 'tagged_operations' in result:
                click.echo(f"   Tagged operations: {result['tagged_operations']}")
            click.echo(f"   Strategy: {result['strategy']}")
        
        # Report on additional exports
        if jit_info and jit_info.get('scope_statistics', {}).get('extraction_success'):
            scopes_found = jit_info['unified_scope_hierarchy']['total_unique_scopes']
            click.echo(f"   JIT Debug: {scopes_found} scopes extracted → {output_base}/{output_name}_jit_debug/")
        
        
        if verbose:
            # Show tag statistics
            try:
                stats = tag_utils.get_tag_statistics(output_path)
                click.echo("\nTag distribution:")
                for tag, count in stats.items():
                    click.echo(f"   {tag}: {count}")
            except Exception as e:
                click.echo(f"Warning: Could not load tag statistics: {e}", err=True)
                
            # Show JIT graph analysis if available
            if jit_info:
                click.echo(f"\nJIT Graph Analysis:")
                click.echo(f"   Extraction success: {jit_info.get('scope_statistics', {}).get('extraction_success', False)}")
                if jit_info.get('scope_statistics', {}).get('coverage_analysis'):
                    coverage = jit_info['scope_statistics']['coverage_analysis']
                    click.echo(f"   BERT modules found: {coverage.get('has_bert_modules', False)}")
                    click.echo(f"   Attention modules: {coverage.get('has_attention_modules', False)}")
                    click.echo(f"   Layer modules: {coverage.get('has_layer_modules', False)}")
            
        
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
        
        # Check sidecar file (try multiple formats)
        try:
            sidecar_data = tag_utils.load_tags_from_sidecar(onnx_path)
            
            # Handle different sidecar formats for counting
            if "tagged_nodes" in sidecar_data:
                # HTP format
                tagged_count = sum(1 for tag in sidecar_data["tagged_nodes"].values() if tag)
                click.echo(f"✅ Found HTP metadata file with {tagged_count} tagged operations")
            elif "node_tags" in sidecar_data:
                # Legacy format
                node_tags = sidecar_data.get('node_tags', {})
                click.echo(f"✅ Found sidecar file with {len(node_tags)} tagged operations")
            else:
                click.echo(f"✅ Found metadata file (unknown format)")
                
        except FileNotFoundError:
            click.echo("⚠️  No sidecar/metadata file found")
        except Exception as e:
            click.echo(f"❌ Invalid sidecar/metadata file: {e}", err=True)
            sys.exit(1)
        
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