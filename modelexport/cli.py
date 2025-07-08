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
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import shutil

from .strategies.htp.htp_hierarchy_exporter import HierarchyExporter
from .strategies.htp.htp_integrated_exporter import export_with_htp_integrated
from .strategies.htp.htp_integrated_exporter_with_reporting import export_with_htp_integrated_reporting
from .strategies.fx.fx_hierarchy_exporter import FXHierarchyExporter
from .strategies.usage_based.usage_based_exporter import UsageBasedExporter
from .core.enhanced_semantic_exporter import EnhancedSemanticExporter
from .core import tag_utils


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
@click.option('--input-shape', help='Input shape as comma-separated values (e.g., 1,3,224,224) for vision models')
@click.option('--strategy', default='htp_integrated', 
              type=click.Choice(['htp_integrated', 'enhanced_semantic', 'usage_based', 'htp', 'fx_graph']), 
              help='Tagging strategy to use')
@click.option('--opset-version', default=14, type=int,
              help='ONNX opset version to use')
@click.option('--config', type=click.Path(exists=True),
              help='Export configuration file (JSON)')
@click.option('--temp-dir', type=click.Path(),
              help='Directory for temporary files (default: system temp)')
@click.option('--jit-graph', is_flag=True,
              help='Dump TorchScript graph information before ONNX export (preserves context)')
@click.option('--fx-graph', type=click.Choice(['symbolic_trace', 'torch_export', 'both']),
              help='Export FX graph representation (dynamo=False alternative)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def export(ctx, model_name_or_path, output_path, input_shape, strategy, opset_version, config, temp_dir, jit_graph, fx_graph, verbose):
    """
    Export a PyTorch model to ONNX with hierarchy preservation.
    
    MODEL_NAME_OR_PATH: HuggingFace model name or local path to model
    OUTPUT_PATH: Path where to save the ONNX model
    
    Examples:
    \b
        # Export BERT model with HTP Integrated strategy (default, recommended)
        modelexport export prajjwal1/bert-tiny bert.onnx --config export_config.json
        
        # Export with verbose output showing optimization details
        modelexport export prajjwal1/bert-tiny bert.onnx --config config.json --verbose
        
        # Export with Enhanced Semantic mapping (alternative strategy)
        modelexport export prajjwal1/bert-tiny bert.onnx --strategy enhanced_semantic --config config.json
        
        # Export with legacy HTP strategy
        modelexport export prajjwal1/bert-tiny bert.onnx --strategy htp --config config.json --jit-graph
        
        # Export with FX graph alternative (for analysis)
        modelexport export prajjwal1/bert-tiny bert.onnx --config config.json --fx-graph both
        
        # Export with FX Graph strategy (structural analysis)
        modelexport export prajjwal1/bert-tiny bert.onnx --strategy fx_graph
        
        # Export vision model with input shape
        modelexport export resnet50 resnet.onnx --input-shape 1,3,224,224
        
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
            from transformers import AutoModel, AutoTokenizer
            
            # Load model
            model = AutoModel.from_pretrained(model_name_or_path)
            
            # Prepare inputs
            if config and Path(config).exists():
                with open(config, 'r') as f:
                    config_data = json.load(f)
                if 'input_specs' in config_data:
                    # Generate dummy inputs from specs
                    inputs = {}
                    for name, spec in config_data['input_specs'].items():
                        dtype = torch.long if spec.get('dtype') == 'int' else torch.float32
                        # Create dummy tensor with shape from dynamic_axes or default
                        if 'dynamic_axes' in config_data and name in config_data['dynamic_axes']:
                            # Default shape: batch_size=1, sequence_length=128
                            shape = [1, 128]  # Common for BERT-like models
                        else:
                            shape = [1, 128]  # Default fallback
                        
                        # Generate values within specified range
                        if 'range' in spec:
                            min_val, max_val = spec['range']
                            if dtype == torch.long:
                                inputs[name] = torch.randint(min_val, max_val + 1, shape, dtype=dtype)
                            else:
                                inputs[name] = torch.rand(shape, dtype=dtype) * (max_val - min_val) + min_val
                        else:
                            inputs[name] = torch.ones(shape, dtype=dtype)
                else:
                    click.echo("Error: Config file must contain 'input_specs' to generate inputs", err=True)
                    sys.exit(1)
            elif input_shape:
                # Use provided input shape for vision models
                shape = [int(x) for x in input_shape.split(',')]
                inputs = torch.randn(shape)
            else:
                click.echo("Error: Either --config with input_specs or --input-shape is required", err=True)
                sys.exit(1)
            
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
            with open(config, 'r') as f:
                export_kwargs = json.load(f)
            
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
        
        # FX Graph export (alternative to ONNX)
        fx_info = None
        if fx_graph:
            if verbose:
                click.echo(f"🔄 Exporting FX graph using method: {fx_graph}")
            
            try:
                # Import the FX graph exporter
                import sys
                sys.path.append(str(Path(__file__).parent.parent))
                from fx_graph_exporter import export_fx_graph_cli
                
                fx_output_path = output_base / f"{output_name}_fx_graph"
                
                fx_info = export_fx_graph_cli(
                    model, inputs, str(fx_output_path), method=fx_graph
                )
                
                if verbose and fx_info.get('success'):
                    click.echo(f"✅ FX graph export complete")
                    click.echo(f"   FX graph saved to: {fx_output_path}")
                elif fx_info.get('error'):
                    click.echo(f"Warning: FX graph export failed: {fx_info['error']}", err=True)
                
            except Exception as e:
                click.echo(f"Warning: FX graph export failed: {e}", err=True)
                if verbose:
                    import traceback
                    click.echo(traceback.format_exc(), err=True)
        
        # Export with hierarchy preservation using appropriate strategy
        if strategy == 'htp_integrated':
            if verbose:
                click.echo("🚀 Using HTP Integrated strategy (TracingHierarchyBuilder + ONNXNodeTagger)")
            
            result = export_with_htp_integrated_reporting(
                model=model,
                example_inputs=inputs,
                output_path=output_path,
                verbose=verbose,
                **export_kwargs
            )
        elif strategy == 'enhanced_semantic':
            if verbose:
                click.echo("🎯 Using Enhanced Semantic mapping with HuggingFace-level understanding")
            
            # For Enhanced Semantic, we need to prepare args tuple for the model
            if isinstance(inputs, dict):
                # Convert dict inputs to tuple of tensors for torch.onnx.export
                args = tuple(inputs.values())
            elif isinstance(inputs, torch.Tensor):
                args = (inputs,)
            else:
                args = inputs
            
            exporter = EnhancedSemanticExporter(verbose=verbose)
            result = exporter.export(
                model=model,
                args=args,
                output_path=output_path,
                **export_kwargs
            )
        elif strategy == 'fx_graph':
            if verbose:
                click.echo("🚀 Using FX Graph-based hierarchy preservation")
            exporter = FXHierarchyExporter()
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=output_path,
                **export_kwargs
            )
        elif strategy == 'usage_based':
            if verbose:
                click.echo("🔄 Using Usage-based hierarchy preservation (legacy)")
            exporter = UsageBasedExporter()
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=output_path,
                **export_kwargs
            )
        else:  # htp strategy
            if verbose:
                click.echo("🧠 Using HTP (Hierarchical Trace-and-Project) strategy")
            exporter = HierarchyExporter(strategy=strategy)
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=output_path,
                **export_kwargs
            )
        
        # Output results
        click.echo(f"✅ Export completed successfully!")
        
        if strategy == 'htp_integrated':
            # HTP Integrated strategy specific output
            click.echo(f"   ONNX Output: {output_path}")
            sidecar_path = output_path.replace('.onnx', '_htp_integrated_metadata.json')
            click.echo(f"   Metadata: {sidecar_path}")
            click.echo(f"   Hierarchy modules: {result['hierarchy_modules']}")
            click.echo(f"   Tagged nodes: {result['tagged_nodes']}")
            click.echo(f"   Coverage: {result['coverage_percentage']:.1f}%")
            click.echo(f"   Export time: {result['export_time']:.2f}s")
            if verbose:
                click.echo(f"   Statistics:")
                click.echo(f"     ONNX nodes: {result['onnx_nodes']}")
                click.echo(f"     Empty tags: {result['empty_tags']} (MUST be 0)")
                click.echo(f"     Optimized hierarchy: {result['hierarchy_modules']} modules (vs ~48 total)")
        elif strategy == 'enhanced_semantic':
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
        elif strategy == 'fx_graph':
            # FX strategy specific output
            click.echo(f"   ONNX Output: {result['onnx_path']}")
            click.echo(f"   Sidecar: {result['sidecar_path']}")
            click.echo(f"   Module Info: {result['module_info_path']}")
            click.echo(f"   FX Nodes with hierarchy: {result['hierarchy_nodes']}")
            click.echo(f"   Unique modules: {result['unique_modules']}")
            click.echo(f"   Strategy: {result['strategy']}")
            click.echo(f"   Export time: {result['export_time']:.2f}s")
            if verbose:
                fx_stats = result['fx_graph_stats']
                click.echo(f"   FX Graph stats:")
                click.echo(f"     Total FX nodes: {fx_stats['total_fx_nodes']}")
                click.echo(f"     Coverage ratio: {fx_stats['coverage_ratio']:.1%}")
                click.echo(f"     Module types: {len(fx_stats['module_types_found'])}")
        else:
            # Legacy strategy output
            click.echo(f"   ONNX Output: {output_path}")
            click.echo(f"   Sidecar: {output_path.replace('.onnx', '_hierarchy.json')}")
            click.echo(f"   Total operations: {result['total_operations']}")
            click.echo(f"   Tagged operations: {result['tagged_operations']}")
            click.echo(f"   Strategy: {result['strategy']}")
        
        # Report on additional exports
        if jit_info and jit_info.get('scope_statistics', {}).get('extraction_success'):
            scopes_found = jit_info['unified_scope_hierarchy']['total_unique_scopes']
            click.echo(f"   JIT Debug: {scopes_found} scopes extracted → {output_base}/{output_name}_jit_debug/")
        
        if fx_info and fx_info.get('success'):
            click.echo(f"   FX Graph: {fx_graph} method → {output_base}/{output_name}_fx_graph")
        
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
            
            # Show FX graph details if available  
            if fx_info and fx_info.get('success'):
                click.echo(f"\nFX Graph Export:")
                click.echo(f"   Method: {fx_info.get('method', 'unknown')}")
                click.echo(f"   Execution test: {fx_info.get('execution_test', 'unknown')}")
                if 'fx_graph_analysis' in fx_info:
                    analysis = fx_info['fx_graph_analysis']
                    click.echo(f"   Total nodes: {analysis.get('total_nodes', 'unknown')}")
                    click.echo(f"   Node types: {len(analysis.get('node_types', {}))}")
        
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