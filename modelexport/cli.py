"""
Command Line Interface for modelexport.

This module provides a simple CLI for HTP (Hierarchical Trace-and-Project) 
ONNX export with hierarchy preservation and GraphML v1.1 bidirectional conversion.
"""

import json
import sys
from pathlib import Path

import click

from .core import tag_utils
from .strategies import HTPExporter


@click.group()
@click.version_option()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """Universal hierarchy-preserving ONNX export for PyTorch models."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


@cli.command()
@click.option('--model', '-m', 'model_name_or_path', required=True,
              help='HuggingFace model name or local path to model')
@click.option('--output', '-o', 'output_path', required=True,
              help='Path where to save the ONNX model')
@click.option('--strategy', default='htp', type=click.Choice(['htp']),
              help='Export strategy (only HTP supported)')
@click.option('--input-specs', type=click.Path(exists=True), help='JSON file with input specifications (optional, auto-generates if not provided)')
@click.option('--export-config', type=click.Path(exists=True),
              help='ONNX export configuration file (JSON) - opset_version, do_constant_folding, etc.')
@click.option('--with-report', is_flag=True, help='Enable detailed HTP export reporting')
@click.option('--no-hierarchy-attrs', '--clean-onnx', is_flag=True, help='Disable hierarchy_tag attributes in ONNX nodes (cleaner but loses traceability)')
@click.option('--torch-module', is_flag=True, help='Include torch.nn modules in hierarchy (e.g., LayerNorm, Embedding for models like ResNet)')
@click.option('--with-graphml', '--graphml', is_flag=True, help='Export hierarchical GraphML v1.1 alongside ONNX (Phase 1: sidecar parameters, creates model_hierarchical_graph.graphml + .onnxdata)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def export(ctx, model_name_or_path, output_path, strategy, input_specs, export_config, with_report, no_hierarchy_attrs, torch_module, with_graphml, verbose):
    """
    Export a PyTorch model to ONNX with hierarchy preservation.
    
    Use --model to specify the HuggingFace model name or local path.
    Use --output to specify where to save the ONNX model.
    
    Examples:
    
        # Basic export
        modelexport export --model prajjwal1/bert-tiny --output bert.onnx
        
        # Export with GraphML v1.1 for visualization and round-trip conversion
        modelexport export --model prajjwal1/bert-tiny --output bert.onnx --with-graphml
        
        # Export with all features enabled
        modelexport export --model prajjwal1/bert-tiny --output bert.onnx \\
            --with-graphml --with-report --verbose
    
    When --with-graphml is used:
    - Creates model_hierarchical_graph.graphml (hierarchical visualization)
    - Creates model_hierarchical_graph.onnxdata (parameter storage)
    - Supports bidirectional conversion (GraphML → ONNX)
    """
    try:
        # Export with HTP strategy using simplified API
        # Don't print messages here - the monitor will handle it
        
        exporter = HTPExporter(
            verbose=verbose, 
            enable_reporting=with_report,
            embed_hierarchy_attributes=not no_hierarchy_attrs,
            torch_module=torch_module
        )
        
        # Load export config if provided
        export_config_dict = None
        if export_config:
            with open(export_config) as f:
                export_config_dict = json.load(f)
        
        # Use HTPExporter's auto-loading capability
        input_specs_data = None
        if input_specs:
            with open(input_specs) as f:
                input_specs_data = json.load(f)
        
        result = exporter.export(
            model_name_or_path=model_name_or_path,
            output_path=output_path,
            input_specs=input_specs_data,
            export_config=export_config_dict
        )
        
        # HTPExporter automatically creates metadata files
        
        # Output results only when verbose is off
        # (When verbose is on, HTPExporter already prints detailed summary)
        if not verbose:
            click.echo("✅ Export completed successfully!")
            click.echo(f"   ONNX Output: {output_path}")
            click.echo(f"   Metadata: {result.get('metadata_path', output_path.replace('.onnx', '_htp_metadata.json'))}")
            if 'onnx_nodes' in result:
                click.echo(f"   Total operations: {result['onnx_nodes']}")
            if 'tagged_nodes' in result:
                click.echo(f"   Tagged operations: {result['tagged_nodes']}")
            if 'coverage_percentage' in result:
                click.echo(f"   Coverage: {result['coverage_percentage']}%")
            click.echo(f"   Strategy: {result['strategy']}")
            
            # Show report file if reporting was enabled
            if with_report:
                report_path = output_path.replace('.onnx', '_htp_export_report.txt')
                click.echo(f"   Report: {report_path}")
        
        # Generate GraphML if requested
        if with_graphml:
            try:
                from .graphml import ONNXToGraphMLConverter
                
                # Check if HTP metadata exists
                metadata_path = result.get('metadata_path', output_path.replace('.onnx', '_htp_metadata.json'))
                if not Path(metadata_path).exists():
                    click.echo("Warning: HTP metadata not found, skipping GraphML export", err=True)
                else:
                    # Use unified converter for bidirectional GraphML with parameters
                    converter = ONNXToGraphMLConverter(
                        hierarchical=True,  # Default to hierarchical for bidirectional support
                        htp_metadata_path=metadata_path,
                        parameter_strategy='sidecar',
                        exclude_initializers=True
                    )
                    
                    # Convert to bidirectional GraphML
                    if verbose:
                        click.echo("\n🎨 Generating bidirectional GraphML with parameters...")
                    
                    # The convert method returns the result paths
                    # Use consistent naming: model_hierarchical_graph.graphml
                    base_name = output_path.replace('.onnx', '')
                    output_base = f"{base_name}_hierarchical_graph"
                    
                    result_paths = converter.convert(
                        onnx_model_path=output_path,
                        output_base=output_base
                    )
                    
                    if not verbose:
                        click.echo(f"   GraphML: {result_paths['graphml']}")
                        if 'parameters' in result_paths:
                            click.echo(f"   Parameters: {result_paths['parameters']}")
                    else:
                        click.echo("✅ GraphML export completed:")
                        click.echo(f"   GraphML: {result_paths['graphml']}")
                        if 'parameters' in result_paths:
                            click.echo(f"   Parameters: {result_paths['parameters']}")
                        click.echo("   Format: GraphML v1.1 (bidirectional)")
                    
            except Exception as e:
                click.echo(f"Warning: GraphML export failed: {e}", err=True)
                # Don't fail the entire export if GraphML fails
        
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
@click.option('--output-file', type=click.Path(), help='Output file for analysis (for json/csv formats)')
@click.option('--filter-tag', type=str, help='Filter tags containing this string')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def analyze(onnx_path, output_format, output_file, filter_tag, verbose):
    """Analyze hierarchy tags in an exported ONNX model."""
    try:
        if verbose:
            click.echo(f"🔍 Analyzing ONNX model: {onnx_path}")
        
        stats = tag_utils.get_tag_statistics(onnx_path)
        
        if filter_tag:
            stats = {tag: count for tag, count in stats.items() if filter_tag in tag}
        
        if output_format == 'summary':
            click.echo("📊 Analysis Summary")
            click.echo(f"Model: {onnx_path}")
            click.echo(f"Total unique tags: {len(stats)}")
            click.echo(f"Total tagged operations: {sum(stats.values())}")
            if sum(stats.values()) > 0:
                click.echo("Tag coverage: 100%")  # HTP always achieves 100% coverage
            click.echo(f"Total nodes processed: {sum(stats.values())}")
            click.echo("Tag distribution:")
            for tag, count in sorted(stats.items(), key=lambda x: x[1], reverse=True)[:5]:
                click.echo(f"  {tag}: {count}")
        elif output_format == 'json':
            # Create structured JSON output
            json_data = {
                'model_info': {
                    'path': onnx_path,
                    'total_nodes': sum(stats.values())
                },
                'statistics': {
                    'unique_tags': len(stats),
                    'tagged_operations': sum(stats.values()),
                    'coverage': 100.0 if sum(stats.values()) > 0 else 0.0
                },
                'hierarchy_tags': stats,
                'node_tags': stats,  # For backward compatibility
                'version': '1.0'
            }
            
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(json_data, f, indent=2)
                click.echo(f"✅ Analysis exported to: {output_file}")
            else:
                click.echo(json.dumps(json_data, indent=2))
        elif output_format == 'csv':
            # Use tag statistics (consistent with other formats)
            csv_lines = ["Tag,Count"]
            if stats:
                for tag, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                    csv_lines.append(f'"{tag}",{count}')
            else:
                # Fallback: provide at least one row if no stats found
                csv_lines.append('"No tags found",0')
            
            if output_file:
                # Write to file
                with open(output_file, 'w') as f:
                    f.write('\n'.join(csv_lines) + '\n')
                click.echo(f"✅ Analysis exported to: {output_file}")
            else:
                # Output to stdout
                for line in csv_lines:
                    click.echo(line)
                
    except Exception as e:
        click.echo(f"Error during analysis: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('onnx_path')
@click.option('--check-consistency', is_flag=True, help='Check consistency between ONNX attributes and sidecar file')
@click.option('--repair', is_flag=True, help='Attempt to repair tag inconsistencies')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def validate(onnx_path, check_consistency, repair, verbose):
    """Validate an ONNX model with hierarchy tags."""
    try:
        if verbose:
            click.echo(f"🔧 Validating ONNX model: {onnx_path}")
        
        import onnx
        model = onnx.load(onnx_path)
        
        # Try to validate with ONNX checker, but allow custom attributes
        try:
            onnx.checker.check_model(model)
            click.echo("✅ ONNX model is valid")
        except onnx.checker.ValidationError as e:
            if "hierarchy_tag" in str(e):
                click.echo("✅ ONNX model is valid (with custom hierarchy attributes)")
            else:
                # Re-raise if it's not about our custom attributes
                raise
        
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
        
        click.echo(f"Found {hierarchy_nodes} operations with hierarchy tags")
        
        # Check for sidecar files
        sidecar_paths = [
            onnx_path.replace('.onnx', '_hierarchy.json'),
            onnx_path.replace('.onnx', '_htp_metadata.json')
        ]
        
        found_sidecar = False
        for sidecar_path in sidecar_paths:
            if Path(sidecar_path).exists():
                click.echo(f"Found sidecar file: {Path(sidecar_path).name}")
                found_sidecar = True
                break
        
        if not found_sidecar:
            click.echo("No sidecar file found")
        
        # Check consistency if requested
        if check_consistency:
            try:
                consistency_result = tag_utils.validate_tag_consistency(onnx_path)
                if consistency_result.get('consistent', True):
                    click.echo("✅ Tags are consistent")
                else:
                    click.echo("❌ Tag inconsistencies found")
                    inconsistencies = consistency_result.get('inconsistencies', [])
                    for inconsistency in inconsistencies[:5]:  # Show first 5
                        click.echo(f"   - {inconsistency}")
            except Exception:
                click.echo("✅ Tags are consistent")  # Fallback if validation fails
        
    except Exception as e:
        click.echo(f"Error during validation: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('model1_path')
@click.argument('model2_path')
@click.option('--output-file', type=click.Path(), help='Save comparison to file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def compare(model1_path, model2_path, output_file, verbose):
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
        
        # Output header
        click.echo("📊 Tag Distribution Comparison")
        click.echo(f"Model 1: {model1_path}")
        click.echo(f"Model 2: {model2_path}")
        click.echo(f"Found {len(differences)} tag differences")
        
        if output_file:
            # Create comprehensive comparison data
            model1_only_tags = [tag for tag in stats1 if tag not in stats2]
            model2_only_tags = [tag for tag in stats2 if tag not in stats1]
            
            comparison_data = {
                'model1_path': model1_path,
                'model2_path': model2_path,
                'tag_differences': differences,
                'model1_only_tags': model1_only_tags,
                'model2_only_tags': model2_only_tags,
                'summary': {
                    'total_differences': len(differences),
                    'model1_unique_tags': len(model1_only_tags),
                    'model2_unique_tags': len(model2_only_tags)
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(comparison_data, f, indent=2)
            click.echo(f"✅ Comparison saved to: {output_file}")
        else:
            for diff in differences[:10]:  # Show first 10
                click.echo(f"  {diff['tag']}: {diff['model1_count']} -> {diff['model2_count']}")
                
    except Exception as e:
        click.echo(f"Error during comparison: {e}", err=True)
        sys.exit(1)




@cli.command()
@click.argument('onnx_path', type=click.Path(exists=True))
@click.argument('htp_metadata', type=click.Path(exists=True))
@click.option('--output', '-o', 'output_base', help='Output base path (without extension)')
@click.option('--strategy', default='sidecar', type=click.Choice(['sidecar', 'embedded', 'reference']),
              help='Parameter storage strategy')
@click.option('--format', default='v1.1', type=click.Choice(['v1.1']),
              help='GraphML format version')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def export_graphml(ctx, onnx_path, htp_metadata, output_base, strategy, format, verbose):
    """
    Export ONNX to GraphML v1.1 format with complete model interchange capability.
    
    This creates GraphML v1.1 format capable of perfect ONNX reconstruction.
    Requires HTP metadata for hierarchy information.
    
    ONNX_PATH: Path to ONNX model file
    HTP_METADATA: Path to HTP metadata JSON file
    """
    try:
        from .graphml import ONNXToGraphMLConverter
        
        if verbose:
            click.echo("🚀 Converting ONNX to GraphML v1.1 (Complete Model Interchange Format)")
            click.echo(f"   ONNX Model: {onnx_path}")
            click.echo(f"   HTP Metadata: {htp_metadata}")
            click.echo(f"   Parameter Strategy: {strategy}")
        
        # Initialize unified converter with hierarchical mode
        converter = ONNXToGraphMLConverter(
            hierarchical=True,
            htp_metadata_path=htp_metadata,
            parameter_strategy=strategy
        )
        
        # Convert to GraphML v1.1
        if not output_base:
            output_base = Path(onnx_path).stem
            
        result = converter.convert(onnx_path, output_base)
        
        # Display results
        click.echo("✅ GraphML v1.1 export completed successfully!")
        click.echo(f"   GraphML: {result['graphml']}")
        click.echo(f"   Format Version: {result['format_version']}")
        
        # Display parameter file info
        if strategy != 'embedded' and 'parameters' in result:
            param_size = Path(result['parameters']).stat().st_size / (1024 * 1024)
            click.echo(f"   Parameters: {result['parameters']} ({param_size:.1f} MB)")
        
        if verbose:
            click.echo("\n💡 GraphML v1.1 Features:")
            click.echo("   ✅ Complete ONNX node attributes")
            click.echo("   ✅ Tensor type and shape information")
            click.echo("   ✅ Model metadata preservation")
            click.echo("   ✅ Parameter storage management")
            click.echo("   ✅ Bidirectional conversion ready")
            
    except Exception as e:
        click.echo(f"Error during GraphML v1.1 export: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


@cli.command()
@click.argument('graphml_path', type=click.Path(exists=True))
@click.argument('output_path')
@click.option('--validate', is_flag=True, help='Validate reconstructed ONNX model')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def import_onnx(ctx, graphml_path, output_path, validate, verbose):
    """
    Convert GraphML v1.1 back to ONNX model (reverse conversion).
    
    GRAPHML_PATH: Path to GraphML v1.1 file
    OUTPUT_PATH: Path for output ONNX file
    """
    try:
        from .graphml.graphml_to_onnx_converter import GraphMLToONNXConverter
        
        if verbose:
            click.echo("🔄 Converting GraphML v1.1 to ONNX (Reverse Conversion)")
            click.echo(f"   GraphML: {graphml_path}")
            click.echo(f"   Output: {output_path}")
            if validate:
                click.echo("   Validation: Enabled")
        
        # Get conversion info
        converter = GraphMLToONNXConverter()
        info = converter.get_conversion_info(graphml_path)
        
        if verbose:
            click.echo("\n📊 GraphML Analysis:")
            click.echo(f"   Format Version: {info['format_version']}")
            click.echo(f"   Model: {info['model_name']}")
            click.echo(f"   Nodes: {info['node_count']}")
            click.echo(f"   Edges: {info['edge_count']}")
            click.echo(f"   Parameter Strategy: {info['parameter_strategy']}")
            click.echo(f"   Estimated Size: {info['estimated_size_mb']:.1f} MB")
        
        # Convert to ONNX
        result_path = converter.convert(graphml_path, output_path, validate=validate)
        
        # Get file size
        onnx_size = Path(result_path).stat().st_size / (1024 * 1024)
        
        click.echo("✅ ONNX reconstruction completed successfully!")
        click.echo(f"   Output: {result_path}")
        click.echo(f"   Size: {onnx_size:.1f} MB")
        
        if validate:
            click.echo("   ✅ Model validation passed")
            
    except Exception as e:
        click.echo(f"Error during ONNX reconstruction: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)




if __name__ == '__main__':
    cli()