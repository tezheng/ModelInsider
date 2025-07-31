"""
Example integration of advanced metadata features into modelexport CLI.

This shows how to add the most valuable features with minimal changes.
"""

import json
from pathlib import Path

import click

# These would be imported from the actual modules
from .metadata_cli_utils import MetadataCLI
from .metadata_patch_cli import MetadataPatchCLI


def add_analyze_query_support(analyze_command):
    """
    Enhance existing analyze command with query support.
    
    This decorator would be applied to the existing analyze command.
    """
    # Add query option
    analyze_command = click.option(
        '--query', '-q',
        help='Query metadata using JSON Pointer (/path/to/field) or pattern (find:type:pattern)'
    )(analyze_command)
    
    # Add validation option
    analyze_command = click.option(
        '--validate-consistency',
        is_flag=True,
        help='Validate internal consistency of metadata'
    )(analyze_command)
    
    # Wrap the original function
    original_func = analyze_command.callback
    
    def enhanced_analyze(onnx_path, query=None, validate_consistency=False, **kwargs):
        # First run original analyze logic
        result = original_func(onnx_path, **kwargs)
        
        # Load metadata (assuming it's in same directory with .json extension)
        metadata_path = Path(onnx_path).with_suffix('.json')
        if not metadata_path.exists():
            metadata_path = Path(onnx_path).parent / 'model_metadata.json'
        
        if metadata_path.exists():
            # Handle query
            if query:
                try:
                    query_result = MetadataCLI.query_metadata(metadata_path, query)
                    click.echo("\nQuery Result:")
                    click.echo(json.dumps(query_result, indent=2))
                except Exception as e:
                    click.echo(f"Query error: {e}", err=True)
            
            # Handle validation
            if validate_consistency:
                validation_result = MetadataCLI.validate_metadata_consistency(metadata_path)
                click.echo("\nConsistency Validation:")
                click.echo(f"Valid: {validation_result['valid']}")
                if not validation_result['valid']:
                    click.echo(f"Errors: {validation_result['summary']['errors']}")
                    click.echo(f"Warnings: {validation_result['summary']['warnings']}")
                    for issue in validation_result['issues'][:5]:  # Show first 5 issues
                        click.echo(f"  - [{issue['severity']}] {issue['type']}: {issue.get('message', '')}")
        
        return result
    
    analyze_command.callback = enhanced_analyze
    return analyze_command


def create_patch_command_group():
    """Create patch command group for metadata updates."""
    
    @click.group()
    def patch():
        """Update metadata files without re-exporting."""
        pass
    
    @patch.command('coverage')
    @click.argument('metadata_path', type=click.Path(exists=True))
    @click.option('--coverage', '-c', type=float, required=True, help='Coverage percentage')
    @click.option('--tagged', '-t', type=int, required=True, help='Number of tagged nodes')
    @click.option('--empty', '-e', type=int, default=0, help='Number of empty tags')
    def update_coverage(metadata_path, coverage, tagged, empty):
        """Update coverage statistics in metadata."""
        try:
            output = MetadataPatchCLI.update_coverage(
                Path(metadata_path), coverage, tagged, empty
            )
            click.echo(f"✅ Updated metadata saved to: {output}")
        except Exception as e:
            click.echo(f"❌ Error updating coverage: {e}", err=True)
    
    @patch.command('add-analysis')
    @click.argument('metadata_path', type=click.Path(exists=True))
    @click.argument('analysis_name')
    @click.argument('analysis_file', type=click.Path(exists=True))
    def add_analysis(metadata_path, analysis_name, analysis_file):
        """Add custom analysis results to metadata."""
        try:
            with open(analysis_file) as f:
                analysis_data = json.load(f)
            
            output = MetadataPatchCLI.add_custom_analysis(
                Path(metadata_path), analysis_name, analysis_data
            )
            click.echo(f"✅ Analysis added. Updated metadata saved to: {output}")
        except Exception as e:
            click.echo(f"❌ Error adding analysis: {e}", err=True)
    
    return patch


def enhance_export_with_validation(export_command):
    """
    Enhance export command with auto-validation.
    
    This would be integrated into the existing export command.
    """
    # Add validation option
    export_command = click.option(
        '--auto-validate/--no-auto-validate',
        default=True,
        help='Enable automatic model-specific validation'
    )(export_command)
    
    # In the export logic, after metadata generation:
    # if auto_validate:
    #     with open(metadata_path) as f:
    #         metadata = json.load(f)
    #     
    #     report = AutoValidationReport(metadata).generate_report()
    #     
    #     # Add validation results to metadata
    #     metadata['validation'] = {
    #         'auto_detected_type': report['detected_model_type'],
    #         'quality_score': report['quality_score'],
    #         'validation_timestamp': datetime.now().isoformat()
    #     }
    #     
    #     # Save updated metadata
    #     with open(metadata_path, 'w') as f:
    #         json.dump(metadata, f, indent=2)
    #     
    #     # Show validation summary
    #     click.echo(f"\nValidation Summary:")
    #     click.echo(f"  Model Type: {report['detected_model_type']}")
    #     click.echo(f"  Quality Score: {report['quality_score']:.1f}%")
    
    return export_command


# Example usage in CLI
def integrate_advanced_features(cli):
    """
    Main integration function that would be called from cli.py
    
    Example:
        from modelexport.strategies.htp.cli_integration_example import integrate_advanced_features
        integrate_advanced_features(cli)
    """
    # Find existing commands and enhance them
    for command in cli.commands.values():
        if command.name == 'analyze':
            add_analyze_query_support(command)
        elif command.name == 'export':
            enhance_export_with_validation(command)
    
    # Add new patch command group
    cli.add_command(create_patch_command_group())
    
    return cli


# Example standalone usage
if __name__ == "__main__":
    # Example queries that would work after integration
    example_queries = [
        # JSON Pointer queries
        "/modules/encoder.layer.0",
        "/tagging/coverage/coverage_percentage",
        "/model/total_parameters",
        
        # Pattern queries
        "find:modules:BertLayer",
        "find:tags:*/attention/*",
        "find:stats",
    ]
    
    print("Example queries after integration:")
    for query in example_queries:
        print(f"  modelexport analyze model.onnx --query '{query}'")
    
    print("\nExample patch commands:")
    print("  modelexport patch coverage metadata.json --coverage 98.5 --tagged 134")
    print("  modelexport patch add-analysis metadata.json complexity analysis.json")
    
    print("\nExample validation:")
    print("  modelexport analyze model.onnx --validate-consistency")
    print("  modelexport export bert model.onnx --auto-validate")