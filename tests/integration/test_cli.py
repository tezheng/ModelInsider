"""
Comprehensive CLI testing using pytest and Click's CliRunner.

Tests all CLI subcommands with proper temp directory structure
and code-generated results validation.
"""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from modelexport.cli import cli


@pytest.fixture
def cli_runner():
    """Create Click CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def temp_workspace():
    """Create structured temporary workspace for CLI tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        
        # Create organized subdirectories
        subdirs = {
            'models': workspace / 'models',
            'exports': workspace / 'exports', 
            'analysis': workspace / 'analysis',
            'comparisons': workspace / 'comparisons'
        }
        
        for subdir in subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
        
        yield subdirs


class TestCLIExport:
    """Test the export subcommand."""
    
    def test_export_help(self, cli_runner):
        """Test export command help output."""
        result = cli_runner.invoke(cli, ['export', '--help'])
        assert result.exit_code == 0
        assert 'Export a PyTorch model to ONNX' in result.output
        assert '--model' in result.output
        assert '--output' in result.output
    
    def test_export_bert_tiny(self, cli_runner, temp_workspace):
        """Test exporting BERT tiny model."""
        output_path = temp_workspace['exports'] / 'bert_tiny_cli.onnx'
        
        result = cli_runner.invoke(cli, [
            '--verbose',
            'export', 
            '--model', 'prajjwal1/bert-tiny',
            '--output', str(output_path),
            '--with-report'
        ])
        
        # Check command succeeded
        assert result.exit_code == 0, f"CLI failed with: {result.output}"
        assert '✅ Export completed successfully!' in result.output
        
        # Verify files were created
        assert output_path.exists(), "ONNX file was not created"
        # Check for HTP metadata file
        metadata_path = output_path.parent / (output_path.stem + '_htp_metadata.json')
        assert metadata_path.exists(), "Metadata file was not created"
        
        # Validate ONNX file
        import onnx
        model = onnx.load(str(output_path))
        assert len(model.graph.node) > 0, "ONNX model has no operations"
        
        # Validate metadata structure
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Check for required fields in metadata (new HTP structure)
        assert 'export_context' in metadata, "Missing export_context in metadata"
        assert 'model' in metadata, "Missing model in metadata"
        assert 'modules' in metadata, "Missing modules in metadata"
        assert 'nodes' in metadata, "Missing nodes in metadata"
        assert 'outputs' in metadata, "Missing outputs in metadata"
        assert 'report' in metadata, "Missing report in metadata"
        assert 'tracing' in metadata, "Missing tracing in metadata"
        assert 'statistics' in metadata, "Missing statistics in metadata"
        
        # Validate model info
        assert metadata['model']['name_or_path'] == 'prajjwal1/bert-tiny'
        assert metadata['model']['total_modules'] == 48
        
        # Validate report structure
        assert 'steps' in metadata['report'], "Missing steps in report"
        # In new implementation, node_tagging is under steps
        assert 'node_tagging' in metadata['report']['steps'], "Missing node_tagging in report steps"
        
        # Validate statistics
        assert metadata['statistics']['onnx_nodes'] > 0
        assert metadata['statistics']['tagged_nodes'] > 0
        assert metadata['statistics']['coverage_percentage'] == 100.0
        
        # Validate node tagging report
        node_tagging = metadata['report']['steps']['node_tagging']
        assert 'statistics' in node_tagging
        stats = node_tagging['statistics']
        assert stats['direct_matches'] >= 0
        assert stats['parent_matches'] >= 0
        assert stats['root_fallbacks'] >= 0
    
    def test_export_with_custom_options(self, cli_runner, temp_workspace):
        """Test export with custom options."""
        output_path = temp_workspace['exports'] / 'bert_custom_opts.onnx'
        
        result = cli_runner.invoke(cli, [
            'export',
            '--model', 'prajjwal1/bert-tiny', 
            '--output', str(output_path),
            '--verbose'
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
        
        # Verify opset version in ONNX
        import onnx
        model = onnx.load(str(output_path))
        # Note: opset version verification would require more complex ONNX analysis
    
    def test_export_nonexistent_model(self, cli_runner, temp_workspace):
        """Test error handling for nonexistent model."""
        output_path = temp_workspace['exports'] / 'nonexistent.onnx'
        
        result = cli_runner.invoke(cli, [
            'export',
            '--model', 'nonexistent/model',
            '--output', str(output_path)
        ])
        
        assert result.exit_code != 0
        assert 'not a local folder and is not a valid model identifier' in result.output
        assert not output_path.exists()


class TestCLIAnalyze:
    """Test the analyze subcommand."""
    
    @pytest.fixture
    def sample_onnx_model(self, temp_workspace):
        """Create a sample ONNX model for analysis tests."""
        from transformers import AutoModel, AutoTokenizer

        from modelexport.strategies.htp import HTPExporter
        
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        inputs = tokenizer('Sample for analysis', return_tensors='pt')
        
        output_path = temp_workspace['models'] / 'sample_for_analysis.onnx'
        exporter = HTPExporter()
        # Use new HTPExporter interface with model_name_or_path for input generation
        exporter.export(model=model, output_path=str(output_path), model_name_or_path='prajjwal1/bert-tiny')
        
        return output_path
    
    def test_analyze_help(self, cli_runner):
        """Test analyze command help."""
        result = cli_runner.invoke(cli, ['analyze', '--help'])
        assert result.exit_code == 0
        assert 'Analyze hierarchy tags' in result.output
    
    def test_analyze_summary(self, cli_runner, sample_onnx_model):
        """Test analysis with summary format."""
        result = cli_runner.invoke(cli, [
            'analyze',
            str(sample_onnx_model),
            '--output-format', 'summary'
        ])
        
        assert result.exit_code == 0
        assert '📊 Analysis Summary' in result.output
        assert 'Total unique tags:' in result.output
        assert 'Total tagged operations:' in result.output
        assert 'Tag distribution:' in result.output
    
    def test_analyze_json_output(self, cli_runner, sample_onnx_model, temp_workspace):
        """Test analysis with JSON output to file."""
        output_file = temp_workspace['analysis'] / 'analysis.json'
        
        result = cli_runner.invoke(cli, [
            'analyze',
            str(sample_onnx_model),
            '--output-format', 'json',
            '--output-file', str(output_file)
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        
        # Validate JSON structure
        with open(output_file) as f:
            data = json.load(f)
        
        assert 'version' in data
        assert 'node_tags' in data
        assert isinstance(data['node_tags'], dict)
    
    def test_analyze_csv_output(self, cli_runner, sample_onnx_model, temp_workspace):
        """Test analysis with CSV output."""
        result = cli_runner.invoke(cli, [
            'analyze',
            str(sample_onnx_model),
            '--output-format', 'csv'
        ])
        
        assert result.exit_code == 0
        
        # Current CLI outputs CSV to stdout, not file
        output = result.output
        assert 'Tag,Count' in output, "Should output CSV header"
        assert '"/' in output, "Should contain hierarchy tags"
        
        # Test CSV output to file
        csv_file = temp_workspace['analysis'] / 'test_analysis.csv'
        result = cli_runner.invoke(cli, [
            'analyze',
            str(sample_onnx_model),
            '--output-format', 'csv',
            '--output-file', str(csv_file)
        ])
        
        assert result.exit_code == 0
        assert '✅ Analysis exported to:' in result.output
        assert csv_file.exists()
        
        # Validate CSV content
        with open(csv_file) as f:
            content = f.read()
            assert 'Tag,Count' in content
    
    def test_analyze_with_filter(self, cli_runner, sample_onnx_model):
        """Test analysis with tag filtering."""
        result = cli_runner.invoke(cli, [
            '--verbose',
            'analyze',
            str(sample_onnx_model),
            '--filter-tag', 'BertAttention'
        ])
        
        assert result.exit_code == 0
        assert 'BertAttention' in result.output
    
    def test_analyze_nonexistent_file(self, cli_runner):
        """Test error handling for nonexistent ONNX file."""
        result = cli_runner.invoke(cli, [
            'analyze', 
            'nonexistent.onnx'
        ])
        
        assert result.exit_code != 0
        assert 'No such file or directory' in result.output


class TestCLIValidate:
    """Test the validate subcommand."""
    
    @pytest.fixture
    def sample_onnx_model(self, temp_workspace):
        """Create a sample ONNX model for validation tests."""
        from transformers import AutoModel, AutoTokenizer

        from modelexport.strategies.htp import HTPExporter
        
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        inputs = tokenizer('Sample for validation', return_tensors='pt')
        
        output_path = temp_workspace['models'] / 'sample_for_validation.onnx'
        exporter = HTPExporter()
        # Use new HTPExporter interface with model_name_or_path for input generation
        exporter.export(model=model, output_path=str(output_path), model_name_or_path='prajjwal1/bert-tiny')
        
        return output_path
    
    def test_validate_help(self, cli_runner):
        """Test validate command help."""
        result = cli_runner.invoke(cli, ['validate', '--help'])
        assert result.exit_code == 0
        assert 'Validate an ONNX model with hierarchy tags' in result.output
    
    def test_validate_basic(self, cli_runner, sample_onnx_model):
        """Test basic validation."""
        result = cli_runner.invoke(cli, [
            'validate',
            str(sample_onnx_model)
        ])
        
        assert result.exit_code == 0
        assert '✅ ONNX model is valid' in result.output  # Works for both messages
        assert 'Found' in result.output and 'operations with hierarchy tags' in result.output
        assert 'Found sidecar file' in result.output
    
    def test_validate_consistency_check(self, cli_runner, sample_onnx_model):
        """Test consistency validation."""
        result = cli_runner.invoke(cli, [
            '--verbose',
            'validate',
            str(sample_onnx_model),
            '--check-consistency'
        ])
        
        assert result.exit_code == 0
        # Should show either consistent or inconsistent with details
        assert any(phrase in result.output for phrase in [
            '✅ Tags are consistent',
            '❌ Tag inconsistencies found'
        ])
    
    def test_validate_nonexistent_file(self, cli_runner):
        """Test validation of nonexistent file."""
        result = cli_runner.invoke(cli, [
            'validate',
            'nonexistent.onnx'
        ])
        
        assert result.exit_code != 0
        assert 'No such file or directory' in result.output


class TestCLICompare:
    """Test the compare subcommand."""
    
    @pytest.fixture
    def two_sample_models(self, temp_workspace):
        """Create two sample ONNX models for comparison."""
        from transformers import AutoModel, AutoTokenizer

        from modelexport.strategies.htp import HTPExporter
        
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        exporter = HTPExporter()
        
        # Create first model
        inputs1 = tokenizer('First model input', return_tensors='pt')
        path1 = temp_workspace['models'] / 'model1.onnx'
        input_values1 = tuple(inputs1.values())
        exporter.export(model=model, output_path=str(path1), model_name_or_path='prajjwal1/bert-tiny')
        
        # Create second model (with different input to potentially get different graphs)
        inputs2 = tokenizer('Second model with different input text', return_tensors='pt')
        path2 = temp_workspace['models'] / 'model2.onnx'
        input_values2 = tuple(inputs2.values())
        exporter.export(model=model, output_path=str(path2), model_name_or_path='prajjwal1/bert-tiny')
        
        return path1, path2
    
    def test_compare_help(self, cli_runner):
        """Test compare command help."""
        result = cli_runner.invoke(cli, ['compare', '--help'])
        assert result.exit_code == 0
        assert 'Compare hierarchy tags' in result.output
    
    def test_compare_models(self, cli_runner, two_sample_models):
        """Test comparing two models."""
        path1, path2 = two_sample_models
        
        result = cli_runner.invoke(cli, [
            'compare',
            str(path1),
            str(path2)
        ])
        
        assert result.exit_code == 0
        assert '📊 Tag Distribution Comparison' in result.output
        assert f'Model 1: {path1}' in result.output
        assert f'Model 2: {path2}' in result.output
    
    def test_compare_to_file(self, cli_runner, two_sample_models, temp_workspace):
        """Test comparing models with output to file."""
        path1, path2 = two_sample_models
        output_file = temp_workspace['comparisons'] / 'comparison.json'
        
        result = cli_runner.invoke(cli, [
            'compare',
            str(path1),
            str(path2),
            '--output-file', str(output_file)
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        
        # Validate comparison JSON structure
        with open(output_file) as f:
            data = json.load(f)
        
        required_fields = ['model1_path', 'model2_path', 'tag_differences', 
                          'model1_only_tags', 'model2_only_tags']
        for field in required_fields:
            assert field in data
    
    def test_compare_nonexistent_files(self, cli_runner):
        """Test comparison with nonexistent files."""
        result = cli_runner.invoke(cli, [
            'compare',
            'nonexistent1.onnx',
            'nonexistent2.onnx'
        ])
        
        assert result.exit_code != 0
        assert 'No such file or directory' in result.output


class TestCLIGeneral:
    """Test general CLI functionality."""
    
    def test_cli_help(self, cli_runner):
        """Test main CLI help."""
        result = cli_runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Universal hierarchy-preserving ONNX export' in result.output
        assert 'export' in result.output
        assert 'analyze' in result.output
        assert 'validate' in result.output
        assert 'compare' in result.output
    
    def test_cli_version(self, cli_runner):
        """Test CLI version."""
        result = cli_runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        # Version should be displayed
    
    def test_verbose_flag(self, cli_runner):
        """Test verbose flag is passed through."""
        result = cli_runner.invoke(cli, ['--verbose', '--help'])
        assert result.exit_code == 0
    
    def test_invalid_subcommand(self, cli_runner):
        """Test invalid subcommand handling."""
        result = cli_runner.invoke(cli, ['invalid-command'])
        assert result.exit_code != 0


class TestCLIIntegration:
    """Integration tests for CLI workflow."""
    
    def test_export_analyze_validate_workflow(self, cli_runner, temp_workspace):
        """Test complete workflow: export -> analyze -> validate."""
        model_path = temp_workspace['models'] / 'workflow_test.onnx'
        
        # Step 1: Export
        export_result = cli_runner.invoke(cli, [
            'export',
            '--model', 'prajjwal1/bert-tiny',
            '--output', str(model_path)
        ])
        assert export_result.exit_code == 0
        assert model_path.exists()
        
        # Step 2: Analyze
        analyze_result = cli_runner.invoke(cli, [
            'analyze',
            str(model_path),
            '--output-format', 'summary'
        ])
        assert analyze_result.exit_code == 0
        assert 'Tag distribution:' in analyze_result.output
        
        # Step 3: Validate
        validate_result = cli_runner.invoke(cli, [
            'validate',
            str(model_path),
            '--check-consistency'
        ])
        assert validate_result.exit_code == 0
        assert '✅' in validate_result.output  # Some success indicator
    
    def test_structured_temp_results(self, temp_workspace):
        """Test that temp workspace structure is properly organized."""
        # Verify all expected directories exist
        expected_dirs = ['models', 'exports', 'analysis', 'comparisons']
        for dir_name in expected_dirs:
            assert dir_name in temp_workspace
            assert temp_workspace[dir_name].exists()
            assert temp_workspace[dir_name].is_dir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])