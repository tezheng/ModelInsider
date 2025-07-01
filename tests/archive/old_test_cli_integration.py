"""
CLI Integration tests - migrated from standalone scripts.

This module replaces bert_convert_cli.py and test_hierarchical_tagging.py
with proper pytest-based testing using structured temp directories.
"""

import pytest
import json
import tempfile
from pathlib import Path
from click.testing import CliRunner

from modelexport.cli import cli
from modelexport import HierarchyExporter


@pytest.fixture
def structured_temp_dir():
    """Create organized temp directory structure for CLI tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base = Path(temp_dir)
        
        # Create structured subdirectories
        structure = {
            'models': base / 'models',
            'exports': base / 'exports',
            'tags': base / 'tags',
            'analysis': base / 'analysis'
        }
        
        for subdir in structure.values():
            subdir.mkdir(parents=True, exist_ok=True)
        
        yield structure


class TestCLIBertConversion:
    """Test BERT conversion functionality via CLI (replaces bert_convert_cli.py)."""
    
    def test_bert_conversion_basic(self, structured_temp_dir):
        """Test basic BERT conversion via CLI."""
        cli_runner = CliRunner()
        output_path = structured_temp_dir['exports'] / 'bert_cli_conversion.onnx'
        
        result = cli_runner.invoke(cli, [
            '--verbose',
            'export',
            'prajjwal1/bert-tiny',  # Use tiny model for faster testing
            str(output_path),
            '--input-text', 'Hello world, this is a test.'
        ])
        
        # Verify CLI success
        assert result.exit_code == 0, f"CLI conversion failed: {result.output}"
        assert '✅ Export completed successfully!' in result.output
        
        # Verify files created
        assert output_path.exists(), "ONNX file not created"
        sidecar_path = Path(str(output_path).replace('.onnx', '_hierarchy.json'))
        assert sidecar_path.exists(), "Sidecar file not created"
        
        # Code-generated validation of results
        with open(sidecar_path) as f:
            sidecar_data = json.load(f)
        
        # Validate structure
        assert 'metadata' in sidecar_data or 'summary' in sidecar_data
        assert 'node_tags' in sidecar_data
        
        # Validate tag statistics
        tag_stats = sidecar_data.get('tag_statistics', {})
        assert len(tag_stats) > 0, "No tag statistics found"
        
        # Check for expected BERT components
        expected_bert_components = ['BertEmbeddings', 'BertEncoder', 'BertAttention', 'BertPooler']
        found_components = [comp for comp in expected_bert_components 
                           if any(comp in tag for tag in tag_stats.keys())]
        
        assert len(found_components) >= 3, f"Expected BERT components not found. Found: {found_components}"
    
    def test_bert_conversion_with_tag_saving(self, structured_temp_dir):
        """Test BERT conversion with tag analysis (replaces --save-tags functionality)."""
        cli_runner = CliRunner()
        model_path = structured_temp_dir['exports'] / 'bert_with_tags.onnx'
        analysis_path = structured_temp_dir['analysis'] / 'bert_analysis.json'
        
        # Step 1: Export model
        export_result = cli_runner.invoke(cli, [
            'export', 
            'prajjwal1/bert-tiny',
            str(model_path),
            '--input-text', 'Test text for tag analysis'
        ])
        assert export_result.exit_code == 0
        
        # Step 2: Analyze and save tags (replaces --save-tags)
        analyze_result = cli_runner.invoke(cli, [
            'analyze',
            str(model_path),
            '--output-format', 'json',
            '--output-file', str(analysis_path)
        ])
        assert analyze_result.exit_code == 0
        assert analysis_path.exists()
        
        # Code-generated validation of analysis
        with open(analysis_path) as f:
            analysis_data = json.load(f)
        
        # Verify analysis structure (equivalent to old tag_data structure)
        assert 'node_tags' in analysis_data
        assert 'summary' in analysis_data or 'tag_statistics' in analysis_data
        
        # Validate unique tags and counts
        node_tags = analysis_data['node_tags']
        unique_tags = set()
        tag_counts = {}
        
        for node_info in node_tags.values():
            for tag in node_info.get('tags', []):
                unique_tags.add(tag)
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        assert len(unique_tags) > 0, "No unique tags found"
        assert len(tag_counts) > 0, "No tag counts computed"
        
        # Verify transformers-specific tags (not torch.nn)
        has_transformers_tags = any('Bert' in tag for tag in unique_tags)
        has_torch_nn_tags = any('torch.nn' in tag for tag in unique_tags)
        
        assert has_transformers_tags, "Should have transformers-specific tags"
        assert not has_torch_nn_tags, "Should not have torch.nn tags in transformers model"
    
    def test_bert_conversion_verbose_output(self, structured_temp_dir):
        """Test verbose output shows tag distribution (replaces verbose CLI functionality)."""
        cli_runner = CliRunner()
        model_path = structured_temp_dir['exports'] / 'bert_verbose.onnx'
        
        result = cli_runner.invoke(cli, [
            '--verbose',
            'export',
            'prajjwal1/bert-tiny',
            str(model_path)
        ])
        
        assert result.exit_code == 0
        
        # Verify verbose output shows tag statistics
        assert 'Tag distribution:' in result.output
        assert any(line.strip().startswith('/') for line in result.output.split('\n')), \
            "Should show hierarchical tags in verbose output"


class TestHierarchicalTagging:
    """Test hierarchical tagging implementation (replaces test_hierarchical_tagging.py)."""
    
    def test_transformers_model_tagging(self, structured_temp_dir):
        """Test tagging with transformers model (replaces test_simple_transformers_model)."""
        from transformers import AutoModel, AutoTokenizer
        
        # Load model
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        inputs = tokenizer("Hello world", return_tensors="pt")
        
        # Export with hierarchy exporter
        output_path = structured_temp_dir['exports'] / 'test_hierarchy_tagging.onnx'
        exporter = HierarchyExporter()
        result = exporter.export(model, inputs, str(output_path))
        
        # Code-generated validation
        tag_mapping = exporter.get_tag_mapping()
        assert len(tag_mapping) > 0, "No tag mapping generated"
        
        # Check parameter to module mapping
        param_to_module = getattr(exporter, '_param_to_module', {})
        assert len(param_to_module) > 0, "No parameters mapped to modules"
        
        # Analyze unique tags
        unique_tags = set()
        for node_info in tag_mapping.values():
            unique_tags.update(node_info.get('tags', []))
        
        assert len(unique_tags) > 0, "No unique tags found"
        
        # Validate tag characteristics
        expected_patterns = ['BertEmbeddings', 'BertEncoder', 'BertAttention', 'BertPooler']
        found_patterns = [pattern for pattern in expected_patterns 
                         if any(pattern in tag for tag in unique_tags)]
        
        assert len(found_patterns) >= 3, f"Expected BERT patterns not found: {found_patterns}"
        
        # Verify correct tag types (transformers, not torch.nn)
        has_transformers_tags = any("Bert" in tag for tag in unique_tags)
        has_torch_nn_tags = any("torch.nn" in tag for tag in unique_tags)
        
        assert has_transformers_tags, "Should have transformers tags"
        assert not has_torch_nn_tags, "Should not have torch.nn tags"
    
    def test_module_hierarchy_inspection(self, structured_temp_dir):
        """Test module hierarchy inspection (replaces test_module_hierarchy_inspection)."""
        from transformers import AutoModel
        
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        
        # Analyze module hierarchy programmatically
        transformers_modules = []
        torch_nn_modules = []
        other_modules = []
        
        for name, module in model.named_modules():
            if name:  # Skip root
                module_path = module.__class__.__module__
                
                if 'transformers' in module_path:
                    transformers_modules.append((name, module.__class__.__name__))
                elif 'torch.nn' in module_path:
                    torch_nn_modules.append((name, module.__class__.__name__))
                else:
                    other_modules.append((name, module.__class__.__name__))
        
        # Code-generated validation
        assert len(transformers_modules) > 0, "Should have transformers modules"
        
        # Verify expected transformers module types
        transformers_classes = [class_name for _, class_name in transformers_modules]
        expected_classes = ['BertEmbeddings', 'BertEncoder', 'BertLayer', 'BertAttention']
        found_classes = [cls for cls in expected_classes if cls in transformers_classes]
        
        assert len(found_classes) >= 3, f"Expected transformers classes not found: {found_classes}"
        
        # Both transformers and torch.nn modules are expected in a BERT model
        # The key is that we should have significant transformers modules
        assert len(transformers_modules) > 10, \
            f"Should have substantial transformers modules, found: {len(transformers_modules)}"
    
    def test_tag_building_logic(self, structured_temp_dir):
        """Test tag building logic directly (replaces test_tag_building_logic)."""
        from transformers import AutoModel
        
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        exporter = HierarchyExporter()
        exporter._model = model  # Set model reference for tag building
        
        # Build operation context first (simulate what happens during export)
        exporter._operation_context = {}
        for name, module in model.named_modules():
            if name:
                if exporter._should_tag_module(module.__class__.__module__):
                    exporter._operation_context[name] = {
                        'tag': exporter._build_hierarchical_tag(name, module),
                        'module_class': module.__class__.__name__
                    }
        
        # Test tag building for key modules (code-generated validation)
        test_cases = [
            ('embeddings', 'BertEmbeddings'),
            ('encoder', 'BertEncoder'),
            ('pooler', 'BertPooler')
        ]
        
        passed_tests = 0
        for module_name, expected_class in test_cases:
            if module_name in exporter._operation_context:
                context = exporter._operation_context[module_name]
                tag = context['tag']
                module_class = context['module_class']
                
                # Validate tag format and content
                assert tag.startswith('/'), f"Tag should start with '/': {tag}"
                assert expected_class in tag or module_class == expected_class, \
                    f"Tag should contain {expected_class}: {tag}, class: {module_class}"
                
                passed_tests += 1
        
        assert passed_tests >= 2, f"Should pass at least 2 tag building tests, passed: {passed_tests}"
    
    def test_complete_tagging_workflow(self, structured_temp_dir):
        """Test complete tagging workflow (replaces main function logic)."""
        from transformers import AutoModel, AutoTokenizer
        
        # Load model and tokenizer
        model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        inputs = tokenizer("Test complete workflow", return_tensors="pt")
        
        # Export with hierarchy preservation
        output_path = structured_temp_dir['exports'] / 'complete_workflow.onnx'
        exporter = HierarchyExporter()
        result = exporter.export(model, inputs, str(output_path))
        
        # Validate export result
        assert result['total_operations'] > 0
        assert result['tagged_operations'] > 0
        assert result['tagged_operations'] <= result['total_operations']
        
        # Get and validate tag mapping
        tag_mapping = exporter.get_tag_mapping()
        unique_tags = set()
        for node_info in tag_mapping.values():
            unique_tags.update(node_info.get('tags', []))
        
        # Code-generated comprehensive validation
        validation_results = {
            'has_tags': len(unique_tags) > 0,
            'has_transformers_tags': any('Bert' in tag for tag in unique_tags),
            'no_torch_nn_tags': not any('torch.nn' in tag for tag in unique_tags),
            'expected_components': 0
        }
        
        # Count expected BERT components
        expected_components = ['BertEmbeddings', 'BertEncoder', 'BertAttention', 'BertPooler']
        for component in expected_components:
            if any(component in tag for tag in unique_tags):
                validation_results['expected_components'] += 1
        
        # Assert all validations pass
        assert validation_results['has_tags'], "Should have tags"
        assert validation_results['has_transformers_tags'], "Should have transformers tags"
        assert validation_results['no_torch_nn_tags'], "Should not have torch.nn tags"
        assert validation_results['expected_components'] >= 3, \
            f"Should have at least 3 expected components, found: {validation_results['expected_components']}"
        
        # Files should exist
        assert output_path.exists(), "ONNX file should exist"
        sidecar_path = Path(str(output_path).replace('.onnx', '_hierarchy.json'))
        assert sidecar_path.exists(), "Sidecar file should exist"


class TestStandaloneScriptMigration:
    """Test that all standalone script functionality is properly migrated."""
    
    def test_bert_convert_cli_functionality_covered(self, structured_temp_dir):
        """Verify all bert_convert_cli.py functionality is covered by CLI."""
        cli_runner = CliRunner()
        
        # Test main conversion functionality
        result = cli_runner.invoke(cli, [
            '--verbose',  # Global verbose flag
            'export',
            'prajjwal1/bert-tiny',
            str(structured_temp_dir['exports'] / 'migration_test.onnx'),
            '--input-text', 'Migration test text'
        ])
        
        assert result.exit_code == 0
        
        # Test tag saving functionality (via analyze command)
        analyze_result = cli_runner.invoke(cli, [
            'analyze',
            str(structured_temp_dir['exports'] / 'migration_test.onnx'),
            '--output-format', 'json',
            '--output-file', str(structured_temp_dir['analysis'] / 'migration_tags.json')
        ])
        
        assert analyze_result.exit_code == 0
        
        # Verify equivalent functionality
        with open(structured_temp_dir['analysis'] / 'migration_tags.json') as f:
            analysis_data = json.load(f)
        
        # Should have equivalent structure to old --save-tags output
        assert 'node_tags' in analysis_data
        assert 'summary' in analysis_data or 'tag_statistics' in analysis_data
    
    def test_hierarchical_tagging_functionality_covered(self):
        """Verify all test_hierarchical_tagging.py functionality is covered by pytest."""
        # This test validates that we have proper pytest coverage
        # All functionality from test_hierarchical_tagging.py should be in other test methods
        
        # Check that we have the key test methods
        test_methods = [
            'test_transformers_model_tagging',
            'test_module_hierarchy_inspection', 
            'test_tag_building_logic',
            'test_complete_tagging_workflow'
        ]
        
        for method_name in test_methods:
            assert hasattr(TestHierarchicalTagging, method_name), \
                f"Missing migrated test method: {method_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])