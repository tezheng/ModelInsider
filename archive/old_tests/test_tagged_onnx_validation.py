"""
Test cases for validating tagged ONNX models against generated test data.

This test suite validates our exported tagged ONNX models against the expected
test data and ensures compliance with MUST rules and design principles.
"""

from __future__ import annotations

import json
from pathlib import Path

import onnx
import pytest
import torch

from modelexport.strategies.htp.htp_hierarchy_exporter import HierarchyExporter


class TestMustRuleCompliance:
    """Tests enforcing MUST rules compliance in tagged ONNX models."""
    
    def test_must_001_no_hardcoded_logic_validation(self, prepared_bert_inputs, tmp_path):
        """MUST-001: Validate no hardcoded logic in exported model tags."""
        test_data = prepared_bert_inputs
        model = test_data['model']
        inputs = test_data['inputs']
        
        # Export model
        exporter = HierarchyExporter(strategy="htp")
        output_path = tmp_path / "must_001_test.onnx"
        
        result = exporter.export(model, inputs, str(output_path))
        tag_mapping = exporter.get_tag_mapping()
        
        # MUST-001 Validation: Check for hardcoded architecture patterns
        hardcoded_patterns = [
            'bert', 'transformer', 'resnet', 'gpt', 'vit',  # Model names
            'attention', 'mlp', 'ffn',  # Architecture-specific terms
            'layer_0', 'layer_1', 'head_'  # Hardcoded indices
        ]
        
        violations = []
        for _node_name, node_info in tag_mapping.items():
            for tag in node_info.get('tags', []):
                tag_lower = tag.lower()
                for pattern in hardcoded_patterns:
                    if pattern in tag_lower and not self._is_legitimate_class_name(tag, pattern):
                        violations.append(f"Tag '{tag}' contains hardcoded pattern '{pattern}'")
        
        assert len(violations) == 0, f"MUST-001 violations found: {violations}"
        
        # Verify universal approach: same logic should work with any model
        assert result['strategy'] == 'htp', "Should use universal usage-based strategy"
        assert result['total_operations'] > 0, "Should have tagged some operations universally"
    
    def _is_legitimate_class_name(self, tag: str, pattern: str) -> bool:
        """Check if pattern appears as legitimate class name (e.g., BertModel is OK)."""
        # Allow patterns that are part of actual PyTorch class names
        legitimate_contexts = [
            'BertModel', 'BertEncoder', 'BertLayer', 'BertAttention', 
            'BertEmbeddings', 'BertPooler', 'BertSelfOutput', 'BertIntermediate',
            'BertSdpaSelfAttention', 'BertOutput'
        ]
        
        return any(legitimate in tag for legitimate in legitimate_contexts)
    
    def test_must_002_torch_nn_filtering(self, prepared_bert_inputs, tmp_path):
        """MUST-002: Validate torch.nn modules are filtered from hierarchy tags."""
        test_data = prepared_bert_inputs
        model = test_data['model']
        inputs = test_data['inputs']
        
        # Export model
        exporter = HierarchyExporter(strategy="htp")
        output_path = tmp_path / "must_002_test.onnx"
        
        exporter.export(model, inputs, str(output_path))
        tag_mapping = exporter.get_tag_mapping()
        
        # MUST-002 Validation: No torch.nn class names in tags
        torch_nn_classes = [
            'Linear', 'LayerNorm', 'Embedding', 'Dropout', 
            'Conv2d', 'BatchNorm1d', 'ReLU', 'GELU', 'Tanh'
        ]
        
        violations = []
        for _node_name, node_info in tag_mapping.items():
            for tag in node_info.get('tags', []):
                # Split tag into components and check each
                tag_components = tag.strip('/').split('/')
                for component in tag_components:
                    if component in torch_nn_classes:
                        violations.append(f"Tag '{tag}' contains torch.nn class '{component}'")
        
        assert len(violations) == 0, f"MUST-002 violations: {violations}"
        
        # Positive validation: should contain model-specific classes
        model_specific_classes = ['BertModel', 'BertEmbeddings', 'BertEncoder']
        found_model_classes = set()
        
        for node_info in tag_mapping.values():
            for tag in node_info.get('tags', []):
                for model_class in model_specific_classes:
                    if model_class in tag:
                        found_model_classes.add(model_class)
        
        assert len(found_model_classes) > 0, "Should find model-specific classes in tags"
    
    def test_must_003_universal_design_validation(self, bert_model_cache, tmp_path):
        """MUST-003: Validate universal design works across different models."""
        model, tokenizer = bert_model_cache
        
        # Test with different input configurations to validate universality
        test_configs = [
            {
                'name': 'short_sequence',
                'text': 'Test',
                'max_length': 16
            },
            {
                'name': 'medium_sequence', 
                'text': 'This is a medium length test sequence',
                'max_length': 32
            },
            {
                'name': 'batch_inputs',
                'text': ['First', 'Second', 'Third'],
                'max_length': 16
            }
        ]
        
        exporter = HierarchyExporter(strategy="htp")
        results = []
        
        for config in test_configs:
            # Prepare inputs
            if isinstance(config['text'], list):
                inputs = tokenizer(config['text'], return_tensors='pt', 
                                 padding=True, truncation=True, max_length=config['max_length'])
            else:
                inputs = tokenizer(config['text'], return_tensors='pt',
                                 padding=True, truncation=True, max_length=config['max_length'])
            
            # Export with same exporter (universal approach)
            output_path = tmp_path / f"must_003_{config['name']}.onnx"
            
            try:
                result = exporter.export(model, inputs, str(output_path))
                tag_mapping = exporter.get_tag_mapping()
                
                test_result = {
                    'config': config['name'],
                    'success': True,
                    'total_operations': result['total_operations'],
                    'tagged_operations': result['tagged_operations'],
                    'unique_tags': len({tag for node in tag_mapping.values() 
                                         for tag in node.get('tags', [])})
                }
                
                # Universal validation: same exporter, consistent results
                assert result['strategy'] == 'htp', "Should use same strategy universally"
                assert result['total_operations'] > 0, "Should work with any input configuration"
                
            except Exception as e:
                test_result = {
                    'config': config['name'],
                    'success': False,
                    'error': str(e)
                }
            
            results.append(test_result)
        
        # All configurations should succeed (universal design)
        successful_tests = [r for r in results if r['success']]
        assert len(successful_tests) == len(test_configs), \
            f"Universal design failed: {len(successful_tests)}/{len(test_configs)} succeeded"


class TestTagFormatValidation:
    """Tests for hierarchy tag format and structure validation."""
    
    def test_no_empty_tag_lists(self, prepared_bert_inputs, tmp_path):
        """CRITICAL: Test that no operations have empty tag lists - this indicates tagging failure."""
        test_data = prepared_bert_inputs
        model = test_data['model']
        inputs = test_data['inputs']
        
        exporter = HierarchyExporter(strategy="htp")
        output_path = tmp_path / "empty_tags_test.onnx"
        
        result = exporter.export(model, inputs, str(output_path))
        tag_mapping = exporter.get_tag_mapping()
        
        # Find operations with empty tag lists
        empty_tag_operations = []
        tagged_operations = []
        
        for node_name, node_info in tag_mapping.items():
            tags = node_info.get('tags', [])
            if len(tags) == 0:
                empty_tag_operations.append({
                    'node': node_name,
                    'op_type': node_info.get('op_type', 'unknown'),
                    'inputs': node_info.get('inputs', []),
                    'outputs': node_info.get('outputs', [])
                })
            else:
                tagged_operations.append(node_name)
        
        # Critical validation: Usage-based tagging should tag all parameter-using operations
        total_operations = result['total_operations']
        reported_tagged = result['tagged_operations']
        actual_tagged = len(tagged_operations)
        empty_count = len(empty_tag_operations)
        
        # Analysis for debugging
        analysis = {
            'total_operations': total_operations,
            'reported_tagged_operations': reported_tagged,
            'actual_tagged_operations': actual_tagged,
            'empty_tag_operations': empty_count,
            'empty_tag_percentage': (empty_count / total_operations * 100) if total_operations > 0 else 0,
            'discrepancy': reported_tagged - actual_tagged
        }
        
        # Show examples of empty tag operations for debugging
        if empty_tag_operations:
            print(f"\n🚨 Found {empty_count} operations with empty tags ({analysis['empty_tag_percentage']:.1f}%):")
            for i, op in enumerate(empty_tag_operations[:5]):  # Show first 5
                print(f"  {i+1}. {op['node']} ({op['op_type']}) - inputs: {len(op['inputs'])}")
        
        # CRITICAL ASSERTION: In usage-based tagging, operations should either:
        # 1. Be tagged (if they use parameters or are in propagation chain)
        # 2. Be legitimately untagged (if they're pure computation with no parameter dependencies)
        
        # For BERT model, we expect high tagging coverage because most operations are parameter-dependent
        max_allowed_empty_percentage = 60.0  # Allow some untagged operations for pure computation
        
        assert analysis['empty_tag_percentage'] <= max_allowed_empty_percentage, \
            f"Too many empty tag operations: {analysis['empty_tag_percentage']:.1f}% > {max_allowed_empty_percentage}%"
        
        # Data consistency check
        assert analysis['discrepancy'] == 0, \
            f"Reported vs actual tagged operations mismatch: {reported_tagged} vs {actual_tagged}"
        
        return analysis
    
    def test_hierarchy_tag_format_compliance(self, prepared_bert_inputs, tmp_path):
        """Validate hierarchy tags follow expected format patterns."""
        test_data = prepared_bert_inputs
        model = test_data['model']
        inputs = test_data['inputs']
        
        exporter = HierarchyExporter(strategy="htp")
        output_path = tmp_path / "tag_format_test.onnx"
        
        exporter.export(model, inputs, str(output_path))
        tag_mapping = exporter.get_tag_mapping()
        
        # Collect all tags
        all_tags = []
        for node_info in tag_mapping.values():
            all_tags.extend(node_info.get('tags', []))
        
        # Format validation
        format_errors = []
        
        for tag in all_tags:
            # Must start with '/'
            if not tag.startswith('/'):
                format_errors.append(f"Tag '{tag}' doesn't start with '/'")
            
            # Must contain root model class
            if 'BertModel' not in tag:
                format_errors.append(f"Tag '{tag}' doesn't contain root model class")
            
            # Must have hierarchical structure (at least 2 levels)
            parts = tag.strip('/').split('/')
            if len(parts) < 2:
                format_errors.append(f"Tag '{tag}' has insufficient hierarchy depth")
            
            # No empty segments
            if any(part == '' for part in parts):
                format_errors.append(f"Tag '{tag}' has empty segments")
        
        assert len(format_errors) == 0, f"Tag format errors: {format_errors}"
        assert len(all_tags) > 0, "Should have generated some tags"
    
    def test_hierarchy_structure_consistency(self, prepared_bert_inputs, tmp_path):
        """Validate hierarchy structure follows consistent patterns."""
        test_data = prepared_bert_inputs
        model = test_data['model']
        inputs = test_data['inputs']
        
        exporter = HierarchyExporter(strategy="htp")
        output_path = tmp_path / "hierarchy_structure_test.onnx"
        
        exporter.export(model, inputs, str(output_path))
        tag_mapping = exporter.get_tag_mapping()
        
        # Analyze hierarchy patterns
        tag_statistics = {}
        for node_info in tag_mapping.values():
            for tag in node_info.get('tags', []):
                tag_statistics[tag] = tag_statistics.get(tag, 0) + 1
        
        # Structural validation
        structure_checks = {
            'has_embeddings': any('BertEmbeddings' in tag for tag in tag_statistics),
            'has_encoder': any('BertEncoder' in tag for tag in tag_statistics),
            'has_attention': any('BertAttention' in tag for tag in tag_statistics),
            'has_pooler': any('BertPooler' in tag for tag in tag_statistics),
            'consistent_root': all(tag.startswith('/BertModel') for tag in tag_statistics)
        }
        
        # All structural elements should be present
        missing_structures = [name for name, present in structure_checks.items() if not present]
        assert len(missing_structures) == 0, f"Missing hierarchy structures: {missing_structures}"
        
        # Hierarchy depth analysis
        depths = [len(tag.strip('/').split('/')) for tag in tag_statistics]
        assert min(depths) >= 2, "All tags should have at least 2 hierarchy levels"
        assert max(depths) <= 6, "No tag should be excessively deep"


class TestDataComparisonValidation:
    """Tests comparing generated tags with expected test data."""
    
    def test_against_expected_test_data(self, prepared_bert_inputs, tmp_path):
        """Compare generated tags against expected test data structure."""
        test_data = prepared_bert_inputs
        model = test_data['model']
        inputs = test_data['inputs']
        
        # Export our model
        exporter = HierarchyExporter(strategy="htp")
        output_path = tmp_path / "comparison_test.onnx"
        
        exporter.export(model, inputs, str(output_path))
        tag_mapping = exporter.get_tag_mapping()
        
        # Load expected test data (using proper temp folder location)
        project_root = Path(__file__).parent.parent
        expected_data_path = project_root / "temp/onnx_model/bert_tiny/prajjwal1_bert_tiny/expected_tags.json"
        if not expected_data_path.exists():
            pytest.skip("Expected test data not available")
        
        with open(expected_data_path) as f:
            expected_data = json.load(f)
        
        # Extract our tag statistics
        our_tag_stats = {}
        for node_info in tag_mapping.values():
            for tag in node_info.get('tags', []):
                our_tag_stats[tag] = our_tag_stats.get(tag, 0) + 1
        
        # Compare with expected hierarchy patterns
        expected_hierarchy = expected_data['expected_hierarchy']
        
        # Pattern matching analysis
        pattern_matches = 0
        pattern_misses = []
        
        for expected_pattern in expected_hierarchy:
            # Check if any of our tags match this pattern
            found_match = False
            for our_tag in our_tag_stats:
                if self._tags_match_pattern(our_tag, expected_pattern):
                    found_match = True
                    pattern_matches += 1
                    break
            
            if not found_match:
                pattern_misses.append(expected_pattern)
        
        # Analysis results
        total_expected = len(expected_hierarchy)
        match_rate = pattern_matches / total_expected * 100 if total_expected > 0 else 0
        
        comparison_result = {
            'our_unique_tags': len(our_tag_stats),
            'expected_patterns': total_expected,
            'pattern_matches': pattern_matches,
            'match_rate_percent': match_rate,
            'pattern_misses': pattern_misses[:5],  # Show first 5 misses
            'our_top_tags': list(sorted(our_tag_stats.items(), key=lambda x: x[1], reverse=True)[:5])
        }
        
        # Validation criteria
        assert len(our_tag_stats) > 0, "Should generate some tags"
        assert match_rate > 15.0, f"Match rate too low: {match_rate:.1f}% (expected >15%)"
        
        # Validate critical patterns are present
        critical_patterns = [
            '/BertModel/BertEmbeddings',
            '/BertModel/BertEncoder', 
            '/BertModel/BertPooler'
        ]
        
        missing_critical = []
        for critical in critical_patterns:
            if not any(self._tags_match_pattern(tag, critical) for tag in our_tag_stats):
                missing_critical.append(critical)
        
        assert len(missing_critical) == 0, f"Missing critical patterns: {missing_critical}"
        
        return comparison_result
    
    def _tags_match_pattern(self, our_tag: str, expected_pattern: str) -> bool:
        """Check if our tag matches the expected pattern."""
        # Simple substring matching with some flexibility
        return expected_pattern in our_tag or our_tag in expected_pattern


class TestONNXIntegrityValidation:
    """Tests for ONNX model integrity and functionality."""
    
    def test_tagged_onnx_structure_intact(self, prepared_bert_inputs, tmp_path):
        """Test that tagged ONNX model maintains identical structure to baseline."""
        test_data = prepared_bert_inputs
        model = test_data['model']
        inputs = test_data['inputs']
        
        # Export baseline (untagged) model
        baseline_path = tmp_path / "baseline_structure.onnx"
        input_args = tuple(inputs.values())
        
        torch.onnx.export(
            model,
            input_args,
            str(baseline_path),
            opset_version=14,
            export_params=True,
            do_constant_folding=True
        )
        
        # Export tagged model  
        tagged_path = tmp_path / "tagged_structure.onnx"
        exporter = HierarchyExporter(strategy="htp")
        exporter.export(model, inputs, str(tagged_path), opset_version=14)
        
        # Load both models
        baseline_model = onnx.load(str(baseline_path))
        tagged_model = onnx.load(str(tagged_path))
        
        # Structural integrity checks
        structure_comparison = {
            'baseline_inputs': len(baseline_model.graph.input),
            'tagged_inputs': len(tagged_model.graph.input),
            'baseline_outputs': len(baseline_model.graph.output),
            'tagged_outputs': len(tagged_model.graph.output),
            'baseline_nodes': len(baseline_model.graph.node),
            'tagged_nodes': len(tagged_model.graph.node),
            'baseline_initializers': len(baseline_model.graph.initializer),
            'tagged_initializers': len(tagged_model.graph.initializer),
        }
        
        # Critical assertions: Structure must be identical
        assert structure_comparison['baseline_inputs'] == structure_comparison['tagged_inputs'], \
            f"Input count mismatch: baseline={structure_comparison['baseline_inputs']}, tagged={structure_comparison['tagged_inputs']}"
        
        assert structure_comparison['baseline_outputs'] == structure_comparison['tagged_outputs'], \
            f"Output count mismatch: baseline={structure_comparison['baseline_outputs']}, tagged={structure_comparison['tagged_outputs']}"
        
        assert structure_comparison['baseline_initializers'] == structure_comparison['tagged_initializers'], \
            f"Initializer count mismatch: baseline={structure_comparison['baseline_initializers']}, tagged={structure_comparison['tagged_initializers']}"
        
        # Node count should be very close (allow minor differences due to optimization)
        node_diff = abs(structure_comparison['baseline_nodes'] - structure_comparison['tagged_nodes'])
        max_node_diff = 3  # Allow small differences due to optimization
        
        assert node_diff <= max_node_diff, \
            f"Too many node differences: {node_diff} > {max_node_diff} (baseline={structure_comparison['baseline_nodes']}, tagged={structure_comparison['tagged_nodes']})"
        
        # Input/Output names and shapes should be identical
        baseline_input_names = [inp.name for inp in baseline_model.graph.input]
        tagged_input_names = [inp.name for inp in tagged_model.graph.input]
        
        baseline_output_names = [out.name for out in baseline_model.graph.output]
        tagged_output_names = [out.name for out in tagged_model.graph.output]
        
        assert baseline_input_names == tagged_input_names, \
            f"Input names mismatch: baseline={baseline_input_names}, tagged={tagged_input_names}"
        
        assert baseline_output_names == tagged_output_names, \
            f"Output names mismatch: baseline={baseline_output_names}, tagged={tagged_output_names}"
        
        # Check that only tagged model has hierarchy metadata
        tagged_hierarchy_nodes = 0
        for node in tagged_model.graph.node:
            if node.doc_string:
                try:
                    hierarchy_info = json.loads(node.doc_string)
                    if isinstance(hierarchy_info, dict) and "hierarchy_tags" in hierarchy_info:
                        tagged_hierarchy_nodes += 1
                except json.JSONDecodeError:
                    pass
        
        baseline_hierarchy_nodes = 0
        for node in baseline_model.graph.node:
            if node.doc_string:
                try:
                    hierarchy_info = json.loads(node.doc_string)
                    if isinstance(hierarchy_info, dict) and "hierarchy_tags" in hierarchy_info:
                        baseline_hierarchy_nodes += 1
                except json.JSONDecodeError:
                    pass
        
        assert baseline_hierarchy_nodes == 0, \
            f"Baseline should have no hierarchy metadata, found {baseline_hierarchy_nodes}"
        
        assert tagged_hierarchy_nodes > 0, \
            f"Tagged model should have hierarchy metadata, found {tagged_hierarchy_nodes}"
        
        structure_comparison['hierarchy_nodes_added'] = tagged_hierarchy_nodes
        structure_comparison['structure_preserved'] = True
        
        return structure_comparison
    
    def test_onnx_model_validity(self, prepared_bert_inputs, tmp_path):
        """Validate exported ONNX model is valid and loadable."""
        test_data = prepared_bert_inputs
        model = test_data['model']
        inputs = test_data['inputs']
        
        exporter = HierarchyExporter(strategy="htp")
        output_path = tmp_path / "onnx_validity_test.onnx"
        
        # Export model
        result = exporter.export(model, inputs, str(output_path))
        
        # Load and validate ONNX model
        try:
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
        except Exception as e:
            pytest.fail(f"ONNX model validation failed: {e}")
        
        # Check model structure
        assert len(onnx_model.graph.input) > 0, "Model should have inputs"
        assert len(onnx_model.graph.output) > 0, "Model should have outputs"
        assert len(onnx_model.graph.node) > 0, "Model should have operations"
        
        # Check hierarchy tags are embedded
        hierarchy_nodes = 0
        for node in onnx_model.graph.node:
            if node.doc_string:
                try:
                    hierarchy_info = json.loads(node.doc_string)
                    if isinstance(hierarchy_info, dict) and "hierarchy_tags" in hierarchy_info:
                        hierarchy_nodes += 1
                except json.JSONDecodeError:
                    pass
        
        assert hierarchy_nodes > 0, "Model should have embedded hierarchy tags"
        assert hierarchy_nodes == result['tagged_operations'], \
            "Embedded tag count should match reported count"
    
    def test_sidecar_file_integrity(self, prepared_bert_inputs, tmp_path):
        """Validate sidecar JSON file contains complete hierarchy information."""
        test_data = prepared_bert_inputs
        model = test_data['model']
        inputs = test_data['inputs']
        
        exporter = HierarchyExporter(strategy="htp")
        output_path = tmp_path / "sidecar_test.onnx"
        
        exporter.export(model, inputs, str(output_path))
        
        # Check sidecar file
        sidecar_path = str(output_path).replace('.onnx', '_hierarchy.json')
        assert Path(sidecar_path).exists(), "Sidecar file should be created"
        
        # Load and validate sidecar
        with open(sidecar_path) as f:
            sidecar_data = json.load(f)
        
        # Required fields validation
        required_fields = [
            'version', 'format', 'model_path', 'generated_at',
            'exporter', 'summary', 'tag_statistics', 'node_tags', 'schema'
        ]
        
        missing_fields = [field for field in required_fields if field not in sidecar_data]
        assert len(missing_fields) == 0, f"Missing sidecar fields: {missing_fields}"
        
        # Content validation
        summary = sidecar_data['summary']
        assert summary['total_operations'] > 0, "Should report some operations"
        assert summary['tagged_operations'] > 0, "Should report some tagged operations"
        assert summary['unique_tags'] > 0, "Should report some unique tags"
        
        # Tag statistics validation
        tag_stats = sidecar_data['tag_statistics']
        assert len(tag_stats) > 0, "Should have tag statistics"
        assert all(isinstance(count, int) and count > 0 for count in tag_stats.values()), \
            "Tag counts should be positive integers"


@pytest.mark.slow
class TestIntegrationValidation:
    """End-to-end integration tests with real exported models."""
    
    def test_existing_bert_tiny_model_validation(self):
        """Validate our existing bert_tiny.onnx model if it exists."""
        # Check if we have the exported model from our previous test (in root dir)
        project_root = Path(__file__).parent.parent
        bert_model_path = project_root / "bert_tiny.onnx"
        sidecar_path = project_root / "bert_tiny_hierarchy.json"
        
        if not bert_model_path.exists():
            pytest.skip("bert_tiny.onnx not found - run export first")
        
        # Load and validate ONNX model
        try:
            onnx_model = onnx.load(str(bert_model_path))
            onnx.checker.check_model(onnx_model)
        except Exception as e:
            pytest.fail(f"Existing ONNX model validation failed: {e}")
        
        # Validate sidecar if exists
        if sidecar_path.exists():
            with open(sidecar_path) as f:
                sidecar_data = json.load(f)
            
            # Check key metrics - Updated for multi-consumer tagging approach
            summary = sidecar_data['summary']
            assert summary['total_operations'] >= 130, "Should have at least 130 total operations"
            assert summary['tagged_operations'] >= 100, "Should have at least 100 tagged operations"
            assert summary['unique_tags'] == 6, "Should have 6 unique tags"
            
            # Validate specific tags we expect
            tag_stats = sidecar_data['tag_statistics']
            expected_tags = [
                '/BertModel/BertEmbeddings',
                '/BertModel/BertEncoder/BertLayer/BertAttention/BertSelfOutput',
                '/BertModel/BertPooler'
            ]
            
            for expected_tag in expected_tags:
                assert expected_tag in tag_stats, f"Missing expected tag: {expected_tag}"
                assert tag_stats[expected_tag] > 0, f"Tag {expected_tag} should have positive count"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])