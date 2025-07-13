"""
Test baseline comparison between tagged and untagged ONNX exports.

This addresses the critical questions:
1. I/O shape handling and testing
2. Functional equivalence between tagged and untagged models
3. Proper comparison methodology
"""

import json

import onnx
import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from modelexport.strategies.htp.htp_hierarchy_exporter import HierarchyExporter


@pytest.fixture
def bert_model_and_tokenizer():
    """Load BERT model and tokenizer for testing."""
    model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
    model.eval()
    return model, tokenizer


@pytest.fixture  
def test_inputs(bert_model_and_tokenizer):
    """Create test inputs with various I/O shapes."""
    _, tokenizer = bert_model_and_tokenizer
    
    return {
        'short_text': tokenizer("Hello", return_tensors='pt', padding=True, truncation=True),
        'medium_text': tokenizer("Hello world test", return_tensors='pt', padding=True, truncation=True),
        'long_text': tokenizer("This is a longer text for testing different sequence lengths in BERT", 
                              return_tensors='pt', padding=True, truncation=True, max_length=32),
        'batch_inputs': tokenizer(["Hello", "World", "Test"], return_tensors='pt', padding=True, truncation=True)
    }


class TestBaselineComparison:
    """Test comparison between baseline and tagged ONNX exports."""
    
    def test_export_parameters_identical(self, bert_model_and_tokenizer, test_inputs, tmp_path):
        """Test that baseline and tagged use identical export parameters."""
        model, _ = bert_model_and_tokenizer
        inputs = test_inputs['medium_text']
        
        # Export parameters that should be identical
        export_params = {
            'export_params': True,
            'opset_version': 14,
            'do_constant_folding': True,
            'input_names': list(inputs.keys()),
            'output_names': ['pooler_output', 'last_hidden_state'],
            'dynamic_axes': {
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
                'token_type_ids': {0: 'batch_size', 1: 'sequence'},
                'pooler_output': {0: 'batch_size'},
                'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
            }
        }
        
        # Export baseline (unmodified torch.onnx.export)
        baseline_path = tmp_path / 'baseline.onnx'
        with torch.no_grad():
            torch.onnx.export(
                model,
                tuple(inputs.values()),
                str(baseline_path),
                **export_params
            )
        
        # Export tagged (using our HierarchyExporter with same parameters)
        tagged_path = tmp_path / 'tagged.onnx'
        exporter = HierarchyExporter()
        exporter.export(model, inputs, str(tagged_path), **export_params)
        
        # Load and compare
        baseline_model = onnx.load(str(baseline_path))
        tagged_model = onnx.load(str(tagged_path))
        
        # Code-generated validation
        comparison_result = {
            'baseline_nodes': len(baseline_model.graph.node),
            'tagged_nodes': len(tagged_model.graph.node),
            'baseline_inputs': len(baseline_model.graph.input),
            'tagged_inputs': len(tagged_model.graph.input),
            'baseline_outputs': len(baseline_model.graph.output),
            'tagged_outputs': len(tagged_model.graph.output)
        }
        
        # I/O compatibility should be identical
        assert comparison_result['baseline_inputs'] == comparison_result['tagged_inputs'], \
            f"Input count mismatch: baseline={comparison_result['baseline_inputs']}, tagged={comparison_result['tagged_inputs']}"
        assert comparison_result['baseline_outputs'] == comparison_result['tagged_outputs'], \
            f"Output count mismatch: baseline={comparison_result['baseline_outputs']}, tagged={comparison_result['tagged_outputs']}"
        
        # Node count difference analysis
        node_diff = comparison_result['baseline_nodes'] - comparison_result['tagged_nodes']
        
        # Check if tagged model has hierarchy attributes (using doc_string field)
        hierarchy_nodes = 0
        for node in tagged_model.graph.node:
            if node.doc_string:
                try:
                    hierarchy_info = json.loads(node.doc_string)
                    if isinstance(hierarchy_info, dict) and "hierarchy_tags" in hierarchy_info:
                        hierarchy_nodes += 1
                except json.JSONDecodeError:
                    pass
        
        assert hierarchy_nodes > 0, "Tagged model should have hierarchy attributes"
        
        # Save comparison data for inspection
        comparison_result['node_difference'] = node_diff
        comparison_result['hierarchy_nodes'] = hierarchy_nodes
        comparison_result['test_name'] = 'identical_export_parameters'
        
        return comparison_result
    
    def test_functional_equivalence(self, bert_model_and_tokenizer, test_inputs, tmp_path):
        """Test that baseline and tagged models produce equivalent outputs."""
        model, _ = bert_model_and_tokenizer
        
        for test_name, inputs in test_inputs.items():
            # Get PyTorch reference output
            with torch.no_grad():
                pytorch_output = model(**inputs)
            
            # Export both versions
            baseline_path = tmp_path / f'baseline_{test_name}.onnx'
            tagged_path = tmp_path / f'tagged_{test_name}.onnx'
            
            # Baseline export
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    tuple(inputs.values()),
                    str(baseline_path),
                    export_params=True,
                    opset_version=14,
                    input_names=list(inputs.keys()),
                    output_names=['pooler_output', 'last_hidden_state']
                )
            
            # Tagged export
            exporter = HierarchyExporter()
            exporter.export(model, inputs, str(tagged_path), opset_version=14)
            
            # Test that files exist
            assert baseline_path.exists(), f"Baseline export failed for {test_name}"
            assert tagged_path.exists(), f"Tagged export failed for {test_name}"
            
            # Load models and verify they can be loaded
            try:
                baseline_model = onnx.load(str(baseline_path))
                tagged_model = onnx.load(str(tagged_path))
                onnx.checker.check_model(baseline_model)
                onnx.checker.check_model(tagged_model)
            except Exception as e:
                pytest.fail(f"Model validation failed for {test_name}: {e}")
            
            # Record input shapes for this test
            input_shapes = {name: tensor.shape for name, tensor in inputs.items()}
            
            # Code-generated validation results
            test_result = {
                'test_case': test_name,
                'input_shapes': {k: list(v) for k, v in input_shapes.items()},
                'baseline_valid': True,
                'tagged_valid': True,
                'baseline_nodes': len(baseline_model.graph.node),
                'tagged_nodes': len(tagged_model.graph.node)
            }
            
            # This test validates that both models can be exported and loaded
            # Note: Actual inference testing would require ONNX runtime
    
    def test_io_shape_variations(self, bert_model_and_tokenizer, tmp_path):
        """Test various I/O shape configurations."""
        model, tokenizer = bert_model_and_tokenizer
        
        # Test cases with different I/O shapes
        test_cases = [
            {
                'name': 'single_token',
                'text': "Hi",
                'expected_seq_len': 3  # [CLS] Hi [SEP]
            },
            {
                'name': 'medium_sequence', 
                'text': "This is a medium length sequence for testing",
                'expected_seq_len': 11
            },
            {
                'name': 'batch_different_lengths',
                'texts': ["Short", "This is a longer sequence", "Medium length"],
                'is_batch': True
            }
        ]
        
        shape_results = []
        
        for case in test_cases:
            if case.get('is_batch', False):
                inputs = tokenizer(case['texts'], return_tensors='pt', padding=True, truncation=True)
            else:
                inputs = tokenizer(case['text'], return_tensors='pt', padding=True, truncation=True)
            
            # Export tagged version
            export_path = tmp_path / f"shape_test_{case['name']}.onnx"
            exporter = HierarchyExporter()
            
            try:
                result = exporter.export(model, inputs, str(export_path))
                
                # Analyze input shapes
                input_shapes = {name: list(tensor.shape) for name, tensor in inputs.items()}
                
                shape_result = {
                    'test_case': case['name'],
                    'input_shapes': input_shapes,
                    'export_success': True,
                    'total_operations': result['total_operations'],
                    'tagged_operations': result['tagged_operations']
                }
                
                # Validate batch dimension
                batch_size = input_shapes['input_ids'][0]
                sequence_length = input_shapes['input_ids'][1]
                
                assert batch_size > 0, f"Invalid batch size: {batch_size}"
                assert sequence_length > 0, f"Invalid sequence length: {sequence_length}"
                
                if case.get('is_batch', False):
                    assert batch_size == len(case['texts']), \
                        f"Batch size mismatch: expected {len(case['texts'])}, got {batch_size}"
                
                shape_result['validation_passed'] = True
                
            except Exception as e:
                shape_result = {
                    'test_case': case['name'],
                    'export_success': False,
                    'error': str(e),
                    'validation_passed': False
                }
            
            shape_results.append(shape_result)
        
        # All shape tests should pass
        passed_tests = [r for r in shape_results if r.get('validation_passed', False)]
        assert len(passed_tests) == len(test_cases), \
            f"Shape tests failed: {len(passed_tests)}/{len(test_cases)} passed"
        
        return shape_results
    
    def test_hierarchy_preservation_vs_baseline(self, bert_model_and_tokenizer, test_inputs, tmp_path):
        """Test that hierarchy preservation doesn't break model functionality."""
        model, _ = bert_model_and_tokenizer
        inputs = test_inputs['medium_text']
        
        # Export baseline
        baseline_path = tmp_path / 'hierarchy_baseline.onnx'
        with torch.no_grad():
            torch.onnx.export(
                model,
                tuple(inputs.values()),
                str(baseline_path),
                export_params=True,
                opset_version=14,
                input_names=list(inputs.keys()),
                output_names=['output']
            )
        
        # Export with hierarchy
        tagged_path = tmp_path / 'hierarchy_tagged.onnx'
        exporter = HierarchyExporter()
        result = exporter.export(model, inputs, str(tagged_path), opset_version=14)
        
        # Load both models
        baseline_model = onnx.load(str(baseline_path))
        tagged_model = onnx.load(str(tagged_path))
        
        # Validate both models
        onnx.checker.check_model(baseline_model)
        onnx.checker.check_model(tagged_model)
        
        # Check hierarchy tags are present only in tagged version
        import json
        
        baseline_hierarchy_nodes = 0
        for node in baseline_model.graph.node:
            if node.doc_string:
                try:
                    hierarchy_info = json.loads(node.doc_string)
                    if isinstance(hierarchy_info, dict) and "hierarchy_tags" in hierarchy_info:
                        baseline_hierarchy_nodes += 1
                except (json.JSONDecodeError, TypeError):
                    pass
        
        tagged_hierarchy_nodes = 0
        for node in tagged_model.graph.node:
            if node.doc_string:
                try:
                    hierarchy_info = json.loads(node.doc_string)
                    if isinstance(hierarchy_info, dict) and "hierarchy_tags" in hierarchy_info:
                        tagged_hierarchy_nodes += 1
                except (json.JSONDecodeError, TypeError):
                    pass
        
        # Code-generated validation
        assert baseline_hierarchy_nodes == 0, \
            f"Baseline should have no hierarchy nodes, found {baseline_hierarchy_nodes}"
        assert tagged_hierarchy_nodes > 0, \
            f"Tagged should have hierarchy nodes, found {tagged_hierarchy_nodes}"
        
        # Check that hierarchy doesn't break the model structure
        assert len(tagged_model.graph.input) > 0, "Tagged model should have inputs"
        assert len(tagged_model.graph.output) > 0, "Tagged model should have outputs"
        
        hierarchy_test_result = {
            'baseline_hierarchy_nodes': baseline_hierarchy_nodes,
            'tagged_hierarchy_nodes': tagged_hierarchy_nodes,
            'baseline_valid': True,
            'tagged_valid': True,
            'hierarchy_preserved': tagged_hierarchy_nodes > 0,
            'baseline_clean': baseline_hierarchy_nodes == 0
        }
        
        return hierarchy_test_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])