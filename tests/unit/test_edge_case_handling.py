"""
Edge Case Handling Unit Tests

Tests for enhanced edge case handling and multi-level fallback strategies
implemented in iteration 7.
"""

import pytest
import torch
import torch.nn as nn
import onnx
from pathlib import Path
import tempfile
import json

from modelexport.strategies.htp.htp_hierarchy_exporter import HierarchyExporter


class EdgeCaseTestModels:
    """Collection of edge case test models."""
    
    @staticmethod
    def create_identity_model():
        """Model with only identity operation."""
        return nn.Identity()
    
    @staticmethod
    def create_constant_heavy_model():
        """Model with many constant operations."""
        class ConstantHeavyModel(nn.Module):
            def forward(self, x):
                c1 = torch.tensor(1.0, dtype=x.dtype)
                c2 = torch.tensor(2.0, dtype=x.dtype)
                c3 = torch.tensor(0.5, dtype=x.dtype)
                return x * c1 + c2 - c3
        return ConstantHeavyModel()
    
    @staticmethod
    def create_reshape_heavy_model():
        """Model with many reshape operations."""
        class ReshapeHeavyModel(nn.Module):
            def forward(self, x):
                batch_size = x.shape[0]
                x = x.reshape(batch_size, -1)
                x = x.reshape(batch_size, 2, -1)
                x = x.transpose(1, 2)
                return x.reshape(batch_size, -1)
        return ReshapeHeavyModel()
    
    @staticmethod
    def create_type_conversion_model():
        """Model with type conversions."""
        class TypeConversionModel(nn.Module):
            def forward(self, x):
                x = x.float()
                x = x.double()
                x = x.int()
                return x.float()
        return TypeConversionModel()
    
    @staticmethod
    def create_minimal_activation_model():
        """Single activation model."""
        class SingleActivationModel(nn.Module):
            def forward(self, x):
                return torch.relu(x)
        return SingleActivationModel()
    
    @staticmethod
    def create_dynamic_shape_model():
        """Model with dynamic shape operations."""
        class DynamicShapeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)
            
            def forward(self, x):
                batch_size = x.shape[0]
                scale = torch.tensor(1.0 / batch_size, dtype=x.dtype)
                return self.linear(x * scale)
        return DynamicShapeModel()
    
    @staticmethod
    def create_custom_operations_model():
        """Model with custom operations using basic torch functions."""
        class CustomOpModel(nn.Module):
            def forward(self, x):
                mean_x = torch.mean(x, dim=-1, keepdim=True)
                std_x = torch.std(x, dim=-1, keepdim=True)
                normalized = (x - mean_x) / (std_x + 1e-6)
                return normalized
        return CustomOpModel()


class TestEdgeCaseHandling:
    """Test edge case handling and fallback strategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = HierarchyExporter(strategy="htp")
        self.temp_dir = tempfile.mkdtemp()
    
    def _validate_100_percent_coverage(self, model, inputs):
        """Helper to validate 100% coverage requirement."""
        model.eval()
        output_path = Path(self.temp_dir) / "edge_case_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Critical requirement: 100% coverage
        coverage_pct = (result['tagged_operations'] / result['total_operations']) * 100
        assert coverage_pct == 100.0, f"Failed 100% coverage requirement: {coverage_pct:.1f}%"
        
        # All operations should have non-empty tags
        if 'node_tags' in result:
            for node_name, node_info in result['node_tags'].items():
                tags = node_info.get('tags', [])
                assert len(tags) > 0, f"Node {node_name} has empty tags"
                assert isinstance(tags[0], str), f"Invalid tag type for {node_name}: {type(tags[0])}"
                assert tags[0].startswith('/'), f"Invalid tag format for {node_name}: {tags[0]}"
        
        return result
    
    def test_identity_model_edge_case(self):
        """Test identity model (previously 0% coverage)."""
        model = EdgeCaseTestModels.create_identity_model()
        inputs = torch.randn(2, 3)
        
        result = self._validate_100_percent_coverage(model, inputs)
        
        # Should have exactly one operation
        assert result['total_operations'] == 1
        assert result['tagged_operations'] == 1
        
        # Check tag quality
        if 'node_tags' in result:
            for node_info in result['node_tags'].values():
                tag = node_info['tags'][0]
                assert 'Identity' in tag, f"Tag should contain operation type: {tag}"
    
    def test_constant_heavy_model_edge_case(self):
        """Test constant-heavy model (previously 0% coverage)."""
        model = EdgeCaseTestModels.create_constant_heavy_model()
        inputs = torch.randn(3, 4)
        
        result = self._validate_100_percent_coverage(model, inputs)
        
        # Should have multiple operations
        assert result['total_operations'] > 3
        
        # Check tag patterns for constants
        if 'node_tags' in result:
            constant_operations = []
            for node_name, node_info in result['node_tags'].items():
                if node_info.get('op_type') == 'Constant':
                    constant_operations.append(node_name)
            
            assert len(constant_operations) > 0, "Should have constant operations"
            
            # Check that constants have meaningful tags
            for const_op in constant_operations:
                tag = result['node_tags'][const_op]['tags'][0]
                assert any(keyword in tag for keyword in ['Parameters', 'Constant']), \
                    f"Constant should have parameter-related tag: {tag}"
    
    def test_type_conversion_model_edge_case(self):
        """Test type conversion model (previously 0% coverage)."""
        model = EdgeCaseTestModels.create_type_conversion_model()
        inputs = torch.randn(2, 3)
        
        result = self._validate_100_percent_coverage(model, inputs)
        
        # Should have multiple Cast operations
        assert result['total_operations'] >= 3
        
        # Check Cast operation tags
        if 'node_tags' in result:
            cast_operations = []
            for node_name, node_info in result['node_tags'].items():
                if node_info.get('op_type') == 'Cast':
                    cast_operations.append(node_name)
            
            assert len(cast_operations) > 0, "Should have cast operations"
            
            # Check that casts have utility-related tags
            for cast_op in cast_operations:
                tag = result['node_tags'][cast_op]['tags'][0]
                assert any(keyword in tag for keyword in ['Utility', 'Cast']), \
                    f"Cast should have utility-related tag: {tag}"
    
    def test_reshape_heavy_model_edge_case(self):
        """Test reshape-heavy model (previously 0% coverage)."""
        model = EdgeCaseTestModels.create_reshape_heavy_model()
        inputs = torch.randn(2, 3, 4)
        
        result = self._validate_100_percent_coverage(model, inputs)
        
        # Should have multiple reshape operations
        assert result['total_operations'] >= 4
        
        # Check reshape operation tags
        if 'node_tags' in result:
            reshape_operations = []
            for node_name, node_info in result['node_tags'].items():
                if node_info.get('op_type') in ['Reshape', 'Transpose']:
                    reshape_operations.append(node_name)
            
            assert len(reshape_operations) > 0, "Should have reshape operations"
            
            # Check that reshapes have data transformation tags
            for reshape_op in reshape_operations:
                tag = result['node_tags'][reshape_op]['tags'][0]
                assert any(keyword in tag for keyword in ['DataTransformation', 'Reshape', 'Transpose']), \
                    f"Reshape should have data transformation tag: {tag}"
    
    def test_minimal_activation_model_edge_case(self):
        """Test single activation model."""
        model = EdgeCaseTestModels.create_minimal_activation_model()
        inputs = torch.randn(2, 5)
        
        result = self._validate_100_percent_coverage(model, inputs)
        
        # Should have exactly one operation
        assert result['total_operations'] == 1
        assert result['tagged_operations'] == 1
        
        # Check activation tag
        if 'node_tags' in result:
            for node_info in result['node_tags'].values():
                tag = node_info['tags'][0]
                op_type = node_info.get('op_type', '')
                assert 'Relu' in tag or 'Activation' in tag, \
                    f"Activation operation should have activation-related tag: {tag} (op_type: {op_type})"
    
    def test_dynamic_shape_model_edge_case(self):
        """Test dynamic shape model."""
        model = EdgeCaseTestModels.create_dynamic_shape_model()
        inputs = torch.randn(3, 4)
        
        result = self._validate_100_percent_coverage(model, inputs)
        
        # Should have multiple operations including shape-related ones
        assert result['total_operations'] > 1
        
        # Validate that all operations are tagged meaningfully
        if 'node_tags' in result:
            for node_name, node_info in result['node_tags'].items():
                tag = node_info['tags'][0]
                op_type = node_info.get('op_type', '')
                
                # Each tag should be semantically appropriate for the operation type
                if op_type == 'Gemm':
                    assert any(keyword in tag for keyword in ['Computation', 'Linear']), \
                        f"Gemm should have computation tag: {tag}"
                elif op_type == 'Constant':
                    assert any(keyword in tag for keyword in ['Parameters', 'Constant']), \
                        f"Constant should have parameters tag: {tag}"
    
    def test_custom_operations_model_edge_case(self):
        """Test custom operations model."""
        model = EdgeCaseTestModels.create_custom_operations_model()
        inputs = torch.randn(3, 5)
        
        result = self._validate_100_percent_coverage(model, inputs)
        
        # Should have multiple operations including reduce operations
        assert result['total_operations'] > 5
        
        # Check for reduce operations
        if 'node_tags' in result:
            reduce_operations = []
            for node_name, node_info in result['node_tags'].items():
                if 'ReduceMean' in node_info.get('op_type', ''):
                    reduce_operations.append(node_name)
            
            # Should have some reduce operations and they should be tagged appropriately
            for reduce_op in reduce_operations:
                tag = result['node_tags'][reduce_op]['tags'][0]
                assert any(keyword in tag for keyword in ['Aggregation', 'ReduceMean']), \
                    f"ReduceMean should have aggregation tag: {tag}"


class TestFallbackStrategyEffectiveness:
    """Test the effectiveness of multi-level fallback strategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = HierarchyExporter(strategy="htp")
        self.temp_dir = tempfile.mkdtemp()
    
    def test_fallback_strategy_distribution(self):
        """Test that fallback strategies are used appropriately."""
        # Test with a mixed model that exercises different fallback levels
        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)
            
            def forward(self, x):
                # This will create a mix of main computation and auxiliary operations
                constant = torch.tensor(2.0, dtype=x.dtype)
                reshaped = x.reshape(-1, 4)
                linear_out = self.linear(reshaped * constant)
                return linear_out.squeeze()
        
        model = MixedModel()
        model.eval()
        inputs = torch.randn(2, 4)
        
        output_path = Path(self.temp_dir) / "mixed_fallback_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Should achieve 100% coverage
        coverage_pct = (result['tagged_operations'] / result['total_operations']) * 100
        assert coverage_pct == 100.0, f"Should achieve 100% coverage: {coverage_pct:.1f}%"
        
        # Analyze fallback strategy usage
        if 'node_tags' in result:
            tag_patterns = {}
            for node_name, node_info in result['node_tags'].items():
                tag = node_info['tags'][0]
                op_type = node_info.get('op_type', 'Unknown')
                
                # Classify tag pattern
                if '/Parameters/' in tag:
                    pattern = 'parameters'
                elif '/Computation/' in tag:
                    pattern = 'computation'
                elif '/DataTransformation/' in tag:
                    pattern = 'data_transformation'
                elif '/Elementwise/' in tag:
                    pattern = 'elementwise'
                elif '/UniversalFallback/' in tag:
                    pattern = 'universal_fallback'
                else:
                    pattern = 'context_inherited'
                
                if pattern not in tag_patterns:
                    tag_patterns[pattern] = []
                tag_patterns[pattern].append((node_name, op_type))
            
            # Should have multiple pattern types (indicating effective fallback)
            assert len(tag_patterns) >= 2, f"Should have multiple fallback patterns: {list(tag_patterns.keys())}"
            
            # Should not rely solely on universal fallback
            total_ops = sum(len(ops) for ops in tag_patterns.values())
            universal_fallback_ops = len(tag_patterns.get('universal_fallback', []))
            universal_fallback_rate = (universal_fallback_ops / total_ops) * 100
            
            assert universal_fallback_rate < 50.0, \
                f"Universal fallback rate too high: {universal_fallback_rate:.1f}%"
    
    def test_tag_semantic_quality(self):
        """Test that fallback tags are semantically meaningful."""
        models_and_inputs = [
            (EdgeCaseTestModels.create_constant_heavy_model(), torch.randn(2, 3)),
            (EdgeCaseTestModels.create_reshape_heavy_model(), torch.randn(2, 3, 4)),
            (EdgeCaseTestModels.create_custom_operations_model(), torch.randn(3, 5))
        ]
        
        for model, inputs in models_and_inputs:
            model.eval()
            output_path = Path(self.temp_dir) / f"semantic_test_{id(model)}.onnx"
            result = self.exporter.export(model, inputs, str(output_path))
            
            # Should achieve 100% coverage
            coverage_pct = (result['tagged_operations'] / result['total_operations']) * 100
            assert coverage_pct == 100.0, f"Should achieve 100% coverage: {coverage_pct:.1f}%"
            
            # Check semantic quality of tags
            if 'node_tags' in result:
                for node_name, node_info in result['node_tags'].items():
                    tag = node_info['tags'][0]
                    op_type = node_info.get('op_type', '')
                    
                    # Tags should be semantically appropriate
                    self._validate_tag_semantic_quality(tag, op_type)
    
    def _validate_tag_semantic_quality(self, tag: str, op_type: str):
        """Validate that a tag is semantically appropriate for the operation type."""
        # Basic format validation
        assert tag.startswith('/'), f"Tag should start with '/': {tag}"
        assert tag.count('/') >= 2, f"Tag should have at least 3 parts: {tag}"
        
        # Semantic validation based on operation type
        tag_lower = tag.lower()
        op_type_lower = op_type.lower()
        
        if op_type == 'Constant':
            assert any(keyword in tag_lower for keyword in ['parameter', 'constant']), \
                f"Constant operation should have parameter-related tag: {tag}"
        elif op_type in ['Reshape', 'Transpose']:
            assert any(keyword in tag_lower for keyword in ['transformation', 'reshape', 'transpose']), \
                f"Reshape operation should have transformation tag: {tag}"
        elif op_type == 'ReduceMean':
            assert any(keyword in tag_lower for keyword in ['aggregation', 'reduce']), \
                f"ReduceMean operation should have aggregation tag: {tag}"
        elif op_type == 'Cast':
            assert any(keyword in tag_lower for keyword in ['utility', 'cast']), \
                f"Cast operation should have utility tag: {tag}"
        elif op_type in ['Add', 'Sub', 'Mul', 'Div']:
            assert any(keyword in tag_lower for keyword in ['elementwise', 'add', 'sub', 'mul', 'div']), \
                f"Elementwise operation should have elementwise tag: {tag}"


class TestEdgeCaseRobustness:
    """Test system robustness across edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = HierarchyExporter(strategy="htp")
        self.temp_dir = tempfile.mkdtemp()
    
    def test_no_system_failures(self):
        """Test that no edge case causes complete system failure."""
        edge_case_models = [
            (EdgeCaseTestModels.create_identity_model(), torch.randn(1, 1)),
            (EdgeCaseTestModels.create_constant_heavy_model(), torch.randn(1, 2)),
            (EdgeCaseTestModels.create_reshape_heavy_model(), torch.randn(1, 2, 3)),
            (EdgeCaseTestModels.create_type_conversion_model(), torch.randn(1, 2)),
            (EdgeCaseTestModels.create_minimal_activation_model(), torch.randn(1, 3)),
            (EdgeCaseTestModels.create_dynamic_shape_model(), torch.randn(2, 4)),
            (EdgeCaseTestModels.create_custom_operations_model(), torch.randn(2, 3))
        ]
        
        successful_exports = 0
        total_models = len(edge_case_models)
        
        for i, (model, inputs) in enumerate(edge_case_models):
            try:
                model.eval()
                output_path = Path(self.temp_dir) / f"robustness_test_{i}.onnx"
                result = self.exporter.export(model, inputs, str(output_path))
                
                # Should achieve some level of coverage (preferably 100%)
                coverage_pct = (result['tagged_operations'] / result['total_operations']) * 100
                assert coverage_pct > 0, f"Model {i} achieved 0% coverage"
                
                successful_exports += 1
                
            except Exception as e:
                # Edge cases should not cause complete failures
                pytest.fail(f"Model {i} caused system failure: {str(e)}")
        
        # Should export all models successfully
        assert successful_exports == total_models, \
            f"Only {successful_exports}/{total_models} models exported successfully"
    
    def test_performance_impact_of_edge_cases(self):
        """Test that edge case handling doesn't significantly impact performance."""
        import time
        
        # Test with a normal model vs edge case models
        normal_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        edge_case_model = EdgeCaseTestModels.create_constant_heavy_model()
        
        models_and_inputs = [
            ("Normal Model", normal_model, torch.randn(3, 10)),
            ("Edge Case Model", edge_case_model, torch.randn(3, 4))
        ]
        
        export_times = {}
        
        for model_name, model, inputs in models_and_inputs:
            model.eval()
            
            start_time = time.perf_counter()
            output_path = Path(self.temp_dir) / f"perf_test_{model_name.replace(' ', '_')}.onnx"
            result = self.exporter.export(model, inputs, str(output_path))
            end_time = time.perf_counter()
            
            export_time = end_time - start_time
            export_times[model_name] = export_time
            
            # Should achieve 100% coverage
            coverage_pct = (result['tagged_operations'] / result['total_operations']) * 100
            assert coverage_pct == 100.0, f"{model_name} should achieve 100% coverage"
        
        # Edge case handling should not significantly slow down exports
        normal_time = export_times["Normal Model"]
        edge_case_time = export_times["Edge Case Model"]
        
        # Allow up to 3x slower for edge cases (very generous threshold)
        time_ratio = edge_case_time / normal_time if normal_time > 0 else 1.0
        assert time_ratio < 3.0, \
            f"Edge case handling too slow: {time_ratio:.1f}x slower than normal"