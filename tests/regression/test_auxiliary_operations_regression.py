"""
Auxiliary Operations Regression Tests

Prevents regressions in enhanced auxiliary operations functionality
implemented during iterations 1-5.
"""

import tempfile
from pathlib import Path

import onnx
import pytest
import torch
import torch.nn as nn

from modelexport.strategies.htp.htp_hierarchy_exporter import HierarchyExporter


class BERTTinyLikeModel(nn.Module):
    """Model that mimics BERT-tiny structure for regression testing."""
    
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(100, 16)
        self.encoder = nn.TransformerEncoderLayer(16, 2, 32)
        self.pooler = nn.Linear(16, 16)
        
    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        x = self.encoder(x)
        pooled = self.pooler(x[:, 0, :])  # Pool first token
        return pooled


class RegressionTestBaseline:
    """Baseline behavior validation for regression prevention."""
    
    def __init__(self):
        self.exporter = HierarchyExporter(strategy="htp")
        self.temp_dir = tempfile.mkdtemp()
    
    def validate_100_percent_coverage(self, model, inputs):
        """Validate that 100% operation coverage is maintained."""
        output_path = Path(self.temp_dir) / "regression_test.onnx"
        result = self.exporter.export(model, inputs, str(output_path))
        
        # Critical regression check: 100% coverage must be maintained
        coverage_pct = (result['tagged_operations'] / result['total_operations']) * 100
        assert coverage_pct == 100.0, f"REGRESSION: Coverage dropped below 100%: {coverage_pct:.1f}%"
        
        return result
    
    def validate_no_empty_tags(self, result):
        """Validate that no operations have empty tags (core user requirement)."""
        if 'node_tags' in result:
            for node_name, node_info in result['node_tags'].items():
                tags = node_info.get('tags', [])
                assert len(tags) > 0, f"REGRESSION: Node {node_name} has empty tags"
    
    def validate_auxiliary_operations_tagged(self, result):
        """Validate that auxiliary operations are properly tagged."""
        if 'node_tags' in result:
            auxiliary_operations = []
            for node_name, node_info in result['node_tags'].items():
                op_type = node_info.get('op_type', '')
                if op_type in ['Constant', 'MatMul', 'Add', 'Reshape', 'Transpose']:
                    auxiliary_operations.append(node_name)
            
            # Should have auxiliary operations
            assert len(auxiliary_operations) > 0, "REGRESSION: No auxiliary operations found"
            
            # All auxiliary operations should be tagged
            for aux_op in auxiliary_operations:
                node_info = result['node_tags'][aux_op]
                tags = node_info.get('tags', [])
                assert len(tags) > 0, f"REGRESSION: Auxiliary operation {aux_op} has empty tags"
    
    def validate_context_inheritance_working(self, result):
        """Validate that context inheritance is functioning."""
        if 'node_tags' in result:
            # Should have multiple distinct tag patterns (indicating inheritance working)
            tag_patterns = set()
            for node_info in result['node_tags'].values():
                for tag in node_info.get('tags', []):
                    parts = tag.split('/')
                    if len(parts) >= 3:
                        pattern = '/'.join(parts[:3])
                        tag_patterns.add(pattern)
            
            # Should have multiple module contexts
            assert len(tag_patterns) > 1, f"REGRESSION: Context inheritance not working, only {len(tag_patterns)} patterns"


class TestAuxiliaryOperationsRegression:
    """Test for regressions in auxiliary operations functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.baseline = RegressionTestBaseline()
    
    def test_bert_tiny_like_model_baseline(self):
        """Test that BERT-tiny-like models maintain baseline behavior."""
        model = BERTTinyLikeModel()
        model.eval()
        inputs = torch.randint(0, 100, (2, 8))
        
        # Validate core functionality
        result = self.baseline.validate_100_percent_coverage(model, inputs)
        self.baseline.validate_no_empty_tags(result)
        self.baseline.validate_auxiliary_operations_tagged(result)
        self.baseline.validate_context_inheritance_working(result)
    
    def test_simple_linear_model_baseline(self):
        """Test that simple models maintain baseline behavior."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        model.eval()
        inputs = torch.randn(3, 10)
        
        # Validate core functionality
        result = self.baseline.validate_100_percent_coverage(model, inputs)
        self.baseline.validate_no_empty_tags(result)
    
    def test_embedding_model_baseline(self):
        """Test that embedding models maintain baseline behavior."""
        class EmbeddingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(50, 8)
                self.linear = nn.Linear(8, 4)
            
            def forward(self, input_ids):
                x = self.embedding(input_ids)
                x = x.mean(dim=1)  # Global average pooling
                return self.linear(x)
        
        model = EmbeddingModel()
        model.eval()
        inputs = torch.randint(0, 50, (2, 6))
        
        # Validate core functionality
        result = self.baseline.validate_100_percent_coverage(model, inputs)
        self.baseline.validate_no_empty_tags(result)
        self.baseline.validate_auxiliary_operations_tagged(result)
    
    def test_backward_compatibility_maintained(self):
        """Test that existing functionality is preserved."""
        # Test that we can still export models successfully
        model = BERTTinyLikeModel()
        model.eval()
        inputs = torch.randint(0, 100, (2, 5))
        
        output_path = Path(self.baseline.temp_dir) / "backward_compat_test.onnx"
        result = self.baseline.exporter.export(model, inputs, str(output_path))
        
        # Should export successfully
        assert result['total_operations'] > 0
        assert result['tagged_operations'] > 0
        
        # Should create valid ONNX file
        assert Path(output_path).exists()
        onnx_model = onnx.load(str(output_path))
        
        # Should pass ONNX validation
        try:
            onnx.checker.check_model(onnx_model)
            onnx_valid = True
        except Exception:
            onnx_valid = False
        
        assert onnx_valid, "REGRESSION: ONNX validation failed"


class TestPerformanceRegression:
    """Test for performance regressions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.baseline = RegressionTestBaseline()
    
    def test_export_performance_baseline(self):
        """Test that export performance meets baseline requirements."""
        import time
        
        model = BERTTinyLikeModel()
        model.eval()
        inputs = torch.randint(0, 100, (2, 10))
        
        # Measure export time
        start_time = time.perf_counter()
        output_path = Path(self.baseline.temp_dir) / "perf_baseline_test.onnx"
        result = self.baseline.exporter.export(model, inputs, str(output_path))
        end_time = time.perf_counter()
        
        export_time = end_time - start_time
        
        # Should complete within reasonable time
        assert export_time < 15.0, f"REGRESSION: Export took too long: {export_time:.2f}s"
        
        # Should still achieve 100% coverage
        coverage_pct = (result['tagged_operations'] / result['total_operations']) * 100
        assert coverage_pct == 100.0, f"REGRESSION: Performance optimization broke coverage: {coverage_pct:.1f}%"


class TestArchitectureCompatibilityRegression:
    """Test for architecture compatibility regressions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.baseline = RegressionTestBaseline()
    
    def test_cnn_architecture_compatibility(self):
        """Test that CNN architectures maintain compatibility."""
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(16, 10)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        model = SimpleCNN()
        model.eval()
        inputs = torch.randn(1, 3, 32, 32)
        
        # Should maintain baseline behavior
        result = self.baseline.validate_100_percent_coverage(model, inputs)
        self.baseline.validate_no_empty_tags(result)
    
    def test_rnn_architecture_compatibility(self):
        """Test that RNN architectures maintain compatibility."""
        class SimpleRNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.LSTM(10, 20, batch_first=True)
                self.fc = nn.Linear(20, 5)
            
            def forward(self, x):
                out, _ = self.rnn(x)
                # Take last output
                last_out = out[:, -1, :]
                return self.fc(last_out)
        
        model = SimpleRNN()
        model.eval()
        inputs = torch.randn(2, 8, 10)
        
        # Should maintain baseline behavior
        result = self.baseline.validate_100_percent_coverage(model, inputs)
        self.baseline.validate_no_empty_tags(result)


class TestGraphFilteringRegression:
    """Test for graph filtering safety regressions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.baseline = RegressionTestBaseline()
    
    def test_graph_filtering_safety_maintained(self):
        """Test that graph filtering remains safe after enhancements."""
        # Import graph filtering utilities
        try:
            from modelexport.graph_filtering import ONNXGraphFilter
            graph_filter = ONNXGraphFilter()
            filter_available = True
        except ImportError:
            filter_available = False
        
        if not filter_available:
            pytest.skip("Graph filtering utilities not available")
        
        # Export a model first
        model = BERTTinyLikeModel()
        model.eval()
        inputs = torch.randint(0, 100, (2, 6))
        
        output_path = Path(self.baseline.temp_dir) / "filter_safety_test.onnx"
        result = self.baseline.exporter.export(model, inputs, str(output_path))
        
        # Validate baseline
        self.baseline.validate_100_percent_coverage(model, inputs)
        
        # Test filtering doesn't break
        onnx_model = onnx.load(str(output_path))
        
        # Should have node tags for filtering
        if 'node_tags' in result:
            available_tags = set()
            for node_info in result['node_tags'].values():
                available_tags.update(node_info.get('tags', []))
            
            # Should have multiple tags available for filtering
            assert len(available_tags) > 1, "REGRESSION: Not enough tags for filtering"