"""
Pytest configuration and fixtures for modelexport tests.

Centralized test configurations including BERT input shapes (1, 128 as default).
"""

import pytest
import torch
from transformers import AutoModel, AutoTokenizer

# Test configurations for BERT input shapes
BERT_TEST_CONFIGS = {
    "default": {
        "batch_size": 1,
        "sequence_length": 128,
        "model": "prajjwal1/bert-tiny",
        "text": "This is a test input for BERT model with default configuration settings that should be around 128 tokens when tokenized properly."
    },
    "short": {
        "batch_size": 1,
        "sequence_length": 32, 
        "model": "prajjwal1/bert-tiny",
        "text": "Short test input."
    },
    "batch": {
        "batch_size": 4,
        "sequence_length": 128,
        "model": "prajjwal1/bert-tiny",
        "text": [
            "First batch item for testing with reasonable length.",
            "Second batch item with different content for variety.",
            "Third item shorter.",
            "Fourth and final batch item for comprehensive testing purposes."
        ]
    }
}


@pytest.fixture(scope="session")
def bert_model_cache():
    """Cache BERT model and tokenizer for session to avoid repeated loading."""
    model_name = "prajjwal1/bert-tiny"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


@pytest.fixture
def bert_test_config():
    """Provide BERT test configurations."""
    return BERT_TEST_CONFIGS


@pytest.fixture
def prepared_bert_inputs(bert_model_cache):
    """Prepare BERT inputs with default configuration (1, 128)."""
    model, tokenizer = bert_model_cache
    config = BERT_TEST_CONFIGS["default"]
    
    inputs = tokenizer(
        config["text"],
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=config["sequence_length"]
    )
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'inputs': inputs,
        'config': config
    }


@pytest.fixture
def temp_test_workspace(tmp_path):
    """Create structured temporary workspace for tests."""
    workspace = tmp_path / "test_workspace"
    
    subdirs = {
        'models': workspace / 'models',
        'exports': workspace / 'exports',
        'analysis': workspace / 'analysis',
        'comparisons': workspace / 'comparisons',
        'reports': workspace / 'reports'
    }
    
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    
    return workspace, subdirs


@pytest.fixture  
def test_data_dir(tmp_path):
    """Create temporary directory for test data."""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture
def simple_pytorch_model():
    """Simple PyTorch model for basic testing."""
    class SimpleTestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(10, 5)
            self.relu = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(5, 2)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    model = SimpleTestModel()
    model.eval()
    return model


@pytest.fixture
def simple_model_input():
    """Simple input tensor for basic testing."""
    return torch.randn(1, 10)


@pytest.fixture
def bert_tiny_model(bert_model_cache):
    """BERT-tiny model for integration testing."""
    model, _ = bert_model_cache
    return model


@pytest.fixture
def bert_model_inputs(bert_model_cache):
    """BERT model inputs for integration testing.""" 
    _, tokenizer = bert_model_cache
    
    text = "This is a test input for BERT model testing."
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=128
    )
    return inputs


@pytest.fixture
def complex_hierarchical_model():
    """Complex model with deep hierarchy for testing edge cases."""
    class AttentionBlock(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.query = torch.nn.Linear(hidden_size, hidden_size)
            self.key = torch.nn.Linear(hidden_size, hidden_size)
            self.value = torch.nn.Linear(hidden_size, hidden_size)
            self.dropout = torch.nn.Dropout(0.1)
            
        def forward(self, x):
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)
            # Simplified attention
            attn = torch.matmul(q, k.transpose(-2, -1))
            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            return torch.matmul(attn, v)
    
    class TransformerLayer(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.attention = AttentionBlock(hidden_size)
            self.layer_norm1 = torch.nn.LayerNorm(hidden_size)
            self.ffn = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size * 4),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_size * 4, hidden_size)
            )
            self.layer_norm2 = torch.nn.LayerNorm(hidden_size)
            
        def forward(self, x):
            attn_out = self.attention(x)
            x = self.layer_norm1(x + attn_out)
            ffn_out = self.ffn(x)
            return self.layer_norm2(x + ffn_out)
    
    class ComplexModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(1000, 256)
            self.pos_encoding = torch.nn.Parameter(torch.randn(512, 256))
            self.layers = torch.nn.ModuleList([
                TransformerLayer(256) for _ in range(3)
            ])
            self.classifier = torch.nn.Linear(256, 10)
            
        def forward(self, x):
            x = self.embedding(x)
            x = x + self.pos_encoding[:x.size(1)]
            for layer in self.layers:
                x = layer(x)
            return self.classifier(x.mean(dim=1))
    
    model = ComplexModel()
    model.eval()
    return model


@pytest.fixture
def complex_model_input():
    """Input for complex hierarchical model."""
    return torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10


# Additional fixtures for new test infrastructure

@pytest.fixture(scope="session")
def test_models():
    """Provide test model fixtures for the entire test session."""
    from .fixtures.test_models import TestModelFixtures
    return TestModelFixtures()


@pytest.fixture(scope="function")
def temp_dir():
    """Provide a temporary directory for each test."""
    import shutil
    import tempfile
    from pathlib import Path
    
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def sample_onnx_path(temp_dir):
    """Provide a sample ONNX file path for testing."""
    return str(temp_dir / "test_model.onnx")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests across components")
    config.addinivalue_line("markers", "strategy: Strategy-specific tests")
    config.addinivalue_line("markers", "htp: HTP strategy tests")
    config.addinivalue_line("markers", "cli: CLI integration tests")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")
    config.addinivalue_line("markers", "requires_transformers: Tests requiring transformers")
    config.addinivalue_line("markers", "requires_onnx: Tests requiring ONNX runtime")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        test_path = str(item.fspath)
        
        if "/unit/" in test_path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in test_path:
            item.add_marker(pytest.mark.integration)
        
        if "/test_strategies/" in test_path:
            item.add_marker(pytest.mark.strategy)
            
        elif "/htp/" in test_path:
            item.add_marker(pytest.mark.htp)
            
        if "/test_cli" in test_path:
            item.add_marker(pytest.mark.cli)
        
        # Add markers based on test content
        if "transformers" in test_path or "huggingface" in test_path.lower():
            item.add_marker(pytest.mark.requires_transformers)
        
        if "slow" in item.name.lower() or "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)


def pytest_runtest_setup(item):
    """Set up each test run."""
    # Skip tests that require optional dependencies
    if item.get_closest_marker("requires_transformers"):
        pytest.importorskip("transformers")
    
    if item.get_closest_marker("requires_onnx"):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")


@pytest.fixture(autouse=True)
def reset_model_state():
    """Reset model state before each test."""
    # Clear any cached models or state
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    
    # Set deterministic behavior for testing
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return {
        "test_timeout": 30,  # seconds
        "temp_file_cleanup": True,
        "verbose_output": False,
        "skip_slow_tests": False
    }