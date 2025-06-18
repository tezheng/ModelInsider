"""
Test configuration and fixtures for modelexport tests.

This module provides common test fixtures and utilities for testing
the universal hierarchy-preserving ONNX export functionality.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple

try:
    from transformers import BertModel, BertTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.fixture(scope="session")
def test_data_dir():
    """Temporary directory for test data that persists across the session."""
    temp_dir = tempfile.mkdtemp(prefix="modelexport_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session") 
def bert_tiny_model():
    """Load BERT tiny model for testing (session-scoped for efficiency)."""
    if not HAS_TRANSFORMERS:
        pytest.skip("transformers library not available")
    
    model_name = "google/bert_uncased_L-2_H-128_A-2"
    try:
        model = BertModel.from_pretrained(model_name)
        model.eval()
        return model
    except Exception as e:
        pytest.skip(f"Could not load BERT model: {e}")


@pytest.fixture(scope="session")
def bert_tiny_tokenizer():
    """Load BERT tiny tokenizer."""
    if not HAS_TRANSFORMERS:
        pytest.skip("transformers library not available")
        
    model_name = "google/bert_uncased_L-2_H-128_A-2"
    try:
        return BertTokenizer.from_pretrained(model_name)
    except Exception as e:
        pytest.skip(f"Could not load BERT tokenizer: {e}")


@pytest.fixture
def sample_input():
    """Generate sample input for BERT model."""
    text = "Hello world, this is a test."
    return text


@pytest.fixture 
def bert_model_inputs(bert_tiny_tokenizer, sample_input):
    """Generate BERT model inputs from sample text."""
    inputs = bert_tiny_tokenizer(
        sample_input, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32
    )
    return inputs


@pytest.fixture
def simple_pytorch_model():
    """Create a simple PyTorch model for basic testing."""
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(10, 5)
            self.relu = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(5, 1)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    model = SimpleModel()
    model.eval()
    return model


@pytest.fixture
def simple_model_input():
    """Input for simple PyTorch model."""
    return torch.randn(1, 10)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may download models)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )