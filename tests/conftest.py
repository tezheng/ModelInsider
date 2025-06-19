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