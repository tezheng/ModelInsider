"""
Pytest configuration and fixtures for ONNX inference tests.

This module provides shared test fixtures and configuration for testing
the ONNX inference implementation components.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import numpy as np
import onnx
import pytest
import torch
from transformers import AutoConfig, AutoTokenizer


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Fixture providing the test data directory path."""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def temp_model_dir() -> Generator[Path, None, None]:
    """Fixture providing a temporary directory for test models."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_bert_config() -> Dict[str, Any]:
    """Fixture providing a sample BERT model configuration."""
    return {
        "model_type": "bert",
        "architectures": ["BertForSequenceClassification"],
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "vocab_size": 1000,
        "max_position_embeddings": 128,
        "num_labels": 2,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "pad_token_id": 0,
        "position_embedding_type": "absolute"
    }


@pytest.fixture
def sample_vision_config() -> Dict[str, Any]:
    """Fixture providing a sample Vision Transformer configuration."""
    return {
        "model_type": "vit",
        "architectures": ["ViTForImageClassification"],
        "hidden_size": 192,
        "num_hidden_layers": 2,
        "num_attention_heads": 3,
        "intermediate_size": 768,
        "image_size": 224,
        "patch_size": 16,
        "num_channels": 3,
        "num_labels": 1000,
        "layer_norm_eps": 1e-12,
        "initializer_range": 0.02
    }


@pytest.fixture
def mock_onnx_model_path(temp_model_dir: Path, sample_bert_config: Dict[str, Any]) -> Path:
    """Fixture providing a mock ONNX model directory with config files."""
    model_dir = temp_model_dir / "mock_bert_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config.json
    config_path = model_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(sample_bert_config, f, indent=2)
    
    # Create tokenizer files
    tokenizer_config = {
        "tokenizer_class": "BertTokenizer",
        "do_lower_case": True,
        "vocab_size": 1000,
        "max_len": 128,
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]"
    }
    
    with open(model_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    
    # Create minimal tokenizer.json
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": {}, "unk_token": "[UNK]"}
    }
    
    with open(model_dir / "tokenizer.json", "w") as f:
        json.dump(tokenizer_json, f, indent=2)
    
    # Create minimal vocab.txt
    vocab_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"token_{i}" for i in range(995)]
    with open(model_dir / "vocab.txt", "w") as f:
        for token in vocab_tokens:
            f.write(f"{token}\n")
    
    # Create special_tokens_map.json
    special_tokens = {
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "mask_token": "[MASK]"
    }
    
    with open(model_dir / "special_tokens_map.json", "w") as f:
        json.dump(special_tokens, f, indent=2)
    
    return model_dir


@pytest.fixture
def sample_onnx_model(temp_model_dir: Path) -> Path:
    """Fixture providing a minimal ONNX model for testing."""
    # Create a simple ONNX model with fixed input shapes
    model_path = temp_model_dir / "sample_model.onnx"
    
    # Create input tensors
    input_ids = onnx.helper.make_tensor_value_info(
        'input_ids', onnx.TensorProto.INT64, [1, 128]
    )
    attention_mask = onnx.helper.make_tensor_value_info(
        'attention_mask', onnx.TensorProto.INT64, [1, 128]
    )
    
    # Create output tensor
    logits = onnx.helper.make_tensor_value_info(
        'logits', onnx.TensorProto.FLOAT, [1, 2]
    )
    
    # Create a simple identity node (for testing purposes)
    node = onnx.helper.make_node(
        'Identity',
        inputs=['input_ids'],
        outputs=['logits'],
    )
    
    # Create the graph
    graph = onnx.helper.make_graph(
        [node],
        'test_model',
        [input_ids, attention_mask],
        [logits]
    )
    
    # Create the model
    model = onnx.helper.make_model(graph)
    model.opset_import[0].version = 17
    
    # Save the model
    onnx.save(model, str(model_path))
    
    return model_path


@pytest.fixture
def mock_onnx_model_with_file(mock_onnx_model_path: Path, sample_onnx_model: Path) -> Path:
    """Fixture providing a complete mock ONNX model directory with .onnx file."""
    # Copy the ONNX model file to the mock model directory
    import shutil
    onnx_file_path = mock_onnx_model_path / "model.onnx"
    shutil.copy(sample_onnx_model, onnx_file_path)
    
    return mock_onnx_model_path


@pytest.fixture
def sample_tokenizer(mock_onnx_model_path: Path):
    """Fixture providing a sample tokenizer."""
    try:
        return AutoTokenizer.from_pretrained(str(mock_onnx_model_path))
    except Exception:
        # Return a mock tokenizer if loading fails
        return MockTokenizer()


@pytest.fixture
def sample_texts() -> list[str]:
    """Fixture providing sample texts for testing."""
    return [
        "Hello world!",
        "This is a test sentence.",
        "BERT is a transformer model.",
        "Testing ONNX inference with multiple inputs.",
        ""  # Empty string edge case
    ]


@pytest.fixture
def sample_batch_texts() -> list[str]:
    """Fixture providing a batch of texts for testing."""
    return [
        "First test sentence for batch processing.",
        "Second sentence in the batch.",
        "Third and final sentence for testing."
    ]


class MockTokenizer:
    """Mock tokenizer for testing when real tokenizer loading fails."""
    
    def __init__(self):
        self.pad_token_id = 0
        self.vocab_size = 1000
        self.model_max_length = 128
    
    def __call__(self, text, **kwargs):
        """Mock tokenization that returns fixed-shape outputs."""
        if isinstance(text, str):
            text = [text]
        
        batch_size = len(text)
        max_length = kwargs.get("max_length", 128)
        
        # Create mock token ids
        input_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
        attention_mask = torch.ones((batch_size, max_length), dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def batch_decode(self, token_ids, **kwargs):
        """Mock batch decode."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return [f"decoded_text_{i}" for i in range(len(token_ids))]
    
    def decode(self, token_ids, **kwargs):
        """Mock decode."""
        return "decoded_text"
    
    def tokenize(self, text):
        """Mock tokenize."""
        return text.split()[:10]  # Simple word splitting, limited length


class MockORTModel:
    """Mock ORTModel for testing."""
    
    def __init__(self, model_path: Path):
        self.path = model_path
        self.model_path = model_path
        self.task = "feature-extraction"
        
        # Mock session
        self.model = MockSession()
        self.session = self.model
    
    def __call__(self, **inputs):
        """Mock forward pass."""
        batch_size = inputs.get("input_ids", torch.tensor([[1]])).shape[0]
        return MockModelOutput(batch_size)
    
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """Mock from_pretrained method."""
        return cls(Path(model_path))


class MockSession:
    """Mock ONNX Runtime session."""
    
    def get_inputs(self):
        """Mock get_inputs method."""
        return [
            MockInputInfo("input_ids", [1, 128]),
            MockInputInfo("attention_mask", [1, 128])
        ]
    
    def get_outputs(self):
        """Mock get_outputs method."""
        return [MockOutputInfo("logits", [1, 2])]


class MockInputInfo:
    """Mock ONNX input info."""
    
    def __init__(self, name: str, shape: list[int]):
        self.name = name
        self.shape = shape


class MockOutputInfo:
    """Mock ONNX output info."""
    
    def __init__(self, name: str, shape: list[int]):
        self.name = name
        self.shape = shape


class MockModelOutput:
    """Mock model output."""
    
    def __init__(self, batch_size: int):
        self.last_hidden_state = torch.randn(batch_size, 128, 128)
        self.logits = torch.randn(batch_size, 2)
        self.pooler_output = torch.randn(batch_size, 128)


@pytest.fixture
def mock_ort_model():
    """Fixture providing a mock ORTModel."""
    return MockORTModel


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.smoke = pytest.mark.smoke
pytest.mark.slow = pytest.mark.slow


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "smoke: Smoke tests")
    config.addinivalue_line("markers", "slow: Slow tests")