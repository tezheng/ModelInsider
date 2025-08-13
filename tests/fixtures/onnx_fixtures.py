"""
ONNX model fixtures for testing.
"""

import json

import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class MediumModel(nn.Module):
    """Medium complexity model for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 6 * 6, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class LargeModel(nn.Module):
    """Large model with multiple layers for testing."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.layers(x)


def create_onnx_model(model, input_shape, output_path):
    """Helper to create ONNX model from PyTorch model."""
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    return output_path


@pytest.fixture
def simple_onnx_model(tmp_path):
    """Create a simple ONNX model for testing."""
    model = SimpleModel()
    model.eval()
    output_path = tmp_path / "simple_model.onnx"
    return create_onnx_model(model, (1, 10), str(output_path))


@pytest.fixture
def medium_onnx_model(tmp_path):
    """Create a medium complexity ONNX model for testing."""
    model = MediumModel()
    model.eval()
    output_path = tmp_path / "medium_model.onnx"
    return create_onnx_model(model, (1, 3, 16, 16), str(output_path))


@pytest.fixture
def large_onnx_model(tmp_path):
    """Create a large ONNX model for testing."""
    model = LargeModel()
    model.eval()
    output_path = tmp_path / "large_model.onnx"
    return create_onnx_model(model, (1, 100), str(output_path))


@pytest.fixture
def malformed_onnx_file(tmp_path):
    """Create a malformed ONNX file for error testing."""
    malformed_path = tmp_path / "malformed.onnx"
    with open(malformed_path, "wb") as f:
        f.write(b"This is not a valid ONNX file")
    return str(malformed_path)


@pytest.fixture
def bert_tiny_onnx_path(tmp_path):
    """Path to a BERT tiny ONNX model (mock for testing)."""
    # For actual tests, this should point to a real BERT model
    # For now, we'll create a simple substitute
    model = SimpleModel()
    model.eval()
    output_path = tmp_path / "bert_tiny.onnx"
    return create_onnx_model(model, (1, 10), str(output_path))


@pytest.fixture
def bert_tiny_metadata_path(tmp_path):
    """Create HTP metadata for BERT tiny model."""
    metadata = {
        "model": {"name_or_path": "prajjwal1/bert-tiny", "class_name": "BertModel"},
        "modules": {
            "scope": "/BertModel",
            "class_name": "BertModel",
            "traced_tag": "/BertModel",
            "execution_order": 0,
            "children": {
                "embeddings": {
                    "scope": "/BertModel/embeddings",
                    "class_name": "BertEmbeddings",
                    "traced_tag": "/BertModel/BertEmbeddings",
                    "execution_order": 1,
                    "children": {
                        "word_embeddings": {
                            "scope": "/BertModel/embeddings/word_embeddings",
                            "class_name": "Embedding",
                            "traced_tag": "/BertModel/BertEmbeddings/Embedding",
                            "execution_order": 2,
                        }
                    },
                },
                "encoder": {
                    "scope": "/BertModel/encoder",
                    "class_name": "BertEncoder",
                    "traced_tag": "/BertModel/BertEncoder",
                    "execution_order": 3,
                    "children": {
                        "layer": {
                            "scope": "/BertModel/encoder/layer",
                            "class_name": "ModuleList",
                            "traced_tag": "/BertModel/BertEncoder/ModuleList",
                            "execution_order": 4,
                        }
                    },
                },
            },
        },
    }

    metadata_path = tmp_path / "bert_tiny_htp_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return str(metadata_path)
