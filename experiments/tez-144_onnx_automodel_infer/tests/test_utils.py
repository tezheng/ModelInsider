"""
Test Utilities and Helper Functions for ONNXAutoProcessor Tests

This module provides utilities for creating mock ONNX models, test fixtures,
and helper functions for comprehensive testing of the ONNX Auto Processor system.

Key Components:
- Mock ONNX model creation for all modalities
- Test data generation and validation
- Helper functions for assertions and comparisons
- Performance measurement utilities
- Fixture management for different model types

Author: Generated for TEZ-144 ONNX AutoProcessor Test Implementation
"""

import json
import os

# Import the modules we're testing
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import onnx
import pytest
from onnx import ValueInfoProto, helper

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from onnx_processor_types import TensorType


class MockONNXModel:
    """
    Mock ONNX model builder for testing various modality configurations.
    
    This class provides a flexible way to create mock ONNX models with
    specific input/output configurations for testing different scenarios.
    """
    
    def __init__(self, model_name: str = "test_model"):
        self.model_name = model_name
        self.inputs: list[ValueInfoProto] = []
        self.outputs: list[ValueInfoProto] = []
        self.metadata_props: list[tuple[str, str]] = []
        self.nodes: list[Any] = []
    
    def add_input(
        self, 
        name: str, 
        shape: list[int], 
        dtype: int | TensorType = TensorType.FLOAT32
    ) -> "MockONNXModel":
        """Add an input tensor to the mock model."""
        if isinstance(dtype, TensorType):
            dtype = int(dtype)
        
        input_tensor = helper.make_tensor_value_info(name, dtype, shape)
        self.inputs.append(input_tensor)
        return self
    
    def add_output(
        self, 
        name: str, 
        shape: list[int], 
        dtype: int | TensorType = TensorType.FLOAT32
    ) -> "MockONNXModel":
        """Add an output tensor to the mock model."""
        if isinstance(dtype, TensorType):
            dtype = int(dtype)
        
        output_tensor = helper.make_tensor_value_info(name, dtype, shape)
        self.outputs.append(output_tensor)
        return self
    
    def add_metadata(self, key: str, value: str) -> "MockONNXModel":
        """Add metadata to the mock model."""
        self.metadata_props.append((key, value))
        return self
    
    def build(self) -> onnx.ModelProto:
        """Build the ONNX model proto."""
        # Create a simple identity node if no nodes specified
        if not self.nodes and self.inputs and self.outputs:
            # Create identity node for first input->output
            identity_node = helper.make_node(
                'Identity',
                inputs=[self.inputs[0].name],
                outputs=[self.outputs[0].name],
                name='identity'
            )
            self.nodes.append(identity_node)
        
        # Create graph
        graph = helper.make_graph(
            nodes=self.nodes,
            name=f"{self.model_name}_graph",
            inputs=self.inputs,
            outputs=self.outputs
        )
        
        # Create model
        model = helper.make_model(graph, producer_name="test_producer")
        
        # Add metadata
        for key, value in self.metadata_props:
            model.metadata_props.add(key=key, value=value)
        
        # Set opset version
        model.opset_import[0].version = 17
        
        return model


def create_text_onnx_model(
    model_name: str = "bert_test",
    batch_size: int = 1,
    sequence_length: int = 128,
    vocab_size: int = 30522
) -> onnx.ModelProto:
    """Create a mock BERT-like text model."""
    builder = MockONNXModel(model_name)
    
    return (builder
        .add_input("input_ids", [batch_size, sequence_length], TensorType.INT64)
        .add_input("attention_mask", [batch_size, sequence_length], TensorType.INT64)
        .add_input("token_type_ids", [batch_size, sequence_length], TensorType.INT64)
        .add_output("last_hidden_state", [batch_size, sequence_length, 768], TensorType.FLOAT32)
        .add_output("pooler_output", [batch_size, 768], TensorType.FLOAT32)
        .add_metadata("model_type", "bert")
        .add_metadata("task", "feature-extraction")
        .add_metadata("modality", "text")
        .add_metadata("processor.batch_size", str(batch_size))
        .add_metadata("processor.sequence_length", str(sequence_length))
        .build())


def create_image_onnx_model(
    model_name: str = "vit_test",
    batch_size: int = 1,
    height: int = 224,
    width: int = 224,
    channels: int = 3
) -> onnx.ModelProto:
    """Create a mock ViT-like image model."""
    builder = MockONNXModel(model_name)
    
    return (builder
        .add_input("pixel_values", [batch_size, channels, height, width], TensorType.FLOAT32)
        .add_output("last_hidden_state", [batch_size, 197, 768], TensorType.FLOAT32)
        .add_output("pooler_output", [batch_size, 768], TensorType.FLOAT32)
        .add_metadata("model_type", "vit")
        .add_metadata("task", "image-classification")
        .add_metadata("modality", "image")
        .add_metadata("processor.batch_size", str(batch_size))
        .add_metadata("processor.height", str(height))
        .add_metadata("processor.width", str(width))
        .add_metadata("processor.channels", str(channels))
        .build())


def create_audio_onnx_model(
    model_name: str = "wav2vec2_test",
    batch_size: int = 1,
    sequence_length: int = 16000,
    feature_size: int = 1
) -> onnx.ModelProto:
    """Create a mock Wav2Vec2-like audio model."""
    builder = MockONNXModel(model_name)
    
    if feature_size == 1:
        # Raw waveform input
        input_shape = [batch_size, sequence_length]
        input_name = "input_values"
    else:
        # Feature-based input (e.g., Whisper)
        input_shape = [batch_size, feature_size, sequence_length]
        input_name = "input_features"
    
    return (builder
        .add_input(input_name, input_shape, TensorType.FLOAT32)
        .add_input("attention_mask", [batch_size, sequence_length], TensorType.INT64)
        .add_output("last_hidden_state", [batch_size, sequence_length // 320, 768], TensorType.FLOAT32)
        .add_metadata("model_type", "wav2vec2")
        .add_metadata("task", "automatic-speech-recognition")
        .add_metadata("modality", "audio")
        .add_metadata("processor.batch_size", str(batch_size))
        .add_metadata("processor.sequence_length", str(sequence_length))
        .add_metadata("processor.sampling_rate", "16000")
        .build())


def create_video_onnx_model(
    model_name: str = "videomae_test",
    batch_size: int = 1,
    num_frames: int = 16,
    height: int = 224,
    width: int = 224,
    channels: int = 3
) -> onnx.ModelProto:
    """Create a mock VideoMAE-like video model."""
    builder = MockONNXModel(model_name)
    
    return (builder
        .add_input("pixel_values", [batch_size, channels, num_frames, height, width], TensorType.FLOAT32)
        .add_output("last_hidden_state", [batch_size, 1568, 768], TensorType.FLOAT32)
        .add_metadata("model_type", "videomae")
        .add_metadata("task", "video-classification")
        .add_metadata("modality", "video")
        .add_metadata("processor.batch_size", str(batch_size))
        .add_metadata("processor.num_frames", str(num_frames))
        .add_metadata("processor.height", str(height))
        .add_metadata("processor.width", str(width))
        .add_metadata("processor.channels", str(channels))
        .build())


def create_multimodal_onnx_model(
    model_name: str = "clip_test",
    batch_size: int = 1,
    text_seq_length: int = 77,
    image_height: int = 224,
    image_width: int = 224,
    image_channels: int = 3
) -> onnx.ModelProto:
    """Create a mock CLIP-like multimodal model."""
    builder = MockONNXModel(model_name)
    
    return (builder
        .add_input("input_ids", [batch_size, text_seq_length], TensorType.INT64)
        .add_input("attention_mask", [batch_size, text_seq_length], TensorType.INT64)
        .add_input("pixel_values", [batch_size, image_channels, image_height, image_width], TensorType.FLOAT32)
        .add_output("text_embeds", [batch_size, 512], TensorType.FLOAT32)
        .add_output("image_embeds", [batch_size, 512], TensorType.FLOAT32)
        .add_output("logits_per_text", [batch_size, batch_size], TensorType.FLOAT32)
        .add_output("logits_per_image", [batch_size, batch_size], TensorType.FLOAT32)
        .add_metadata("model_type", "clip")
        .add_metadata("task", "zero-shot-image-classification")
        .add_metadata("modality", "multimodal")
        .add_metadata("processor.batch_size", str(batch_size))
        .add_metadata("processor.text_sequence_length", str(text_seq_length))
        .add_metadata("processor.image_height", str(image_height))
        .add_metadata("processor.image_width", str(image_width))
        .add_metadata("processor.image_channels", str(image_channels))
        .build())


def create_mock_base_processor(processor_type: str = "tokenizer") -> Mock:
    """Create mock HuggingFace base processor for testing."""
    mock_processor = Mock()
    
    if processor_type == "tokenizer":
        # Set the mock to appear as a PreTrainedTokenizerBase instance
        try:
            from transformers import PreTrainedTokenizerBase
            mock_processor.__class__ = type("MockTokenizer", (PreTrainedTokenizerBase,), {})
        except ImportError:
            # Fallback if transformers not available
            mock_processor.__class__.__name__ = "PreTrainedTokenizerBase"
            mock_processor.__class__.__bases__ = (object,)
        # Mock tokenizer attributes and methods
        mock_processor.vocab_size = 30522
        mock_processor.model_max_length = 512
        mock_processor.padding_side = "right"
        mock_processor.truncation_side = "right"
        mock_processor.pad_token_id = 0
        mock_processor.cls_token_id = 101
        mock_processor.sep_token_id = 102
        mock_processor.unk_token_id = 100
        mock_processor.mask_token_id = 103
        mock_processor.__class__.__name__ = "BertTokenizer"
        
        # Create a proper BatchEncoding-like object 
        from transformers import BatchEncoding
        
        class MockBatchEncoding(BatchEncoding):
            """Mock BatchEncoding that properly mimics the transformers BatchEncoding."""
            
            def __init__(self, data):
                # Initialize the parent BatchEncoding with our data
                super().__init__(data)
                # Ensure data attribute is properly set
                if not hasattr(self, 'data') or self.data is None:
                    self.data = data
        
        def mock_call(*args, **kwargs):
            max_length = kwargs.get('max_length', 128)
            # Create mock output with proper shape that mimics BatchEncoding
            data = {
                'input_ids': np.array([[101, 7592, 2088, 102] + [0] * (max_length - 4)]),
                'attention_mask': np.array([[1, 1, 1, 1] + [0] * (max_length - 4)]),
                'token_type_ids': np.array([[0] * max_length])
            }
            result = MockBatchEncoding(data)
            # Double-check the data attribute is set correctly
            if not hasattr(result, 'data') or not hasattr(result.data, 'items'):
                result.data = data
            return result
        
        # Set the call method correctly
        mock_processor.__call__ = mock_call
        mock_processor.side_effect = mock_call
        
    elif processor_type == "image_processor":
        # Set the mock to appear as a BaseImageProcessor instance
        try:
            from transformers.image_processing_utils import BaseImageProcessor
            mock_processor.__class__ = type("MockImageProcessor", (BaseImageProcessor,), {})
        except ImportError:
            # Fallback if transformers not available
            mock_processor.__class__.__name__ = "BaseImageProcessor"
            mock_processor.__class__.__bases__ = (object,)
        # Mock image processor
        mock_processor.image_mean = [0.485, 0.456, 0.406]
        mock_processor.image_std = [0.229, 0.224, 0.225]
        mock_processor.do_resize = True
        mock_processor.do_normalize = True
        mock_processor.__class__.__name__ = "ViTImageProcessor"
        
        # Mock processing output
        def mock_image_call(*args, **kwargs):
            return {
                'pixel_values': np.random.randn(1, 3, 224, 224).astype(np.float32)
            }
        
        mock_processor.__call__ = mock_image_call
        mock_processor.side_effect = mock_image_call
        
    elif processor_type == "feature_extractor":
        # Set the mock to appear as a FeatureExtractionMixin instance
        try:
            from transformers.feature_extraction_utils import FeatureExtractionMixin
            mock_processor.__class__ = type("MockFeatureExtractor", (FeatureExtractionMixin,), {})
        except ImportError:
            # Fallback if transformers not available
            mock_processor.__class__.__name__ = "FeatureExtractionMixin"
            mock_processor.__class__.__bases__ = (object,)
        # Mock audio feature extractor
        mock_processor.sampling_rate = 16000
        mock_processor.feature_size = 1
        mock_processor.return_attention_mask = True
        mock_processor.__class__.__name__ = "Wav2Vec2FeatureExtractor"
        
        # Mock processing output
        def mock_audio_call(*args, **kwargs):
            return {
                'input_values': np.random.randn(1, 16000).astype(np.float32),
                'attention_mask': np.ones((1, 16000), dtype=np.int64)
            }
        
        mock_processor.__call__ = mock_audio_call
        mock_processor.side_effect = mock_audio_call
        
    elif processor_type == "video_processor":
        # Set the mock to appear as a BaseImageProcessor instance (videos use image processing)
        try:
            from transformers.image_processing_utils import BaseImageProcessor
            mock_processor.__class__ = type("MockVideoProcessor", (BaseImageProcessor,), {})
        except ImportError:
            # Fallback if transformers not available
            mock_processor.__class__.__name__ = "BaseImageProcessor"
            mock_processor.__class__.__bases__ = (object,)
        # Mock video processor
        mock_processor.image_mean = [0.485, 0.456, 0.406]
        mock_processor.image_std = [0.229, 0.224, 0.225]
        mock_processor.do_resize = True
        mock_processor.do_normalize = True
        mock_processor.__class__.__name__ = "VideoMAEImageProcessor"
        
        # Mock processing output
        def mock_video_call(*args, **kwargs):
            return {
                'pixel_values': np.random.randn(1, 3, 16, 224, 224).astype(np.float32)
            }
        
        mock_processor.__call__ = mock_video_call
        mock_processor.side_effect = mock_video_call
        
    elif processor_type == "multimodal":
        # Set the mock to appear as a ProcessorMixin instance
        try:
            from transformers.processing_utils import ProcessorMixin
            mock_processor.__class__ = type("MockMultimodalProcessor", (ProcessorMixin,), {})
        except ImportError:
            # Fallback if transformers not available
            mock_processor.__class__.__name__ = "ProcessorMixin"
            mock_processor.__class__.__bases__ = (object,)
        # Mock multimodal processor (like CLIP)
        mock_processor.__class__.__name__ = "CLIPProcessor"
        
        # Add sub-processors
        mock_processor.tokenizer = create_mock_base_processor("tokenizer")
        mock_processor.image_processor = create_mock_base_processor("image_processor")
        
        # Mock processing output
        def mock_multimodal_call(*args, **kwargs):
            return {
                'input_ids': np.array([[49406, 320, 5655, 49407] + [0] * 73]),  # CLIP sequence length
                'attention_mask': np.array([[1, 1, 1, 1] + [0] * 73]),
                'pixel_values': np.random.randn(1, 3, 224, 224).astype(np.float32)
            }
        
        mock_processor.__call__ = mock_multimodal_call
        mock_processor.side_effect = mock_multimodal_call
    
    return mock_processor


def save_onnx_model_to_temp(model: onnx.ModelProto, filename: str | None = None) -> Path:
    """Save ONNX model to temporary file and return path."""
    if filename is None:
        filename = f"{model.graph.name}.onnx"
    
    temp_dir = Path(tempfile.mkdtemp())
    model_path = temp_dir / filename
    onnx.save(model, str(model_path))
    
    return model_path


def create_test_model_directory(
    model_type: str = "bert",
    include_metadata: bool = True,
    include_configs: bool = True
) -> Path:
    """Create a complete test model directory with all required files."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create ONNX model
    if model_type == "bert":
        model = create_text_onnx_model()
    elif model_type == "vit":
        model = create_image_onnx_model()
    elif model_type == "wav2vec2":
        model = create_audio_onnx_model()
    elif model_type == "videomae":
        model = create_video_onnx_model()
    elif model_type == "clip":
        model = create_multimodal_onnx_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Save ONNX model
    model_path = temp_dir / "model.onnx"
    onnx.save(model, str(model_path))
    
    # Create companion metadata JSON if requested
    if include_metadata:
        metadata = {
            "model_name": f"{model_type}_test",
            "model_type": model_type,
            "task": _get_default_task(model_type),
            "modalities": _get_default_modalities(model_type),
            "onnx_opset_version": 17,
            "metadata_source": "json"
        }
        
        metadata_path = temp_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Create HuggingFace config files if requested
    if include_configs:
        _create_hf_config_files(temp_dir, model_type)
    
    return temp_dir


def _get_default_task(model_type: str) -> str:
    """Get default task for model type."""
    task_mapping = {
        "bert": "feature-extraction",
        "vit": "image-classification",
        "wav2vec2": "automatic-speech-recognition",
        "videomae": "video-classification",
        "clip": "zero-shot-image-classification"
    }
    return task_mapping.get(model_type, "unknown")


def _get_default_modalities(model_type: str) -> dict[str, Any]:
    """Get default modality configuration for model type."""
    modality_configs = {
        "bert": {
            "text": {
                "type": "text",
                "batch_size": 1,
                "sequence_length": 128,
                "tensors": [
                    {"name": "input_ids", "shape": [1, 128], "dtype": 7},
                    {"name": "attention_mask", "shape": [1, 128], "dtype": 7}
                ]
            }
        },
        "vit": {
            "image": {
                "type": "image",
                "batch_size": 1,
                "height": 224,
                "width": 224,
                "num_channels": 3,
                "tensors": [
                    {"name": "pixel_values", "shape": [1, 3, 224, 224], "dtype": 1}
                ]
            }
        },
        "clip": {
            "text": {
                "type": "text",
                "batch_size": 1,
                "sequence_length": 77,
                "tensors": [
                    {"name": "input_ids", "shape": [1, 77], "dtype": 7}
                ]
            },
            "image": {
                "type": "image",
                "batch_size": 1,
                "height": 224,
                "width": 224,
                "num_channels": 3,
                "tensors": [
                    {"name": "pixel_values", "shape": [1, 3, 224, 224], "dtype": 1}
                ]
            }
        }
    }
    return modality_configs.get(model_type, {})


def _create_hf_config_files(model_dir: Path, model_type: str) -> None:
    """Create HuggingFace configuration files."""
    if model_type in ["bert", "clip"]:
        # Create tokenizer config
        tokenizer_config = {
            "model_max_length": 512 if model_type == "bert" else 77,
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
            "tokenizer_class": "BertTokenizer" if model_type == "bert" else "CLIPTokenizer"
        }
        
        with open(model_dir / "tokenizer_config.json", 'w') as f:
            json.dump(tokenizer_config, f, indent=2)
        
        # Create basic vocab file
        vocab_lines = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "hello", "world", "test", "example"
        ]
        with open(model_dir / "vocab.txt", 'w') as f:
            f.write('\n'.join(vocab_lines))
    
    if model_type in ["vit", "clip"]:
        # Create image processor config
        processor_config = {
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "size": {"height": 224, "width": 224},
            "do_resize": True,
            "do_normalize": True,
            "processor_class": "ViTImageProcessor" if model_type == "vit" else "CLIPImageProcessor"
        }
        
        with open(model_dir / "preprocessor_config.json", 'w') as f:
            json.dump(processor_config, f, indent=2)
    
    # Create model config
    model_config = {
        "model_type": model_type,
        "architectures": [f"{model_type.upper()}Model"],
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12
    }
    
    if model_type == "clip":
        model_config.update({
            "text_config": {"vocab_size": 49408, "max_position_embeddings": 77},
            "vision_config": {"image_size": 224, "patch_size": 32}
        })
    
    with open(model_dir / "config.json", 'w') as f:
        json.dump(model_config, f, indent=2)


def assert_tensor_dict_valid(tensor_dict: dict[str, np.ndarray]) -> None:
    """Assert that a tensor dictionary is valid."""
    assert isinstance(tensor_dict, dict), "Result must be a dictionary"
    assert len(tensor_dict) > 0, "Result must contain at least one tensor"
    
    for name, tensor in tensor_dict.items():
        assert isinstance(name, str), f"Tensor name must be string, got {type(name)}"
        assert isinstance(tensor, np.ndarray), f"Tensor {name} must be numpy array, got {type(tensor)}"
        assert tensor.size > 0, f"Tensor {name} must not be empty"


def assert_fixed_shape(tensor_dict: dict[str, np.ndarray], expected_shapes: dict[str, list[int]]) -> None:
    """Assert that tensors have expected fixed shapes."""
    for name, expected_shape in expected_shapes.items():
        assert name in tensor_dict, f"Expected tensor {name} not found in output"
        actual_shape = list(tensor_dict[name].shape)
        assert actual_shape == expected_shape, f"Tensor {name} shape mismatch: expected {expected_shape}, got {actual_shape}"


def measure_processing_time(processor_func, *args, **kwargs) -> tuple[Any, float]:
    """Measure processing time and return result and elapsed time."""
    import time
    
    start_time = time.perf_counter()
    result = processor_func(*args, **kwargs)
    end_time = time.perf_counter()
    
    return result, end_time - start_time


def create_corrupted_onnx_model() -> bytes:
    """Create corrupted ONNX model data for error testing."""
    return b"INVALID_ONNX_DATA_FOR_TESTING"


def create_performance_test_data(
    modality: str = "text",
    num_samples: int = 100
) -> list[str] | list[np.ndarray]:
    """Create test data for performance benchmarking."""
    if modality == "text":
        # Create text samples of varying lengths
        texts = []
        for _i in range(num_samples):
            length = np.random.randint(5, 100)
            text = " ".join([f"word{j}" for j in range(length)])
            texts.append(text)
        return texts
    
    elif modality == "image":
        # Create random image arrays
        images = []
        for _i in range(num_samples):
            # Random image (224x224x3)
            image = np.random.rand(224, 224, 3).astype(np.float32)
            images.append(image)
        return images
    
    elif modality == "audio":
        # Create random audio waveforms
        audio_samples = []
        for _i in range(num_samples):
            # Random waveform (16000 samples = 1 second at 16kHz)
            audio = np.random.randn(16000).astype(np.float32)
            audio_samples.append(audio)
        return audio_samples
    
    else:
        raise ValueError(f"Unsupported modality for performance test: {modality}")


class PerformanceBenchmark:
    """Helper class for performance benchmarking."""
    
    def __init__(self, name: str):
        self.name = name
        self.times = []
        self.memory_usage = []
    
    def __enter__(self):
        import time

        import psutil
        self.start_time = time.perf_counter()
        self.start_memory = psutil.Process().memory_info().rss
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        import psutil
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss
        
        elapsed_time = end_time - self.start_time
        memory_delta = end_memory - self.start_memory
        
        self.times.append(elapsed_time)
        self.memory_usage.append(memory_delta)
    
    def get_stats(self) -> dict[str, float]:
        """Get performance statistics."""
        if not self.times:
            return {}
        
        return {
            "avg_time": np.mean(self.times),
            "min_time": np.min(self.times),
            "max_time": np.max(self.times),
            "std_time": np.std(self.times),
            "avg_memory": np.mean(self.memory_usage),
            "max_memory": np.max(self.memory_usage) if self.memory_usage else 0
        }


# Test fixtures for pytest
@pytest.fixture
def bert_onnx_model():
    """Fixture providing BERT ONNX model."""
    return create_text_onnx_model("bert_fixture")


@pytest.fixture
def vit_onnx_model():
    """Fixture providing ViT ONNX model."""
    return create_image_onnx_model("vit_fixture")


@pytest.fixture
def wav2vec2_onnx_model():
    """Fixture providing Wav2Vec2 ONNX model."""
    return create_audio_onnx_model("wav2vec2_fixture")


@pytest.fixture
def videomae_onnx_model():
    """Fixture providing VideoMAE ONNX model."""
    return create_video_onnx_model("videomae_fixture")


@pytest.fixture
def clip_onnx_model():
    """Fixture providing CLIP ONNX model."""
    return create_multimodal_onnx_model("clip_fixture")


@pytest.fixture
def mock_tokenizer():
    """Fixture providing mock tokenizer."""
    return create_mock_base_processor("tokenizer")


@pytest.fixture
def mock_image_processor():
    """Fixture providing mock image processor."""
    return create_mock_base_processor("image_processor")


@pytest.fixture
def mock_feature_extractor():
    """Fixture providing mock feature extractor."""
    return create_mock_base_processor("feature_extractor")


@pytest.fixture
def mock_multimodal_processor():
    """Fixture providing mock multimodal processor."""
    return create_mock_base_processor("multimodal")


@pytest.fixture
def temp_model_directory():
    """Fixture providing temporary model directory."""
    model_dir = create_test_model_directory("bert")
    yield model_dir
    # Cleanup handled by tempfile


@pytest.fixture
def performance_text_data():
    """Fixture providing text data for performance testing."""
    return create_performance_test_data("text", 50)


@pytest.fixture
def performance_image_data():
    """Fixture providing image data for performance testing."""
    return create_performance_test_data("image", 20)


@pytest.fixture
def corrupted_onnx_data():
    """Fixture providing corrupted ONNX data."""
    return create_corrupted_onnx_model()