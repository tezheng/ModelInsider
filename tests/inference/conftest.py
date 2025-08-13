"""
Pytest configuration and shared fixtures for ONNXAutoProcessor tests.

This module provides shared fixtures, test configuration, and setup/teardown
functionality for the comprehensive test suite.

Author: Generated for TEZ-144 ONNX AutoProcessor Test Implementation
"""

import os
import sys
import tempfile
import warnings
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

# Add src directory to path for imports
TEST_DIR = Path(__file__).parent
SRC_DIR = TEST_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

# Import test utilities
from test_utils import (
    create_audio_onnx_model,
    create_image_onnx_model,
    create_mock_base_processor,
    create_multimodal_onnx_model,
    create_performance_test_data,
    create_test_model_directory,
    create_text_onnx_model,
    create_video_onnx_model,
)

# Suppress known warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="onnx")


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    # Configure test output
    config.option.verbose = max(config.option.verbose, 1)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and organize tests."""
    for item in items:
        # Add markers based on test names and patterns
        if "smoke" in item.name or item.name.startswith("test_import"):
            item.add_marker(pytest.mark.smoke)
        
        if "sanity" in item.name or "all_five_processor" in item.name:
            item.add_marker(pytest.mark.sanity)
        
        if "unit" in item.name or item.name.startswith("test_extract_") or "detection" in item.name:
            item.add_marker(pytest.mark.unit)
        
        if "integration" in item.name or "end_to_end" in item.name or "e2e" in item.name:
            item.add_marker(pytest.mark.integration)
        
        if "performance" in item.name or "speed" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance)
        
        if "multimodal" in item.name or "clip" in item.name.lower():
            item.add_marker(pytest.mark.multimodal)
        
        if "slow" in item.name or "memory" in item.name or "concurrent" in item.name:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def test_models_dir() -> Generator[Path, None, None]:
    """Session-scoped fixture providing temporary directory for test models."""
    with tempfile.TemporaryDirectory(prefix="onnx_test_models_") as temp_dir:
        models_dir = Path(temp_dir)
        
        # Pre-create common test models to avoid recreation in each test
        bert_model = create_text_onnx_model("session_bert")
        vit_model = create_image_onnx_model("session_vit")
        clip_model = create_multimodal_onnx_model("session_clip")
        
        # Save models
        (models_dir / "bert.onnx").write_bytes(bert_model.SerializeToString())
        (models_dir / "vit.onnx").write_bytes(vit_model.SerializeToString())
        (models_dir / "clip.onnx").write_bytes(clip_model.SerializeToString())
        
        yield models_dir


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Function-scoped fixture providing clean temporary directory."""
    with tempfile.TemporaryDirectory(prefix="onnx_test_temp_") as temp_dir:
        yield Path(temp_dir)


# Model fixtures - these create fresh ONNX models for each test
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


# Mock processor fixtures
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
def mock_video_processor():
    """Fixture providing mock video processor."""
    return create_mock_base_processor("video_processor")


@pytest.fixture
def mock_multimodal_processor():
    """Fixture providing mock multimodal processor."""
    return create_mock_base_processor("multimodal")


# Test data fixtures
@pytest.fixture
def sample_texts():
    """Fixture providing sample text data for testing."""
    return [
        "Hello world",
        "This is a longer sentence for testing tokenization with various lengths",
        "Short text",
        "",  # Empty string
        "Text with special characters: !@#$%^&*()",
        "Multi-sentence text. This has multiple sentences. Testing various scenarios.",
        "Numbers and text: 123 456 789",
        "Mixed case Text WITH various CASE patterns"
    ]


@pytest.fixture
def sample_images():
    """Fixture providing sample image data for testing."""
    images = []
    
    # Various image sizes and formats
    sizes = [(224, 224, 3), (256, 256, 3), (299, 299, 3), (224, 224, 1)]
    
    for height, width, channels in sizes:
        # Create random image data
        image = np.random.rand(height, width, channels).astype(np.float32)
        images.append(image)
    
    return images


@pytest.fixture
def sample_audio():
    """Fixture providing sample audio data for testing."""
    audio_samples = []
    
    # Various audio lengths (in samples at 16kHz)
    lengths = [8000, 16000, 24000, 32000]  # 0.5s, 1s, 1.5s, 2s
    
    for length in lengths:
        # Create random waveform
        audio = np.random.randn(length).astype(np.float32)
        # Normalize to [-1, 1]
        audio = audio / np.max(np.abs(audio))
        audio_samples.append(audio)
    
    return audio_samples


@pytest.fixture
def sample_video():
    """Fixture providing sample video data for testing."""
    # Create sample video as list of frames
    num_frames = 16
    height, width, channels = 224, 224, 3
    
    frames = []
    for i in range(num_frames):
        # Create frame with slight variation
        frame = np.random.rand(height, width, channels).astype(np.float32)
        # Add temporal pattern
        frame = frame * (1.0 + 0.1 * np.sin(i * 0.5))
        frames.append(frame)
    
    return frames


# Performance test data fixtures
@pytest.fixture
def performance_text_data():
    """Fixture providing text data for performance testing."""
    return create_performance_test_data("text", 50)


@pytest.fixture
def performance_image_data():
    """Fixture providing image data for performance testing."""
    return create_performance_test_data("image", 20)


@pytest.fixture
def performance_audio_data():
    """Fixture providing audio data for performance testing."""
    return create_performance_test_data("audio", 30)


# Test model directory fixtures
@pytest.fixture
def bert_model_directory(temp_dir):
    """Fixture providing complete BERT model directory."""
    return create_test_model_directory("bert", include_metadata=True, include_configs=True)


@pytest.fixture
def vit_model_directory(temp_dir):
    """Fixture providing complete ViT model directory."""
    return create_test_model_directory("vit", include_metadata=True, include_configs=True)


@pytest.fixture
def clip_model_directory(temp_dir):
    """Fixture providing complete CLIP model directory."""
    return create_test_model_directory("clip", include_metadata=True, include_configs=True)


# Utility fixtures
@pytest.fixture
def assert_valid_tensor_dict():
    """Fixture providing tensor dictionary validation function."""
    def _assert_valid(tensor_dict, expected_keys=None, expected_shapes=None):
        assert isinstance(tensor_dict, dict), "Result must be a dictionary"
        assert len(tensor_dict) > 0, "Result must contain at least one tensor"
        
        for name, tensor in tensor_dict.items():
            assert isinstance(name, str), f"Tensor name must be string, got {type(name)}"
            assert isinstance(tensor, np.ndarray), f"Tensor {name} must be numpy array, got {type(tensor)}"
            assert tensor.size > 0, f"Tensor {name} must not be empty"
        
        if expected_keys:
            for key in expected_keys:
                assert key in tensor_dict, f"Expected tensor {key} not found"
        
        if expected_shapes:
            for name, expected_shape in expected_shapes.items():
                if name in tensor_dict:
                    actual_shape = list(tensor_dict[name].shape)
                    assert actual_shape == expected_shape, f"Shape mismatch for {name}: expected {expected_shape}, got {actual_shape}"
    
    return _assert_valid


@pytest.fixture
def measure_time():
    """Fixture providing time measurement utility."""
    import time
    
    def _measure(func, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    
    return _measure


# Test environment setup/teardown
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically run for each test to set up clean environment."""
    # Set environment variables for testing
    original_env = os.environ.copy()
    
    # Disable various caches and optimizations for consistent testing
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/test_transformers_cache'
    os.environ['HF_HOME'] = '/tmp/test_hf_home'
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Skip markers for CI/CD
def pytest_runtest_setup(item):
    """Set up individual tests with conditional skipping."""
    # Skip GPU tests if no GPU available
    if item.get_closest_marker("requires_gpu"):
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("PyTorch not available for GPU testing")
    
    # Skip model tests if models not available (for CI)
    if item.get_closest_marker("requires_models"):
        # Could implement model availability check here
        pass


# Custom assertions
def assert_tensor_shapes_match(tensor_dict, expected_shapes):
    """Custom assertion for tensor shape validation."""
    for name, expected_shape in expected_shapes.items():
        assert name in tensor_dict, f"Expected tensor {name} not found in output"
        actual_shape = list(tensor_dict[name].shape)
        assert actual_shape == expected_shape, f"Tensor {name} shape mismatch: expected {expected_shape}, got {actual_shape}"


def assert_processing_time_acceptable(elapsed_time, target_time, operation="processing"):
    """Custom assertion for performance validation."""
    assert elapsed_time < target_time, f"{operation} took {elapsed_time:.4f}s, expected <{target_time:.4f}s"


# Register custom assertions with pytest
pytest.assert_tensor_shapes_match = assert_tensor_shapes_match
pytest.assert_processing_time_acceptable = assert_processing_time_acceptable