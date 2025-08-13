"""
Common test helper functions and utilities.

This module provides reusable helper functions to reduce duplication
across test files and improve test maintainability.
"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import onnx
import pytest
from onnx import helper

from tests.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEQUENCE_LENGTH,
    ONNX_OPSET_VERSION,
    ONNX_PRODUCER_NAME,
)


def create_mock_onnx_model(
    name: str = "test_model",
    inputs: list[tuple[str, list[int], int]] | None = None,
    outputs: list[tuple[str, list[int], int]] | None = None,
    metadata: dict[str, str] | None = None,
) -> onnx.ModelProto:
    """
    Create a mock ONNX model for testing.

    Args:
        name: Model name
        inputs: List of (name, shape, dtype) tuples for inputs
        outputs: List of (name, shape, dtype) tuples for outputs
        metadata: Dictionary of metadata key-value pairs

    Returns:
        ONNX ModelProto
    """
    if inputs is None:
        inputs = [
            (
                "input",
                [DEFAULT_BATCH_SIZE, DEFAULT_SEQUENCE_LENGTH],
                onnx.TensorProto.FLOAT,
            )
        ]
    if outputs is None:
        outputs = [
            (
                "output",
                [DEFAULT_BATCH_SIZE, DEFAULT_SEQUENCE_LENGTH],
                onnx.TensorProto.FLOAT,
            )
        ]

    # Create input/output value infos
    input_value_infos = [
        helper.make_tensor_value_info(input_name, dtype, shape)
        for input_name, shape, dtype in inputs
    ]
    output_value_infos = [
        helper.make_tensor_value_info(output_name, dtype, shape)
        for output_name, shape, dtype in outputs
    ]

    # Create a simple identity node if we have inputs and outputs
    nodes = []
    if inputs and outputs:
        # Create identity node from first input to first output
        identity_node = helper.make_node(
            "Identity", inputs=[inputs[0][0]], outputs=[outputs[0][0]], name="identity"
        )
        nodes.append(identity_node)

    # Create graph
    graph = helper.make_graph(
        nodes=nodes,
        name=f"{name}_graph",
        inputs=input_value_infos,
        outputs=output_value_infos,
    )

    # Create model
    model = helper.make_model(graph, producer_name=ONNX_PRODUCER_NAME)

    # Add metadata
    if metadata:
        for key, value in metadata.items():
            model.metadata_props.add(key=key, value=value)

    # Set opset version
    model.opset_import[0].version = ONNX_OPSET_VERSION

    return model


def save_model_to_temp(model: onnx.ModelProto, filename: str | None = None) -> Path:
    """
    Save ONNX model to temporary file and return path.

    Args:
        model: ONNX model to save
        filename: Optional filename (will use model name if not provided)

    Returns:
        Path to saved model file
    """
    if filename is None:
        filename = f"{model.graph.name}.onnx"

    temp_dir = Path(tempfile.mkdtemp())
    model_path = temp_dir / filename
    onnx.save(model, str(model_path))

    return model_path


def assert_valid_tensor_dict(
    tensor_dict: dict[str, np.ndarray], min_tensors: int = 1
) -> None:
    """
    Assert that a tensor dictionary is valid.

    Args:
        tensor_dict: Dictionary of tensors to validate
        min_tensors: Minimum number of tensors expected
    """
    assert isinstance(tensor_dict, dict), f"Expected dict, got {type(tensor_dict)}"
    assert len(tensor_dict) >= min_tensors, (
        f"Expected at least {min_tensors} tensors, got {len(tensor_dict)}"
    )

    for name, tensor in tensor_dict.items():
        assert isinstance(name, str), f"Tensor name must be string, got {type(name)}"
        assert isinstance(tensor, np.ndarray), (
            f"Tensor {name} must be numpy array, got {type(tensor)}"
        )
        assert tensor.size > 0, f"Tensor {name} must not be empty"


def assert_shapes_match(
    tensor_dict: dict[str, np.ndarray], expected_shapes: dict[str, list[int]]
) -> None:
    """
    Assert that tensor shapes match expected shapes.

    Args:
        tensor_dict: Dictionary of tensors
        expected_shapes: Dictionary of expected shapes by tensor name
    """
    for name, expected_shape in expected_shapes.items():
        assert name in tensor_dict, f"Expected tensor {name} not found in output"
        actual_shape = list(tensor_dict[name].shape)
        assert actual_shape == expected_shape, (
            f"Shape mismatch for {name}: expected {expected_shape}, got {actual_shape}"
        )


def create_mock_processor(processor_type: str = "base") -> Mock:
    """
    Create a mock processor for testing.

    Args:
        processor_type: Type of processor to mock

    Returns:
        Mock processor object
    """
    mock_processor = Mock()

    # Common attributes
    mock_processor.__class__.__name__ = f"Mock{processor_type.title()}Processor"

    # Common methods
    mock_processor.__call__ = Mock(
        return_value={"output": np.random.randn(1, 128).astype(np.float32)}
    )
    mock_processor.preprocess = Mock(
        return_value={"input": np.random.randn(1, 128).astype(np.float32)}
    )

    return mock_processor


@contextlib.contextmanager
def temp_directory():
    """Context manager for temporary directory that cleans up automatically."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        yield temp_dir
    finally:
        # Cleanup is handled by tempfile.mkdtemp automatically
        pass


def skip_if_no_transformers():
    """Skip test if transformers library is not available."""
    try:
        import transformers  # noqa: F401

        return pytest.mark.skipif(False, reason="")
    except ImportError:
        return pytest.mark.skip(reason="transformers library not available")


def skip_if_no_torch():
    """Skip test if torch library is not available."""
    try:
        import torch  # noqa: F401

        return pytest.mark.skipif(False, reason="")
    except ImportError:
        return pytest.mark.skip(reason="torch library not available")


class PerformanceTimer:
    """Simple performance timer for testing."""

    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def __enter__(self) -> PerformanceTimer:
        import time

        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        import time

        self.end_time = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return self.end_time - self.start_time


def parametrize_model_types():
    """Pytest parametrize decorator for common model types."""
    return pytest.mark.parametrize(
        "model_type,expected_modality",
        [
            ("bert", "text"),
            ("vit", "image"),
            ("wav2vec2", "audio"),
            ("videomae", "video"),
            ("clip", "multimodal"),
        ],
    )


def create_test_data(data_type: str = "text", count: int = 10) -> list[Any]:
    """
    Create test data for different modalities.

    Args:
        data_type: Type of data to create (text, image, audio)
        count: Number of items to create

    Returns:
        List of test data items
    """
    if data_type == "text":
        return [
            f"Test sentence number {i} with varying length content."
            for i in range(count)
        ]
    elif data_type == "image":
        return [np.random.rand(224, 224, 3).astype(np.float32) for _ in range(count)]
    elif data_type == "audio":
        return [np.random.randn(16000).astype(np.float32) for _ in range(count)]
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def assert_no_duplicate_names(items: list[Any], name_attr: str = "name") -> None:
    """
    Assert that items don't have duplicate names.

    Args:
        items: List of items to check
        name_attr: Attribute name to check for uniqueness
    """
    names = [getattr(item, name_attr) for item in items]
    duplicates = [name for name in set(names) if names.count(name) > 1]
    assert not duplicates, f"Found duplicate names: {duplicates}"


def assert_within_tolerance(
    actual: float, expected: float, tolerance: float = 0.1
) -> None:
    """
    Assert that actual value is within tolerance of expected value.

    Args:
        actual: Actual value
        expected: Expected value
        tolerance: Tolerance as a fraction (0.1 = 10%)
    """
    lower_bound = expected * (1 - tolerance)
    upper_bound = expected * (1 + tolerance)
    assert lower_bound <= actual <= upper_bound, (
        f"Value {actual} not within {tolerance * 100}% of {expected}"
    )


# Common pytest fixtures
@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory."""
    with temp_directory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def mock_model():
    """Fixture providing a mock ONNX model."""
    return create_mock_onnx_model("test_model")


@pytest.fixture
def performance_timer():
    """Fixture providing a performance timer."""
    return PerformanceTimer()
