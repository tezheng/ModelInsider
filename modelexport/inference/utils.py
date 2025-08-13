"""
Utility functions for ONNX model inference with Optimum.

This module provides helper functions for loading models, preprocessors,
and creating inference pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)

logger = logging.getLogger(__name__)


def detect_model_task(model_path: str | Path) -> str:
    """
    Detect the task for a model based on its configuration.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Detected task name
        
    Raises:
        ValueError: If task cannot be detected
    """
    from .auto_model_loader import AutoModelForONNX
    
    model_path = Path(model_path)
    config = AutoConfig.from_pretrained(model_path)
    return AutoModelForONNX._detect_task(config, model_path)


def get_ort_model_class(task: str) -> type:
    """
    Get the appropriate ORTModel class for a given task.
    
    Args:
        task: Task name
        
    Returns:
        ORTModel class
        
    Raises:
        ImportError: If Optimum is not installed
        ValueError: If task is not supported
    """
    from .auto_model_loader import AutoModelForONNX
    
    class_name = AutoModelForONNX._get_ort_model_class(task)
    
    try:
        import optimum.onnxruntime
        
        if class_name == "ORTModel":
            from optimum.onnxruntime import ORTModel
            return ORTModel
        else:
            return getattr(optimum.onnxruntime, class_name)
    except ImportError as e:
        raise ImportError(
            "Optimum is not installed. Please install with: "
            "pip install optimum[onnxruntime]"
        ) from e
    except AttributeError as e:
        raise ValueError(f"ORTModel class {class_name} not found in Optimum") from e


def load_preprocessor(
    model_path: str | Path,
    preprocessor_type: str | None = None
) -> Any:
    """
    Load the appropriate preprocessor for a model.
    
    Automatically detects and loads the correct preprocessor type
    (tokenizer, processor, image processor, or feature extractor).
    
    Args:
        model_path: Path to the model directory
        preprocessor_type: Optional preprocessor type to force
                          ("tokenizer", "processor", "image_processor", "feature_extractor")
    
    Returns:
        Loaded preprocessor
        
    Raises:
        ValueError: If no suitable preprocessor is found
    """
    model_path = Path(model_path)
    
    if preprocessor_type:
        # Use specified type
        if preprocessor_type == "tokenizer":
            return AutoTokenizer.from_pretrained(model_path)
        elif preprocessor_type == "processor":
            return AutoProcessor.from_pretrained(model_path)
        elif preprocessor_type == "image_processor":
            return AutoImageProcessor.from_pretrained(model_path)
        elif preprocessor_type == "feature_extractor":
            return AutoFeatureExtractor.from_pretrained(model_path)
        else:
            raise ValueError(f"Unknown preprocessor type: {preprocessor_type}")
    
    # Auto-detect preprocessor type
    # Try processor first (for multimodal models)
    try:
        logger.debug("Trying to load processor...")
        return AutoProcessor.from_pretrained(model_path)
    except Exception:
        pass
    
    # Try tokenizer (for text models)
    try:
        logger.debug("Trying to load tokenizer...")
        return AutoTokenizer.from_pretrained(model_path)
    except Exception:
        pass
    
    # Try image processor (for vision models)
    try:
        logger.debug("Trying to load image processor...")
        return AutoImageProcessor.from_pretrained(model_path)
    except Exception:
        pass
    
    # Try feature extractor (for audio models)
    try:
        logger.debug("Trying to load feature extractor...")
        return AutoFeatureExtractor.from_pretrained(model_path)
    except Exception:
        pass
    
    raise ValueError(
        f"Could not load any preprocessor from {model_path}. "
        "Make sure the model was exported with preprocessor files."
    )


def create_inference_pipeline(
    model_path: str | Path,
    task: str | None = None,
    device: int = -1,
    batch_size: int = 1,
    **kwargs
) -> Any:
    """
    Create a HuggingFace pipeline for inference with an ONNX model.
    
    This is a convenience function that loads both the model and preprocessor
    and creates a pipeline for easy inference.
    
    Args:
        model_path: Path to the exported ONNX model
        task: Optional task specification (auto-detected if None)
        device: Device to run on (-1 for CPU, 0+ for GPU)
        batch_size: Batch size for inference
        **kwargs: Additional arguments for pipeline
        
    Returns:
        HuggingFace pipeline configured for ONNX inference
        
    Example:
        >>> pipe = create_inference_pipeline("path/to/model", task="text-classification")
        >>> results = pipe(["Text 1", "Text 2"])
    """
    from .auto_model_loader import AutoModelForONNX
    
    model_path = Path(model_path)
    
    # Load model with auto-detection
    if device >= 0:
        # Use CUDA provider for GPU
        model = AutoModelForONNX.from_pretrained(
            model_path,
            task=task,
            provider="CUDAExecutionProvider"
        )
    else:
        # Use CPU provider
        model = AutoModelForONNX.from_pretrained(
            model_path,
            task=task,
            provider="CPUExecutionProvider"
        )
    
    # Get the actual task used
    task = model.task
    
    # Load preprocessor
    preprocessor = load_preprocessor(model_path)
    
    # Create pipeline
    logger.info(f"Creating pipeline for task: {task}")
    
    # Build pipeline kwargs
    pipe_kwargs = {
        "task": task,
        "model": model,
        "tokenizer": preprocessor if hasattr(preprocessor, "tokenize") else None,
        "feature_extractor": preprocessor if not hasattr(preprocessor, "tokenize") else None,
        "device": device if device >= 0 else -1,
        **kwargs
    }
    
    # Handle different preprocessor types
    if hasattr(preprocessor, "image_processor"):
        # It's a processor with image processor
        pipe_kwargs["image_processor"] = preprocessor.image_processor
    
    # Remove None values
    pipe_kwargs = {k: v for k, v in pipe_kwargs.items() if v is not None}
    
    # Create and return pipeline
    pipe = pipeline(**pipe_kwargs)
    
    # Set batch size if specified
    if batch_size > 1:
        pipe.batch_size = batch_size
    
    return pipe


def benchmark_inference(
    model: Any,
    inputs: dict,
    num_runs: int = 100,
    warmup_runs: int = 10
) -> dict[str, float]:
    """
    Benchmark inference performance of a model.
    
    Args:
        model: The model to benchmark
        inputs: Input dictionary for the model
        num_runs: Number of inference runs for benchmarking
        warmup_runs: Number of warmup runs before benchmarking
        
    Returns:
        Dictionary with performance metrics:
        - mean_latency: Average inference time in ms
        - std_latency: Standard deviation of inference time
        - min_latency: Minimum inference time
        - max_latency: Maximum inference time
        - throughput: Inferences per second
    """
    import time

    import numpy as np
    
    # Warmup runs
    logger.info(f"Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        _ = model(**inputs)
    
    # Benchmark runs
    logger.info(f"Running {num_runs} benchmark iterations...")
    latencies = []
    
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(**inputs)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = np.array(latencies)
    
    return {
        "mean_latency": float(np.mean(latencies)),
        "std_latency": float(np.std(latencies)),
        "min_latency": float(np.min(latencies)),
        "max_latency": float(np.max(latencies)),
        "throughput": 1000.0 / float(np.mean(latencies)),  # inferences per second
    }


def compare_with_pytorch(
    onnx_model_path: str | Path,
    pytorch_model_name: str,
    test_inputs: list[str],
    task: str | None = None
) -> dict:
    """
    Compare ONNX model performance with PyTorch model.
    
    Args:
        onnx_model_path: Path to ONNX model
        pytorch_model_name: HuggingFace model name for PyTorch version
        test_inputs: List of test input strings
        task: Optional task specification
        
    Returns:
        Dictionary with comparison results
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    # Load ONNX model
    from .auto_model_loader import AutoModelForONNX
    
    onnx_model = AutoModelForONNX.from_pretrained(onnx_model_path, task=task)
    tokenizer = AutoTokenizer.from_pretrained(onnx_model_path)
    
    # Load PyTorch model
    pytorch_model = AutoModel.from_pretrained(pytorch_model_name)
    pytorch_model.eval()
    
    # Prepare inputs
    inputs = tokenizer(test_inputs[0], return_tensors="pt")
    
    # Benchmark ONNX
    logger.info("Benchmarking ONNX model...")
    onnx_metrics = benchmark_inference(onnx_model, inputs)
    
    # Benchmark PyTorch
    logger.info("Benchmarking PyTorch model...")
    with torch.no_grad():
        pytorch_metrics = benchmark_inference(pytorch_model, inputs)
    
    # Calculate speedup
    speedup = pytorch_metrics["mean_latency"] / onnx_metrics["mean_latency"]
    
    return {
        "onnx": onnx_metrics,
        "pytorch": pytorch_metrics,
        "speedup": speedup,
        "speedup_percentage": (speedup - 1) * 100,
    }