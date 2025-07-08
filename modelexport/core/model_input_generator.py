#!/usr/bin/env python3
"""
Universal Model Input Generator

Simple wrapper around Optimum's TasksManager to generate input data
for any HuggingFace model. No reinventing the wheel - just using what exists.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoConfig

logger = logging.getLogger(__name__)


def get_export_config_from_model_path(
    model_name_or_path: str,
    exporter: str = "onnx",
    task: str | None = None,
    library_name: str | None = None,
    **config_kwargs,
):
    """
    Get export config from HuggingFace model path using Optimum's TasksManager.

    This is the correct way to use Optimum - no wheel reinvention!

    Args:
        model_name_or_path: HuggingFace model name or local path
        exporter: Export backend ("onnx", "tflite", etc.)
        task: Task name (auto-detects if None)
        library_name: Library name ("transformers", "diffusers", etc.)
        **config_kwargs: Additional config constructor arguments

    Returns:
        Export config object that can generate dummy inputs

    Example:
        >>> config = get_export_config_from_model_path("bert-base-uncased")
        >>> inputs = config.generate_dummy_inputs()
    """
    try:
        from optimum.exporters.tasks import TasksManager
    except ImportError as e:
        msg = "optimum is required. Install with: pip install optimum"
        raise ImportError(msg) from e

    # Load config to get model_type - this is always available
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_type = config.model_type

    logger.info(f"Model type: {model_type}")

    # Auto-detect task if not provided using TasksManager
    if task is None:
        try:
            supported_tasks = TasksManager.get_supported_tasks_for_model_type(
                model_type,
                exporter=exporter,
                library_name=library_name or "transformers",
            )
            if supported_tasks:
                task = list(supported_tasks.keys())[0]  # Use first available task
                logger.info(f"Auto-detected task: {task}")
            else:
                task = "feature-extraction"
                logger.info(f"No supported tasks found, using default: {task}")
        except Exception as e:
            logger.warning(f"Task detection failed: {e}")
            task = "feature-extraction"
            logger.info(f"Using fallback task: {task}")

    # This is the key function - directly from Optimum TasksManager
    try:
        constructor = TasksManager.get_exporter_config_constructor(
            exporter=exporter,
            model_type=model_type,
            task=task,
            library_name=library_name or "transformers",
            exporter_config_kwargs=config_kwargs if config_kwargs else None,
        )

        # Create the actual export config
        export_config = constructor(config)

        logger.info(
            f"✅ Created {exporter} export config for {model_type} with task {task}"
        )
        return export_config

    except Exception as e:
        logger.error(f"Failed to create export config: {e}")
        raise ValueError(
            f"Could not create export config for {model_type}. "
            f"This model type may not be supported by Optimum for {exporter} export. "
            f"Supported model types: https://huggingface.co/docs/optimum/exporters/overview"
        ) from e


def generate_dummy_inputs_from_model_path(
    model_name_or_path: str,
    exporter: str = "onnx",
    task: str | None = None,
    library_name: str | None = None,
    **shape_kwargs,
) -> dict[str, torch.Tensor]:
    """
    Generate dummy inputs for any HuggingFace model using Optimum.

    Args:
        model_name_or_path: HuggingFace model name or local path
        exporter: Export backend ("onnx", "tflite", etc.)
        task: Task name (auto-detects if None)
        library_name: Library name ("transformers", "diffusers", etc.)
        **shape_kwargs: Optional shape parameters for input generation.
            These are model-specific and passed directly to Optimum's
            dummy input generator. Common parameters include batch_size,
            sequence_length, height, width, etc., depending on the model type.

    Returns:
        Dictionary of input tensors ready for model inference
    """
    # Get export config
    export_config = get_export_config_from_model_path(
        model_name_or_path=model_name_or_path,
        exporter=exporter,
        task=task,
        library_name=library_name,
    )

    # Set up input generation parameters with defaults
    from optimum.utils import DEFAULT_DUMMY_SHAPES

    shapes = DEFAULT_DUMMY_SHAPES.copy()

    # Apply defaults for common parameters if not provided
    if "batch_size" not in shape_kwargs:
        shape_kwargs["batch_size"] = 2
    if "sequence_length" not in shape_kwargs:
        shape_kwargs["sequence_length"] = 16

    # Update shapes with user-provided values
    shapes.update(shape_kwargs)

    # Generate dummy inputs using Optimum
    dummy_inputs = export_config.generate_dummy_inputs(framework="pt", **shapes)

    logger.info(f"Generated inputs: {list(dummy_inputs.keys())}")
    for name, tensor in dummy_inputs.items():
        logger.debug(f"  {name}: {list(tensor.shape)} ({tensor.dtype})")

    return dummy_inputs


def get_supported_tasks_for_model_path(
    model_name_or_path: str,
    exporter: str = "onnx",
    library_name: str | None = None,
) -> dict[str, Any]:
    """
    Get all supported tasks for a model using Optimum's TasksManager.

    Args:
        model_name_or_path: HuggingFace model name or local path
        exporter: Export backend ("onnx", "tflite", etc.)
        library_name: Library name ("transformers", "diffusers", etc.)

    Returns:
        Dictionary of supported tasks and their configurations
    """
    try:
        from optimum.exporters.tasks import TasksManager
    except ImportError as e:
        msg = "optimum is required. Install with: pip install optimum"
        raise ImportError(msg) from e

    # Load config to get model_type
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_type = config.model_type

    # Get supported tasks
    try:
        supported_tasks = TasksManager.get_supported_tasks_for_model_type(
            model_type, exporter=exporter, library_name=library_name or "transformers"
        )
        return supported_tasks
    except Exception as e:
        logger.warning(f"Could not get supported tasks: {e}")
        return {}


def get_model_info(model_name_or_path: str) -> dict[str, Any]:
    """
    Get comprehensive model information for debugging.

    Args:
        model_name_or_path: HuggingFace model name or local path

    Returns:
        Dictionary with model information
    """
    try:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

        info = {
            "model_type": config.model_type,
            "architectures": getattr(config, "architectures", []),
            "supported_tasks": get_supported_tasks_for_model_path(model_name_or_path),
            "config_keys": [k for k in dir(config) if not k.startswith("_")],
        }

        return info

    except Exception as e:
        return {"error": str(e)}
