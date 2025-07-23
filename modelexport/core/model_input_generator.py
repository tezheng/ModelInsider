#!/usr/bin/env python3
"""
Universal Model Input Generator

Unified input generation supporting both manual input specs and automatic
generation via Optimum's TasksManager.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoConfig

logger = logging.getLogger(__name__)


def _generate_from_specs(input_specs: dict[str, dict[str, Any]]) -> dict[str, torch.Tensor]:
    """
    Generate dummy inputs from manual specifications.
    
    Args:
        input_specs: Input specifications with format:
            {
                "input_name": {
                    "dtype": "int" | "float",
                    "shape": [1, 128],  # Required
                    "range": [0, 1000]  # Optional
                }
            }
    
    Returns:
        Dictionary of generated tensors
        
    Raises:
        ValueError: If specs are invalid
    """
    inputs = {}
    
    for name, spec in input_specs.items():
        # Validate required fields
        if "dtype" not in spec:
            raise ValueError(f"Missing 'dtype' in input spec for '{name}'")
        if "shape" not in spec:
            raise ValueError(f"Missing 'shape' in input spec for '{name}'")
        
        # Parse dtype
        dtype_str = spec["dtype"].lower()
        if dtype_str in ["int", "long", "int64"]:
            dtype = torch.long
        elif dtype_str in ["float", "float32"]:
            dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype '{spec['dtype']}' for '{name}'. Use 'int' or 'float'")
        
        # Parse shape
        shape = spec["shape"]
        if not isinstance(shape, list):
            raise ValueError(f"Shape must be a list for '{name}', got {type(shape)}")
        
        # Generate values
        if "range" in spec:
            if len(spec["range"]) != 2:
                raise ValueError(f"Range must have exactly 2 values [min, max] for '{name}'")
            min_val, max_val = spec["range"]
            
            if dtype == torch.long:
                inputs[name] = torch.randint(min_val, max_val + 1, shape, dtype=dtype)
            else:
                inputs[name] = torch.rand(shape, dtype=dtype) * (max_val - min_val) + min_val
        else:
            # Default ranges
            if dtype == torch.long:
                inputs[name] = torch.randint(0, 2, shape, dtype=dtype)  # Default: 0 or 1
            else:
                inputs[name] = torch.rand(shape, dtype=dtype)  # Default: [0, 1)
        
        logger.info(f"Generated '{name}': shape={list(inputs[name].shape)}, dtype={inputs[name].dtype}")
    
    return inputs




def generate_dummy_inputs(
    model_name_or_path: str | None = None,
    input_specs: dict[str, dict[str, Any]] | None = None,
    exporter: str = "onnx",
    **kwargs
) -> dict[str, torch.Tensor]:
    """
    Generate dummy inputs based on input specs or model auto-detection.
    
    Priority: input_specs > model_name_or_path (NO fallback)
    
    Args:
        model_name_or_path: HuggingFace model name/path for auto-generation
        input_specs: Manual input specifications. If provided, overrides everything.
            Format: {
                "input_ids": {
                    "dtype": "int",     # Required: "int" or "float"
                    "shape": [1, 128],  # Required: list of dimensions
                    "range": [0, 1000]  # Optional: [min, max] for value generation
                }
            }
        exporter: Export format for auto-generation
        **kwargs: Additional arguments for auto-generation
        
    Returns:
        Dictionary of input tensors
        
    Raises:
        ValueError: If neither input_specs nor model_name_or_path provided
        ValueError: If input_specs is invalid (fail fast)
    """
    if input_specs is not None:
        # Use ONLY input_specs, no fallback
        try:
            logger.info("Generating inputs from provided specs")
            return _generate_from_specs(input_specs)
        except Exception as e:
            logger.error(f"Failed to generate inputs from specs: {e}")
            logger.error("input_specs is optional - if not provided, the exporter will auto-generate inputs based on the model")
            raise ValueError(f"Invalid input_specs: {e}") from e
    
    elif model_name_or_path is not None:
        # Use Optimum auto-generation
        logger.info(f"Auto-generating inputs for model: {model_name_or_path}")
        return generate_dummy_inputs_from_model_path(
            model_name_or_path=model_name_or_path,
            exporter=exporter,
            **kwargs
        )
    else:
        raise ValueError(
            "Either input_specs or model_name_or_path must be provided. "
            "input_specs is optional - if not provided, the exporter will auto-generate inputs based on the model."
        )


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
                task = next(iter(supported_tasks.keys()))  # Use first available task
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


def patch_export_config(export_config) -> None:
    """
    Apply model-specific patches to export configurations.
    
    Args:
        export_config: Optimum export config instance to patch
        
    This function checks the config type and applies appropriate patches:
    - SamOnnxConfig: Forces pixel_values generation for full model export
    - Future models: Add additional patches as needed
    """
    config_type = type(export_config).__name__
    
    if config_type == "SamOnnxConfig":
        # TEZ-48 Fix: Override generate_dummy_inputs to force pixel_values generation
        original_generate = export_config.generate_dummy_inputs
        
        def generate_full_model_inputs(framework="pt", **kwargs):
            """Generate inputs for full SAM model export with pixel_values"""
            import torch
            from optimum.utils import DEFAULT_DUMMY_SHAPES
            
            # Merge default shapes with user overrides
            shapes = DEFAULT_DUMMY_SHAPES.copy()
            shapes.update(kwargs)
            
            # Generate inputs for full model export
            batch_size = shapes.get("batch_size", 1)
            
            inputs = {
                # Generate pixel_values for full model (includes vision encoder)
                "pixel_values": torch.randn(batch_size, 3, 1024, 1024, dtype=torch.float32),
                
                # Generate semantic input points (center region)
                "input_points": torch.tensor([[[[512.0, 512.0]]]] * batch_size, dtype=torch.float32),
                
                # Generate input labels (foreground point)
                "input_labels": torch.tensor([[[1]]] * batch_size, dtype=torch.long),
            }
            
            logger.info(f"Generated full model SAM inputs: {list(inputs.keys())}")
            for name, tensor in inputs.items():
                logger.debug(f"  {name}: {list(tensor.shape)} ({tensor.dtype})")
            
            return inputs
        
        # Replace the generate_dummy_inputs method
        export_config.generate_dummy_inputs = generate_full_model_inputs
        logger.info(f"🎯 Applied full model export fix for {config_type} (pixel_values instead of embeddings)")
    
    # Future model patches can be added here
    # elif config_type == "SomeOtherConfig":
    #     apply_other_patch(export_config)


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
    
    # Update shapes with user-provided values
    shapes.update(shape_kwargs)

    # Apply model-specific patches if needed
    patch_export_config(export_config)
    
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
