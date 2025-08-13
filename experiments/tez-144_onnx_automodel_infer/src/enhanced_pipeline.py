"""Enhanced Pipeline Wrapper with data_processor Parameter

This module provides a wrapper around the standard transformers pipeline
that accepts a generic `data_processor` parameter and automatically routes
it to the correct pipeline parameter (tokenizer, image_processor, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from transformers import PretrainedConfig, PreTrainedModel, TFPreTrainedModel
from transformers import pipeline as hf_pipeline

if TYPE_CHECKING:
    import torch


def create_pipeline(
    task: str,
    model: str | PreTrainedModel | TFPreTrainedModel | None = None,
    data_processor: Any | None = None,
    config: str | PretrainedConfig | None = None,
    framework: str | None = None,
    revision: str | None = None,
    use_fast: bool = True,
    token: str | bool | None = None,
    device: int | str | torch.device | None = None,
    device_map: str | dict[str, int | str] | None = None,
    torch_dtype: str | torch.dtype | None = None,
    trust_remote_code: bool | None = None,
    model_kwargs: dict[str, Any] | None = None,
    pipeline_class: Any | None = None,
    **kwargs: Any,
) -> Any:
    """
    Create a pipeline with a generic data_processor parameter.

    This wrapper function automatically routes the data_processor to the correct
    pipeline parameter based on the task type and processor characteristics.

    Args:
        task: The task defining which pipeline will be returned. Available tasks:
            - Text: "feature-extraction", "text-classification", "sentiment-analysis",
                   "token-classification", "ner", "question-answering", "fill-mask",
                   "summarization", "translation", "text-generation", etc.
            - Vision: "image-classification", "image-segmentation", "object-detection",
                     "image-to-text", "image-feature-extraction", etc.
            - Audio: "audio-classification", "automatic-speech-recognition", etc.
            - Multimodal: "zero-shot-image-classification", "document-question-answering", etc.

        model: The model to use. Can be a model instance, or a string model identifier.

        data_processor: The data processor to use. This can be:
            - A tokenizer (for text tasks)
            - An image processor (for vision tasks)
            - A feature extractor (for audio tasks)
            - A processor (for multimodal tasks)
            - A custom processor with appropriate attributes

        config: Model configuration (optional).
        framework: The framework to use ("pt" for PyTorch, "tf" for TensorFlow).
        revision: Model revision to use.
        use_fast: Whether to use fast tokenizer if available.
        token: Hugging Face Hub token for private models.
        device: Device to run the pipeline on.
        device_map: Device map for model parallelism.
        torch_dtype: Torch dtype for model weights.
        trust_remote_code: Whether to trust remote code.
        model_kwargs: Additional model loading arguments.
        pipeline_class: Custom pipeline class to use.
        **kwargs: Additional pipeline arguments.

    Returns:
        A pipeline object configured for the specified task.

    Examples:
        >>> from enhanced_pipeline import create_pipeline
        >>> from src.onnx_tokenizer import ONNXTokenizer

        >>> # Text task with ONNX tokenizer
        >>> onnx_tokenizer = ONNXTokenizer(base_tokenizer, onnx_model=model)
        >>> pipe = create_pipeline("feature-extraction", model=model, data_processor=onnx_tokenizer)

        >>> # Vision task with image processor
        >>> pipe = create_pipeline("image-classification", model=model, data_processor=image_processor)

        >>> # Audio task with feature extractor
        >>> pipe = create_pipeline("automatic-speech-recognition", model=model, data_processor=feature_extractor)

        >>> # Multimodal task with processor
        >>> pipe = create_pipeline("image-to-text", model=model, data_processor=processor)
    """

    # Task categories for routing logic
    TEXT_TASKS = {
        "feature-extraction",
        "text-classification",
        "sentiment-analysis",
        "token-classification",
        "ner",
        "named-entity-recognition",
        "question-answering",
        "fill-mask",
        "summarization",
        "translation",
        "text2text-generation",
        "text-generation",
        "zero-shot-classification",
        "conversational",
        "table-question-answering",
    }

    VISION_TASKS = {
        "image-classification",
        "image-segmentation",
        "object-detection",
        "image-feature-extraction",
        "depth-estimation",
        "image-to-image",
        "mask-generation",
    }

    AUDIO_TASKS = {
        "audio-classification",
        "automatic-speech-recognition",
        "asr",
        "text-to-audio",
        "text-to-speech",
        "audio-to-audio",
    }

    MULTIMODAL_TASKS = {
        "image-to-text",
        "document-question-answering",
        "vqa",
        "visual-question-answering",
        "zero-shot-image-classification",
        "image-text-to-text",
        "video-classification",
    }

    # Prepare arguments for the standard pipeline
    pipeline_kwargs = {
        "task": task,
        "model": model,
        "config": config,
        "framework": framework,
        "revision": revision,
        "use_fast": use_fast,
        "token": token,
        "device": device,
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        "model_kwargs": model_kwargs,
        "pipeline_class": pipeline_class,
    }

    # Add any additional kwargs
    pipeline_kwargs.update(kwargs)

    # Route data_processor to the appropriate parameter
    if data_processor is not None:
        # Try to detect the type of processor
        processor_type = _detect_processor_type(data_processor)

        if processor_type == "tokenizer":
            pipeline_kwargs["tokenizer"] = data_processor
        elif processor_type == "image_processor":
            pipeline_kwargs["image_processor"] = data_processor
        elif processor_type == "feature_extractor":
            pipeline_kwargs["feature_extractor"] = data_processor
        elif processor_type == "processor":
            pipeline_kwargs["processor"] = data_processor
        else:
            # Default routing based on task
            if task in TEXT_TASKS:
                pipeline_kwargs["tokenizer"] = data_processor
            elif task in VISION_TASKS:
                pipeline_kwargs["image_processor"] = data_processor
            elif task in AUDIO_TASKS:
                pipeline_kwargs["feature_extractor"] = data_processor
            elif task in MULTIMODAL_TASKS:
                pipeline_kwargs["processor"] = data_processor
            else:
                # Unknown task, try tokenizer as default for backward compatibility
                pipeline_kwargs["tokenizer"] = data_processor

    # Create and return the pipeline
    return hf_pipeline(**pipeline_kwargs)


def _detect_processor_type(processor: Any) -> str:
    """
    Detect the type of data processor based on its attributes and class.

    Args:
        processor: The data processor to analyze.

    Returns:
        A string indicating the processor type:
        "tokenizer", "image_processor", "feature_extractor", "processor", or "unknown"
    """

    # Check by class inheritance (most reliable)
    processor_class_name = processor.__class__.__name__

    # Check for ONNXAutoProcessor (our universal processor)
    if processor_class_name == "ONNXAutoProcessor":
        # Check modality type if available
        if hasattr(processor, "modality_type"):
            modality = str(processor.modality_type)
            if "TEXT" in modality:
                return "tokenizer"
            elif "IMAGE" in modality:
                return "image_processor"
            elif "AUDIO" in modality:
                return "feature_extractor"
            elif "MULTIMODAL" in modality:
                return "processor"
        # Fallback to checking for tokenizer attribute
        if hasattr(processor, "tokenizer"):
            return "tokenizer"

    # Check for tokenizer types
    if any(name in processor_class_name for name in ["Tokenizer", "TokenizerFast"]):
        return "tokenizer"

    # Check for image processor types
    if any(
        name in processor_class_name for name in ["ImageProcessor", "ImageProcessing"]
    ):
        return "image_processor"

    # Check for feature extractor types
    if any(
        name in processor_class_name
        for name in ["FeatureExtractor", "FeatureExtraction"]
    ):
        return "feature_extractor"

    # Check for multimodal processor types
    if "Processor" in processor_class_name and not any(
        name in processor_class_name for name in ["ImageProcessor", "TokenProcessor"]
    ):
        return "processor"

    # Check by attributes (fallback method)
    if hasattr(processor, "tokenize") or hasattr(processor, "encode"):
        return "tokenizer"

    if hasattr(processor, "pixel_values") or hasattr(processor, "preprocess_image"):
        return "image_processor"

    if hasattr(processor, "feature_size") or hasattr(processor, "sampling_rate"):
        return "feature_extractor"

    if hasattr(processor, "tokenizer") and hasattr(processor, "image_processor"):
        return "processor"

    # Check for custom wrapper classes (like FixedShapeTokenizer)
    if hasattr(processor, "tokenizer") and not hasattr(processor, "image_processor"):
        # This is likely a tokenizer wrapper
        return "tokenizer"

    return "unknown"


# Convenience function for backward compatibility
def pipeline(task: str, model=None, data_processor=None, **kwargs):
    """
    Convenience function that mimics the standard pipeline API but with data_processor support.

    This is a shorter alias for create_pipeline that can be used as a drop-in replacement
    for the standard transformers.pipeline function.

    Examples:
        >>> from enhanced_pipeline import pipeline
        >>> pipe = pipeline("feature-extraction", model=model, data_processor=my_tokenizer)
    """
    return create_pipeline(task, model=model, data_processor=data_processor, **kwargs)




if __name__ == "__main__":
    # Example usage and testing
    print("Enhanced Pipeline Module")
    print("=" * 60)
    print("This module provides:")
    print("1. create_pipeline() - Full-featured pipeline with data_processor")
    print("2. pipeline() - Drop-in replacement for transformers.pipeline")
    print("3. create_fixed_shape_pipeline() - Convenience for ONNX models")
    print("\nUsage:")
    print("  from enhanced_pipeline import pipeline")
    print("  pipe = pipeline('task', model=model, data_processor=processor)")
    print("\nThe data_processor is automatically routed to the correct parameter!")
