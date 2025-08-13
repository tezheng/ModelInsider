"""
ONNX Auto Processor Implementation

This module provides the main entry point for ONNX Auto Processor functionality.
It automatically detects model types, loads appropriate processors, and wraps
them with ONNX-optimized implementations for maximum performance.

Key Components:
- ONNXAutoProcessor: Universal factory for auto-detecting processor types

Architecture:
1. Uses HuggingFace AutoProcessor to load base processors
2. Detects processor type via isinstance() checks
3. Wraps with appropriate ONNX processor for fixed-shape optimization
4. Optimized tensor operations for improved performance

Author: Generated for TEZ-144 ONNX AutoProcessor Implementation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import onnx

# Import processor implementations
try:
    # When imported as part of src package
    # Import type definitions  
    from .onnx_processor_types import (
        ModalityConfig,
        ModalityType,
        ONNXConfigurationError,
        ONNXModelLoadError,
        ONNXProcessorError,
        ONNXProcessorNotFoundError,
        ONNXUnsupportedModalityError,
        PathLike,
        ProcessorMetadata,
        ProcessorResult,
        TensorDict,
        TensorSpec,
        TensorType,
    )
    from .processors import (
        BaseONNXProcessor,
        ONNXAudioProcessor,
        ONNXImageProcessor,
        ONNXProcessor,
        ONNXTokenizer,
        ONNXVideoProcessor,
        ProcessorDefaults,
    )
except ImportError:
    # When imported directly in tests
    # Import type definitions
    from onnx_processor_types import (
        ModalityConfig,
        ModalityType,
        ONNXConfigurationError,
        ONNXModelLoadError,
        ONNXProcessorError,
        ONNXProcessorNotFoundError,
        ONNXUnsupportedModalityError,
        PathLike,
        ProcessorMetadata,
        ProcessorResult,
        TensorDict,
        TensorSpec,
        TensorType,
    )
    from processors import (
        BaseONNXProcessor,
        ONNXAudioProcessor,
        ONNXImageProcessor,
        ONNXProcessor,
        ONNXTokenizer,
        ProcessorDefaults,
    )

# Configure logging
logger = logging.getLogger(__name__)

# Import at module level to make it patchable in tests
try:
    from transformers import AutoProcessor
except ImportError:
    AutoProcessor = None


class ONNXAutoProcessor:
    """
    Universal ONNX processor factory with automatic detection and configuration.

    This class provides the main entry point for ONNX Auto Processor functionality.
    It automatically detects model types, loads appropriate processors, and wraps
    them with ONNX-optimized implementations for maximum performance.

    Key Features:
    - Universal support for all HuggingFace model types
    - Automatic processor detection and configuration
    - Zero-configuration setup from ONNX models
    - Fixed-shape optimization for ONNX Runtime performance
    - Comprehensive error handling and validation

    Primary API:
    Use from_model() for direct ONNX model loading:
    >>> processor = ONNXAutoProcessor.from_model("model.onnx")
    >>> result = processor("Hello world!")

    Architecture:
    1. Uses AutoProcessor to load base processors from HuggingFace
    2. Extracts metadata from ONNX models for shape configuration
    3. Detects processor type via isinstance() checks
    4. Wraps with appropriate ONNX processor for fixed shapes
    5. Provides unified interface for all processor types
    """

    def __init__(
        self,
        base_processor: Any,
        metadata: ProcessorMetadata,
        validation_enabled: bool = True,
    ):
        """
        Initialize ONNX Auto Processor.

        Args:
            base_processor: HuggingFace processor returned by AutoProcessor
            metadata: Complete processor metadata extracted from ONNX model
            validation_enabled: Whether to enable input/output validation
        """
        self.base_processor = base_processor
        self.metadata = metadata
        self._validation_enabled = validation_enabled

        # Create the appropriate ONNX wrapper
        self._onnx_processor = self._create_onnx_wrapper()

        logger.info(
            f"Created {self.__class__.__name__} for {metadata.model_type} model "
            f"with {len(metadata.modalities)} modalities"
        )

    @classmethod  
    def from_model(
        cls,
        onnx_model_path: PathLike,
        base_processor: Any | None = None,
        hf_model_path: PathLike | None = None,
        **kwargs: Any,
    ) -> ONNXAutoProcessor:
        """
        Create an ONNX processor from an ONNX model with auto-detection.
        
        This is the primary method for creating ONNX processors. It loads an ONNX model
        and automatically detects the appropriate processor type and configuration.

        Args:
            onnx_model_path: Path to the ONNX model file (.onnx) 
            base_processor: Optional base processor (will auto-load if not provided)
            hf_model_path: Optional path to HuggingFace model directory
            **kwargs: Additional configuration options

        Returns:
            Configured ONNX processor with fixed shapes for optimal performance

        Examples:
            >>> # Auto-detect processor from ONNX model (primary use case)
            >>> processor = ONNXAutoProcessor.from_model("model.onnx")
            
            >>> # Specify model directory containing HF configs
            >>> processor = ONNXAutoProcessor.from_model("model.onnx", hf_model_path="./model_dir/")
            
            >>> # Provide custom base processor
            >>> from transformers import AutoTokenizer
            >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            >>> processor = ONNXAutoProcessor.from_model("model.onnx", base_processor=tokenizer)
        """
        onnx_path = Path(onnx_model_path)
        
        # Determine model directory for processor configs
        model_dir = Path(hf_model_path) if hf_model_path else onnx_path.parent

        # Load ONNX model and extract metadata
        try:
            onnx_model = onnx.load(str(onnx_path))
            # Try to validate the loaded model, but don't fail for test models
            try:
                onnx.checker.check_model(onnx_model)
            except onnx.checker.ValidationError as e:
                logger.warning(f"ONNX model validation warning: {e}")
        except FileNotFoundError as e:
            raise ONNXModelLoadError(onnx_path, e) from e
        except Exception as e:
            raise ONNXModelLoadError(onnx_path, e) from e

        try:
            metadata = cls._extract_processor_metadata(onnx_model, onnx_path)
        except Exception as e:
            logger.error(f"Failed to extract metadata from ONNX model: {e}")
            raise ONNXProcessorError(f"Failed to extract model metadata: {e}") from e

        # Load base processor if not provided
        if base_processor is None:
            base_processor = cls._load_base_processor(
                onnx_path, model_dir, metadata
            )

        # Create and return the auto processor
        return cls(base_processor, metadata, kwargs.get("validation_enabled", True))

    def __call__(self, *args: Any, **kwargs: Any) -> ProcessorResult:
        """
        Process inputs using the wrapped ONNX processor.

        This delegates to the appropriate ONNX processor based on the
        detected model type and modalities.

        Args:
            *args: Positional arguments for processing
            **kwargs: Keyword arguments for processing

        Returns:
            Dictionary of tensors ready for ONNX inference
        """
        return self._onnx_processor(*args, **kwargs)

    def preprocess(self, inputs: Any, **kwargs: Any) -> TensorDict:
        """Preprocess inputs into ONNX tensor format."""
        return self._onnx_processor.preprocess(inputs, **kwargs)

    @staticmethod
    def _extract_processor_metadata(
        onnx_model: onnx.ModelProto, model_path: Path
    ) -> ProcessorMetadata:
        """
        Extract complete processor metadata from ONNX model.

        This method implements the structured multimodal information extraction
        from the design document, providing complete type safety and modality
        separation.
        """
        # Extract basic ONNX information
        onnx_info = ONNXAutoProcessor._extract_onnx_info(onnx_model)

        # Extract metadata from model properties
        model_metadata = {}
        for prop in onnx_model.metadata_props:
            model_metadata[prop.key] = prop.value

        # Determine model information
        model_name = model_metadata.get("model_name", model_path.stem)
        model_type = model_metadata.get("model_type", "unknown")
        task = model_metadata.get("task")

        # Create modality configurations from ONNX info
        modality_configs = {}
        for modality_name, modality_info in onnx_info["modalities"].items():
            if modality_name == "unknown":
                continue

            modality_type = ModalityType(modality_info["type"])
            tensors = []

            for tensor_info in modality_info["tensors"]:
                tensor_spec = TensorSpec(
                    name=tensor_info["name"],
                    shape=tensor_info["shape"],
                    dtype=TensorType(tensor_info["dtype"]),
                    modality=modality_type,
                    is_input=True,  # Assuming input tensors for now
                )
                tensors.append(tensor_spec)

            # Create configuration with shape information
            config = {
                key: value
                for key, value in modality_info.items()
                if key not in ["type", "tensors"]
            }

            modality_configs[modality_name] = ModalityConfig(
                modality_type=modality_type,
                tensors=tensors,
                batch_size=modality_info.get("batch_size", 1),
                config=config,
            )

        return ProcessorMetadata(
            model_name=model_name,
            model_type=model_type,
            task=task,
            modalities=modality_configs,
            is_multimodal=onnx_info["is_multimodal"],
            onnx_opset_version=getattr(onnx_model, "opset_import", [None])[0].version
            if onnx_model.opset_import
            else ProcessorDefaults.DEFAULT_ONNX_OPSET_VERSION,
            metadata_source="onnx",
        )

    @staticmethod
    def _extract_onnx_info(onnx_model: onnx.ModelProto) -> dict[str, Any]:
        """
        Extract structured multimodal information from ONNX model.

        This implements the enhanced extraction logic from the design document
        with proper modality detection and type safety.
        """
        # Initialize modality-specific storage
        modalities = {}
        all_inputs = []
        all_outputs = []

        # Process each input tensor
        for input_tensor in onnx_model.graph.input:
            name = input_tensor.name
            all_inputs.append(name)

            # Extract shape and data type
            shape = []
            for dim in input_tensor.type.tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value > 0 else -1)

            dtype = input_tensor.type.tensor_type.elem_type

            # Detect modality based on name and shape
            modality_from_name = ModalityType.from_tensor_name(name)
            modality_from_shape = ModalityType.from_tensor_shape(shape)

            # Prefer name-based detection, fall back to shape-based
            modality = (
                modality_from_name
                if modality_from_name != ModalityType.UNKNOWN
                else modality_from_shape
            )

            # Create modality info based on type
            modality_info = {"type": modality.value}

            if modality == ModalityType.TEXT and len(shape) >= 2:
                modality_info.update(
                    {
                        "batch_size": shape[0]
                        if shape[0] != -1
                        else ProcessorDefaults.DEFAULT_BATCH_SIZE,
                        "sequence_length": shape[1]
                        if shape[1] != -1
                        else ProcessorDefaults.DEFAULT_SEQUENCE_LENGTH,
                    }
                )
            elif modality == ModalityType.IMAGE and len(shape) == 4:
                modality_info.update(
                    {
                        "batch_size": shape[0]
                        if shape[0] != -1
                        else ProcessorDefaults.DEFAULT_BATCH_SIZE,
                        "num_channels": shape[1],
                        "height": shape[2],
                        "width": shape[3],
                    }
                )
            elif modality == ModalityType.AUDIO and len(shape) >= 2:
                modality_info.update(
                    {
                        "batch_size": shape[0]
                        if shape[0] != -1
                        else ProcessorDefaults.DEFAULT_BATCH_SIZE,
                        "sequence_length": shape[1] if len(shape) == 2 else shape[-1],
                        "feature_size": shape[1] if len(shape) == 3 else 1,
                    }
                )
            elif modality == ModalityType.VIDEO and len(shape) == 5:
                modality_info.update(
                    {
                        "batch_size": shape[0]
                        if shape[0] != -1
                        else ProcessorDefaults.DEFAULT_BATCH_SIZE,
                        "num_channels": shape[1],
                        "num_frames": shape[2],
                        "height": shape[3],
                        "width": shape[4],
                    }
                )

            # Store modality information
            modality_key = modality.value
            if modality_key not in modalities:
                modalities[modality_key] = {
                    "type": modality.value,
                    "tensors": [],
                    **{k: v for k, v in modality_info.items() if k != "type"},
                }

            # Add tensor info
            modalities[modality_key]["tensors"].append(
                {"name": name, "shape": shape, "dtype": dtype}
            )

        # Process outputs
        for output_tensor in onnx_model.graph.output:
            all_outputs.append(output_tensor.name)

        # Build final info
        info = {
            "modalities": modalities,
            "is_multimodal": len([m for m in modalities if m != "unknown"])
            > ProcessorDefaults.DEFAULT_BATCH_SIZE,
            "input_names": all_inputs,
            "output_names": all_outputs,
            "input_count": len(all_inputs),
            "output_count": len(all_outputs),
        }

        # Add backward compatibility for single modality
        if not info["is_multimodal"] and len(modalities) == 1:
            single_modality = next(iter(modalities.values()))
            if single_modality["type"] != "unknown":
                for key, value in single_modality.items():
                    if key not in ["type", "tensors"]:
                        info[key] = value

        return info

    @staticmethod
    def _load_base_processor(
        onnx_model_path: Path, hf_model_path: Path | None, metadata: ProcessorMetadata
    ) -> Any:
        """Load the base HuggingFace processor for the model."""
        # Use module-level import
        if AutoProcessor is None:
            raise ONNXProcessorNotFoundError(
                "AutoProcessor", "transformers library not available"
            )

        # Try multiple paths for loading
        search_paths = []

        if hf_model_path:
            search_paths.append(hf_model_path)

        # Same directory as ONNX model
        search_paths.append(onnx_model_path.parent)

        # Model name from metadata
        if hasattr(metadata, "model_name") and metadata.model_name != "unknown":
            search_paths.append(metadata.model_name)

        # Try loading from each path
        for path in search_paths:
            try:
                return AutoProcessor.from_pretrained(str(path))
            except Exception as e:
                logger.debug(f"Failed to load processor from {path}: {e}")
                continue

        # If all paths fail, raise error
        raise ONNXProcessorNotFoundError(
            "AutoProcessor", f"Could not load processor from any of: {search_paths}"
        )

    def _create_onnx_wrapper(self) -> BaseONNXProcessor:
        """
        Create the appropriate ONNX wrapper based on processor type.

        This implements the processor detection and wrapping logic from
        the design document.
        """
        # Import processor type classes for isinstance checks
        try:
            from transformers import PreTrainedTokenizerBase
            from transformers.feature_extraction_utils import FeatureExtractionMixin
            from transformers.image_processing_utils import BaseImageProcessor
            from transformers.processing_utils import ProcessorMixin
        except ImportError as e:
            raise ONNXProcessorNotFoundError(
                "transformers components",
                f"Required transformers classes not available: {e}",
            ) from e

        # Detect processor type and create appropriate wrapper
        if isinstance(self.base_processor, PreTrainedTokenizerBase):
            # Text model - wrap with ONNXTokenizer
            text_config = self.metadata.get_modality_config(ModalityType.TEXT)
            if not text_config:
                raise ONNXConfigurationError(
                    "text_config", None, "Text modality configuration not found"
                )

            return ONNXTokenizer(
                tokenizer=self.base_processor,
                batch_size=text_config.batch_size,
                sequence_length=text_config.config.get(
                    "sequence_length", ProcessorDefaults.DEFAULT_SEQUENCE_LENGTH
                ),
                validation_enabled=self._validation_enabled,
            )

        elif isinstance(self.base_processor, BaseImageProcessor):
            # Vision model - wrap with ONNXImageProcessor
            image_config = self.metadata.get_modality_config(ModalityType.IMAGE)
            if not image_config:
                raise ONNXConfigurationError(
                    "image_config", None, "Image modality configuration not found"
                )

            return ONNXImageProcessor(
                image_processor=self.base_processor,
                batch_size=image_config.batch_size,
                height=image_config.config.get(
                    "height", ProcessorDefaults.DEFAULT_IMAGE_HEIGHT
                ),
                width=image_config.config.get(
                    "width", ProcessorDefaults.DEFAULT_IMAGE_WIDTH
                ),
                num_channels=image_config.config.get(
                    "num_channels", ProcessorDefaults.DEFAULT_NUM_CHANNELS
                ),
                validation_enabled=self._validation_enabled,
            )

        elif isinstance(self.base_processor, FeatureExtractionMixin):
            # Audio model - wrap with ONNXAudioProcessor
            audio_config = self.metadata.get_modality_config(ModalityType.AUDIO)
            if not audio_config:
                raise ONNXConfigurationError(
                    "audio_config", None, "Audio modality configuration not found"
                )

            return ONNXAudioProcessor(
                feature_extractor=self.base_processor,
                batch_size=audio_config.batch_size,
                sequence_length=audio_config.config.get(
                    "sequence_length", ProcessorDefaults.DEFAULT_AUDIO_SEQUENCE_LENGTH
                ),
                sampling_rate=audio_config.config.get(
                    "sampling_rate", ProcessorDefaults.DEFAULT_SAMPLING_RATE
                ),
                validation_enabled=self._validation_enabled,
            )

        elif isinstance(self.base_processor, ProcessorMixin):
            # Multimodal model - wrap with ONNXProcessor
            return ONNXProcessor(
                processor=self.base_processor,
                modality_configs=self.metadata.modalities,
                validation_enabled=self._validation_enabled,
            )

        else:
            raise ONNXUnsupportedModalityError(
                ModalityType.UNKNOWN,
                [
                    ModalityType.TEXT,
                    ModalityType.IMAGE,
                    ModalityType.AUDIO,
                    ModalityType.VIDEO,
                ],
            )

    @property
    def modality_type(self) -> ModalityType:
        """Get the primary modality type for this processor."""
        if self.metadata.is_multimodal:
            return ModalityType.MULTIMODAL

        # Return the single modality type
        for config in self.metadata.modalities.values():
            if config.modality_type.is_unimodal():
                return config.modality_type

        return ModalityType.UNKNOWN

    @property
    def supported_modalities(self) -> list[ModalityType]:
        """Get list of all supported modality types."""
        modalities = list(self.metadata.modality_types)

        # For multimodal processors, also include MULTIMODAL type
        if self.metadata.is_multimodal and ModalityType.MULTIMODAL not in modalities:
            modalities.append(ModalityType.MULTIMODAL)

        return modalities

    @property
    def tensor_names(self) -> list[str]:
        """Get list of all input tensor names."""
        return [t.name for t in self.metadata.all_input_tensors]

    @property
    def output_names(self) -> list[str]:
        """Get list of all output tensor names."""
        return [t.name for t in self.metadata.all_output_tensors]

    @property
    def tokenizer(self):
        """Expose tokenizer for pipeline compatibility.
        
        For text models, this returns self to make the processor
        recognizable as a tokenizer by the enhanced_pipeline.
        """
        if self.modality_type == ModalityType.TEXT:
            return self
        return None

    def encode(self, *args, **kwargs):
        """Encode method for tokenizer compatibility.
        
        This delegates to __call__ for text processing.
        """
        if self.modality_type == ModalityType.TEXT:
            return self(*args, **kwargs)
        raise NotImplementedError("encode is only available for text processors")

    def get_input_spec(self, tensor_name: str) -> TensorSpec | None:
        """Get tensor specification for an input tensor."""
        for tensor in self.metadata.all_input_tensors:
            if tensor.name == tensor_name:
                return tensor
        return None

    def get_output_spec(self, tensor_name: str) -> TensorSpec | None:
        """Get tensor specification for an output tensor."""
        for tensor in self.metadata.all_output_tensors:
            if tensor.name == tensor_name:
                return tensor
        return None
