"""
ONNX Text Processor Implementation

This module provides the ONNX-optimized tokenizer for text processing with fixed shapes.
It wraps HuggingFace tokenizers (BERT, GPT, T5, etc.) and provides fixed-shape tokenization
optimized for ONNX inference.

Author: Generated for TEZ-144 ONNX AutoProcessor Implementation
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    # When imported as part of package structure
    from ..onnx_processor_types import (
        ModalityConfig,
        ModalityType,
        ONNXProcessorError,
        ONNXShapeError,
        ProcessorResult,
        TensorDict,
        TensorSpec,
        TensorType,
    )
    from .base import BaseONNXProcessor, ProcessorDefaults
except ImportError:
    # When imported directly in tests
    from onnx_processor_types import (
        ModalityConfig,
        ModalityType,
        ONNXProcessorError,
        ONNXShapeError,
        ProcessorResult,
        TensorDict,
        TensorSpec,
        TensorType,
    )
    from processors.base import BaseONNXProcessor, ProcessorDefaults

# Configure logging
logger = logging.getLogger(__name__)


class ONNXTokenizer(BaseONNXProcessor):
    """
    ONNX-optimized tokenizer for text processing with fixed shapes.

    This class wraps HuggingFace tokenizers (BERT, GPT, T5, etc.) and provides
    fixed-shape tokenization optimized for ONNX inference. It handles padding,
    truncation, and batch size enforcement automatically.

    Key Features:
    - Fixed sequence length for optimal ONNX performance
    - Automatic padding and truncation
    - Support for all HuggingFace tokenizer features
    - Batch size enforcement
    - Input validation and error handling

    Attributes:
        sequence_length: Fixed sequence length for all outputs
        vocab_size: Vocabulary size of the tokenizer
        special_tokens: Dictionary of special token IDs
    """

    def __init__(
        self,
        tokenizer: Any,
        batch_size: int = ProcessorDefaults.DEFAULT_BATCH_SIZE,
        sequence_length: int = ProcessorDefaults.DEFAULT_SEQUENCE_LENGTH,
        **kwargs: Any,
    ):
        """
        Initialize ONNX tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer instance
            batch_size: Fixed batch size for ONNX optimization
            sequence_length: Fixed sequence length for padding/truncation
            **kwargs: Additional configuration options

        Raises:
            ONNXConfigurationError: If configuration is invalid
        """
        # Create modality configuration
        modality_config = self._create_modality_config(
            tokenizer, batch_size, sequence_length, **kwargs
        )

        super().__init__(
            tokenizer, modality_config, kwargs.get("validation_enabled", True)
        )

        self.sequence_length = sequence_length
        self.vocab_size = getattr(
            tokenizer, "vocab_size", ProcessorDefaults.DEFAULT_VOCAB_SIZE
        )
        self.special_tokens = self._extract_special_tokens(tokenizer)

        # Configure tokenizer for fixed shapes
        self._configure_tokenizer_for_fixed_shapes()

    @staticmethod
    def _create_modality_config(
        tokenizer: Any, batch_size: int, sequence_length: int, **kwargs: Any
    ) -> ModalityConfig:
        """Create modality configuration for text processing."""
        # Standard text tensors
        tensors = [
            TensorSpec(
                name="input_ids",
                shape=[batch_size, sequence_length],
                dtype=TensorType.INT64,
                modality=ModalityType.TEXT,
                is_input=True,
                description="Token IDs for input text",
            ),
            TensorSpec(
                name="attention_mask",
                shape=[batch_size, sequence_length],
                dtype=TensorType.INT64,
                modality=ModalityType.TEXT,
                is_input=True,
                description="Attention mask for input tokens",
            ),
        ]

        # Add token_type_ids for models that use them (BERT-like)
        if (
            hasattr(tokenizer, "token_type_ids")
            or getattr(tokenizer, "model_max_length", ProcessorDefaults.BERT_MAX_LENGTH)
            > 0
        ):
            tensors.append(
                TensorSpec(
                    name="token_type_ids",
                    shape=[batch_size, sequence_length],
                    dtype=TensorType.INT64,
                    modality=ModalityType.TEXT,
                    is_input=True,
                    description="Token type IDs for segment separation",
                )
            )

        config = {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "vocab_size": getattr(
                tokenizer, "vocab_size", ProcessorDefaults.DEFAULT_VOCAB_SIZE
            ),
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "np",
            **kwargs,
        }

        return ModalityConfig(
            modality_type=ModalityType.TEXT,
            tensors=tensors,
            batch_size=batch_size,
            config=config,
            processor_class=tokenizer.__class__.__name__,
        )

    def _extract_special_tokens(self, tokenizer: Any) -> dict[str, int]:
        """Extract special token IDs from tokenizer."""
        special_tokens = {}

        for token_name in [
            "pad_token_id",
            "cls_token_id",
            "sep_token_id",
            "unk_token_id",
            "mask_token_id",
            "bos_token_id",
            "eos_token_id",
        ]:
            token_id = getattr(tokenizer, token_name, None)
            if token_id is not None:
                special_tokens[token_name] = token_id

        return special_tokens

    def _configure_tokenizer_for_fixed_shapes(self) -> None:
        """Configure tokenizer for optimal fixed-shape processing."""
        # Set tokenizer parameters for fixed shapes
        if hasattr(self.base_processor, "model_max_length"):
            self.base_processor.model_max_length = self.sequence_length

        # Configure padding and truncation
        self.base_processor.padding_side = getattr(
            self.base_processor, "padding_side", "right"
        )
        self.base_processor.truncation_side = getattr(
            self.base_processor, "truncation_side", "right"
        )

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | list[str] | None = None,
        **kwargs: Any,
    ) -> ProcessorResult:
        """
        Tokenize text with fixed shapes for ONNX inference.

        Args:
            text: Input text or list of texts
            text_pair: Optional second text for pair classification
            **kwargs: Additional tokenization arguments

        Returns:
            Dictionary with tokenized tensors ready for ONNX inference

        Examples:
            >>> tokenizer = ONNXTokenizer(hf_tokenizer, batch_size=1, sequence_length=128)
            >>> result = tokenizer("Hello world!")
            >>> result['input_ids'].shape
            (1, 128)
        """
        self._validate_inputs(text)

        # Ensure text is a list for batch processing
        if isinstance(text, str):
            text = [text]

        # Handle batch size mismatch
        if len(text) > self.batch_size:
            logger.warning(
                f"Input batch size {len(text)} exceeds configured batch size {self.batch_size}. "
                f"Truncating to {self.batch_size} samples."
            )
            text = text[: self.batch_size]
        elif len(text) < self.batch_size:
            # Pad with empty strings
            text = text + [""] * (self.batch_size - len(text))

        # Handle text pairs
        if text_pair is not None:
            if isinstance(text_pair, str):
                text_pair = [text_pair]
            if len(text_pair) < len(text):
                text_pair = text_pair + [None] * (len(text) - len(text_pair))

        # Tokenize with fixed parameters
        tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": self.sequence_length,
            "return_tensors": "np",
            **kwargs,
        }

        try:
            if text_pair is not None:
                result = self.base_processor(text, text_pair, **tokenizer_kwargs)
            else:
                result = self.base_processor(text, **tokenizer_kwargs)
        except Exception as e:
            raise ONNXProcessorError(
                f"Tokenization failed: {e}", processor_type=self.__class__.__name__
            ) from e

        # Convert to standard tensor dictionary
        tensor_dict = self._convert_to_tensor_dict(result)

        # Ensure fixed batch size
        tensor_dict = self._ensure_fixed_batch_size(tensor_dict)

        # Validate outputs
        self._validate_outputs(tensor_dict)

        return tensor_dict

    def preprocess(self, text: str | list[str], **kwargs: Any) -> TensorDict:
        """
        Preprocess text into ONNX tensor format.

        This is an alias for __call__ to match the base class interface.
        """
        return self.__call__(text, **kwargs)

    def encode(
        self, text: str | list[str], add_special_tokens: bool = True, **kwargs: Any
    ) -> ProcessorResult:
        """
        Encode text into token tensors with fixed shapes.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens
            **kwargs: Additional encoding arguments

        Returns:
            Dictionary with encoded tensors
        """
        kwargs["add_special_tokens"] = add_special_tokens
        return self.__call__(text, **kwargs)

    def decode(
        self,
        token_ids: NDArray[np.integer],
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> str | list[str]:
        """
        Decode token tensors back to text.

        Args:
            token_ids: Token ID array of shape [batch_size, sequence_length]
            skip_special_tokens: Whether to skip special tokens in output
            **kwargs: Additional decoding arguments

        Returns:
            Decoded text string or list of strings
        """
        if token_ids.ndim != 2:
            raise ONNXShapeError(
                [self.batch_size, self.sequence_length], list(token_ids.shape)
            )

        try:
            if token_ids.shape[0] == 1:
                # Single sample
                return self.base_processor.decode(
                    token_ids[0], skip_special_tokens=skip_special_tokens, **kwargs
                )
            else:
                # Batch of samples
                return self.base_processor.batch_decode(
                    token_ids, skip_special_tokens=skip_special_tokens, **kwargs
                )
        except Exception as e:
            raise ONNXProcessorError(
                f"Decoding failed: {e}", processor_type=self.__class__.__name__
            ) from e

    def _convert_to_tensor_dict(self, tokenizer_output: Any) -> TensorDict:
        """Convert tokenizer output to tensor dictionary."""
        tensor_dict = {}

        # Handle BatchEncoding or dict output
        if hasattr(tokenizer_output, "data"):
            # BatchEncoding object
            for key, value in tokenizer_output.data.items():
                if isinstance(value, np.ndarray):
                    tensor_dict[key] = value
                else:
                    tensor_dict[key] = np.array(value)
        elif isinstance(tokenizer_output, dict):
            # Dictionary output
            for key, value in tokenizer_output.items():
                if isinstance(value, np.ndarray):
                    tensor_dict[key] = value
                else:
                    tensor_dict[key] = np.array(value)
        else:
            raise ONNXProcessorError(
                f"Unexpected tokenizer output type: {type(tokenizer_output)}"
            )

        return tensor_dict
