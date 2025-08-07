"""
HTP Config Builder for Optimum Compatibility.

This module provides functionality to generate config.json files that are
compatible with Hugging Face Optimum's ONNX Runtime models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoProcessor,
    AutoImageProcessor,
    AutoFeatureExtractor,
    PretrainedConfig,
)

logger = logging.getLogger(__name__)


class HTPConfigBuilder:
    """
    Builder for generating Optimum-compatible configuration files.
    
    This class handles the generation of config.json and preprocessing files
    (tokenizer, processor, image processor, feature extractor) needed for 
    loading ONNX models with Optimum's ORTModel classes.
    """
    
    def __init__(self, model_name_or_path: str | None = None):
        """
        Initialize the config builder.
        
        Args:
            model_name_or_path: HuggingFace model name or path to load config from
        """
        self.model_name_or_path = model_name_or_path
        self._config: PretrainedConfig | None = None
        self._tokenizer: Any = None
        self._processor: Any = None
        self._image_processor: Any = None
        self._feature_extractor: Any = None
    
    def load_config(self) -> PretrainedConfig | None:
        """
        Load the model configuration from HuggingFace.
        
        Returns:
            PretrainedConfig object if successful, None otherwise
        """
        if not self.model_name_or_path:
            logger.warning("No model name or path provided, cannot load config")
            return None
            
        try:
            self._config = AutoConfig.from_pretrained(self.model_name_or_path)
            logger.info(f"Loaded config for model: {self.model_name_or_path}")
            return self._config
        except Exception as e:
            logger.warning(f"Failed to load config from {self.model_name_or_path}: {e}")
            return None
    
    def detect_preprocessor_type(self) -> str:
        """
        Detect what type of preprocessor this model uses.
        
        Returns:
            Type of preprocessor: 'processor', 'tokenizer', 'image_processor', 
            'feature_extractor', or 'unknown'
        """
        if not self.model_name_or_path:
            return "unknown"
        
        # Try processor first (for multimodal models)
        try:
            _ = AutoProcessor.from_pretrained(self.model_name_or_path)
            return "processor"
        except Exception:
            pass
        
        # Try tokenizer (for text models)
        try:
            _ = AutoTokenizer.from_pretrained(self.model_name_or_path)
            return "tokenizer"
        except Exception:
            pass
        
        # Try image processor (for vision models)
        try:
            _ = AutoImageProcessor.from_pretrained(self.model_name_or_path)
            return "image_processor"
        except Exception:
            pass
        
        # Try feature extractor (for audio models)
        try:
            _ = AutoFeatureExtractor.from_pretrained(self.model_name_or_path)
            return "feature_extractor"
        except Exception:
            pass
        
        return "unknown"
    
    def load_processor(self) -> Any:
        """
        Load the processor for multimodal models.
        
        Returns:
            Processor object if successful, None otherwise
        """
        if not self.model_name_or_path:
            logger.warning("No model name or path provided, cannot load processor")
            return None
        
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_name_or_path)
            logger.info(f"Loaded processor for model: {self.model_name_or_path}")
            return self._processor
        except Exception as e:
            logger.debug(f"No processor available for {self.model_name_or_path}: {e}")
            return None
    
    def load_image_processor(self) -> Any:
        """
        Load the image processor for vision models.
        
        Returns:
            Image processor object if successful, None otherwise
        """
        if not self.model_name_or_path:
            logger.warning("No model name or path provided, cannot load image processor")
            return None
        
        try:
            self._image_processor = AutoImageProcessor.from_pretrained(self.model_name_or_path)
            logger.info(f"Loaded image processor for model: {self.model_name_or_path}")
            return self._image_processor
        except Exception as e:
            logger.debug(f"No image processor available for {self.model_name_or_path}: {e}")
            return None
    
    def load_feature_extractor(self) -> Any:
        """
        Load the feature extractor for audio models.
        
        Returns:
            Feature extractor object if successful, None otherwise
        """
        if not self.model_name_or_path:
            logger.warning("No model name or path provided, cannot load feature extractor")
            return None
        
        try:
            self._feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name_or_path)
            logger.info(f"Loaded feature extractor for model: {self.model_name_or_path}")
            return self._feature_extractor
        except Exception as e:
            logger.debug(f"No feature extractor available for {self.model_name_or_path}: {e}")
            return None
    
    def load_tokenizer(self) -> Any:
        """
        Load the tokenizer from HuggingFace.
        
        Returns:
            Tokenizer object if successful, None otherwise
        """
        if not self.model_name_or_path:
            logger.warning("No model name or path provided, cannot load tokenizer")
            return None
            
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            logger.info(f"Loaded tokenizer for model: {self.model_name_or_path}")
            return self._tokenizer
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {self.model_name_or_path}: {e}")
            return None
    
    def save_config(self, output_dir: str | Path, config: PretrainedConfig | None = None) -> bool:
        """
        Save the model configuration to config.json.
        
        Args:
            output_dir: Directory to save the config file
            config: Optional config to save (uses loaded config if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config_to_save = config or self._config
        if not config_to_save:
            logger.warning("No config available to save")
            return False
        
        try:
            config_path = output_dir / "config.json"
            config_to_save.save_pretrained(output_dir)
            logger.info(f"Saved config.json to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def save_processor(self, output_dir: str | Path, processor: Any = None) -> bool:
        """
        Save the processor files (for multimodal models).
        
        Args:
            output_dir: Directory to save the processor files
            processor: Optional processor to save (uses loaded processor if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processor_to_save = processor or self._processor
        if not processor_to_save:
            logger.warning("No processor available to save")
            return False
        
        try:
            processor_to_save.save_pretrained(output_dir)
            logger.info(f"Saved processor files to {output_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to save processor: {e}")
            return False
    
    def save_image_processor(self, output_dir: str | Path, image_processor: Any = None) -> bool:
        """
        Save the image processor files (for vision models).
        
        Args:
            output_dir: Directory to save the image processor files
            image_processor: Optional image processor to save (uses loaded if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processor_to_save = image_processor or self._image_processor
        if not processor_to_save:
            logger.warning("No image processor available to save")
            return False
        
        try:
            processor_to_save.save_pretrained(output_dir)
            logger.info(f"Saved image processor files to {output_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to save image processor: {e}")
            return False
    
    def save_feature_extractor(self, output_dir: str | Path, feature_extractor: Any = None) -> bool:
        """
        Save the feature extractor files (for audio models).
        
        Args:
            output_dir: Directory to save the feature extractor files
            feature_extractor: Optional feature extractor to save (uses loaded if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extractor_to_save = feature_extractor or self._feature_extractor
        if not extractor_to_save:
            logger.warning("No feature extractor available to save")
            return False
        
        try:
            extractor_to_save.save_pretrained(output_dir)
            logger.info(f"Saved feature extractor files to {output_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to save feature extractor: {e}")
            return False
    
    def save_tokenizer(self, output_dir: str | Path, tokenizer: Any = None) -> bool:
        """
        Save the tokenizer files.
        
        Args:
            output_dir: Directory to save the tokenizer files
            tokenizer: Optional tokenizer to save (uses loaded tokenizer if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tokenizer_to_save = tokenizer or self._tokenizer
        if not tokenizer_to_save:
            logger.warning("No tokenizer available to save")
            return False
        
        try:
            tokenizer_to_save.save_pretrained(output_dir)
            logger.info(f"Saved tokenizer files to {output_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to save tokenizer: {e}")
            return False
    
    def generate_optimum_config(self, output_dir: str | Path, 
                                save_preprocessor: bool = True,
                                additional_config: dict[str, Any] | None = None) -> dict[str, bool]:
        """
        Generate all necessary files for Optimum compatibility.
        
        This method intelligently detects and saves the appropriate preprocessor
        (processor, tokenizer, image processor, or feature extractor) based on
        the model type.
        
        Args:
            output_dir: Directory to save all files
            save_preprocessor: Whether to save preprocessor files (tokenizer/processor/etc)
            additional_config: Additional config values to merge
            
        Returns:
            Dictionary with success status for each component
        """
        output_dir = Path(output_dir)
        results = {
            "config": False,
            "preprocessor": False,
            "preprocessor_type": "unknown",
        }
        
        # Load and save config
        if self.load_config():
            if additional_config:
                # Merge additional config values
                for key, value in additional_config.items():
                    setattr(self._config, key, value)
            results["config"] = self.save_config(output_dir)
        
        # Detect and save appropriate preprocessor if requested
        if save_preprocessor:
            preprocessor_type = self.detect_preprocessor_type()
            results["preprocessor_type"] = preprocessor_type
            
            if preprocessor_type == "processor":
                # Multimodal models use processor
                if self.load_processor():
                    results["preprocessor"] = self.save_processor(output_dir)
                    logger.info(f"Saved processor for multimodal model")
            elif preprocessor_type == "tokenizer":
                # Text models use tokenizer
                if self.load_tokenizer():
                    results["preprocessor"] = self.save_tokenizer(output_dir)
                    logger.info(f"Saved tokenizer for text model")
            elif preprocessor_type == "image_processor":
                # Vision models use image processor
                if self.load_image_processor():
                    results["preprocessor"] = self.save_image_processor(output_dir)
                    logger.info(f"Saved image processor for vision model")
            elif preprocessor_type == "feature_extractor":
                # Audio models use feature extractor
                if self.load_feature_extractor():
                    results["preprocessor"] = self.save_feature_extractor(output_dir)
                    logger.info(f"Saved feature extractor for audio model")
            else:
                logger.warning(f"Could not detect preprocessor type for {self.model_name_or_path}")
        
        # For backward compatibility, also set 'tokenizer' key
        if results["preprocessor_type"] == "tokenizer":
            results["tokenizer"] = results["preprocessor"]
        
        return results
    
    @staticmethod
    def create_minimal_config(model_type: str, 
                             hidden_size: int = 768,
                             num_attention_heads: int = 12,
                             num_hidden_layers: int = 12,
                             vocab_size: int = 30522,
                             **kwargs) -> dict[str, Any]:
        """
        Create a minimal config dict for cases where we can't load from HuggingFace.
        
        Args:
            model_type: Type of model (e.g., 'bert', 'gpt2', 'roberta')
            hidden_size: Hidden dimension size
            num_attention_heads: Number of attention heads
            num_hidden_layers: Number of transformer layers
            vocab_size: Vocabulary size
            **kwargs: Additional config parameters
            
        Returns:
            Minimal config dictionary
        """
        base_config = {
            "model_type": model_type,
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": num_hidden_layers,
            "vocab_size": vocab_size,
            "architectures": [f"{model_type.title()}Model"],
            "torch_dtype": "float32",
        }
        
        # Model-specific defaults
        if model_type == "bert":
            base_config.update({
                "intermediate_size": hidden_size * 4,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1,
                "max_position_embeddings": 512,
                "type_vocab_size": 2,
                "initializer_range": 0.02,
                "layer_norm_eps": 1e-12,
                "pad_token_id": 0,
            })
        elif model_type == "gpt2":
            base_config.update({
                "n_embd": hidden_size,
                "n_head": num_attention_heads,
                "n_layer": num_hidden_layers,
                "n_positions": 1024,
                "n_ctx": 1024,
                "activation_function": "gelu_new",
                "resid_pdrop": 0.1,
                "embd_pdrop": 0.1,
                "attn_pdrop": 0.1,
                "initializer_range": 0.02,
                "layer_norm_epsilon": 1e-5,
            })
        elif model_type == "roberta":
            base_config.update({
                "intermediate_size": hidden_size * 4,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1,
                "max_position_embeddings": 514,
                "type_vocab_size": 1,
                "initializer_range": 0.02,
                "layer_norm_eps": 1e-5,
                "pad_token_id": 1,
                "bos_token_id": 0,
                "eos_token_id": 2,
            })
        
        # Merge any additional kwargs
        base_config.update(kwargs)
        
        return base_config
    
    @staticmethod
    def save_minimal_config(output_dir: str | Path, config_dict: dict[str, Any]) -> bool:
        """
        Save a minimal config dictionary as config.json.
        
        Args:
            output_dir: Directory to save the config
            config_dict: Config dictionary to save
            
        Returns:
            True if successful, False otherwise
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            config_path = output_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Saved minimal config to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save minimal config: {e}")
            return False