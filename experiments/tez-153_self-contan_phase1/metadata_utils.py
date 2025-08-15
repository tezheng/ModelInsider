"""
Metadata utilities for self-contained ONNX models.

This module provides functionality to extract, embed, and read metadata
from ONNX models, enabling self-contained deployment with zero external dependencies.
"""

import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import onnx
from onnx import helper

logger = logging.getLogger(__name__)


class MetadataDiscovery:
    """Discover and aggregate metadata from HuggingFace models and processors."""

# Keep alias for backward compatibility
MetadataExtractor = MetadataDiscovery
    
    def discover_from_hf(
        self,
        model: Any,
        processor: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Discover and aggregate metadata from HuggingFace model and processor.
        Honors all original configuration fields without filtering.
        
        Args:
            model: HuggingFace model instance
            processor: HuggingFace processor (tokenizer/image_processor/feature_extractor)
            
        Returns:
            Dictionary containing discovered metadata with complete configs
        """
        metadata = {}
        
        # Discover complete model configuration
        if hasattr(model, 'config'):
            model_config = self._discover_model_config(model.config)
            metadata['hf_model_config'] = model_config
            
            # Infer task type from model config
            task_type = self._infer_task_type(model.config)
            if task_type:
                metadata['hf_pipeline_task'] = task_type
        
        # Discover complete processor configuration
        if processor:
            proc_metadata = self._discover_processor_config(processor)
            metadata.update(proc_metadata)
        
        # Add metadata version and export info
        metadata['hf_metadata_version'] = "2.0"
        
        # Add transformers version if available
        try:
            import transformers
            metadata['hf_transformers_version'] = transformers.__version__
        except ImportError:
            pass
        
        # Add export timestamp
        from datetime import datetime
        metadata['hf_export_timestamp'] = datetime.utcnow().isoformat() + "Z"
        
        return metadata
    
    def _discover_model_config(self, config: Any) -> Dict[str, Any]:
        """Discover complete model configuration - honor all fields.
        
        Core Principle: We preserve all configuration fields exactly as
        provided by HuggingFace, trusting that every field exists for a reason.
        """
        # Always return complete configuration
        return config.to_dict() if hasattr(config, 'to_dict') else dict(config)
    
    def _discover_processor_config(self, processor: Any) -> Dict[str, Any]:
        """Discover processor configuration - preserve complete original.
        
        Core Principle: Honor all processor configuration fields without
        filtering or judging which are "essential".
        """
        metadata = {}
        
        # Handle tokenizer - preserve complete config
        if hasattr(processor, 'tokenizer'):
            tokenizer = processor.tokenizer
            original_config = tokenizer.to_dict() if hasattr(tokenizer, 'to_dict') else {}
            metadata['hf_tokenizer_config'] = original_config
            metadata['feature_engineering.type'] = 'tokenizer'
        
        # Handle image processor - preserve complete config
        elif hasattr(processor, 'image_processor'):
            image_proc = processor.image_processor
            original_config = image_proc.to_dict() if hasattr(image_proc, 'to_dict') else {}
            metadata['hf_image_processor_config'] = original_config
            metadata['feature_engineering.type'] = 'image_processor'
        
        # Handle feature extractor (audio) - preserve complete config
        elif hasattr(processor, 'feature_extractor'):
            feat_ext = processor.feature_extractor
            original_config = feat_ext.to_dict() if hasattr(feat_ext, 'to_dict') else {}
            metadata['hf_feature_extractor_config'] = original_config
            metadata['feature_engineering.type'] = 'feature_extractor'
        
        return metadata
    
    def _infer_task_type(self, config: Any) -> Optional[str]:
        """Infer pipeline task type from model configuration."""
        # Check architectures field
        architectures = getattr(config, 'architectures', [])
        if architectures:
            arch_name = architectures[0].lower()
            
            # Text classification
            if 'sequenceclassification' in arch_name:
                return 'text-classification'
            elif 'tokenclassification' in arch_name:
                return 'token-classification'
            elif 'questionanswering' in arch_name:
                return 'question-answering'
            elif 'maskedlm' in arch_name or 'fill-mask' in arch_name:
                return 'fill-mask'
            
            # Generation tasks
            elif 'causallm' in arch_name or 'generation' in arch_name:
                return 'text-generation'
            elif 'seq2seq' in arch_name:
                # Could be translation, summarization, etc.
                if hasattr(config, 'task_specific_params'):
                    tasks = config.task_specific_params
                    if 'translation' in str(tasks):
                        return 'translation'
                    elif 'summarization' in str(tasks):
                        return 'summarization'
                return 'text2text-generation'
            
            # Vision tasks
            elif 'imageclassification' in arch_name:
                return 'image-classification'
            elif 'objectdetection' in arch_name:
                return 'object-detection'
            elif 'imagesegmentation' in arch_name or 'segmentation' in arch_name:
                return 'image-segmentation'
            
            # Audio tasks
            elif 'audioclassification' in arch_name:
                return 'audio-classification'
            elif 'ctc' in arch_name or 'speechrecognition' in arch_name:
                return 'automatic-speech-recognition'
            
            # Multimodal
            elif 'clip' in arch_name:
                return 'zero-shot-image-classification'
            elif 'visionencoderdecoder' in arch_name:
                return 'image-to-text'
        
        # Fallback: check model_type
        model_type = getattr(config, 'model_type', '').lower()
        if 'bert' in model_type and hasattr(config, 'num_labels'):
            return 'text-classification'
        elif 'gpt' in model_type:
            return 'text-generation'
        elif 'vit' in model_type:
            return 'image-classification'
        
        return None


class ONNXMetadataEmbedder:
    """Embed metadata into ONNX models."""
    
    def embed(
        self,
        onnx_path: Union[str, Path],
        metadata: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Embed metadata into ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            metadata: Metadata dictionary to embed
            output_path: Output path (if None, overwrites input)
        """
        # Load ONNX model
        model = onnx.load(str(onnx_path))
        
        # Clear existing metadata_props
        del model.metadata_props[:]
        
        # Add new metadata
        for key, value in metadata.items():
            if value is not None:
                prop = model.metadata_props.add()
                prop.key = key
                
                # Convert value to string
                if isinstance(value, (dict, list)):
                    prop.value = json.dumps(value, separators=(',', ':'))
                else:
                    prop.value = str(value)
        
        # Save model
        output = output_path or onnx_path
        onnx.save(model, str(output))
        
        logger.info(f"Embedded {len(metadata)} metadata entries into {output}")
    
    def embed_with_compression(
        self,
        onnx_path: Union[str, Path],
        metadata: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
        compress_threshold: int = 10240  # 10KB
    ) -> None:
        """
        Embed metadata with automatic compression for large configs.
        
        Args:
            onnx_path: Path to ONNX model
            metadata: Metadata dictionary to embed
            output_path: Output path (if None, overwrites input)
            compress_threshold: Compress values larger than this (bytes)
        """
        import gzip
        import base64
        
        # Load ONNX model
        model = onnx.load(str(onnx_path))
        
        # Clear existing metadata_props
        del model.metadata_props[:]
        
        # Add new metadata with compression
        for key, value in metadata.items():
            if value is not None:
                prop = model.metadata_props.add()
                prop.key = key
                
                # Convert value to string
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, separators=(',', ':'))
                else:
                    value_str = str(value)
                
                # Compress if large
                if len(value_str) > compress_threshold:
                    compressed = gzip.compress(value_str.encode('utf-8'))
                    prop.value = f"gzip:{base64.b64encode(compressed).decode('ascii')}"
                    logger.debug(f"Compressed {key}: {len(value_str)} -> {len(prop.value)} bytes")
                else:
                    prop.value = value_str
        
        # Save model
        output = output_path or onnx_path
        onnx.save(model, str(output))
        
        logger.info(f"Embedded {len(metadata)} metadata entries into {output}")


class ONNXMetadataReader:
    """Read metadata from ONNX models."""
    
    def read(self, onnx_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Read all metadata from ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            Dictionary containing all metadata
        """
        model = onnx.load(str(onnx_path))
        metadata = {}
        
        for prop in model.metadata_props:
            key = prop.key
            value = prop.value
            
            # Handle compressed values
            if value.startswith('gzip:'):
                import gzip
                import base64
                compressed = base64.b64decode(value[5:])
                value = gzip.decompress(compressed).decode('utf-8')
            
            # Try to parse JSON
            if value.startswith(('{', '[')):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass  # Keep as string
            
            metadata[key] = value
        
        return metadata
    
    def read_feature_engineering(self, onnx_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Read feature engineering metadata specifically.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            Dictionary with feature_engineering and image metadata
        """
        all_metadata = self.read(onnx_path)
        
        feature_engineering = {}
        image_metadata = {}
        
        for key, value in all_metadata.items():
            if key.startswith('feature_engineering.'):
                feature_key = key[20:]  # Remove prefix
                feature_engineering[feature_key] = value
            elif key.startswith('Image.'):
                image_metadata[key] = value
        
        return {
            'feature_engineering': feature_engineering,
            'image': image_metadata,
            'task': all_metadata.get('hf_pipeline_task'),
            'model_config': all_metadata.get('hf_model_config'),
            'version': all_metadata.get('hf_metadata_version')
        }


class MetadataValidator:
    """Validate metadata completeness and compatibility."""
    
    def validate(self, metadata: Dict[str, Any]) -> bool:
        """
        Validate metadata completeness.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Check required fields
        if not metadata:
            raise ValueError("Metadata is empty")
        
        # Check version
        if 'hf_metadata_version' not in metadata:
            raise ValueError("Missing metadata version")
        
        version = metadata['hf_metadata_version']
        if version not in ['1.0', '2.0']:
            raise ValueError(f"Unsupported metadata version: {version}")
        
        # Check model config
        if 'hf_model_config' in metadata:
            config = metadata['hf_model_config']
            if not isinstance(config, dict):
                raise ValueError("Model config must be a dictionary")
        
        # Check feature engineering
        if 'feature_engineering.type' in metadata:
            fe_type = metadata['feature_engineering.type']
            if fe_type not in ['tokenizer', 'image_processor', 'feature_extractor']:
                raise ValueError(f"Unknown feature engineering type: {fe_type}")
            
            if 'feature_engineering.config' not in metadata:
                raise ValueError(f"Missing config for feature engineering type: {fe_type}")
        
        return True


class MetadataManager:
    """Central utility for metadata operations."""
    
    def __init__(self):
        self.discovery = MetadataDiscovery()
        self.embedder = ONNXMetadataEmbedder()
        self.reader = ONNXMetadataReader()
        self.validator = MetadataValidator()
    
    def process_model(
        self,
        model: Any,
        processor: Optional[Any],
        onnx_path: Union[str, Path],
        compress: bool = True
    ) -> Dict[str, Any]:
        """
        Main workflow for embedding metadata into ONNX model.
        Preserves complete configurations without filtering.
        
        Args:
            model: HuggingFace model instance
            processor: HuggingFace processor
            onnx_path: Path to exported ONNX model
            compress: Whether to compress large metadata
            
        Returns:
            Embedded metadata dictionary with complete configs
        """
        # Discover complete metadata from source (no filtering)
        metadata = self.discovery.discover_from_hf(model, processor)
        
        # Validate completeness and compatibility
        self.validator.validate(metadata)
        
        # Embed into ONNX
        if compress:
            self.embedder.embed_with_compression(onnx_path, metadata)
        else:
            self.embedder.embed(onnx_path, metadata)
        
        logger.info(f"Successfully embedded complete metadata into {onnx_path}")
        
        return metadata
    
    def read_and_validate(self, onnx_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Read and validate metadata from ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            Validated metadata dictionary
        """
        metadata = self.reader.read(onnx_path)
        self.validator.validate(metadata)
        return metadata