"""
AutoModel-like interface for ONNX models using Optimum.

This module provides a familiar AutoModel interface for loading ONNX models
exported by ModelExport, automatically detecting the appropriate task and
loading the correct ORTModel class.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

from transformers import AutoConfig

logger = logging.getLogger(__name__)


class AutoModelForONNX:
    """
    AutoModel-like interface for ONNX models.
    
    Automatically detects the model task and loads the appropriate
    ORTModel class from Optimum.
    
    Examples:
        >>> # Load any ONNX model automatically
        >>> model = AutoModelForONNX.from_pretrained("path/to/exported/model")
        >>> 
        >>> # Specify task explicitly
        >>> model = AutoModelForONNX.from_pretrained(
        ...     "path/to/model",
        ...     task="text-classification"
        ... )
    """
    
    # Mapping from model types to common tasks
    MODEL_TYPE_TO_TASKS = {
        # Text models
        "bert": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "roberta": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "distilbert": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "albert": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "electra": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "xlm-roberta": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        
        # Generative models
        "gpt2": ["text-generation", "feature-extraction"],
        "gpt-neo": ["text-generation", "feature-extraction"],
        "gpt-j": ["text-generation", "feature-extraction"],
        "opt": ["text-generation", "feature-extraction"],
        "bloom": ["text-generation", "feature-extraction"],
        "llama": ["text-generation", "feature-extraction"],
        "mistral": ["text-generation", "feature-extraction"],
        
        # Encoder-decoder models
        "t5": ["text2text-generation", "summarization", "translation"],
        "bart": ["text2text-generation", "summarization", "feature-extraction"],
        "mbart": ["text2text-generation", "translation"],
        "pegasus": ["summarization"],
        "marian": ["translation"],
        
        # Vision models
        "vit": ["image-classification", "feature-extraction"],
        "deit": ["image-classification", "feature-extraction"],
        "beit": ["image-classification", "feature-extraction"],
        "swin": ["image-classification", "feature-extraction"],
        "convnext": ["image-classification", "feature-extraction"],
        "resnet": ["image-classification", "feature-extraction"],
        
        # Multimodal models
        "clip": ["zero-shot-image-classification", "feature-extraction"],
        "layoutlm": ["document-question-answering", "token-classification"],
        "layoutlmv2": ["document-question-answering", "token-classification"],
        "layoutlmv3": ["document-question-answering", "token-classification"],
        
        # Audio models
        "wav2vec2": ["automatic-speech-recognition", "audio-classification", "feature-extraction"],
        "whisper": ["automatic-speech-recognition", "feature-extraction"],
        "hubert": ["automatic-speech-recognition", "feature-extraction"],
        "wavlm": ["automatic-speech-recognition", "feature-extraction"],
    }
    
    # Mapping from tasks to ORTModel classes
    TASK_TO_ORT_MODEL = {
        # Text tasks
        "text-classification": "ORTModelForSequenceClassification",
        "token-classification": "ORTModelForTokenClassification",
        "question-answering": "ORTModelForQuestionAnswering",
        "feature-extraction": "ORTModelForFeatureExtraction",
        
        # Generative tasks
        "text-generation": "ORTModelForCausalLM",
        "text2text-generation": "ORTModelForSeq2SeqLM",
        "summarization": "ORTModelForSeq2SeqLM",
        "translation": "ORTModelForSeq2SeqLM",
        
        # Vision tasks
        "image-classification": "ORTModelForImageClassification",
        "zero-shot-image-classification": "ORTModel",  # Generic for CLIP
        
        # Audio tasks
        "automatic-speech-recognition": "ORTModelForSpeechSeq2Seq",
        "audio-classification": "ORTModelForAudioClassification",
        
        # Document tasks
        "document-question-answering": "ORTModelForQuestionAnswering",
    }
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        task: Optional[str] = None,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[Any] = None,
        provider_options: Optional[dict] = None,
        **kwargs
    ) -> Any:
        """
        Load an ONNX model with automatic task detection.
        
        Args:
            model_path: Path to the exported ONNX model directory
            task: Optional task specification. If None, auto-detects from config
            provider: ONNX Runtime execution provider (CPU, CUDA, etc.)
            session_options: ONNX Runtime session options
            provider_options: Provider-specific options
            **kwargs: Additional arguments passed to ORTModel
            
        Returns:
            Appropriate ORTModel instance for the detected/specified task
            
        Raises:
            ValueError: If task cannot be detected or is not supported
            ImportError: If Optimum is not installed
        """
        model_path = Path(model_path)
        
        # Check if path exists
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # Load config to detect model type
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise ValueError(
                f"config.json not found in {model_path}. "
                "Make sure the model was exported with Optimum compatibility enabled."
            )
        
        config = AutoConfig.from_pretrained(model_path)
        
        # Detect task if not specified
        if task is None:
            task = cls._detect_task(config, model_path)
            logger.info(f"Auto-detected task: {task}")
        
        # Get the appropriate ORTModel class
        ort_model_class = cls._get_ort_model_class(task)
        
        # Import the class dynamically
        try:
            from optimum.onnxruntime import ORTModel
            
            # Import specific class
            if ort_model_class == "ORTModel":
                model_class = ORTModel
            else:
                # Dynamic import of specific model class
                import optimum.onnxruntime
                model_class = getattr(optimum.onnxruntime, ort_model_class)
        except ImportError as e:
            raise ImportError(
                "Optimum is not installed. Please install with: "
                "pip install optimum[onnxruntime]"
            ) from e
        except AttributeError as e:
            raise ValueError(f"ORTModel class {ort_model_class} not found in Optimum") from e
        
        # Set up provider configuration
        if provider == "CUDAExecutionProvider":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = [provider]
        
        # Load the model
        logger.info(f"Loading {ort_model_class} from {model_path}")
        
        # Build kwargs for model loading
        load_kwargs = {
            "provider": providers[0] if len(providers) == 1 else providers,
            **kwargs
        }
        
        if session_options is not None:
            load_kwargs["session_options"] = session_options
        if provider_options is not None:
            load_kwargs["provider_options"] = provider_options
        
        # Load and return the model
        model = model_class.from_pretrained(model_path, **load_kwargs)
        
        # Add task info as attribute
        model.task = task
        
        return model
    
    @classmethod
    def _detect_task(cls, config: AutoConfig, model_path: Path) -> str:
        """
        Detect the task from model configuration.
        
        Args:
            config: Model configuration
            model_path: Path to model directory
            
        Returns:
            Detected task name
            
        Raises:
            ValueError: If task cannot be detected
        """
        model_type = config.model_type.lower()
        
        # Check for explicit task in config
        if hasattr(config, "task"):
            return config.task
        
        # Check for task-specific architectures
        architectures = getattr(config, "architectures", [])
        if architectures:
            arch = architectures[0].lower()
            
            # Classification models
            if "forsequenceclassification" in arch:
                return "text-classification"
            elif "fortokenclassification" in arch:
                return "token-classification"
            elif "forquestionanswering" in arch:
                return "question-answering"
            elif "forimageclassification" in arch:
                return "image-classification"
            
            # Generative models
            elif "forcausallm" in arch or "lmhead" in arch:
                return "text-generation"
            elif "forconditionalgeneration" in arch:
                return "text2text-generation"
            
            # Feature extraction (fallback)
            elif "model" in arch:
                return "feature-extraction"
        
        # Use model type mapping
        if model_type in cls.MODEL_TYPE_TO_TASKS:
            # Return the first (most common) task for this model type
            return cls.MODEL_TYPE_TO_TASKS[model_type][0]
        
        # Check for task hints in model files
        task_hints_file = model_path / "task.txt"
        if task_hints_file.exists():
            return task_hints_file.read_text().strip()
        
        # Default fallback
        logger.warning(
            f"Could not detect task for model type '{model_type}'. "
            "Defaulting to 'feature-extraction'. "
            "Consider specifying the task explicitly."
        )
        return "feature-extraction"
    
    @classmethod
    def _get_ort_model_class(cls, task: str) -> str:
        """
        Get the appropriate ORTModel class name for a task.
        
        Args:
            task: Task name
            
        Returns:
            ORTModel class name
            
        Raises:
            ValueError: If task is not supported
        """
        if task not in cls.TASK_TO_ORT_MODEL:
            supported_tasks = ", ".join(sorted(cls.TASK_TO_ORT_MODEL.keys()))
            raise ValueError(
                f"Task '{task}' is not supported. "
                f"Supported tasks: {supported_tasks}"
            )
        
        return cls.TASK_TO_ORT_MODEL[task]
    
    @classmethod
    def list_supported_tasks(cls) -> list[str]:
        """
        List all supported tasks.
        
        Returns:
            List of supported task names
        """
        return sorted(cls.TASK_TO_ORT_MODEL.keys())
    
    @classmethod
    def list_supported_model_types(cls) -> list[str]:
        """
        List all supported model types.
        
        Returns:
            List of supported model type names
        """
        return sorted(cls.MODEL_TYPE_TO_TASKS.keys())