"""
Task detection from model architectures and configurations.

This module automatically detects the appropriate task for a model based on its
architecture and configuration.
"""

from typing import Optional, List, Dict, Any
from .patterns import ARCHITECTURE_TO_TASK, SPECIAL_MODEL_FAMILIES


class TaskDetector:
    """Detect tasks from model configurations and architectures."""
    
    # Extended task mappings for more specific patterns
    TASK_PATTERNS = {
        # Text tasks
        "SequenceClassification": "text-classification",
        "TokenClassification": "token-classification",
        "QuestionAnswering": "question-answering",
        "MultipleChoice": "multiple-choice",
        "NextSentencePrediction": "next-sentence-prediction",
        
        # Generation tasks  
        "CausalLM": "text-generation",
        "ConditionalGeneration": "text2text-generation",
        "Seq2SeqLM": "text2text-generation",
        "MaskedLM": "fill-mask",
        
        # Vision tasks
        "ImageClassification": "image-classification",
        "ObjectDetection": "object-detection",
        "ImageSegmentation": "image-segmentation",
        "SemanticSegmentation": "semantic-segmentation",
        "InstanceSegmentation": "instance-segmentation",
        "PanopticSegmentation": "panoptic-segmentation",
        "DepthEstimation": "depth-estimation",
        
        # Audio tasks
        "AudioClassification": "audio-classification",
        "AudioFrameClassification": "audio-frame-classification",
        "CTC": "automatic-speech-recognition",
        "SpeechSeq2Seq": "automatic-speech-recognition",
        "AudioXVector": "audio-xvector",
        
        # Multimodal tasks
        "VisionTextDualEncoder": "feature-extraction",
        "ImageTextRetrieval": "image-text-retrieval",
        "ZeroShotImageClassification": "zero-shot-image-classification",
        "ZeroShotObjectDetection": "zero-shot-object-detection",
        "DocumentQuestionAnswering": "document-question-answering",
        "VisualQuestionAnswering": "visual-question-answering",
        
        # Video tasks
        "VideoClassification": "video-classification",
        "VideoFrameClassification": "video-classification",
        
        # Other tasks
        "Embedding": "feature-extraction",
        "Regression": "regression",
        "Retrieval": "feature-extraction",
    }
    
    # Model type to default task mapping
    MODEL_TYPE_TO_DEFAULT_TASK = {
        # Text encoder models
        "bert": "feature-extraction",
        "roberta": "feature-extraction",
        "albert": "feature-extraction",
        "electra": "feature-extraction",
        "distilbert": "feature-extraction",
        "xlm": "feature-extraction",
        "xlm-roberta": "feature-extraction",
        "deberta": "feature-extraction",
        "deberta-v2": "feature-extraction",
        "mpnet": "feature-extraction",
        
        # Decoder models
        "gpt2": "text-generation",
        "gpt-neo": "text-generation",
        "gpt-j": "text-generation",
        "gptj": "text-generation",
        "opt": "text-generation",
        "bloom": "text-generation",
        "llama": "text-generation",
        "mistral": "text-generation",
        "falcon": "text-generation",
        "mpt": "text-generation",
        "phi": "text-generation",
        "qwen": "text-generation",
        "qwen2": "text-generation",
        "gemma": "text-generation",
        "stablelm": "text-generation",
        
        # Encoder-decoder models
        "t5": "text2text-generation",
        "bart": "text2text-generation",
        "mbart": "text2text-generation",
        "pegasus": "text2text-generation",
        "marian": "translation",
        "mt5": "text2text-generation",
        "blenderbot": "conversational",
        "blenderbot-small": "conversational",
        "led": "text2text-generation",
        "longformer": "feature-extraction",
        "longt5": "text2text-generation",
        
        # Vision models
        "vit": "image-classification",
        "deit": "image-classification", 
        "beit": "image-classification",
        "swin": "image-classification",
        "convnext": "image-classification",
        "resnet": "image-classification",
        "efficientnet": "image-classification",
        "regnet": "image-classification",
        "mobilenet": "image-classification",
        "densenet": "image-classification",
        "dinov2": "feature-extraction",
        
        # Object detection models
        "detr": "object-detection",
        "yolos": "object-detection",
        "conditional-detr": "object-detection",
        "deformable-detr": "object-detection",
        "table-transformer": "object-detection",
        "deta": "object-detection",
        
        # Segmentation models
        "sam": "image-segmentation",
        "maskformer": "image-segmentation",
        "mask2former": "image-segmentation",
        "oneformer": "image-segmentation",
        "segformer": "semantic-segmentation",
        "upernet": "semantic-segmentation",
        
        # Audio models
        "wav2vec2": "automatic-speech-recognition",
        "whisper": "automatic-speech-recognition",
        "hubert": "automatic-speech-recognition",
        "wavlm": "automatic-speech-recognition",
        "unispeech": "automatic-speech-recognition",
        "seamless_m4t": "automatic-speech-recognition",
        "speecht5": "text-to-speech",
        
        # Multimodal models
        "clip": "feature-extraction",
        "align": "feature-extraction",
        "blip": "visual-question-answering",
        "blip-2": "visual-question-answering",
        "bridgetower": "feature-extraction",
        "chinese_clip": "feature-extraction",
        "flava": "feature-extraction",
        "git": "image-to-text",
        "groupvit": "zero-shot-image-classification",
        "owlvit": "zero-shot-object-detection",
        "owlv2": "zero-shot-object-detection",
        "x-clip": "feature-extraction",
        
        # Document models
        "layoutlm": "token-classification",
        "layoutlmv2": "token-classification",
        "layoutlmv3": "token-classification",
        "layoutxlm": "token-classification",
        "markuplm": "token-classification",
        "donut": "document-question-answering",
        "nougat": "image-to-text",
        
        # Vision-language generation
        "vision-encoder-decoder": "image-to-text",
        "trocr": "image-to-text",
        "pix2struct": "visual-question-answering",
    }
    
    @classmethod
    def detect_from_config(cls, config: Dict[str, Any]) -> str:
        """
        Detect task from model configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Detected task name
        """
        # Check architectures field first (most reliable)
        if "architectures" in config and config["architectures"]:
            task = cls._detect_from_architecture(config["architectures"])
            if task:
                return task
        
        # Check model type
        if "model_type" in config:
            task = cls._detect_from_model_type(config["model_type"])
            if task:
                return task
        
        # Check for task-specific configuration hints
        task = cls._detect_from_config_hints(config)
        if task:
            return task
        
        # Default to feature extraction
        return "feature-extraction"
    
    @classmethod
    def _detect_from_architecture(cls, architectures: List[str]) -> Optional[str]:
        """Detect task from architecture names."""
        if not architectures:
            return None
        
        # Get the first architecture
        arch = architectures[0]
        
        # Check exact patterns from ARCHITECTURE_TO_TASK
        for pattern, task in ARCHITECTURE_TO_TASK.items():
            if pattern in arch:
                return task
        
        # Check our extended patterns
        for pattern, task in cls.TASK_PATTERNS.items():
            if pattern in arch:
                return task
        
        return None
    
    @classmethod
    def _detect_from_model_type(cls, model_type: str) -> Optional[str]:
        """Detect task from model type."""
        model_type_lower = model_type.lower()
        
        # Direct lookup
        if model_type_lower in cls.MODEL_TYPE_TO_DEFAULT_TASK:
            return cls.MODEL_TYPE_TO_DEFAULT_TASK[model_type_lower]
        
        # Check if it's a variant (e.g., "microsoft/deberta-v3" -> "deberta")
        for base_type in cls.MODEL_TYPE_TO_DEFAULT_TASK:
            if base_type in model_type_lower:
                return cls.MODEL_TYPE_TO_DEFAULT_TASK[base_type]
        
        return None
    
    @classmethod
    def _detect_from_config_hints(cls, config: Dict[str, Any]) -> Optional[str]:
        """Detect task from configuration hints."""
        
        # Check for task-specific parameters
        hints = {
            "num_labels": "classification",
            "id2label": "classification",
            "label2id": "classification",
            "is_encoder_decoder": "seq2seq",
            "is_decoder": "generation",
            "tie_word_embeddings": "generation",
            "vocab_size": "text",
            "image_size": "vision",
            "num_channels": "vision",
            "sample_rate": "audio",
            "num_mel_bins": "audio",
        }
        
        detected_modalities = set()
        for hint_key, modality in hints.items():
            if hint_key in config:
                detected_modalities.add(modality)
        
        # Determine task based on detected hints
        if "classification" in detected_modalities:
            if "vision" in detected_modalities:
                return "image-classification"
            elif "audio" in detected_modalities:
                return "audio-classification"
            else:
                return "text-classification"
        
        if "seq2seq" in detected_modalities:
            return "text2text-generation"
        
        if "generation" in detected_modalities:
            return "text-generation"
        
        if "vision" in detected_modalities:
            return "image-classification"
        
        if "audio" in detected_modalities:
            return "automatic-speech-recognition"
        
        return None
    
    @classmethod
    def get_task_family(cls, task: str) -> str:
        """
        Get the family of a task (e.g., classification, generation, etc.).
        
        Args:
            task: Task name
            
        Returns:
            Task family name
        """
        families = {
            "classification": [
                "text-classification", "token-classification", "image-classification",
                "audio-classification", "video-classification", "zero-shot-image-classification",
                "audio-frame-classification"
            ],
            "generation": [
                "text-generation", "text2text-generation", "image-to-text",
                "automatic-speech-recognition", "text-to-speech", "translation"
            ],
            "question-answering": [
                "question-answering", "visual-question-answering",
                "document-question-answering"
            ],
            "feature-extraction": [
                "feature-extraction", "image-text-retrieval", "audio-xvector"
            ],
            "detection": [
                "object-detection", "zero-shot-object-detection"
            ],
            "segmentation": [
                "image-segmentation", "semantic-segmentation",
                "instance-segmentation", "panoptic-segmentation"
            ],
            "other": [
                "fill-mask", "next-sentence-prediction", "multiple-choice",
                "regression", "depth-estimation", "conversational"
            ]
        }
        
        for family, tasks in families.items():
            if task in tasks:
                return family
        
        return "unknown"
    
    @classmethod
    def requires_past_key_values(cls, task: str, model_type: Optional[str] = None) -> bool:
        """
        Check if a task/model requires past key values for generation.
        
        Args:
            task: Task name
            model_type: Optional model type
            
        Returns:
            Whether past key values are needed
        """
        # Tasks that typically use past key values
        generation_tasks = [
            "text-generation", "text2text-generation",
            "conversational", "translation"
        ]
        
        if task not in generation_tasks:
            return False
        
        # Models that definitely use past key values
        if model_type:
            models_with_pkv = [
                "gpt2", "gpt-neo", "gpt-j", "opt", "bloom",
                "llama", "mistral", "falcon", "mpt", "phi",
                "qwen", "gemma", "stablelm", "bart", "mbart",
                "t5", "mt5", "pegasus"
            ]
            
            model_type_lower = model_type.lower()
            return any(m in model_type_lower for m in models_with_pkv)
        
        return task in generation_tasks