"""
AutoModel-like interface for ONNX models using Optimum.

This module provides a familiar AutoModel interface for loading ONNX models
exported by ModelExport, automatically detecting the appropriate task and
loading the correct ORTModel class.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
        # === BERT Family ===
        "bert": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "roberta": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "distilbert": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "albert": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "electra": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "xlm-roberta": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "deberta": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "deberta-v2": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "deberta-v3": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "camembert": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "flaubert": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "xlm": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "squeezbert": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "mobilebert": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "funnel": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "convbert": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "ibert": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "luke": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "rembert": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "ernie": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "ernie-m": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "roformer": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "nezha": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "qdqbert": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "fnet": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "megatron-bert": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "big_bird": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "bigbird": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "longformer": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "reformer": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "canine": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "yoso": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "nystromformer": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        
        # === GPT/Causal LM Family ===
        "gpt2": ["text-generation", "feature-extraction"],
        "gpt-neo": ["text-generation", "feature-extraction"],
        "gpt-neox": ["text-generation", "feature-extraction"],
        "gpt-neox-japanese": ["text-generation", "feature-extraction"],
        "gpt-j": ["text-generation", "feature-extraction"],
        "gptj": ["text-generation", "feature-extraction"],
        "opt": ["text-generation", "feature-extraction"],
        "bloom": ["text-generation", "feature-extraction"],
        "bloomz": ["text-generation", "feature-extraction"],
        "llama": ["text-generation", "feature-extraction"],
        "llama2": ["text-generation", "feature-extraction"],
        "codellama": ["text-generation", "feature-extraction"],
        "mistral": ["text-generation", "feature-extraction"],
        "mixtral": ["text-generation", "feature-extraction"],
        "falcon": ["text-generation", "feature-extraction"],
        "rwkv": ["text-generation", "feature-extraction"],
        "mpt": ["text-generation", "feature-extraction"],
        "stablelm": ["text-generation", "feature-extraction"],
        "stablelm2": ["text-generation", "feature-extraction"],
        "phi": ["text-generation", "feature-extraction"],
        "phi-2": ["text-generation", "feature-extraction"],
        "phi-3": ["text-generation", "feature-extraction"],
        "qwen": ["text-generation", "feature-extraction"],
        "qwen2": ["text-generation", "feature-extraction"],
        "yi": ["text-generation", "feature-extraction"],
        "gemma": ["text-generation", "feature-extraction"],
        "gemma2": ["text-generation", "feature-extraction"],
        "starcoder": ["text-generation", "feature-extraction"],
        "starcoder2": ["text-generation", "feature-extraction"],
        "codegen": ["text-generation", "feature-extraction"],
        "codegen2": ["text-generation", "feature-extraction"],
        "santacoder": ["text-generation", "feature-extraction"],
        "gpt_bigcode": ["text-generation", "feature-extraction"],
        "gpt-sw3": ["text-generation", "feature-extraction"],
        "xglm": ["text-generation", "feature-extraction"],
        "ctrl": ["text-generation", "feature-extraction"],
        "open-llama": ["text-generation", "feature-extraction"],
        "openai-gpt": ["text-generation", "feature-extraction"],
        "persimmon": ["text-generation", "feature-extraction"],
        "fuyu": ["text-generation", "feature-extraction"],
        "olmo": ["text-generation", "feature-extraction"],
        "mamba": ["text-generation", "feature-extraction"],
        "jamba": ["text-generation", "feature-extraction"],
        "recurrentgemma": ["text-generation", "feature-extraction"],
        "cohere": ["text-generation", "feature-extraction"],
        "dbrx": ["text-generation", "feature-extraction"],
        "granite": ["text-generation", "feature-extraction"],
        "graniitemoe": ["text-generation", "feature-extraction"],
        
        # === Encoder-Decoder Models ===
        "t5": ["text2text-generation", "summarization", "translation", "feature-extraction"],
        "t5-v1_1": ["text2text-generation", "summarization", "translation", "feature-extraction"],
        "mt5": ["text2text-generation", "summarization", "translation", "feature-extraction"],
        "byt5": ["text2text-generation", "summarization", "translation", "feature-extraction"],
        "ul2": ["text2text-generation", "summarization", "translation", "feature-extraction"],
        "flan-t5": ["text2text-generation", "summarization", "translation", "feature-extraction"],
        "flan-ul2": ["text2text-generation", "summarization", "translation", "feature-extraction"],
        "switch-transformers": ["text2text-generation", "summarization", "translation", "feature-extraction"],
        "bart": ["text2text-generation", "summarization", "feature-extraction"],
        "mbart": ["text2text-generation", "translation", "feature-extraction"],
        "mbart-50": ["text2text-generation", "translation", "feature-extraction"],
        "pegasus": ["summarization", "feature-extraction"],
        "pegasus-x": ["summarization", "feature-extraction"],
        "bigbird-pegasus": ["summarization", "feature-extraction"],
        "led": ["summarization", "feature-extraction"],
        "longt5": ["text2text-generation", "summarization", "translation", "feature-extraction"],
        "encoder-decoder": ["text2text-generation", "summarization", "translation", "feature-extraction"],
        "marian": ["translation"],
        "m2m_100": ["translation"],
        "seamless_m4t": ["translation", "automatic-speech-recognition"],
        "seamless_m4t_v2": ["translation", "automatic-speech-recognition"],
        "nllb": ["translation"],
        "nllb-moe": ["translation"],
        "plbart": ["text2text-generation", "feature-extraction"],
        "blenderbot": ["text-generation", "feature-extraction"],
        "blenderbot-small": ["text-generation", "feature-extraction"],
        "prophetnet": ["text2text-generation", "summarization", "feature-extraction"],
        "rag": ["question-answering", "feature-extraction"],
        
        # === Vision Models ===
        "vit": ["image-classification", "feature-extraction"],
        "vit-mae": ["image-classification", "feature-extraction"],
        "vit-msn": ["image-classification", "feature-extraction"],
        "deit": ["image-classification", "feature-extraction"],
        "beit": ["image-classification", "feature-extraction"],
        "beitv2": ["image-classification", "feature-extraction"],
        "swin": ["image-classification", "feature-extraction"],
        "swinv2": ["image-classification", "feature-extraction"],
        "swin2sr": ["image-to-image", "feature-extraction"],
        "convnext": ["image-classification", "feature-extraction"],
        "convnextv2": ["image-classification", "feature-extraction"],
        "resnet": ["image-classification", "feature-extraction"],
        "regnet": ["image-classification", "feature-extraction"],
        "mobilenet_v1": ["image-classification", "feature-extraction"],
        "mobilenet_v2": ["image-classification", "feature-extraction"],
        "mobilevit": ["image-classification", "feature-extraction"],
        "mobilevitv2": ["image-classification", "feature-extraction"],
        "efficientnet": ["image-classification", "feature-extraction"],
        "efficientformer": ["image-classification", "feature-extraction"],
        "pvt": ["image-classification", "feature-extraction"],
        "pvt_v2": ["image-classification", "feature-extraction"],
        "poolformer": ["image-classification", "feature-extraction"],
        "focalnet": ["image-classification", "feature-extraction"],
        "nat": ["image-classification", "feature-extraction"],
        "dinat": ["image-classification", "feature-extraction"],
        "van": ["image-classification", "feature-extraction"],
        "levit": ["image-classification", "feature-extraction"],
        "cvt": ["image-classification", "feature-extraction"],
        "segformer": ["image-segmentation", "feature-extraction"],
        "maskformer": ["image-segmentation", "feature-extraction"],
        "mask2former": ["image-segmentation", "feature-extraction"],
        "oneformer": ["image-segmentation", "feature-extraction"],
        "dinov2": ["image-classification", "feature-extraction"],
        "sam": ["image-segmentation", "feature-extraction"],
        "vit-hybrid": ["image-classification", "feature-extraction"],
        "data2vec-vision": ["image-classification", "feature-extraction"],
        "timesformer": ["video-classification", "feature-extraction"],
        "videomae": ["video-classification", "feature-extraction"],
        "vivit": ["video-classification", "feature-extraction"],
        "x-clip": ["video-classification", "feature-extraction"],
        
        # === Object Detection Models ===
        "detr": ["object-detection", "feature-extraction"],
        "deformable-detr": ["object-detection", "feature-extraction"],
        "conditional-detr": ["object-detection", "feature-extraction"],
        "deta": ["object-detection", "feature-extraction"],
        "table-transformer": ["object-detection", "feature-extraction"],
        "yolos": ["object-detection", "feature-extraction"],
        "owlvit": ["zero-shot-object-detection", "feature-extraction"],
        "owlv2": ["zero-shot-object-detection", "feature-extraction"],
        "grounding-dino": ["zero-shot-object-detection", "feature-extraction"],
        "rt-detr": ["object-detection", "feature-extraction"],
        
        # === Multimodal Models ===
        "clip": ["zero-shot-image-classification", "feature-extraction"],
        "clip-vision-model": ["image-classification", "feature-extraction"],
        "clipseg": ["image-segmentation", "feature-extraction"],
        "x-clip": ["video-classification", "feature-extraction"],
        "align": ["zero-shot-image-classification", "feature-extraction"],
        "altclip": ["zero-shot-image-classification", "feature-extraction"],
        "bridgetower": ["image-text-to-text", "feature-extraction"],
        "chinese_clip": ["zero-shot-image-classification", "feature-extraction"],
        "clip_vision_model": ["image-classification", "feature-extraction"],
        "flava": ["image-text-classification", "feature-extraction"],
        "groupvit": ["zero-shot-image-classification", "feature-extraction"],
        "siglip": ["zero-shot-image-classification", "feature-extraction"],
        "blip": ["image-to-text", "visual-question-answering", "feature-extraction"],
        "blip-2": ["image-to-text", "visual-question-answering", "feature-extraction"],
        "instructblip": ["image-to-text", "visual-question-answering", "feature-extraction"],
        "kosmos-2": ["image-to-text", "visual-question-answering", "feature-extraction"],
        "pix2struct": ["image-to-text", "visual-question-answering", "feature-extraction"],
        "vilt": ["visual-question-answering", "feature-extraction"],
        "visual_bert": ["visual-question-answering", "feature-extraction"],
        "git": ["image-to-text", "feature-extraction"],
        "mgp-str": ["image-to-text", "feature-extraction"],
        "donut": ["document-question-answering", "feature-extraction"],
        "layoutlm": ["document-question-answering", "token-classification"],
        "layoutlmv2": ["document-question-answering", "token-classification"],
        "layoutlmv3": ["document-question-answering", "token-classification"],
        "layoutxlm": ["document-question-answering", "token-classification"],
        "lilt": ["document-question-answering", "token-classification"],
        "bros": ["document-question-answering", "token-classification"],
        "udop": ["document-question-answering", "feature-extraction"],
        "matcha": ["image-to-text", "feature-extraction"],
        "paligemma": ["image-to-text", "visual-question-answering", "feature-extraction"],
        "llava": ["image-to-text", "visual-question-answering", "feature-extraction"],
        "llava_next": ["image-to-text", "visual-question-answering", "feature-extraction"],
        "video_llava": ["video-text-to-text", "feature-extraction"],
        "vipllava": ["image-to-text", "visual-question-answering", "feature-extraction"],
        "idefics": ["image-to-text", "visual-question-answering", "feature-extraction"],
        "idefics2": ["image-to-text", "visual-question-answering", "feature-extraction"],
        "chameleon": ["image-text-to-text", "feature-extraction"],
        
        # === Audio Models ===
        "wav2vec2": ["automatic-speech-recognition", "audio-classification", "feature-extraction"],
        "wav2vec2-conformer": ["automatic-speech-recognition", "audio-classification", "feature-extraction"],
        "wav2vec2-bert": ["automatic-speech-recognition", "audio-classification", "feature-extraction"],
        "whisper": ["automatic-speech-recognition", "feature-extraction"],
        "hubert": ["automatic-speech-recognition", "audio-classification", "feature-extraction"],
        "wavlm": ["automatic-speech-recognition", "audio-classification", "feature-extraction"],
        "sew": ["automatic-speech-recognition", "audio-classification", "feature-extraction"],
        "sew-d": ["automatic-speech-recognition", "audio-classification", "feature-extraction"],
        "unispeech": ["automatic-speech-recognition", "audio-classification", "feature-extraction"],
        "unispeech-sat": ["automatic-speech-recognition", "audio-classification", "feature-extraction"],
        "data2vec-audio": ["automatic-speech-recognition", "audio-classification", "feature-extraction"],
        "mctct": ["automatic-speech-recognition", "feature-extraction"],
        "speechbert": ["automatic-speech-recognition", "feature-extraction"],
        "speecht5": ["text-to-speech", "automatic-speech-recognition", "feature-extraction"],
        "mms": ["automatic-speech-recognition", "text-to-speech", "feature-extraction"],
        "vits": ["text-to-speech", "feature-extraction"],
        "encodec": ["audio-to-audio", "feature-extraction"],
        "musicgen": ["text-to-audio", "feature-extraction"],
        "musicgen_melody": ["text-to-audio", "feature-extraction"],
        "audio-spectrogram-transformer": ["audio-classification", "feature-extraction"],
        "ast": ["audio-classification", "feature-extraction"],
        "clap": ["zero-shot-audio-classification", "feature-extraction"],
        "pop2piano": ["audio-to-audio", "feature-extraction"],
        "univnet": ["text-to-speech", "feature-extraction"],
        "fastspeech2": ["text-to-speech", "feature-extraction"],
        "bark": ["text-to-speech", "feature-extraction"],
        
        # === Protein/Biology Models ===
        "esm": ["fill-mask", "feature-extraction"],
        "esmfold": ["protein-folding", "feature-extraction"],
        "biogpt": ["text-generation", "feature-extraction"],
        
        # === Recommendation Models ===
        "bert4rec": ["recommendation", "feature-extraction"],
        "sasrec": ["recommendation", "feature-extraction"],
        
        # === Time Series Models ===
        "informer": ["time-series-forecasting", "feature-extraction"],
        "autoformer": ["time-series-forecasting", "feature-extraction"],
        "time_series_transformer": ["time-series-forecasting", "feature-extraction"],
        "patchtst": ["time-series-forecasting", "feature-extraction"],
        "patchtsmixer": ["time-series-forecasting", "feature-extraction"],
        
        # === Graph Models ===
        "graphormer": ["graph-modeling", "feature-extraction"],
        
        # === Structured Data ===
        "tapas": ["table-question-answering", "feature-extraction"],
        "tapex": ["table-question-answering", "feature-extraction"],
        
        # === Specialized Language Models ===
        "markuplm": ["token-classification", "question-answering", "feature-extraction"],
        "codebert": ["text-classification", "feature-extraction"],
        "graphcodebert": ["text-classification", "feature-extraction"],
        "unixcoder": ["text-classification", "feature-extraction"],
        "splinter": ["question-answering", "feature-extraction"],
        "retribert": ["text-classification", "feature-extraction"],
        "realm": ["question-answering", "feature-extraction"],
        "dpr": ["question-answering", "feature-extraction"],
        "transfo-xl": ["text-generation", "feature-extraction"],
        "xlnet": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "mega": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "mpnet": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "megatron-gpt2": ["text-generation", "feature-extraction"],
        "fsmt": ["translation", "feature-extraction"],
        "distilgpt2": ["text-generation", "feature-extraction"],
        "distilroberta": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "distilbart": ["text2text-generation", "summarization", "feature-extraction"],
        "trocr": ["image-to-text", "feature-extraction"],
        "xmod": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "bert-generation": ["text-generation", "feature-extraction"],
        "bert-japanese": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "bertweet": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "herbert": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "phobert": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "barthez": ["text2text-generation", "summarization", "feature-extraction"],
        "bartpho": ["text2text-generation", "summarization", "feature-extraction"],
        "cpm": ["text-generation", "feature-extraction"],
        "cpmant": ["text-generation", "feature-extraction"],
        "mluke": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        "mobilebert": ["text-classification", "token-classification", "question-answering", "feature-extraction"],
        
        # === Decision/Policy Models ===
        "decision_transformer": ["reinforcement-learning", "feature-extraction"],
        "trajectory_transformer": ["reinforcement-learning", "feature-extraction"],
        
        # === Diffusion Models (if needed for feature extraction) ===
        "dit": ["image-generation", "feature-extraction"],
        "bit": ["image-classification", "feature-extraction"],
    }
    
    # Mapping from tasks to ORTModel classes
    TASK_TO_ORT_MODEL = {
        # === Text Tasks ===
        "text-classification": "ORTModelForSequenceClassification",
        "token-classification": "ORTModelForTokenClassification",
        "question-answering": "ORTModelForQuestionAnswering",
        "feature-extraction": "ORTModelForFeatureExtraction",
        "fill-mask": "ORTModelForMaskedLM",
        
        # === Generative Tasks ===
        "text-generation": "ORTModelForCausalLM",
        "text2text-generation": "ORTModelForSeq2SeqLM",
        "summarization": "ORTModelForSeq2SeqLM",
        "translation": "ORTModelForSeq2SeqLM",
        
        # === Vision Tasks ===
        "image-classification": "ORTModelForImageClassification",
        "zero-shot-image-classification": "ORTModel",  # Generic for CLIP-like models
        "image-segmentation": "ORTModelForSemanticSegmentation",
        "object-detection": "ORTModelForObjectDetection",
        "zero-shot-object-detection": "ORTModel",  # Generic for OWL-ViT-like models
        "image-to-text": "ORTModelForVision2Seq",
        "image-to-image": "ORTModel",  # Generic for image transformation
        "video-classification": "ORTModel",  # Generic for video models
        
        # === Audio Tasks ===
        "automatic-speech-recognition": "ORTModelForSpeechSeq2Seq",
        "audio-classification": "ORTModelForAudioClassification",
        "text-to-speech": "ORTModel",  # Generic for TTS models
        "text-to-audio": "ORTModel",  # Generic for audio generation
        "audio-to-audio": "ORTModel",  # Generic for audio transformation
        "zero-shot-audio-classification": "ORTModel",  # Generic for CLAP-like models
        
        # === Multimodal Tasks ===
        "document-question-answering": "ORTModelForQuestionAnswering",
        "visual-question-answering": "ORTModel",  # Generic for VQA models
        "image-text-classification": "ORTModel",  # Generic for multimodal classification
        "image-text-to-text": "ORTModel",  # Generic for multimodal generation
        "video-text-to-text": "ORTModel",  # Generic for video-text models
        
        # === Specialized Tasks ===
        "table-question-answering": "ORTModel",  # Generic for table QA
        "graph-modeling": "ORTModel",  # Generic for graph models
        "time-series-forecasting": "ORTModel",  # Generic for time series
        "protein-folding": "ORTModel",  # Generic for protein models
        "recommendation": "ORTModel",  # Generic for recommendation systems
        "reinforcement-learning": "ORTModel",  # Generic for RL models
        "image-generation": "ORTModel",  # Generic for diffusion models
    }
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        task: str | None = None,
        provider: str = "CPUExecutionProvider",
        session_options: Any | None = None,
        provider_options: dict | None = None,
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