"""
Architecture patterns and mappings for model type detection.

This module contains patterns and mappings used to identify model architectures,
tasks, and input/output specifications.
"""

from typing import Dict, List, Set

# Architecture to task mapping based on class name patterns
ARCHITECTURE_TO_TASK = {
    # Text classification
    "ForSequenceClassification": "text-classification",
    "ForSentenceEmbedding": "feature-extraction",
    
    # Token classification
    "ForTokenClassification": "token-classification",
    "ForNER": "token-classification",
    
    # Question answering
    "ForQuestionAnswering": "question-answering",
    
    # Text generation
    "ForCausalLM": "text-generation",
    "LMHeadModel": "text-generation",
    "ForConditionalGeneration": "text2text-generation",
    "ForSeq2SeqLM": "text2text-generation",
    
    # Masked language modeling
    "ForMaskedLM": "fill-mask",
    
    # Image classification
    "ForImageClassification": "image-classification",
    
    # Audio
    "ForCTC": "automatic-speech-recognition",
    "ForAudioClassification": "audio-classification",
    "ForSpeechSeq2Seq": "automatic-speech-recognition",
    
    # Multimodal
    "ForImageTextRetrieval": "feature-extraction",
    "ForZeroShotImageClassification": "zero-shot-image-classification",
    
    # Default
    "Model": "feature-extraction",
    "PreTrainedModel": "feature-extraction",
}

# Model type to typical input names
MODEL_TYPE_TO_INPUTS = {
    # Text models
    "bert": ["input_ids", "attention_mask", "token_type_ids"],
    "roberta": ["input_ids", "attention_mask"],
    "distilbert": ["input_ids", "attention_mask"],
    "albert": ["input_ids", "attention_mask", "token_type_ids"],
    "electra": ["input_ids", "attention_mask", "token_type_ids"],
    "xlm": ["input_ids", "attention_mask"],
    "xlm-roberta": ["input_ids", "attention_mask"],
    "bart": ["input_ids", "attention_mask"],
    "mbart": ["input_ids", "attention_mask"],
    "t5": ["input_ids", "attention_mask", "decoder_input_ids"],
    "pegasus": ["input_ids", "attention_mask"],
    "marian": ["input_ids", "attention_mask"],
    
    # Generative models
    "gpt2": ["input_ids", "attention_mask"],
    "gpt-neo": ["input_ids", "attention_mask"],
    "gpt-j": ["input_ids", "attention_mask"],
    "opt": ["input_ids", "attention_mask"],
    "bloom": ["input_ids", "attention_mask"],
    "llama": ["input_ids", "attention_mask"],
    "mistral": ["input_ids", "attention_mask"],
    "falcon": ["input_ids", "attention_mask"],
    "phi": ["input_ids", "attention_mask"],
    "qwen": ["input_ids", "attention_mask"],
    
    # Vision models
    "vit": ["pixel_values"],
    "deit": ["pixel_values"],
    "beit": ["pixel_values"],
    "swin": ["pixel_values"],
    "convnext": ["pixel_values"],
    "resnet": ["pixel_values"],
    "efficientnet": ["pixel_values"],
    "regnet": ["pixel_values"],
    "mobilenet": ["pixel_values"],
    "densenet": ["pixel_values"],
    
    # Audio models
    "wav2vec2": ["input_values", "attention_mask"],
    "whisper": ["input_features"],
    "hubert": ["input_values", "attention_mask"],
    "wavlm": ["input_values", "attention_mask"],
    "unispeech": ["input_values", "attention_mask"],
    "seamless": ["input_features"],
    
    # Multimodal models
    "clip": ["input_ids", "attention_mask", "pixel_values"],
    "align": ["input_ids", "attention_mask", "pixel_values"],
    "blip": ["input_ids", "attention_mask", "pixel_values"],
    "blip-2": ["input_ids", "attention_mask", "pixel_values"],
    "bridgetower": ["input_ids", "attention_mask", "pixel_values"],
    "chinese_clip": ["input_ids", "attention_mask", "pixel_values"],
    "flava": ["input_ids", "attention_mask", "pixel_values"],
    "git": ["input_ids", "attention_mask", "pixel_values"],
    "groupvit": ["input_ids", "attention_mask", "pixel_values"],
    "owlvit": ["input_ids", "attention_mask", "pixel_values"],
    "x-clip": ["input_ids", "attention_mask", "pixel_values"],
    
    # Document models
    "layoutlm": ["input_ids", "attention_mask", "token_type_ids", "bbox"],
    "layoutlmv2": ["input_ids", "attention_mask", "token_type_ids", "bbox", "image"],
    "layoutlmv3": ["input_ids", "attention_mask", "bbox", "pixel_values"],
    "layoutxlm": ["input_ids", "attention_mask", "bbox", "image"],
    "markuplm": ["input_ids", "attention_mask", "token_type_ids", "xpath_tags_seq", "xpath_subs_seq"],
    
    # Special models
    "sam": ["pixel_values", "input_points", "input_labels"],
    "detr": ["pixel_values", "pixel_mask"],
    "yolos": ["pixel_values"],
}

# Task to typical output names
TASK_TO_OUTPUTS = {
    # Classification tasks
    "text-classification": ["logits"],
    "token-classification": ["logits"],
    "image-classification": ["logits"],
    "audio-classification": ["logits"],
    
    # Generation tasks
    "text-generation": ["logits", "past_key_values"],
    "text2text-generation": ["logits", "past_key_values", "encoder_last_hidden_state"],
    
    # Question answering
    "question-answering": ["start_logits", "end_logits"],
    
    # Feature extraction
    "feature-extraction": ["last_hidden_state", "pooler_output"],
    
    # Masked LM
    "fill-mask": ["logits"],
    
    # Speech
    "automatic-speech-recognition": ["logits"],
    
    # Vision
    "object-detection": ["logits", "pred_boxes"],
    "image-segmentation": ["logits", "pred_masks"],
    
    # Default
    "default": ["logits", "last_hidden_state"],
}

# Input shapes by modality
DEFAULT_SHAPES = {
    # Text
    "input_ids": (1, 128),  # batch_size, sequence_length
    "attention_mask": (1, 128),
    "token_type_ids": (1, 128),
    "position_ids": (1, 128),
    
    # Text generation
    "decoder_input_ids": (1, 128),
    "decoder_attention_mask": (1, 128),
    
    # Vision
    "pixel_values": (1, 3, 224, 224),  # batch_size, channels, height, width
    "pixel_mask": (1, 224, 224),
    
    # Audio
    "input_values": (1, 16000),  # batch_size, sequence_length
    "input_features": (1, 80, 3000),  # batch_size, feature_dim, sequence_length
    
    # Document understanding
    "bbox": (1, 128, 4),  # batch_size, sequence_length, 4
    "image": (1, 3, 224, 224),
    
    # Special
    "input_points": (1, 1, 1, 2),  # batch_size, point_batch_size, nb_points_per_image, 2
    "input_labels": (1, 1, 1),  # batch_size, point_batch_size, nb_points_per_image
    
    # Markup
    "xpath_tags_seq": (1, 128, 50),  # batch_size, sequence_length, max_depth
    "xpath_subs_seq": (1, 128, 50),  # batch_size, sequence_length, max_depth
}

# Dynamic axes for ONNX export
DYNAMIC_AXES_PATTERNS = {
    # Text inputs
    "input_ids": {0: "batch_size", 1: "sequence_length"},
    "attention_mask": {0: "batch_size", 1: "sequence_length"},
    "token_type_ids": {0: "batch_size", 1: "sequence_length"},
    "position_ids": {0: "batch_size", 1: "sequence_length"},
    
    # Decoder inputs
    "decoder_input_ids": {0: "batch_size", 1: "decoder_sequence_length"},
    "decoder_attention_mask": {0: "batch_size", 1: "decoder_sequence_length"},
    
    # Vision inputs
    "pixel_values": {0: "batch_size"},
    "pixel_mask": {0: "batch_size"},
    
    # Audio inputs
    "input_values": {0: "batch_size", 1: "sequence_length"},
    "input_features": {0: "batch_size", 2: "sequence_length"},
    
    # Document inputs
    "bbox": {0: "batch_size", 1: "sequence_length"},
    "image": {0: "batch_size"},
    
    # Outputs
    "logits": {0: "batch_size"},
    "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
    "pooler_output": {0: "batch_size"},
    "start_logits": {0: "batch_size"},
    "end_logits": {0: "batch_size"},
}

# Model families that need special handling
SPECIAL_MODEL_FAMILIES = {
    "encoder-decoder": ["t5", "bart", "mbart", "pegasus", "marian", "blenderbot", "led", "longformer"],
    "vision-encoder-decoder": ["vision-encoder-decoder", "trocr", "donut"],
    "multimodal": ["clip", "align", "flava", "blip", "bridgetower", "chinese_clip"],
    "document": ["layoutlm", "layoutlmv2", "layoutlmv3", "layoutxlm", "markuplm"],
    "object-detection": ["detr", "yolos", "conditional-detr", "deformable-detr", "table-transformer"],
    "segmentation": ["sam", "maskformer", "mask2former", "oneformer"],
}

def get_model_family(model_type: str) -> str:
    """
    Determine the model family from model type.
    
    Args:
        model_type: The model type string
        
    Returns:
        Model family name
    """
    model_type_lower = model_type.lower()
    
    for family, types in SPECIAL_MODEL_FAMILIES.items():
        if model_type_lower in types:
            return family
    
    # Check for patterns
    if "encoder-decoder" in model_type_lower:
        return "encoder-decoder"
    elif any(vision in model_type_lower for vision in ["vit", "resnet", "convnext", "swin", "deit"]):
        return "vision"
    elif any(audio in model_type_lower for audio in ["wav2vec", "whisper", "hubert", "wavlm"]):
        return "audio"
    elif any(text in model_type_lower for text in ["bert", "roberta", "gpt", "llama", "mistral"]):
        return "text"
    
    return "unknown"