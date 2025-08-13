"""
Shape inference for model outputs.

This module infers output shapes and names based on model task and architecture.
"""

from typing import Any

from .patterns import TASK_TO_OUTPUTS


class ShapeInferencer:
    """Infer output shapes and specifications for models."""
    
    # Extended output mappings by task
    EXTENDED_OUTPUTS = {
        # Classification outputs
        "text-classification": ["logits"],
        "token-classification": ["logits"],
        "image-classification": ["logits"],
        "audio-classification": ["logits"],
        "video-classification": ["logits"],
        
        # Sequence labeling
        "named-entity-recognition": ["logits", "predictions"],
        "part-of-speech-tagging": ["logits", "predictions"],
        
        # Generation outputs
        "text-generation": ["logits", "past_key_values"],
        "text2text-generation": [
            "logits", "past_key_values",
            "encoder_last_hidden_state", "encoder_hidden_states"
        ],
        "translation": ["logits", "past_key_values", "encoder_last_hidden_state"],
        "summarization": ["logits", "past_key_values", "encoder_last_hidden_state"],
        
        # Question answering
        "question-answering": ["start_logits", "end_logits"],
        "extractive-question-answering": ["start_logits", "end_logits", "has_answer"],
        "visual-question-answering": ["logits", "predictions"],
        "document-question-answering": ["start_logits", "end_logits"],
        
        # Feature extraction
        "feature-extraction": ["last_hidden_state", "pooler_output", "hidden_states"],
        "sentence-similarity": ["sentence_embedding"],
        "image-feature-extraction": ["image_embeds", "pooler_output"],
        
        # Fill mask
        "fill-mask": ["logits"],
        "masked-language-modeling": ["logits", "predictions"],
        
        # Object detection
        "object-detection": ["logits", "pred_boxes", "pred_masks"],
        "zero-shot-object-detection": ["logits", "pred_boxes", "objectness"],
        
        # Image segmentation
        "image-segmentation": ["logits", "pred_masks"],
        "semantic-segmentation": ["logits"],
        "instance-segmentation": ["masks", "boxes", "labels", "scores"],
        "panoptic-segmentation": ["masks", "boxes", "labels", "scores", "segments_info"],
        
        # Audio
        "automatic-speech-recognition": ["logits", "predicted_ids"],
        "audio-frame-classification": ["logits"],
        "audio-xvector": ["logits", "embeddings"],
        "text-to-speech": ["waveform", "spectrogram"],
        
        # Multimodal
        "image-to-text": ["logits", "generated_ids", "past_key_values"],
        "image-text-retrieval": ["logits_per_image", "logits_per_text"],
        "zero-shot-image-classification": ["logits_per_image", "logits_per_text"],
        
        # Other
        "depth-estimation": ["predicted_depth", "depth"],
        "regression": ["logits"],
        "multiple-choice": ["logits"],
        "next-sentence-prediction": ["logits"],
        "conversational": ["logits", "past_key_values"],
    }
    
    @classmethod
    def get_output_names(cls, task: str, model_type: str, config: dict[str, Any]) -> list[str]:
        """
        Get output names for a model.
        
        Args:
            task: Task name
            model_type: Model type string
            config: Model configuration
            
        Returns:
            List of output names
        """
        # Check extended outputs first
        if task in cls.EXTENDED_OUTPUTS:
            outputs = cls.EXTENDED_OUTPUTS[task].copy()
        # Then check pattern outputs
        elif task in TASK_TO_OUTPUTS:
            outputs = TASK_TO_OUTPUTS[task].copy()
        else:
            # Default outputs
            outputs = ["logits", "last_hidden_state"]
        
        # Filter outputs based on model configuration
        outputs = cls._filter_outputs_by_config(outputs, model_type, config)
        
        # Add model-specific outputs
        outputs = cls._add_model_specific_outputs(outputs, model_type, config, task)
        
        return outputs
    
    @classmethod
    def _filter_outputs_by_config(
        cls,
        outputs: list[str],
        model_type: str,
        config: dict[str, Any]
    ) -> list[str]:
        """Filter outputs based on model configuration."""
        filtered = []
        model_type_lower = model_type.lower()
        
        for output in outputs:
            # Check if output is relevant for this model
            if output == "pooler_output":
                # Not all models have pooler
                has_pooler = config.get("add_pooling_layer", True)
                if not has_pooler:
                    continue
                # Some models don't have pooler even if config says so
                no_pooler_models = ["gpt", "llama", "mistral", "falcon", "opt", "bloom"]
                if any(m in model_type_lower for m in no_pooler_models):
                    continue
            
            elif output == "past_key_values":
                # Only decoder models have past key values
                is_decoder = config.get("is_decoder", False)
                is_encoder_decoder = config.get("is_encoder_decoder", False)
                
                # Check model type
                decoder_models = [
                    "gpt", "llama", "mistral", "falcon", "opt", "bloom",
                    "mpt", "phi", "qwen", "stablelm", "bart", "t5", "marian"
                ]
                has_decoder = any(m in model_type_lower for m in decoder_models)
                
                if not (is_decoder or is_encoder_decoder or has_decoder):
                    continue
            
            elif output in ["encoder_last_hidden_state", "encoder_hidden_states"]:
                # Only encoder-decoder models
                is_encoder_decoder = config.get("is_encoder_decoder", False)
                encoder_decoder_models = ["t5", "bart", "mbart", "marian", "pegasus", "led"]
                has_encoder_decoder = any(m in model_type_lower for m in encoder_decoder_models)
                
                if not (is_encoder_decoder or has_encoder_decoder):
                    continue
            
            elif output == "hidden_states":
                # Check if output_hidden_states is enabled
                if not config.get("output_hidden_states", False):
                    continue
            
            elif output == "attentions":
                # Check if output_attentions is enabled
                if not config.get("output_attentions", False):
                    continue
            
            filtered.append(output)
        
        return filtered if filtered else ["logits"]
    
    @classmethod
    def _add_model_specific_outputs(
        cls,
        outputs: list[str],
        model_type: str,
        config: dict[str, Any],
        task: str
    ) -> list[str]:
        """Add model-specific outputs."""
        model_type_lower = model_type.lower()
        
        # CLIP-like models
        if any(clip in model_type_lower for clip in ["clip", "align", "chinese_clip"]):
            if task == "feature-extraction":
                # Add vision and text embeddings
                if "text_embeds" not in outputs:
                    outputs.append("text_embeds")
                if "image_embeds" not in outputs:
                    outputs.append("image_embeds")
        
        # Detection models
        elif any(det in model_type_lower for det in ["detr", "yolos", "conditional-detr"]):
            if "pred_boxes" not in outputs:
                outputs.append("pred_boxes")
            if "logits" not in outputs:
                outputs.append("logits")
        
        # SAM model
        elif "sam" in model_type_lower:
            outputs = ["pred_masks", "iou_scores", "low_res_masks"]
        
        # Wav2Vec2-like models
        elif any(audio in model_type_lower for audio in ["wav2vec2", "hubert", "wavlm"]):
            if task == "automatic-speech-recognition":
                if "projected_states" not in outputs:
                    outputs.append("projected_states")
        
        # Whisper
        elif "whisper" in model_type_lower:
            if task == "automatic-speech-recognition":
                outputs = ["logits", "past_key_values", "encoder_last_hidden_state"]
        
        # LayoutLM family
        elif "layoutlm" in model_type_lower:
            if "pooler_output" not in outputs and "layoutlmv3" not in model_type_lower:
                outputs.append("pooler_output")
        
        return outputs
    
    @classmethod
    def get_output_shapes(
        cls,
        output_names: list[str],
        batch_size: int,
        seq_length: int,
        config: dict[str, Any],
        task: str
    ) -> dict[str, tuple]:
        """
        Get expected output shapes.
        
        Args:
            output_names: List of output names
            batch_size: Batch size
            seq_length: Sequence length
            config: Model configuration
            task: Task name
            
        Returns:
            Dictionary mapping output names to shapes
        """
        shapes = {}
        
        # Get dimensions from config
        hidden_size = config.get("hidden_size", config.get("d_model", 768))
        num_labels = config.get("num_labels", 2)
        vocab_size = config.get("vocab_size", 30000)
        
        # For encoder-decoder models
        if config.get("is_encoder_decoder", False):
            encoder_hidden_size = config.get("encoder", {}).get("hidden_size", hidden_size)
            decoder_hidden_size = config.get("decoder", {}).get("hidden_size", hidden_size)
        else:
            encoder_hidden_size = decoder_hidden_size = hidden_size
        
        for output_name in output_names:
            if output_name == "logits":
                # Determine logits shape based on task
                if "classification" in task:
                    if "token" in task:
                        shapes[output_name] = (batch_size, seq_length, num_labels)
                    else:
                        shapes[output_name] = (batch_size, num_labels)
                elif "generation" in task or "mask" in task:
                    shapes[output_name] = (batch_size, seq_length, vocab_size)
                elif task == "multiple-choice":
                    num_choices = config.get("num_choices", 4)
                    shapes[output_name] = (batch_size, num_choices)
                else:
                    shapes[output_name] = (batch_size, seq_length, vocab_size)
            
            elif output_name == "last_hidden_state":
                shapes[output_name] = (batch_size, seq_length, hidden_size)
            
            elif output_name == "pooler_output":
                shapes[output_name] = (batch_size, hidden_size)
            
            elif output_name in ["start_logits", "end_logits"]:
                shapes[output_name] = (batch_size, seq_length)
            
            elif output_name == "encoder_last_hidden_state":
                shapes[output_name] = (batch_size, seq_length, encoder_hidden_size)
            
            elif output_name == "past_key_values":
                # Past key values have complex nested structure
                # This is a simplified representation
                num_layers = config.get("num_hidden_layers", config.get("n_layer", 12))
                num_heads = config.get("num_attention_heads", config.get("n_head", 12))
                head_dim = hidden_size // num_heads
                # Shape: (num_layers, 2, batch_size, num_heads, seq_length, head_dim)
                shapes[output_name] = f"({num_layers}, 2, {batch_size}, {num_heads}, seq_length, {head_dim})"
            
            elif output_name == "hidden_states":
                num_layers = config.get("num_hidden_layers", config.get("n_layer", 12))
                shapes[output_name] = f"tuple of {num_layers+1} x ({batch_size}, {seq_length}, {hidden_size})"
            
            elif output_name == "attentions":
                num_layers = config.get("num_hidden_layers", config.get("n_layer", 12))
                num_heads = config.get("num_attention_heads", config.get("n_head", 12))
                shapes[output_name] = f"tuple of {num_layers} x ({batch_size}, {num_heads}, {seq_length}, {seq_length})"
            
            elif output_name in ["text_embeds", "image_embeds", "sentence_embedding"]:
                embedding_dim = config.get("projection_dim", hidden_size)
                shapes[output_name] = (batch_size, embedding_dim)
            
            elif output_name == "pred_boxes":
                # Object detection boxes
                num_queries = config.get("num_queries", 100)
                shapes[output_name] = (batch_size, num_queries, 4)
            
            elif output_name == "pred_masks":
                # Segmentation masks
                if task == "object-detection":
                    num_queries = config.get("num_queries", 100)
                    shapes[output_name] = (batch_size, num_queries, seq_length, seq_length)
                else:
                    # Semantic segmentation
                    num_classes = config.get("num_labels", num_labels)
                    image_size = config.get("image_size", 224)
                    shapes[output_name] = (batch_size, num_classes, image_size, image_size)
            
            else:
                # Default shape
                shapes[output_name] = (batch_size, seq_length, hidden_size)
        
        return shapes
    
    @classmethod
    def requires_attention_mask(cls, model_type: str) -> bool:
        """
        Check if model requires attention mask.
        
        Args:
            model_type: Model type string
            
        Returns:
            Whether attention mask is required
        """
        model_type_lower = model_type.lower()
        
        # Most models need attention mask
        # Only some vision models don't
        no_mask_models = ["vit", "deit", "beit", "swin", "convnext", "resnet", "efficientnet"]
        
        return not any(m in model_type_lower for m in no_mask_models)