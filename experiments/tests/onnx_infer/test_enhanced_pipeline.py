"""
Unit tests for Enhanced Pipeline module.

Tests the enhanced pipeline wrapper functionality including:
- Generic data_processor parameter routing
- Multi-modal task support
- Processor type detection
- Pipeline creation with various configurations
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tez-144_onnx_automodel_infer" / "src"))

from enhanced_pipeline import (
    create_pipeline,
    pipeline,
    _detect_processor_type
)


@pytest.mark.unit
class TestDetectProcessorType:
    """Test the _detect_processor_type function."""
    
    def test_detect_tokenizer_by_class_name(self):
        """Test detection of tokenizer by class name."""
        # Mock tokenizer with Tokenizer in class name
        mock_tokenizer = Mock()
        mock_tokenizer.__class__.__name__ = "BertTokenizer"
        
        result = _detect_processor_type(mock_tokenizer)
        assert result == "tokenizer"
    
    def test_detect_tokenizer_fast_by_class_name(self):
        """Test detection of fast tokenizer by class name."""
        mock_tokenizer = Mock()
        mock_tokenizer.__class__.__name__ = "BertTokenizerFast"
        
        result = _detect_processor_type(mock_tokenizer)
        assert result == "tokenizer"
    
    def test_detect_image_processor_by_class_name(self):
        """Test detection of image processor by class name."""
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "ViTImageProcessor"
        
        result = _detect_processor_type(mock_processor)
        assert result == "image_processor"
    
    def test_detect_feature_extractor_by_class_name(self):
        """Test detection of feature extractor by class name."""
        mock_extractor = Mock()
        mock_extractor.__class__.__name__ = "Wav2Vec2FeatureExtractor"
        
        result = _detect_processor_type(mock_extractor)
        assert result == "feature_extractor"
    
    def test_detect_processor_by_class_name(self):
        """Test detection of multimodal processor by class name."""
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "CLIPProcessor"
        
        result = _detect_processor_type(mock_processor)
        assert result == "processor"
    
    def test_detect_tokenizer_by_attributes(self):
        """Test detection of tokenizer by attributes."""
        mock_tokenizer = Mock()
        mock_tokenizer.__class__.__name__ = "CustomProcessor"
        mock_tokenizer.tokenize = Mock()
        
        result = _detect_processor_type(mock_tokenizer)
        assert result == "tokenizer"
    
    def test_detect_tokenizer_by_encode_attribute(self):
        """Test detection of tokenizer by encode attribute."""
        mock_tokenizer = Mock()
        mock_tokenizer.__class__.__name__ = "CustomProcessor"
        mock_tokenizer.encode = Mock()
        
        result = _detect_processor_type(mock_tokenizer)
        assert result == "tokenizer"
    
    def test_detect_image_processor_by_attributes(self):
        """Test detection of image processor by attributes."""
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "CustomProcessor"
        mock_processor.pixel_values = Mock()
        
        result = _detect_processor_type(mock_processor)
        assert result == "image_processor"
    
    def test_detect_feature_extractor_by_attributes(self):
        """Test detection of feature extractor by attributes."""
        mock_extractor = Mock()
        mock_extractor.__class__.__name__ = "CustomProcessor"
        mock_extractor.feature_size = 512
        
        result = _detect_processor_type(mock_extractor)
        assert result == "feature_extractor"
    
    def test_detect_feature_extractor_by_sampling_rate(self):
        """Test detection of feature extractor by sampling rate."""
        mock_extractor = Mock()
        mock_extractor.__class__.__name__ = "CustomProcessor"
        mock_extractor.sampling_rate = 16000
        
        result = _detect_processor_type(mock_extractor)
        assert result == "feature_extractor"
    
    def test_detect_multimodal_processor_by_attributes(self):
        """Test detection of multimodal processor by attributes."""
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "CustomProcessor"
        mock_processor.tokenizer = Mock()
        mock_processor.image_processor = Mock()
        
        result = _detect_processor_type(mock_processor)
        assert result == "processor"
    
    def test_detect_tokenizer_wrapper(self):
        """Test detection of tokenizer wrapper (like FixedShapeTokenizer)."""
        mock_wrapper = Mock()
        mock_wrapper.__class__.__name__ = "ONNXTokenizer"
        mock_wrapper.tokenizer = Mock()
        # No image_processor attribute
        
        result = _detect_processor_type(mock_wrapper)
        assert result == "tokenizer"
    
    def test_detect_unknown_processor(self):
        """Test handling of unknown processor type."""
        mock_unknown = Mock()
        mock_unknown.__class__.__name__ = "UnknownProcessor"
        
        result = _detect_processor_type(mock_unknown)
        assert result == "unknown"
    
    def test_exclude_image_processor_from_processor(self):
        """Test that ImageProcessor class name is not detected as generic processor."""
        mock_image_proc = Mock()
        mock_image_proc.__class__.__name__ = "SomeImageProcessor"
        
        result = _detect_processor_type(mock_image_proc)
        assert result == "image_processor"
        # Should not be "processor"


@pytest.mark.unit
class TestCreatePipeline:
    """Test the create_pipeline function."""
    
    @patch('enhanced_pipeline.hf_pipeline')
    def test_create_pipeline_basic(self, mock_hf_pipeline):
        """Test basic pipeline creation without data_processor."""
        mock_hf_pipeline.return_value = Mock()
        
        result = create_pipeline(
            task="text-classification",
            model="bert-base-uncased"
        )
        
        mock_hf_pipeline.assert_called_once()
        args, kwargs = mock_hf_pipeline.call_args
        assert kwargs["task"] == "text-classification"
        assert kwargs["model"] == "bert-base-uncased"
        assert "tokenizer" not in kwargs
    
    @patch('enhanced_pipeline.hf_pipeline')
    def test_create_pipeline_with_tokenizer(self, mock_hf_pipeline):
        """Test pipeline creation with tokenizer data_processor."""
        mock_hf_pipeline.return_value = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.__class__.__name__ = "BertTokenizer"
        
        create_pipeline(
            task="text-classification",
            model="bert-base-uncased",
            data_processor=mock_tokenizer
        )
        
        args, kwargs = mock_hf_pipeline.call_args
        assert kwargs["tokenizer"] == mock_tokenizer
    
    @patch('enhanced_pipeline.hf_pipeline')
    def test_create_pipeline_with_image_processor(self, mock_hf_pipeline):
        """Test pipeline creation with image processor data_processor."""
        mock_hf_pipeline.return_value = Mock()
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "ViTImageProcessor"
        
        create_pipeline(
            task="image-classification",
            model="google/vit-base-patch16-224",
            data_processor=mock_processor
        )
        
        args, kwargs = mock_hf_pipeline.call_args
        assert kwargs["image_processor"] == mock_processor
    
    @patch('enhanced_pipeline.hf_pipeline')
    def test_create_pipeline_with_feature_extractor(self, mock_hf_pipeline):
        """Test pipeline creation with feature extractor data_processor."""
        mock_hf_pipeline.return_value = Mock()
        mock_extractor = Mock()
        mock_extractor.__class__.__name__ = "Wav2Vec2FeatureExtractor"
        
        create_pipeline(
            task="automatic-speech-recognition",
            model="facebook/wav2vec2-base",
            data_processor=mock_extractor
        )
        
        args, kwargs = mock_hf_pipeline.call_args
        assert kwargs["feature_extractor"] == mock_extractor
    
    @patch('enhanced_pipeline.hf_pipeline')
    def test_create_pipeline_with_multimodal_processor(self, mock_hf_pipeline):
        """Test pipeline creation with multimodal processor."""
        mock_hf_pipeline.return_value = Mock()
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "CLIPProcessor"
        
        create_pipeline(
            task="zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
            data_processor=mock_processor
        )
        
        args, kwargs = mock_hf_pipeline.call_args
        assert kwargs["processor"] == mock_processor
    
    @patch('enhanced_pipeline.hf_pipeline')
    def test_create_pipeline_task_based_routing_text(self, mock_hf_pipeline):
        """Test task-based routing for text tasks."""
        mock_hf_pipeline.return_value = Mock()
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "UnknownProcessor"  # Will use task-based routing
        
        create_pipeline(
            task="text-classification",
            model="bert-base-uncased",
            data_processor=mock_processor
        )
        
        args, kwargs = mock_hf_pipeline.call_args
        assert kwargs["tokenizer"] == mock_processor
    
    @patch('enhanced_pipeline.hf_pipeline')
    def test_create_pipeline_task_based_routing_vision(self, mock_hf_pipeline):
        """Test task-based routing for vision tasks."""
        mock_hf_pipeline.return_value = Mock()
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "UnknownProcessor"
        
        create_pipeline(
            task="image-classification",
            model="google/vit-base-patch16-224",
            data_processor=mock_processor
        )
        
        args, kwargs = mock_hf_pipeline.call_args
        assert kwargs["image_processor"] == mock_processor
    
    @patch('enhanced_pipeline.hf_pipeline')
    def test_create_pipeline_task_based_routing_audio(self, mock_hf_pipeline):
        """Test task-based routing for audio tasks."""
        mock_hf_pipeline.return_value = Mock()
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "UnknownProcessor"
        
        create_pipeline(
            task="automatic-speech-recognition",
            model="facebook/wav2vec2-base",
            data_processor=mock_processor
        )
        
        args, kwargs = mock_hf_pipeline.call_args
        assert kwargs["feature_extractor"] == mock_processor
    
    @patch('enhanced_pipeline.hf_pipeline')
    def test_create_pipeline_task_based_routing_multimodal(self, mock_hf_pipeline):
        """Test task-based routing for multimodal tasks."""
        mock_hf_pipeline.return_value = Mock()
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "UnknownProcessor"
        
        create_pipeline(
            task="image-to-text",
            model="nlpconnect/vit-gpt2-image-captioning",
            data_processor=mock_processor
        )
        
        args, kwargs = mock_hf_pipeline.call_args
        assert kwargs["processor"] == mock_processor
    
    @patch('enhanced_pipeline.hf_pipeline')
    def test_create_pipeline_unknown_task_defaults_to_tokenizer(self, mock_hf_pipeline):
        """Test that unknown tasks default to tokenizer routing."""
        mock_hf_pipeline.return_value = Mock()
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "UnknownProcessor"
        
        create_pipeline(
            task="custom-unknown-task",
            model="some-model",
            data_processor=mock_processor
        )
        
        args, kwargs = mock_hf_pipeline.call_args
        assert kwargs["tokenizer"] == mock_processor
    
    @patch('enhanced_pipeline.hf_pipeline')
    def test_create_pipeline_all_parameters(self, mock_hf_pipeline):
        """Test pipeline creation with all possible parameters."""
        mock_hf_pipeline.return_value = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.__class__.__name__ = "BertTokenizer"
        
        create_pipeline(
            task="text-classification",
            model="bert-base-uncased",
            data_processor=mock_tokenizer,
            config="bert-base-uncased",
            framework="pt",
            revision="main",
            use_fast=True,
            token="hf_token",
            device=0,
            device_map="auto",
            torch_dtype="float16",
            trust_remote_code=True,
            model_kwargs={"output_hidden_states": True},
            pipeline_class=None,
            return_all_scores=True
        )
        
        args, kwargs = mock_hf_pipeline.call_args
        assert kwargs["task"] == "text-classification"
        assert kwargs["model"] == "bert-base-uncased"
        assert kwargs["tokenizer"] == mock_tokenizer
        assert kwargs["config"] == "bert-base-uncased"
        assert kwargs["framework"] == "pt"
        assert kwargs["revision"] == "main"
        assert kwargs["use_fast"] is True
        assert kwargs["token"] == "hf_token"
        assert kwargs["device"] == 0
        assert kwargs["device_map"] == "auto"
        assert kwargs["torch_dtype"] == "float16"
        assert kwargs["trust_remote_code"] is True
        assert kwargs["model_kwargs"] == {"output_hidden_states": True}
        assert kwargs["pipeline_class"] is None
        assert kwargs["return_all_scores"] is True


@pytest.mark.unit
class TestPipelineFunction:
    """Test the convenience pipeline function."""
    
    @patch('enhanced_pipeline.create_pipeline')
    def test_pipeline_convenience_function(self, mock_create_pipeline):
        """Test that pipeline function is a wrapper for create_pipeline."""
        mock_create_pipeline.return_value = Mock()
        mock_tokenizer = Mock()
        
        result = pipeline(
            task="text-classification",
            model="bert-base-uncased",
            data_processor=mock_tokenizer,
            device=0
        )
        
        mock_create_pipeline.assert_called_once_with(
            "text-classification",
            model="bert-base-uncased",
            data_processor=mock_tokenizer,
            device=0
        )


@pytest.mark.integration
class TestEnhancedPipelineIntegration:
    """Integration tests for enhanced pipeline functionality."""
    
    @patch('enhanced_pipeline.hf_pipeline')
    def test_text_task_integration(self, mock_hf_pipeline):
        """Test integration with text classification task."""
        mock_pipeline = Mock()
        mock_pipeline.return_value = ["positive", "negative"]
        mock_hf_pipeline.return_value = mock_pipeline
        
        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.__class__.__name__ = "BertTokenizer"
        
        pipe = create_pipeline(
            task="text-classification",
            model="bert-base-uncased",
            data_processor=mock_tokenizer
        )
        
        # Test the pipeline
        result = pipe(["Good movie!", "Bad movie!"])
        assert result == ["positive", "negative"]
    
    @patch('enhanced_pipeline.hf_pipeline')
    def test_vision_task_integration(self, mock_hf_pipeline):
        """Test integration with image classification task."""
        mock_pipeline = Mock()
        mock_pipeline.return_value = [{"label": "cat", "score": 0.9}]
        mock_hf_pipeline.return_value = mock_pipeline
        
        # Create mock image processor
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "ViTImageProcessor"
        
        pipe = create_pipeline(
            task="image-classification",
            model="google/vit-base-patch16-224",
            data_processor=mock_processor
        )
        
        # Test the pipeline
        result = pipe("image.jpg")
        assert result[0]["label"] == "cat"
        assert result[0]["score"] == 0.9
    
    @patch('enhanced_pipeline.hf_pipeline')
    def test_multimodal_task_integration(self, mock_hf_pipeline):
        """Test integration with multimodal task."""
        mock_pipeline = Mock()
        mock_pipeline.return_value = "A cat sitting on a table"
        mock_hf_pipeline.return_value = mock_pipeline
        
        # Create mock multimodal processor
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "BlipProcessor"
        
        pipe = create_pipeline(
            task="image-to-text",
            model="Salesforce/blip-image-captioning-base",
            data_processor=mock_processor
        )
        
        # Test the pipeline
        result = pipe("image.jpg")
        assert result == "A cat sitting on a table"


@pytest.mark.unit
class TestTaskCategorization:
    """Test task categorization for different modalities."""
    
    def test_text_tasks_complete_list(self):
        """Test that all expected text tasks are in TEXT_TASKS."""
        from enhanced_pipeline import create_pipeline
        
        # Get TEXT_TASKS from the function's local scope
        # We'll test through the routing behavior
        text_tasks = [
            "feature-extraction", "text-classification", "sentiment-analysis",
            "token-classification", "ner", "named-entity-recognition",
            "question-answering", "fill-mask", "summarization", "translation",
            "text2text-generation", "text-generation", "zero-shot-classification",
            "conversational", "table-question-answering"
        ]
        
        # Each should be recognized as a text task (through behavior testing)
        for task in text_tasks:
            # Test that a processor with unknown type gets routed to tokenizer for text tasks
            mock_processor = Mock()
            mock_processor.__class__.__name__ = "UnknownProcessor"
            
            with patch('enhanced_pipeline.hf_pipeline') as mock_hf_pipeline:
                create_pipeline(task=task, data_processor=mock_processor)
                args, kwargs = mock_hf_pipeline.call_args
                assert "tokenizer" in kwargs
    
    def test_vision_tasks_complete_list(self):
        """Test that all expected vision tasks are in VISION_TASKS."""
        vision_tasks = [
            "image-classification", "image-segmentation", "object-detection",
            "image-feature-extraction", "depth-estimation", "image-to-image",
            "mask-generation"
        ]
        
        for task in vision_tasks:
            mock_processor = Mock()
            mock_processor.__class__.__name__ = "UnknownProcessor"
            
            with patch('enhanced_pipeline.hf_pipeline') as mock_hf_pipeline:
                create_pipeline(task=task, data_processor=mock_processor)
                args, kwargs = mock_hf_pipeline.call_args
                assert "image_processor" in kwargs
    
    def test_audio_tasks_complete_list(self):
        """Test that all expected audio tasks are in AUDIO_TASKS."""
        audio_tasks = [
            "audio-classification", "automatic-speech-recognition", "asr",
            "text-to-audio", "text-to-speech", "audio-to-audio"
        ]
        
        for task in audio_tasks:
            mock_processor = Mock()
            mock_processor.__class__.__name__ = "UnknownProcessor"
            
            with patch('enhanced_pipeline.hf_pipeline') as mock_hf_pipeline:
                create_pipeline(task=task, data_processor=mock_processor)
                args, kwargs = mock_hf_pipeline.call_args
                assert "feature_extractor" in kwargs
    
    def test_multimodal_tasks_complete_list(self):
        """Test that all expected multimodal tasks are in MULTIMODAL_TASKS."""
        multimodal_tasks = [
            "image-to-text", "document-question-answering", "vqa",
            "visual-question-answering", "zero-shot-image-classification",
            "image-text-to-text", "video-classification"
        ]
        
        for task in multimodal_tasks:
            mock_processor = Mock()
            mock_processor.__class__.__name__ = "UnknownProcessor"
            
            with patch('enhanced_pipeline.hf_pipeline') as mock_hf_pipeline:
                create_pipeline(task=task, data_processor=mock_processor)
                args, kwargs = mock_hf_pipeline.call_args
                assert "processor" in kwargs


@pytest.mark.smoke
class TestEnhancedPipelineSmoke:
    """Smoke tests for basic functionality."""
    
    @patch('enhanced_pipeline.hf_pipeline')
    def test_basic_pipeline_creation(self, mock_hf_pipeline):
        """Test basic pipeline creation doesn't crash."""
        mock_hf_pipeline.return_value = Mock()
        
        result = create_pipeline(task="text-classification", model="bert-base-uncased")
        assert result is not None
    
    @patch('enhanced_pipeline.hf_pipeline')
    def test_basic_pipeline_with_processor(self, mock_hf_pipeline):
        """Test pipeline creation with processor doesn't crash."""
        mock_hf_pipeline.return_value = Mock()
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "BertTokenizer"
        
        result = create_pipeline(
            task="text-classification",
            model="bert-base-uncased",
            data_processor=mock_processor
        )
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])