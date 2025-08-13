"""
Tests for various pipeline tasks with ONNX models.

Tests different types of NLP, Vision, and Audio pipeline tasks
to ensure comprehensive task support with ONNX inference.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import torch
import numpy as np

# Import the modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tez-144_onnx_automodel_infer" / "src"))

from auto_model_loader import AutoModelForONNX
from enhanced_pipeline import create_pipeline
from onnx_tokenizer import create_auto_shape_tokenizer


@pytest.mark.integration
class TestTextPipelineTasks:
    """Test various text-based pipeline tasks."""
    
    @patch('optimum.onnxruntime.ORTModelForSequenceClassification')
    @patch('transformers.AutoTokenizer')
    @patch('enhanced_pipeline.hf_pipeline')
    def test_text_classification_task(self, mock_hf_pipeline, mock_tokenizer_class, mock_ort_model):
        """Test text classification pipeline task."""
        # Setup mocks
        mock_model = self._setup_mock_model(mock_ort_model, "text-classification")
        mock_tokenizer = self._setup_mock_tokenizer(mock_tokenizer_class)
        mock_pipe = self._setup_mock_pipeline(mock_hf_pipeline, [
            {"label": "POSITIVE", "score": 0.95},
            {"label": "NEGATIVE", "score": 0.85}
        ])
        
        with patch.object(Path, 'exists', return_value=True):
            # Create pipeline
            model = AutoModelForONNX.from_pretrained("/fake/path", task="text-classification")
            onnx_tokenizer = create_auto_shape_tokenizer(mock_tokenizer, model)
            
            pipe = create_pipeline(
                task="text-classification",
                model=model,
                data_processor=onnx_tokenizer
            )
            
            # Test inference
            results = pipe(["Great movie!", "Terrible film!"])
            
            assert len(results) == 2
            assert results[0]["label"] == "POSITIVE"
            assert results[1]["label"] == "NEGATIVE"
    
    @patch('optimum.onnxruntime.ORTModelForTokenClassification')
    @patch('transformers.AutoTokenizer')
    @patch('enhanced_pipeline.hf_pipeline')
    def test_token_classification_task(self, mock_hf_pipeline, mock_tokenizer_class, mock_ort_model):
        """Test token classification (NER) pipeline task."""
        mock_model = self._setup_mock_model(mock_ort_model, "token-classification")
        mock_tokenizer = self._setup_mock_tokenizer(mock_tokenizer_class)
        mock_pipe = self._setup_mock_pipeline(mock_hf_pipeline, [
            [
                {"entity": "B-PER", "score": 0.99, "word": "John", "start": 0, "end": 4},
                {"entity": "I-PER", "score": 0.98, "word": "Doe", "start": 5, "end": 8},
            ]
        ])
        
        with patch.object(Path, 'exists', return_value=True):
            model = AutoModelForONNX.from_pretrained("/fake/path", task="token-classification")
            onnx_tokenizer = create_auto_shape_tokenizer(mock_tokenizer, model)
            
            pipe = create_pipeline(
                task="token-classification",
                model=model,
                data_processor=onnx_tokenizer
            )
            
            results = pipe(["John Doe is a person."])
            
            assert len(results[0]) == 2
            assert results[0][0]["entity"] == "B-PER"
            assert results[0][0]["word"] == "John"
            assert results[0][1]["entity"] == "I-PER"
            assert results[0][1]["word"] == "Doe"
    
    @patch('optimum.onnxruntime.ORTModelForQuestionAnswering')
    @patch('transformers.AutoTokenizer')
    @patch('enhanced_pipeline.hf_pipeline')
    def test_question_answering_task(self, mock_hf_pipeline, mock_tokenizer_class, mock_ort_model):
        """Test question answering pipeline task."""
        mock_model = self._setup_mock_model(mock_ort_model, "question-answering")
        mock_tokenizer = self._setup_mock_tokenizer(mock_tokenizer_class)
        mock_pipe = self._setup_mock_pipeline(mock_hf_pipeline, {
            "answer": "Paris",
            "score": 0.95,
            "start": 25,
            "end": 30
        })
        
        with patch.object(Path, 'exists', return_value=True):
            model = AutoModelForONNX.from_pretrained("/fake/path", task="question-answering")
            onnx_tokenizer = create_auto_shape_tokenizer(mock_tokenizer, model)
            
            pipe = create_pipeline(
                task="question-answering",
                model=model,
                data_processor=onnx_tokenizer
            )
            
            result = pipe({
                "question": "What is the capital of France?",
                "context": "The capital of France is Paris."
            })
            
            assert result["answer"] == "Paris"
            assert result["score"] == 0.95
    
    @patch('optimum.onnxruntime.ORTModelForFeatureExtraction')
    @patch('transformers.AutoTokenizer')
    @patch('enhanced_pipeline.hf_pipeline')
    def test_feature_extraction_task(self, mock_hf_pipeline, mock_tokenizer_class, mock_ort_model):
        """Test feature extraction pipeline task."""
        mock_model = self._setup_mock_model(mock_ort_model, "feature-extraction")
        mock_tokenizer = self._setup_mock_tokenizer(mock_tokenizer_class)
        mock_pipe = self._setup_mock_pipeline(mock_hf_pipeline, [
            np.random.randn(1, 768).tolist()  # Feature vector
        ])
        
        with patch.object(Path, 'exists', return_value=True):
            model = AutoModelForONNX.from_pretrained("/fake/path", task="feature-extraction")
            onnx_tokenizer = create_auto_shape_tokenizer(mock_tokenizer, model)
            
            pipe = create_pipeline(
                task="feature-extraction",
                model=model,
                data_processor=onnx_tokenizer
            )
            
            results = pipe(["Extract features from this text."])
            
            assert len(results) == 1
            assert len(results[0]) == 768  # Feature dimension
    
    @patch('optimum.onnxruntime.ORTModelForMaskedLM')
    @patch('transformers.AutoTokenizer')
    @patch('enhanced_pipeline.hf_pipeline')
    def test_fill_mask_task(self, mock_hf_pipeline, mock_tokenizer_class, mock_ort_model):
        """Test fill mask pipeline task."""
        mock_model = self._setup_mock_model(mock_ort_model, "fill-mask")
        mock_tokenizer = self._setup_mock_tokenizer(mock_tokenizer_class)
        mock_pipe = self._setup_mock_pipeline(mock_hf_pipeline, [
            {
                "token_str": "amazing",
                "score": 0.3,
                "token": 1234,
                "sequence": "This movie is amazing."
            },
            {
                "token_str": "great", 
                "score": 0.25,
                "token": 5678,
                "sequence": "This movie is great."
            }
        ])
        
        with patch.object(Path, 'exists', return_value=True):
            model = AutoModelForONNX.from_pretrained("/fake/path", task="fill-mask")
            onnx_tokenizer = create_auto_shape_tokenizer(mock_tokenizer, model)
            
            pipe = create_pipeline(
                task="fill-mask",
                model=model,
                data_processor=onnx_tokenizer
            )
            
            results = pipe("This movie is [MASK].")
            
            assert len(results) == 2
            assert results[0]["token_str"] == "amazing"
            assert results[1]["token_str"] == "great"


@pytest.mark.integration
class TestVisionPipelineTasks:
    """Test various vision-based pipeline tasks."""
    
    @patch('optimum.onnxruntime.ORTModelForImageClassification')
    @patch('transformers.AutoImageProcessor')
    @patch('enhanced_pipeline.hf_pipeline')
    def test_image_classification_task(self, mock_hf_pipeline, mock_processor_class, mock_ort_model):
        """Test image classification pipeline task."""
        mock_model = self._setup_mock_model(mock_ort_model, "image-classification")
        mock_processor = self._setup_mock_image_processor(mock_processor_class)
        mock_pipe = self._setup_mock_pipeline(mock_hf_pipeline, [
            {"label": "cat", "score": 0.95},
            {"label": "dog", "score": 0.85},
            {"label": "bird", "score": 0.15}
        ])
        
        with patch.object(Path, 'exists', return_value=True):
            model = AutoModelForONNX.from_pretrained("/fake/path", task="image-classification")
            
            pipe = create_pipeline(
                task="image-classification",
                model=model,
                data_processor=mock_processor
            )
            
            results = pipe("cat_image.jpg")
            
            assert len(results) == 3
            assert results[0]["label"] == "cat"
            assert results[0]["score"] == 0.95
    
    @patch('optimum.onnxruntime.ORTModelForObjectDetection')
    @patch('transformers.AutoImageProcessor')
    @patch('enhanced_pipeline.hf_pipeline')
    def test_object_detection_task(self, mock_hf_pipeline, mock_processor_class, mock_ort_model):
        """Test object detection pipeline task."""
        mock_model = self._setup_mock_model(mock_ort_model, "object-detection")
        mock_processor = self._setup_mock_image_processor(mock_processor_class)
        mock_pipe = self._setup_mock_pipeline(mock_hf_pipeline, [
            {
                "label": "cat",
                "score": 0.95,
                "box": {"xmin": 100, "ymin": 150, "xmax": 300, "ymax": 400}
            },
            {
                "label": "dog",
                "score": 0.88,
                "box": {"xmin": 400, "ymin": 200, "xmax": 600, "ymax": 450}
            }
        ])
        
        with patch.object(Path, 'exists', return_value=True):
            model = AutoModelForONNX.from_pretrained("/fake/path", task="object-detection")
            
            pipe = create_pipeline(
                task="object-detection",
                model=model,
                data_processor=mock_processor
            )
            
            results = pipe("pets_image.jpg")
            
            assert len(results) == 2
            assert results[0]["label"] == "cat"
            assert "box" in results[0]
    
    @patch('optimum.onnxruntime.ORTModel')
    @patch('transformers.AutoImageProcessor')
    @patch('enhanced_pipeline.hf_pipeline')
    def test_image_segmentation_task(self, mock_hf_pipeline, mock_processor_class, mock_ort_model):
        """Test image segmentation pipeline task."""
        mock_model = self._setup_mock_model(mock_ort_model, "image-segmentation")
        mock_processor = self._setup_mock_image_processor(mock_processor_class)
        mock_pipe = self._setup_mock_pipeline(mock_hf_pipeline, [
            {
                "label": "cat",
                "score": 0.95,
                "mask": np.random.randint(0, 2, (224, 224), dtype=np.uint8)
            },
            {
                "label": "background",
                "score": 0.88,
                "mask": np.random.randint(0, 2, (224, 224), dtype=np.uint8)
            }
        ])
        
        with patch.object(Path, 'exists', return_value=True):
            model = AutoModelForONNX.from_pretrained("/fake/path", task="image-segmentation")
            
            pipe = create_pipeline(
                task="image-segmentation",
                model=model,
                data_processor=mock_processor
            )
            
            results = pipe("segmentation_image.jpg")
            
            assert len(results) == 2
            assert results[0]["label"] == "cat"
            assert "mask" in results[0]


@pytest.mark.integration
class TestAudioPipelineTasks:
    """Test various audio-based pipeline tasks."""
    
    @patch('optimum.onnxruntime.ORTModelForSpeechSeq2Seq')
    @patch('transformers.AutoFeatureExtractor')
    @patch('enhanced_pipeline.hf_pipeline')
    def test_automatic_speech_recognition_task(self, mock_hf_pipeline, mock_extractor_class, mock_ort_model):
        """Test automatic speech recognition pipeline task."""
        mock_model = self._setup_mock_model(mock_ort_model, "automatic-speech-recognition")
        mock_extractor = self._setup_mock_feature_extractor(mock_extractor_class)
        mock_pipe = self._setup_mock_pipeline(mock_hf_pipeline, {
            "text": "Hello, this is a transcribed audio message."
        })
        
        with patch.object(Path, 'exists', return_value=True):
            model = AutoModelForONNX.from_pretrained("/fake/path", task="automatic-speech-recognition")
            
            pipe = create_pipeline(
                task="automatic-speech-recognition",
                model=model,
                data_processor=mock_extractor
            )
            
            result = pipe("audio_file.wav")
            
            assert result["text"] == "Hello, this is a transcribed audio message."
    
    @patch('optimum.onnxruntime.ORTModelForAudioClassification')
    @patch('transformers.AutoFeatureExtractor')
    @patch('enhanced_pipeline.hf_pipeline')
    def test_audio_classification_task(self, mock_hf_pipeline, mock_extractor_class, mock_ort_model):
        """Test audio classification pipeline task."""
        mock_model = self._setup_mock_model(mock_ort_model, "audio-classification")
        mock_extractor = self._setup_mock_feature_extractor(mock_extractor_class)
        mock_pipe = self._setup_mock_pipeline(mock_hf_pipeline, [
            {"label": "speech", "score": 0.92},
            {"label": "music", "score": 0.85},
            {"label": "noise", "score": 0.15}
        ])
        
        with patch.object(Path, 'exists', return_value=True):
            model = AutoModelForONNX.from_pretrained("/fake/path", task="audio-classification")
            
            pipe = create_pipeline(
                task="audio-classification",
                model=model,
                data_processor=mock_extractor
            )
            
            results = pipe("classification_audio.wav")
            
            assert len(results) == 3
            assert results[0]["label"] == "speech"
            assert results[0]["score"] == 0.92


@pytest.mark.integration
class TestMultimodalPipelineTasks:
    """Test multimodal pipeline tasks."""
    
    @patch('optimum.onnxruntime.ORTModel')
    @patch('transformers.AutoProcessor')
    @patch('enhanced_pipeline.hf_pipeline')
    def test_image_to_text_task(self, mock_hf_pipeline, mock_processor_class, mock_ort_model):
        """Test image-to-text pipeline task."""
        mock_model = self._setup_mock_model(mock_ort_model, "image-to-text")
        mock_processor = self._setup_mock_multimodal_processor(mock_processor_class)
        mock_pipe = self._setup_mock_pipeline(mock_hf_pipeline, [
            {"generated_text": "A cat sitting on a table."}
        ])
        
        with patch.object(Path, 'exists', return_value=True):
            model = AutoModelForONNX.from_pretrained("/fake/path", task="image-to-text")
            
            pipe = create_pipeline(
                task="image-to-text",
                model=model,
                data_processor=mock_processor
            )
            
            results = pipe("image_for_captioning.jpg")
            
            assert len(results) == 1
            assert results[0]["generated_text"] == "A cat sitting on a table."
    
    @patch('optimum.onnxruntime.ORTModel')
    @patch('transformers.AutoProcessor')
    @patch('enhanced_pipeline.hf_pipeline')
    def test_visual_question_answering_task(self, mock_hf_pipeline, mock_processor_class, mock_ort_model):
        """Test visual question answering pipeline task."""
        mock_model = self._setup_mock_model(mock_ort_model, "visual-question-answering")
        mock_processor = self._setup_mock_multimodal_processor(mock_processor_class)
        mock_pipe = self._setup_mock_pipeline(mock_hf_pipeline, [
            {"answer": "brown", "score": 0.95}
        ])
        
        with patch.object(Path, 'exists', return_value=True):
            model = AutoModelForONNX.from_pretrained("/fake/path", task="visual-question-answering")
            
            pipe = create_pipeline(
                task="visual-question-answering",
                model=model,
                data_processor=mock_processor
            )
            
            result = pipe({
                "image": "dog_image.jpg",
                "question": "What color is the dog?"
            })
            
            assert len(result) == 1
            assert result[0]["answer"] == "brown"
    
    @patch('optimum.onnxruntime.ORTModel')
    @patch('transformers.AutoProcessor')
    @patch('enhanced_pipeline.hf_pipeline')
    def test_zero_shot_image_classification_task(self, mock_hf_pipeline, mock_processor_class, mock_ort_model):
        """Test zero-shot image classification pipeline task."""
        mock_model = self._setup_mock_model(mock_ort_model, "zero-shot-image-classification")
        mock_processor = self._setup_mock_multimodal_processor(mock_processor_class)
        mock_pipe = self._setup_mock_pipeline(mock_hf_pipeline, [
            {"label": "a photo of a cat", "score": 0.92},
            {"label": "a photo of a dog", "score": 0.85},
            {"label": "a photo of a bird", "score": 0.23}
        ])
        
        with patch.object(Path, 'exists', return_value=True):
            model = AutoModelForONNX.from_pretrained("/fake/path", task="zero-shot-image-classification")
            
            pipe = create_pipeline(
                task="zero-shot-image-classification",
                model=model,
                data_processor=mock_processor
            )
            
            results = pipe({
                "image": "animal_image.jpg",
                "candidate_labels": ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
            })
            
            assert len(results) == 3
            assert results[0]["label"] == "a photo of a cat"


@pytest.mark.integration
class TestGenerationPipelineTasks:
    """Test text generation pipeline tasks."""
    
    @patch('optimum.onnxruntime.ORTModelForCausalLM')
    @patch('transformers.AutoTokenizer')
    @patch('enhanced_pipeline.hf_pipeline')
    def test_text_generation_task(self, mock_hf_pipeline, mock_tokenizer_class, mock_ort_model):
        """Test text generation pipeline task."""
        mock_model = self._setup_mock_model(mock_ort_model, "text-generation")
        mock_tokenizer = self._setup_mock_tokenizer(mock_tokenizer_class)
        mock_pipe = self._setup_mock_pipeline(mock_hf_pipeline, [
            {"generated_text": "Once upon a time, there was a brave knight who saved the kingdom."}
        ])
        
        with patch.object(Path, 'exists', return_value=True):
            model = AutoModelForONNX.from_pretrained("/fake/path", task="text-generation")
            onnx_tokenizer = create_auto_shape_tokenizer(mock_tokenizer, model)
            
            pipe = create_pipeline(
                task="text-generation",
                model=model,
                data_processor=onnx_tokenizer
            )
            
            results = pipe("Once upon a time,")
            
            assert len(results) == 1
            assert "generated_text" in results[0]
    
    @patch('optimum.onnxruntime.ORTModelForSeq2SeqLM')
    @patch('transformers.AutoTokenizer')
    @patch('enhanced_pipeline.hf_pipeline')
    def test_summarization_task(self, mock_hf_pipeline, mock_tokenizer_class, mock_ort_model):
        """Test summarization pipeline task."""
        mock_model = self._setup_mock_model(mock_ort_model, "summarization")
        mock_tokenizer = self._setup_mock_tokenizer(mock_tokenizer_class)
        mock_pipe = self._setup_mock_pipeline(mock_hf_pipeline, [
            {"summary_text": "This is a concise summary of the input text."}
        ])
        
        with patch.object(Path, 'exists', return_value=True):
            model = AutoModelForONNX.from_pretrained("/fake/path", task="summarization")
            onnx_tokenizer = create_auto_shape_tokenizer(mock_tokenizer, model)
            
            pipe = create_pipeline(
                task="summarization",
                model=model,
                data_processor=onnx_tokenizer
            )
            
            long_text = "This is a very long text that needs to be summarized. " * 10
            results = pipe(long_text)
            
            assert len(results) == 1
            assert "summary_text" in results[0]
    
    @patch('optimum.onnxruntime.ORTModelForSeq2SeqLM')
    @patch('transformers.AutoTokenizer')
    @patch('enhanced_pipeline.hf_pipeline')
    def test_translation_task(self, mock_hf_pipeline, mock_tokenizer_class, mock_ort_model):
        """Test translation pipeline task."""
        mock_model = self._setup_mock_model(mock_ort_model, "translation")
        mock_tokenizer = self._setup_mock_tokenizer(mock_tokenizer_class)
        mock_pipe = self._setup_mock_pipeline(mock_hf_pipeline, [
            {"translation_text": "Bonjour le monde!"}
        ])
        
        with patch.object(Path, 'exists', return_value=True):
            model = AutoModelForONNX.from_pretrained("/fake/path", task="translation")
            onnx_tokenizer = create_auto_shape_tokenizer(mock_tokenizer, model)
            
            pipe = create_pipeline(
                task="translation",
                model=model,
                data_processor=onnx_tokenizer
            )
            
            results = pipe("Hello world!")
            
            assert len(results) == 1
            assert "translation_text" in results[0]


# Helper methods for setting up mocks
def _setup_mock_model(mock_ort_model, task):
    """Setup mock ONNX model."""
    mock_model_instance = Mock()
    mock_model_instance.task = task
    mock_model_instance.path = Path(f"/fake/{task}_model.onnx")
    mock_ort_model.from_pretrained.return_value = mock_model_instance
    return mock_model_instance

def _setup_mock_tokenizer(mock_tokenizer_class):
    """Setup mock tokenizer."""
    mock_tokenizer = Mock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.__call__ = Mock(return_value={
        "input_ids": torch.zeros((1, 128), dtype=torch.long),
        "attention_mask": torch.ones((1, 128), dtype=torch.long)
    })
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    return mock_tokenizer

def _setup_mock_image_processor(mock_processor_class):
    """Setup mock image processor."""
    mock_processor = Mock()
    mock_processor.__class__.__name__ = "ViTImageProcessor"
    mock_processor_class.from_pretrained.return_value = mock_processor
    return mock_processor

def _setup_mock_feature_extractor(mock_extractor_class):
    """Setup mock feature extractor."""
    mock_extractor = Mock()
    mock_extractor.__class__.__name__ = "Wav2Vec2FeatureExtractor"
    mock_extractor_class.from_pretrained.return_value = mock_extractor
    return mock_extractor

def _setup_mock_multimodal_processor(mock_processor_class):
    """Setup mock multimodal processor."""
    mock_processor = Mock()
    mock_processor.__class__.__name__ = "BlipProcessor"
    mock_processor.tokenizer = Mock()
    mock_processor.image_processor = Mock()
    mock_processor_class.from_pretrained.return_value = mock_processor
    return mock_processor

def _setup_mock_pipeline(mock_hf_pipeline, return_value):
    """Setup mock pipeline."""
    mock_pipeline_instance = Mock()
    mock_pipeline_instance.return_value = return_value
    mock_hf_pipeline.return_value = mock_pipeline_instance
    return mock_pipeline_instance

# Inject helper methods into test classes
for cls in [TestTextPipelineTasks, TestVisionPipelineTasks, TestAudioPipelineTasks, 
            TestMultimodalPipelineTasks, TestGenerationPipelineTasks]:
    cls._setup_mock_model = _setup_mock_model
    cls._setup_mock_tokenizer = _setup_mock_tokenizer
    cls._setup_mock_image_processor = _setup_mock_image_processor
    cls._setup_mock_feature_extractor = _setup_mock_feature_extractor
    cls._setup_mock_multimodal_processor = _setup_mock_multimodal_processor
    cls._setup_mock_pipeline = _setup_mock_pipeline


@pytest.mark.smoke
class TestPipelineTasksSmoke:
    """Smoke tests for pipeline tasks."""
    
    def test_task_to_ort_model_completeness(self):
        """Test that all tasks have corresponding ORTModel mappings."""
        from auto_model_loader import AutoModelForONNX
        
        # Test a sample of important tasks
        important_tasks = [
            "text-classification", "token-classification", "question-answering",
            "feature-extraction", "text-generation", "text2text-generation",
            "image-classification", "object-detection", "automatic-speech-recognition"
        ]
        
        for task in important_tasks:
            assert task in AutoModelForONNX.TASK_TO_ORT_MODEL, f"Task {task} not mapped to ORTModel"
    
    def test_processor_routing_completeness(self):
        """Test that processor routing covers all major task categories."""
        from enhanced_pipeline import create_pipeline
        
        # Test processor routing for different task types
        task_processor_pairs = [
            ("text-classification", "BertTokenizer", "tokenizer"),
            ("image-classification", "ViTImageProcessor", "image_processor"),
            ("automatic-speech-recognition", "Wav2Vec2FeatureExtractor", "feature_extractor"),
            ("image-to-text", "BlipProcessor", "processor")
        ]
        
        for task, processor_class, expected_param in task_processor_pairs:
            mock_processor = Mock()
            mock_processor.__class__.__name__ = processor_class
            
            with patch('enhanced_pipeline.hf_pipeline') as mock_hf_pipeline:
                create_pipeline(task=task, data_processor=mock_processor)
                args, kwargs = mock_hf_pipeline.call_args
                assert expected_param in kwargs, f"Task {task} not routed to {expected_param}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])