"""
Unit tests for AutoModelForONNX module.

Tests the auto model loader functionality including:
- Automatic task detection from model configuration
- ORTModel class selection and loading
- Support for various model types and tasks
- Error handling for missing dependencies
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tez-144_onnx_automodel_infer" / "src"))

from auto_model_loader import AutoModelForONNX


@pytest.mark.unit
class TestAutoModelForONNX:
    """Test the AutoModelForONNX class."""
    
    def test_model_type_to_tasks_mappings(self):
        """Test that model type to tasks mappings are comprehensive."""
        # Test some key model types
        bert_tasks = AutoModelForONNX.MODEL_TYPE_TO_TASKS.get("bert", [])
        assert "text-classification" in bert_tasks
        assert "token-classification" in bert_tasks
        assert "question-answering" in bert_tasks
        assert "feature-extraction" in bert_tasks
        
        gpt2_tasks = AutoModelForONNX.MODEL_TYPE_TO_TASKS.get("gpt2", [])
        assert "text-generation" in gpt2_tasks
        assert "feature-extraction" in gpt2_tasks
        
        vit_tasks = AutoModelForONNX.MODEL_TYPE_TO_TASKS.get("vit", [])
        assert "image-classification" in vit_tasks
        assert "feature-extraction" in vit_tasks
    
    def test_task_to_ort_model_mappings(self):
        """Test that task to ORTModel mappings are complete."""
        # Test text tasks
        assert AutoModelForONNX.TASK_TO_ORT_MODEL["text-classification"] == "ORTModelForSequenceClassification"
        assert AutoModelForONNX.TASK_TO_ORT_MODEL["token-classification"] == "ORTModelForTokenClassification"
        assert AutoModelForONNX.TASK_TO_ORT_MODEL["question-answering"] == "ORTModelForQuestionAnswering"
        assert AutoModelForONNX.TASK_TO_ORT_MODEL["feature-extraction"] == "ORTModelForFeatureExtraction"
        
        # Test generation tasks
        assert AutoModelForONNX.TASK_TO_ORT_MODEL["text-generation"] == "ORTModelForCausalLM"
        assert AutoModelForONNX.TASK_TO_ORT_MODEL["text2text-generation"] == "ORTModelForSeq2SeqLM"
        
        # Test vision tasks
        assert AutoModelForONNX.TASK_TO_ORT_MODEL["image-classification"] == "ORTModelForImageClassification"
        assert AutoModelForONNX.TASK_TO_ORT_MODEL["object-detection"] == "ORTModelForObjectDetection"
    
    def test_detect_task_from_architecture(self):
        """Test task detection from architecture field."""
        # Text classification
        config = {"architectures": ["BertForSequenceClassification"], "model_type": "bert"}
        task = AutoModelForONNX._detect_task(Mock(config=config), Path("/fake"))
        assert task == "text-classification"
        
        # Token classification
        config = {"architectures": ["BertForTokenClassification"], "model_type": "bert"}
        task = AutoModelForONNX._detect_task(Mock(config=config), Path("/fake"))
        assert task == "token-classification"
        
        # Question answering
        config = {"architectures": ["BertForQuestionAnswering"], "model_type": "bert"}
        task = AutoModelForONNX._detect_task(Mock(config=config), Path("/fake"))
        assert task == "question-answering"
        
        # Image classification
        config = {"architectures": ["ViTForImageClassification"], "model_type": "vit"}
        task = AutoModelForONNX._detect_task(Mock(config=config), Path("/fake"))
        assert task == "image-classification"
        
        # Causal LM
        config = {"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2"}
        task = AutoModelForONNX._detect_task(Mock(config=config), Path("/fake"))
        assert task == "text-generation"
        
        # Conditional generation
        config = {"architectures": ["BartForConditionalGeneration"], "model_type": "bart"}
        task = AutoModelForONNX._detect_task(Mock(config=config), Path("/fake"))
        assert task == "text2text-generation"
    
    def test_detect_task_from_model_type(self):
        """Test task detection from model type when architecture is ambiguous."""
        # BERT model without specific architecture
        config = {"architectures": ["BertModel"], "model_type": "bert"}
        task = AutoModelForONNX._detect_task(Mock(config=config), Path("/fake"))
        assert task == "text-classification"  # First task in BERT's list
        
        # GPT-2 model
        config = {"model_type": "gpt2"}
        task = AutoModelForONNX._detect_task(Mock(config=config), Path("/fake"))
        assert task == "text-generation"
        
        # Vision Transformer
        config = {"model_type": "vit"}
        task = AutoModelForONNX._detect_task(Mock(config=config), Path("/fake"))
        assert task == "image-classification"
    
    def test_detect_task_with_task_hint_file(self, temp_model_dir):
        """Test task detection from task.txt file."""
        model_dir = temp_model_dir / "model_with_task_hint"
        model_dir.mkdir()
        
        # Create task.txt file
        task_file = model_dir / "task.txt"
        task_file.write_text("custom-task")
        
        config = {"model_type": "unknown"}
        task = AutoModelForONNX._detect_task(Mock(config=config), model_dir)
        assert task == "custom-task"
    
    def test_detect_task_fallback_to_feature_extraction(self):
        """Test fallback to feature-extraction for unknown models."""
        config = {"model_type": "unknown-model-type"}
        task = AutoModelForONNX._detect_task(Mock(config=config), Path("/fake"))
        assert task == "feature-extraction"
    
    def test_get_ort_model_class_valid_tasks(self):
        """Test getting ORTModel class for valid tasks."""
        assert AutoModelForONNX._get_ort_model_class("text-classification") == "ORTModelForSequenceClassification"
        assert AutoModelForONNX._get_ort_model_class("image-classification") == "ORTModelForImageClassification"
        assert AutoModelForONNX._get_ort_model_class("feature-extraction") == "ORTModelForFeatureExtraction"
    
    def test_get_ort_model_class_invalid_task(self):
        """Test error handling for invalid task."""
        with pytest.raises(ValueError, match="Task 'invalid-task' is not supported"):
            AutoModelForONNX._get_ort_model_class("invalid-task")
    
    def test_list_supported_tasks(self):
        """Test listing all supported tasks."""
        tasks = AutoModelForONNX.list_supported_tasks()
        assert isinstance(tasks, list)
        assert "text-classification" in tasks
        assert "image-classification" in tasks
        assert "feature-extraction" in tasks
        assert len(tasks) > 20  # Should have many supported tasks
    
    def test_list_supported_model_types(self):
        """Test listing all supported model types."""
        model_types = AutoModelForONNX.list_supported_model_types()
        assert isinstance(model_types, list)
        assert "bert" in model_types
        assert "gpt2" in model_types
        assert "vit" in model_types
        assert len(model_types) > 50  # Should have many supported model types
    
    @patch('auto_model_loader.AutoConfig')
    @patch('optimum.onnxruntime.ORTModelForFeatureExtraction')
    def test_from_pretrained_with_auto_detection(self, mock_ort_model, mock_config):
        """Test from_pretrained with automatic task detection."""
        # Setup mocks
        mock_config_obj = Mock()
        mock_config_obj.model_type = "bert"
        mock_config_obj.architectures = ["BertModel"]
        mock_config.from_pretrained.return_value = mock_config_obj
        
        mock_model_instance = Mock()
        mock_model_instance.task = "feature-extraction"
        mock_ort_model.from_pretrained.return_value = mock_model_instance
        
        # Test
        with patch.object(Path, 'exists', return_value=True):
            result = AutoModelForONNX.from_pretrained("/fake/path")
        
        assert result.task == "feature-extraction"
        mock_config.from_pretrained.assert_called_once_with(Path("/fake/path"))
        mock_ort_model.from_pretrained.assert_called_once()
    
    @patch('auto_model_loader.AutoConfig')
    @patch('optimum.onnxruntime.ORTModelForSequenceClassification')
    def test_from_pretrained_with_explicit_task(self, mock_ort_model, mock_config):
        """Test from_pretrained with explicitly specified task."""
        # Setup mocks
        mock_config_obj = Mock()
        mock_config.from_pretrained.return_value = mock_config_obj
        
        mock_model_instance = Mock()
        mock_model_instance.task = "text-classification"
        mock_ort_model.from_pretrained.return_value = mock_model_instance
        
        # Test
        with patch.object(Path, 'exists', return_value=True):
            result = AutoModelForONNX.from_pretrained("/fake/path", task="text-classification")
        
        assert result.task == "text-classification"
        mock_ort_model.from_pretrained.assert_called_once()
    
    @patch('auto_model_loader.AutoConfig')
    def test_from_pretrained_path_not_exists(self, mock_config):
        """Test error handling when model path doesn't exist."""
        with patch.object(Path, 'exists', return_value=False):
            with pytest.raises(ValueError, match="Model path does not exist"):
                AutoModelForONNX.from_pretrained("/nonexistent/path")
    
    @patch('auto_model_loader.AutoConfig')
    def test_from_pretrained_config_not_exists(self, mock_config):
        """Test error handling when config.json doesn't exist."""
        with patch.object(Path, 'exists', side_effect=lambda x: str(x).endswith("/fake/path")):
            with pytest.raises(ValueError, match="config.json not found"):
                AutoModelForONNX.from_pretrained("/fake/path")
    
    def test_from_pretrained_optimum_not_installed(self):
        """Test error handling when Optimum is not installed."""
        with patch.object(Path, 'exists', return_value=True):
            with patch('auto_model_loader.AutoConfig') as mock_config:
                mock_config_obj = Mock()
                mock_config_obj.model_type = "bert"
                mock_config.from_pretrained.return_value = mock_config_obj
                
                # Mock import error
                with patch('builtins.__import__', side_effect=ImportError("No module named 'optimum'")):
                    with pytest.raises(ImportError, match="Optimum is not installed"):
                        AutoModelForONNX.from_pretrained("/fake/path")
    
    @patch('auto_model_loader.AutoConfig')
    def test_from_pretrained_unknown_ort_model_class(self, mock_config):
        """Test error handling for unknown ORTModel class."""
        mock_config_obj = Mock()
        mock_config_obj.model_type = "unknown"
        mock_config.from_pretrained.return_value = mock_config_obj
        
        # Patch to simulate missing ORTModel class
        with patch.object(Path, 'exists', return_value=True):
            with patch('auto_model_loader.AutoModelForONNX._get_ort_model_class', return_value="NonExistentORTModel"):
                with patch('optimum.onnxruntime.ORTModel'):
                    with patch('optimum.onnxruntime', spec=[]):  # Empty spec means no attributes
                        with pytest.raises(ValueError, match="ORTModel class .* not found"):
                            AutoModelForONNX.from_pretrained("/fake/path")
    
    @patch('auto_model_loader.AutoConfig')
    @patch('optimum.onnxruntime.ORTModelForFeatureExtraction')
    def test_from_pretrained_with_provider_options(self, mock_ort_model, mock_config):
        """Test from_pretrained with provider and session options."""
        # Setup mocks
        mock_config_obj = Mock()
        mock_config_obj.model_type = "bert"
        mock_config.from_pretrained.return_value = mock_config_obj
        
        mock_model_instance = Mock()
        mock_ort_model.from_pretrained.return_value = mock_model_instance
        
        # Test with CUDA provider
        with patch.object(Path, 'exists', return_value=True):
            result = AutoModelForONNX.from_pretrained(
                "/fake/path",
                provider="CUDAExecutionProvider",
                session_options={"option": "value"},
                provider_options={"device_id": 0}
            )
        
        # Check that provider was passed correctly
        call_kwargs = mock_ort_model.from_pretrained.call_args[1]
        assert "CUDAExecutionProvider" in call_kwargs["provider"]
        assert "CPUExecutionProvider" in call_kwargs["provider"]
        assert call_kwargs["session_options"] == {"option": "value"}
        assert call_kwargs["provider_options"] == {"device_id": 0}


@pytest.mark.unit
class TestTaskDetectionDetails:
    """Test detailed task detection scenarios."""
    
    def test_detect_task_with_explicit_task_in_config(self):
        """Test task detection when task is explicitly in config."""
        config = Mock()
        config.task = "custom-task"
        config.model_type = "bert"
        
        task = AutoModelForONNX._detect_task(config, Path("/fake"))
        assert task == "custom-task"
    
    def test_detect_task_architecture_patterns(self):
        """Test various architecture patterns for task detection."""
        test_cases = [
            (["RobertaForSequenceClassification"], "text-classification"),
            (["AlbertForTokenClassification"], "token-classification"),
            (["DistilBertForQuestionAnswering"], "question-answering"),
            (["GPT2LMHeadModel"], "text-generation"),
            (["T5ForConditionalGeneration"], "text2text-generation"),
            (["ViTForImageClassification"], "image-classification"),
            (["BertModel"], "feature-extraction"),  # Generic model -> feature extraction
        ]
        
        for architectures, expected_task in test_cases:
            config = Mock()
            config.model_type = "test"
            config.architectures = architectures
            
            task = AutoModelForONNX._detect_task(config, Path("/fake"))
            assert task == expected_task, f"Failed for {architectures}"
    
    def test_detect_task_model_type_variants(self):
        """Test task detection for model type variants."""
        # Test that model types with variants are handled
        test_cases = [
            ("deberta-v2", "text-classification"),
            ("deberta-v3", "text-classification"),
            ("gpt-neo", "text-generation"),
            ("xlm-roberta", "text-classification"),
        ]
        
        for model_type, expected_task in test_cases:
            config = Mock()
            config.model_type = model_type
            config.architectures = []
            
            task = AutoModelForONNX._detect_task(config, Path("/fake"))
            assert task == expected_task, f"Failed for {model_type}"


@pytest.mark.integration
class TestAutoModelForONNXIntegration:
    """Integration tests for AutoModelForONNX."""
    
    @patch('optimum.onnxruntime.ORTModelForFeatureExtraction')
    def test_end_to_end_model_loading(self, mock_ort_model, mock_onnx_model_with_file):
        """Test end-to-end model loading workflow."""
        # Setup mock
        mock_model_instance = Mock()
        mock_model_instance.task = "feature-extraction"
        mock_ort_model.from_pretrained.return_value = mock_model_instance
        
        # Test loading
        result = AutoModelForONNX.from_pretrained(str(mock_onnx_model_with_file))
        
        assert result.task == "feature-extraction"
        mock_ort_model.from_pretrained.assert_called_once()
    
    @patch('optimum.onnxruntime.ORTModelForSequenceClassification')
    def test_text_classification_model_loading(self, mock_ort_model, temp_model_dir):
        """Test loading text classification model."""
        # Create model directory with text classification config
        model_dir = temp_model_dir / "text_classifier"
        model_dir.mkdir()
        
        config = {
            "model_type": "bert",
            "architectures": ["BertForSequenceClassification"],
            "num_labels": 2
        }
        
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)
        
        # Setup mock
        mock_model_instance = Mock()
        mock_model_instance.task = "text-classification"
        mock_ort_model.from_pretrained.return_value = mock_model_instance
        
        # Test loading
        result = AutoModelForONNX.from_pretrained(str(model_dir))
        
        assert result.task == "text-classification"
        mock_ort_model.from_pretrained.assert_called_once()
    
    @patch('optimum.onnxruntime.ORTModelForImageClassification')
    def test_image_classification_model_loading(self, mock_ort_model, temp_model_dir):
        """Test loading image classification model."""
        # Create model directory with image classification config
        model_dir = temp_model_dir / "image_classifier"
        model_dir.mkdir()
        
        config = {
            "model_type": "vit",
            "architectures": ["ViTForImageClassification"],
            "num_labels": 1000,
            "image_size": 224
        }
        
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)
        
        # Setup mock
        mock_model_instance = Mock()
        mock_model_instance.task = "image-classification"
        mock_ort_model.from_pretrained.return_value = mock_model_instance
        
        # Test loading
        result = AutoModelForONNX.from_pretrained(str(model_dir))
        
        assert result.task == "image-classification"
        mock_ort_model.from_pretrained.assert_called_once()


@pytest.mark.smoke
class TestAutoModelForONNXSmoke:
    """Smoke tests for basic functionality."""
    
    def test_class_attributes_exist(self):
        """Test that essential class attributes exist."""
        assert hasattr(AutoModelForONNX, 'MODEL_TYPE_TO_TASKS')
        assert hasattr(AutoModelForONNX, 'TASK_TO_ORT_MODEL')
        assert isinstance(AutoModelForONNX.MODEL_TYPE_TO_TASKS, dict)
        assert isinstance(AutoModelForONNX.TASK_TO_ORT_MODEL, dict)
    
    def test_static_methods_exist(self):
        """Test that essential static methods exist."""
        assert hasattr(AutoModelForONNX, 'from_pretrained')
        assert hasattr(AutoModelForONNX, 'list_supported_tasks')
        assert hasattr(AutoModelForONNX, 'list_supported_model_types')
        assert hasattr(AutoModelForONNX, '_detect_task')
        assert hasattr(AutoModelForONNX, '_get_ort_model_class')
    
    def test_mappings_not_empty(self):
        """Test that key mappings are not empty."""
        assert len(AutoModelForONNX.MODEL_TYPE_TO_TASKS) > 0
        assert len(AutoModelForONNX.TASK_TO_ORT_MODEL) > 0
    
    def test_common_tasks_supported(self):
        """Test that common tasks are supported."""
        common_tasks = [
            "text-classification", "token-classification", "question-answering",
            "feature-extraction", "text-generation", "image-classification"
        ]
        
        supported_tasks = AutoModelForONNX.list_supported_tasks()
        for task in common_tasks:
            assert task in supported_tasks, f"Common task {task} not supported"
    
    def test_common_model_types_supported(self):
        """Test that common model types are supported."""
        common_model_types = ["bert", "gpt2", "t5", "vit", "roberta", "albert"]
        
        supported_types = AutoModelForONNX.list_supported_model_types()
        for model_type in common_model_types:
            assert model_type in supported_types, f"Common model type {model_type} not supported"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])