"""
Sanity tests for ONNX inference implementation.

These tests verify that the implementation behaves correctly
under normal conditions and follows expected patterns.
Basic functionality checks and regression prevention.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import torch
import numpy as np

# Import the modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tez-144_onnx_automodel_infer" / "src"))

from onnx_tokenizer import ONNXTokenizer, create_auto_shape_tokenizer
from enhanced_pipeline import create_pipeline, _detect_processor_type
from auto_model_loader import AutoModelForONNX
from inference_utils import detect_model_task, load_preprocessor


@pytest.mark.sanity
class TestONNXTokenizerSanity:
    """Sanity tests for ONNXTokenizer functionality."""
    
    def test_tokenizer_produces_fixed_shapes(self):
        """Test that tokenizer always produces fixed shapes."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.__call__ = Mock(return_value={
            "input_ids": torch.zeros((2, 64), dtype=torch.long),
            "attention_mask": torch.ones((2, 64), dtype=torch.long)
        })
        
        onnx_tokenizer = ONNXTokenizer(
            tokenizer=mock_tokenizer,
            fixed_batch_size=2,
            fixed_sequence_length=64
        )
        
        # Test various inputs - should always return same shape
        test_inputs = [
            "Single text",
            ["Text 1", "Text 2"],
            ["Only one text"],  # Less than batch size
            ["Text 1", "Text 2", "Text 3", "Text 4"]  # More than batch size
        ]
        
        for text_input in test_inputs:
            result = onnx_tokenizer(text_input)
            assert result["input_ids"].shape == (2, 64)
            assert result["attention_mask"].shape == (2, 64)
    
    def test_tokenizer_passthrough_methods(self):
        """Test that tokenizer methods pass through correctly."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.vocab_size = 30000
        mock_tokenizer.model_max_length = 512
        mock_tokenizer.decode.return_value = "decoded text"
        mock_tokenizer.batch_decode.return_value = ["decoded text 1", "decoded text 2"]
        
        onnx_tokenizer = ONNXTokenizer(
            tokenizer=mock_tokenizer,
            fixed_batch_size=1,
            fixed_sequence_length=128
        )
        
        # Test attribute access
        assert onnx_tokenizer.vocab_size == 30000
        assert onnx_tokenizer.model_max_length == 512
        
        # Test decode methods
        assert onnx_tokenizer.decode([1, 2, 3]) == "decoded text"
        assert onnx_tokenizer.batch_decode([[1, 2], [3, 4]]) == ["decoded text 1", "decoded text 2"]
    
    def test_auto_shape_detection_fallback(self):
        """Test graceful fallback when auto-detection fails."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        
        # Mock an ONNX model that will cause detection to fail
        mock_onnx_model = Mock()
        mock_onnx_model.path = None
        mock_onnx_model.model_path = None
        # Remove model attribute to cause error
        del mock_onnx_model.model
        
        # Should fall back to defaults without crashing
        onnx_tokenizer = ONNXTokenizer(
            tokenizer=mock_tokenizer,
            onnx_model=mock_onnx_model
        )
        
        assert onnx_tokenizer.fixed_batch_size == 1
        assert onnx_tokenizer.fixed_sequence_length == 128


@pytest.mark.sanity
class TestEnhancedPipelineSanity:
    """Sanity tests for enhanced pipeline functionality."""
    
    def test_processor_routing_consistency(self):
        """Test that processor routing is consistent and predictable."""
        test_cases = [
            # (class_name, expected_type)
            ("BertTokenizer", "tokenizer"),
            ("BertTokenizerFast", "tokenizer"),
            ("ViTImageProcessor", "image_processor"),
            ("Wav2Vec2FeatureExtractor", "feature_extractor"),
            ("CLIPProcessor", "processor"),
            ("BlipProcessor", "processor")
        ]
        
        for class_name, expected_type in test_cases:
            mock_processor = Mock()
            mock_processor.__class__.__name__ = class_name
            
            detected_type = _detect_processor_type(mock_processor)
            assert detected_type == expected_type, \
                f"Processor {class_name} incorrectly detected as {detected_type}, expected {expected_type}"
    
    def test_task_based_routing_fallback(self):
        """Test that task-based routing works as fallback."""
        # Unknown processor should be routed based on task
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "UnknownCustomProcessor"
        
        task_routing_tests = [
            ("text-classification", "tokenizer"),
            ("image-classification", "image_processor"),
            ("automatic-speech-recognition", "feature_extractor"),
            ("image-to-text", "processor")
        ]
        
        for task, expected_param in task_routing_tests:
            with patch('enhanced_pipeline.hf_pipeline') as mock_hf_pipeline:
                mock_hf_pipeline.return_value = Mock()
                
                create_pipeline(task=task, data_processor=mock_processor)
                
                args, kwargs = mock_hf_pipeline.call_args
                assert expected_param in kwargs, \
                    f"Task {task} not routed to {expected_param}"
                assert kwargs[expected_param] == mock_processor
    
    def test_pipeline_parameter_preservation(self):
        """Test that all pipeline parameters are preserved."""
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "BertTokenizer"
        
        with patch('enhanced_pipeline.hf_pipeline') as mock_hf_pipeline:
            mock_hf_pipeline.return_value = Mock()
            
            # Test with many parameters
            create_pipeline(
                task="text-classification",
                model="bert-base-uncased",
                data_processor=mock_processor,
                config="config",
                framework="pt",
                revision="main",
                use_fast=True,
                token="token",
                device=0,
                device_map="auto",
                torch_dtype="float16",
                trust_remote_code=True,
                model_kwargs={"key": "value"},
                custom_param="custom_value"
            )
            
            args, kwargs = mock_hf_pipeline.call_args
            
            # Check that all parameters are passed through
            assert kwargs["task"] == "text-classification"
            assert kwargs["model"] == "bert-base-uncased"
            assert kwargs["tokenizer"] == mock_processor
            assert kwargs["config"] == "config"
            assert kwargs["framework"] == "pt"
            assert kwargs["revision"] == "main"
            assert kwargs["use_fast"] is True
            assert kwargs["token"] == "token"
            assert kwargs["device"] == 0
            assert kwargs["device_map"] == "auto"
            assert kwargs["torch_dtype"] == "float16"
            assert kwargs["trust_remote_code"] is True
            assert kwargs["model_kwargs"] == {"key": "value"}
            assert kwargs["custom_param"] == "custom_value"


@pytest.mark.sanity
class TestAutoModelLoaderSanity:
    """Sanity tests for AutoModelForONNX functionality."""
    
    def test_task_detection_precedence(self):
        """Test that task detection follows correct precedence."""
        # 1. Explicit task in config should have highest precedence
        config_with_task = Mock()
        config_with_task.task = "explicit-task"
        config_with_task.model_type = "bert"
        config_with_task.architectures = ["BertForSequenceClassification"]
        
        task = AutoModelForONNX._detect_task(config_with_task, Path("/fake"))
        assert task == "explicit-task"
        
        # 2. Architecture should have precedence over model type
        config_with_arch = Mock()
        config_with_arch.model_type = "bert"  # Would suggest feature-extraction
        config_with_arch.architectures = ["BertForSequenceClassification"]  # Should suggest text-classification
        del config_with_arch.task
        
        task = AutoModelForONNX._detect_task(config_with_arch, Path("/fake"))
        assert task == "text-classification"
        
        # 3. Model type should be used if architecture is generic
        config_generic_arch = Mock()
        config_generic_arch.model_type = "gpt2"
        config_generic_arch.architectures = ["GPT2Model"]  # Generic model
        del config_generic_arch.task
        
        task = AutoModelForONNX._detect_task(config_generic_arch, Path("/fake"))
        assert task == "text-generation"  # From model type
    
    def test_model_type_variants_handling(self):
        """Test handling of model type variants."""
        variant_tests = [
            ("deberta-v2", "text-classification"),
            ("deberta-v3", "text-classification"),
            ("xlm-roberta", "text-classification"),
            ("gpt-neo", "text-generation"),
            ("mt5", "text2text-generation")
        ]
        
        for model_type, expected_task in variant_tests:
            config = Mock()
            config.model_type = model_type
            config.architectures = []
            del config.task
            
            task = AutoModelForONNX._detect_task(config, Path("/fake"))
            assert task == expected_task, \
                f"Model type variant {model_type} not handled correctly"
    
    def test_ort_model_class_mapping_completeness(self):
        """Test that all tasks have valid ORTModel class mappings."""
        supported_tasks = AutoModelForONNX.list_supported_tasks()
        
        for task in supported_tasks:
            ort_class = AutoModelForONNX._get_ort_model_class(task)
            assert ort_class is not None
            assert isinstance(ort_class, str)
            assert "ORT" in ort_class  # Should be an ORTModel class
    
    def test_supported_tasks_consistency(self):
        """Test consistency between task mappings and supported tasks."""
        supported_tasks = set(AutoModelForONNX.list_supported_tasks())
        mapped_tasks = set(AutoModelForONNX.TASK_TO_ORT_MODEL.keys())
        
        assert supported_tasks == mapped_tasks, \
            "Mismatch between supported tasks and task mappings"
    
    def test_model_type_tasks_validity(self):
        """Test that all tasks in model type mappings are supported."""
        supported_tasks = set(AutoModelForONNX.list_supported_tasks())
        
        for model_type, tasks in AutoModelForONNX.MODEL_TYPE_TO_TASKS.items():
            for task in tasks:
                assert task in supported_tasks, \
                    f"Task {task} for model type {model_type} is not in supported tasks"


@pytest.mark.sanity
class TestInferenceUtilsSanity:
    """Sanity tests for inference utilities."""
    
    @patch('inference_utils.AutoConfig')
    def test_detect_model_task_integration(self, mock_config):
        """Test task detection integration."""
        mock_config_obj = Mock()
        mock_config_obj.model_type = "bert"
        mock_config_obj.architectures = ["BertForSequenceClassification"]
        mock_config.from_pretrained.return_value = mock_config_obj
        
        task = detect_model_task("/fake/path")
        assert task == "text-classification"
        mock_config.from_pretrained.assert_called_once_with(Path("/fake/path"))
    
    def test_load_preprocessor_precedence(self):
        """Test preprocessor loading precedence."""
        # Test that processor has highest precedence
        with patch('inference_utils.AutoProcessor') as mock_auto_processor:
            mock_processor = Mock()
            mock_auto_processor.from_pretrained.return_value = mock_processor
            
            result = load_preprocessor("/fake/path")
            assert result == mock_processor
            mock_auto_processor.from_pretrained.assert_called_once()
    
    def test_load_preprocessor_fallback_chain(self):
        """Test preprocessor loading fallback chain."""
        # Test fallback chain: processor -> tokenizer -> image_processor -> feature_extractor
        with patch('inference_utils.AutoProcessor') as mock_processor:
            with patch('inference_utils.AutoTokenizer') as mock_tokenizer:
                with patch('inference_utils.AutoImageProcessor') as mock_image_proc:
                    with patch('inference_utils.AutoFeatureExtractor') as mock_feature_ext:
                        
                        # Processor fails, tokenizer succeeds
                        mock_processor.from_pretrained.side_effect = Exception("No processor")
                        mock_tokenizer_obj = Mock()
                        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_obj
                        
                        result = load_preprocessor("/fake/path")
                        assert result == mock_tokenizer_obj
    
    def test_load_preprocessor_explicit_type(self):
        """Test loading preprocessor with explicit type."""
        with patch('inference_utils.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer_obj = Mock()
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_obj
            
            result = load_preprocessor("/fake/path", preprocessor_type="tokenizer")
            assert result == mock_tokenizer_obj
            mock_tokenizer.from_pretrained.assert_called_once()


@pytest.mark.sanity
class TestConfigurationSanity:
    """Sanity tests for configuration handling."""
    
    def test_universal_config_input_names(self):
        """Test that universal config generates reasonable input names."""
        from onnx_config.universal_config import UniversalOnnxConfig
        
        # Test text model
        text_config = {"model_type": "bert", "hidden_size": 768}
        onnx_config = UniversalOnnxConfig(config=text_config)
        input_names = onnx_config.get_input_names()
        
        assert isinstance(input_names, list)
        assert len(input_names) > 0
        # Should include common text inputs
        assert any("input_ids" in name or "input" in name.lower() for name in input_names)
    
    def test_universal_config_output_names(self):
        """Test that universal config generates reasonable output names."""
        from onnx_config.universal_config import UniversalOnnxConfig
        
        text_config = {"model_type": "bert", "hidden_size": 768}
        onnx_config = UniversalOnnxConfig(config=text_config, task="text-classification")
        output_names = onnx_config.get_output_names()
        
        assert isinstance(output_names, list)
        assert len(output_names) > 0
        # Should include logits for classification
        assert any("logits" in name.lower() for name in output_names)
    
    def test_task_detector_families(self):
        """Test task family detection consistency."""
        from onnx_config.task_detector import TaskDetector
        
        family_tests = [
            ("text-classification", "classification"),
            ("token-classification", "classification"),
            ("image-classification", "classification"),
            ("text-generation", "generation"),
            ("text2text-generation", "generation"),
            ("question-answering", "question-answering"),
            ("visual-question-answering", "question-answering"),
            ("feature-extraction", "feature-extraction"),
            ("object-detection", "detection"),
            ("image-segmentation", "segmentation")
        ]
        
        for task, expected_family in family_tests:
            family = TaskDetector.get_task_family(task)
            assert family == expected_family, \
                f"Task {task} incorrectly assigned to family {family}, expected {expected_family}"
    
    def test_past_key_values_detection(self):
        """Test past key values requirement detection."""
        from onnx_config.task_detector import TaskDetector
        
        # Generation tasks with appropriate models should require past key values
        generation_models = ["gpt2", "gpt-neo", "t5", "bart"]
        for model_type in generation_models:
            requires_pkv = TaskDetector.requires_past_key_values("text-generation", model_type)
            assert requires_pkv, f"Model {model_type} should require past key values for generation"
        
        # Non-generation tasks should not require past key values
        non_gen_tasks = ["text-classification", "feature-extraction", "image-classification"]
        for task in non_gen_tasks:
            requires_pkv = TaskDetector.requires_past_key_values(task, "bert")
            assert not requires_pkv, f"Task {task} should not require past key values"


@pytest.mark.sanity
class TestDataFlowSanity:
    """Sanity tests for data flow through the system."""
    
    @patch('optimum.onnxruntime.ORTModelForFeatureExtraction')
    @patch('transformers.AutoTokenizer')
    def test_end_to_end_data_flow(self, mock_tokenizer_class, mock_ort_model):
        """Test basic end-to-end data flow."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_model_instance.task = "feature-extraction"
        mock_model_instance.path = Path("/fake/model.onnx")
        mock_ort_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.__call__ = Mock(return_value={
            "input_ids": torch.zeros((1, 128), dtype=torch.long),
            "attention_mask": torch.ones((1, 128), dtype=torch.long)
        })
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        with patch('onnx_tokenizer.parse_onnx_input_shapes') as mock_parse:
            mock_parse.return_value = {"input_ids": [1, 128], "attention_mask": [1, 128]}
            
            with patch.object(Path, 'exists', return_value=True):
                # Test the flow: model loading -> tokenizer creation -> text processing
                model = AutoModelForONNX.from_pretrained("/fake/path")
                base_tokenizer = mock_tokenizer_class.from_pretrained("/fake/path")
                onnx_tokenizer = create_auto_shape_tokenizer(base_tokenizer, model)
                
                # Process text
                result = onnx_tokenizer("Test input text")
                
                # Verify data flow
                assert model.task == "feature-extraction"
                assert onnx_tokenizer.fixed_batch_size == 1
                assert onnx_tokenizer.fixed_sequence_length == 128
                assert result["input_ids"].shape == (1, 128)
                assert result["attention_mask"].shape == (1, 128)
    
    def test_tensor_type_consistency(self):
        """Test that tensor types are handled consistently."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        
        # Test different return tensor types
        tensor_types = ["pt", "np"]
        
        for tensor_type in tensor_types:
            if tensor_type == "pt":
                mock_tokenizer.__call__ = Mock(return_value={
                    "input_ids": torch.zeros((1, 128), dtype=torch.long),
                    "attention_mask": torch.ones((1, 128), dtype=torch.long)
                })
            else:  # numpy
                mock_tokenizer.__call__ = Mock(return_value={
                    "input_ids": np.zeros((1, 128), dtype=np.int64),
                    "attention_mask": np.ones((1, 128), dtype=np.int64)
                })
            
            onnx_tokenizer = ONNXTokenizer(
                tokenizer=mock_tokenizer,
                fixed_batch_size=1,
                fixed_sequence_length=128
            )
            
            result = onnx_tokenizer("Test text", return_tensors=tensor_type)
            
            if tensor_type == "pt":
                assert isinstance(result["input_ids"], torch.Tensor)
            else:
                assert isinstance(result["input_ids"], np.ndarray)


@pytest.mark.sanity
class TestRegressionPrevention:
    """Tests to prevent common regressions."""
    
    def test_tokenizer_shape_validation_strictness(self):
        """Test that shape validation is strict and catches errors."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        
        onnx_tokenizer = ONNXTokenizer(
            tokenizer=mock_tokenizer,
            fixed_batch_size=2,
            fixed_sequence_length=64
        )
        
        # Create wrong-shaped tensors
        wrong_batch = {
            "input_ids": torch.zeros((3, 64)),  # Wrong batch size
            "attention_mask": torch.ones((2, 64))
        }
        
        wrong_sequence = {
            "input_ids": torch.zeros((2, 128)),  # Wrong sequence length
            "attention_mask": torch.ones((2, 64))
        }
        
        # Should raise errors for wrong shapes
        with pytest.raises(ValueError):
            onnx_tokenizer._validate_shapes(wrong_batch)
            
        with pytest.raises(ValueError):
            onnx_tokenizer._validate_shapes(wrong_sequence)
    
    def test_model_type_case_sensitivity(self):
        """Test that model type detection is case-insensitive where appropriate."""
        from onnx_config.task_detector import TaskDetector
        
        # Test case variations
        test_cases = [
            ("bert", "BERT"),
            ("gpt2", "GPT2"),
            ("roberta", "RoBERTa")
        ]
        
        for lower_case, mixed_case in test_cases:
            config_lower = {"model_type": lower_case}
            config_mixed = {"model_type": mixed_case.lower()}  # TaskDetector converts to lower
            
            task_lower = TaskDetector.detect_from_config(config_lower)
            task_mixed = TaskDetector.detect_from_config(config_mixed)
            
            assert task_lower == task_mixed, \
                f"Case sensitivity issue with model type {lower_case}/{mixed_case}"
    
    def test_empty_input_handling(self):
        """Test handling of edge cases like empty inputs."""
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.__call__ = Mock(return_value={
            "input_ids": torch.zeros((1, 128), dtype=torch.long),
            "attention_mask": torch.ones((1, 128), dtype=torch.long)
        })
        
        onnx_tokenizer = ONNXTokenizer(
            tokenizer=mock_tokenizer,
            fixed_batch_size=1,
            fixed_sequence_length=128
        )
        
        # Test empty string
        result = onnx_tokenizer("")
        assert result["input_ids"].shape == (1, 128)
        
        # Test empty list (should be padded to batch size)
        onnx_tokenizer.fixed_batch_size = 2
        result = onnx_tokenizer([])
        assert result["input_ids"].shape == (2, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "sanity"])