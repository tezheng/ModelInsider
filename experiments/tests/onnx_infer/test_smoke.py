"""
Smoke tests for ONNX inference implementation.

These tests verify that basic functionality works without crashes
and that the main components can be imported and instantiated.
Quick validation tests for CI/CD pipelines.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Import the modules under test
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tez-144_onnx_automodel_infer" / "src"))


@pytest.mark.smoke
class TestImports:
    """Test that all modules can be imported without errors."""
    
    def test_import_onnx_tokenizer(self):
        """Test importing onnx_tokenizer module."""
        try:
            import onnx_tokenizer
            assert hasattr(onnx_tokenizer, 'ONNXTokenizer')
            assert hasattr(onnx_tokenizer, 'parse_onnx_input_shapes')
            assert hasattr(onnx_tokenizer, 'create_auto_shape_tokenizer')
        except ImportError as e:
            pytest.fail(f"Failed to import onnx_tokenizer: {e}")
    
    def test_import_enhanced_pipeline(self):
        """Test importing enhanced_pipeline module."""
        try:
            import enhanced_pipeline
            assert hasattr(enhanced_pipeline, 'create_pipeline')
            assert hasattr(enhanced_pipeline, 'pipeline')
            assert hasattr(enhanced_pipeline, '_detect_processor_type')
        except ImportError as e:
            pytest.fail(f"Failed to import enhanced_pipeline: {e}")
    
    def test_import_auto_model_loader(self):
        """Test importing auto_model_loader module."""
        try:
            import auto_model_loader
            assert hasattr(auto_model_loader, 'AutoModelForONNX')
        except ImportError as e:
            pytest.fail(f"Failed to import auto_model_loader: {e}")
    
    def test_import_inference_utils(self):
        """Test importing inference_utils module."""
        try:
            import inference_utils
            assert hasattr(inference_utils, 'detect_model_task')
            assert hasattr(inference_utils, 'load_preprocessor')
            assert hasattr(inference_utils, 'create_inference_pipeline')
        except ImportError as e:
            pytest.fail(f"Failed to import inference_utils: {e}")
    
    def test_import_onnx_config_modules(self):
        """Test importing onnx_config modules."""
        try:
            from onnx_config import universal_config, task_detector
            assert hasattr(universal_config, 'UniversalOnnxConfig')
            assert hasattr(task_detector, 'TaskDetector')
        except ImportError as e:
            pytest.fail(f"Failed to import onnx_config modules: {e}")


@pytest.mark.smoke
class TestBasicInstantiation:
    """Test that main classes can be instantiated without crashes."""
    
    def test_onnx_tokenizer_instantiation(self):
        """Test ONNXTokenizer can be instantiated."""
        from onnx_tokenizer import ONNXTokenizer
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        
        # Should not crash
        tokenizer = ONNXTokenizer(
            tokenizer=mock_tokenizer,
            fixed_batch_size=1,
            fixed_sequence_length=128
        )
        
        assert tokenizer is not None
        assert tokenizer.fixed_batch_size == 1
        assert tokenizer.fixed_sequence_length == 128
    
    def test_auto_model_for_onnx_class_attributes(self):
        """Test AutoModelForONNX has required class attributes."""
        from auto_model_loader import AutoModelForONNX
        
        assert hasattr(AutoModelForONNX, 'MODEL_TYPE_TO_TASKS')
        assert hasattr(AutoModelForONNX, 'TASK_TO_ORT_MODEL')
        assert hasattr(AutoModelForONNX, 'from_pretrained')
        assert hasattr(AutoModelForONNX, 'list_supported_tasks')
        assert hasattr(AutoModelForONNX, 'list_supported_model_types')
        
        # Check that mappings are not empty
        assert len(AutoModelForONNX.MODEL_TYPE_TO_TASKS) > 0
        assert len(AutoModelForONNX.TASK_TO_ORT_MODEL) > 0
    
    def test_universal_onnx_config_instantiation(self):
        """Test UniversalOnnxConfig can be instantiated."""
        from onnx_config.universal_config import UniversalOnnxConfig
        
        config = {"model_type": "bert", "hidden_size": 768}
        
        # Should not crash
        onnx_config = UniversalOnnxConfig(config=config)
        
        assert onnx_config is not None
        assert onnx_config.model_type == "bert"
        assert onnx_config.task is not None
    
    def test_task_detector_instantiation(self):
        """Test TaskDetector can be used."""
        from onnx_config.task_detector import TaskDetector
        
        config = {"model_type": "bert", "architectures": ["BertForSequenceClassification"]}
        
        # Should not crash
        task = TaskDetector.detect_from_config(config)
        
        assert task is not None
        assert isinstance(task, str)


@pytest.mark.smoke
class TestBasicFunctionality:
    """Test basic functionality without external dependencies."""
    
    def test_create_pipeline_basic(self):
        """Test create_pipeline function basic usage."""
        from enhanced_pipeline import create_pipeline
        
        with patch('enhanced_pipeline.hf_pipeline') as mock_hf_pipeline:
            mock_hf_pipeline.return_value = Mock()
            
            # Should not crash
            result = create_pipeline(
                task="text-classification",
                model="bert-base-uncased"
            )
            
            assert result is not None
            mock_hf_pipeline.assert_called_once()
    
    def test_processor_type_detection(self):
        """Test processor type detection functionality."""
        from enhanced_pipeline import _detect_processor_type
        
        # Test various processor types
        test_cases = [
            ("BertTokenizer", "tokenizer"),
            ("ViTImageProcessor", "image_processor"),
            ("Wav2Vec2FeatureExtractor", "feature_extractor"),
            ("CLIPProcessor", "processor"),
            ("UnknownProcessor", "processor")
        ]
        
        for class_name, expected_type in test_cases:
            mock_processor = Mock()
            mock_processor.__class__.__name__ = class_name
            
            result = _detect_processor_type(mock_processor)
            assert result == expected_type
    
    def test_task_detection_basic(self):
        """Test basic task detection functionality."""
        from auto_model_loader import AutoModelForONNX
        
        # Test architecture-based detection
        config = Mock(spec=['model_type', 'architectures'])  # Only these attributes exist
        config.model_type = "bert"
        config.architectures = ["BertForSequenceClassification"]
        
        task = AutoModelForONNX._detect_task(config, Path("/fake"))
        assert task == "text-classification"
        
        # Test model type-based detection
        config2 = Mock(spec=['model_type', 'architectures'])  # Only these attributes exist
        config2.model_type = "gpt2"
        config2.architectures = []
        
        task2 = AutoModelForONNX._detect_task(config2, Path("/fake"))
        assert task2 == "text-generation"
    
    def test_onnx_shape_parsing_mock(self):
        """Test ONNX shape parsing with mock data."""
        from onnx_tokenizer import parse_onnx_input_shapes
        
        # Create mock ONNX model structure
        mock_model = Mock()
        mock_graph = Mock()
        mock_model.graph = mock_graph
        
        # Mock input tensor
        mock_input = Mock()
        mock_input.name = "input_ids"
        mock_input.type.tensor_type.shape.dim = []
        
        # Add dimension mocks
        dim1 = Mock()
        dim1.HasField.return_value = True
        dim1.dim_value = 1
        
        dim2 = Mock() 
        dim2.HasField.return_value = True
        dim2.dim_value = 128
        
        mock_input.type.tensor_type.shape.dim = [dim1, dim2]
        mock_graph.input = [mock_input]
        
        # Test parsing
        with patch('onnx.load', return_value=mock_model):
            shapes = parse_onnx_input_shapes("/fake/model.onnx")
            
            assert "input_ids" in shapes
            assert shapes["input_ids"] == [1, 128]


@pytest.mark.smoke
class TestDataStructures:
    """Test data structures and configurations."""
    
    def test_model_type_mappings_completeness(self):
        """Test that model type mappings include common models."""
        from auto_model_loader import AutoModelForONNX
        
        # Common model types that should be supported
        common_models = ["bert", "gpt2", "t5", "vit", "roberta"]
        
        for model_type in common_models:
            assert model_type in AutoModelForONNX.MODEL_TYPE_TO_TASKS, \
                f"Common model type {model_type} not in mappings"
    
    def test_task_mappings_completeness(self):
        """Test that task mappings include common tasks."""
        from auto_model_loader import AutoModelForONNX
        
        # Common tasks that should be supported
        common_tasks = [
            "text-classification", "token-classification", "question-answering",
            "feature-extraction", "text-generation", "image-classification"
        ]
        
        for task in common_tasks:
            assert task in AutoModelForONNX.TASK_TO_ORT_MODEL, \
                f"Common task {task} not in mappings"
    
    def test_task_categories_in_enhanced_pipeline(self):
        """Test that enhanced pipeline has task categories defined."""
        from enhanced_pipeline import create_pipeline
        
        # Test that different task types are handled
        test_tasks = [
            "text-classification",  # TEXT_TASKS
            "image-classification",  # VISION_TASKS 
            "automatic-speech-recognition",  # AUDIO_TASKS
            "image-to-text"  # MULTIMODAL_TASKS
        ]
        
        for task in test_tasks:
            mock_processor = Mock()
            mock_processor.__class__.__name__ = "UnknownProcessor"
            
            with patch('enhanced_pipeline.hf_pipeline') as mock_hf_pipeline:
                mock_hf_pipeline.return_value = Mock()
                
                # Should not crash and should route to appropriate parameter
                result = create_pipeline(task=task, data_processor=mock_processor)
                assert result is not None


@pytest.mark.smoke
class TestErrorHandling:
    """Test basic error handling."""
    
    def test_invalid_onnx_tokenizer_shapes(self):
        """Test that ONNXTokenizer handles edge cases with fallback values."""
        from onnx_tokenizer import ONNXTokenizer
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        
        # Test that 0 values fall back to defaults
        tokenizer = ONNXTokenizer(
            tokenizer=mock_tokenizer,
            fixed_batch_size=0,  # Falls back to 1
            fixed_sequence_length=0  # Falls back to 128
        )
        assert tokenizer.fixed_batch_size == 1  # Default fallback
        assert tokenizer.fixed_sequence_length == 128  # Default fallback
        
        # Test that None values also fall back to defaults
        tokenizer2 = ONNXTokenizer(
            tokenizer=mock_tokenizer,
            fixed_batch_size=None,
            fixed_sequence_length=None
        )
        assert tokenizer2.fixed_batch_size == 1
        assert tokenizer2.fixed_sequence_length == 128
    
    def test_unsupported_task_error(self):
        """Test error handling for unsupported tasks."""
        from auto_model_loader import AutoModelForONNX
        
        with pytest.raises(ValueError, match="Task .* is not supported"):
            AutoModelForONNX._get_ort_model_class("completely-invalid-task")
    
    def test_model_path_validation(self):
        """Test model path validation."""
        from auto_model_loader import AutoModelForONNX
        
        with patch.object(Path, 'exists', return_value=False):
            with pytest.raises(ValueError, match="Model path does not exist"):
                AutoModelForONNX.from_pretrained("/nonexistent/path")


@pytest.mark.smoke
class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_list_supported_tasks(self):
        """Test listing supported tasks."""
        from auto_model_loader import AutoModelForONNX
        
        tasks = AutoModelForONNX.list_supported_tasks()
        
        assert isinstance(tasks, list)
        assert len(tasks) > 0
        assert "text-classification" in tasks
        assert "feature-extraction" in tasks
    
    def test_list_supported_model_types(self):
        """Test listing supported model types."""
        from auto_model_loader import AutoModelForONNX
        
        model_types = AutoModelForONNX.list_supported_model_types()
        
        assert isinstance(model_types, list)
        assert len(model_types) > 0
        assert "bert" in model_types
        assert "gpt2" in model_types
    
    def test_task_family_detection(self):
        """Test task family detection."""
        from onnx_config.task_detector import TaskDetector
        
        # Test various task families
        test_cases = [
            ("text-classification", "classification"),
            ("text-generation", "generation"),
            ("question-answering", "question-answering"),
            ("feature-extraction", "feature-extraction"),
            ("object-detection", "detection"),
            ("unknown-task", "unknown")
        ]
        
        for task, expected_family in test_cases:
            family = TaskDetector.get_task_family(task)
            assert family == expected_family


@pytest.mark.smoke
class TestConfigurationHandling:
    """Test configuration handling."""
    
    def test_config_dict_handling(self):
        """Test handling of configuration dictionaries."""
        from onnx_config.universal_config import UniversalOnnxConfig
        
        # Test with dict config
        config_dict = {"model_type": "bert", "hidden_size": 768}
        onnx_config = UniversalOnnxConfig(config=config_dict)
        
        assert onnx_config.model_type == "bert"
        assert onnx_config.config_dict["hidden_size"] == 768
    
    def test_config_object_handling(self):
        """Test handling of configuration objects."""
        from onnx_config.universal_config import UniversalOnnxConfig
        
        # Test with mock config object
        mock_config = Mock()
        mock_config.model_type = "gpt2"
        mock_config.to_dict.return_value = {"model_type": "gpt2", "vocab_size": 50257}
        
        onnx_config = UniversalOnnxConfig(config=mock_config)
        
        assert onnx_config.model_type == "gpt2"
        assert onnx_config.config_obj == mock_config


@pytest.mark.smoke
class TestModuleStructure:
    """Test module structure and organization."""
    
    def test_src_directory_structure(self):
        """Test that source directory has expected structure."""
        src_path = Path(__file__).parent.parent.parent / "tez-144_onnx_automodel_infer" / "src"
        
        # Key files should exist
        expected_files = [
            "onnx_tokenizer.py",
            "enhanced_pipeline.py", 
            "auto_model_loader.py",
            "inference_utils.py"
        ]
        
        for file_name in expected_files:
            file_path = src_path / file_name
            assert file_path.exists(), f"Expected file {file_name} not found"
    
    def test_onnx_config_submodule(self):
        """Test onnx_config submodule structure."""
        onnx_config_path = Path(__file__).parent.parent.parent / "tez-144_onnx_automodel_infer" / "src" / "onnx_config"
        
        expected_files = [
            "__init__.py",
            "universal_config.py",
            "task_detector.py"
        ]
        
        for file_name in expected_files:
            file_path = onnx_config_path / file_name
            assert file_path.exists(), f"Expected onnx_config file {file_name} not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "smoke"])