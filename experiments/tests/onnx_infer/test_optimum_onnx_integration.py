"""
Integration tests for Optimum + ONNX integration.

Tests the complete integration between our ONNX inference components
and Optimum ONNX runtime models, including:
- Model loading and inference workflows
- Pipeline integration with ONNX models
- End-to-end inference scenarios
- Error handling and edge cases
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
from onnx_tokenizer import ONNXTokenizer, create_auto_shape_tokenizer
from enhanced_pipeline import create_pipeline, pipeline
from inference_utils import (
    detect_model_task,
    load_preprocessor,
    create_inference_pipeline,
    benchmark_inference,
    compare_with_pytorch
)


@pytest.mark.integration
class TestOptimumONNXIntegration:
    """Test integration with Optimum ONNX runtime."""
    
    @patch('optimum.onnxruntime.ORTModelForFeatureExtraction')
    @patch('transformers.AutoTokenizer')
    def test_model_and_tokenizer_integration(self, mock_tokenizer_class, mock_ort_model):
        """Test integration between model loading and tokenizer creation."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_model_instance.path = Path("/fake/model.onnx")
        mock_model_instance.task = "feature-extraction"
        mock_ort_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.__call__ = Mock(return_value={
            "input_ids": torch.zeros((1, 128), dtype=torch.long),
            "attention_mask": torch.ones((1, 128), dtype=torch.long)
        })
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock ONNX shape parsing
        with patch('onnx_tokenizer.parse_onnx_input_shapes') as mock_parse:
            mock_parse.return_value = {"input_ids": [1, 128], "attention_mask": [1, 128]}
            
            # Test workflow
            with patch.object(Path, 'exists', return_value=True):
                with patch('transformers.AutoConfig.from_pretrained') as mock_config:
                    mock_config.return_value = Mock(
                        model_type="bert",
                        architectures=["BertForFeatureExtraction"]
                    )
                    # Load model
                    model = AutoModelForONNX.from_pretrained("/fake/path")
                
                # Create tokenizer with auto-shape detection
                tokenizer = create_auto_shape_tokenizer(mock_tokenizer, model)
                
                # Test inference
                inputs = tokenizer("Hello world!")
                assert inputs["input_ids"].shape == (1, 128)
                assert inputs["attention_mask"].shape == (1, 128)
    
    @patch('enhanced_pipeline.hf_pipeline')
    @patch('optimum.onnxruntime.ORTModelForSequenceClassification')
    def test_pipeline_integration_with_onnx_model(self, mock_ort_model, mock_hf_pipeline):
        """Test enhanced pipeline integration with ONNX model."""
        # Setup model mock
        mock_model_instance = Mock()
        mock_model_instance.task = "text-classification"
        mock_ort_model.from_pretrained.return_value = mock_model_instance
        
        # Setup pipeline mock
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{"label": "POSITIVE", "score": 0.9}]
        mock_hf_pipeline.return_value = mock_pipeline_instance
        
        # Setup tokenizer mock
        mock_tokenizer = Mock()
        mock_tokenizer.__class__.__name__ = "BertTokenizer"
        
        # Test pipeline creation with ONNX model
        with patch.object(Path, 'exists', return_value=True):
            model = AutoModelForONNX.from_pretrained("/fake/path", task="text-classification")
            
            pipe = create_pipeline(
                task="text-classification",
                model=model,
                data_processor=mock_tokenizer
            )
            
            # Test inference
            result = pipe("This is a positive sentence.")
            assert result[0]["label"] == "POSITIVE"
            assert result[0]["score"] == 0.9
    
    @patch('inference_utils.AutoTokenizer')
    @patch('optimum.onnxruntime.ORTModelForFeatureExtraction')
    def test_inference_utils_integration(self, mock_ort_model, mock_tokenizer_class):
        """Test inference utils integration with ONNX models."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_model_instance.task = "feature-extraction"
        mock_ort_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Test create_inference_pipeline
        with patch.object(Path, 'exists', return_value=True):
            with patch('inference_utils.AutoModelForONNX') as mock_auto_model:
                mock_auto_model.from_pretrained.return_value = mock_model_instance
                
                with patch('transformers.pipeline') as mock_pipeline:
                    mock_pipeline.return_value = Mock()
                    
                    pipe = create_inference_pipeline(
                        "/fake/path",
                        task="feature-extraction",
                        device=-1
                    )
                    
                    assert pipe is not None
                    mock_auto_model.from_pretrained.assert_called_once()
    
    @patch('optimum.onnxruntime.ORTModelForFeatureExtraction')
    def test_task_detection_integration(self, mock_ort_model, sample_bert_config):
        """Test task detection integration with model configuration."""
        # Test different architecture configurations
        configs = [
            {**sample_bert_config, "architectures": ["BertForSequenceClassification"]},
            {**sample_bert_config, "architectures": ["BertForTokenClassification"]},
            {**sample_bert_config, "architectures": ["BertForQuestionAnswering"]},
        ]
        
        expected_tasks = ["text-classification", "token-classification", "question-answering"]
        
        for config, expected_task in zip(configs, expected_tasks):
            with patch('auto_model_loader.AutoConfig') as mock_config:
                mock_config_obj = Mock()
                for key, value in config.items():
                    setattr(mock_config_obj, key, value)
                mock_config.from_pretrained.return_value = mock_config_obj
                
                mock_model_instance = Mock()
                mock_model_instance.task = expected_task
                mock_ort_model.from_pretrained.return_value = mock_model_instance
                
                with patch.object(Path, 'exists', return_value=True):
                    model = AutoModelForONNX.from_pretrained("/fake/path")
                    assert model.task == expected_task


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    @patch('optimum.onnxruntime.ORTModelForSequenceClassification')
    @patch('transformers.AutoTokenizer')
    @patch('transformers.pipeline')
    def test_text_classification_workflow(self, mock_pipeline, mock_tokenizer_class, mock_ort_model):
        """Test complete text classification workflow."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_model_instance.task = "text-classification"
        mock_model_instance.path = Path("/fake/model.onnx")
        mock_ort_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.__call__ = Mock(return_value={
            "input_ids": torch.zeros((1, 128), dtype=torch.long),
            "attention_mask": torch.ones((1, 128), dtype=torch.long)
        })
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [{"label": "POSITIVE", "score": 0.95}]
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Mock ONNX shape parsing
        with patch('onnx_tokenizer.parse_onnx_input_shapes') as mock_parse:
            mock_parse.return_value = {"input_ids": [1, 128], "attention_mask": [1, 128]}
            
            with patch.object(Path, 'exists', return_value=True):
                # Step 1: Load model
                model = AutoModelForONNX.from_pretrained("/fake/path", task="text-classification")
                
                # Step 2: Create ONNX tokenizer
                base_tokenizer = mock_tokenizer_class.from_pretrained("/fake/path")
                onnx_tokenizer = create_auto_shape_tokenizer(base_tokenizer, model)
                
                # Step 3: Create pipeline
                pipe = create_pipeline(
                    task="text-classification",
                    model=model,
                    data_processor=onnx_tokenizer
                )
                
                # Step 4: Run inference
                result = pipe("This movie is excellent!")
                
                assert result[0]["label"] == "POSITIVE"
                assert result[0]["score"] == 0.95
    
    @patch('optimum.onnxruntime.ORTModelForFeatureExtraction')
    @patch('transformers.AutoTokenizer')
    def test_feature_extraction_workflow(self, mock_tokenizer_class, mock_ort_model):
        """Test feature extraction workflow."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_model_instance.task = "feature-extraction"
        mock_model_instance.path = Path("/fake/model.onnx")
        mock_model_instance.return_value = Mock(last_hidden_state=torch.randn(1, 128, 768))
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
                # Load model and tokenizer
                model = AutoModelForONNX.from_pretrained("/fake/path")
                base_tokenizer = mock_tokenizer_class.from_pretrained("/fake/path")
                onnx_tokenizer = create_auto_shape_tokenizer(base_tokenizer, model)
                
                # Process text
                inputs = onnx_tokenizer("Extract features from this text.")
                
                # Run inference
                outputs = model(**inputs)
                
                assert hasattr(outputs, 'last_hidden_state')
                assert outputs.last_hidden_state.shape == (1, 128, 768)
    
    @patch('inference_utils.pipeline')
    @patch('inference_utils.load_preprocessor')
    @patch('inference_utils.AutoModelForONNX')
    def test_inference_utils_workflow(self, mock_auto_model, mock_load_preprocessor, mock_pipeline):
        """Test inference utils convenience functions."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_model_instance.task = "text-classification"
        mock_auto_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer = Mock()
        mock_load_preprocessor.return_value = mock_tokenizer
        
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Test create_inference_pipeline
        pipe = create_inference_pipeline(
            "/fake/path",
            task="text-classification",
            device=-1
        )
        
        assert pipe is not None
        mock_auto_model.from_pretrained.assert_called_once()
        mock_load_preprocessor.assert_called_once()
        mock_pipeline.assert_called_once()


@pytest.mark.integration
class TestMultiModalSupport:
    """Test multi-modal model support."""
    
    @patch('optimum.onnxruntime.ORTModelForImageClassification')
    @patch('transformers.AutoImageProcessor')
    def test_vision_model_integration(self, mock_processor_class, mock_ort_model):
        """Test vision model integration."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_model_instance.task = "image-classification"
        mock_ort_model.from_pretrained.return_value = mock_model_instance
        
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "ViTImageProcessor"
        mock_processor_class.from_pretrained.return_value = mock_processor
        
        with patch.object(Path, 'exists', return_value=True):
            # Load vision model
            model = AutoModelForONNX.from_pretrained("/fake/path", task="image-classification")
            
            # Create pipeline with image processor
            with patch('enhanced_pipeline.hf_pipeline') as mock_hf_pipeline:
                pipe = create_pipeline(
                    task="image-classification",
                    model=model,
                    data_processor=mock_processor
                )
                
                # Check that image_processor was used
                args, kwargs = mock_hf_pipeline.call_args
                assert kwargs["image_processor"] == mock_processor
    
    @patch('optimum.onnxruntime.ORTModel')
    @patch('transformers.AutoProcessor')
    def test_multimodal_processor_integration(self, mock_processor_class, mock_ort_model):
        """Test multimodal processor integration."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_model_instance.task = "image-to-text"
        mock_ort_model.from_pretrained.return_value = mock_model_instance
        
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "BlipProcessor"
        mock_processor.tokenizer = Mock()
        mock_processor.image_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        
        with patch.object(Path, 'exists', return_value=True):
            # Load multimodal model
            model = AutoModelForONNX.from_pretrained("/fake/path", task="image-to-text")
            
            # Create pipeline with multimodal processor
            with patch('enhanced_pipeline.hf_pipeline') as mock_hf_pipeline:
                pipe = create_pipeline(
                    task="image-to-text",
                    model=model,
                    data_processor=mock_processor
                )
                
                # Check that processor was used
                args, kwargs = mock_hf_pipeline.call_args
                assert kwargs["processor"] == mock_processor


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test error handling in integrated scenarios."""
    
    def test_model_path_not_found_integration(self):
        """Test handling of non-existent model paths."""
        with pytest.raises(ValueError, match="Model path does not exist"):
            AutoModelForONNX.from_pretrained("/definitely/does/not/exist")
    
    @patch('optimum.onnxruntime.ORTModelForFeatureExtraction')
    def test_tokenizer_shape_detection_failure_integration(self, mock_ort_model):
        """Test graceful handling of tokenizer shape detection failures."""
        # Setup model mock
        mock_model_instance = Mock()
        mock_model_instance.path = None  # Will cause shape detection to fail
        mock_model_instance.model_path = None
        del mock_model_instance.model  # Remove model attribute
        mock_ort_model.from_pretrained.return_value = mock_model_instance
        
        # Setup tokenizer mock
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.__call__ = Mock(return_value={
            "input_ids": torch.zeros((1, 128), dtype=torch.long),
            "attention_mask": torch.ones((1, 128), dtype=torch.long)
        })
        
        # Should fall back to defaults without crashing
        onnx_tokenizer = ONNXTokenizer(
            tokenizer=mock_tokenizer,
            onnx_model=mock_model_instance
        )
        
        assert onnx_tokenizer.fixed_batch_size == 1
        assert onnx_tokenizer.fixed_sequence_length == 128
    
    @patch('enhanced_pipeline.hf_pipeline')
    def test_unknown_processor_type_integration(self, mock_hf_pipeline):
        """Test handling of unknown processor types."""
        mock_hf_pipeline.return_value = Mock()
        
        # Create processor with unknown class name
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "VeryUnknownProcessor"
        
        # Should default to tokenizer for unknown tasks
        pipe = create_pipeline(
            task="unknown-task",
            model="some-model",
            data_processor=mock_processor
        )
        
        # Check that it was routed to tokenizer
        args, kwargs = mock_hf_pipeline.call_args
        assert kwargs["tokenizer"] == mock_processor


@pytest.mark.integration
class TestBenchmarkingIntegration:
    """Test benchmarking and performance testing integration."""
    
    @patch('optimum.onnxruntime.ORTModelForFeatureExtraction')
    def test_benchmark_inference_integration(self, mock_ort_model):
        """Test benchmarking integration with ONNX models."""
        # Setup model mock
        mock_model_instance = Mock()
        mock_model_instance.return_value = Mock(last_hidden_state=torch.randn(1, 128, 768))
        mock_ort_model.from_pretrained.return_value = mock_model_instance
        
        # Test inputs
        inputs = {
            "input_ids": torch.zeros((1, 128), dtype=torch.long),
            "attention_mask": torch.ones((1, 128), dtype=torch.long)
        }
        
        # Run benchmark
        with patch('time.perf_counter', side_effect=[0.0, 0.001, 0.0, 0.002, 0.0, 0.001]):
            metrics = benchmark_inference(
                mock_model_instance,
                inputs,
                num_runs=3,
                warmup_runs=1
            )
        
        assert "mean_latency" in metrics
        assert "std_latency" in metrics
        assert "min_latency" in metrics
        assert "max_latency" in metrics
        assert "throughput" in metrics
        
        assert metrics["mean_latency"] > 0
        assert metrics["throughput"] > 0
    
    @patch('optimum.onnxruntime.ORTModelForFeatureExtraction')
    @patch('transformers.AutoModel')
    @patch('transformers.AutoTokenizer')
    def test_pytorch_comparison_integration(self, mock_tokenizer, mock_pytorch_model, mock_ort_model):
        """Test PyTorch vs ONNX comparison integration."""
        # Setup ONNX model mock
        mock_onnx_instance = Mock()
        mock_onnx_instance.return_value = Mock(last_hidden_state=torch.randn(1, 128, 768))
        mock_ort_model.from_pretrained.return_value = mock_onnx_instance
        
        # Setup PyTorch model mock
        mock_pytorch_instance = Mock()
        mock_pytorch_instance.eval.return_value = None
        mock_pytorch_instance.return_value = Mock(last_hidden_state=torch.randn(1, 128, 768))
        mock_pytorch_model.from_pretrained.return_value = mock_pytorch_instance
        
        # Setup tokenizer mock
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            "input_ids": torch.zeros((1, 128), dtype=torch.long),
            "attention_mask": torch.ones((1, 128), dtype=torch.long)
        }
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock timing
        with patch('time.perf_counter', side_effect=[0.0, 0.002, 0.0, 0.001]):
            with patch('torch.no_grad'):
                comparison = compare_with_pytorch(
                    "/fake/onnx/path",
                    "pytorch-model-name",
                    ["Test input"],
                    task="feature-extraction"
                )
        
        assert "onnx" in comparison
        assert "pytorch" in comparison
        assert "speedup" in comparison
        assert "speedup_percentage" in comparison
        
        # ONNX should be faster (2ms vs 1ms in our mock)
        assert comparison["speedup"] == 2.0
        assert comparison["speedup_percentage"] == 100.0


@pytest.mark.smoke
class TestIntegrationSmoke:
    """Smoke tests for integration functionality."""
    
    def test_basic_integration_smoke(self):
        """Basic smoke test for integration components."""
        # Test that we can import and use the main components
        from onnx_tokenizer import ONNXTokenizer
        from enhanced_pipeline import create_pipeline
        from auto_model_loader import AutoModelForONNX
        
        # Verify classes exist and have expected methods
        assert hasattr(ONNXTokenizer, '__init__')
        assert hasattr(AutoModelForONNX, 'from_pretrained')
        assert callable(create_pipeline)
        
        # Verify model type mappings are loaded
        assert len(AutoModelForONNX.MODEL_TYPE_TO_TASKS) > 100  # We added 250+ models
        assert len(AutoModelForONNX.TASK_TO_ORT_MODEL) > 20  # We added 30+ tasks
    
    @patch('enhanced_pipeline.hf_pipeline')
    def test_pipeline_creation_smoke(self, mock_hf_pipeline):
        """Smoke test for pipeline creation."""
        mock_hf_pipeline.return_value = Mock()
        mock_processor = Mock()
        mock_processor.__class__.__name__ = "BertTokenizer"
        
        # Test that pipeline creation doesn't crash
        pipe = create_pipeline(
            task="text-classification",
            model="bert-base-uncased",
            data_processor=mock_processor
        )
        assert pipe is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])