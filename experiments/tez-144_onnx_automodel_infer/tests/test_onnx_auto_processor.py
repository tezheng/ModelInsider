"""
Comprehensive Test Suite for ONNXAutoProcessor

This module provides complete test coverage for the ONNX Auto Processor system,
including smoke tests, sanity tests, unit tests, integration tests, and performance tests.

Test Categories:
- 🚬 Smoke Tests: Basic functionality verification (5 min)
- ✅ Sanity Tests: Core feature validation (15 min)  
- 🔧 Unit Tests: Component isolation testing (30 min)
- 🔄 Integration Tests: End-to-end workflows (45 min)
- ⚡ Performance Tests: Speed/memory benchmarks (20 min)

Usage:
    # Quick validation
    pytest -m smoke tests/test_onnx_auto_processor.py -v
    
    # Core features
    pytest -m "smoke or sanity" tests/test_onnx_auto_processor.py -v
    
    # All except slow tests
    pytest -m "not slow" tests/test_onnx_auto_processor.py -v

Author: Generated for TEZ-144 ONNX AutoProcessor Test Implementation
"""

import os

# Import modules under test
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import onnx
import pytest

# Import test utilities
from test_utils import (
    MockONNXModel,
    PerformanceBenchmark,
    assert_tensor_dict_valid,
    create_audio_onnx_model,
    create_image_onnx_model,
    create_mock_base_processor,
    create_multimodal_onnx_model,
    create_test_model_directory,
    create_text_onnx_model,
    create_video_onnx_model,
    save_onnx_model_to_temp,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from onnx_auto_processor import ONNXAutoProcessor
from onnx_processor_types import (
    ModalityConfig,
    ModalityType,
    ONNXModelLoadError,
    ONNXProcessorError,
    ONNXProcessorNotFoundError,
    ONNXUnsupportedModalityError,
    ProcessorMetadata,
    TensorSpec,
    TensorType,
    create_tensor_spec_from_dict,
    validate_tensor_spec,
)
from processors import (
    BaseONNXProcessor,
    ONNXAudioProcessor,
    ONNXImageProcessor,
    ONNXProcessor,
    ONNXTokenizer,
    ONNXVideoProcessor,
)


class TestONNXAutoProcessor:
    """Comprehensive test suite for ONNXAutoProcessor with pytest markers."""
    
    # ========================================================================
    # 🚬 SMOKE TESTS - Basic functionality check (5 minutes)
    # ========================================================================
    
    @pytest.mark.smoke
    def test_import_onnx_auto_processor(self):
        """Test that ONNXAutoProcessor can be imported."""
        assert ONNXAutoProcessor is not None
        assert hasattr(ONNXAutoProcessor, 'from_model')
    
    @pytest.mark.smoke
    def test_import_processor_types(self):
        """Test that all processor types can be imported."""
        assert BaseONNXProcessor is not None
        assert ONNXTokenizer is not None
        assert ONNXImageProcessor is not None
        assert ONNXAudioProcessor is not None
        assert ONNXVideoProcessor is not None
        assert ONNXProcessor is not None
    
    @pytest.mark.smoke
    def test_import_type_definitions(self):
        """Test that type definitions can be imported."""
        assert ModalityType is not None
        assert TensorType is not None
        assert TensorSpec is not None
        assert ProcessorMetadata is not None
    
    @pytest.mark.smoke
    def test_create_processor_from_bert_onnx(self, bert_onnx_model, mock_tokenizer):
        """Test basic BERT processor creation."""
        # Save model to temp file
        model_path = save_onnx_model_to_temp(bert_onnx_model, "bert_smoke.onnx")
        
        # Mock the AutoProcessor loading
        with patch('onnx_auto_processor.AutoProcessor') as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            
            try:
                processor = ONNXAutoProcessor.from_model(model_path, base_processor=mock_tokenizer)
                assert processor is not None
                assert hasattr(processor, '_onnx_processor')
                assert processor.modality_type in [ModalityType.TEXT, ModalityType.UNKNOWN]
            except Exception as e:
                pytest.fail(f"Basic processor creation failed: {e}")
    
    @pytest.mark.smoke
    def test_process_simple_text_input(self, bert_onnx_model, mock_tokenizer):
        """Test processing a simple text input."""
        model_path = save_onnx_model_to_temp(bert_onnx_model, "bert_process.onnx")
        
        with patch('onnx_auto_processor.AutoProcessor') as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            
            try:
                processor = ONNXAutoProcessor.from_model(model_path, base_processor=mock_tokenizer)
                output = processor("Hello world")
                
                assert isinstance(output, dict)
                assert len(output) > 0
                assert_tensor_dict_valid(output)
            except Exception as e:
                pytest.fail(f"Simple text processing failed: {e}")
    
    @pytest.mark.smoke
    def test_metadata_extraction_works(self, bert_onnx_model):
        """Test that metadata can be extracted from ONNX."""
        try:
            info = ONNXAutoProcessor._extract_onnx_info(bert_onnx_model)
            assert isinstance(info, dict)
            assert 'modalities' in info
            assert 'is_multimodal' in info
            assert 'input_names' in info
            assert 'output_names' in info
        except Exception as e:
            pytest.fail(f"Metadata extraction failed: {e}")
    
    @pytest.mark.smoke
    @pytest.mark.multimodal
    def test_multimodal_detection_clip(self, clip_onnx_model, mock_multimodal_processor):
        """Test that CLIP is detected as multimodal."""
        model_path = save_onnx_model_to_temp(clip_onnx_model, "clip_smoke.onnx")
        
        with patch('onnx_auto_processor.AutoProcessor') as mock_auto:
            mock_auto.from_pretrained.return_value = mock_multimodal_processor
            
            try:
                processor = ONNXAutoProcessor.from_model(
                    model_path, 
                    base_processor=mock_multimodal_processor
                )
                assert processor is not None
                assert processor.metadata.is_multimodal == True
                assert ModalityType.MULTIMODAL in processor.supported_modalities
            except Exception as e:
                pytest.fail(f"Multimodal detection failed: {e}")
    
    # ========================================================================
    # ✅ SANITY TESTS - Core feature validation (15 minutes)
    # ========================================================================
    
    @pytest.mark.sanity
    def test_all_five_processor_types(self):
        """Test that all 5 processor types can be created."""
        test_models = {
            "text": (create_text_onnx_model("bert_sanity"), create_mock_base_processor("tokenizer")),
            "image": (create_image_onnx_model("vit_sanity"), create_mock_base_processor("image_processor")),
            "audio": (create_audio_onnx_model("wav2vec2_sanity"), create_mock_base_processor("feature_extractor")),
            "video": (create_video_onnx_model("videomae_sanity"), create_mock_base_processor("video_processor")),
            "multimodal": (create_multimodal_onnx_model("clip_sanity"), create_mock_base_processor("multimodal"))
        }
        
        for modality, (model, base_processor) in test_models.items():
            model_path = save_onnx_model_to_temp(model, f"{modality}_sanity.onnx")
            
            with patch('onnx_auto_processor.AutoProcessor') as mock_auto:
                mock_auto.from_pretrained.return_value = base_processor
                
                processor = ONNXAutoProcessor.from_model(model_path, base_processor=base_processor)
                assert processor is not None, f"Failed to create {modality} processor"
                print(f"✓ {modality} processor created successfully")
    
    @pytest.mark.sanity
    def test_processor_wrapping_correct(self, mock_tokenizer):
        """Test that processors are wrapped with correct ONNX wrapper."""
        # Create BERT model
        bert_model = create_text_onnx_model("bert_wrap_test")
        model_path = save_onnx_model_to_temp(bert_model, "bert_wrap.onnx")
        
        with patch('onnx_auto_processor.AutoProcessor') as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            
            processor = ONNXAutoProcessor.from_model(model_path, base_processor=mock_tokenizer)
            
            # Text processor should be wrapped with ONNXTokenizer
            assert hasattr(processor._onnx_processor, 'batch_size')
            assert hasattr(processor._onnx_processor, 'sequence_length')
            assert isinstance(processor._onnx_processor, BaseONNXProcessor)
    
    @pytest.mark.sanity
    def test_fixed_shape_enforcement(self, bert_onnx_model, mock_tokenizer):
        """Test that fixed shapes are enforced."""
        model_path = save_onnx_model_to_temp(bert_onnx_model, "bert_fixed_shape.onnx")
        
        with patch('onnx_auto_processor.AutoProcessor') as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            
            processor = ONNXAutoProcessor.from_model(model_path, base_processor=mock_tokenizer)
            
            # Process varying length inputs
            output1 = processor("Hi")
            output2 = processor("This is a much longer sentence that should be padded or truncated")
            
            # Both should have same shape
            for key in output1:
                if key in output2:
                    assert output1[key].shape == output2[key].shape, f"Shape mismatch for {key}"
    
    @pytest.mark.sanity
    def test_metadata_priority_order(self):
        """Test metadata loading priority: ONNX > JSON > auto-detect."""
        # Create model with rich metadata
        model = (MockONNXModel("priority_test")
                .add_input("input_ids", [1, 128], TensorType.INT64)
                .add_output("logits", [1, 2], TensorType.FLOAT32)
                .add_metadata("model_type", "bert")
                .add_metadata("processor.batch_size", "1")
                .add_metadata("processor.sequence_length", "128")
                .build())
        
        model_path = save_onnx_model_to_temp(model, "priority_test.onnx")
        
        processor_metadata = ONNXAutoProcessor._extract_processor_metadata(model, model_path)
        
        # Verify metadata was extracted from ONNX (not auto-detected)
        assert processor_metadata.model_type == "bert"
        assert processor_metadata.metadata_source == "onnx"
    
    @pytest.mark.sanity
    def test_backward_compatibility_interface(self, bert_onnx_model, mock_tokenizer):
        """Test that the interface remains backward compatible."""
        model_path = save_onnx_model_to_temp(bert_onnx_model, "compat_test.onnx")
        
        with patch('onnx_auto_processor.AutoProcessor') as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            
            # New style - auto processor
            auto_processor = ONNXAutoProcessor.from_model(model_path, base_processor=mock_tokenizer)
            
            # Both should be callable
            assert callable(auto_processor)
            
            # Should provide common interface methods
            assert hasattr(auto_processor, 'preprocess')
            assert hasattr(auto_processor, 'tensor_names')
            assert hasattr(auto_processor, 'supported_modalities')
    
    # ========================================================================
    # 🔧 UNIT TESTS - Component isolation testing (30 minutes)
    # ========================================================================
    
    @pytest.mark.unit
    def test_extract_onnx_info_text_modality(self):
        """Unit test: _extract_onnx_info for text inputs."""
        mock_model = (MockONNXModel("text_unit")
                     .add_input("input_ids", [1, 128], TensorType.INT64)
                     .add_input("attention_mask", [1, 128], TensorType.INT64)
                     .build())
        
        info = ONNXAutoProcessor._extract_onnx_info(mock_model)
        
        assert 'text' in info['modalities']
        assert info['modalities']['text']['batch_size'] == 1
        assert info['modalities']['text']['sequence_length'] == 128
        assert len(info['modalities']['text']['tensors']) == 2
        assert info['is_multimodal'] == False
    
    @pytest.mark.unit
    @pytest.mark.multimodal
    def test_extract_onnx_info_multimodal(self):
        """Unit test: _extract_onnx_info for CLIP-like model."""
        mock_model = (MockONNXModel("multimodal_unit")
                     .add_input("input_ids", [1, 77], TensorType.INT64)
                     .add_input("pixel_values", [1, 3, 224, 224], TensorType.FLOAT32)
                     .build())
        
        info = ONNXAutoProcessor._extract_onnx_info(mock_model)
        
        assert info['is_multimodal'] == True
        assert 'text' in info['modalities']
        assert 'image' in info['modalities']
        assert info['modalities']['text']['sequence_length'] == 77
        assert info['modalities']['image']['height'] == 224
    
    @pytest.mark.unit
    def test_modality_detection_by_name(self):
        """Unit test: Modality detection by tensor name."""
        test_cases = [
            ("input_ids", ModalityType.TEXT),
            ("attention_mask", ModalityType.TEXT),
            ("token_type_ids", ModalityType.TEXT),
            ("pixel_values", ModalityType.IMAGE),
            ("image", ModalityType.IMAGE),
            ("input_values", ModalityType.AUDIO),
            ("input_features", ModalityType.AUDIO),
            ("mel_spectrogram", ModalityType.AUDIO),
            ("video_frames", ModalityType.VIDEO),
            ("temporal_features", ModalityType.VIDEO),
            ("unknown_tensor", ModalityType.UNKNOWN)
        ]
        
        for tensor_name, expected_modality in test_cases:
            detected = ModalityType.from_tensor_name(tensor_name)
            assert detected == expected_modality, f"Failed for {tensor_name}: expected {expected_modality}, got {detected}"
    
    @pytest.mark.unit
    def test_modality_detection_by_shape(self):
        """Unit test: Modality detection by tensor shape."""
        test_cases = [
            ([1, 128], ModalityType.TEXT),           # 2D likely text
            ([2, 512], ModalityType.TEXT),           # 2D likely text
            ([1, 3, 224, 224], ModalityType.IMAGE),  # 4D NCHW
            ([4, 1, 256, 256], ModalityType.IMAGE),  # 4D grayscale
            ([1, 16000], ModalityType.AUDIO),         # 2D waveform
            ([1, 80, 3000], ModalityType.AUDIO),      # 3D spectrogram
            ([1, 3, 16, 224, 224], ModalityType.VIDEO), # 5D NCTHW
            ([], ModalityType.UNKNOWN),               # Empty shape
            ([1], ModalityType.UNKNOWN),              # 1D
        ]
        
        for shape, expected_modality in test_cases:
            detected = ModalityType.from_tensor_shape(shape)
            assert detected == expected_modality, f"Failed for {shape}: expected {expected_modality}, got {detected}"
    
    @pytest.mark.unit
    def test_tensor_spec_creation_and_validation(self):
        """Unit test: TensorSpec creation and validation."""
        # Valid tensor spec
        spec = TensorSpec(
            name="test_tensor",
            shape=[1, 128],
            dtype=TensorType.INT64,
            modality=ModalityType.TEXT,
            is_input=True,
            description="Test tensor"
        )
        
        assert spec.name == "test_tensor"
        assert spec.shape == [1, 128]
        assert spec.dtype == TensorType.INT64
        assert spec.rank == 2
        assert spec.size == 128
        assert not spec.is_dynamic
        assert spec.memory_size() == 128 * 8  # INT64 = 8 bytes
    
    @pytest.mark.unit
    def test_tensor_spec_validation_errors(self):
        """Unit test: TensorSpec validation error cases."""
        # Test validation during construction - should raise ValueError immediately
        
        # Test empty name
        with pytest.raises(ValueError, match="Tensor name cannot be empty"):
            TensorSpec("", [1, 128], TensorType.INT64)
        
        # Test empty shape
        with pytest.raises(ValueError, match="Tensor shape cannot be empty"):
            TensorSpec("test", [], TensorType.INT64)
        
        # Test invalid dimension
        with pytest.raises(ValueError, match="Invalid dimension size"):
            TensorSpec("test", [1, -2], TensorType.INT64)
    
    @pytest.mark.unit
    def test_modality_config_creation(self):
        """Unit test: ModalityConfig creation and validation."""
        tensors = [
            TensorSpec("input_ids", [1, 128], TensorType.INT64, ModalityType.TEXT),
            TensorSpec("attention_mask", [1, 128], TensorType.INT64, ModalityType.TEXT)
        ]
        
        config = ModalityConfig(
            modality_type=ModalityType.TEXT,
            tensors=tensors,
            batch_size=1,
            config={"sequence_length": 128}
        )
        
        assert config.modality_type == ModalityType.TEXT
        assert len(config.input_tensors) == 2
        assert len(config.output_tensors) == 0
        assert config.tensor_names == ["input_ids", "attention_mask"]
        assert not config.has_dynamic_shapes
    
    @pytest.mark.unit
    def test_processor_metadata_creation(self):
        """Unit test: ProcessorMetadata creation and properties."""
        text_config = ModalityConfig(
            modality_type=ModalityType.TEXT,
            tensors=[TensorSpec("input_ids", [1, 128], TensorType.INT64, ModalityType.TEXT)],
            batch_size=1
        )
        
        metadata = ProcessorMetadata(
            model_name="test_model",
            model_type="bert",
            task="feature-extraction",
            modalities={"text": text_config},
            is_multimodal=False
        )
        
        assert metadata.model_name == "test_model"
        assert metadata.model_type == "bert"
        assert not metadata.is_multimodal
        assert len(metadata.all_input_tensors) == 1
        assert ModalityType.TEXT in metadata.modality_types
        assert metadata.has_modality(ModalityType.TEXT)
        assert not metadata.has_modality(ModalityType.IMAGE)
    
    @pytest.mark.unit
    def test_handle_missing_metadata(self):
        """Unit test: Handle missing metadata gracefully."""
        # Model with no inputs (edge case)
        mock_model = MockONNXModel("empty_model").build()
        
        info = ONNXAutoProcessor._extract_onnx_info(mock_model)
        
        assert 'modalities' in info
        assert info['input_count'] == 0
        assert info['output_count'] == 0
        assert info['is_multimodal'] == False
    
    @pytest.mark.unit
    def test_tensor_type_conversions(self):
        """Unit test: TensorType conversion methods."""
        # Test numpy dtype conversion
        assert TensorType.FLOAT32.to_numpy_dtype() == np.float32
        assert TensorType.INT64.to_numpy_dtype() == np.int64
        assert TensorType.BOOL.to_numpy_dtype() == np.bool_
        
        # Test Python type conversion
        assert TensorType.FLOAT32.to_python_type() is float
        assert TensorType.INT64.to_python_type() is int
        assert TensorType.BOOL.to_python_type() is bool
        assert TensorType.STRING.to_python_type() is str
        
        # Test type checking methods
        assert TensorType.INT64.is_integer()
        assert TensorType.FLOAT32.is_floating_point()
        assert not TensorType.BOOL.is_integer()
        assert not TensorType.INT32.is_floating_point()
        
        # Test size calculation
        assert TensorType.FLOAT32.size_in_bytes() == 4
        assert TensorType.INT64.size_in_bytes() == 8
        assert TensorType.BOOL.size_in_bytes() == 1
    
    @pytest.mark.unit
    def test_shape_compatibility_checking(self):
        """Unit test: Shape compatibility validation."""
        spec = TensorSpec("test", [1, -1, 128], TensorType.FLOAT32)  # Dynamic middle dimension
        
        # Compatible shapes
        assert spec.is_compatible_shape([1, 256, 128])
        assert spec.is_compatible_shape([1, 512, 128])
        
        # Incompatible shapes
        assert not spec.is_compatible_shape([2, 256, 128])  # Wrong batch size
        assert not spec.is_compatible_shape([1, 256, 64])   # Wrong last dimension
        assert not spec.is_compatible_shape([1, 256])       # Wrong rank
    
    @pytest.mark.unit
    def test_utility_functions(self):
        """Unit test: Utility functions."""
        # Test validate_tensor_spec
        valid_spec = TensorSpec("test", [1, 128], TensorType.FLOAT32, ModalityType.TEXT)
        errors = validate_tensor_spec(valid_spec)
        assert len(errors) == 0
        
        # Test invalid spec creation - should fail during construction
        with pytest.raises(ValueError):
            TensorSpec("", [], TensorType.UNDEFINED)
        
        # Test create_tensor_spec_from_dict
        tensor_dict = {
            "name": "test_tensor",
            "shape": [1, 128],
            "dtype": int(TensorType.INT64),
            "modality": ModalityType.TEXT.value,
            "is_input": True,
            "description": "Test tensor"
        }
        
        spec = create_tensor_spec_from_dict(tensor_dict)
        assert spec.name == "test_tensor"
        assert spec.shape == [1, 128]
        assert spec.dtype == TensorType.INT64
    
    # ========================================================================
    # 🔄 INTEGRATION TESTS - End-to-end workflows (45 minutes)
    # ========================================================================
    
    @pytest.mark.integration
    def test_end_to_end_bert_processor(self, mock_tokenizer):
        """Integration test: Complete BERT processor workflow."""
        # Create realistic BERT model
        bert_model = create_text_onnx_model("bert_e2e", batch_size=1, sequence_length=512)
        model_dir = create_test_model_directory("bert", include_metadata=True, include_configs=True)
        
        # Save model to directory
        bert_path = model_dir / "model.onnx"
        onnx.save(bert_model, str(bert_path))
        
        with patch('onnx_auto_processor.AutoProcessor') as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            
            # Test from_model
            processor = ONNXAutoProcessor.from_model(bert_path, base_processor=mock_tokenizer)
            
            # Verify processor configuration
            assert processor.modality_type == ModalityType.TEXT
            assert "input_ids" in processor.tensor_names
            assert "attention_mask" in processor.tensor_names
            
            # Test processing
            test_inputs = [
                "Hello world",
                "This is a longer sentence for testing",
                ["Multiple", "sentences", "in", "a", "list"]
            ]
            
            for input_text in test_inputs:
                output = processor(input_text)
                assert_tensor_dict_valid(output)
            
            # Test empty string handling - should raise an error
            with pytest.raises(ONNXProcessorError, match="Text input cannot be empty"):
                processor("")
            
            # Verify fixed shapes for the last valid output
            expected_shapes = {
                "input_ids": [1, 512],
                "attention_mask": [1, 512]
            }
            # Only check shapes for tensors that exist
            actual_shapes = {k: list(v.shape) for k, v in output.items() if k in expected_shapes}
            for name, expected_shape in expected_shapes.items():
                if name in actual_shapes:
                    assert actual_shapes[name] == expected_shape
    
    @pytest.mark.integration
    @pytest.mark.multimodal
    def test_end_to_end_clip_processor(self, mock_multimodal_processor):
        """Integration test: Complete CLIP multimodal processor workflow."""
        # Create CLIP model
        clip_model = create_multimodal_onnx_model("clip_e2e")
        model_dir = create_test_model_directory("clip", include_metadata=True, include_configs=True)
        
        clip_path = model_dir / "model.onnx"
        onnx.save(clip_model, str(clip_path))
        
        with patch('onnx_auto_processor.AutoProcessor') as mock_auto:
            mock_auto.from_pretrained.return_value = mock_multimodal_processor
            
            processor = ONNXAutoProcessor.from_model(clip_path, base_processor=mock_multimodal_processor)
            
            # Verify multimodal configuration
            assert processor.metadata.is_multimodal == True
            assert ModalityType.TEXT in processor.supported_modalities
            assert ModalityType.IMAGE in processor.supported_modalities
            
            # Test text-only processing
            text_output = processor(text="A photo of a cat")
            assert_tensor_dict_valid(text_output)
            
            # Test image-only processing
            fake_image = np.random.rand(224, 224, 3).astype(np.float32)
            image_output = processor(images=fake_image)
            assert_tensor_dict_valid(image_output)
            
            # Test multimodal processing
            multimodal_output = processor(text="A photo of a cat", images=fake_image)
            assert_tensor_dict_valid(multimodal_output)
            assert "input_ids" in multimodal_output
            assert "pixel_values" in multimodal_output
    
    @pytest.mark.integration
    def test_processor_from_model_directory_workflow(self):
        """Integration test: from_model workflow with directory."""
        # Create a complete model directory
        model_dir = create_test_model_directory("bert", include_metadata=True, include_configs=True)
        
        with patch('onnx_auto_processor.AutoProcessor') as mock_auto:
            mock_auto.from_pretrained.return_value = create_mock_base_processor("tokenizer")
            
            # Test from_model with the ONNX file directly
            onnx_file = model_dir / "model.onnx"
            processor = ONNXAutoProcessor.from_model(onnx_file, hf_model_path=model_dir)
            
            assert processor is not None
            assert processor.metadata.model_name is not None
            assert len(processor.tensor_names) > 0
    
    @pytest.mark.integration
    def test_error_handling_workflow(self):
        """Integration test: Error handling scenarios."""
        # Test missing ONNX file
        with pytest.raises(ONNXModelLoadError):
            ONNXAutoProcessor.from_model("nonexistent.onnx")
        
        # Test corrupted ONNX file
        temp_file = Path(tempfile.mktemp(suffix=".onnx"))
        temp_file.write_bytes(b"INVALID_ONNX_DATA")
        
        with pytest.raises(ONNXModelLoadError):
            ONNXAutoProcessor.from_model(temp_file)
        
        temp_file.unlink()
        
        # Test unsupported processor type
        mock_unsupported = Mock()
        mock_unsupported.__class__.__name__ = "UnsupportedProcessor"
        
        bert_model = create_text_onnx_model("error_test")
        model_path = save_onnx_model_to_temp(bert_model)
        
        with pytest.raises((ONNXUnsupportedModalityError, ONNXProcessorNotFoundError)):
            ONNXAutoProcessor.from_model(model_path, base_processor=mock_unsupported)
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_processor_pipeline_integration(self, mock_tokenizer):
        """Integration test: Pipeline integration."""
        # This would test integration with enhanced pipeline
        # For now, test basic compatibility
        bert_model = create_text_onnx_model("pipeline_test")
        model_path = save_onnx_model_to_temp(bert_model)
        
        with patch('onnx_auto_processor.AutoProcessor') as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            
            processor = ONNXAutoProcessor.from_model(model_path, base_processor=mock_tokenizer)
            
            # Test that processor can be used like a standard processor
            assert callable(processor)
            assert hasattr(processor, 'preprocess')
            
            # Test batch processing
            batch_texts = ["Text 1", "Text 2", "Text 3"]
            outputs = []
            for text in batch_texts:
                output = processor(text)
                outputs.append(output)
            
            # All outputs should have consistent shapes
            if len(outputs) > 1:
                first_keys = set(outputs[0].keys())
                for output in outputs[1:]:
                    assert set(output.keys()) == first_keys
                    for key in first_keys:
                        assert outputs[0][key].shape == output[key].shape
    
    # ========================================================================
    # ⚡ PERFORMANCE TESTS - Speed/memory benchmarks (20 minutes)
    # ========================================================================
    
    @pytest.mark.performance
    def test_processor_creation_speed(self, performance_text_data):
        """Performance test: Processor instantiation time."""
        bert_model = create_text_onnx_model("perf_creation")
        model_path = save_onnx_model_to_temp(bert_model)
        mock_tokenizer = create_mock_base_processor("tokenizer")
        
        with patch('onnx_auto_processor.AutoProcessor') as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            
            with PerformanceBenchmark("processor_creation") as benchmark:
                for _ in range(10):
                    processor = ONNXAutoProcessor.from_model(model_path, base_processor=mock_tokenizer)
                    # Ensure processor is actually created
                    assert processor is not None
            
            stats = benchmark.get_stats()
            assert stats["avg_time"] < 0.1, f"Processor creation too slow: {stats['avg_time']:.3f}s (target: <0.1s)"
            print(f"Processor creation: {stats['avg_time']:.3f}s avg")
    
    @pytest.mark.performance
    def test_text_preprocessing_speed(self, performance_text_data):
        """Performance test: Text preprocessing speed."""
        bert_model = create_text_onnx_model("perf_text", batch_size=1, sequence_length=128)
        model_path = save_onnx_model_to_temp(bert_model)
        mock_tokenizer = create_mock_base_processor("tokenizer")
        
        with patch('onnx_auto_processor.AutoProcessor') as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            
            processor = ONNXAutoProcessor.from_model(model_path, base_processor=mock_tokenizer)
            
            # Warmup
            for text in performance_text_data[:5]:
                processor(text)
            
            # Benchmark
            with PerformanceBenchmark("text_preprocessing") as benchmark:
                for text in performance_text_data:
                    output = processor(text)
                    assert_tensor_dict_valid(output)
            
            stats = benchmark.get_stats()
            
            # Target: Process 50 texts in under 0.5 seconds (0.01s per text)
            assert stats["avg_time"] < 0.01, f"Text preprocessing too slow: {stats['avg_time']:.4f}s per text"
            print(f"Text preprocessing: {stats['avg_time']:.4f}s per text")
    
    @pytest.mark.performance
    def test_image_preprocessing_speed(self, performance_image_data):
        """Performance test: Image preprocessing speed."""
        vit_model = create_image_onnx_model("perf_image")
        model_path = save_onnx_model_to_temp(vit_model)
        mock_processor = create_mock_base_processor("image_processor")
        
        with patch('onnx_auto_processor.AutoProcessor') as mock_auto:
            mock_auto.from_pretrained.return_value = mock_processor
            
            processor = ONNXAutoProcessor.from_model(model_path, base_processor=mock_processor)
            
            # Warmup
            for image in performance_image_data[:3]:
                processor(image)
            
            # Benchmark
            with PerformanceBenchmark("image_preprocessing") as benchmark:
                for image in performance_image_data:
                    output = processor(image)
                    assert_tensor_dict_valid(output)
            
            stats = benchmark.get_stats()
            
            # Target: Process 20 images in under 1 second (0.05s per image)
            assert stats["avg_time"] < 0.05, f"Image preprocessing too slow: {stats['avg_time']:.4f}s per image"
            print(f"Image preprocessing: {stats['avg_time']:.4f}s per image")
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_memory_usage_stability(self):
        """Performance test: Memory usage stability."""
        import psutil
        
        bert_model = create_text_onnx_model("memory_test")
        model_path = save_onnx_model_to_temp(bert_model)
        mock_tokenizer = create_mock_base_processor("tokenizer")
        
        with patch('onnx_auto_processor.AutoProcessor') as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            
            processor = ONNXAutoProcessor.from_model(model_path, base_processor=mock_tokenizer)
            
            # Measure baseline memory
            process = psutil.Process()
            baseline_memory = process.memory_info().rss
            
            # Process many inputs
            test_texts = [f"Test sentence number {i}" for i in range(1000)]
            
            for i, text in enumerate(test_texts):
                processor(text)
                
                # Check memory every 100 iterations
                if i % 100 == 0:
                    current_memory = process.memory_info().rss
                    memory_growth = current_memory - baseline_memory
                    
                    # Memory growth should be minimal (<100MB)
                    assert memory_growth < 100 * 1024 * 1024, f"Memory leak detected: {memory_growth / 1024 / 1024:.1f}MB growth"
            
            print("Memory stable after 1000 iterations")
    
    @pytest.mark.performance
    def test_batch_processing_efficiency(self):
        """Performance test: Batch processing efficiency."""
        bert_model = create_text_onnx_model("batch_perf", batch_size=4)
        model_path = save_onnx_model_to_temp(bert_model)
        mock_tokenizer = create_mock_base_processor("tokenizer")
        
        with patch('onnx_auto_processor.AutoProcessor') as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            
            processor = ONNXAutoProcessor.from_model(model_path, base_processor=mock_tokenizer)
            
            test_texts = ["Sample text for batch test"] * 100
            
            # Time single processing
            single_start = time.perf_counter()
            for text in test_texts:
                processor(text)
            single_time = time.perf_counter() - single_start
            
            # Time batch processing (simulate batching)
            batch_start = time.perf_counter()
            for i in range(0, len(test_texts), 4):
                batch = test_texts[i:i+4]
                for text in batch:  # Simulate batch processing
                    processor(text)
            batch_time = time.perf_counter() - batch_start
            
            # Batch processing should be at least as fast
            print(f"Single: {single_time:.3f}s, Batch: {batch_time:.3f}s")
            assert batch_time <= single_time * 1.1  # Allow 10% overhead
    
    # ========================================================================
    # Additional Test Categories
    # ========================================================================
    
    @pytest.mark.unit
    def test_edge_cases_handling(self):
        """Test edge cases and boundary conditions."""
        # Test empty inputs
        with pytest.raises((ValueError, ONNXProcessorError)):
            spec = TensorSpec("", [1, 128], TensorType.FLOAT32)
            spec.__post_init__()
        
        # Test very large dimensions
        large_spec = TensorSpec("large", [1, 1000000], TensorType.FLOAT32)
        assert large_spec.size == 1000000
        assert large_spec.memory_size() == 4000000  # 4MB
        
        # Test minimum dimensions
        min_spec = TensorSpec("min", [1, 1], TensorType.FLOAT32)
        assert min_spec.size == 1
        assert min_spec.memory_size() == 4
    
    @pytest.mark.unit
    def test_error_message_quality(self):
        """Test that error messages are clear and helpful."""
        # Test missing file error
        try:
            ONNXAutoProcessor.from_model("nonexistent_file.onnx")
        except ONNXModelLoadError as e:
            assert "nonexistent_file.onnx" in str(e)
            assert "Failed to load" in str(e)
        
        # Test configuration error
        try:
            config = ModalityConfig(
                modality_type=ModalityType.TEXT,
                tensors=[],  # Empty tensors list
                batch_size=0  # Invalid batch size
            )
            config.__post_init__()
        except ValueError as e:
            assert "Batch size must be positive" in str(e) or "At least one tensor" in str(e)
    
    @pytest.mark.integration
    def test_concurrent_processing(self):
        """Test concurrent processing safety."""
        import queue
        import threading
        
        bert_model = create_text_onnx_model("concurrent_test")
        model_path = save_onnx_model_to_temp(bert_model)
        mock_tokenizer = create_mock_base_processor("tokenizer")
        
        with patch('onnx_auto_processor.AutoProcessor') as mock_auto:
            mock_auto.from_pretrained.return_value = mock_tokenizer
            
            processor = ONNXAutoProcessor.from_model(model_path, base_processor=mock_tokenizer)
            
            results_queue = queue.Queue()
            errors_queue = queue.Queue()
            
            def worker(worker_id):
                try:
                    for i in range(10):
                        text = f"Worker {worker_id} processing item {i}"
                        result = processor(text)
                        results_queue.put((worker_id, i, result))
                except Exception as e:
                    errors_queue.put((worker_id, e))
            
            # Start multiple threads
            threads = []
            for i in range(3):
                t = threading.Thread(target=worker, args=(i,))
                t.start()
                threads.append(t)
            
            # Wait for completion
            for t in threads:
                t.join()
            
            # Check results
            assert errors_queue.empty(), f"Concurrent processing errors: {list(errors_queue.queue)}"
            assert results_queue.qsize() == 30  # 3 workers * 10 items each
    
    # ========================================================================
    # Fixture-based Tests
    # ========================================================================
    
    def test_with_all_fixtures(
        self, 
        bert_onnx_model, vit_onnx_model, wav2vec2_onnx_model, 
        videomae_onnx_model, clip_onnx_model,
        mock_tokenizer, mock_image_processor, mock_feature_extractor
    ):
        """Test using all provided fixtures."""
        models_and_processors = [
            (bert_onnx_model, mock_tokenizer, "text"),
            (vit_onnx_model, mock_image_processor, "image"),
            (wav2vec2_onnx_model, mock_feature_extractor, "audio"),
        ]
        
        for model, processor, modality_name in models_and_processors:
            model_path = save_onnx_model_to_temp(model, f"fixture_{modality_name}.onnx")
            
            with patch('onnx_auto_processor.AutoProcessor') as mock_auto:
                mock_auto.from_pretrained.return_value = processor
                
                onnx_processor = ONNXAutoProcessor.from_model(model_path, base_processor=processor)
                assert onnx_processor is not None
                print(f"✓ Successfully created processor for {modality_name} using fixtures")


# Additional test configuration
pytest_plugins = ["test_utils"]

# Mark configuration for pytest
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")