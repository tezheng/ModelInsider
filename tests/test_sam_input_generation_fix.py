"""
Test SAM model input generation fix (TEZ-48).

This test validates that SAM models generate pixel_values instead of embeddings
for full model export, ensuring the vision encoder is included in the trace.
"""

import pytest
import torch
from modelexport.core.model_input_generator import generate_dummy_inputs_from_model_path


class TestSAMInputGenerationFix:
    """Test TEZ-48 fix for SAM model input generation."""
    
    def test_sam_generates_pixel_values(self):
        """Test that SAM models generate pixel_values for full model export."""
        # Generate inputs for SAM model
        inputs = generate_dummy_inputs_from_model_path('facebook/sam-vit-base')
        
        # TEZ-48 Fix: Should generate pixel_values, not embeddings
        assert 'pixel_values' in inputs, "SAM should generate pixel_values for full model export"
        assert 'image_embeddings' not in inputs, "SAM should not generate embeddings (bypasses vision encoder)"
        assert 'image_positional_embeddings' not in inputs, "SAM should not generate positional embeddings"
        
        # Check required inputs are present
        assert 'input_points' in inputs, "SAM should generate input_points"
        assert 'input_labels' in inputs, "SAM should generate input_labels"
        
        # Validate tensor shapes and types
        pixel_values = inputs['pixel_values']
        assert pixel_values.shape == torch.Size([2, 3, 1024, 1024]), f"Unexpected pixel_values shape: {pixel_values.shape}"
        assert pixel_values.dtype == torch.float32, f"Unexpected pixel_values dtype: {pixel_values.dtype}"
        
        input_points = inputs['input_points']
        assert input_points.shape == torch.Size([2, 1, 1, 2]), f"Unexpected input_points shape: {input_points.shape}"
        assert input_points.dtype == torch.float32, f"Unexpected input_points dtype: {input_points.dtype}"
        
        input_labels = inputs['input_labels']
        assert input_labels.shape == torch.Size([2, 1, 1]), f"Unexpected input_labels shape: {input_labels.shape}"
        assert input_labels.dtype == torch.int64, f"Unexpected input_labels dtype: {input_labels.dtype}"
    
    def test_sam_inputs_work_with_model(self):
        """Test that generated SAM inputs work with the actual model."""
        from transformers import SamModel
        
        # Generate inputs
        inputs = generate_dummy_inputs_from_model_path('facebook/sam-vit-base')
        
        # Load model and test inference
        model = SamModel.from_pretrained('facebook/sam-vit-base')
        model.eval()
        
        # Test that inputs work with the model
        with torch.no_grad():
            try:
                # Use batch_size=1 for testing
                test_inputs = {
                    'pixel_values': inputs['pixel_values'][:1],  # Take first batch
                    'input_points': inputs['input_points'][:1],
                    'input_labels': inputs['input_labels'][:1]
                }
                
                output = model(**test_inputs)
                
                # Validate output structure
                assert 'pred_masks' in output, "SAM should output pred_masks"
                assert 'iou_scores' in output, "SAM should output iou_scores"
                
                # Validate output shapes
                pred_masks = output.pred_masks
                iou_scores = output.iou_scores
                
                assert len(pred_masks.shape) == 5, f"pred_masks should be 5D, got shape: {pred_masks.shape}"
                assert len(iou_scores.shape) == 3, f"iou_scores should be 3D, got shape: {iou_scores.shape}"
                
            except Exception as e:
                pytest.fail(f"Generated inputs failed model inference: {e}")
    
    def test_sam_export_includes_vision_encoder(self):
        """Test that SAM export with fixed inputs includes the vision encoder."""
        import tempfile
        from pathlib import Path
        from modelexport.strategies.htp_new.htp_exporter import HTPExporter
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "sam_test.onnx"
            
            # Export SAM with HTP strategy
            exporter = HTPExporter()
            result = exporter.export(
                model_name_or_path='facebook/sam-vit-base',
                output_path=str(output_path)
            )
            
            # Validate that vision encoder modules are traced  
            hierarchy_data = exporter._hierarchy_data
            
            # Check for vision encoder presence
            vision_encoder_found = False
            for module_name in hierarchy_data.keys():
                if 'vision_encoder' in module_name.lower():
                    vision_encoder_found = True
                    break
            
            assert vision_encoder_found, "Vision encoder modules should be present in hierarchy"
            
            # TEZ-48 Success Criteria: Should trace significantly more modules than before
            traced_modules = len(hierarchy_data)
            assert traced_modules > 200, f"Should trace >200 modules (got {traced_modules}), indicating full model export"
            
            # Validate ONNX file exists and has reasonable size
            assert output_path.exists(), "ONNX file should be created"
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            assert file_size_mb > 300, f"ONNX file should be >300MB for full model (got {file_size_mb:.1f}MB)"


class TestSAMInputGenerationRegression:
    """Regression test to ensure we don't revert to embedding generation."""
    
    def test_sam_does_not_generate_embeddings(self):
        """Regression test: Ensure SAM doesn't generate embeddings that bypass vision encoder."""
        inputs = generate_dummy_inputs_from_model_path('facebook/sam-vit-base')
        
        # These inputs would indicate a regression to the old behavior
        regression_inputs = [
            'image_embeddings',
            'image_positional_embeddings'
        ]
        
        for bad_input in regression_inputs:
            assert bad_input not in inputs, f"Regression detected: {bad_input} should not be generated (TEZ-48)"
        
        # Ensure we have the correct inputs
        required_inputs = ['pixel_values', 'input_points', 'input_labels']
        for required_input in required_inputs:
            assert required_input in inputs, f"Required input missing: {required_input} (TEZ-48 fix)"
    
    def test_sam_separate_component_export(self):
        """Test how to properly export SAM components separately using Optimum."""
        from optimum.exporters.onnx.model_configs import SamOnnxConfig
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained('facebook/sam-vit-base')
        
        # Test 1: Vision encoder export (Optimum's intended way)
        encoder_config = SamOnnxConfig(
            config=config,
            task='feature-extraction',
            vision_encoder=True  # This is the key!
        )
        encoder_inputs = encoder_config.generate_dummy_inputs(framework='pt')
        
        # Vision encoder should only have pixel_values input
        assert 'pixel_values' in encoder_inputs, "Vision encoder should have pixel_values"
        assert 'image_embeddings' not in encoder_inputs, "Vision encoder should not have embeddings as input"
        assert 'input_points' not in encoder_inputs, "Vision encoder should not have prompts"
        
        # Test 2: Mask decoder export (default Optimum behavior)
        decoder_config = SamOnnxConfig(
            config=config,
            task='feature-extraction',
            vision_encoder=False  # Default, but explicit for clarity
        )
        decoder_inputs = decoder_config.generate_dummy_inputs(framework='pt')
        
        # Mask decoder should have embeddings and prompts
        assert 'image_embeddings' in decoder_inputs, "Mask decoder needs embeddings"
        assert 'image_positional_embeddings' in decoder_inputs, "Mask decoder needs positional embeddings"
        assert 'input_points' in decoder_inputs, "Mask decoder needs points"
        assert 'input_labels' in decoder_inputs, "Mask decoder needs labels"
        assert 'pixel_values' not in decoder_inputs, "Mask decoder should not have pixel_values"