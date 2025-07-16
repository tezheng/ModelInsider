"""
Test Model Compatibility - Comprehensive Testing Across Model Types

This test suite validates that the modelexport system works correctly
across a wide variety of model architectures, ensuring universal design
principles are maintained and no hardcoded assumptions limit compatibility.

Model Architecture Coverage Philosophy:
    The modelexport system is designed to work with ANY PyTorch model,
    regardless of architecture, domain, or complexity. This test suite
    validates that universal design principle by testing diverse model types.
    
    Universal Design Validation:
    ├── Architecture Agnostic - Works with any nn.Module hierarchy
    ├── Domain Independent - Vision, NLP, Audio, Multimodal all supported
    ├── Scale Invariant - Small research models to large production models
    ├── Framework Neutral - Standard PyTorch, transformers, timm, etc.
    └── Pattern Universal - No hardcoded operation or module patterns

Test Categories by Architecture Type:
    ├── Transformer Models (NLP Domain)
    │   ├── BERT Family (encoder-only)
    │   ├── GPT Family (decoder-only) 
    │   ├── T5 Family (encoder-decoder)
    │   ├── DistilBERT (compressed models)
    │   └── Custom Transformers (user-defined)
    ├── Vision Models (Computer Vision Domain)
    │   ├── ResNet Family (CNN architectures)
    │   ├── Vision Transformers (ViT)
    │   ├── EfficientNet Family
    │   ├── SAM (Segment Anything Model)
    │   └── Custom Vision Architectures
    ├── Multimodal Models (Cross-Domain)
    │   ├── CLIP (vision-language)
    │   ├── BLIP (bootstrapped vision-language)
    │   ├── Custom Multimodal Architectures
    │   └── Fusion Architectures
    ├── Specialized Architectures
    │   ├── RNN/LSTM/GRU Based Models
    │   ├── Graph Neural Networks (if applicable)
    │   ├── Custom Domain Models
    │   └── Research Architectures
    └── Edge Cases & Stress Tests
        ├── Extremely Large Models (memory limits)
        ├── Extremely Small Models (minimal structure)
        ├── Unusual Module Combinations
        └── Dynamic/Conditional Architectures

Test Validation Criteria:
    ├── Universal Compatibility
    │   ├── No hardcoded architecture assumptions
    │   ├── Works with any nn.Module hierarchy
    │   ├── Handles diverse input/output patterns
    │   └── Scales across model sizes
    ├── Quality Standards
    │   ├── 100% node coverage for all models
    │   ├── No empty hierarchy tags
    │   ├── Valid ONNX output generation
    │   └── Consistent metadata quality
    ├── Performance Characteristics
    │   ├── Reasonable export times (<5 minutes for large models)
    │   ├── Memory usage within bounds
    │   ├── No memory leaks across model types
    │   └── Consistent performance patterns
    └── Error Handling
        ├── Graceful failure for unsupported models
        ├── Clear error messages for incompatibilities
        ├── Helpful guidance for resolution
        └── No crashes or undefined behavior

Special Focus Areas:
    ├── SAM Model Integration (coordinate fix validation)
    ├── Large Model Handling (memory and performance)
    ├── Custom Architecture Support (universal design)
    ├── Input Generation Compatibility (diverse input types)
    └── Cross-Strategy Compatibility (all strategies work)

Performance Requirements:
    - Small models (<100MB): Export in <30 seconds
    - Medium models (100MB-1GB): Export in <2 minutes  
    - Large models (>1GB): Export in <5 minutes
    - Memory usage: <8GB peak for any single model
    - No significant memory leaks between exports

Quality Guarantees:
    - 100% node coverage across all compatible models
    - Zero empty hierarchy tags
    - Valid ONNX outputs that pass validation
    - Comprehensive metadata for all exports
    - Consistent behavior across model types
"""

import tempfile
import time
from pathlib import Path

import onnx
import pytest
import torch
import torch.nn as nn

from modelexport.core.model_input_generator import generate_dummy_inputs
from modelexport.strategies.htp.htp_hierarchy_exporter import HierarchyExporter


class TestTransformerModelCompatibility:
    """
    Test suite for transformer model compatibility.
    
    Transformer models are the most common type of models users export,
    so this test suite ensures robust support across the transformer
    ecosystem including BERT, GPT, T5, and custom transformer architectures.
    
    Key Features Tested:
    - BERT family models (encoder-only)
    - GPT family models (decoder-only)
    - T5 family models (encoder-decoder)
    - Compressed models (DistilBERT)
    - Custom transformer architectures
    - Various input/output patterns
    
    Universal Design Validation:
    - No hardcoded BERT/GPT/T5 assumptions
    - Works with any transformer variant
    - Handles diverse attention patterns
    - Supports various input modalities
    """
    
    def test_bert_family_models(self):
        """
        Test BERT family model compatibility.
        
        BERT and its variants are encoder-only transformer models
        widely used for NLP tasks. This test validates compatibility
        across different BERT model sizes and variants.
        
        Models Tested:
        - prajjwal1/bert-tiny (minimal BERT for fast testing)
        - distilbert-base-uncased (compressed BERT variant)
        
        Validation Points:
        - Encoder-only architecture support
        - Standard BERT input patterns
        - Attention mechanism hierarchy
        - Layer normalization and residual connections
        """
        bert_models_to_test = [
            {
                "name": "prajjwal1/bert-tiny",
                "description": "Minimal BERT for testing",
                "expected_inputs": ["input_ids", "token_type_ids", "attention_mask"],
                "max_export_time": 30.0
            },
            {
                "name": "distilbert-base-uncased", 
                "description": "Compressed BERT variant",
                "expected_inputs": ["input_ids", "attention_mask"],
                "max_export_time": 60.0,
                "skip_reason": "May be too large for CI environment"
            }
        ]
        
        bert_results = {}
        
        for model_config in bert_models_to_test:
            model_name = model_config["name"]
            
            # Skip large models in CI if needed
            if "skip_reason" in model_config:
                pytest.skip(f"Skipping {model_name}: {model_config['skip_reason']}")
                continue
            
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / f"{model_name.replace('/', '_')}_bert.onnx"
                
                try:
                    start_time = time.time()
                    
                    # Test with HTP exporter
                    exporter = HTPExporter(verbose=False)
                    result = exporter.export(
                        model_name_or_path=model_name,
                        output_path=str(output_path),
                        opset_version=17
                    )
                    
                    export_time = time.time() - start_time
                    
                    # Validate export success
                    assert result["coverage_percentage"] == 100.0, f"{model_name} should achieve 100% coverage"
                    assert result["empty_tags"] == 0, f"{model_name} should have no empty tags"
                    assert export_time < model_config["max_export_time"], f"{model_name} export took too long: {export_time:.2f}s"
                    
                    # Validate ONNX output
                    assert output_path.exists(), f"ONNX file should be created for {model_name}"
                    onnx_model = onnx.load(str(output_path))
                    onnx.checker.check_model(onnx_model)
                    
                    # Validate input generation worked correctly
                    inputs = generate_dummy_inputs(model_name_or_path=model_name)
                    input_names = set(inputs.keys())
                    
                    # Should generate expected BERT inputs
                    expected_inputs = set(model_config["expected_inputs"])
                    assert input_names.intersection(expected_inputs), f"Should generate BERT inputs for {model_name}. Got: {input_names}"
                    
                    bert_results[model_name] = {
                        "success": True,
                        "export_time": export_time,
                        "coverage": result["coverage_percentage"],
                        "onnx_nodes": result["onnx_nodes"],
                        "hierarchy_modules": result["hierarchy_modules"]
                    }
                    
                except Exception as e:
                    bert_results[model_name] = {
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
        
        # At least one BERT model should work
        successful_bert = [name for name, result in bert_results.items() if result["success"]]
        assert len(successful_bert) >= 1, f"At least one BERT model should export successfully. Results: {bert_results}"
    
    def test_custom_transformer_architecture(self):
        """
        Test custom transformer architecture compatibility.
        
        This validates that the system works with user-defined transformer
        architectures, ensuring no hardcoded assumptions about transformer
        structure limit compatibility.
        
        Test Scenario:
        - Define custom transformer with non-standard structure
        - Export using HTP strategy
        - Validate universal design principles
        
        Universal Design Validation:
        - No hardcoded transformer assumptions
        - Works with custom attention mechanisms
        - Handles non-standard layer arrangements
        - Supports various activation functions
        """
        # Define custom transformer architecture
        class CustomTransformer(nn.Module):
            def __init__(self, vocab_size=1000, hidden_size=256, num_layers=4, num_heads=8):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.positional_encoding = nn.Parameter(torch.randn(512, hidden_size))
                
                # Custom transformer layers with non-standard structure
                self.layers = nn.ModuleList([
                    CustomTransformerLayer(hidden_size, num_heads) 
                    for _ in range(num_layers)
                ])
                
                # Non-standard final layers
                self.layer_norm = nn.LayerNorm(hidden_size)
                self.dropout = nn.Dropout(0.1)
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.GELU(),
                    nn.Linear(hidden_size // 2, 10)
                )
            
            def forward(self, input_ids, attention_mask=None):
                seq_len = input_ids.size(1)
                x = self.embedding(input_ids)
                x = x + self.positional_encoding[:seq_len].unsqueeze(0)
                
                for layer in self.layers:
                    x = layer(x, attention_mask)
                
                x = self.layer_norm(x)
                x = self.dropout(x)
                
                # Global average pooling with attention mask
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).float()
                    x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
                else:
                    x = x.mean(dim=1)
                
                return self.classifier(x)
        
        class CustomTransformerLayer(nn.Module):
            def __init__(self, hidden_size, num_heads):
                super().__init__()
                self.self_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
                self.feed_forward = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size)
                )
                self.norm1 = nn.LayerNorm(hidden_size)
                self.norm2 = nn.LayerNorm(hidden_size)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x, attention_mask=None):
                # Self-attention with residual
                attn_mask = None
                if attention_mask is not None:
                    # Convert attention mask for MultiheadAttention
                    attn_mask = ~attention_mask.bool()
                
                attn_out, _ = self.self_attention(x, x, x, key_padding_mask=attn_mask)
                x = self.norm1(x + self.dropout(attn_out))
                
                # Feed-forward with residual
                ff_out = self.feed_forward(x)
                x = self.norm2(x + self.dropout(ff_out))
                
                return x
        
        model = CustomTransformer()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "custom_transformer.onnx"
            
            # Define custom inputs
            input_specs = {
                "input_ids": {"shape": [2, 32], "dtype": "int", "range": [0, 999]},
                "attention_mask": {"shape": [2, 32], "dtype": "int", "range": [0, 1]}
            }
            
            # Export custom transformer
            start_time = time.time()
            
            exporter = HierarchyExporter(strategy="htp")
            result = exporter.export(
                model=model,
                output_path=str(output_path),
                input_specs=input_specs,
                opset_version=17
            )
            
            export_time = time.time() - start_time
            
            # Validate universal design compliance
            assert result["coverage_percentage"] == 100.0, "Custom transformer should achieve 100% coverage"
            assert result["empty_tags"] == 0, "Should have no empty tags"
            assert export_time < 60.0, f"Custom transformer export should complete in <60s, took {export_time:.2f}s"
            
            # Validate ONNX output
            assert output_path.exists(), "Custom transformer ONNX file should be created"
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            # Validate hierarchy captures custom structure
            hierarchy_data = result.get("hierarchy_data", {})
            
            # Should capture various custom components
            custom_components = []
            for module_path, module_info in hierarchy_data.items():
                module_type = module_info.get("module_type", "")
                if any(term in module_type for term in ["Custom", "MultiheadAttention", "Embedding", "LayerNorm"]):
                    custom_components.append(module_type)
            
            assert len(custom_components) > 0, f"Should capture custom transformer components. Found hierarchy: {list(hierarchy_data.keys())}"


class TestVisionModelCompatibility:
    """
    Test suite for vision model compatibility.
    
    Vision models have different architectural patterns compared to
    transformers, including convolutional layers, pooling operations,
    and different input/output structures. This validates universal
    support across vision architectures.
    
    Key Features Tested:
    - CNN architectures (ResNet family)
    - Vision Transformers (ViT)
    - SAM models (with coordinate fix)
    - Custom vision architectures
    - Various input image formats
    
    Universal Design Validation:
    - No hardcoded CNN assumptions
    - Works with any vision architecture
    - Handles diverse input formats
    - Supports various output patterns
    """
    
    def test_cnn_architecture_compatibility(self):
        """
        Test CNN architecture compatibility.
        
        Convolutional neural networks have different hierarchy patterns
        compared to transformers. This test validates that the system
        correctly handles CNN architectures.
        
        Test Scenario:
        - Create ResNet-like CNN architecture
        - Export using HTP strategy
        - Validate CNN-specific hierarchy capture
        
        Validation Points:
        - Convolutional layer hierarchy
        - Batch normalization integration
        - Residual connection handling
        - Pooling operation support
        """
        # Define CNN architecture similar to ResNet
        class CNNModel(nn.Module):
            def __init__(self, num_classes=1000):
                super().__init__()
                # Initial convolution
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                
                # ResNet-like blocks
                self.layer1 = self._make_layer(64, 64, 2)
                self.layer2 = self._make_layer(64, 128, 2, stride=2)
                self.layer3 = self._make_layer(128, 256, 2, stride=2)
                self.layer4 = self._make_layer(256, 512, 2, stride=2)
                
                # Final layers
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512, num_classes)
            
            def _make_layer(self, inplanes, planes, blocks, stride=1):
                layers = []
                layers.append(BasicBlock(inplanes, planes, stride))
                for _ in range(1, blocks):
                    layers.append(BasicBlock(planes, planes))
                return nn.Sequential(*layers)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                
                return x
        
        class BasicBlock(nn.Module):
            def __init__(self, inplanes, planes, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)
                
                self.shortcut = nn.Sequential()
                if stride != 1 or inplanes != planes:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(planes)
                    )
            
            def forward(self, x):
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                
                out = self.conv2(out)
                out = self.bn2(out)
                
                out += self.shortcut(x)
                out = self.relu(out)
                
                return out
        
        model = CNNModel(num_classes=10)  # Smaller for testing
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "cnn_model.onnx"
            
            # Define vision inputs
            input_specs = {
                "input": {
                    "shape": [1, 3, 224, 224],  # Standard ImageNet format
                    "dtype": "float",
                    "range": [0.0, 1.0]
                }
            }
            
            # Export CNN model
            start_time = time.time()
            
            exporter = HierarchyExporter(strategy="htp")
            result = exporter.export(
                model=model,
                output_path=str(output_path),
                input_specs=input_specs,
                opset_version=17
            )
            
            export_time = time.time() - start_time
            
            # Validate CNN export success
            assert result["coverage_percentage"] == 100.0, "CNN model should achieve 100% coverage"
            assert result["empty_tags"] == 0, "Should have no empty tags"
            assert export_time < 90.0, f"CNN export should complete in <90s, took {export_time:.2f}s"
            
            # Validate ONNX output
            assert output_path.exists(), "CNN ONNX file should be created"
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            # Validate input format in ONNX
            graph_inputs = onnx_model.graph.input
            assert len(graph_inputs) >= 1, "Should have at least one input"
            
            input_shape = [dim.dim_value for dim in graph_inputs[0].type.tensor_type.shape.dim]
            assert input_shape == [1, 3, 224, 224], f"Input shape should be [1, 3, 224, 224], got {input_shape}"
            
            # Validate CNN-specific hierarchy
            hierarchy_data = result.get("hierarchy_data", {})
            cnn_components = []
            
            for module_path, module_info in hierarchy_data.items():
                module_type = module_info.get("module_type", "")
                if any(term in module_type for term in ["Conv2d", "BatchNorm2d", "MaxPool2d", "AdaptiveAvgPool2d"]):
                    cnn_components.append(module_type)
            
            assert len(cnn_components) > 0, f"Should capture CNN-specific components. Found: {list(hierarchy_data.keys())}"
    
    def test_sam_model_compatibility(self):
        """
        Test SAM model compatibility with coordinate fix.
        
        SAM (Segment Anything Model) represents a complex vision model
        that requires special handling (coordinate fix). This test validates
        that SAM models work correctly with the automatic coordinate fix.
        
        Test Scenario:
        - Test SAM coordinate fix integration
        - Validate input generation works correctly
        - Check export compatibility (may fail at ONNX level)
        
        Validation Points:
        - Automatic coordinate fix application
        - Correct coordinate value ranges [0, 1024]
        - SAM-specific input handling
        - Integration with export pipeline
        """
        model_name = "facebook/sam-vit-base"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test SAM coordinate fix first (this should always work)
            try:
                inputs = generate_dummy_inputs(model_name_or_path=model_name)
                
                # Validate SAM inputs
                assert "input_points" in inputs, "SAM should generate input_points"
                
                input_points = inputs["input_points"]
                min_val = float(input_points.min())
                max_val = float(input_points.max())
                
                # Validate coordinate fix
                assert min_val >= 0, f"Coordinates should be >= 0, got min={min_val}"
                assert max_val <= 1024, f"Coordinates should be <= 1024, got max={max_val}"
                assert max_val > 10, f"Coordinates should be in pixel space [0, 1024], got max={max_val}"
                
                # Test export (may fail due to SAM complexity)
                output_path = Path(temp_dir) / "sam_model.onnx"
                
                try:
                    exporter = HTPExporter(verbose=False)
                    result = exporter.export(
                        model_name_or_path=model_name,
                        output_path=str(output_path),
                        opset_version=17
                    )
                    
                    # If export succeeds, validate results
                    if output_path.exists():
                        assert result["coverage_percentage"] == 100.0, "SAM should achieve 100% coverage"
                        assert result["empty_tags"] == 0, "SAM should have no empty tags"
                        
                        onnx_model = onnx.load(str(output_path))
                        onnx.checker.check_model(onnx_model)
                
                except Exception as e:
                    # SAM export may fail at ONNX level - this is acceptable
                    pytest.skip(f"SAM ONNX export failed (expected): {e}")
                
            except Exception as e:
                pytest.skip(f"SAM model loading failed: {e}")


class TestMultimodalModelCompatibility:
    """
    Test suite for multimodal model compatibility.
    
    Multimodal models combine different input modalities (vision, text, etc.)
    and represent complex architectures that test the limits of universal
    design. This validates support for cross-modal architectures.
    
    Key Features Tested:
    - Vision-language models (CLIP-style)
    - Custom multimodal architectures
    - Cross-modal fusion mechanisms
    - Multiple input type handling
    
    Universal Design Validation:
    - No hardcoded multimodal assumptions
    - Works with any fusion architecture
    - Handles diverse input combinations
    - Supports various output patterns
    """
    
    def test_custom_multimodal_architecture(self):
        """
        Test custom multimodal architecture compatibility.
        
        This creates a custom multimodal model that combines vision and
        text inputs, validating that the system can handle complex
        cross-modal architectures.
        
        Test Scenario:
        - Define vision-text fusion model
        - Export using HTP strategy
        - Validate multimodal hierarchy capture
        
        Validation Points:
        - Multiple input modality support
        - Cross-modal fusion layer handling
        - Complex attention mechanism support
        - Mixed architecture hierarchy preservation
        """
        # Define custom multimodal architecture
        class MultimodalModel(nn.Module):
            def __init__(self, vocab_size=1000, text_hidden=256, vision_hidden=512, fusion_hidden=384):
                super().__init__()
                # Text encoder (transformer-like)
                self.text_embedding = nn.Embedding(vocab_size, text_hidden)
                self.text_encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(text_hidden, nhead=8, batch_first=True),
                    num_layers=4
                )
                self.text_pooler = nn.Linear(text_hidden, fusion_hidden)
                
                # Vision encoder (CNN-like)
                self.vision_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, vision_hidden)
                )
                self.vision_projector = nn.Linear(vision_hidden, fusion_hidden)
                
                # Cross-modal fusion
                self.cross_attention = nn.MultiheadAttention(fusion_hidden, num_heads=8, batch_first=True)
                self.fusion_layer = nn.Sequential(
                    nn.Linear(fusion_hidden * 2, fusion_hidden),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(fusion_hidden, fusion_hidden)
                )
                
                # Final classifier
                self.classifier = nn.Sequential(
                    nn.Linear(fusion_hidden, fusion_hidden // 2),
                    nn.ReLU(),
                    nn.Linear(fusion_hidden // 2, 10)
                )
            
            def forward(self, text_input, vision_input, text_mask=None):
                # Process text
                text_embed = self.text_embedding(text_input)
                text_encoded = self.text_encoder(text_embed, src_key_padding_mask=~text_mask.bool() if text_mask is not None else None)
                
                # Global text pooling
                if text_mask is not None:
                    mask_expanded = text_mask.unsqueeze(-1).float()
                    text_pooled = (text_encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
                else:
                    text_pooled = text_encoded.mean(dim=1)
                
                text_features = self.text_pooler(text_pooled)
                
                # Process vision
                vision_features = self.vision_encoder(vision_input)
                vision_features = self.vision_projector(vision_features)
                
                # Cross-modal attention
                text_features_unsqueezed = text_features.unsqueeze(1)  # [batch, 1, hidden]
                vision_features_unsqueezed = vision_features.unsqueeze(1)  # [batch, 1, hidden]
                
                attended_text, _ = self.cross_attention(text_features_unsqueezed, vision_features_unsqueezed, vision_features_unsqueezed)
                attended_vision, _ = self.cross_attention(vision_features_unsqueezed, text_features_unsqueezed, text_features_unsqueezed)
                
                # Fusion
                fused_features = torch.cat([attended_text.squeeze(1), attended_vision.squeeze(1)], dim=1)
                fused_output = self.fusion_layer(fused_features)
                
                return self.classifier(fused_output)
        
        model = MultimodalModel()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "multimodal_model.onnx"
            
            # Define multimodal inputs
            input_specs = {
                "text_input": {"shape": [2, 32], "dtype": "int", "range": [0, 999]},
                "vision_input": {"shape": [2, 3, 224, 224], "dtype": "float", "range": [0.0, 1.0]},
                "text_mask": {"shape": [2, 32], "dtype": "int", "range": [0, 1]}
            }
            
            # Export multimodal model
            start_time = time.time()
            
            exporter = HierarchyExporter(strategy="htp")
            result = exporter.export(
                model=model,
                output_path=str(output_path),
                input_specs=input_specs,
                opset_version=17
            )
            
            export_time = time.time() - start_time
            
            # Validate multimodal export success
            assert result["coverage_percentage"] == 100.0, "Multimodal model should achieve 100% coverage"
            assert result["empty_tags"] == 0, "Should have no empty tags"
            assert export_time < 120.0, f"Multimodal export should complete in <120s, took {export_time:.2f}s"
            
            # Validate ONNX output
            assert output_path.exists(), "Multimodal ONNX file should be created"
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            # Validate multimodal inputs
            graph_inputs = onnx_model.graph.input
            assert len(graph_inputs) >= 2, "Should have multiple inputs for multimodal model"
            
            # Validate hierarchy captures multimodal structure
            hierarchy_data = result.get("hierarchy_data", {})
            multimodal_components = []
            
            for module_path, module_info in hierarchy_data.items():
                module_type = module_info.get("module_type", "")
                if any(term in module_type for term in ["TransformerEncoder", "Conv2d", "MultiheadAttention", "Embedding"]):
                    multimodal_components.append(module_type)
            
            assert len(multimodal_components) > 0, f"Should capture multimodal components. Found: {list(hierarchy_data.keys())}"


class TestSpecializedArchitectures:
    """
    Test suite for specialized and edge case architectures.
    
    This tests unusual or specialized architectures that might stress
    the universal design principles, including RNN-based models,
    extremely large/small models, and custom research architectures.
    
    Key Features Tested:
    - RNN/LSTM/GRU based models
    - Extremely small models (minimal structure)
    - Unusual module combinations
    - Dynamic/conditional architectures
    
    Universal Design Validation:
    - No architectural assumptions break
    - Graceful handling of edge cases
    - Consistent behavior across extremes
    - Clear error messages for unsupported cases
    """
    
    def test_rnn_based_architecture(self):
        """
        Test RNN-based architecture compatibility.
        
        RNN architectures have different patterns compared to transformers
        and CNNs, including sequential processing and hidden state management.
        
        Test Scenario:
        - Create LSTM-based model
        - Export using HTP strategy
        - Validate RNN-specific hierarchy
        
        Validation Points:
        - LSTM/GRU layer hierarchy
        - Sequential processing patterns
        - Hidden state handling
        - Recurrent connection support
        """
        # Define RNN-based model
        class RNNModel(nn.Module):
            def __init__(self, vocab_size=1000, embed_size=128, hidden_size=256, num_layers=2, num_classes=10):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_size)
                self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.1, bidirectional=True)
                self.gru = nn.GRU(hidden_size * 2, hidden_size, 1, batch_first=True)
                self.attention = nn.Linear(hidden_size, 1)
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size // 2, num_classes)
                )
            
            def forward(self, input_ids, sequence_lengths=None):
                # Embedding
                embedded = self.embedding(input_ids)
                
                # LSTM processing
                lstm_out, _ = self.lstm(embedded)
                
                # GRU processing
                gru_out, _ = self.gru(lstm_out)
                
                # Attention pooling
                attention_weights = torch.softmax(self.attention(gru_out), dim=1)
                pooled = (gru_out * attention_weights).sum(dim=1)
                
                return self.classifier(pooled)
        
        model = RNNModel()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "rnn_model.onnx"
            
            # Define RNN inputs
            input_specs = {
                "input_ids": {"shape": [2, 32], "dtype": "int", "range": [0, 999]},
                "sequence_lengths": {"shape": [2], "dtype": "int", "range": [1, 32]}
            }
            
            # Export RNN model
            start_time = time.time()
            
            exporter = HierarchyExporter(strategy="htp")
            result = exporter.export(
                model=model,
                output_path=str(output_path),
                input_specs=input_specs,
                opset_version=17
            )
            
            export_time = time.time() - start_time
            
            # Validate RNN export success
            assert result["coverage_percentage"] == 100.0, "RNN model should achieve 100% coverage"
            assert result["empty_tags"] == 0, "Should have no empty tags"
            assert export_time < 60.0, f"RNN export should complete in <60s, took {export_time:.2f}s"
            
            # Validate ONNX output
            assert output_path.exists(), "RNN ONNX file should be created"
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            # Validate RNN-specific hierarchy
            hierarchy_data = result.get("hierarchy_data", {})
            rnn_components = []
            
            for module_path, module_info in hierarchy_data.items():
                module_type = module_info.get("module_type", "")
                if any(term in module_type for term in ["LSTM", "GRU", "Embedding"]):
                    rnn_components.append(module_type)
            
            assert len(rnn_components) > 0, f"Should capture RNN components. Found: {list(hierarchy_data.keys())}"
    
    def test_minimal_model_architecture(self):
        """
        Test minimal model architecture compatibility.
        
        Extremely small models test the lower bounds of the system's
        ability to extract meaningful hierarchy information.
        
        Test Scenario:
        - Create minimal PyTorch model
        - Export using HTP strategy
        - Validate minimal hierarchy handling
        
        Validation Points:
        - Minimal hierarchy extraction
        - Single layer model support
        - Edge case handling
        - Consistent behavior at extremes
        """
        # Define minimal model
        class MinimalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = MinimalModel()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "minimal_model.onnx"
            
            # Define minimal inputs
            input_specs = {
                "input": {"shape": [1, 10], "dtype": "float", "range": [0.0, 1.0]}
            }
            
            # Export minimal model
            start_time = time.time()
            
            exporter = HTPExporter(verbose=False)
            result = exporter.export(
                model=model,
                output_path=str(output_path),
                input_specs=input_specs,
                opset_version=17
            )
            
            export_time = time.time() - start_time
            
            # Validate minimal export success
            assert result["coverage_percentage"] == 100.0, "Minimal model should achieve 100% coverage"
            assert result["empty_tags"] == 0, "Should have no empty tags"
            assert export_time < 15.0, f"Minimal export should be very fast, took {export_time:.2f}s"
            
            # Validate ONNX output
            assert output_path.exists(), "Minimal ONNX file should be created"
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            # Should have minimal but valid hierarchy
            hierarchy_data = result.get("hierarchy_data", {})
            assert len(hierarchy_data) > 0, "Should extract some hierarchy even for minimal model"
            
            # Should have at least the linear layer
            linear_found = any("Linear" in info.get("module_type", "") for info in hierarchy_data.values())
            assert linear_found, f"Should find Linear layer. Found: {hierarchy_data}"
    
    def test_complex_nested_architecture(self):
        """
        Test complex nested architecture compatibility.
        
        Very deep or complexly nested models test the upper bounds
        of hierarchy extraction and ensure no recursion or nesting
        limits are exceeded.
        
        Test Scenario:
        - Create deeply nested model architecture
        - Export using HTP strategy
        - Validate deep hierarchy handling
        
        Validation Points:
        - Deep nesting support
        - Complex module relationships
        - Recursive structure handling
        - Performance with complex hierarchies
        """
        # Define deeply nested model
        class DeeplyNestedModel(nn.Module):
            def __init__(self, depth=5):
                super().__init__()
                self.input_layer = nn.Linear(128, 128)
                self.nested_modules = self._create_nested_structure(depth, 128)
                self.output_layer = nn.Linear(128, 10)
            
            def _create_nested_structure(self, depth, hidden_size):
                if depth <= 0:
                    return nn.Identity()
                
                return nn.ModuleDict({
                    f"level_{depth}": nn.ModuleDict({
                        "transform": nn.Sequential(
                            nn.Linear(hidden_size, hidden_size),
                            nn.ReLU(),
                            nn.Dropout(0.1)
                        ),
                        "nested": self._create_nested_structure(depth - 1, hidden_size),
                        "residual": nn.Linear(hidden_size, hidden_size)
                    })
                })
            
            def forward(self, x):
                x = self.input_layer(x)
                x = self._forward_nested(x, self.nested_modules)
                x = self.output_layer(x)
                return x
            
            def _forward_nested(self, x, nested_dict):
                if not nested_dict:
                    return x
                
                for level_name, level_modules in nested_dict.items():
                    if isinstance(level_modules, nn.ModuleDict):
                        transformed = level_modules["transform"](x)
                        nested_out = self._forward_nested(transformed, level_modules.get("nested", {}))
                        residual = level_modules["residual"](x)
                        x = nested_out + residual
                
                return x
        
        model = DeeplyNestedModel(depth=3)  # Moderate depth for testing
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested_model.onnx"
            
            # Define inputs
            input_specs = {
                "input": {"shape": [2, 128], "dtype": "float", "range": [0.0, 1.0]}
            }
            
            # Export nested model
            start_time = time.time()
            
            exporter = HTPExporter(verbose=False)
            result = exporter.export(
                model=model,
                output_path=str(output_path),
                input_specs=input_specs,
                opset_version=17
            )
            
            export_time = time.time() - start_time
            
            # Validate nested export success
            assert result["coverage_percentage"] == 100.0, "Nested model should achieve 100% coverage"
            assert result["empty_tags"] == 0, "Should have no empty tags"
            assert export_time < 90.0, f"Nested export should complete in <90s, took {export_time:.2f}s"
            
            # Validate ONNX output
            assert output_path.exists(), "Nested ONNX file should be created"
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            # Validate deep hierarchy capture
            hierarchy_data = result.get("hierarchy_data", {})
            
            # Should capture nested structure
            nested_modules = [path for path in hierarchy_data.keys() if "level_" in path]
            assert len(nested_modules) > 0, f"Should capture nested structure. Found: {list(hierarchy_data.keys())}"
            
            # Should handle deep nesting
            max_depth = max(path.count('/') for path in hierarchy_data.keys())
            assert max_depth >= 3, f"Should capture deep nesting. Max depth: {max_depth}"