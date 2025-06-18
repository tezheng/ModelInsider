#!/usr/bin/env python3
"""
Universal Input Generator for Different Model Architectures
Generate appropriate dummy inputs for any Hugging Face or torchvision model
"""

import torch
import torch.nn as nn
import inspect
from typing import Dict, Any, Union, Tuple
from transformers import AutoModel, AutoTokenizer
import torchvision.models as models


class UniversalInputGenerator:
    """Generate appropriate inputs for different model architectures"""
    
    def __init__(self):
        self.batch_size = 1
        self.default_seq_length = 32
        self.default_vocab_size = 30000
        self.default_image_size = 224
    
    def generate_inputs(self, model: nn.Module, model_name: str = None) -> Dict[str, torch.Tensor]:
        """
        Generate appropriate inputs for any model based on its architecture
        
        Args:
            model: The PyTorch model
            model_name: Optional model name for additional context
            
        Returns:
            Dictionary of input tensors
        """
        # Detect model type
        model_type = self._detect_model_type(model, model_name)
        
        if model_type == "transformer":
            return self._generate_transformer_inputs(model, model_name)
        elif model_type == "vision":
            return self._generate_vision_inputs(model, model_name)
        elif model_type == "vision_transformer":
            return self._generate_vision_transformer_inputs(model, model_name)
        else:
            # Fallback: try to infer from forward signature
            return self._generate_generic_inputs(model)
    
    def _detect_model_type(self, model: nn.Module, model_name: str = None) -> str:
        """Detect the type of model architecture"""
        model_class_name = type(model).__name__.lower()
        
        # Vision transformers (ViT)
        if any(keyword in model_class_name for keyword in ['vit', 'vision_transformer']):
            return "vision_transformer"
        
        # Traditional vision models (ResNet, etc.)
        if any(keyword in model_class_name for keyword in ['resnet', 'efficientnet', 'mobilenet', 'densenet', 'vgg']):
            return "vision"
        
        # Check for torchvision models
        if hasattr(model, 'features') or hasattr(model, 'classifier'):
            return "vision"
        
        # Transformer models (BERT, GPT, T5, etc.)
        if any(keyword in model_class_name for keyword in ['bert', 'gpt', 'transformer', 't5', 'roberta', 'electra']):
            return "transformer"
        
        # Check model name for additional context
        if model_name:
            model_name_lower = model_name.lower()
            if any(keyword in model_name_lower for keyword in ['vit', 'vision']):
                return "vision_transformer"
            if any(keyword in model_name_lower for keyword in ['resnet', 'efficientnet']):
                return "vision"
            if any(keyword in model_name_lower for keyword in ['bert', 'gpt', 't5']):
                return "transformer"
        
        # Default fallback based on forward signature
        sig = inspect.signature(model.forward)
        param_names = list(sig.parameters.keys())
        
        if any('input_ids' in param or 'attention_mask' in param for param in param_names):
            return "transformer"
        
        # If only one parameter and it's likely an image tensor
        if len(param_names) == 2 and 'self' in param_names:  # self + one input
            return "vision"
        
        return "generic"
    
    def _generate_transformer_inputs(self, model: nn.Module, model_name: str = None) -> Dict[str, torch.Tensor]:
        """Generate inputs for transformer models (BERT, GPT, T5, etc.)"""
        # Get vocab size from model config if available
        vocab_size = getattr(model.config, 'vocab_size', self.default_vocab_size)
        seq_length = getattr(model.config, 'max_position_embeddings', self.default_seq_length)
        # Use smaller sequence length for testing
        seq_length = min(seq_length, self.default_seq_length)
        
        # Minimal inputs to avoid ONNX export issues
        inputs = {
            'input_ids': torch.randint(
                0, min(vocab_size, 1000), 
                (self.batch_size, seq_length), 
                dtype=torch.long
            ),
            'attention_mask': torch.ones(
                self.batch_size, seq_length, dtype=torch.long
            )
        }
        
        return inputs
    
    def _generate_vision_inputs(self, model: nn.Module, model_name: str = None) -> Dict[str, torch.Tensor]:
        """Generate inputs for vision models (ResNet, EfficientNet, etc.)"""
        # Standard ImageNet input format
        inputs = {
            'x': torch.randn(
                self.batch_size, 3, self.default_image_size, self.default_image_size,
                dtype=torch.float32
            )
        }
        
        # Check if model expects different input name
        sig = inspect.signature(model.forward)
        param_names = [p for p in sig.parameters.keys() if p != 'self']
        
        if param_names:
            first_param = param_names[0]
            if first_param != 'x':
                inputs = {first_param: inputs['x']}
        
        return inputs
    
    def _generate_vision_transformer_inputs(self, model: nn.Module, model_name: str = None) -> Dict[str, torch.Tensor]:
        """Generate inputs for Vision Transformer models"""
        # ViT typically expects pixel_values
        image_size = getattr(model.config, 'image_size', self.default_image_size)
        
        inputs = {
            'pixel_values': torch.randn(
                self.batch_size, 3, image_size, image_size,
                dtype=torch.float32
            )
        }
        
        return inputs
    
    def _generate_generic_inputs(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Fallback: generate generic inputs based on forward signature"""
        sig = inspect.signature(model.forward)
        inputs = {}
        
        param_count = 0
        for param_name in sig.parameters:
            if param_name in ['self', 'args', 'kwargs']:
                continue
                
            param_count += 1
            
            # For generic models, assume first parameter is main input
            if param_count == 1:
                # Try to guess based on parameter name
                if 'image' in param_name.lower() or 'pixel' in param_name.lower():
                    inputs[param_name] = torch.randn(
                        self.batch_size, 3, self.default_image_size, self.default_image_size
                    )
                elif 'input_ids' in param_name.lower():
                    inputs[param_name] = torch.randint(
                        0, 1000, (self.batch_size, self.default_seq_length), dtype=torch.long
                    )
                else:
                    # Default to image-like input
                    inputs[param_name] = torch.randn(
                        self.batch_size, 3, self.default_image_size, self.default_image_size
                    )
        
        if not inputs:
            # Ultimate fallback
            inputs['input'] = torch.randn(
                self.batch_size, 3, self.default_image_size, self.default_image_size
            )
        
        return inputs
    
    def get_test_models(self) -> Dict[str, Dict[str, Any]]:
        """Get test models for different architectures"""
        return {
            'bert': {
                'name': 'google/bert_uncased_L-2_H-128_A-2',
                'type': 'transformer',
                'loader': lambda: AutoModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
            },
            'resnet': {
                'name': 'resnet18',
                'type': 'vision',
                'loader': lambda: models.resnet18(pretrained=False)
            },
            'vit': {
                'name': 'google/vit-base-patch16-224',
                'type': 'vision_transformer',
                'loader': lambda: AutoModel.from_pretrained('google/vit-base-patch16-224')
            }
        }


def test_input_generation():
    """Test input generation for different model types"""
    generator = UniversalInputGenerator()
    test_models = generator.get_test_models()
    
    for model_key, model_info in test_models.items():
        print(f"\n=== Testing {model_key.upper()} ({model_info['type']}) ===")
        try:
            model = model_info['loader']()
            inputs = generator.generate_inputs(model, model_info['name'])
            
            print(f"Model: {model_info['name']}")
            print(f"Generated inputs:")
            for input_name, input_tensor in inputs.items():
                print(f"  {input_name}: {input_tensor.shape} ({input_tensor.dtype})")
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                try:
                    if len(inputs) == 1:
                        output = model(list(inputs.values())[0])
                    else:
                        output = model(**inputs)
                    print(f"Forward pass successful!")
                    if hasattr(output, 'last_hidden_state'):
                        print(f"Output shape: {output.last_hidden_state.shape}")
                    elif isinstance(output, torch.Tensor):
                        print(f"Output shape: {output.shape}")
                    else:
                        print(f"Output type: {type(output)}")
                except Exception as e:
                    print(f"Forward pass failed: {e}")
                    
        except Exception as e:
            print(f"Failed to load/test {model_key}: {e}")


if __name__ == "__main__":
    test_input_generation()