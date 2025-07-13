"""
Test Model Fixtures

Provides standard test models for consistent testing across all strategies.
"""

from typing import Any

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple CNN for vision tasks - excellent for FX tracing."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ComplexMLP(nn.Module):
    """Multi-layer perceptron with dropout and normalization."""
    
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        return self.layers(x)


class AttentionModel(nn.Module):
    """Simple attention model to test attention mechanism handling."""
    
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(1000, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(100, d_model))
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 10)
        
    def forward(self, x):
        # x is token indices
        x = self.embedding(x)  # [batch, seq, d_model]
        seq_len = x.size(1)
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(attn_output + x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)


class ConditionalModel(nn.Module):
    """Model with conditional execution to test control flow limitations."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 15)
        self.fc3 = nn.Linear(20, 10)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        
        # Control flow - problematic for FX tracing
        if x.sum() > 0:
            return self.fc2(x)
        else:
            return self.fc3(x)


class TestModelFixtures:
    """Central fixture provider for test models."""
    
    @staticmethod
    def get_simple_cnn() -> tuple[nn.Module, torch.Tensor]:
        """Get SimpleCNN model with appropriate input."""
        model = SimpleCNN()
        model.eval()
        input_tensor = torch.randn(1, 3, 32, 32)
        return model, input_tensor
    
    @staticmethod
    def get_complex_mlp() -> tuple[nn.Module, torch.Tensor]:
        """Get ComplexMLP model with appropriate input."""
        model = ComplexMLP()
        model.eval()
        input_tensor = torch.randn(1, 784)
        return model, input_tensor
    
    @staticmethod
    def get_attention_model() -> tuple[nn.Module, torch.Tensor]:
        """Get AttentionModel with appropriate input."""
        model = AttentionModel()
        model.eval()
        input_tensor = torch.randint(0, 1000, (1, 20))  # batch=1, seq_len=20
        return model, input_tensor
    
    @staticmethod
    def get_conditional_model() -> tuple[nn.Module, torch.Tensor]:
        """Get ConditionalModel with appropriate input (FX incompatible)."""
        model = ConditionalModel()
        model.eval()
        input_tensor = torch.randn(1, 10)
        return model, input_tensor
    
    @staticmethod
    def get_all_models() -> dict[str, tuple[nn.Module, torch.Tensor]]:
        """Get all test models as a dictionary."""
        return {
            'simple_cnn': TestModelFixtures.get_simple_cnn(),
            'complex_mlp': TestModelFixtures.get_complex_mlp(),
            'attention_model': TestModelFixtures.get_attention_model(),
            'conditional_model': TestModelFixtures.get_conditional_model()
        }
    
    @staticmethod
    @staticmethod
    def get_model_metadata() -> dict[str, dict[str, Any]]:
        """Get metadata about test models."""
        return {
            'simple_cnn': {
                'name': 'SimpleCNN',
                'type': 'vision',
                'expected_coverage': 0.95,
                'architecture': 'cnn'
            },
            'complex_mlp': {
                'name': 'ComplexMLP',
                'type': 'feedforward',
                'expected_coverage': 0.90,
                'architecture': 'mlp'
            },
            'attention_model': {
                'name': 'AttentionModel',
                'type': 'attention',
                'expected_coverage': 0.85,
                'architecture': 'transformer'
            },
            'conditional_model': {
                'name': 'ConditionalModel',
                'type': 'conditional',
                'expected_coverage': None,
                'architecture': 'control_flow'
            }
        }