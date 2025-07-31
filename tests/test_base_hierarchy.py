#!/usr/bin/env python3
"""
Unit tests for base hierarchy utilities.
"""

import pytest
import torch.nn as nn

from modelexport.core.base import should_include_in_hierarchy


class TestShouldIncludeInHierarchy:
    """Test the should_include_in_hierarchy function."""
    
    def test_torch_nn_modules_excluded(self):
        """Test that torch.nn modules are excluded by default."""
        linear = nn.Linear(10, 5)
        assert should_include_in_hierarchy(linear) == False
        
        relu = nn.ReLU()
        assert should_include_in_hierarchy(relu) == False
        
        dropout = nn.Dropout(0.1)
        assert should_include_in_hierarchy(dropout) == False
        
        layer_norm = nn.LayerNorm(10)
        assert should_include_in_hierarchy(layer_norm) == False
        
        embedding = nn.Embedding(1000, 100)
        assert should_include_in_hierarchy(embedding) == False
    
    def test_torch_nn_modules_with_exceptions(self):
        """Test that torch.nn modules can be included with exceptions."""
        linear = nn.Linear(10, 5)
        layer_norm = nn.LayerNorm(10)
        
        # Test with exceptions
        assert should_include_in_hierarchy(linear, exceptions=["Linear"]) == True
        assert should_include_in_hierarchy(layer_norm, exceptions=["LayerNorm"]) == True
        assert should_include_in_hierarchy(linear, exceptions=["LayerNorm"]) == False
    
    def test_custom_modules_included(self):
        """Test that custom (non-torch.nn) modules are included."""
        class CustomModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        custom = CustomModule()
        assert should_include_in_hierarchy(custom) == True
    
    def test_transformers_modules_included(self):
        """Test that HuggingFace transformers modules are included."""
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
            
            # Test root model
            assert should_include_in_hierarchy(model) == True
            
            # Test embeddings
            assert should_include_in_hierarchy(model.embeddings) == True
            
            # Test encoder
            assert should_include_in_hierarchy(model.encoder) == True
            
        except ImportError:
            pytest.skip("transformers not available")
    
    def test_custom_torch_nn_subclass(self):
        """Test custom classes that inherit from torch.nn modules."""
        class CustomLinear(nn.Linear):
            def __init__(self, in_features, out_features):
                super().__init__(in_features, out_features)
        
        # Custom torch.nn subclasses should be included (not filtered as infrastructure)
        # because they have custom __module__ paths
        custom_linear = CustomLinear(10, 5)
        assert should_include_in_hierarchy(custom_linear) == True
    
    def test_input_validation(self):
        """Test that the function properly validates inputs."""
        # None input should raise TypeError
        with pytest.raises(TypeError, match="Expected torch.nn.Module"):
            should_include_in_hierarchy(None)
        
        # String input should raise TypeError
        with pytest.raises(TypeError, match="Expected torch.nn.Module"):
            should_include_in_hierarchy("not a module")
        
        # Number input should raise TypeError
        with pytest.raises(TypeError, match="Expected torch.nn.Module"):
            should_include_in_hierarchy(42)
        
        # List input should raise TypeError  
        with pytest.raises(TypeError, match="Expected torch.nn.Module"):
            should_include_in_hierarchy([])
    
    def test_empty_exceptions_list(self):
        """Test behavior with empty exceptions list."""
        linear = nn.Linear(10, 5)
        
        # Empty list should exclude torch.nn modules
        assert should_include_in_hierarchy(linear, exceptions=[]) == False
        
        # None should also exclude torch.nn modules (default behavior)
        assert should_include_in_hierarchy(linear, exceptions=None) == False
    
    def test_must_002_compliance(self):
        """Test MUST-002 compliance - no torch.nn modules by default."""
        forbidden_modules = [
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(10),
            nn.Embedding(1000, 100),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm1d(10),
            nn.MultiheadAttention(64, 8),
            nn.Sequential(nn.Linear(10, 5), nn.ReLU()),
            nn.ModuleList([nn.Linear(10, 5)])
        ]
        
        # All torch.nn modules should be excluded by default
        for module in forbidden_modules:
            assert should_include_in_hierarchy(module) == False, f"{module.__class__.__name__} should be excluded"