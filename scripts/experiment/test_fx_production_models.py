#!/usr/bin/env python3
"""
Iteration 12: Test production-scale models and optimize performance.

This script tests the FX exporter on production-scale models that are within
FX constraints and focuses on performance optimization for supported architectures.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import tempfile
import time
import statistics
from typing import Dict, List, Any, Tuple

# Add modelexport to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modelexport.fx_hierarchy_exporter import FXHierarchyExporter

def test_production_vision_models():
    """Test production-scale vision models that should work well with FX."""
    print("=" * 60)
    print("PRODUCTION TESTING: Vision Models")
    print("=" * 60)
    
    # Production-scale vision models that avoid FX limitations
    production_models = [
        {
            'name': 'ProductionResNet',
            'model': create_production_resnet(),
            'inputs': torch.randn(1, 3, 224, 224),
            'description': 'Production-scale ResNet architecture',
            'expected_coverage': '>90%'
        },
        {
            'name': 'EfficientNet_Block',
            'model': create_efficientnet_block(),
            'inputs': torch.randn(1, 32, 112, 112),
            'description': 'EfficientNet-style block with squeeze-excitation',
            'expected_coverage': '>85%'
        },
        {
            'name': 'MobileNet_Block',
            'model': create_mobilenet_block(),
            'inputs': torch.randn(1, 32, 112, 112),
            'description': 'MobileNet depthwise separable convolutions',
            'expected_coverage': '>80%'
        },
        {
            'name': 'VGG_Production',
            'model': create_vgg_production(),
            'inputs': torch.randn(1, 3, 224, 224),
            'description': 'Production VGG-style architecture',
            'expected_coverage': '>95%'
        }
    ]
    
    results = {}
    performance_data = []
    
    for model_spec in production_models:
        print(f"\nTesting {model_spec['name']}: {model_spec['description']}")
        
        model = model_spec['model']
        inputs = model_spec['inputs']
        
        # Performance testing with multiple runs
        times = []
        coverage_results = []
        
        for run in range(3):  # 3 runs for statistical significance
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
                try:
                    exporter = FXHierarchyExporter(auto_fallback=False)
                    
                    start_time = time.time()
                    result = exporter.export(model, inputs, tmp.name)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    coverage_results.append(result['fx_graph_stats']['coverage_ratio'])
                    
                    if run == 0:  # Store detailed results from first run
                        fx_stats = result['fx_graph_stats']
                        results[model_spec['name']] = {
                            'success': True,
                            'coverage': fx_stats['coverage_ratio'],
                            'coverage_pct': f"{fx_stats['coverage_ratio'] * 100:.1f}%",
                            'total_nodes': fx_stats['total_fx_nodes'],
                            'hierarchy_nodes': fx_stats['hierarchy_nodes'],
                            'node_types': fx_stats['node_type_distribution'],
                            'confidence_dist': fx_stats['confidence_distribution'],
                            'description': model_spec['description'],
                            'expected': model_spec['expected_coverage']
                        }
                    
                    # Cleanup
                    for cleanup_file in [tmp.name, result.get('sidecar_path', ''), result.get('module_info_path', '')]:
                        if cleanup_file and os.path.exists(cleanup_file):
                            os.unlink(cleanup_file)
                            
                except Exception as e:
                    print(f"  ❌ Run {run + 1} failed: {e}")
                    if model_spec['name'] not in results:
                        results[model_spec['name']] = {'success': False, 'error': str(e)}
                    break
        
        # Calculate performance statistics
        if times:
            param_count = sum(p.numel() for p in model.parameters())
            avg_time = statistics.mean(times)
            avg_coverage = statistics.mean(coverage_results)
            time_per_param = avg_time / param_count * 1000000  # microseconds
            
            performance_data.append({
                'name': model_spec['name'],
                'avg_time': avg_time,
                'avg_coverage': avg_coverage,
                'param_count': param_count,
                'time_per_param': time_per_param,
                'efficiency_score': avg_coverage / avg_time,
                'description': model_spec['description']
            })
            
            print(f"  ✅ Performance: {avg_time:.3f}s, {avg_coverage * 100:.1f}% coverage")
            print(f"     {param_count:,} params, {time_per_param:.2f} μs/param")
            print(f"     Efficiency: {avg_coverage / avg_time:.1f} coverage/sec")
    
    return results, performance_data

def test_optimized_attention_models():
    """Test optimized attention models without complex control flow."""
    print("=" * 60)
    print("PRODUCTION TESTING: Optimized Attention Models")
    print("=" * 60)
    
    attention_models = [
        {
            'name': 'MultiLayer_Attention',
            'model': create_multilayer_attention(),
            'inputs': torch.randn(1, 128, 512),
            'description': 'Multi-layer attention without complex control flow',
            'expected_coverage': '>90%'
        },
        {
            'name': 'Cross_Attention',
            'model': create_cross_attention(),
            'inputs': (torch.randn(1, 64, 256), torch.randn(1, 32, 256)),
            'description': 'Cross-attention between different sequences',
            'expected_coverage': '>85%'
        },
        {
            'name': 'Vision_Attention',
            'model': create_vision_attention(),
            'inputs': torch.randn(1, 3, 128, 128),
            'description': 'Vision transformer style patch attention',
            'expected_coverage': '>80%'
        }
    ]
    
    results = {}
    
    for model_spec in attention_models:
        print(f"\nTesting {model_spec['name']}: {model_spec['description']}")
        
        model = model_spec['model']
        inputs = model_spec['inputs']
        
        exporter = FXHierarchyExporter(auto_fallback=False)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            try:
                start_time = time.time()
                result = exporter.export(model, inputs, tmp.name)
                end_time = time.time()
                
                fx_stats = result['fx_graph_stats']
                coverage = fx_stats['coverage_ratio']
                
                results[model_spec['name']] = {
                    'success': True,
                    'coverage': coverage,
                    'coverage_pct': f"{coverage * 100:.1f}%",
                    'export_time': end_time - start_time,
                    'total_nodes': fx_stats['total_fx_nodes'],
                    'hierarchy_nodes': fx_stats['hierarchy_nodes'],
                    'node_types': fx_stats['node_type_distribution'],
                    'confidence_dist': fx_stats['confidence_distribution'],
                    'description': model_spec['description'],
                    'expected': model_spec['expected_coverage']
                }
                
                print(f"  ✅ Coverage: {results[model_spec['name']]['coverage_pct']} ({fx_stats['hierarchy_nodes']}/{fx_stats['total_fx_nodes']} nodes)")
                print(f"     Export time: {end_time - start_time:.3f}s")
                print(f"     Node types: {fx_stats['node_type_distribution']}")
                
                # Cleanup
                for cleanup_file in [tmp.name, result.get('sidecar_path', ''), result.get('module_info_path', '')]:
                    if cleanup_file and os.path.exists(cleanup_file):
                        os.unlink(cleanup_file)
                        
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                results[model_spec['name']] = {'success': False, 'error': str(e), 'description': model_spec['description']}
    
    return results

def test_performance_optimization():
    """Test performance optimization techniques."""
    print("=" * 60)
    print("PERFORMANCE OPTIMIZATION: Efficiency Analysis")
    print("=" * 60)
    
    # Test models of different scales to understand performance characteristics
    scale_models = [
        ("Small_Efficient", create_efficient_small_model(), torch.randn(1, 64)),
        ("Medium_Optimized", create_optimized_medium_model(), torch.randn(1, 3, 128, 128)),
        ("Large_Vision", create_large_vision_model(), torch.randn(1, 3, 224, 224))
    ]
    
    optimization_results = []
    
    for name, model, inputs in scale_models:
        print(f"\nOptimization testing: {name}")
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Model size: {param_count:,} parameters")
        
        # Test with different optimization settings
        configs = [
            ("Standard", {'auto_fallback': False}),
            ("Optimized", {'auto_fallback': False})  # Same config for now, could add optimizations
        ]
        
        config_results = {}
        
        for config_name, config in configs:
            times = []
            coverage_results = []
            
            for run in range(3):
                with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
                    try:
                        exporter = FXHierarchyExporter(**config)
                        
                        start_time = time.time()
                        result = exporter.export(model, inputs, tmp.name)
                        end_time = time.time()
                        
                        times.append(end_time - start_time)
                        coverage_results.append(result['fx_graph_stats']['coverage_ratio'])
                        
                        # Cleanup
                        for cleanup_file in [tmp.name, result.get('sidecar_path', ''), result.get('module_info_path', '')]:
                            if cleanup_file and os.path.exists(cleanup_file):
                                os.unlink(cleanup_file)
                                
                    except Exception as e:
                        print(f"    ❌ {config_name} run {run + 1} failed: {e}")
                        break
            
            if times:
                avg_time = statistics.mean(times)
                avg_coverage = statistics.mean(coverage_results)
                time_per_param = avg_time / param_count * 1000000
                
                config_results[config_name] = {
                    'avg_time': avg_time,
                    'avg_coverage': avg_coverage,
                    'time_per_param': time_per_param,
                    'efficiency': avg_coverage / avg_time
                }
                
                print(f"    {config_name}: {avg_time:.3f}s, {avg_coverage * 100:.1f}% coverage")
                print(f"      Efficiency: {avg_coverage / avg_time:.1f} coverage/sec, {time_per_param:.2f} μs/param")
        
        optimization_results.append({
            'name': name,
            'param_count': param_count,
            'configs': config_results
        })
    
    return optimization_results

def test_fx_limitations_documentation():
    """Document FX limitations with specific examples."""
    print("=" * 60)
    print("FX LIMITATIONS: Documentation and Examples")
    print("=" * 60)
    
    # Test models that specifically trigger known FX limitations
    limitation_tests = [
        {
            'name': 'Dynamic_Tensor_Ops',
            'model': create_dynamic_tensor_model(),
            'inputs': torch.randn(1, 10, 64),
            'expected_error': 'torch.eye with dynamic size',
            'limitation': 'Dynamic tensor operations'
        },
        {
            'name': 'Control_Flow',
            'model': create_control_flow_model(),
            'inputs': torch.randn(1, 128),
            'expected_error': 'control flow',
            'limitation': 'Conditional operations'
        },
        {
            'name': 'Complex_Indexing',
            'model': create_complex_indexing_model(),
            'inputs': torch.randn(1, 100, 256),
            'expected_error': 'indexing',
            'limitation': 'Dynamic indexing operations'
        }
    ]
    
    limitation_results = {}
    
    for test_spec in limitation_tests:
        print(f"\nTesting {test_spec['name']} (Expected: {test_spec['limitation']})")
        
        model = test_spec['model']
        inputs = test_spec['inputs']
        
        exporter = FXHierarchyExporter(auto_fallback=False)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            try:
                result = exporter.export(model, inputs, tmp.name)
                
                print(f"  ⚠️ Unexpected success: {test_spec['limitation']} should have failed")
                limitation_results[test_spec['name']] = {
                    'expected_failure': True,
                    'actual_result': 'success',
                    'limitation': test_spec['limitation']
                }
                
                # Cleanup if successful
                for cleanup_file in [tmp.name, result.get('sidecar_path', ''), result.get('module_info_path', '')]:
                    if cleanup_file and os.path.exists(cleanup_file):
                        os.unlink(cleanup_file)
                        
            except Exception as e:
                error_str = str(e).lower()
                expected_in_error = any(keyword in error_str for keyword in ['eye', 'control flow', 'indexing', 'proxy'])
                
                print(f"  ✅ Expected failure: {e}")
                limitation_results[test_spec['name']] = {
                    'expected_failure': True,
                    'actual_result': 'failed_as_expected',
                    'error': str(e),
                    'limitation': test_spec['limitation'],
                    'error_matches_expectation': expected_in_error
                }
    
    return limitation_results

# Model creation functions for production testing

def create_production_resnet():
    """Create a production-scale ResNet."""
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.downsample = downsample
            
        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
                
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += identity
            return F.relu(out)
    
    class ProductionResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            
            # Layer 1
            self.layer1_0 = ResidualBlock(64, 64)
            self.layer1_1 = ResidualBlock(64, 64)
            
            # Layer 2 with downsampling
            downsample2 = nn.Sequential(
                nn.Conv2d(64, 128, 1, stride=2, bias=False),
                nn.BatchNorm2d(128)
            )
            self.layer2_0 = ResidualBlock(64, 128, stride=2, downsample=downsample2)
            self.layer2_1 = ResidualBlock(128, 128)
            
            # Layer 3 with downsampling
            downsample3 = nn.Sequential(
                nn.Conv2d(128, 256, 1, stride=2, bias=False),
                nn.BatchNorm2d(256)
            )
            self.layer3_0 = ResidualBlock(128, 256, stride=2, downsample=downsample3)
            self.layer3_1 = ResidualBlock(256, 256)
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(256, 1000)
            
        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            
            x = self.layer1_0(x)
            x = self.layer1_1(x)
            x = self.layer2_0(x)
            x = self.layer2_1(x)
            x = self.layer3_0(x)
            x = self.layer3_1(x)
            
            x = self.avgpool(x)
            x = x.flatten(1)
            return self.fc(x)
    
    return ProductionResNet()

def create_efficientnet_block():
    """Create EfficientNet-style block with squeeze-excitation."""
    class SqueezeExcitation(nn.Module):
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.squeeze = nn.AdaptiveAvgPool2d(1)
            self.excitation = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias=False),
                nn.ReLU(),
                nn.Linear(channels // reduction, channels, bias=False),
                nn.Sigmoid()
            )
            
        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.squeeze(x).view(b, c)
            y = self.excitation(y).view(b, c, 1, 1)
            return x * y.expand_as(x)
    
    class EfficientNetBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.expand_conv = nn.Conv2d(32, 192, 1, bias=False)
            self.expand_bn = nn.BatchNorm2d(192)
            
            self.depthwise_conv = nn.Conv2d(192, 192, 3, padding=1, groups=192, bias=False)
            self.depthwise_bn = nn.BatchNorm2d(192)
            
            self.se = SqueezeExcitation(192)
            
            self.project_conv = nn.Conv2d(192, 32, 1, bias=False)
            self.project_bn = nn.BatchNorm2d(32)
            
        def forward(self, x):
            identity = x
            
            # Expand
            out = F.relu6(self.expand_bn(self.expand_conv(x)))
            
            # Depthwise
            out = F.relu6(self.depthwise_bn(self.depthwise_conv(out)))
            
            # Squeeze-excitation
            out = self.se(out)
            
            # Project
            out = self.project_bn(self.project_conv(out))
            
            # Skip connection
            return out + identity
    
    return EfficientNetBlock()

def create_mobilenet_block():
    """Create MobileNet depthwise separable convolution block."""
    class MobileNetBlock(nn.Module):
        def __init__(self):
            super().__init__()
            # Depthwise convolution
            self.depthwise = nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False)
            self.depthwise_bn = nn.BatchNorm2d(32)
            
            # Pointwise convolution
            self.pointwise = nn.Conv2d(32, 64, 1, bias=False)
            self.pointwise_bn = nn.BatchNorm2d(64)
            
            # Second depthwise
            self.depthwise2 = nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False)
            self.depthwise2_bn = nn.BatchNorm2d(64)
            
            # Second pointwise
            self.pointwise2 = nn.Conv2d(64, 32, 1, bias=False)
            self.pointwise2_bn = nn.BatchNorm2d(32)
            
        def forward(self, x):
            # First depthwise separable block
            out = F.relu(self.depthwise_bn(self.depthwise(x)))
            out = F.relu(self.pointwise_bn(self.pointwise(out)))
            
            # Second depthwise separable block
            out = F.relu(self.depthwise2_bn(self.depthwise2(out)))
            out = self.pointwise2_bn(self.pointwise2(out))
            
            return out + x  # Residual connection
    
    return MobileNetBlock()

def create_vgg_production():
    """Create production VGG-style architecture."""
    class VGGProduction(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                
                # Block 2
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                
                # Block 3
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                
                # Block 4
                nn.Conv2d(256, 512, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
            )
            
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 1000)
            )
            
        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)
    
    return VGGProduction()

def create_multilayer_attention():
    """Create multi-layer attention model."""
    class MultiLayerAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(512, 512)
            self.attention1 = nn.MultiheadAttention(512, 8, batch_first=True)
            self.norm1 = nn.LayerNorm(512)
            self.attention2 = nn.MultiheadAttention(512, 8, batch_first=True)
            self.norm2 = nn.LayerNorm(512)
            self.attention3 = nn.MultiheadAttention(512, 8, batch_first=True)
            self.norm3 = nn.LayerNorm(512)
            
            self.feed_forward = nn.Sequential(
                nn.Linear(512, 2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, 512)
            )
            self.norm4 = nn.LayerNorm(512)
            self.classifier = nn.Linear(512, 10)
            
        def forward(self, x):
            x = self.embedding(x)
            
            # First attention layer
            attn1, _ = self.attention1(x, x, x)
            x = self.norm1(x + attn1)
            
            # Second attention layer
            attn2, _ = self.attention2(x, x, x)
            x = self.norm2(x + attn2)
            
            # Third attention layer
            attn3, _ = self.attention3(x, x, x)
            x = self.norm3(x + attn3)
            
            # Feed-forward
            ff_out = self.feed_forward(x)
            x = self.norm4(x + ff_out)
            
            # Global pooling and classification
            x = x.mean(dim=1)
            return self.classifier(x)
    
    return MultiLayerAttention()

def create_cross_attention():
    """Create cross-attention model."""
    class CrossAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.query_proj = nn.Linear(256, 256)
            self.key_proj = nn.Linear(256, 256)
            self.value_proj = nn.Linear(256, 256)
            
            self.cross_attention = nn.MultiheadAttention(256, 8, batch_first=True)
            self.norm1 = nn.LayerNorm(256)
            self.norm2 = nn.LayerNorm(256)
            
            self.feed_forward = nn.Sequential(
                nn.Linear(256, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256)
            )
            
            self.classifier = nn.Linear(256, 10)
            
        def forward(self, inputs):
            query_seq, key_value_seq = inputs
            
            # Project inputs
            query = self.query_proj(query_seq)
            key = self.key_proj(key_value_seq)
            value = self.value_proj(key_value_seq)
            
            # Cross-attention
            attn_out, _ = self.cross_attention(query, key, value)
            query = self.norm1(query + attn_out)
            
            # Feed-forward
            ff_out = self.feed_forward(query)
            query = self.norm2(query + ff_out)
            
            # Global pooling and classification
            pooled = query.mean(dim=1)
            return self.classifier(pooled)
    
    return CrossAttention()

def create_vision_attention():
    """Create vision attention model (simplified ViT-style)."""
    class VisionAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, 384, kernel_size=16, stride=16)
            self.pos_embed = nn.Parameter(torch.randn(1, 65, 384))  # 64 patches + 1 cls token
            self.cls_token = nn.Parameter(torch.randn(1, 1, 384))
            
            self.attention = nn.MultiheadAttention(384, 6, batch_first=True)
            self.norm1 = nn.LayerNorm(384)
            self.norm2 = nn.LayerNorm(384)
            
            self.mlp = nn.Sequential(
                nn.Linear(384, 1536),
                nn.GELU(),
                nn.Linear(1536, 384)
            )
            
            self.classifier = nn.Linear(384, 1000)
            
        def forward(self, x):
            B = x.shape[0]
            
            # Patch embedding
            x = self.patch_embed(x).flatten(2).transpose(1, 2)  # B, N, C
            
            # Add class token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            # Add position embedding
            x = x + self.pos_embed
            
            # Attention block
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)
            
            # MLP block
            mlp_out = self.mlp(x)
            x = self.norm2(x + mlp_out)
            
            # Classification using cls token
            cls_token_final = x[:, 0]
            return self.classifier(cls_token_final)
    
    return VisionAttention()

# Optimization test models
def create_efficient_small_model():
    """Create a small efficient model for performance testing."""
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

def create_optimized_medium_model():
    """Create a medium-sized optimized model."""
    class OptimizedMedium(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_block = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 10)
            )
            
        def forward(self, x):
            x = self.conv_block(x)
            return self.classifier(x)
    
    return OptimizedMedium()

def create_large_vision_model():
    """Create a large vision model for performance testing."""
    class LargeVision(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                # Block 1
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                
                # Block 2
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Block 3
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Block 4
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
            
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 1000)
            )
            
        def forward(self, x):
            x = self.backbone(x)
            return self.classifier(x)
    
    return LargeVision()

# Limitation test models
def create_dynamic_tensor_model():
    """Create model with dynamic tensor operations (known FX limitation)."""
    class DynamicTensorModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 32)
            
        def forward(self, x):
            batch_size, seq_len, features = x.shape
            # This will fail in FX because seq_len is a Proxy, not an int
            identity_matrix = torch.eye(seq_len)
            return self.linear(x) @ identity_matrix
    
    return DynamicTensorModel()

def create_control_flow_model():
    """Create model with control flow (known FX limitation)."""
    class ControlFlowModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, 64)
            
        def forward(self, x):
            x = self.linear(x)
            # Control flow based on tensor values
            if x.sum() > 0:
                return F.relu(x)
            else:
                return F.tanh(x)
    
    return ControlFlowModel()

def create_complex_indexing_model():
    """Create model with complex indexing (potential FX limitation)."""
    class ComplexIndexingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(256, 128)
            
        def forward(self, x):
            batch_size, seq_len, features = x.shape
            x = self.linear(x)
            
            # Complex indexing that might fail in FX
            indices = torch.randperm(seq_len)[:seq_len//2]
            selected = x[:, indices, :]
            
            return selected.mean(dim=1)
    
    return ComplexIndexingModel()

def main():
    """Run Iteration 12 production model tests."""
    print("🚀 Iteration 12: Production Model Testing and Performance Optimization")
    print(f"PyTorch version: {torch.__version__}")
    
    all_results = {}
    
    # Test 1: Production vision models
    print(f"\n📋 Running Production Vision Model Tests")
    try:
        vision_results, vision_performance = test_production_vision_models()
        all_results['production_vision'] = vision_results
        all_results['vision_performance'] = vision_performance
    except Exception as e:
        print(f"❌ Production vision tests crashed: {e}")
        all_results['production_vision'] = {}
        all_results['vision_performance'] = []
    
    # Test 2: Optimized attention models
    print(f"\n📋 Running Optimized Attention Model Tests")
    try:
        attention_results = test_optimized_attention_models()
        all_results['optimized_attention'] = attention_results
    except Exception as e:
        print(f"❌ Optimized attention tests crashed: {e}")
        all_results['optimized_attention'] = {}
    
    # Test 3: Performance optimization
    print(f"\n📋 Running Performance Optimization Tests")
    try:
        optimization_results = test_performance_optimization()
        all_results['performance_optimization'] = optimization_results
    except Exception as e:
        print(f"❌ Performance optimization tests crashed: {e}")
        all_results['performance_optimization'] = []
    
    # Test 4: FX limitations documentation
    print(f"\n📋 Running FX Limitations Documentation Tests")
    try:
        limitation_results = test_fx_limitations_documentation()
        all_results['fx_limitations'] = limitation_results
    except Exception as e:
        print(f"❌ FX limitations tests crashed: {e}")
        all_results['fx_limitations'] = {}
    
    # Analysis and summary
    print("\n" + "=" * 60)
    print("ITERATION 12 ANALYSIS AND RESULTS")
    print("=" * 60)
    
    # Production vision analysis
    if all_results['production_vision']:
        vision_data = all_results['production_vision']
        successful_models = {k: v for k, v in vision_data.items() if v.get('success', False)}
        
        print("🏭 Production Vision Model Results:")
        for model_name, data in successful_models.items():
            coverage_pct = data['coverage_pct']
            description = data['description']
            print(f"   {model_name}: {coverage_pct} - {description}")
        
        if successful_models:
            avg_coverage = sum(data['coverage'] for data in successful_models.values()) / len(successful_models)
            print(f"   📊 Average production coverage: {avg_coverage * 100:.1f}%")
    
    # Performance analysis
    if all_results['vision_performance']:
        perf_data = all_results['vision_performance']
        print(f"\n⚡ Production Model Performance:")
        
        # Sort by efficiency
        sorted_perf = sorted(perf_data, key=lambda x: x['efficiency_score'], reverse=True)
        
        for result in sorted_perf:
            print(f"   {result['name']}: {result['avg_time']:.3f}s, {result['avg_coverage'] * 100:.1f}% coverage")
            print(f"     Efficiency: {result['efficiency_score']:.1f} coverage/sec, {result['param_count']:,} params")
    
    # Attention model analysis
    if all_results['optimized_attention']:
        attention_data = all_results['optimized_attention']
        successful_attention = {k: v for k, v in attention_data.items() if v.get('success', False)}
        
        print(f"\n🎯 Optimized Attention Model Results:")
        for model_name, data in successful_attention.items():
            coverage_pct = data['coverage_pct']
            description = data['description']
            print(f"   {model_name}: {coverage_pct} - {description}")
    
    # FX limitations analysis
    if all_results['fx_limitations']:
        limitation_data = all_results['fx_limitations']
        
        print(f"\n⚠️ FX Limitations Documentation:")
        for test_name, data in limitation_data.items():
            limitation = data['limitation']
            result = data['actual_result']
            print(f"   {test_name}: {limitation} - {result}")
            if data.get('error_matches_expectation'):
                print(f"     ✅ Error matches expected FX limitation")
    
    # Overall assessment
    total_production_models = len(all_results.get('production_vision', {}))
    successful_production = len([v for v in all_results.get('production_vision', {}).values() if v.get('success', False)])
    
    total_attention_models = len(all_results.get('optimized_attention', {}))
    successful_attention = len([v for v in all_results.get('optimized_attention', {}).values() if v.get('success', False)])
    
    print(f"\n💡 Iteration 12 Key Achievements:")
    print(f"   🏭 Production Vision Success: {successful_production}/{total_production_models} models")
    print(f"   🎯 Optimized Attention Success: {successful_attention}/{total_attention_models} models")
    print(f"   ⚡ Performance characteristics documented across model scales")
    print(f"   📚 FX limitations systematically documented with examples")
    
    if successful_production > 0 and all_results.get('production_vision'):
        production_avg = sum(v['coverage'] for v in all_results['production_vision'].values() 
                           if v.get('success', False)) / successful_production
        print(f"   📈 Production model average coverage: {production_avg * 100:.1f}%")
    
    print(f"\n🎉 Iteration 12 completed!")
    print(f"   ➡️  Next: Continue with specialized optimizations and approach HuggingFace compatibility")
    
    return 0

if __name__ == '__main__':
    exit(main())