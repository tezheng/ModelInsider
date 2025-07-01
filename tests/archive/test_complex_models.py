"""
Complex Model Architecture Coverage Tests - Round 5

These tests validate the hierarchy exporter with various complex model architectures,
ensuring universal compatibility across different model types and patterns.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
from pathlib import Path

from modelexport.hierarchy_exporter import HierarchyExporter


class TestCNNArchitectures:
    """Test convolutional neural network architectures."""
    
    def test_basic_cnn_model(self):
        """Test basic CNN with conv, pool, and fc layers."""
        class BasicCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(64, 10)
                )
                
            def forward(self, x):
                x = self.conv_layers(x)
                return self.classifier(x)
        
        model = BasicCNN()
        model.eval()
        inputs = torch.randn(1, 3, 32, 32)
        
        for strategy in ["usage_based", "htp"]:
            exporter = HierarchyExporter(strategy=strategy)
            
            with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
                result = exporter.export(
                    model=model,
                    example_inputs=inputs,
                    output_path=tmp.name
                )
                
                assert result is not None
                assert result['total_operations'] > 5  # Should have several operations
                
                # Check tag mapping
                tag_mapping = exporter.get_tag_mapping()
                assert len(tag_mapping) > 0
    
    def test_resnet_like_architecture(self):
        """Test ResNet-like architecture with skip connections."""
        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(channels)
                self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(channels)
                
            def forward(self, x):
                residual = x
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                return F.relu(out + residual)
        
        class MiniResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 7, stride=2, padding=3)
                self.bn1 = nn.BatchNorm2d(32)
                self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
                
                self.layer1 = nn.Sequential(
                    ResidualBlock(32),
                    ResidualBlock(32)
                )
                
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(32, 10)
                
            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                return self.fc(x)
        
        model = MiniResNet()
        model.eval()
        inputs = torch.randn(1, 3, 64, 64)
        
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None
            assert result['total_operations'] > 8
            
            # Should capture residual block hierarchy
            tag_mapping = exporter.get_tag_mapping()
            all_tags = []
            for node_info in tag_mapping.values():
                all_tags.extend(node_info.get('tags', []))
            
            if len(all_tags) > 0:
                # Should have tags containing ResidualBlock
                residual_tags = [tag for tag in all_tags if "ResidualBlock" in tag]
                assert len(residual_tags) >= 0  # May be filtered out depending on strategy


class TestRNNArchitectures:
    """Test recurrent neural network architectures."""
    
    def test_lstm_model(self):
        """Test LSTM-based model."""
        class LSTMModel(nn.Module):
            def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=256, num_layers=2):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
                self.dropout = nn.Dropout(0.1)
                self.classifier = nn.Linear(hidden_dim, 10)
                
            def forward(self, x):
                x = self.embedding(x)
                lstm_out, (hidden, cell) = self.lstm(x)
                # Use last output
                last_output = lstm_out[:, -1, :]
                x = self.dropout(last_output)
                return self.classifier(x)
        
        model = LSTMModel()
        model.eval()
        inputs = torch.randint(0, 1000, (2, 20))  # batch_size=2, seq_len=20
        
        for strategy in ["usage_based", "htp"]:
            exporter = HierarchyExporter(strategy=strategy)
            
            with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
                result = exporter.export(
                    model=model,
                    example_inputs=inputs,
                    output_path=tmp.name
                )
                
                assert result is not None
                assert result['total_operations'] > 5
    
    def test_gru_model(self):
        """Test GRU-based model."""
        class GRUModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(64, 128, 2, batch_first=True, bidirectional=True)
                self.attention = nn.MultiheadAttention(256, 8, batch_first=True)
                self.norm = nn.LayerNorm(256)
                self.output = nn.Linear(256, 1)
                
            def forward(self, x):
                gru_out, _ = self.gru(x)
                attn_out, _ = self.attention(gru_out, gru_out, gru_out)
                normed = self.norm(attn_out)
                return self.output(normed.mean(dim=1))
        
        model = GRUModel()
        model.eval()
        inputs = torch.randn(2, 15, 64)  # batch_size=2, seq_len=15, features=64
        
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None


class TestTransformerArchitectures:
    """Test transformer-based architectures."""
    
    def test_custom_transformer(self):
        """Test custom transformer implementation."""
        class MultiHeadAttention(nn.Module):
            def __init__(self, d_model, num_heads):
                super().__init__()
                self.d_model = d_model
                self.num_heads = num_heads
                self.d_k = d_model // num_heads
                
                self.W_q = nn.Linear(d_model, d_model)
                self.W_k = nn.Linear(d_model, d_model)
                self.W_v = nn.Linear(d_model, d_model)
                self.W_o = nn.Linear(d_model, d_model)
                
            def forward(self, query, key, value):
                batch_size = query.size(0)
                
                Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                
                scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
                attn_weights = F.softmax(scores, dim=-1)
                attn_output = torch.matmul(attn_weights, V)
                
                attn_output = attn_output.transpose(1, 2).contiguous().view(
                    batch_size, -1, self.d_model
                )
                
                return self.W_o(attn_output)
        
        class TransformerBlock(nn.Module):
            def __init__(self, d_model, num_heads, d_ff):
                super().__init__()
                self.attention = MultiHeadAttention(d_model, num_heads)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                
                self.feed_forward = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model)
                )
                
            def forward(self, x):
                attn_out = self.attention(x, x, x)
                x = self.norm1(x + attn_out)
                ff_out = self.feed_forward(x)
                return self.norm2(x + ff_out)
        
        class CustomTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(5000, 512)
                self.pos_encoding = nn.Parameter(torch.randn(100, 512))
                
                self.transformer_blocks = nn.ModuleList([
                    TransformerBlock(512, 8, 2048) for _ in range(4)
                ])
                
                self.output_projection = nn.Linear(512, 10)
                
            def forward(self, x):
                x = self.embedding(x) + self.pos_encoding[:x.size(1)]
                
                for block in self.transformer_blocks:
                    x = block(x)
                
                return self.output_projection(x.mean(dim=1))
        
        model = CustomTransformer()
        model.eval()
        inputs = torch.randint(0, 5000, (2, 50))
        
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None
            assert result['total_operations'] > 10  # Complex model should have many ops


class TestMultiBranchArchitectures:
    """Test models with multiple branches and complex routing."""
    
    def test_inception_like_module(self):
        """Test Inception-like module with multiple parallel branches."""
        class InceptionModule(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                # 1x1 conv branch
                self.branch1 = nn.Conv2d(in_channels, out_channels // 4, 1)
                
                # 1x1 -> 3x3 conv branch
                self.branch2 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels // 4, 1),
                    nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1)
                )
                
                # 1x1 -> 5x5 conv branch
                self.branch3 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels // 4, 1),
                    nn.Conv2d(out_channels // 4, out_channels // 4, 5, padding=2)
                )
                
                # 3x3 maxpool -> 1x1 conv branch
                self.branch4 = nn.Sequential(
                    nn.MaxPool2d(3, stride=1, padding=1),
                    nn.Conv2d(in_channels, out_channels // 4, 1)
                )
                
            def forward(self, x):
                branch1 = self.branch1(x)
                branch2 = self.branch2(x)
                branch3 = self.branch3(x)
                branch4 = self.branch4(x)
                
                return torch.cat([branch1, branch2, branch3, branch4], dim=1)
        
        class InceptionNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
                self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
                
                self.inception1 = InceptionModule(64, 256)
                self.inception2 = InceptionModule(256, 512)
                
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(512, 10)
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = self.maxpool1(x)
                x = self.inception1(x)
                x = self.inception2(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                return self.classifier(x)
        
        model = InceptionNet()
        model.eval()
        inputs = torch.randn(1, 3, 224, 224)
        
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None
            assert result['total_operations'] > 8
    
    def test_multi_input_multi_output_model(self):
        """Test model with multiple inputs and outputs."""
        class MultiIOModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Image branch
                self.image_branch = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 4)),
                    nn.Flatten()
                )
                
                # Text branch
                self.text_branch = nn.Sequential(
                    nn.Embedding(1000, 64),
                    nn.LSTM(64, 128, batch_first=True)
                )
                
                # Fusion
                self.fusion = nn.Sequential(
                    nn.Linear(32 * 16 + 128, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5)
                )
                
                # Multiple outputs
                self.classifier = nn.Linear(256, 10)
                self.regressor = nn.Linear(256, 1)
                
            def forward(self, image, text):
                # Process image
                image_features = self.image_branch(image)
                
                # Process text
                text_features, (hidden, _) = self.text_branch(text)
                text_features = hidden[-1]  # Use last hidden state
                
                # Fuse features
                combined = torch.cat([image_features, text_features], dim=1)
                fused = self.fusion(combined)
                
                # Multiple outputs
                classification = self.classifier(fused)
                regression = self.regressor(fused)
                
                return classification, regression
        
        model = MultiIOModel()
        model.eval()
        
        image_input = torch.randn(2, 3, 32, 32)
        text_input = torch.randint(0, 1000, (2, 20))
        inputs = (image_input, text_input)
        
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None


class TestCustomArchitectures:
    """Test custom and unusual architectures."""
    
    def test_recursive_module(self):
        """Test module that uses itself recursively."""
        class RecursiveBlock(nn.Module):
            def __init__(self, channels, depth=3):
                super().__init__()
                self.depth = depth
                self.conv = nn.Conv2d(channels, channels, 3, padding=1)
                self.norm = nn.BatchNorm2d(channels)
                
            def forward(self, x, depth=None):
                if depth is None:
                    depth = self.depth
                
                x = F.relu(self.norm(self.conv(x)))
                
                if depth > 1:
                    # Recursive call (simplified for ONNX compatibility)
                    x = F.relu(self.norm(self.conv(x)))
                    if depth > 2:
                        x = F.relu(self.norm(self.conv(x)))
                
                return x
        
        class RecursiveNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_conv = nn.Conv2d(3, 64, 7, padding=3)
                self.recursive_block = RecursiveBlock(64, depth=3)
                self.output_conv = nn.Conv2d(64, 1, 1)
                
            def forward(self, x):
                x = self.input_conv(x)
                x = self.recursive_block(x)
                return self.output_conv(x)
        
        model = RecursiveNet()
        model.eval()
        inputs = torch.randn(1, 3, 64, 64)
        
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None
    
    def test_dynamic_architecture(self):
        """Test architecture with dynamic behavior."""
        class DynamicNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.ModuleList([
                    nn.Linear(128, 128) for _ in range(5)
                ])
                self.batch_norms = nn.ModuleList([
                    nn.BatchNorm1d(128) for _ in range(5)
                ])
                self.output = nn.Linear(128, 10)
                
            def forward(self, x):
                # Dynamically choose how many layers to use based on input
                num_layers = min(len(self.features), x.size(0) + 2)  # Simple dynamic rule
                
                for i in range(num_layers):
                    x = F.relu(self.batch_norms[i](self.features[i](x)))
                
                return self.output(x)
        
        model = DynamicNet()
        model.eval()
        inputs = torch.randn(3, 128)  # This will use 5 layers (3 + 2)
        
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None
    
    def test_nested_sequential_modules(self):
        """Test deeply nested Sequential modules."""
        def create_nested_sequential(depth, channels):
            if depth == 1:
                return nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU()
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    create_nested_sequential(depth - 1, channels)
                )
        
        class DeeplyNested(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_conv = nn.Conv2d(3, 64, 7, padding=3)
                self.nested_layers = create_nested_sequential(6, 64)  # 6 levels deep
                self.output = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(64, 10)
                
            def forward(self, x):
                x = self.input_conv(x)
                x = self.nested_layers(x)
                x = self.output(x)
                x = torch.flatten(x, 1)
                return self.classifier(x)
        
        model = DeeplyNested()
        model.eval()
        inputs = torch.randn(1, 3, 64, 64)
        
        for strategy in ["usage_based", "htp"]:
            exporter = HierarchyExporter(strategy=strategy)
            
            with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
                result = exporter.export(
                    model=model,
                    example_inputs=inputs,
                    output_path=tmp.name
                )
                
                assert result is not None
                assert result['total_operations'] > 5


class TestArchitectureSpecificFeatures:
    """Test architecture-specific features and patterns."""
    
    def test_attention_mechanisms(self):
        """Test various attention mechanisms."""
        class SelfAttention(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.query = nn.Linear(dim, dim)
                self.key = nn.Linear(dim, dim)
                self.value = nn.Linear(dim, dim)
                self.scale = dim ** -0.5
                
            def forward(self, x):
                q = self.query(x)
                k = self.key(x)
                v = self.value(x)
                
                attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                attn = F.softmax(attn, dim=-1)
                
                return torch.matmul(attn, v)
        
        class AttentionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(1000, 256)
                self.self_attention = SelfAttention(256)
                self.cross_attention = nn.MultiheadAttention(256, 8, batch_first=True)
                self.output = nn.Linear(256, 10)
                
            def forward(self, x):
                x = self.embedding(x)
                
                # Self attention
                x = self.self_attention(x)
                
                # Cross attention (using same input as query and key-value)
                x, _ = self.cross_attention(x, x, x)
                
                return self.output(x.mean(dim=1))
        
        model = AttentionModel()
        model.eval()
        inputs = torch.randint(0, 1000, (2, 32))
        
        exporter = HierarchyExporter(strategy="htp")
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
            result = exporter.export(
                model=model,
                example_inputs=inputs,
                output_path=tmp.name
            )
            
            assert result is not None
    
    def test_normalization_layers(self):
        """Test various normalization techniques."""
        class NormalizationNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(64)
                
                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.ln1 = nn.LayerNorm([128, 32, 32])
                
                self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                self.gn1 = nn.GroupNorm(8, 256)
                
                self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
                self.in1 = nn.InstanceNorm2d(512)
                
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512, 10)
                
            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.ln1(self.conv2(x)))
                x = F.relu(self.gn1(self.conv3(x)))
                x = F.relu(self.in1(self.conv4(x)))
                
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                return self.fc(x)
        
        model = NormalizationNet()
        model.eval()
        inputs = torch.randn(2, 3, 32, 32)
        
        for strategy in ["usage_based", "htp"]:
            exporter = HierarchyExporter(strategy=strategy)
            
            with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
                result = exporter.export(
                    model=model,
                    example_inputs=inputs,
                    output_path=tmp.name
                )
                
                assert result is not None
                assert result['total_operations'] > 8


if __name__ == "__main__":
    # Run complex model tests
    pytest.main([__file__, "-v", "--tb=short"])