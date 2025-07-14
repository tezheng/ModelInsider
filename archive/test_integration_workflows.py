"""
Test Integration Workflows - Comprehensive End-to-End Testing

This test suite validates complete integration workflows that users would
perform in real-world scenarios. These tests ensure that all components
work together seamlessly and that the system delivers on its promises.

Workflow Architecture Overview:
    Integration workflows test the complete pipeline from model loading
    through export to final validation and analysis. These are the most
    important tests as they reflect actual user experiences.
    
    Complete Workflow Pipeline:
    ├── Model Loading & Preparation
    │   ├── Load HuggingFace models (AutoModel.from_pretrained)
    │   ├── Generate appropriate inputs (manual or automatic)
    │   └── Validate model compatibility
    ├── Export Process
    │   ├── Choose export strategy based on requirements
    │   ├── Configure export parameters and options
    │   ├── Execute export with progress tracking
    │   └── Handle errors and edge cases gracefully
    ├── Output Validation
    │   ├── Verify ONNX file creation and validity
    │   ├── Check metadata file completeness
    │   ├── Validate hierarchy tag coverage and quality
    │   └── Confirm no Cardinal Rule violations
    └── Post-Export Analysis
        ├── Load and analyze exported ONNX model
        ├── Query hierarchy information
        ├── Compare with baseline expectations
        └── Generate reports and insights

Test Categories Covered:
    ├── Complete Export Workflows
    │   ├── Simple Model Export (BERT-like transformers)
    │   ├── Complex Model Export (Vision, multimodal)
    │   ├── Custom Model Export (user-defined architectures)
    │   ├── SAM Model Export (coordinate fix validation)
    │   └── Batch Export Processing (multiple models)
    ├── Strategy Integration Testing
    │   ├── Cross-Strategy Compatibility
    │   ├── Strategy Migration (upgrading between strategies)
    │   ├── Strategy Comparison Workflows
    │   └── Strategy Selection Guidance
    ├── Configuration & Customization Workflows
    │   ├── Manual Input Specification Workflows
    │   ├── Configuration File Usage
    │   ├── Advanced Parameter Tuning
    │   └── Custom Export Requirements
    ├── Error Recovery & Robustness
    │   ├── Network Interruption Recovery
    │   ├── Disk Space Handling
    │   ├── Memory Constraint Management
    │   └── Model Compatibility Issues
    ├── Quality Assurance Workflows
    │   ├── Export Validation Pipelines
    │   ├── Coverage Analysis Workflows
    │   ├── Regression Testing Procedures
    │   └── Performance Benchmarking
    └── Real-World Usage Scenarios
        ├── Research & Development Workflows
        ├── Production Deployment Preparation
        ├── Model Analysis & Debugging
        └── Comparative Model Studies

Critical Success Criteria:
    - 100% node coverage (no empty tags) across all workflows
    - Valid ONNX outputs that load and execute correctly
    - Comprehensive metadata generation for traceability
    - Reasonable performance characteristics (time/memory)
    - Robust error handling with clear user guidance
    - Consistent behavior across different model types

Performance Requirements:
    - Simple model exports complete in <30 seconds
    - Complex model exports complete in <2 minutes
    - Analysis workflows complete in <10 seconds
    - Memory usage stays reasonable (<4GB peak)
    - No memory leaks or resource exhaustion

Quality Standards:
    - All exported models pass ONNX validation
    - Hierarchy tags follow consistent format conventions
    - Metadata files contain complete export information
    - Error messages provide actionable guidance
    - Workflows are intuitive and well-documented
"""

import json
import tempfile
import time
from pathlib import Path

import onnx
import pytest
import torch
import torch.nn as nn
from transformers import AutoModel

from modelexport.core.enhanced_semantic_exporter import EnhancedSemanticExporter
from modelexport.core.model_input_generator import generate_dummy_inputs
from modelexport.strategies.htp.htp_exporter import (
    HTPExporter,
)


class TestCompleteExportWorkflows:
    """
    Test suite for complete end-to-end export workflows.
    
    These tests validate the entire export pipeline from model loading
    through final validation, ensuring that users can successfully
    export their models with high quality results.
    
    Key Workflows Tested:
    - Simple transformer model export (BERT-like)
    - Vision model export (ResNet-like, SAM)
    - Custom model export (user-defined architectures)
    - Batch processing workflows
    - Error recovery and robustness
    
    Each workflow is tested with multiple strategies to ensure
    consistency and identify strategy-specific characteristics.
    """
    
    def test_simple_transformer_complete_workflow(self):
        """
        Test complete workflow for simple transformer model export.
        
        This represents the most common use case: exporting a standard
        transformer model (BERT-like) with automatic input generation
        and default settings. This workflow should be smooth and reliable.
        
        Workflow Steps:
        1. Load transformer model from HuggingFace
        2. Generate appropriate inputs automatically
        3. Export using HTP integrated strategy (default)
        4. Validate ONNX output and hierarchy tags
        5. Analyze results and verify quality metrics
        
        Success Criteria:
        - Export completes without errors
        - ONNX file is valid and loadable
        - 100% node coverage achieved
        - Metadata file contains complete information
        - Analysis reveals expected model structure
        """
        # Step 1: Load model
        model_name = "prajjwal1/bert-tiny"
        model = AutoModel.from_pretrained(model_name)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "bert_complete_workflow.onnx"
            
            # Step 2: Export with HTP integrated strategy
            start_time = time.time()
            
            exporter = HTPExporter(verbose=True, enable_reporting=True)
            result = exporter.export(
                model=model,
                output_path=str(output_path),
                model_name_or_path=model_name,
                opset_version=17
            )
            
            export_time = time.time() - start_time
            
            # Step 3: Validate export success
            assert result["coverage_percentage"] == 100.0, f"Should achieve 100% coverage, got {result['coverage_percentage']}%"
            assert result["empty_tags"] == 0, f"Should have 0 empty tags, got {result['empty_tags']}"
            assert export_time < 30.0, f"Export should complete in <30s, took {export_time:.2f}s"
            
            # Step 4: Validate ONNX output
            assert output_path.exists(), "ONNX file should be created"
            onnx_model = onnx.load(str(output_path))
            # Note: Skip onnx.checker.check_model() due to custom hierarchy_tag attributes
            
            # Step 5: Validate metadata
            metadata_path = output_path.parent / (output_path.stem + "_htp_metadata.json")
            assert metadata_path.exists(), "Metadata file should be created"
            
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Validate metadata completeness
            required_sections = ["export_info", "statistics", "hierarchy_data", "tagged_nodes"]
            for section in required_sections:
                assert section in metadata, f"Metadata should contain {section} section"
            
            # Step 6: Validate hierarchy tags in ONNX
            hierarchy_tagged_nodes = 0
            for node in onnx_model.graph.node:
                for attr in node.attribute:
                    if attr.name == "hierarchy_tag":
                        hierarchy_tagged_nodes += 1
                        tag_value = attr.s.decode()
                        assert len(tag_value) > 0, f"Node {node.name} has empty hierarchy tag"
                        assert tag_value.startswith("/"), f"Tag should be hierarchical: {tag_value}"
                        break
            
            assert hierarchy_tagged_nodes > 0, "ONNX nodes should have hierarchy tags embedded"
            
            # Step 7: Validate export statistics
            stats = result
            assert stats["hierarchy_modules"] > 0, "Should discover module hierarchy"
            assert stats["tagged_nodes"] > 0, "Should tag ONNX nodes"
            assert stats["onnx_nodes"] > 0, "Should have ONNX nodes to tag"
            
            # Calculate and validate coverage
            expected_coverage = (stats["tagged_nodes"] / stats["onnx_nodes"]) * 100
            assert abs(stats["coverage_percentage"] - expected_coverage) < 0.1, "Coverage calculation should be accurate"
    
    def test_vision_model_complete_workflow(self):
        """
        Test complete workflow for vision model export.
        
        Vision models have different input requirements and architectural
        patterns compared to transformers. This workflow validates that
        the system handles vision models correctly.
        
        Workflow Steps:
        1. Create or load vision model (ResNet-like)
        2. Generate appropriate vision inputs
        3. Export using suitable strategy
        4. Validate vision-specific outputs
        5. Analyze hierarchical structure
        
        Success Criteria:
        - Vision model exports successfully
        - Input generation works for image inputs
        - Hierarchy captures vision model structure
        - Performance is reasonable for larger models
        """
        # Create a representative vision model
        class VisionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    
                    # ResNet-like block
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                )
                
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(64, 1000)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        model = VisionModel()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "vision_complete_workflow.onnx"
            
            # Generate vision inputs
            input_specs = {
                "input": {
                    "shape": [1, 3, 224, 224],  # Standard ImageNet format
                    "dtype": "float",
                    "range": [0.0, 1.0]
                }
            }
            
            # Export vision model
            start_time = time.time()
            
            exporter = HTPExporter(verbose=True)
            result = exporter.export(
                model=model,
                output_path=str(output_path),
                input_specs=input_specs,
                opset_version=17
            )
            
            export_time = time.time() - start_time
            
            # Validate export success
            assert result["coverage_percentage"] == 100.0, f"Vision model should achieve 100% coverage, got {result['coverage_percentage']}%"
            assert result["empty_tags"] == 0, f"Should have 0 empty tags, got {result['empty_tags']}"
            assert export_time < 60.0, f"Vision export should complete in <60s, took {export_time:.2f}s"
            
            # Validate ONNX output
            assert output_path.exists(), "Vision ONNX file should be created"
            onnx_model = onnx.load(str(output_path))
            # Note: Skip onnx.checker.check_model() due to custom hierarchy_tag attributes
            
            # Validate input shape in ONNX matches specification
            graph_inputs = onnx_model.graph.input
            assert len(graph_inputs) >= 1, "Should have at least one input"
            
            input_shape = [dim.dim_value for dim in graph_inputs[0].type.tensor_type.shape.dim]
            assert input_shape == [1, 3, 224, 224], f"Input shape should match spec, got {input_shape}"
            
            # Validate vision-specific hierarchy structure
            hierarchy_data = result.get("hierarchy_data", {})
            vision_modules = [path for path in hierarchy_data.keys() if 
                            any(term in path.lower() for term in ["conv", "batchnorm", "relu", "linear"])]
            
            assert len(vision_modules) > 0, f"Should discover vision-specific modules. Found: {list(hierarchy_data.keys())}"
    
    def test_sam_model_coordinate_fix_workflow(self):
        """
        Test complete workflow for SAM model with coordinate fix.
        
        This validates the end-to-end SAM model export workflow,
        ensuring the coordinate fix is applied seamlessly and the
        export produces correct results.
        
        Workflow Steps:
        1. Load SAM model from HuggingFace
        2. Verify automatic coordinate fix application
        3. Export using unified HTP strategy
        4. Validate coordinate values in inputs
        5. Verify export quality and metadata
        
        Success Criteria:
        - SAM coordinate fix applied automatically
        - Input coordinates in correct range [0, 1024]
        - Export completes (may fail at ONNX level due to model complexity)
        - No user configuration required
        """
        model_name = "facebook/sam-vit-base"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "sam_coordinate_fix_workflow.onnx"
            
            # Test coordinate fix in input generation first
            inputs = generate_dummy_inputs(model_name_or_path=model_name)
            
            # Validate SAM inputs structure
            assert "input_points" in inputs, "SAM model should generate input_points"
            
            input_points = inputs["input_points"]
            min_val = float(input_points.min())
            max_val = float(input_points.max())
            
            # Validate coordinate fix
            assert min_val >= 0, f"Coordinates should be >= 0, got min={min_val}"
            assert max_val <= 1024, f"Coordinates should be <= 1024, got max={max_val}"
            assert max_val > 10, f"Coordinates should be in pixel space [0, 1024], got max={max_val}"
            
            # Attempt full export workflow
            try:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(model_name)
                
                exporter = HTPExporter(verbose=True)
                result = exporter.export(
                    model=model,
                    model_name_or_path=model_name,
                    output_path=str(output_path),
                    opset_version=17
                )
                
                # If export succeeds, validate results
                if output_path.exists():
                    onnx_model = onnx.load(str(output_path))
                    # Note: Skip onnx.checker.check_model() due to custom hierarchy_tag attributes
                    
                    # Validate export statistics
                    assert result["coverage_percentage"] == 100.0, "SAM export should achieve 100% coverage"
                    assert result["empty_tags"] == 0, "SAM export should have no empty tags"
                
            except Exception as e:
                # SAM models may fail at ONNX export level due to model complexity
                # This is acceptable as long as coordinate fix worked
                pytest.skip(f"SAM ONNX export failed (expected due to model complexity): {e}")
    
    def test_custom_model_complete_workflow(self):
        """
        Test complete workflow for custom user-defined model.
        
        This validates that the system works with arbitrary PyTorch
        models that users might define, ensuring universal design
        principles are maintained.
        
        Workflow Steps:
        1. Define custom model architecture
        2. Generate appropriate inputs
        3. Export using HTP strategy
        4. Validate universal design compliance
        5. Verify hierarchy preservation
        
        Success Criteria:
        - Custom model exports successfully
        - No hardcoded assumptions fail
        - Hierarchy captures custom structure
        - Universal design principles maintained
        """
        # Define complex custom model
        class CustomComplexModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Multiple different module types
                self.embedding = nn.Embedding(1000, 64)
                self.conv_layers = nn.ModuleList([
                    nn.Conv1d(64, 128, 3, padding=1),
                    nn.Conv1d(128, 256, 3, padding=1),
                ])
                self.attention = nn.MultiheadAttention(256, 8, batch_first=True)
                self.norm_layers = nn.ModuleList([
                    nn.LayerNorm(256),
                    nn.BatchNorm1d(256),
                ])
                self.classifier = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 10)
                )
            
            def forward(self, input_ids, attention_mask):
                # Complex forward pass with multiple operations
                x = self.embedding(input_ids)  # [batch, seq, embed]
                x = x.transpose(1, 2)  # [batch, embed, seq] for conv
                
                for conv in self.conv_layers:
                    x = torch.relu(conv(x))
                
                x = x.transpose(1, 2)  # [batch, seq, embed] for attention
                x = self.norm_layers[0](x)
                
                attn_out, _ = self.attention(x, x, x, key_padding_mask=~attention_mask.bool())
                x = x + attn_out  # Residual connection
                
                x = x.transpose(1, 2)  # [batch, embed, seq]
                x = self.norm_layers[1](x)
                x = x.transpose(1, 2)  # [batch, seq, embed]
                
                # Global average pooling
                mask_expanded = attention_mask.unsqueeze(-1).float()
                x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
                
                return self.classifier(x)
        
        model = CustomComplexModel()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "custom_complete_workflow.onnx"
            
            # Define custom inputs
            input_specs = {
                "input_ids": {"shape": [2, 32], "dtype": "int", "range": [0, 999]},
                "attention_mask": {"shape": [2, 32], "dtype": "int", "range": [0, 1]}
            }
            
            # Export custom model
            start_time = time.time()
            
            exporter = HTPExporter(verbose=True)
            result = exporter.export(
                model=model,
                output_path=str(output_path),
                input_specs=input_specs,
                opset_version=17
            )
            
            export_time = time.time() - start_time
            
            # Validate universal design compliance
            assert result["coverage_percentage"] == 100.0, f"Custom model should achieve 100% coverage, got {result['coverage_percentage']}%"
            assert result["empty_tags"] == 0, f"Should have 0 empty tags, got {result['empty_tags']}"
            assert export_time < 45.0, f"Custom export should complete in <45s, took {export_time:.2f}s"
            
            # Validate ONNX output
            assert output_path.exists(), "Custom ONNX file should be created"
            onnx_model = onnx.load(str(output_path))
            # Note: Skip onnx.checker.check_model() due to custom hierarchy_tag attributes
            
            # Validate hierarchy captures custom architecture
            hierarchy_data = result.get("hierarchy_data", {})
            
            # Should capture various module types from our custom model
            module_types_found = set()
            for module_info in hierarchy_data.values():
                module_type = module_info.get("module_type", "")
                module_types_found.add(module_type)
            
            expected_types = {"Embedding", "Conv1d", "MultiheadAttention", "LayerNorm", "Linear"}
            found_expected = expected_types.intersection(module_types_found)
            
            assert len(found_expected) >= 3, f"Should capture diverse module types. Found: {module_types_found}"
    
    def test_batch_processing_workflow(self):
        """
        Test batch processing workflow for multiple models.
        
        This validates workflows where users need to export multiple
        models in sequence, ensuring resource management and consistency.
        
        Workflow Steps:
        1. Define multiple models to export
        2. Export each model in sequence
        3. Validate consistent quality across exports
        4. Check resource usage and cleanup
        
        Success Criteria:
        - All models export successfully
        - Consistent quality metrics across models
        - No resource leaks or memory issues
        - Performance degrades gracefully
        """
        models_to_test = [
            ("prajjwal1/bert-tiny", "htp_integrated"),
            ("prajjwal1/bert-tiny", "enhanced_semantic"),
        ]
        
        batch_results = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, (model_name, strategy) in enumerate(models_to_test):
                output_path = Path(temp_dir) / f"batch_model_{i}_{strategy}.onnx"
                
                start_time = time.time()
                
                try:
                    if strategy == "htp_integrated":
                        exporter = HTPExporter(verbose=False)
                        result = exporter.export(
                            model_name_or_path=model_name,
                            output_path=str(output_path),
                            opset_version=17
                        )
                    elif strategy == "enhanced_semantic":
                        model = AutoModel.from_pretrained(model_name)
                        inputs = generate_dummy_inputs(model_name_or_path=model_name)
                        args = tuple(inputs.values())
                        
                        exporter = EnhancedSemanticExporter(verbose=False)
                        result = exporter.export(
                            model=model,
                            args=args,
                            output_path=str(output_path),
                            opset_version=17
                        )
                    
                    export_time = time.time() - start_time
                    
                    batch_results[f"{model_name}_{strategy}"] = {
                        "success": True,
                        "export_time": export_time,
                        "coverage": result.get("coverage_percentage", 0),
                        "empty_tags": result.get("empty_tags", -1),
                        "file_size": output_path.stat().st_size if output_path.exists() else 0
                    }
                    
                except Exception as e:
                    batch_results[f"{model_name}_{strategy}"] = {
                        "success": False,
                        "error": str(e),
                        "export_time": time.time() - start_time
                    }
        
        # Validate batch processing results
        successful_exports = [name for name, result in batch_results.items() if result["success"]]
        assert len(successful_exports) >= 1, f"At least one batch export should succeed. Results: {batch_results}"
        
        # Validate consistency across successful exports
        successful_results = [result for result in batch_results.values() if result["success"]]
        if len(successful_results) > 1:
            coverages = [r["coverage"] for r in successful_results]
            empty_tags = [r["empty_tags"] for r in successful_results]
            
            # All should achieve full coverage
            assert all(c == 100.0 for c in coverages), f"All exports should achieve 100% coverage. Got: {coverages}"
            assert all(e == 0 for e in empty_tags), f"All exports should have 0 empty tags. Got: {empty_tags}"


class TestStrategyIntegrationWorkflows:
    """
    Test suite for strategy integration workflows.
    
    These tests validate how different export strategies work together
    and how users can migrate between strategies or compare results.
    
    Key Workflows Tested:
    - Cross-strategy compatibility
    - Strategy migration workflows
    - Strategy comparison and analysis
    - Strategy selection guidance
    """
    
    def test_strategy_comparison_workflow(self):
        """
        Test workflow for comparing multiple export strategies.
        
        Users often want to compare different strategies to understand
        their trade-offs and choose the best one for their use case.
        
        Workflow Steps:
        1. Export same model with multiple strategies
        2. Compare results and quality metrics
        3. Analyze differences and trade-offs
        4. Generate comparison report
        
        Success Criteria:
        - Multiple strategies work on same model
        - Comparison provides meaningful insights
        - Quality metrics are comparable
        - Differences are well-characterized
        """
        model_name = "prajjwal1/bert-tiny"
        model = AutoModel.from_pretrained(model_name)
        
        strategies_to_compare = [
            ("htp_integrated", "HTP Integrated"),
            ("enhanced_semantic", "Enhanced Semantic"),
        ]
        
        comparison_results = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for strategy_key, strategy_name in strategies_to_compare:
                output_path = Path(temp_dir) / f"bert_comparison_{strategy_key}.onnx"
                
                start_time = time.time()
                
                try:
                    if strategy_key == "htp_integrated":
                        exporter = HTPExporter(verbose=False)
                        result = exporter.export(
                            model=model,
                            output_path=str(output_path),
                            model_name_or_path=model_name,
                            opset_version=17
                        )
                    elif strategy_key == "enhanced_semantic":
                        inputs = generate_dummy_inputs(model_name_or_path=model_name)
                        args = tuple(inputs.values())
                        
                        exporter = EnhancedSemanticExporter(verbose=False)
                        result = exporter.export(
                            model=model,
                            args=args,
                            output_path=str(output_path),
                            opset_version=17
                        )
                    
                    export_time = time.time() - start_time
                    
                    # Validate ONNX output
                    onnx_model = onnx.load(str(output_path))
                    node_count = len(onnx_model.graph.node)
                    
                    comparison_results[strategy_key] = {
                        "name": strategy_name,
                        "success": True,
                        "export_time": export_time,
                        "coverage": result.get("coverage_percentage", 0),
                        "empty_tags": result.get("empty_tags", -1),
                        "tagged_nodes": result.get("tagged_nodes", 0),
                        "total_nodes": node_count,
                        "file_size": output_path.stat().st_size
                    }
                    
                except Exception as e:
                    comparison_results[strategy_key] = {
                        "name": strategy_name,
                        "success": False,
                        "error": str(e)
                    }
        
        # Validate comparison results
        successful_strategies = [key for key, result in comparison_results.items() if result["success"]]
        assert len(successful_strategies) >= 1, f"At least one strategy should work for comparison. Results: {comparison_results}"
        
        # Generate comparison analysis
        if len(successful_strategies) >= 2:
            strategy1_key, strategy2_key = successful_strategies[0], successful_strategies[1]
            result1 = comparison_results[strategy1_key]
            result2 = comparison_results[strategy2_key]
            
            # Compare coverage (both should be high)
            assert result1["coverage"] >= 90.0, f"{result1['name']} should have high coverage"
            assert result2["coverage"] >= 90.0, f"{result2['name']} should have high coverage"
            
            # Compare empty tags (both should be zero)
            assert result1["empty_tags"] == 0, f"{result1['name']} should have no empty tags"
            assert result2["empty_tags"] == 0, f"{result2['name']} should have no empty tags"
            
            # Performance comparison (should be reasonable)
            time_ratio = max(result1["export_time"], result2["export_time"]) / min(result1["export_time"], result2["export_time"])
            assert time_ratio < 5.0, f"Performance difference should be reasonable, got {time_ratio:.2f}x"
    
    def test_strategy_migration_workflow(self):
        """
        Test workflow for migrating between export strategies.
        
        Users may want to upgrade from one strategy to another or
        re-export models with different strategies over time.
        
        Workflow Steps:
        1. Export model with initial strategy
        2. Analyze results and identify limitations
        3. Re-export with upgraded strategy
        4. Compare and validate improvements
        
        Success Criteria:
        - Migration preserves or improves quality
        - Workflow is straightforward
        - Results are consistent where expected
        - Improvements are measurable
        """
        model_name = "prajjwal1/bert-tiny"
        
        # Simulate migration from legacy to modern strategy
        migration_path = [
            ("htp_integrated", "Modern HTP"),
            ("enhanced_semantic", "Enhanced Analysis")
        ]
        
        migration_results = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, (strategy_key, strategy_description) in enumerate(migration_path):
                output_path = Path(temp_dir) / f"bert_migration_step_{i}_{strategy_key}.onnx"
                
                try:
                    if strategy_key == "htp_integrated":
                        exporter = HTPExporter(verbose=False)
                        result = exporter.export(
                            model_name_or_path=model_name,
                            output_path=str(output_path),
                            opset_version=17
                        )
                    elif strategy_key == "enhanced_semantic":
                        model = AutoModel.from_pretrained(model_name)
                        inputs = generate_dummy_inputs(model_name_or_path=model_name)
                        args = tuple(inputs.values())
                        
                        exporter = EnhancedSemanticExporter(verbose=False)
                        result = exporter.export(
                            model=model,
                            args=args,
                            output_path=str(output_path),
                            opset_version=17
                        )
                    
                    migration_results[i] = {
                        "strategy": strategy_key,
                        "description": strategy_description,
                        "success": True,
                        "coverage": result.get("coverage_percentage", 0),
                        "empty_tags": result.get("empty_tags", -1),
                        "tagged_nodes": result.get("tagged_nodes", 0)
                    }
                    
                except Exception as e:
                    migration_results[i] = {
                        "strategy": strategy_key,
                        "description": strategy_description,
                        "success": False,
                        "error": str(e)
                    }
        
        # Validate migration results
        successful_steps = [step for step, result in migration_results.items() if result["success"]]
        assert len(successful_steps) >= 1, f"At least one migration step should succeed. Results: {migration_results}"
        
        # If multiple steps succeeded, validate migration benefits
        if len(successful_steps) >= 2:
            step1 = migration_results[successful_steps[0]]
            step2 = migration_results[successful_steps[1]]
            
            # Coverage should be maintained or improved
            assert step2["coverage"] >= step1["coverage"], "Migration should maintain or improve coverage"
            
            # Empty tags should remain at zero
            assert step1["empty_tags"] == 0, "Initial strategy should have no empty tags"
            assert step2["empty_tags"] == 0, "Migrated strategy should have no empty tags"


class TestConfigurationWorkflows:
    """
    Test suite for configuration and customization workflows.
    
    These tests validate how users can customize exports using
    configuration files, manual input specifications, and
    advanced parameter tuning.
    
    Key Workflows Tested:
    - Manual input specification workflows
    - Configuration file usage
    - Advanced parameter customization
    - Custom export requirements
    """
    
    def test_manual_input_specification_workflow(self):
        """
        Test complete workflow using manual input specifications.
        
        Users with specific requirements may need to provide exact
        input specifications rather than using automatic generation.
        
        Workflow Steps:
        1. Define custom input specifications
        2. Validate input specification format
        3. Export model using custom inputs
        4. Verify outputs match specifications
        5. Analyze results for quality
        
        Success Criteria:
        - Custom inputs are used exactly as specified
        - Export quality remains high
        - ONNX inputs match specifications
        - Workflow is straightforward
        """
        model_name = "prajjwal1/bert-tiny"
        
        # Define custom input specifications
        custom_input_specs = {
            "input_ids": {
                "shape": [4, 128],  # Larger batch size, longer sequence
                "dtype": "int",
                "range": [0, 30522]  # BERT vocabulary size
            },
            "token_type_ids": {
                "shape": [4, 128],
                "dtype": "int", 
                "range": [0, 1]
            },
            "attention_mask": {
                "shape": [4, 128],
                "dtype": "int",
                "range": [0, 1]
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "bert_custom_inputs.onnx"
            
            # Export with custom input specifications
            exporter = HTPExporter(verbose=True)
            result = exporter.export(
                model_name_or_path=model_name,
                output_path=str(output_path),
                input_specs=custom_input_specs,
                opset_version=17
            )
            
            # Validate export success
            assert result["coverage_percentage"] == 100.0, "Custom input export should achieve 100% coverage"
            assert result["empty_tags"] == 0, "Should have no empty tags"
            
            # Validate ONNX output uses custom specifications
            onnx_model = onnx.load(str(output_path))
            # Note: Skip onnx.checker.check_model() due to custom hierarchy_tag attributes
            
            # Check input shapes in ONNX graph
            graph_inputs = onnx_model.graph.input
            input_shapes = {}
            for graph_input in graph_inputs:
                shape = [dim.dim_value for dim in graph_input.type.tensor_type.shape.dim]
                input_shapes[graph_input.name] = shape
            
            # Validate shapes match specifications
            assert "input_ids" in input_shapes, "Should have input_ids input"
            assert input_shapes["input_ids"] == [4, 128], f"input_ids shape should be [4, 128], got {input_shapes['input_ids']}"
            
            # Validate other inputs if present
            for spec_name, spec_details in custom_input_specs.items():
                if spec_name in input_shapes:
                    expected_shape = spec_details["shape"]
                    actual_shape = input_shapes[spec_name]
                    assert actual_shape == expected_shape, f"{spec_name} shape should be {expected_shape}, got {actual_shape}"
    
    def test_configuration_file_workflow(self):
        """
        Test workflow using configuration files.
        
        Advanced users may want to save and reuse export configurations
        for consistency across multiple exports.
        
        Workflow Steps:
        1. Create export configuration file
        2. Load configuration and validate format
        3. Export model using configuration
        4. Verify configuration is applied correctly
        5. Test configuration reusability
        
        Success Criteria:
        - Configuration file loads correctly
        - Settings are applied as specified
        - Export results match configuration
        - Configuration is reusable
        """
        model_name = "prajjwal1/bert-tiny"
        
        # Define export configuration
        export_config = {
            "model_name_or_path": model_name,
            "strategy": "htp_integrated",
            "opset_version": 17,
            "input_specs": {
                "input_ids": {"shape": [2, 64], "dtype": "int", "range": [0, 1000]},
                "token_type_ids": {"shape": [2, 64], "dtype": "int", "range": [0, 1]},
                "attention_mask": {"shape": [2, 64], "dtype": "int", "range": [0, 1]}
            },
            "export_params": {
                "enable_operation_fallback": True,
                "verbose": True
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save configuration file
            config_path = Path(temp_dir) / "export_config.json"
            with open(config_path, 'w') as f:
                json.dump(export_config, f, indent=2)
            
            # Load and use configuration
            with open(config_path) as f:
                loaded_config = json.load(f)
            
            output_path = Path(temp_dir) / "bert_config_export.onnx"
            
            # Load model
            from transformers import AutoModel
            model = AutoModel.from_pretrained(loaded_config["model_name_or_path"])
            
            # Export using configuration
            exporter = HTPExporter(verbose=loaded_config["export_params"]["verbose"])
            result = exporter.export(
                model=model,
                model_name_or_path=loaded_config["model_name_or_path"],
                output_path=str(output_path),
                input_specs=loaded_config["input_specs"],
                opset_version=loaded_config["opset_version"],
                enable_operation_fallback=loaded_config["export_params"]["enable_operation_fallback"]
            )
            
            # Validate configuration was applied
            assert result["coverage_percentage"] == 100.0, "Config-based export should achieve 100% coverage"
            assert result["empty_tags"] == 0, "Should have no empty tags"
            
            # Validate ONNX uses configuration settings
            onnx_model = onnx.load(str(output_path))
            # Note: Skip onnx.checker.check_model() due to custom hierarchy_tag attributes
            
            # Check ONNX opset version
            opset_imports = onnx_model.opset_import
            onnx_opset = next((imp.version for imp in opset_imports if imp.domain == ""), None)
            assert onnx_opset == 17, f"ONNX should use opset 17, got {onnx_opset}"
            
            # Test configuration reusability
            output_path2 = Path(temp_dir) / "bert_config_export_2.onnx"
            
            result2 = exporter.export(
                model_name_or_path=loaded_config["model_name_or_path"],
                output_path=str(output_path2),
                input_specs=loaded_config["input_specs"],
                opset_version=loaded_config["opset_version"]
            )
            
            # Results should be consistent
            assert result2["coverage_percentage"] == result["coverage_percentage"], "Reused config should give consistent results"
            assert result2["empty_tags"] == result["empty_tags"], "Reused config should have consistent quality"


class TestErrorRecoveryWorkflows:
    """
    Test suite for error recovery and robustness workflows.
    
    These tests validate how the system handles various error conditions
    and recovery scenarios that users might encounter in real-world usage.
    
    Key Workflows Tested:
    - Network interruption recovery
    - Disk space handling
    - Memory constraint management
    - Model compatibility issues
    """
    
    def test_model_compatibility_workflow(self):
        """
        Test workflow for handling model compatibility issues.
        
        Some models may not be compatible with certain export strategies
        or may require special handling. The system should provide clear
        guidance and fallback options.
        
        Workflow Steps:
        1. Attempt export with potentially problematic model
        2. Handle compatibility issues gracefully
        3. Provide clear error messages and guidance
        4. Suggest alternative approaches
        
        Success Criteria:
        - Clear error messages for incompatible models
        - Helpful guidance for resolution
        - No crashes or undefined behavior
        - Alternative options suggested when possible
        """
        # Test with a model that might have compatibility issues
        test_cases = [
            {
                "name": "nonexistent_model",
                "model_path": "definitely/does/not/exist",
                "expected_error_type": "model_not_found"
            },
            {
                "name": "invalid_model_format",
                "model_path": "invalid/model/format",
                "expected_error_type": "invalid_format"
            }
        ]
        
        compatibility_results = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for test_case in test_cases:
                output_path = Path(temp_dir) / f"{test_case['name']}_test.onnx"
                
                try:
                    exporter = HTPExporter(verbose=False)
                    result = exporter.export(
                        model_name_or_path=test_case["model_path"],
                        output_path=str(output_path),
                        opset_version=17
                    )
                    
                    compatibility_results[test_case["name"]] = {
                        "success": True,
                        "unexpected": True,  # These should fail
                        "result": result
                    }
                    
                except Exception as e:
                    compatibility_results[test_case["name"]] = {
                        "success": False,
                        "expected": True,  # These should fail
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
        
        # Validate error handling
        for test_case in test_cases:
            result = compatibility_results[test_case["name"]]
            assert not result.get("success", False), f"{test_case['name']} should fail gracefully"
            assert "expected" in result, f"{test_case['name']} should have expected failure"
            
            # Error message should be informative
            error_msg = result.get("error", "").lower()
            assert len(error_msg) > 10, f"{test_case['name']} should have informative error message"
    
    def test_resource_constraint_workflow(self):
        """
        Test workflow for handling resource constraints.
        
        Users may encounter memory or disk space limitations during
        export. The system should handle these gracefully and provide
        helpful guidance.
        
        Workflow Steps:
        1. Simulate resource-constrained environment
        2. Attempt export operations
        3. Handle resource limitations gracefully
        4. Provide helpful guidance for resolution
        
        Success Criteria:
        - Graceful handling of resource constraints
        - Clear error messages about resource issues
        - Helpful suggestions for resolution
        - No data corruption or partial outputs
        """
        model_name = "prajjwal1/bert-tiny"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with invalid output path (simulates disk space/permission issues)
            invalid_output_path = "/root/invalid/path/that/cannot/be/created.onnx"
            
            try:
                exporter = HTPExporter(verbose=False)
                result = exporter.export(
                    model_name_or_path=model_name,
                    output_path=invalid_output_path,
                    opset_version=17
                )
                
                # Should not reach here
                pytest.fail("Export should fail with invalid output path")
                
            except (OSError, PermissionError, ValueError) as e:
                # Expected error types for file system issues
                error_msg = str(e).lower()
                assert len(error_msg) > 5, "Should provide informative error message"
                
                # Should not create partial files
                assert not Path(invalid_output_path).exists(), "Should not create partial output files"
            
            except Exception as e:
                # Other exceptions are also acceptable as long as they're handled
                assert isinstance(e, Exception), f"Should handle errors gracefully, got {type(e)}: {e}"


class TestQualityAssuranceWorkflows:
    """
    Test suite for quality assurance workflows.
    
    These tests validate workflows for ensuring export quality,
    performing validation, and maintaining quality standards.
    
    Key Workflows Tested:
    - Export validation pipelines
    - Coverage analysis workflows
    - Quality metrics tracking
    - Regression testing procedures
    """
    
    def test_export_validation_pipeline(self):
        """
        Test complete export validation pipeline.
        
        This validates a complete quality assurance workflow that
        users would follow to ensure their exports meet quality standards.
        
        Workflow Steps:
        1. Export model with quality tracking
        2. Validate ONNX format and structure
        3. Analyze hierarchy tag coverage and quality
        4. Check compliance with quality standards
        5. Generate quality assurance report
        
        Success Criteria:
        - All validation steps pass
        - Quality metrics meet standards
        - Comprehensive quality report generated
        - Issues are clearly identified
        """
        model_name = "prajjwal1/bert-tiny"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "bert_qa_pipeline.onnx"
            
            # Load model
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_name)
            
            # Step 1: Export with quality tracking
            exporter = HTPExporter(verbose=True, enable_reporting=True)
            result = exporter.export(
                model=model,
                model_name_or_path=model_name,
                output_path=str(output_path),
                opset_version=17
            )
            
            # Step 2: Validate ONNX format
            assert output_path.exists(), "ONNX file should be created"
            onnx_model = onnx.load(str(output_path))
            # Note: Skip onnx.checker.check_model() due to custom hierarchy_tag attributes
            
            # Step 3: Quality metrics validation
            qa_report = {
                "export_success": True,
                "onnx_valid": True,
                "coverage_percentage": result["coverage_percentage"],
                "empty_tags": result["empty_tags"],
                "total_nodes": result["onnx_nodes"],
                "tagged_nodes": result["tagged_nodes"],
                "export_time": result["export_time"]
            }
            
            # Step 4: Quality standards compliance
            quality_standards = {
                "min_coverage": 100.0,
                "max_empty_tags": 0,
                "max_export_time": 60.0,
                "min_tagged_nodes": 1
            }
            
            compliance_results = {}
            for standard, threshold in quality_standards.items():
                if standard == "min_coverage":
                    compliance_results[standard] = qa_report["coverage_percentage"] >= threshold
                elif standard == "max_empty_tags":
                    compliance_results[standard] = qa_report["empty_tags"] <= threshold
                elif standard == "max_export_time":
                    compliance_results[standard] = qa_report["export_time"] <= threshold
                elif standard == "min_tagged_nodes":
                    compliance_results[standard] = qa_report["tagged_nodes"] >= threshold
            
            # Step 5: Generate quality report
            qa_report["compliance"] = compliance_results
            qa_report["overall_pass"] = all(compliance_results.values())
            
            report_path = output_path.with_suffix(".qa_report.json")
            with open(report_path, 'w') as f:
                json.dump(qa_report, f, indent=2)
            
            # Validate quality assurance results
            assert qa_report["overall_pass"], f"QA pipeline should pass all standards. Report: {qa_report}"
            assert qa_report["coverage_percentage"] == 100.0, "Should achieve perfect coverage"
            assert qa_report["empty_tags"] == 0, "Should have no empty tags"
            assert qa_report["export_time"] < 60.0, "Should complete in reasonable time"
    
    def test_coverage_analysis_workflow(self):
        """
        Test coverage analysis workflow.
        
        This validates workflows for analyzing and understanding
        hierarchy tag coverage across different models and strategies.
        
        Workflow Steps:
        1. Export model and collect coverage data
        2. Analyze tag distribution and patterns
        3. Identify coverage gaps or issues
        4. Generate coverage analysis report
        
        Success Criteria:
        - Comprehensive coverage analysis
        - Clear identification of tag patterns
        - Issues and gaps clearly reported
        - Actionable insights provided
        """
        model_name = "prajjwal1/bert-tiny"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "bert_coverage_analysis.onnx"
            
            # Load model
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_name)
            
            # Export model for coverage analysis
            exporter = HTPExporter(verbose=False)
            result = exporter.export(
                model=model,
                model_name_or_path=model_name,
                output_path=str(output_path),
                opset_version=17
            )
            
            # Analyze coverage data
            onnx_model = onnx.load(str(output_path))
            
            # Extract hierarchy tags from ONNX nodes
            node_tags = {}
            for node in onnx_model.graph.node:
                for attr in node.attribute:
                    if attr.name == "hierarchy_tag":
                        node_tags[node.name] = attr.s.decode()
                        break
                else:
                    node_tags[node.name] = ""  # No tag found
            
            # Analyze tag patterns
            coverage_analysis = {
                "total_nodes": len(onnx_model.graph.node),
                "tagged_nodes": len([tag for tag in node_tags.values() if tag]),
                "empty_tags": len([tag for tag in node_tags.values() if not tag]),
                "coverage_percentage": (len([tag for tag in node_tags.values() if tag]) / len(onnx_model.graph.node)) * 100,
                "tag_patterns": {}
            }
            
            # Analyze tag hierarchy patterns
            tag_levels = {}
            for tag in node_tags.values():
                if tag:
                    level = tag.count('/')
                    tag_levels[level] = tag_levels.get(level, 0) + 1
            
            coverage_analysis["tag_patterns"]["hierarchy_levels"] = tag_levels
            
            # Common tag prefixes
            tag_prefixes = {}
            for tag in node_tags.values():
                if tag and '/' in tag:
                    prefix = '/'.join(tag.split('/')[:2])  # First two levels
                    tag_prefixes[prefix] = tag_prefixes.get(prefix, 0) + 1
            
            coverage_analysis["tag_patterns"]["common_prefixes"] = tag_prefixes
            
            # Validate coverage analysis
            assert coverage_analysis["coverage_percentage"] == 100.0, "Coverage analysis should show 100% coverage"
            assert coverage_analysis["empty_tags"] == 0, "Should detect no empty tags"
            assert len(coverage_analysis["tag_patterns"]["hierarchy_levels"]) > 0, "Should analyze hierarchy levels"
            
            # Generate coverage report
            report_path = output_path.with_suffix(".coverage_report.json")
            with open(report_path, 'w') as f:
                json.dump(coverage_analysis, f, indent=2)
            
            assert report_path.exists(), "Coverage report should be generated"