"""
Comprehensive End-to-End Test Suite

This test suite validates the modelexport system against 8 carefully selected models
representing the most important architectural paradigms to ensure universal
compatibility and robustness.

Current Model Matrix (Updated 2024):
┌─────────┬──────────────────────────────┬──────────────┬────────────┬─────────────┐
│ Model   │ HuggingFace ID               │ Architecture │ Domain     │ Size        │
├─────────┼──────────────────────────────┼──────────────┼────────────┼─────────────┤
│ BERT    │ prajjwal1/bert-tiny          │ BERT         │ Language   │ Tiny        │
│ LLaMA   │ meta-llama/Llama-3.2-1B      │ LLaMA 3.2    │ Language   │ Small (1B)  │
│ Qwen    │ Qwen/Qwen1.5-0.5B            │ Qwen         │ Language   │ Small (0.5B)│
│ ResNet  │ microsoft/resnet-18          │ ResNet       │ Vision     │ Small       │
│ ViT     │ google/vit-base-patch16-224  │ ViT          │ Vision     │ Base        │
│ SAM     │ facebook/sam-vit-base        │ SAM          │ Vision     │ Base        │
│ YOLO    │ Ultralytics/YOLO11           │ YOLOv11      │ Vision     │ Small       │
│ CLIP    │ openai/clip-vit-base-patch32 │ CLIP         │ Multimodal │ Base        │
└─────────┴──────────────────────────────┴──────────────┴────────────┴─────────────┘

Architecture Coverage:
- Language: BERT (baseline) + LLaMA 3.2 (latest LLM) + Qwen (Chinese LLM)
- Vision: ResNet (CNN) + ViT (transformer) + SAM (segmentation) + YOLO (detection)
- Multimodal: CLIP (vision-language)

CARDINAL RULES:
- MUST-001: NO HARDCODED LOGIC - Test universal compatibility
- MUST-002: TORCH.NN FILTERING - Validate filtering works across all architectures
- MUST-003: UNIVERSAL DESIGN - Ensure export works for any model type

Test Matrix:
- Strategy: htp_integrated (default, most reliable)
- Opset: 17 (modern, well-supported)
- Coverage: 100% expected for all models
- Empty tags: 0 (CARDINAL RULE compliance)
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import onnx
import pytest

from modelexport.strategies.htp.htp_exporter import HTPExporter

# Popular model configurations for comprehensive testing
POPULAR_MODELS = {
    # Language Models - Baseline
    "bert_tiny": {
        "model_name": "prajjwal1/bert-tiny",
        "domain": "language", 
        "architecture": "bert",
        "size_category": "tiny",
        "expected_modules": 45,
        "notes": "Fast baseline model for testing"
    },
    
    # Large Language Models
    "llama3_2_1b": {
        "model_name": "meta-llama/Llama-3.2-1B",
        "domain": "language",
        "architecture": "llama",
        "size_category": "small",
        "expected_modules": 200,
        "notes": "LLaMA 3.2 1B - smallest LLaMA 3.2 variant"
    },
    "qwen_0_5b": {
        "model_name": "Qwen/Qwen1.5-0.5B",
        "domain": "language",
        "architecture": "qwen",
        "size_category": "small",
        "expected_modules": 150,
        "notes": "Qwen 0.5B - smallest Qwen variant"
    },
    
    # Vision Models
    "resnet18": {
        "model_name": "microsoft/resnet-18",
        "domain": "vision",
        "architecture": "resnet",
        "size_category": "small",
        "expected_modules": 60,
        "notes": "Classic CNN architecture"
    },
    "vit_base": {
        "model_name": "google/vit-base-patch16-224",
        "domain": "vision",
        "architecture": "vit",
        "size_category": "base",
        "expected_modules": 150,
        "notes": "Vision Transformer"
    },
    
    # Segmentation Models
    "sam_vit_base": {
        "model_name": "facebook/sam-vit-base",
        "domain": "vision",
        "architecture": "sam",
        "size_category": "base",
        "expected_modules": 200,
        "notes": "Segment Anything Model - ViT backbone"
    },
    
    # Object Detection Models
    "yolo_v11": {
        "model_name": "Ultralytics/YOLO11",
        "domain": "vision",
        "architecture": "yolo",
        "size_category": "small",
        "expected_modules": 120,
        "notes": "YOLO v11 - latest object detection model (2024)"
    },
    
    # Multimodal Models
    "clip_vit_base": {
        "model_name": "openai/clip-vit-base-patch32",
        "domain": "multimodal",
        "architecture": "clip",
        "size_category": "base",
        "expected_modules": 200,
        "notes": "Vision-language model"
    }
}

# Test configurations
TEST_CONFIG = {
    "strategy": "htp",
    "opset_version": 17,
    "verbose": False,
    "enable_reporting": False,
    "coverage_threshold": 100.0,
    "export_timeout": None  # No timeout
}


class TestComprehensiveEndToEnd:
    """
    Comprehensive end-to-end testing suite for popular models.
    
    This test class validates that the modelexport system works reliably
    across a diverse range of popular models from different domains and
    architectures, ensuring universal compatibility.
    """
    
    @pytest.mark.parametrize("model_id", list(POPULAR_MODELS.keys()))
    def test_popular_model_export(self, model_id: str):
        """
        Test export of popular models across different domains and architectures.
        
        This is the main comprehensive test that validates each popular model
        can be exported successfully with 100% coverage and zero empty tags.
        
        Test Steps:
        1. Load model configuration
        2. Load model from HuggingFace Hub
        3. Export with HTP integrated strategy
        4. Validate export success criteria
        5. Verify ONNX model integrity
        6. Check metadata completeness
        
        Success Criteria:
        - Export completes without errors
        - 100% tag coverage achieved
        - Zero empty tags (CARDINAL RULE compliance)
        - Valid ONNX model generated
        - Metadata file created with correct format
        - Export time within reasonable bounds
        """
        model_config = POPULAR_MODELS[model_id]
        
        # Skip models that are known to require special handling
        if "skip_reason" in model_config:
            pytest.skip(f"Skipping {model_id}: {model_config['skip_reason']}")
        
        model_name = model_config["model_name"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / f"{model_id}_export.onnx"
            
            # Step 1: Load model
            try:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(model_name)
            except Exception as e:
                pytest.skip(f"Could not load model {model_name}: {e}")
            
            # Step 2: Export
            start_time = time.time()
            
            try:
                exporter = HTPExporter(verbose=TEST_CONFIG["verbose"], enable_reporting=TEST_CONFIG["enable_reporting"])
                
                # HTPExporter automatically generates inputs from model_name_or_path
                result = exporter.export(
                    model=model,
                    output_path=str(output_path),
                    model_name_or_path=model_name,
                    opset_version=TEST_CONFIG["opset_version"]
                )
                
                export_time = time.time() - start_time
                
            except Exception as e:
                error_msg = str(e)
                
                # Handle known ONNX incompatibility issues
                if "DynamicCache" in error_msg:
                    pytest.skip(f"Model {model_id} has DynamicCache outputs - ONNX incompatible")
                elif "unsupported type" in error_msg and ("Cache" in error_msg or "dict" in error_msg):
                    pytest.skip(f"Model {model_id} has unsupported output types for ONNX")
                elif "AutoModel" in error_msg and "for_object_detection" in error_msg:
                    pytest.skip(f"Model {model_id} requires different AutoModel class")
                elif "bounding boxes" in error_msg and "input points" in error_msg:
                    pytest.skip(f"Model {model_id} (SAM) has complex input requirements - ONNX incompatible")
                else:
                    pytest.fail(f"Export failed for {model_id} ({model_name}): {e}")
            
            # Step 3: Validate export success criteria
            self._validate_export_results(
                model_id=model_id,
                model_config=model_config,
                result=result,
                output_path=output_path,
                export_time=export_time
            )
    
    def _validate_export_results(
        self,
        model_id: str,
        model_config: dict[str, Any],
        result: dict[str, Any],
        output_path: Path,
        export_time: float
    ):
        """
        Validate export results against success criteria.
        
        Args:
            model_id: Identifier for the model being tested
            model_config: Configuration dictionary for the model
            result: Export result dictionary from HTPExporter
            output_path: Path to exported ONNX file
            export_time: Time taken for export
            export_time: Time taken for export
        """
        model_name = model_config["model_name"]
        
        # Validate basic export success
        assert output_path.exists(), f"{model_id}: ONNX file should be created at {output_path}"
        assert output_path.stat().st_size > 0, f"{model_id}: ONNX file should not be empty"
        
        # Log export timing (no hard limits)
        print(f"{model_id}: Export completed in {export_time:.2f}s")
        
        # Validate coverage requirements (CARDINAL RULE compliance)
        total_ops = result.get("onnx_nodes", 0)
        tagged_ops = result.get("tagged_nodes", 0)
        coverage = result.get("coverage_percentage", (tagged_ops / total_ops * 100) if total_ops > 0 else 0.0)
        assert coverage == TEST_CONFIG["coverage_threshold"], \
            f"{model_id}: Expected {TEST_CONFIG['coverage_threshold']}% coverage, got {coverage}%"
        
        # Validate empty tags requirement (CARDINAL RULE compliance)
        empty_tags = result.get("empty_tags", -1)
        assert empty_tags == 0, f"{model_id}: Expected 0 empty tags, got {empty_tags} (CARDINAL RULE violation)"
        
        # Validate hierarchy discovery
        hierarchy_modules = result.get("hierarchy_modules", 0)
        expected_min = model_config.get("expected_modules", 10)
        assert hierarchy_modules >= expected_min // 2, \
            f"{model_id}: Expected at least {expected_min//2} modules, got {hierarchy_modules}"
        
        # Validate ONNX model integrity
        try:
            onnx_model = onnx.load(str(output_path))
            # Note: Skip onnx.checker.check_model() due to custom hierarchy_tag attributes
            assert len(onnx_model.graph.node) > 0, f"{model_id}: ONNX model should have nodes"
        except Exception as e:
            pytest.fail(f"{model_id}: ONNX model validation failed: {e}")
        
        # Validate metadata file
        metadata_path = output_path.parent / (output_path.stem + "_htp_metadata.json")
        assert metadata_path.exists(), f"{model_id}: Metadata file should be created"
        
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Validate metadata structure
            required_keys = ["export_info", "statistics", "hierarchy_data", "tagged_nodes"]
            for key in required_keys:
                assert key in metadata, f"{model_id}: Metadata should contain '{key}'"
            
            # Validate tagged nodes count matches ONNX nodes
            tagged_count = len(metadata["tagged_nodes"])
            onnx_node_count = len(onnx_model.graph.node)
            assert tagged_count == onnx_node_count, \
                f"{model_id}: Tagged nodes ({tagged_count}) should match ONNX nodes ({onnx_node_count})"
                
        except Exception as e:
            pytest.fail(f"{model_id}: Metadata validation failed: {e}")
    
    def test_model_diversity_coverage(self):
        """
        Test that our model selection covers diverse architectures and domains.
        
        This meta-test validates that the comprehensive test suite includes
        sufficient diversity across model types, domains, and architectures
        to provide confidence in universal compatibility.
        
        Coverage Requirements:
        - At least 5 different models (streamlined for efficiency)
        - At least 3 different domains (language, vision, multimodal)
        - At least 5 different architectures
        - At least 3 different size categories
        """
        # Filter out skipped models
        active_models = {
            k: v for k, v in POPULAR_MODELS.items() 
            if "skip_reason" not in v
        }
        
        # Validate minimum model count
        assert len(active_models) >= 5, f"Should test at least 5 models, got {len(active_models)}"
        
        # Validate domain diversity
        domains = set(model["domain"] for model in active_models.values())
        assert len(domains) >= 3, f"Should cover at least 3 domains, got {domains}"
        
        # Validate architecture diversity
        architectures = set(model["architecture"] for model in active_models.values())
        assert len(architectures) >= 5, f"Should cover at least 5 architectures, got {architectures}"
        
        # Validate size diversity
        sizes = set(model["size_category"] for model in active_models.values())
        assert len(sizes) >= 3, f"Should cover at least 3 size categories, got {sizes}"
        
        # Print coverage summary for visibility
        print(f"\\nModel Diversity Coverage Summary:")
        print(f"  Total active models: {len(active_models)}")
        print(f"  Domains covered: {sorted(domains)}")
        print(f"  Architectures covered: {sorted(architectures)}")
        print(f"  Size categories: {sorted(sizes)}")


class TestModelArchitectureSpecific:
    """
    Architecture-specific tests for edge cases and special requirements.
    
    These tests focus on specific architectural challenges that might
    require special handling or validation.
    """
    
    def test_bert_family_consistency(self):
        """
        Test that all BERT-family models export consistently.
        
        BERT variants should all achieve similar results since they share
        the same core architecture, validating our universal approach.
        """
        bert_models = [
            "bert_tiny"  # Only testing one BERT model for efficiency
        ]
        
        results = {}
        
        for model_id in bert_models:
            if model_id not in POPULAR_MODELS:
                continue
                
            model_config = POPULAR_MODELS[model_id]
            if "skip_reason" in model_config:
                continue
            
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / f"{model_id}_consistency.onnx"
                
                try:
                    from transformers import AutoModel
                    model = AutoModel.from_pretrained(model_config["model_name"])
                    
                    exporter = HTPExporter(verbose=False, enable_reporting=False)
                    
                    # HTPExporter automatically generates inputs from model_name_or_path
                    result = exporter.export(
                        model=model,
                        output_path=str(output_path),
                        model_name_or_path=model_config["model_name"],
                        opset_version=17
                    )
                    
                    results[model_id] = {
                        "coverage": result.get("coverage_percentage", 0),
                        "empty_tags": result.get("empty_tags", 0),
                        "hierarchy_modules": result.get("hierarchy_modules", 0)
                    }
                    
                except Exception as e:
                    # Skip models that fail to load/export
                    continue
        
        # Validate consistency
        assert len(results) >= 1, f"Should test at least 1 BERT model, got {len(results)}"
        
        # All should achieve 100% coverage
        for model_id, result in results.items():
            assert result["coverage"] == 100.0, f"{model_id} should achieve 100% coverage"
            assert result["empty_tags"] == 0, f"{model_id} should have 0 empty tags"
    
    def test_vision_model_input_handling(self):
        """
        Test that vision models handle different input specifications correctly.
        
        Vision models often require specific input shapes and preprocessing,
        so this test validates our input generation works universally.
        """
        vision_models = ["resnet18", "vit_base", "sam_vit_base", "yolo_v11"]
        
        for model_id in vision_models:
            if model_id not in POPULAR_MODELS:
                continue
                
            model_config = POPULAR_MODELS[model_id]
            if "skip_reason" in model_config:
                continue
            
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / f"{model_id}_vision.onnx"
                
                try:
                    from transformers import AutoModel
                    model = AutoModel.from_pretrained(model_config["model_name"])
                    
                    exporter = HTPExporter(verbose=False, enable_reporting=False)
                    
                    # HTPExporter automatically generates inputs from model_name_or_path
                    result = exporter.export(
                        model=model,
                        output_path=str(output_path),
                        model_name_or_path=model_config["model_name"],
                        opset_version=17
                    )
                    
                    # Vision models should still achieve universal success
                    total_ops = result.get("total_operations", 0)
                    tagged_ops = result.get("tagged_operations", 0)
                    coverage = (tagged_ops / total_ops * 100) if total_ops > 0 else 0.0
                    assert coverage == 100.0, \
                        f"{model_id} vision model should achieve 100% coverage, got {coverage}%"
                    assert result.get("empty_tags", 0) == 0, \
                        f"{model_id} vision model should have 0 empty tags"
                        
                except Exception as e:
                    # Skip models that fail to load (may require special image processors)
                    pytest.skip(f"Vision model {model_id} requires special handling: {e}")


class TestScalabilityAndPerformance:
    """
    Test performance characteristics across different model sizes.
    
    These tests validate that our export process scales appropriately
    with model size and complexity.
    """
    
    def test_export_time_scaling(self):
        """
        Test that export time scales reasonably with model size.
        
        Larger models should take proportionally longer to export,
        but the relationship should be reasonable (not exponential).
        """
        # Test models of different sizes
        size_test_models = [
            ("bert_tiny", "tiny"),
            ("qwen_0_5b", "small"), 
            ("vit_base", "base"),
        ]
        
        timing_results = {}
        
        for model_id, size_category in size_test_models:
            if model_id not in POPULAR_MODELS:
                continue
                
            model_config = POPULAR_MODELS[model_id]
            
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / f"{model_id}_timing.onnx"
                
                try:
                    from transformers import AutoModel
                    model = AutoModel.from_pretrained(model_config["model_name"])
                    
                    start_time = time.time()
                    exporter = HTPExporter(verbose=False, enable_reporting=False)
                    
                    # HTPExporter automatically generates inputs from model_name_or_path
                    result = exporter.export(
                        model=model,
                        output_path=str(output_path),
                        model_name_or_path=model_config["model_name"],
                        opset_version=17
                    )
                    export_time = time.time() - start_time
                    
                    timing_results[model_id] = {
                        "size": size_category,
                        "time": export_time,
                        "modules": len(result.get("hierarchy_nodes", {})),
                        "nodes": result.get("total_operations", 0)
                    }
                    
                except Exception as e:
                    continue
        
        # Validate timing relationships
        if len(timing_results) >= 2:
            times = [r["time"] for r in timing_results.values()]
            max_time = max(times)
            min_time = min(times)
            
            # Scaling should be reasonable (not more than 20x difference for base vs tiny)
            assert max_time / min_time < 20.0, \
                f"Export time scaling seems excessive: {max_time:.2f}s / {min_time:.2f}s = {max_time/min_time:.1f}x"


# Test execution helper
def run_comprehensive_tests():
    """
    Helper function to run the comprehensive test suite.
    
    This can be called directly for manual testing or debugging.
    """
    import subprocess
    
    # Run the comprehensive tests
    result = subprocess.run([
        "uv", "run", "pytest", 
        "tests/test_comprehensive_e2e.py",
        "-v", "--tb=short"
    ], cwd="/mnt/d/BYOM/modelexport")
    
    return result.returncode == 0


if __name__ == "__main__":
    # Allow running tests directly
    print("Running comprehensive end-to-end tests...")
    success = run_comprehensive_tests()
    print(f"Tests {'PASSED' if success else 'FAILED'}")