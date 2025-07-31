"""
Auto-validation system using conditional JSON Schema features.

This module provides automatic model type detection and validation
using JSON Schema 2020-12's conditional validation features.
"""

from __future__ import annotations

from typing import Any


class ModelTypeDetector:
    """Detect model type and apply appropriate validation rules."""
    
    # Model patterns for auto-detection
    MODEL_PATTERNS = {
        "bert": {
            "class_pattern": r".*Bert.*",
            "expected_inputs": ["input_ids", "attention_mask"],
            "optional_inputs": ["token_type_ids", "position_ids"],
            "expected_outputs": ["last_hidden_state", "pooler_output"],
            "module_indicators": ["embeddings", "encoder", "pooler"],
        },
        "gpt": {
            "class_pattern": r".*GPT.*",
            "expected_inputs": ["input_ids"],
            "optional_inputs": ["attention_mask", "position_ids"],
            "expected_outputs": ["last_hidden_state"],
            "module_indicators": ["wte", "wpe", "h", "ln_f"],  # word/position embeddings, layers, final layernorm
        },
        "resnet": {
            "class_pattern": r".*ResNet.*",
            "expected_inputs": ["pixel_values"],
            "optional_inputs": [],
            "expected_outputs": ["last_hidden_state", "pooler_output"],
            "module_indicators": ["embedder", "encoder", "stages"],
        },
        "vit": {
            "class_pattern": r".*ViT.*",
            "expected_inputs": ["pixel_values"],
            "optional_inputs": ["head_mask"],
            "expected_outputs": ["last_hidden_state", "pooler_output"],
            "module_indicators": ["embeddings", "encoder", "layernorm"],
        },
        "t5": {
            "class_pattern": r".*T5.*",
            "expected_inputs": ["input_ids"],
            "optional_inputs": ["attention_mask", "decoder_input_ids"],
            "expected_outputs": ["last_hidden_state"],
            "module_indicators": ["encoder", "decoder"],
        }
    }
    
    @classmethod
    def detect_model_type(cls, metadata: dict[str, Any]) -> str | None:
        """
        Detect model type from metadata.
        
        Returns detected model type or None if unknown.
        """
        model_class = metadata.get("model", {}).get("class", "")
        modules = metadata.get("modules", {})
        inputs = metadata.get("tracing", {}).get("inputs", {})
        
        # Try class name pattern matching first
        import re
        for model_type, patterns in cls.MODEL_PATTERNS.items():
            if re.match(patterns["class_pattern"], model_class):
                return model_type
        
        # Fall back to module structure analysis
        module_names = set(modules.keys())
        for model_type, patterns in cls.MODEL_PATTERNS.items():
            indicators = patterns["module_indicators"]
            # Check if most indicators are present
            matches = sum(1 for ind in indicators if any(ind in name for name in module_names))
            if matches >= len(indicators) * 0.6:  # 60% match threshold
                return model_type
        
        # Final fallback: check input structure
        input_names = set(inputs.keys())
        for model_type, patterns in cls.MODEL_PATTERNS.items():
            expected = set(patterns["expected_inputs"])
            if expected.issubset(input_names):
                return model_type
        
        return None
    
    @classmethod
    def validate_model_specific_rules(
        cls,
        metadata: dict[str, Any],
        model_type: str
    ) -> tuple[bool, list[str]]:
        """
        Validate metadata against model-specific rules.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        patterns = cls.MODEL_PATTERNS.get(model_type)
        
        if not patterns:
            return False, [f"Unknown model type: {model_type}"]
        
        # Validate inputs
        actual_inputs = set(metadata.get("tracing", {}).get("inputs", {}).keys())
        expected_inputs = set(patterns["expected_inputs"])
        
        missing_inputs = expected_inputs - actual_inputs
        if missing_inputs:
            errors.append(f"Missing required inputs for {model_type}: {missing_inputs}")
        
        # Validate outputs
        actual_outputs = metadata.get("tracing", {}).get("outputs", [])
        if actual_outputs:
            expected_outputs = patterns["expected_outputs"]
            missing_outputs = set(expected_outputs) - set(actual_outputs)
            if missing_outputs:
                errors.append(f"Missing expected outputs for {model_type}: {missing_outputs}")
        
        # Validate module structure
        modules = metadata.get("modules", {})
        module_names = set(modules.keys())
        indicators = patterns["module_indicators"]
        
        missing_indicators = []
        for indicator in indicators:
            if not any(indicator in name for name in module_names):
                missing_indicators.append(indicator)
        
        if missing_indicators:
            errors.append(f"Missing expected module indicators for {model_type}: {missing_indicators}")
        
        # Model-specific validations
        if model_type == "bert":
            # BERT should have attention modules
            attention_modules = [m for m in module_names if "attention" in m.lower()]
            if not attention_modules:
                errors.append("BERT model missing attention modules")
        
        elif model_type == "resnet":
            # ResNet should have stages
            stages = [m for m in module_names if "stage" in m.lower()]
            if not stages:
                errors.append("ResNet model missing stage modules")
        
        return len(errors) == 0, errors


class AutoValidationReport:
    """Generate validation reports with auto-detected rules."""
    
    def __init__(self, metadata: dict[str, Any]):
        """Initialize with metadata."""
        self.metadata = metadata
        self.model_type = ModelTypeDetector.detect_model_type(metadata)
    
    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            "detected_model_type": self.model_type,
            "model_class": self.metadata.get("model", {}).get("class", "Unknown"),
            "validation_results": {},
            "recommendations": [],
            "quality_score": 0.0,
        }
        
        # Basic structure validation
        structure_valid, structure_errors = self._validate_structure()
        report["validation_results"]["structure"] = {
            "valid": structure_valid,
            "errors": structure_errors
        }
        
        # Model-specific validation
        if self.model_type:
            model_valid, model_errors = ModelTypeDetector.validate_model_specific_rules(
                self.metadata,
                self.model_type
            )
            report["validation_results"]["model_specific"] = {
                "valid": model_valid,
                "errors": model_errors
            }
        else:
            report["recommendations"].append(
                "Could not auto-detect model type. Consider adding model_type to metadata."
            )
        
        # Coverage validation
        coverage_valid, coverage_info = self._validate_coverage()
        report["validation_results"]["coverage"] = coverage_info
        
        # Calculate quality score
        total_checks = len(report["validation_results"])
        passed_checks = sum(
            1 for result in report["validation_results"].values()
            if result.get("valid", False)
        )
        report["quality_score"] = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        # Generate recommendations
        if report["quality_score"] < 100 and not coverage_valid:
            report["recommendations"].append(
                f"Coverage is {coverage_info.get('coverage', 0):.1f}%. "
                "Consider investigating untagged nodes."
            )
        
        return report
    
    def _validate_structure(self) -> tuple[bool, list[str]]:
        """Validate basic metadata structure."""
        errors = []
        required_sections = ["export_context", "model", "tracing", "modules", "tagging"]
        
        for section in required_sections:
            if section not in self.metadata:
                errors.append(f"Missing required section: {section}")
        
        return len(errors) == 0, errors
    
    def _validate_coverage(self) -> tuple[bool, dict[str, Any]]:
        """Validate tagging coverage."""
        coverage_data = self.metadata.get("tagging", {}).get("coverage", {})
        coverage_pct = coverage_data.get("coverage_percentage", 0)
        empty_tags = coverage_data.get("empty_tags", 0)
        
        info = {
            "valid": coverage_pct >= 95.0 and empty_tags == 0,
            "coverage": coverage_pct,
            "empty_tags": empty_tags,
        }
        
        if not info["valid"]:
            if coverage_pct < 95.0:
                info["message"] = f"Coverage {coverage_pct:.1f}% is below 95% threshold"
            if empty_tags > 0:
                info["message"] = f"{empty_tags} empty tags found"
        
        return info["valid"], info


# Integration with HTP exporter
def add_auto_validation_to_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Add auto-validation section to metadata.
    
    This could be called in the HTP exporter to enrich metadata.
    """
    detector = ModelTypeDetector()
    model_type = detector.detect_model_type(metadata)
    
    if model_type:
        # Add detected model type
        if "tracing" in metadata:
            metadata["tracing"]["detected_model_type"] = model_type
        
        # Add validation report
        report = AutoValidationReport(metadata).generate_report()
        metadata["validation"] = {
            "auto_detected_type": model_type,
            "quality_score": report["quality_score"],
            "checks_passed": report["validation_results"],
        }
    
    return metadata