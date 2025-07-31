"""
Universal auto-validation system without hardcoded model logic.

This module provides automatic validation based on metadata structure
without any model-specific hardcoded patterns.
"""

from __future__ import annotations

from typing import Any


class UniversalModelValidator:
    """Universal validation without hardcoded model patterns."""
    
    @classmethod
    def analyze_model_structure(cls, metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze model structure from metadata universally.
        
        Returns analysis results without assuming specific architectures.
        """
        modules = metadata.get("modules", {})
        inputs = metadata.get("tracing", {}).get("inputs", {})
        outputs = metadata.get("tracing", {}).get("outputs", [])
        
        # Analyze module hierarchy depth
        max_depth = 0
        for module_name in modules:
            depth = module_name.count(".")
            max_depth = max(max_depth, depth)
        
        # Analyze module types present
        module_types = set()
        for module_info in modules.values():
            class_name = module_info.get("class_name", "")
            module_types.add(class_name)
        
        # Analyze input/output structure
        input_types = []
        for input_name, input_info in inputs.items():
            shape = input_info.get("shape", [])
            dtype = input_info.get("dtype", "")
            input_types.append({
                "name": input_name,
                "dimensions": len(shape),
                "dtype": dtype
            })
        
        return {
            "structure_analysis": {
                "hierarchy_depth": max_depth,
                "unique_module_types": len(module_types),
                "total_modules": len(modules),
                "input_count": len(inputs),
                "output_count": len(outputs),
                "module_type_distribution": cls._get_type_distribution(modules),
                "input_characteristics": input_types
            }
        }
    
    @classmethod
    def _get_type_distribution(cls, modules: dict[str, dict[str, Any]]) -> dict[str, int]:
        """Get distribution of module types."""
        distribution = {}
        for module_info in modules.values():
            class_name = module_info.get("class_name", "Unknown")
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution
    
    @classmethod
    def validate_universal_rules(
        cls,
        metadata: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """
        Validate metadata against universal rules that apply to all models.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Universal structure validation
        required_sections = ["export_context", "model", "tracing", "modules", "tagging"]
        for section in required_sections:
            if section not in metadata:
                errors.append(f"Missing required section: {section}")
        
        # Validate model has basic info
        model_info = metadata.get("model", {})
        if not model_info.get("class"):
            errors.append("Model class information is missing")
        
        if not model_info.get("name_or_path"):
            errors.append("Model name or path is missing")
        
        # Validate tracing has inputs
        tracing = metadata.get("tracing", {})
        if not tracing.get("inputs"):
            errors.append("No input tensors found in tracing")
        
        # Validate modules exist
        modules = metadata.get("modules", {})
        if not modules:
            errors.append("No modules found in hierarchy")
        
        # Validate each module has required fields
        for module_name, module_info in modules.items():
            if not module_info.get("class_name"):
                errors.append(f"Module {module_name} missing class_name")
            if "traced_tag" not in module_info:
                errors.append(f"Module {module_name} missing traced_tag")
        
        # Validate tagging coverage
        tagging = metadata.get("tagging", {})
        coverage = tagging.get("coverage", {})
        
        if "coverage_percentage" not in coverage:
            errors.append("Coverage percentage not calculated")
        else:
            # Validate coverage calculation
            total_nodes = coverage.get("total_onnx_nodes", 0)
            tagged_nodes = coverage.get("tagged_nodes", 0)
            reported_coverage = coverage.get("coverage_percentage", 0)
            
            if total_nodes > 0:
                calculated_coverage = (tagged_nodes / total_nodes) * 100
                if abs(calculated_coverage - reported_coverage) > 0.1:
                    errors.append(
                        f"Coverage calculation mismatch: "
                        f"reported {reported_coverage:.1f}% vs calculated {calculated_coverage:.1f}%"
                    )
        
        return len(errors) == 0, errors


class UniversalValidationReport:
    """Generate validation reports based on universal patterns."""
    
    def __init__(self, metadata: dict[str, Any]):
        """Initialize with metadata."""
        self.metadata = metadata
        self.structure_analysis = UniversalModelValidator.analyze_model_structure(metadata)
    
    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive validation report using universal patterns."""
        report = {
            "model_class": self.metadata.get("model", {}).get("class", "Unknown"),
            "structure_analysis": self.structure_analysis["structure_analysis"],
            "validation_results": {},
            "recommendations": [],
            "quality_score": 0.0,
        }
        
        # Universal validation
        valid, errors = UniversalModelValidator.validate_universal_rules(self.metadata)
        report["validation_results"]["universal"] = {
            "valid": valid,
            "errors": errors
        }
        
        # Coverage validation
        coverage_valid, coverage_info = self._validate_coverage()
        report["validation_results"]["coverage"] = coverage_info
        
        # Hierarchy validation
        hierarchy_valid, hierarchy_info = self._validate_hierarchy()
        report["validation_results"]["hierarchy"] = hierarchy_info
        
        # Calculate quality score
        total_checks = len(report["validation_results"])
        passed_checks = sum(
            1 for result in report["validation_results"].values()
            if result.get("valid", False)
        )
        report["quality_score"] = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        # Generate recommendations based on analysis
        self._generate_recommendations(report)
        
        return report
    
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
    
    def _validate_hierarchy(self) -> tuple[bool, dict[str, Any]]:
        """Validate module hierarchy consistency."""
        modules = self.metadata.get("modules", {})
        tagged_nodes = self.metadata.get("tagging", {}).get("tagged_nodes", {})
        
        issues = []
        
        # Check for modules without any tagged nodes
        module_tags = {info.get("traced_tag") for info in modules.values() if info.get("traced_tag")}
        used_tags = set(tagged_nodes.values())
        
        for module_name, module_info in modules.items():
            module_tag = module_info.get("traced_tag")
            if module_tag and module_tag not in used_tags:
                # Check if any child modules have tags
                has_child_tags = any(
                    tag.startswith(module_tag + "/") for tag in used_tags
                )
                if not has_child_tags:
                    issues.append(f"Module {module_name} has no tagged operations")
        
        info = {
            "valid": len(issues) == 0,
            "issues": issues[:5],  # Limit to first 5 issues
            "total_issues": len(issues)
        }
        
        return info["valid"], info
    
    def _generate_recommendations(self, report: dict[str, Any]) -> None:
        """Generate recommendations based on universal patterns."""
        structure = self.structure_analysis["structure_analysis"]
        
        # Hierarchy depth recommendation
        if structure["hierarchy_depth"] > 10:
            report["recommendations"].append(
                f"Model has deep hierarchy (depth={structure['hierarchy_depth']}). "
                "Consider reviewing deeply nested modules for optimization."
            )
        
        # Coverage recommendations
        coverage_info = report["validation_results"]["coverage"]
        if not coverage_info["valid"]:
            report["recommendations"].append(
                f"Coverage is {coverage_info['coverage']:.1f}%. "
                "Review untagged nodes to improve hierarchy tracking."
            )
        
        # Module diversity recommendation
        if structure["unique_module_types"] < 5:
            report["recommendations"].append(
                "Low module type diversity detected. "
                "Ensure all model components are properly traced."
            )
        
        # Input validation recommendation
        if structure["input_count"] == 0:
            report["recommendations"].append(
                "No inputs detected. Verify model tracing was successful."
            )


# Integration function
def add_universal_validation_to_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Add universal validation section to metadata.
    
    This can be called in the HTP exporter to enrich metadata.
    """
    # Add structure analysis
    analysis = UniversalModelValidator.analyze_model_structure(metadata)
    
    # Add validation report
    report = UniversalValidationReport(metadata).generate_report()
    
    metadata["validation"] = {
        "structure_analysis": analysis["structure_analysis"],
        "quality_score": report["quality_score"],
        "checks_passed": report["validation_results"],
    }
    
    return metadata