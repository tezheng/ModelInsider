"""
Round-Trip Validation Framework for GraphML v1.1

Validates the bidirectional conversion between ONNX and GraphML v1.1 formats
to ensure perfect model reconstruction capability.

Validation Levels:
- Structural: Graph topology and node/edge counts
- Functional: Model behavior with sample inputs  
- Numerical: Output value accuracy within tolerance
- Performance: Speed and memory usage metrics

Linear Task: TEZ-124
"""

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from .graphml_to_onnx_converter import GraphMLToONNXConverter
from .onnx_to_graphml_converter import ONNXToGraphMLConverter


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self):
        self.passed = False
        self.errors = []
        self.warnings = []
        self.metrics = {}
        self.details = {}
        
    def add_error(self, message: str, category: str = "general"):
        """Add validation error."""
        self.errors.append({"message": message, "category": category})
        self.passed = False
        
    def add_warning(self, message: str, category: str = "general"):
        """Add validation warning.""" 
        self.warnings.append({"message": message, "category": category})
        
    def add_metric(self, name: str, value: Any, unit: str = ""):
        """Add performance metric."""
        self.metrics[name] = {"value": value, "unit": unit}
        
    def add_detail(self, name: str, value: Any):
        """Add detailed information."""
        self.details[name] = value
        
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "passed": self.passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
            "details": self.details
        }


class RoundTripValidator:
    """Validates bidirectional ONNX ↔ GraphML conversion."""
    
    def __init__(
        self, 
        numerical_tolerance: float = 1e-6,
        parameter_strategy: str = "sidecar"
    ):
        """Initialize validator with configuration."""
        self.numerical_tolerance = numerical_tolerance
        self.parameter_strategy = parameter_strategy
        
        # Initialize converters
        self.graphml_converter = None  # Will be initialized per model
        self.onnx_converter = GraphMLToONNXConverter()
        
    def validate_round_trip(
        self, 
        original_onnx_path: str,
        htp_metadata_path: str,
        temp_dir: str | None = None
    ) -> ValidationResult:
        """
        Perform complete round-trip validation.
        
        Args:
            original_onnx_path: Path to original ONNX model
            htp_metadata_path: Path to HTP metadata  
            temp_dir: Temporary directory for intermediate files
            
        Returns:
            ValidationResult with comprehensive validation report
        """
        
        result = ValidationResult()
        
        # Setup temporary directory
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="roundtrip_")
        temp_path = Path(temp_dir)
        temp_path.mkdir(exist_ok=True)
        
        try:
            # Initialize GraphML converter
            self.graphml_converter = ONNXToGraphMLConverter(
                hierarchical=True,
                htp_metadata_path=htp_metadata_path,
                parameter_strategy=self.parameter_strategy
            )
            
            # Load original model
            original_model = onnx.load(original_onnx_path)
            result.add_detail("original_model_size", Path(original_onnx_path).stat().st_size)
            
            # Step 1: ONNX → GraphML v1.1
            print("🔄 Step 1: ONNX → GraphML v1.1...")
            start_time = time.time()
            
            graphml_files = self.graphml_converter.convert(
                onnx_model_path=original_onnx_path,
                output_base=str(temp_path / "model")
            )
            
            forward_time = time.time() - start_time
            result.add_metric("forward_conversion_time", forward_time, "seconds")
            
            graphml_path = graphml_files["graphml"]
            result.add_detail("graphml_path", graphml_path)
            result.add_detail("graphml_size", Path(graphml_path).stat().st_size)
            
            # Step 2: GraphML v1.1 → ONNX  
            print("🔄 Step 2: GraphML v1.1 → ONNX...")
            start_time = time.time()
            
            reconstructed_path = str(temp_path / "reconstructed.onnx")
            self.onnx_converter.convert(
                graphml_path=graphml_path,
                output_path=reconstructed_path
            )
            
            reverse_time = time.time() - start_time
            result.add_metric("reverse_conversion_time", reverse_time, "seconds")
            result.add_metric("total_conversion_time", forward_time + reverse_time, "seconds")
            
            # Load reconstructed model
            reconstructed_model = onnx.load(reconstructed_path)
            result.add_detail("reconstructed_model_size", Path(reconstructed_path).stat().st_size)
            
            # Step 3: Structural Validation
            print("🔍 Step 3: Structural validation...")
            self._validate_structure(original_model, reconstructed_model, result)
            
            # Step 4: Functional Validation
            print("🔍 Step 4: Functional validation...")
            self._validate_functionality(
                original_onnx_path, reconstructed_path, result
            )
            
            # Step 5: Numerical Validation
            print("🔍 Step 5: Numerical validation...")
            self._validate_numerical_accuracy(
                original_onnx_path, reconstructed_path, result
            )
            
            # Step 6: Performance Analysis
            print("📊 Step 6: Performance analysis...")
            self._analyze_performance(result, graphml_files)
            
            # Mark as passed if no errors
            if not result.errors:
                result.passed = True
                print("✅ Round-trip validation PASSED")
            else:
                print(f"❌ Round-trip validation FAILED with {len(result.errors)} errors")
                
        except Exception as e:
            result.add_error(f"Round-trip validation failed with exception: {e}", "exception")
            print(f"💥 Round-trip validation crashed: {e}")
            
        return result
    
    def _validate_structure(
        self, 
        original: onnx.ModelProto, 
        reconstructed: onnx.ModelProto, 
        result: ValidationResult
    ):
        """Validate structural equivalence."""
        
        # Graph structure
        orig_graph = original.graph
        recon_graph = reconstructed.graph
        
        # Node counts
        orig_nodes = len(orig_graph.node)
        recon_nodes = len(recon_graph.node)
        
        if orig_nodes != recon_nodes:
            result.add_error(
                f"Node count mismatch: original={orig_nodes}, reconstructed={recon_nodes}",
                "structure"
            )
        else:
            result.add_metric("node_count", orig_nodes, "nodes")
        
        # Input/output counts
        orig_inputs = len(orig_graph.input)
        recon_inputs = len(recon_graph.input)
        
        if orig_inputs != recon_inputs:
            result.add_error(
                f"Input count mismatch: original={orig_inputs}, reconstructed={recon_inputs}",
                "structure"
            )
        
        orig_outputs = len(orig_graph.output)
        recon_outputs = len(recon_graph.output)
        
        if orig_outputs != recon_outputs:
            result.add_error(
                f"Output count mismatch: original={orig_outputs}, reconstructed={recon_outputs}",
                "structure"
            )
        
        # Initializer counts
        orig_init = len(orig_graph.initializer)
        recon_init = len(recon_graph.initializer)
        
        if orig_init != recon_init:
            result.add_error(
                f"Initializer count mismatch: original={orig_init}, reconstructed={recon_init}",
                "structure"
            )
        else:
            result.add_metric("parameter_count", orig_init, "tensors")
        
        # Model metadata
        if original.model_version != reconstructed.model_version:
            result.add_warning(
                f"Model version mismatch: {original.model_version} vs {reconstructed.model_version}",
                "metadata"
            )
        
        # Opset versions
        orig_opsets = {(imp.domain, imp.version) for imp in original.opset_import}
        recon_opsets = {(imp.domain, imp.version) for imp in reconstructed.opset_import}
        
        if orig_opsets != recon_opsets:
            result.add_warning(
                f"Opset mismatch: original={orig_opsets}, reconstructed={recon_opsets}",
                "metadata"
            )
    
    def _validate_functionality(
        self, 
        original_path: str, 
        reconstructed_path: str,
        result: ValidationResult
    ):
        """Validate functional equivalence."""
        
        try:
            # Create inference sessions
            orig_session = ort.InferenceSession(original_path)
            recon_session = ort.InferenceSession(reconstructed_path)
            
            # Compare input/output specifications
            orig_inputs = orig_session.get_inputs()
            recon_inputs = recon_session.get_inputs()
            
            if len(orig_inputs) != len(recon_inputs):
                result.add_error(
                    f"Input count mismatch in runtime: {len(orig_inputs)} vs {len(recon_inputs)}",
                    "functionality"
                )
                return
            
            # Check input compatibility
            for orig_input, recon_input in zip(orig_inputs, recon_inputs, strict=False):
                if orig_input.name != recon_input.name:
                    result.add_warning(
                        f"Input name mismatch: {orig_input.name} vs {recon_input.name}",
                        "functionality"
                    )
                
                if orig_input.type != recon_input.type:
                    result.add_error(
                        f"Input type mismatch for {orig_input.name}: {orig_input.type} vs {recon_input.type}",
                        "functionality"
                    )
            
            # Check outputs
            orig_outputs = orig_session.get_outputs()
            recon_outputs = recon_session.get_outputs()
            
            if len(orig_outputs) != len(recon_outputs):
                result.add_error(
                    f"Output count mismatch in runtime: {len(orig_outputs)} vs {len(recon_outputs)}",
                    "functionality"
                )
                return
                
            result.add_metric("input_count", len(orig_inputs), "inputs")
            result.add_metric("output_count", len(orig_outputs), "outputs")
                
        except Exception as e:
            result.add_error(f"Functionality validation failed: {e}", "functionality")  
    
    def _validate_numerical_accuracy(
        self, 
        original_path: str, 
        reconstructed_path: str,
        result: ValidationResult
    ):
        """Validate numerical accuracy with sample inputs."""
        
        try:
            # Create inference sessions  
            orig_session = ort.InferenceSession(original_path)
            recon_session = ort.InferenceSession(reconstructed_path)
            
            # Generate sample inputs
            sample_inputs = self._generate_sample_inputs(orig_session)
            
            # Run inference on both models
            orig_outputs = orig_session.run(None, sample_inputs)
            recon_outputs = recon_session.run(None, sample_inputs)
            
            # Compare outputs
            max_diff = 0.0
            mean_diff = 0.0
            total_elements = 0
            
            for orig_out, recon_out in zip(orig_outputs, recon_outputs, strict=False):
                # Calculate differences
                diff = np.abs(orig_out - recon_out)
                max_diff = max(max_diff, np.max(diff))
                mean_diff += np.sum(diff)
                total_elements += diff.size
            
            mean_diff /= total_elements if total_elements > 0 else 1
            
            # Check tolerance
            if max_diff > self.numerical_tolerance:
                result.add_error(
                    f"Numerical accuracy failed: max_diff={max_diff:.2e} > tolerance={self.numerical_tolerance:.2e}",
                    "numerical"
                )
            
            result.add_metric("max_numerical_difference", max_diff, "absolute")
            result.add_metric("mean_numerical_difference", mean_diff, "absolute")
            result.add_metric("numerical_tolerance", self.numerical_tolerance, "absolute")
            
        except Exception as e:
            result.add_error(f"Numerical validation failed: {e}", "numerical")
    
    def _generate_sample_inputs(self, session: ort.InferenceSession) -> dict[str, np.ndarray]:
        """Generate sample inputs for testing."""
        
        inputs = {}
        
        for input_meta in session.get_inputs():
            name = input_meta.name
            shape = input_meta.shape
            dtype = input_meta.type
            
            # Handle dynamic shapes
            concrete_shape = []
            for dim in shape:
                if isinstance(dim, str) or dim < 0:
                    # Use reasonable default for dynamic dimensions
                    if "batch" in str(dim).lower():
                        concrete_shape.append(2)
                    elif "seq" in str(dim).lower():
                        concrete_shape.append(16)
                    else:
                        concrete_shape.append(10)
                else:
                    concrete_shape.append(dim)
            
            # Generate appropriate data based on type
            if dtype == 'tensor(int64)':
                # For input_ids, attention_mask, etc.
                inputs[name] = np.random.randint(0, 1000, concrete_shape, dtype=np.int64)
            elif dtype == 'tensor(float)':
                inputs[name] = np.random.randn(*concrete_shape).astype(np.float32)
            else:
                # Default to float32
                inputs[name] = np.random.randn(*concrete_shape).astype(np.float32)
        
        return inputs
    
    def _analyze_performance(
        self, 
        result: ValidationResult,
        graphml_files: dict[str, str]
    ):
        """Analyze performance metrics."""
        
        # File size analysis
        original_size = result.details.get("original_model_size", 0)
        reconstructed_size = result.details.get("reconstructed_model_size", 0)
        
        # Size comparison
        if original_size > 0:
            size_ratio = reconstructed_size / original_size
            result.add_metric("size_preservation_ratio", size_ratio, "ratio")
            
            if abs(size_ratio - 1.0) < 0.01:  # Within 1%
                result.add_detail("size_preservation", "excellent")
            elif abs(size_ratio - 1.0) < 0.05:  # Within 5%
                result.add_detail("size_preservation", "good")
            else:
                result.add_detail("size_preservation", "poor")
                result.add_warning(
                    f"Significant size difference: {size_ratio:.2%} of original",
                    "performance"
                )
        
        # Conversion speed analysis
        forward_time = result.metrics.get("forward_conversion_time", {}).get("value", 0)
        reverse_time = result.metrics.get("reverse_conversion_time", {}).get("value", 0)
        total_time = forward_time + reverse_time
        
        if total_time > 0:
            forward_speed = original_size / (forward_time * 1024 * 1024)  # MB/s
            reverse_speed = reconstructed_size / (reverse_time * 1024 * 1024)  # MB/s
            
            result.add_metric("forward_speed", forward_speed, "MB/s")
            result.add_metric("reverse_speed", reverse_speed, "MB/s")
        
        # Parameter file analysis
        if "parameters" in graphml_files:
            param_size = Path(graphml_files["parameters"]).stat().st_size
            result.add_metric("parameter_file_size", param_size, "bytes")
            result.add_metric("parameter_file_size_mb", param_size / (1024 * 1024), "MB")
    
    def generate_report(self, result: ValidationResult) -> str:
        """Generate human-readable validation report."""
        
        report = []
        report.append("=" * 80)
        report.append("ROUND-TRIP VALIDATION REPORT")
        report.append("=" * 80)
        
        # Overall result
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        report.append(f"Overall Status: {status}")
        report.append(f"Errors: {len(result.errors)}")
        report.append(f"Warnings: {len(result.warnings)}")
        report.append("")
        
        # Performance metrics
        if result.metrics:
            report.append("📊 PERFORMANCE METRICS")
            report.append("-" * 40)
            for name, metric in result.metrics.items():
                value = metric["value"]
                unit = metric["unit"]
                if isinstance(value, float):
                    report.append(f"{name}: {value:.4f} {unit}")
                else:
                    report.append(f"{name}: {value} {unit}")
            report.append("")
        
        # Errors
        if result.errors:
            report.append("❌ ERRORS")
            report.append("-" * 40)
            for error in result.errors:
                report.append(f"[{error['category'].upper()}] {error['message']}")
            report.append("")
        
        # Warnings
        if result.warnings:
            report.append("⚠️ WARNINGS")
            report.append("-" * 40)
            for warning in result.warnings:
                report.append(f"[{warning['category'].upper()}] {warning['message']}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)