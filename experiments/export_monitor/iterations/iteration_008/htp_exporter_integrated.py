"""
HTP Exporter - Iteration 8: Integrated with Export Monitor.
All logging and metadata generation delegated to export monitor.
"""

import time
import torch
import onnx
from pathlib import Path
from typing import Dict, Any, Optional

# Import from relative paths matching the original structure
from ...core.onnx_node_tagger import create_node_tagger_from_hierarchy
from ...core.onnx_utils import infer_output_names
from ...core.tracing_hierarchy_builder import TracingHierarchyBuilder

# Import the rich export monitor
from .export_monitor import HTPExportMonitor


class HTPExporter:
    """HTP (Hierarchy-preserving Tags Protocol) Exporter using Export Monitor."""
    
    def __init__(self, verbose: bool = True, embed_hierarchy_attributes: bool = True):
        """Initialize HTP exporter.
        
        Args:
            verbose: Whether to output progress to console
            embed_hierarchy_attributes: Whether to embed hierarchy tags as ONNX attributes
        """
        self.verbose = verbose
        self.embed_hierarchy_attributes = embed_hierarchy_attributes
        self.monitor = None
        
    def export(self, model: torch.nn.Module, output_path: str, 
               dummy_input: Optional[Dict[str, torch.Tensor]] = None,
               export_params: Optional[Dict[str, Any]] = None) -> str:
        """Export model using HTP strategy with export monitor.
        
        Args:
            model: PyTorch model to export
            output_path: Path for the exported ONNX file
            dummy_input: Optional dummy input for tracing
            export_params: Optional export parameters
            
        Returns:
            Path to the exported ONNX file
        """
        start_time = time.time()
        
        # Initialize export monitor
        model_name = export_params.get("model_name", "Unknown") if export_params else "Unknown"
        enable_report = export_params.get("enable_report", True) if export_params else True
        
        self.monitor = HTPExportMonitor(
            output_path=output_path,
            model_name=model_name,
            verbose=self.verbose,
            enable_report=enable_report
        )
        
        # Step 1: Model preparation
        model_class = model.__class__.__name__
        total_modules = sum(1 for _ in model.named_modules())
        total_parameters = sum(p.numel() for p in model.parameters())
        
        model.eval()
        
        self.monitor.model_preparation(
            model_class=model_class,
            total_modules=total_modules,
            total_parameters=total_parameters,
            embed_hierarchy_attributes=self.embed_hierarchy_attributes
        )
        
        # Step 2: Input generation
        if dummy_input is None:
            model_config = get_model_config(model_name) if model_name != "Unknown" else None
            
            if model_config:
                model_type = model_config.model_type.value
                task = model_config.default_task
            else:
                model_type = "unknown"
                task = "unknown"
            
            # Generate dummy input
            dummy_input = self._generate_dummy_input(model, model_name)
            
            # Convert to format for monitor
            input_info = {}
            for name, tensor in dummy_input.items():
                input_info[name] = {
                    "shape": str(tensor.shape),
                    "dtype": str(tensor.dtype)
                }
            
            self.monitor.input_generation(
                model_type=model_type,
                task=task,
                inputs=input_info
            )
        else:
            # Use provided input
            input_info = {}
            for name, tensor in dummy_input.items():
                input_info[name] = {
                    "shape": str(tensor.shape),
                    "dtype": str(tensor.dtype)
                }
            
            self.monitor.input_generation(
                model_type="custom",
                task="custom",
                inputs=input_info
            )
        
        # Step 3: Hierarchy building
        hierarchy_builder = TracingHierarchyBuilder(verbose=False)
        hierarchy = hierarchy_builder.build_hierarchy(model, dummy_input)
        execution_steps = len(hierarchy_builder.execution_order)
        
        self.monitor.hierarchy_building(
            hierarchy=hierarchy,
            execution_steps=execution_steps
        )
        
        # Step 4: ONNX export
        input_names = list(dummy_input.keys())
        output_names = ['output']
        
        # Default export parameters
        opset_version = export_params.get("opset_version", 17) if export_params else 17
        do_constant_folding = export_params.get("do_constant_folding", True) if export_params else True
        
        self.monitor.onnx_export(
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            input_names=input_names
        )
        
        # Perform actual export
        torch.onnx.export(
            model,
            tuple(dummy_input.values()),
            output_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            verbose=False
        )
        
        # Step 5: Tagger creation
        enable_operation_fallback = export_params.get("enable_operation_fallback", False) if export_params else False
        
        self.monitor.tagger_creation(
            enable_operation_fallback=enable_operation_fallback
        )
        
        tagger = OnnxNodeTagger(
            hierarchy=hierarchy,
            execution_order=hierarchy_builder.execution_order,
            enable_operation_fallback=enable_operation_fallback,
            verbose=False
        )
        
        # Step 6: Node tagging
        onnx_model = onnx.load(output_path)
        tagged_nodes = tagger.tag_onnx_nodes(onnx_model)
        
        self.monitor.node_tagging(
            total_nodes=tagger.total_nodes,
            tagged_nodes=tagged_nodes,
            statistics=tagger.get_statistics()
        )
        
        # Step 7: Tag injection (if enabled)
        self.monitor.tag_injection()
        
        if self.embed_hierarchy_attributes:
            inject_hierarchy_tags(onnx_model, tagged_nodes)
            onnx.save(onnx_model, output_path)
        
        # Update monitor state with output names
        self.monitor.state.output_names = output_names
        
        # Step 8: Metadata generation
        self.monitor.metadata_generation()
        
        # Complete
        export_time = time.time() - start_time
        self.monitor.complete(export_time=export_time)
        
        return output_path
    
    def _generate_dummy_input(self, model: torch.nn.Module, model_name: str) -> Dict[str, torch.Tensor]:
        """Generate dummy input for the model."""
        # Simple dummy input generation
        batch_size = 1
        sequence_length = 128
        
        # For BERT-like models
        if any(name in model_name.lower() for name in ['bert', 'roberta', 'distilbert']):
            return {
                'input_ids': torch.randint(0, 1000, (batch_size, sequence_length)),
                'attention_mask': torch.ones(batch_size, sequence_length, dtype=torch.long)
            }
        
        # For vision models
        elif any(name in model_name.lower() for name in ['resnet', 'vit', 'mobilenet']):
            return {
                'pixel_values': torch.randn(batch_size, 3, 224, 224)
            }
        
        # Default
        else:
            return {
                'input': torch.randn(batch_size, sequence_length)
            }
