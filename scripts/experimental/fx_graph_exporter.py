#!/usr/bin/env python3
"""
FX Graph Export Module (dynamo=False)

This module implements FX graph export functionality without using TorchDynamo,
providing an alternative approach to capture model execution graphs.
"""

import torch
import torch.fx as fx
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path


class FXGraphExporter:
    """
    Exports models to FX graph representation without using TorchDynamo.
    
    This provides an alternative to ONNX export that preserves more context
    and is more suitable for analysis and debugging.
    """
    
    def __init__(self):
        self.trace_info = {}
        self.export_info = {}
    
    def export_fx_graph(
        self,
        model: torch.nn.Module,
        example_inputs: Union[torch.Tensor, Tuple, Dict],
        output_path: str,
        method: str = "symbolic_trace",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Export model to FX graph format.
        
        Args:
            model: PyTorch model to export
            example_inputs: Example inputs for tracing
            output_path: Path to save the FX graph
            method: Export method ('symbolic_trace', 'torch_export', 'both')
            **kwargs: Additional arguments for the export method
            
        Returns:
            Dictionary containing export information and results
        """
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            "model_name": model.__class__.__name__,
            "export_method": method,
            "export_timestamp": str(torch.random.initial_seed()),
            "success": False,
            "error": None,
            "graph_info": {},
            "node_analysis": {}
        }
        
        try:
            if method == "symbolic_trace":
                results.update(self._export_symbolic_trace(model, example_inputs, output_path, **kwargs))
            elif method == "torch_export":
                results.update(self._export_torch_export(model, example_inputs, output_path, **kwargs))
            elif method == "both":
                # Try both methods
                symbolic_results = self._export_symbolic_trace(model, example_inputs, 
                                                             output_path.with_suffix('.symbolic.json'), **kwargs)
                export_results = self._export_torch_export(model, example_inputs,
                                                          output_path.with_suffix('.export.json'), **kwargs)
                
                results.update({
                    "symbolic_trace_results": symbolic_results,
                    "torch_export_results": export_results,
                    "success": symbolic_results.get("success", False) or export_results.get("success", False)
                })
            else:
                raise ValueError(f"Unknown export method: {method}")
                
        except Exception as e:
            results["error"] = str(e)
            results["success"] = False
        
        # Save results
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ FX graph export {'successful' if results['success'] else 'failed'}")
        if results["success"]:
            print(f"📁 Results saved to: {output_path.with_suffix('.json')}")
        
        return results
    
    def _export_symbolic_trace(
        self, 
        model: torch.nn.Module, 
        example_inputs: Any, 
        output_path: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """Export using torch.fx.symbolic_trace."""
        
        results = {
            "method": "symbolic_trace",
            "success": False,
            "error": None
        }
        
        try:
            print("🔄 Attempting symbolic tracing...")
            
            # Prepare the model
            model.eval()
            
            # Create a wrapper to handle dict inputs for FX tracing
            class FXTraceWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, input_ids, attention_mask):
                    return self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Use wrapper if we have dict inputs
            if isinstance(example_inputs, dict):
                wrapper_model = FXTraceWrapper(model)
                traced_graph = fx.symbolic_trace(wrapper_model)
            else:
                traced_graph = fx.symbolic_trace(model)
            
            print("✅ Symbolic tracing successful!")
            results["success"] = True
            
            # Analyze the FX graph
            graph_analysis = self._analyze_fx_graph(traced_graph.graph)
            results.update(graph_analysis)
            
            # Save the graph code
            graph_code = traced_graph.code
            code_output_path = output_path.with_suffix('.py')
            with open(code_output_path, 'w') as f:
                f.write(graph_code)
            
            results["graph_code_path"] = str(code_output_path)
            results["graph_code_preview"] = graph_code[:500] + "..." if len(graph_code) > 500 else graph_code
            
            # Try to execute the traced graph
            try:
                with torch.no_grad():
                    if isinstance(example_inputs, dict):
                        # Handle dict inputs (from tokenizer)
                        traced_output = traced_graph(**example_inputs)
                    elif isinstance(example_inputs, (tuple, list)):
                        traced_output = traced_graph(*example_inputs)
                    else:
                        traced_output = traced_graph(example_inputs)
                
                results["execution_test"] = "successful"
                results["output_type"] = str(type(traced_output))
                
            except Exception as e:
                results["execution_test"] = f"failed: {e}"
            
        except Exception as e:
            results["error"] = str(e)
            results["success"] = False
            print(f"❌ Symbolic tracing failed: {e}")
        
        return results
    
    def _export_torch_export(
        self, 
        model: torch.nn.Module, 
        example_inputs: Any, 
        output_path: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """Export using torch.export (PyTorch 2.0+)."""
        
        results = {
            "method": "torch_export",
            "success": False,
            "error": None
        }
        
        try:
            print("🔄 Attempting torch.export...")
            
            # Check if torch.export is available
            if not hasattr(torch, 'export'):
                results["error"] = "torch.export not available in this PyTorch version"
                return results
            
            # Prepare inputs for export
            if isinstance(example_inputs, dict):
                # Convert dict to args for export
                args = tuple(example_inputs.values())
            elif isinstance(example_inputs, (tuple, list)):
                args = tuple(example_inputs)
            else:
                args = (example_inputs,)
            
            # Export the program
            exported_program = torch.export.export(model, args)
            
            print("✅ torch.export successful!")
            results["success"] = True
            
            # Analyze the exported program
            export_analysis = self._analyze_exported_program(exported_program)
            results.update(export_analysis)
            
            # Save the exported program graph code
            graph_module = exported_program.module
            graph_code = str(graph_module.graph)
            
            code_output_path = output_path.with_suffix('.py')
            with open(code_output_path, 'w') as f:
                f.write(f"# Exported Program Graph\n\n{graph_code}")
            
            results["graph_code_path"] = str(code_output_path)
            results["graph_code_preview"] = graph_code[:500] + "..." if len(graph_code) > 500 else graph_code
            
            # Test execution
            try:
                with torch.no_grad():
                    exported_output = exported_program(*args)
                
                results["execution_test"] = "successful"
                results["output_type"] = str(type(exported_output))
                
            except Exception as e:
                results["execution_test"] = f"failed: {e}"
        
        except Exception as e:
            results["error"] = str(e)
            results["success"] = False
            print(f"❌ torch.export failed: {e}")
        
        return results
    
    def _analyze_fx_graph(self, graph: fx.Graph) -> Dict[str, Any]:
        """Analyze FX graph structure and extract information."""
        
        nodes = list(graph.nodes)
        
        analysis = {
            "total_nodes": len(nodes),
            "node_types": {},
            "node_targets": {},
            "nodes_with_meta": 0,
            "sample_nodes": []
        }
        
        for i, node in enumerate(nodes):
            # Count node types
            op_type = node.op
            analysis["node_types"][op_type] = analysis["node_types"].get(op_type, 0) + 1
            
            # Count targets
            target = str(node.target)
            analysis["node_targets"][target] = analysis["node_targets"].get(target, 0) + 1
            
            # Check for metadata
            if hasattr(node, 'meta') and node.meta:
                analysis["nodes_with_meta"] += 1
            
            # Sample first 10 nodes
            if i < 10:
                node_info = {
                    "index": i,
                    "op": node.op,
                    "name": node.name,
                    "target": str(node.target),
                    "args_count": len(node.args),
                    "kwargs_count": len(node.kwargs),
                    "has_meta": hasattr(node, 'meta') and bool(node.meta)
                }
                
                if node_info["has_meta"]:
                    node_info["meta_keys"] = list(node.meta.keys())
                
                analysis["sample_nodes"].append(node_info)
        
        return {"fx_graph_analysis": analysis}
    
    def _analyze_exported_program(self, exported_program) -> Dict[str, Any]:
        """Analyze torch.export ExportedProgram."""
        
        analysis = {
            "export_program_analysis": {
                "has_graph_module": hasattr(exported_program, 'module'),
                "has_graph_signature": hasattr(exported_program, 'graph_signature'),
                "has_call_spec": hasattr(exported_program, 'call_spec'),
            }
        }
        
        # Analyze the graph module
        if hasattr(exported_program, 'module'):
            graph_module = exported_program.module
            if hasattr(graph_module, 'graph'):
                fx_analysis = self._analyze_fx_graph(graph_module.graph)
                analysis["export_program_analysis"].update(fx_analysis)
        
        # Analyze graph signature
        if hasattr(exported_program, 'graph_signature'):
            signature = exported_program.graph_signature
            analysis["export_program_analysis"]["graph_signature"] = {
                "input_specs": len(signature.input_specs) if hasattr(signature, 'input_specs') else 0,
                "output_specs": len(signature.output_specs) if hasattr(signature, 'output_specs') else 0,
            }
        
        return analysis


def export_fx_graph_cli(
    model: torch.nn.Module,
    example_inputs: Any,
    output_path: str,
    method: str = "both"
) -> Dict[str, Any]:
    """
    CLI interface for FX graph export.
    
    Args:
        model: PyTorch model to export
        example_inputs: Example inputs for tracing
        output_path: Path to save the FX graph
        method: Export method ('symbolic_trace', 'torch_export', 'both')
        
    Returns:
        Export results dictionary
    """
    
    exporter = FXGraphExporter()
    return exporter.export_fx_graph(model, example_inputs, output_path, method=method)


if __name__ == "__main__":
    # Example usage
    from transformers import AutoModel, AutoTokenizer
    
    # Load test model
    model_name = "prajjwal1/bert-tiny"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare inputs
    inputs = tokenizer("Hello world", return_tensors="pt")
    
    # Export FX graph
    results = export_fx_graph_cli(
        model, 
        inputs, 
        "temp/fx_graph_export",
        method="both"
    )
    
    print("\n🎯 FX EXPORT SUMMARY:")
    print(f"Export successful: {results['success']}")
    if results.get("error"):
        print(f"Error: {results['error']}")
    
    if results['success']:
        print("✅ Successfully exported FX graph representation!")