# HTP Metadata Template with Descriptions

This document shows the complete JSON metadata structure with descriptions for each field.

```json
{
  "export_context": {
    "timestamp": "ISO 8601 timestamp of export",
    "strategy": "htp (always)",
    "version": "1.0 (metadata format version)",
    "exporter": "HTPExporter (exporter class name)",
    "embed_hierarchy_attributes": "boolean - whether tags are embedded in ONNX",
    "export_time_seconds": "total export time as float"
  },
  
  "model": {
    "name_or_path": "HuggingFace model name or local path",
    "class": "PyTorch model class name (e.g., BertModel)",
    "framework": "transformers (usually)",
    "total_modules": "number of nn.Module instances in model",
    "total_parameters": "total parameter count"
  },
  
  "modules": {
    "module.path": {
      "class_name": "Python class name of the module",
      "traced_tag": "/Hierarchical/Tag/Path assigned to module",
      "execution_order": "integer showing when module was executed during tracing"
    }
    // ... more modules
  },
  
  "nodes": {
    "onnx_node_name": "/Hierarchical/Tag/Path of source module"
    // ... mapping for all ONNX nodes
  },
  
  "outputs": {
    "onnx_model": {
      "path": "filename.onnx",
      "size_mb": "file size in megabytes",
      "opset_version": "ONNX opset version (e.g., 17)",
      "output_names": ["list", "of", "output", "tensor", "names"]
    },
    "metadata": {
      "path": "filename_htp_metadata.json"
    },
    "report": {
      "path": "filename_htp_export_report.txt (if enabled)"
    }
  },
  
  "report": {
    "steps": {
      "input_generation": {
        "method": "auto_generated or provided",
        "model_type": "bert, gpt2, etc.",
        "task": "feature-extraction, text-generation, etc.",
        "inputs": {
          "input_name": {
            "shape": [batch, sequence_length],
            "dtype": "torch.int64 or torch.float32"
          }
        }
      },
      "onnx_export": {
        "opset_version": "ONNX opset version",
        "do_constant_folding": "boolean",
        "onnx_size_mb": "exported file size",
        "output_names": ["output", "tensor", "names"]
      },
      "tag_injection": {
        "tags_injected": "boolean - were tags added to ONNX",
        "tags_stripped": "boolean - were tags removed"
      }
    },
    "node_tagging": {
      "statistics": {
        "direct_matches": "nodes matched directly to executed modules",
        "parent_matches": "nodes matched to parent modules",
        "root_fallbacks": "nodes that fell back to root module",
        "tagged_nodes": "total nodes that received tags"
      },
      "coverage": {
        "percentage": "tag coverage as percentage (always 100.0)",
        "total_onnx_nodes": "total ONNX nodes in graph",
        "tagged_nodes": "nodes with hierarchy tags"
      }
    }
  },
  
  "tracing": {
    "builder": "TracingHierarchyBuilder (always)",
    "modules_traced": "number of modules discovered",
    "execution_steps": "number of execution steps during tracing",
    "model_type": "detected model type",
    "task": "detected or specified task",
    "inputs": {
      "input_name": {
        "shape": [dimensions],
        "dtype": "data type"
      }
    },
    "outputs": ["list", "of", "output", "names"]
  },
  
  "statistics": {
    "export_time": "total time in seconds",
    "hierarchy_modules": "modules in hierarchy",
    "onnx_nodes": "total ONNX nodes",
    "tagged_nodes": "nodes with tags",
    "empty_tags": "always 0 (validation metric)",
    "coverage_percentage": "always 100.0",
    "module_types": ["unique", "module", "class", "names"]
  }
}
```

## Key Points:

1. **Required sections** (per schema): `export_context`, `model`, `modules`, `nodes`
2. **Optional sections**: `outputs`, `report`, `tracing`, `statistics`
3. **No duplication**: Each piece of data appears only once
4. **Clean hierarchy**: 
   - Data at root level (modules, nodes)
   - Metadata in appropriate sections (report, statistics)
5. **Validation**: The schema ensures data integrity and proper types