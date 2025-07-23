#!/usr/bin/env python3
"""
Fix the empty report issue in metadata.

The report section should contain all the detailed information
that's logged to the console, not just completion status.
"""

def show_current_vs_expected_report():
    """Show what's currently in report vs what should be there."""
    print("Current Report Structure (Empty):")
    print("=" * 60)
    
    current = {
        "report": {
            "steps": {
                "hierarchy_building": {
                    "modules_traced": 76,
                    "execution_steps": 152,
                    "timestamp": "2025-07-19T09:56:51Z"
                },
                "onnx_export": {
                    "completed": True,  # This is too minimal!
                    "timestamp": "2025-07-19T09:56:52Z"
                }
            }
        }
    }
    
    print(current)
    
    print("\n\nExpected Report Structure (Full Details):")
    print("=" * 60)
    
    expected = {
        "report": {
            "steps": {
                "model_preparation": {
                    "model_class": "BertModel",
                    "total_modules": 48,
                    "total_parameters": 4385536,
                    "parameters_formatted": "4.4M",
                    "export_target": "model.onnx",
                    "strategy": "HTP (Hierarchy-Preserving)",
                    "embed_hierarchy_attributes": True,
                    "evaluation_mode": True,
                    "timestamp": "2025-07-19T09:56:51Z"
                },
                "input_generation": {
                    "model_name": "prajjwal1/bert-tiny",
                    "model_type": "bert",
                    "task": "feature-extraction",
                    "config_created": True,
                    "inputs_generated": {
                        "count": 3,
                        "tensors": {
                            "input_ids": {"shape": [2, 16], "dtype": "torch.int64"},
                            "attention_mask": {"shape": [2, 16], "dtype": "torch.int64"},
                            "token_type_ids": {"shape": [2, 16], "dtype": "torch.int64"}
                        }
                    },
                    "timestamp": "2025-07-19T09:56:51Z"
                },
                "hierarchy_building": {
                    "builder": "TracingHierarchyBuilder",
                    "modules_traced": 18,
                    "execution_steps": 36,
                    "hierarchy_depth": 7,
                    "module_tree": {
                        "root": "BertModel",
                        "total_modules": 18,
                        "structure": "hierarchical"  # Could include full tree
                    },
                    "timestamp": "2025-07-19T09:56:51Z"
                },
                "onnx_export": {
                    "target_file": "model.onnx",
                    "opset_version": 17,
                    "do_constant_folding": True,
                    "verbose": False,
                    "input_names": ["input_ids", "attention_mask", "token_type_ids"],
                    "output_names": ["last_hidden_state", "pooler_output"],
                    "dynamic_axes": None,
                    "export_successful": True,
                    "file_size_mb": 17.5,
                    "timestamp": "2025-07-19T09:56:52Z"
                },
                "node_tagger_creation": {
                    "tagger_created": True,
                    "model_root_tag": "/BertModel",
                    "operation_fallback": "disabled",
                    "timestamp": "2025-07-19T09:56:52Z"
                },
                "node_tagging": {
                    "total_nodes": 136,
                    "tagged_nodes": 136,
                    "coverage_percentage": 100.0,
                    "tagging_statistics": {
                        "direct_matches": 83,
                        "direct_percentage": 61.0,
                        "parent_matches": 34,
                        "parent_percentage": 25.0,
                        "root_fallbacks": 19,
                        "root_percentage": 14.0,
                        "empty_tags": 0
                    },
                    "top_nodes_by_hierarchy": [
                        {
                            "rank": 1,
                            "tag": "/BertModel/BertEncoder/BertLayer.0/BertAttention/BertSdpaSelfAttention",
                            "count": 35
                        },
                        {
                            "rank": 2,
                            "tag": "/BertModel/BertEncoder/BertLayer.1/BertAttention/BertSdpaSelfAttention",
                            "count": 35
                        }
                        # ... more top nodes
                    ],
                    "timestamp": "2025-07-19T09:56:53Z"
                },
                "model_save": {
                    "output_path": "model.onnx",
                    "hierarchy_attributes_embedded": True,
                    "file_saved": True,
                    "timestamp": "2025-07-19T09:56:53Z"
                },
                "export_complete": {
                    "export_time_seconds": 2.35,
                    "export_statistics": {
                        "hierarchy_modules": 18,
                        "onnx_nodes": 136,
                        "tagged_nodes": 136,
                        "coverage": 100.0
                    },
                    "output_files": {
                        "onnx_model": "model.onnx",
                        "metadata": "model_htp_metadata.json",
                        "report": "model_htp_export_report.txt",
                        "console_log": "model_console.log"
                    },
                    "timestamp": "2025-07-19T09:56:53Z"
                }
            }
        }
    }
    
    import json
    print(json.dumps(expected, indent=2))
    
    return current, expected


def create_fix_for_metadata_writer():
    """Create the fix for HTPMetadataWriter."""
    print("\n\nFix for HTPMetadataWriter:")
    print("=" * 60)
    
    fix_code = '''
class HTPMetadataWriter(StepAwareWriter):
    """JSON metadata writer for HTP export."""
    
    @step(HTPExportStep.MODEL_PREP)
    def write_model_prep(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Record detailed model preparation info."""
        # ... existing code ...
        
        # ADD: Full report details
        self.metadata["report"]["steps"]["model_preparation"] = {
            "model_class": data.model_class,
            "total_modules": data.total_modules,
            "total_parameters": data.total_parameters,
            "parameters_formatted": f"{data.total_parameters/1e6:.1f}M",
            "export_target": data.output_path,
            "strategy": f"{data.strategy} (Hierarchy-Preserving)",
            "embed_hierarchy_attributes": data.embed_hierarchy_attributes,
            "evaluation_mode": True,
            "timestamp": data.timestamp
        }
        return 1
    
    @step(HTPExportStep.INPUT_GEN)
    def write_input_gen(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Record detailed input generation info."""
        if "input_generation" in data.steps:
            step_data = data.steps["input_generation"]
            
            # ... existing code ...
            
            # ADD: Full report details
            self.metadata["report"]["steps"]["input_generation"] = {
                "model_name": data.model_name,
                "model_type": step_data.get("model_type", "unknown"),
                "task": step_data.get("task", "unknown"),
                "config_created": True,
                "inputs_generated": {
                    "count": len(step_data.get("inputs", {})),
                    "tensors": step_data.get("inputs", {})
                },
                "timestamp": data.timestamp
            }
        return 1
    
    @step(HTPExportStep.ONNX_EXPORT)
    def write_onnx_export(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Record detailed ONNX export info."""
        export_config = data.steps.get("onnx_export", {}).get("config", {})
        
        self.metadata["report"]["steps"]["onnx_export"] = {
            "target_file": data.output_path,
            "opset_version": export_config.get("opset_version", 17),
            "do_constant_folding": export_config.get("do_constant_folding", True),
            "verbose": export_config.get("verbose", False),
            "input_names": export_config.get("input_names", []),
            "output_names": data.output_names,
            "dynamic_axes": export_config.get("dynamic_axes"),
            "export_successful": True,
            "file_size_mb": data.onnx_size_mb,
            "timestamp": data.timestamp
        }
        return 1
    
    @step(HTPExportStep.NODE_TAGGING)
    def write_node_tagging(self, export_step: HTPExportStep, data: HTPExportData) -> int:
        """Record detailed tagging results."""
        stats = data.tagging_stats
        
        # Build top nodes list
        from collections import Counter
        tag_counts = Counter(data.tagged_nodes.values())
        top_nodes = [
            {
                "rank": i + 1,
                "tag": tag,
                "count": count
            }
            for i, (tag, count) in enumerate(tag_counts.most_common(20))
        ]
        
        self.metadata["report"]["steps"]["node_tagging"] = {
            "total_nodes": data.total_nodes,
            "tagged_nodes": len(data.tagged_nodes),
            "coverage_percentage": data.coverage,
            "tagging_statistics": {
                "direct_matches": stats.get("direct_matches", 0),
                "direct_percentage": round(stats.get("direct_matches", 0) / data.total_nodes * 100, 1),
                "parent_matches": stats.get("parent_matches", 0),
                "parent_percentage": round(stats.get("parent_matches", 0) / data.total_nodes * 100, 1),
                "root_fallbacks": stats.get("root_fallbacks", 0),
                "root_percentage": round(stats.get("root_fallbacks", 0) / data.total_nodes * 100, 1),
                "empty_tags": stats.get("empty_tags", 0)
            },
            "top_nodes_by_hierarchy": top_nodes,
            "timestamp": data.timestamp
        }
        return 1
'''
    
    print(fix_code)
    return fix_code


def show_report_purpose():
    """Explain the purpose of the report section."""
    print("\n\nPurpose of Report Section:")
    print("=" * 60)
    
    print("""
The 'report' section in metadata should serve as a complete record of the export process,
containing ALL the information that was displayed in the console output, but in structured JSON format.

This allows users to:
1. Review what happened during export without parsing console logs
2. Access all statistics and metrics programmatically
3. Debug issues by seeing the complete export flow
4. Compare exports by diffing metadata files
5. Generate documentation or analysis from the structured data

Current issue: Only storing completion status instead of full details.
Solution: Capture all console output data in the report section.
""")


def main():
    """Analyze and fix the empty report issue."""
    print("Metadata Report Section Analysis")
    print("=" * 80)
    
    # Show current vs expected
    current, expected = show_current_vs_expected_report()
    
    # Create fix
    fix = create_fix_for_metadata_writer()
    
    # Explain purpose
    show_report_purpose()
    
    print("\n\nSummary:")
    print("=" * 60)
    print("The report section should mirror ALL console output in JSON format.")
    print("Currently it only stores minimal completion status.")
    print("Need to update each step handler to store full details.")


if __name__ == "__main__":
    main()