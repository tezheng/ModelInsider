"""
Conditional schema validation for different model types.

This module demonstrates how to use JSON Schema 2020-12's conditional
validation features for model-specific metadata validation.
"""

from __future__ import annotations

from typing import Any


def create_conditional_htp_schema() -> dict[str, Any]:
    """
    Create a conditional schema that validates differently based on model type.
    
    This schema uses if-then-else and unevaluatedProperties to ensure
    model-specific fields are properly validated.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://modelexport/schemas/htp-conditional/v2.0",
        "title": "HTP Metadata with Conditional Validation",
        "description": "Validates metadata differently based on model architecture",
        
        "type": "object",
        "required": ["export_context", "model", "tracing", "modules", "tagging"],
        
        # Base properties that all models must have
        "properties": {
            "export_context": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "format": "date-time"},
                    "strategy": {"const": "htp"},
                    "version": {"type": "string", "pattern": "^\\d+\\.\\d+$"},
                    "exporter": {"type": "string"},
                    "embed_hierarchy_attributes": {"type": "boolean"}
                },
                "required": ["timestamp", "strategy", "version"]
            },
            
            "model": {
                "type": "object",
                "properties": {
                    "name_or_path": {"type": "string"},
                    "class": {"type": "string"},
                    "framework": {"type": "string"},
                    "total_modules": {"type": "integer", "minimum": 1},
                    "total_parameters": {"type": "integer", "minimum": 0}
                },
                "required": ["name_or_path", "class"]
            },
            
            "tracing": {
                "type": "object",
                "properties": {
                    "builder": {"type": "string"},
                    "modules_traced": {"type": "integer", "minimum": 0},
                    "execution_steps": {"type": "integer", "minimum": 0},
                    "model_type": {"type": "string"},
                    "task": {"type": "string"},
                    "inputs": {"type": "object"},
                    "outputs": {"type": "array", "items": {"type": "string"}}
                }
            },
            
            "modules": {
                "type": "object",
                "additionalProperties": {
                    "$ref": "#/$defs/moduleInfo"
                }
            },
            
            "tagging": {
                "type": "object",
                "properties": {
                    "tagged_nodes": {"type": "object"},
                    "statistics": {"type": "object"},
                    "coverage": {
                        "type": "object",
                        "properties": {
                            "total_onnx_nodes": {"type": "integer", "minimum": 0},
                            "tagged_nodes": {"type": "integer", "minimum": 0},
                            "coverage_percentage": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 100.0
                            },
                            "empty_tags": {"type": "integer", "minimum": 0}
                        }
                    }
                }
            }
        },
        
        # Conditional validation based on model type
        "allOf": [
            # BERT-specific validation
            {
                "if": {
                    "properties": {
                        "model": {
                            "properties": {
                                "class": {"pattern": ".*Bert.*", "type": "string"}
                            }
                        }
                    }
                },
                "then": {
                    "properties": {
                        "tracing": {
                            "properties": {
                                "model_type": {"const": "bert"},
                                "inputs": {
                                    "required": ["input_ids", "attention_mask"],
                                    "properties": {
                                        "input_ids": {"$ref": "#/$defs/tensorInfo"},
                                        "attention_mask": {"$ref": "#/$defs/tensorInfo"},
                                        "token_type_ids": {"$ref": "#/$defs/tensorInfo"}
                                    }
                                },
                                "outputs": {
                                    "contains": {"const": "last_hidden_state"}
                                }
                            }
                        }
                    }
                }
            },
            
            # ResNet-specific validation
            {
                "if": {
                    "properties": {
                        "model": {
                            "properties": {
                                "class": {"pattern": ".*ResNet.*", "type": "string"}
                            }
                        }
                    }
                },
                "then": {
                    "properties": {
                        "tracing": {
                            "properties": {
                                "model_type": {"const": "resnet"},
                                "inputs": {
                                    "required": ["pixel_values"],
                                    "properties": {
                                        "pixel_values": {"$ref": "#/$defs/tensorInfo"}
                                    }
                                }
                            }
                        },
                        "custom_resnet_info": {
                            "type": "object",
                            "properties": {
                                "num_stages": {"type": "integer", "minimum": 1},
                                "has_pooling": {"type": "boolean"}
                            }
                        }
                    }
                }
            },
            
            # GPT-specific validation
            {
                "if": {
                    "properties": {
                        "model": {
                            "properties": {
                                "class": {"pattern": ".*GPT.*", "type": "string"}
                            }
                        }
                    }
                },
                "then": {
                    "properties": {
                        "tracing": {
                            "properties": {
                                "model_type": {"pattern": "^gpt"},
                                "inputs": {
                                    "required": ["input_ids"],
                                    "properties": {
                                        "input_ids": {"$ref": "#/$defs/tensorInfo"},
                                        "attention_mask": {"$ref": "#/$defs/tensorInfo"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        ],
        
        # Use unevaluatedProperties to catch any properties not covered by conditions
        "unevaluatedProperties": false,
        
        # Definitions for reuse
        "$defs": {
            "tensorInfo": {
                "type": "object",
                "properties": {
                    "shape": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1}
                    },
                    "dtype": {
                        "type": "string",
                        "enum": ["torch.float32", "torch.float16", "torch.int64", "torch.int32"]
                    }
                },
                "required": ["shape", "dtype"]
            },
            
            "moduleInfo": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "class_name": {"type": "string"},
                    "module_type": {"type": "string"},
                    "traced_tag": {"type": "string"},
                    "execution_order": {"type": "integer", "minimum": 0},
                    "expected_tag": {"type": "string"}
                },
                "required": ["name", "class_name"],
                
                # Module-specific properties based on class
                "allOf": [
                    {
                        "if": {
                            "properties": {
                                "class_name": {"pattern": ".*Attention.*"}
                            }
                        },
                        "then": {
                            "properties": {
                                "attention_config": {
                                    "type": "object",
                                    "properties": {
                                        "num_heads": {"type": "integer", "minimum": 1},
                                        "head_dim": {"type": "integer", "minimum": 1}
                                    }
                                }
                            }
                        }
                    },
                    {
                        "if": {
                            "properties": {
                                "class_name": {"pattern": ".*Conv.*"}
                            }
                        },
                        "then": {
                            "properties": {
                                "conv_config": {
                                    "type": "object",
                                    "properties": {
                                        "kernel_size": {
                                            "oneOf": [
                                                {"type": "integer", "minimum": 1},
                                                {
                                                    "type": "array",
                                                    "items": {"type": "integer", "minimum": 1}
                                                }
                                            ]
                                        },
                                        "stride": {"type": "integer", "minimum": 1},
                                        "padding": {"type": "integer", "minimum": 0}
                                    }
                                }
                            }
                        }
                    }
                ]
            }
        }
    }


def create_version_aware_schema() -> dict[str, Any]:
    """
    Create a schema that handles multiple metadata versions.
    
    Uses conditional validation to support backward compatibility.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://modelexport/schemas/htp-versioned",
        
        "type": "object",
        
        # First, check the version
        "if": {
            "properties": {
                "export_context": {
                    "properties": {
                        "version": {"const": "2.0"}
                    }
                }
            }
        },
        "then": {
            # Version 2.0 schema
            "$ref": "#/$defs/v2_schema"
        },
        "else": {
            "if": {
                "properties": {
                    "export_context": {
                        "properties": {
                            "version": {"const": "1.0"}
                        }
                    }
                }
            },
            "then": {
                # Version 1.0 schema (backward compatibility)
                "$ref": "#/$defs/v1_schema"
            },
            "else": {
                # Unknown version
                "not": true,
                "errorMessage": "Unsupported metadata version"
            }
        },
        
        "$defs": {
            "v1_schema": {
                "properties": {
                    # Original v1 properties
                    "export_info": {"type": "object"},
                    "statistics": {"type": "object"},
                    "hierarchy_summary": {"type": "object"},
                    "hierarchy_data": {"type": "object"},
                    "tagged_nodes": {"type": "object"}
                }
            },
            
            "v2_schema": {
                "properties": {
                    # New v2 structure
                    "export_context": {"type": "object"},
                    "model": {"type": "object"},
                    "tracing": {"type": "object"},
                    "modules": {"type": "object"},
                    "tagging": {"type": "object"},
                    "outputs": {"type": "object"},
                    "report": {"type": "object"},
                    "statistics": {"type": "object"}
                },
                "required": ["export_context", "model", "tracing", "modules"]
            }
        }
    }


def create_prefixItems_pipeline_schema() -> dict[str, Any]:
    """
    Use prefixItems to validate export pipeline steps in order.
    
    This ensures the export process follows the correct sequence.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        
        "type": "object",
        "properties": {
            "export_pipeline": {
                "type": "array",
                "prefixItems": [
                    {
                        # Step 1: Model Preparation
                        "type": "object",
                        "properties": {
                            "step": {"const": "model_preparation"},
                            "status": {"enum": ["pending", "in_progress", "completed", "failed"]},
                            "timestamp": {"type": "string", "format": "date-time"}
                        },
                        "required": ["step", "status"]
                    },
                    {
                        # Step 2: Input Generation
                        "type": "object",
                        "properties": {
                            "step": {"const": "input_generation"},
                            "status": {"enum": ["pending", "in_progress", "completed", "failed"]},
                            "method": {"enum": ["auto_generated", "provided", "custom"]}
                        },
                        "required": ["step", "status", "method"]
                    },
                    {
                        # Step 3: Hierarchy Building
                        "type": "object",
                        "properties": {
                            "step": {"const": "hierarchy_building"},
                            "status": {"enum": ["pending", "in_progress", "completed", "failed"]},
                            "modules_traced": {"type": "integer", "minimum": 0}
                        },
                        "required": ["step", "status", "modules_traced"]
                    },
                    {
                        # Step 4: ONNX Export
                        "type": "object",
                        "properties": {
                            "step": {"const": "onnx_export"},
                            "status": {"enum": ["pending", "in_progress", "completed", "failed"]},
                            "file_size_mb": {"type": "number", "minimum": 0}
                        },
                        "required": ["step", "status"]
                    },
                    {
                        # Step 5: Node Tagging
                        "type": "object",
                        "properties": {
                            "step": {"const": "node_tagging"},
                            "status": {"enum": ["pending", "in_progress", "completed", "failed"]},
                            "coverage_percentage": {"type": "number", "minimum": 0, "maximum": 100}
                        },
                        "required": ["step", "status", "coverage_percentage"]
                    }
                ],
                "items": false,  # No additional items allowed
                "minItems": 5,   # All steps required
                "maxItems": 5    # Exactly 5 steps
            }
        }
    }


# Example validation function
def validate_metadata_with_schema(metadata: dict[str, Any], schema: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate metadata against a schema.
    
    This would use jsonschema library with 2020-12 support.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    # Placeholder for actual validation
    # In practice, would use: jsonschema.Draft202012Validator
    
    # Example validation logic
    errors = []
    
    # Check model-specific requirements
    model_class = metadata.get("model", {}).get("class", "")
    model_type = metadata.get("tracing", {}).get("model_type", "")
    
    if "Bert" in model_class and model_type != "bert":
        errors.append("BERT model must have model_type='bert'")
    
    if "ResNet" in model_class and model_type != "resnet":
        errors.append("ResNet model must have model_type='resnet'")
    
    # Check coverage percentage bounds
    coverage = metadata.get("tagging", {}).get("coverage", {}).get("coverage_percentage", 0)
    if not 0 <= coverage <= 100:
        errors.append(f"Coverage percentage {coverage} out of bounds [0, 100]")
    
    return len(errors) == 0, errors