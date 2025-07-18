"""
Universal JSON Schema definitions without hardcoded model logic.

This module provides schema definitions that work for any model architecture
by focusing on structural patterns rather than specific model names.
"""

from __future__ import annotations

from typing import Any


def create_universal_htp_schema() -> dict[str, Any]:
    """
    Create a universal schema that validates any model type.
    
    This schema focuses on structural requirements without
    hardcoding specific model architectures.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://modelexport/schemas/htp-universal/v2.0",
        "title": "Universal HTP Metadata Schema",
        "description": "Validates metadata for any model architecture",
        
        "type": "object",
        "required": ["export_context", "model", "tracing", "modules", "tagging"],
        
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
                    "name_or_path": {"type": "string", "minLength": 1},
                    "class": {"type": "string", "minLength": 1},
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
                    "model_type": {"type": "string"},  # Optional, no hardcoded values
                    "task": {"type": "string"},
                    "inputs": {
                        "type": "object",
                        "additionalProperties": {"$ref": "#/$defs/tensorInfo"},
                        "minProperties": 1  # At least one input required
                    },
                    "outputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1  # At least one output
                    }
                },
                "required": ["inputs"]
            },
            
            "modules": {
                "type": "object",
                "additionalProperties": {
                    "$ref": "#/$defs/moduleInfo"
                },
                "minProperties": 1  # At least one module
            },
            
            "tagging": {
                "type": "object",
                "properties": {
                    "tagged_nodes": {
                        "type": "object",
                        "additionalProperties": {"type": "string"}
                    },
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
                        },
                        "required": ["total_onnx_nodes", "tagged_nodes", "coverage_percentage"]
                    }
                },
                "required": ["tagged_nodes", "coverage"]
            },
            
            "outputs": {
                "type": "object",
                "properties": {
                    "onnx_model": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "size_mb": {"type": "number", "minimum": 0},
                            "opset_version": {"type": "integer", "minimum": 1},
                            "output_names": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["path", "size_mb"]
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        }
                    }
                }
            },
            
            "report": {
                "type": "object",
                "properties": {
                    "export_time_seconds": {"type": "number", "minimum": 0},
                    "steps": {"type": "object"},
                    "quality_guarantees": {
                        "type": "object",
                        "properties": {
                            "no_hardcoded_logic": {"const": true},
                            "universal_module_tracking": {"type": "string"},
                            "empty_tags_guarantee": {"type": "integer"},
                            "coverage_guarantee": {"type": "string"},
                            "optimum_compatible": {"type": "boolean"}
                        }
                    }
                }
            },
            
            "statistics": {
                "type": "object",
                "properties": {
                    "export_time": {"type": "number", "minimum": 0},
                    "hierarchy_modules": {"type": "integer", "minimum": 0},
                    "onnx_nodes": {"type": "integer", "minimum": 0},
                    "tagged_nodes": {"type": "integer", "minimum": 0},
                    "empty_tags": {"type": "integer", "minimum": 0},
                    "coverage_percentage": {"type": "number", "minimum": 0, "maximum": 100},
                    "module_types": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            
            "validation": {
                "type": "object",
                "properties": {
                    "structure_analysis": {"type": "object"},
                    "quality_score": {"type": "number", "minimum": 0, "maximum": 100},
                    "checks_passed": {"type": "object"}
                }
            }
        },
        
        # Definitions for reuse
        "$defs": {
            "tensorInfo": {
                "type": "object",
                "properties": {
                    "shape": {
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "dtype": {
                        "type": "string",
                        "pattern": "^torch\\.(float32|float16|int64|int32|bool|uint8)$"
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
                    "expected_tag": {"type": "string"},
                    "parameters": {"type": "integer", "minimum": 0}
                },
                "required": ["name", "class_name"]
            }
        }
    }


def create_structural_validation_schema() -> dict[str, Any]:
    """
    Create a schema that validates based on structural patterns.
    
    This uses conditional validation based on input/output patterns
    rather than hardcoded model names.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://modelexport/schemas/htp-structural",
        "title": "Structural Pattern Validation",
        
        "type": "object",
        
        # Conditional validation based on input structure
        "allOf": [
            # Text-based models (have input_ids)
            {
                "if": {
                    "properties": {
                        "tracing": {
                            "properties": {
                                "inputs": {
                                    "required": ["input_ids"]
                                }
                            }
                        }
                    }
                },
                "then": {
                    "properties": {
                        "tracing": {
                            "properties": {
                                "inputs": {
                                    "properties": {
                                        "input_ids": {"$ref": "#/$defs/integerTensor"},
                                        "attention_mask": {"$ref": "#/$defs/integerTensor"}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            
            # Vision-based models (have pixel_values)
            {
                "if": {
                    "properties": {
                        "tracing": {
                            "properties": {
                                "inputs": {
                                    "required": ["pixel_values"]
                                }
                            }
                        }
                    }
                },
                "then": {
                    "properties": {
                        "tracing": {
                            "properties": {
                                "inputs": {
                                    "properties": {
                                        "pixel_values": {"$ref": "#/$defs/floatTensor"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        ],
        
        "$defs": {
            "integerTensor": {
                "type": "object",
                "properties": {
                    "dtype": {"pattern": "^torch\\.(int64|int32)$"}
                }
            },
            "floatTensor": {
                "type": "object",
                "properties": {
                    "dtype": {"pattern": "^torch\\.(float32|float16)$"}
                }
            }
        }
    }


def create_version_aware_schema() -> dict[str, Any]:
    """
    Create a schema that handles multiple metadata versions.
    
    Uses conditional validation to support backward compatibility
    without hardcoding model-specific logic.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://modelexport/schemas/htp-versioned",
        
        "type": "object",
        
        # Version detection
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
            # Version 2.0 uses new structure
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
                # Version 1.0 (backward compatibility)
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
                "$ref": "https://modelexport/schemas/htp-universal/v2.0"
            }
        }
    }


def create_quality_validation_schema() -> dict[str, Any]:
    """
    Create a schema focused on quality metrics validation.
    
    This ensures quality standards are met regardless of model type.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://modelexport/schemas/htp-quality",
        
        "type": "object",
        
        "properties": {
            "tagging": {
                "properties": {
                    "coverage": {
                        "properties": {
                            "coverage_percentage": {
                                "minimum": 95.0,  # Require 95% coverage
                                "errorMessage": "Coverage must be at least 95%"
                            },
                            "empty_tags": {
                                "maximum": 0,  # No empty tags allowed
                                "errorMessage": "Empty tags are not allowed"
                            }
                        }
                    }
                }
            },
            
            "report": {
                "properties": {
                    "quality_guarantees": {
                        "properties": {
                            "no_hardcoded_logic": {
                                "const": true,
                                "errorMessage": "Hardcoded logic detected"
                            }
                        }
                    }
                }
            },
            
            "validation": {
                "properties": {
                    "quality_score": {
                        "minimum": 90.0,  # Require 90% quality score
                        "errorMessage": "Quality score must be at least 90%"
                    }
                }
            }
        }
    }


# Example validation function
def validate_metadata_universally(metadata: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate metadata using universal patterns.
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check basic structure
    required_sections = ["export_context", "model", "tracing", "modules", "tagging"]
    for section in required_sections:
        if section not in metadata:
            errors.append(f"Missing required section: {section}")
    
    # Check coverage requirements
    coverage = metadata.get("tagging", {}).get("coverage", {})
    coverage_pct = coverage.get("coverage_percentage", 0)
    if coverage_pct < 95.0:
        errors.append(f"Coverage {coverage_pct:.1f}% is below 95% requirement")
    
    empty_tags = coverage.get("empty_tags", 0)
    if empty_tags > 0:
        errors.append(f"Found {empty_tags} empty tags")
    
    # Check quality guarantees
    guarantees = metadata.get("report", {}).get("quality_guarantees", {})
    if not guarantees.get("no_hardcoded_logic", False):
        errors.append("Quality guarantee violation: hardcoded logic detected")
    
    return len(errors) == 0, errors