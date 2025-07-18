# Advanced JSON Schema Features for HTP Metadata

## Overview

JSON Schema 2020-12 and related extensions provide powerful features we can leverage to make our HTP metadata system more flexible, extensible, and maintainable.

## 1. JSON Pointer for Efficient Navigation

JSON Pointer (RFC 6901) allows us to reference specific parts of our metadata:

```python
# Example: Reference specific module in hierarchy
pointer = "/modules/encoder.layer.0.attention.self"

# Use case: Quickly access deeply nested data
def get_module_info(metadata: dict, module_path: str) -> dict:
    """Get module info using JSON pointer syntax."""
    from jsonpointer import resolve_pointer
    return resolve_pointer(metadata, f"/modules/{module_path}")

# Example usage in analysis tools
attention_info = get_module_info(metadata, "encoder.layer.0.attention")
```

### Benefits for HTP:
- **Efficient Navigation**: Direct access to nested module data
- **Cross-References**: Link between tagging data and module hierarchy
- **Tooling**: Enable query languages for metadata analysis

## 2. Dynamic References for Extensible Schemas

Use `$dynamicRef` and `$dynamicAnchor` for creating extensible metadata schemas:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://modelexport/schemas/base-metadata",
  "$dynamicAnchor": "module-info",
  
  "properties": {
    "modules": {
      "type": "object",
      "additionalProperties": {
        "$dynamicRef": "#module-info"
      }
    }
  },
  
  "$defs": {
    "baseModule": {
      "$dynamicAnchor": "module-info",
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "class_name": { "type": "string" },
        "module_type": { "type": "string" }
      }
    }
  }
}
```

Extended schema for specific model types:
```json
{
  "$id": "https://modelexport/schemas/transformer-metadata",
  "$ref": "base-metadata",
  "$dynamicAnchor": "module-info",
  
  "$defs": {
    "transformerModule": {
      "$dynamicAnchor": "module-info",
      "$ref": "base-metadata#/$defs/baseModule",
      "properties": {
        "attention_heads": { "type": "integer" },
        "hidden_size": { "type": "integer" }
      }
    }
  }
}
```

## 3. Conditional Validation with unevaluatedProperties

Use conditional schemas to validate different model types:

```json
{
  "if": {
    "properties": {
      "model": {
        "properties": {
          "class": { "pattern": ".*BERT.*" }
        }
      }
    }
  },
  "then": {
    "properties": {
      "tracing": {
        "required": ["model_type"],
        "properties": {
          "model_type": { "const": "bert" }
        }
      }
    }
  },
  "else": {
    "if": {
      "properties": {
        "model": {
          "properties": {
            "class": { "pattern": ".*ResNet.*" }
          }
        }
      }
    },
    "then": {
      "properties": {
        "tracing": {
          "properties": {
            "model_type": { "const": "resnet" }
          }
        }
      }
    }
  },
  "unevaluatedProperties": false
}
```

## 4. JSON Patch for Metadata Updates

Use JSON Patch (RFC 6902) for efficient metadata updates:

```python
from jsonpatch import JsonPatch

# Update coverage statistics after re-tagging
patch = JsonPatch([
    {"op": "replace", "path": "/tagging/coverage/coverage_percentage", "value": 98.5},
    {"op": "replace", "path": "/tagging/coverage/tagged_nodes", "value": 134},
    {"op": "add", "path": "/report/steps/re_tagging", "value": {
        "status": "completed",
        "timestamp": "2024-01-01T12:00:00Z"
    }}
])

updated_metadata = patch.apply(original_metadata)
```

## 5. JSON Merge Patch for Simple Extensions

For simpler updates, use JSON Merge Patch (RFC 7396):

```python
import json_merge_patch

# Add custom fields to metadata
extension = {
    "custom_analysis": {
        "layer_complexity": {
            "encoder.layer.0": 0.8,
            "encoder.layer.1": 0.7
        }
    }
}

extended_metadata = json_merge_patch.merge(original_metadata, extension)
```

## 6. Vocabulary System for Modular Features

Define custom vocabularies for different aspects:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://modelexport/schemas/htp-metadata",
  "$vocabulary": {
    "https://json-schema.org/draft/2020-12/vocab/core": true,
    "https://json-schema.org/draft/2020-12/vocab/validation": true,
    "https://modelexport/vocab/hierarchy": true,
    "https://modelexport/vocab/tagging": true
  }
}
```

## 7. prefixItems for Ordered Metadata

Use `prefixItems` for metadata that has specific ordering:

```json
{
  "properties": {
    "export_pipeline": {
      "type": "array",
      "prefixItems": [
        { "$ref": "#/$defs/model_preparation" },
        { "$ref": "#/$defs/input_generation" },
        { "$ref": "#/$defs/hierarchy_building" },
        { "$ref": "#/$defs/onnx_export" },
        { "$ref": "#/$defs/node_tagging" }
      ],
      "items": false
    }
  }
}
```

## 8. Relative JSON Pointers for Context-Aware References

Use relative pointers for referencing related data:

```python
# In a module context, reference its parent
# "1/traced_tag" means: go up 1 level, then access traced_tag
relative_ref = "1/traced_tag"

# In tagging data, reference the corresponding module
# "2/modules/{module_name}" from within tagged_nodes
```

## Implementation Strategy

### Phase 1: Enhanced Validation
```python
class HTPMetadataValidator:
    def __init__(self, schema_version: str = "1.0"):
        self.schema = self._load_schema(schema_version)
        self.validator = jsonschema.Draft202012Validator(self.schema)
    
    def validate_with_patches(self, metadata: dict, patches: list[dict]) -> bool:
        """Validate metadata with potential patches applied."""
        # Apply patches
        patched = JsonPatch(patches).apply(metadata)
        # Validate against schema
        self.validator.validate(patched)
        return True
```

### Phase 2: Extensible Schemas
```python
class ExtensibleHTPSchema:
    """Support for model-specific schema extensions."""
    
    def create_extended_schema(self, base_schema: dict, model_type: str) -> dict:
        """Create model-specific schema using dynamic references."""
        extension = self._get_model_extension(model_type)
        # Use $dynamicRef to extend base schema
        return {
            "$ref": base_schema["$id"],
            "$dynamicAnchor": "module-info",
            "$defs": extension
        }
```

### Phase 3: Query and Analysis Tools
```python
class MetadataQuery:
    """Query metadata using JSON Pointer and JSONPath."""
    
    def query(self, metadata: dict, query: str) -> Any:
        if query.startswith("/"):
            # JSON Pointer
            return resolve_pointer(metadata, query)
        else:
            # JSONPath expression
            return jsonpath.findall(query, metadata)
    
    def find_modules_by_type(self, metadata: dict, module_type: str) -> list:
        """Find all modules of a specific type."""
        query = f"$.modules[?(@.class_name == '{module_type}')]"
        return self.query(metadata, query)
```

## Benefits for HTP Metadata

1. **Extensibility**: Dynamic references allow model-specific extensions
2. **Validation**: Conditional validation for different model types
3. **Efficiency**: JSON Pointer for fast data access
4. **Updates**: JSON Patch for efficient metadata updates
5. **Modularity**: Vocabulary system for feature organization
6. **Tooling**: Enable powerful query and analysis tools
7. **Evolution**: Schema versioning with backward compatibility

## Example: Complete Advanced Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://modelexport/schemas/htp-metadata/v2.0",
  "$vocabulary": {
    "https://json-schema.org/draft/2020-12/vocab/core": true,
    "https://json-schema.org/draft/2020-12/vocab/validation": true,
    "https://json-schema.org/draft/2020-12/vocab/unevaluated": true,
    "https://modelexport/vocab/hierarchy": true
  },
  
  "type": "object",
  "properties": {
    "export_context": {
      "type": "object",
      "properties": {
        "version": { 
          "type": "string",
          "pattern": "^\\d+\\.\\d+$"
        }
      }
    }
  },
  
  "if": {
    "properties": {
      "export_context": {
        "properties": {
          "version": { "const": "2.0" }
        }
      }
    }
  },
  "then": {
    "$ref": "#/$defs/v2_schema"
  },
  "else": {
    "$ref": "#/$defs/v1_schema"
  },
  
  "$defs": {
    "v2_schema": {
      "$dynamicAnchor": "metadata",
      "properties": {
        "advanced_features": {
          "type": "object",
          "properties": {
            "query_indexes": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "name": { "type": "string" },
                  "pointer": { "type": "string", "format": "json-pointer" }
                }
              }
            }
          }
        }
      }
    }
  }
}
```

## Conclusion

These advanced JSON Schema features provide powerful capabilities for:
- Creating extensible, model-specific metadata schemas
- Efficient querying and navigation of complex hierarchies
- Validation that adapts to different model types
- Clean update mechanisms for metadata evolution
- Modular feature organization

By leveraging these features, we can build a more robust and flexible metadata system that scales with the complexity of modern ML models.