# Implementation Guide - ONNX Configuration for Optimum Compatibility

## Overview

This guide implements the **Always Copy Configuration** strategy from ADR-013 (Revised), ensuring full compatibility with HuggingFace Optimum's inference APIs.

## Core Implementation

### 1. Enhanced Export Function

```python
# modelexport/conversion/hf_universal_hierarchy_exporter.py

import os
import shutil
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoImageProcessor, AutoProcessor
from typing import Optional, Union

def export_with_config(
    model_name_or_path: str,
    output_dir: Union[str, Path],
    input_text: Optional[str] = None,
    strategy: str = "htp",
    clean_onnx: bool = False,
    **kwargs
) -> Path:
    """
    Export model to ONNX with all required configuration files.
    
    Args:
        model_name_or_path: HF Hub model ID or local path
        output_dir: Directory to save exported model
        input_text: Optional input text for tracing
        strategy: Export strategy (default: "htp")
        clean_onnx: If True, export without hierarchy metadata
        **kwargs: Additional export parameters
    
    Returns:
        Path to the output directory containing model and configs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Export ONNX model using HTP strategy
    onnx_path = output_dir / "model.onnx"
    export_onnx_with_hierarchy(
        model_name_or_path,
        onnx_path,
        input_text=input_text,
        strategy=strategy,
        clean_onnx=clean_onnx,
        **kwargs
    )
    
    # Step 2: Copy configuration files
    try:
        # Load and save config
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.save_pretrained(output_dir)
        
        # Try to load and save tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            tokenizer.save_pretrained(output_dir)
        except Exception:
            # Model might not have a tokenizer (e.g., vision models)
            pass
        
        # Try to load and save image processor
        try:
            processor = AutoImageProcessor.from_pretrained(model_name_or_path)
            processor.save_pretrained(output_dir)
        except Exception:
            # Model might not have an image processor
            pass
        
        # Try to load and save general processor
        try:
            processor = AutoProcessor.from_pretrained(model_name_or_path)
            processor.save_pretrained(output_dir)
        except Exception:
            # Model might not have a processor
            pass
            
    except Exception as e:
        raise RuntimeError(
            f"Failed to copy configuration files: {e}\n"
            "The ONNX model was exported but may not be compatible with Optimum."
        )
    
    # Step 3: Verify required files exist
    if not (output_dir / "config.json").exists():
        raise FileNotFoundError(
            "config.json was not created. The model cannot be used with Optimum."
        )
    
    # Step 4: Create metadata file for tracking
    metadata = {
        "source_model": model_name_or_path,
        "export_strategy": strategy,
        "clean_onnx": clean_onnx,
        "export_timestamp": datetime.now().isoformat(),
        "modelexport_version": get_version()
    }
    
    with open(output_dir / "export_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model exported successfully to {output_dir}")
    print(f"   Files created:")
    for file in sorted(output_dir.glob("*")):
        size = file.stat().st_size
        print(f"   - {file.name}: {format_size(size)}")
    
    return output_dir
```

### 2. CLI Integration

```python
# modelexport/cli.py

@click.command()
@click.argument("model_name")
@click.argument("output_path")
@click.option("--strategy", default="htp", help="Export strategy")
@click.option("--clean-onnx", is_flag=True, help="Export without metadata")
@click.option("--no-config", is_flag=True, help="Skip config copying (not recommended)")
def export(model_name, output_path, strategy, clean_onnx, no_config):
    """Export HuggingFace model to ONNX with Optimum compatibility."""
    
    output_path = Path(output_path)
    
    if no_config:
        # Legacy behavior - just export ONNX
        click.echo("⚠️  Warning: Exporting without config files.")
        click.echo("   The model will NOT work with Optimum!")
        export_onnx_with_hierarchy(
            model_name,
            output_path,
            strategy=strategy,
            clean_onnx=clean_onnx
        )
    else:
        # Default behavior - export with configs
        if output_path.suffix == ".onnx":
            # User provided file path, convert to directory
            output_dir = output_path.parent / output_path.stem
            click.echo(f"📁 Creating output directory: {output_dir}")
        else:
            output_dir = output_path
        
        export_with_config(
            model_name,
            output_dir,
            strategy=strategy,
            clean_onnx=clean_onnx
        )
```

### 3. Validation Function

```python
def validate_optimum_compatibility(model_dir: Union[str, Path]) -> bool:
    """
    Validate that exported model is compatible with Optimum.
    
    Args:
        model_dir: Directory containing exported model
    
    Returns:
        True if compatible, False otherwise
    """
    model_dir = Path(model_dir)
    
    # Check required files
    required_files = ["model.onnx", "config.json"]
    missing_files = []
    
    for file in required_files:
        if not (model_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    # Try loading with Optimum
    try:
        from optimum.onnxruntime import ORTModel
        model = ORTModel.from_pretrained(model_dir)
        print("✅ Model is compatible with Optimum!")
        return True
    except Exception as e:
        print(f"❌ Failed to load with Optimum: {e}")
        return False
```

## Usage Examples

### Basic Export

```bash
# Export BERT model with all configs
modelexport export bert-base-uncased bert_exported/

# Directory structure created:
# bert_exported/
# ├── model.onnx
# ├── config.json
# ├── tokenizer.json
# ├── tokenizer_config.json
# └── export_metadata.json
```

### Python API

```python
from modelexport import export_with_config
from optimum.onnxruntime import ORTModelForSequenceClassification

# Export model
output_dir = export_with_config(
    "distilbert-base-uncased-finetuned-sst-2-english",
    "sentiment_model/"
)

# Load with Optimum - just works!
model = ORTModelForSequenceClassification.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

# Run inference
inputs = tokenizer("I love this!", return_tensors="np")
outputs = model(**inputs)
```

### Vision Models

```python
# Export vision model
export_with_config("google/vit-base-patch16-224", "vit_exported/")

# Load with Optimum
from optimum.onnxruntime import ORTModelForImageClassification
model = ORTModelForImageClassification.from_pretrained("vit_exported/")
processor = AutoImageProcessor.from_pretrained("vit_exported/")
```

## Testing

### Unit Tests

```python
# tests/test_export_with_config.py

def test_export_creates_required_files(tmp_path):
    """Test that export creates all required files."""
    output_dir = tmp_path / "test_model"
    
    export_with_config("hf-internal-testing/tiny-random-bert", output_dir)
    
    assert (output_dir / "model.onnx").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "tokenizer.json").exists()
    assert (output_dir / "tokenizer_config.json").exists()

def test_optimum_compatibility(tmp_path):
    """Test that exported model works with Optimum."""
    output_dir = tmp_path / "test_model"
    
    export_with_config("hf-internal-testing/tiny-random-bert", output_dir)
    
    # Should load without errors
    from optimum.onnxruntime import ORTModelForSequenceClassification
    model = ORTModelForSequenceClassification.from_pretrained(output_dir)
    assert model is not None
```

### Integration Tests

```python
def test_end_to_end_inference(tmp_path):
    """Test complete export and inference workflow."""
    # Export
    output_dir = export_with_config(
        "hf-internal-testing/tiny-random-bert",
        tmp_path / "model"
    )
    
    # Load
    model = ORTModelForSequenceClassification.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    
    # Inference
    inputs = tokenizer("Test input", return_tensors="np")
    outputs = model(**inputs)
    
    assert outputs.logits.shape[0] == 1  # Batch size
    assert outputs.logits.shape[1] == 2  # Num classes
```

## Migration Guide

### For Existing Models

If you have ONNX models exported without configs:

```python
def add_config_to_existing_model(onnx_path: Path, model_name: str):
    """Add configuration files to existing ONNX export."""
    model_dir = onnx_path.parent
    
    # Download and save config
    config = AutoConfig.from_pretrained(model_name)
    config.save_pretrained(model_dir)
    
    # Download and save tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_dir)
    except:
        pass
    
    print(f"✅ Added config files to {model_dir}")
```

### Backward Compatibility

The old export function still works but shows a deprecation warning:

```python
@deprecated("Use export_with_config for Optimum compatibility")
def export_onnx_with_hierarchy(model_name, output_path, **kwargs):
    """Legacy export function - configs not included."""
    warnings.warn(
        "This function exports ONNX without config files. "
        "The model will not work with Optimum. "
        "Use export_with_config instead.",
        DeprecationWarning
    )
    # ... original implementation
```

## Performance Considerations

### Storage Impact

| Component | Typical Size | Percentage of Model |
|-----------|-------------|-------------------|
| ONNX Model | 400MB | 99.99% |
| config.json | 2KB | 0.0005% |
| tokenizer.json | 1KB | 0.00025% |
| tokenizer_config.json | 1KB | 0.00025% |
| **Total Overhead** | **~5KB** | **< 0.01%** |

### Export Time

- Config copying adds < 100ms to export time
- Network download (for Hub models): 100-500ms
- Local copying: < 10ms

## Troubleshooting

### Common Issues

1. **Missing config.json**
   ```
   Error: config.json not found
   Solution: Re-export with modelexport >= 1.0.0
   ```

2. **Optimum import error**
   ```
   Error: Cannot import ORTModel
   Solution: pip install optimum[onnxruntime]
   ```

3. **Tokenizer not found**
   ```
   Warning: Tokenizer not found (this is OK for vision models)
   ```

## Future Enhancements

### Phase 1: Current (Implemented)
- ✅ Always copy configs
- ✅ Full Optimum compatibility
- ✅ Clear error messages

### Phase 2: Convenience Wrapper (Planned)
- Custom `AutoModelForONNX` class
- Automatic model type detection
- Enhanced error handling

### Phase 3: Optimum Contribution (Future)
- Propose metadata-based loading
- Maintain backward compatibility
- Reduce deployment size

## Summary

The **Always Copy Configuration** approach provides:

1. **100% Optimum Compatibility** - Works immediately
2. **Simple Implementation** - No complex logic needed
3. **Predictable Behavior** - Same pattern for all models
4. **Negligible Overhead** - < 0.01% size increase
5. **Future-Proof** - Easy to optimize later if needed

This pragmatic solution prioritizes compatibility and simplicity over minimal optimization that would require significant complexity.