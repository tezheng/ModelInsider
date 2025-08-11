# TEZ-144 Experiments

This directory contains experimental scripts that validate our ONNX configuration strategy.

## Files

### `test_optimum_requirement.py`
Demonstrates that Optimum **requires** config.json to be present locally:
- ❌ WITHOUT config.json: Optimum fails to load model
- ✅ WITH config.json: Optimum loads successfully

**Run with:**
```bash
uv run python experiments/test_optimum_requirement.py
```

### `test_config_only_copy.py`
Shows the efficient approach using only AutoConfig and AutoProcessor:
- Tests multiple model types (NLP, Vision, Multimodal)
- Demonstrates we don't need to load full model weights
- Measures performance and storage overhead

**Run with:**
```bash
uv run python experiments/test_config_only_copy.py
```

## Key Findings

1. **Optimum Requirement Validated**: config.json MUST be present locally
2. **Efficient Copying**: Use `AutoConfig.from_pretrained(model_id)` directly
3. **Universal Approach**: Works for any HuggingFace model type
4. **Minimal Overhead**: Config files are typically < 1% of model size

## Next Steps

Integrate these findings into the production HTP exporter with:
```python
def export_with_config(model_id: str, output_dir: Path):
    # 1. Export ONNX with HTP
    export_onnx_with_hierarchy(model_id, output_dir / "model.onnx")
    
    # 2. Copy configs using model ID only
    config = AutoConfig.from_pretrained(model_id)
    config.save_pretrained(output_dir)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(output_dir)
    except:
        pass  # Not all models have tokenizers
```