# TEZ-144 Notebooks

Interactive demonstrations of our ONNX configuration strategy.

## Main Notebooks

### `config_only_demo.ipynb` ⭐ **RECOMMENDED**
**Latest and most efficient approach**
- Demonstrates using `AutoConfig.from_pretrained(model_id)` directly
- No need to load full model weights
- Tests NLP, Vision, and Multimodal models
- Shows performance benchmarks
- Includes complete implementation pattern

### `optimum_feasibility_demo.ipynb` 
**Validation of Optimum requirements**
- Proves that Optimum REQUIRES config.json locally
- Shows before/after loading behavior
- Performance comparison PyTorch vs ONNX Runtime
- Storage overhead analysis

## Legacy Notebooks

### `bert_onnx_inference_final.ipynb`
Original BERT inference demo

### `hub_integration_demo.ipynb` 
Early Hub integration experiments

### `understanding_onnxconfig.ipynb`
Initial exploration of ONNX configuration approaches

## Quick Start

1. **Run the main demo:**
```bash
jupyter notebook config_only_demo.ipynb
```

2. **Or execute directly:**
```bash
uv run jupyter nbconvert --to notebook --execute config_only_demo.ipynb
```

## Key Findings

- ✅ `AutoConfig.from_pretrained(model_id)` is fastest approach
- ✅ Works for any HuggingFace model without loading weights
- ✅ Config overhead is < 1% of model size
- ✅ Optimum loads and runs inference successfully

## Implementation Ready

The notebooks validate our production implementation:
```python
def export_with_config(model_id, output_dir):
    # Export ONNX
    export_onnx_with_hierarchy(model_id, output_dir / "model.onnx")
    
    # Copy configs efficiently 
    AutoConfig.from_pretrained(model_id).save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(output_dir)
```