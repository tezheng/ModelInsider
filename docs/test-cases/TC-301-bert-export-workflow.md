# Test Case: BERT Export Workflow

## Type
**Integration Test** 🔗

## Purpose
Verify the complete end-to-end workflow for exporting a BERT model with hierarchy preservation, including CLI usage, tag generation, and analysis.

## Test Data (Fixtures)
- BERT-tiny model (`prajjwal1/bert-tiny`)
- Default text input or custom input text
- Output directory for ONNX and sidecar files

## Test Command
```bash
# Complete CLI workflow
uv run modelexport export prajjwal1/bert-tiny my_bert.onnx --verbose

# With custom input
uv run modelexport export prajjwal1/bert-tiny bert_custom.onnx \
  --input-text "The quick brown fox jumps over the lazy dog" \
  --input-shape 1,64

# Analysis workflow
uv run modelexport analyze my_bert.onnx --format summary
uv run modelexport validate my_bert.onnx

# Pytest integration test
uv run python -m pytest tests/test_cli_integration.py::TestCLIBertConversion::test_bert_conversion_basic -v
```

## Expected Behavior
- BERT model loads successfully (~40-50s)
- ONNX export completes without errors
- Hierarchy tags generated with proper BERT structure:
  - `/BertModel/BertEmbeddings`
  - `/BertModel/BertEncoder/BertLayer/BertAttention/BertSelfOutput`
  - `/BertModel/BertPooler`
- Sidecar JSON file created with tag statistics
- CLI analysis shows tag distribution
- Validation passes ONNX compliance

## Failure Modes
- **Model Loading Timeout**: BERT model takes >60s to load
- **Memory Issues**: Insufficient memory for model + export
- **Tag Generation Issues**: Missing or incorrect hierarchy tags
- **ONNX Export Error**: PyTorch → ONNX conversion failures
- **CLI Interface Problems**: Argument parsing or command execution errors

## Dependencies
- transformers library
- BERT-tiny model (auto-downloaded)
- Sufficient memory (~2GB)
- Internet connection (for model download)
- CLI interface properly installed

## Notes
- This is the primary integration test for the entire system
- Tests real-world usage with actual transformer model
- Validates CLI interface and Python API consistency
- Model download happens once, then cached
- Export time varies: 40-60s for BERT-tiny

## Performance Expectations
- Model loading: 30-60s (first time, then cached)
- ONNX export: 30-60s  
- Tag generation: <10s
- Analysis: <5s
- Total workflow: ~90-120s

## Validation Checklist
- [ ] BERT model loads without timeout
- [ ] ONNX file created with reasonable size (>1MB)
- [ ] Sidecar JSON contains tag statistics
- [ ] BertSelfOutput tags present (critical for tests)
- [ ] No torch.nn modules in hierarchy tags
- [ ] CLI commands execute without errors
- [ ] Analysis shows expected tag distribution