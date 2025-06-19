# Test Case: Export Time Bounds

## Type
**Performance Test** 🚀

## Purpose
Verify that model export completes within reasonable time bounds and doesn't hang indefinitely. Establish performance baselines for different model sizes.

## Test Data (Fixtures)
- BERT-tiny model (~4M parameters)
- Simple model (<1M parameters)
- Baseline timing measurements
- System performance metrics

## Test Command
```bash
# Timed export test
time uv run modelexport export prajjwal1/bert-tiny test_timing.onnx

# Performance test via pytest (with extended timeout)
timeout 600 uv run python -m pytest tests/test_cli_integration.py::TestCLIBertConversion::test_bert_conversion_basic -v

# Simple model timing
time uv run python -c "
import time
from modelexport import HierarchyExporter
# ... simple model export code ...
"
```

## Expected Behavior
- **BERT-tiny export**: <120 seconds total
  - Model loading: 30-60s
  - Export process: 30-60s  
- **Simple model export**: <10 seconds total
- **No timeouts**: Process completes successfully
- **Memory usage**: <4GB peak

## Failure Modes
- **Export Timeout**: Process hangs or takes >10 minutes
- **Memory Explosion**: Usage exceeds 8GB
- **Infinite Loops**: Process never completes
- **Model Loading Hang**: Download/loading takes >5 minutes

## Dependencies
- Stable internet connection (for model download)
- Sufficient system memory (4GB+)
- No competing heavy processes
- Fresh model cache (or cached model)

## Notes
### Performance Baselines (Target)
| Model Type | Size | Load Time | Export Time | Total |
|------------|------|-----------|-------------|-------|
| BERT-tiny | 4.4M | 30-60s | 30-60s | <120s |
| Simple | <1M | <1s | <5s | <10s |

### Performance Factors
- **First run**: Includes model download time
- **Cached run**: Model already downloaded
- **System specs**: CPU, memory, disk speed
- **Network**: Download speed for transformers models

### Timeout Strategy
- Tests use 600s (10 minute) timeout
- 2 minute timeout too aggressive for BERT models
- Simple models should complete much faster
- Extended timeout accounts for model loading

## Historical Context
- Initial 2-minute timeouts caused CI failures
- Model loading dominates export time
- PyTorch ONNX export itself is relatively fast
- Hierarchy processing adds minimal overhead

## Validation Checklist
- [ ] Export completes within timeout bounds
- [ ] No infinite loops or hangs
- [ ] Memory usage stays reasonable
- [ ] Performance degradation doesn't exceed 2x baseline
- [ ] Simple models much faster than complex models
- [ ] Time scales reasonably with model complexity