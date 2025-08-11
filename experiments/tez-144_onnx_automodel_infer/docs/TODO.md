# TEZ-144: ONNX Optimum Compatibility - TODO List

## ✅ Completed Tasks

1. **Review TEZ-144 requirements and current implementation status**
   - Analyzed Linear task requirements
   - Reviewed current HTP export implementation

2. **Create notebook that demonstrates ONNX inference with exported BERT model**
   - Created initial demo notebook with bert-tiny

3. **Review ADR-012 and update TEZ-144 experiment docs**
   - Resolved ADR numbering conflicts
   - Updated experiment documentation

4. **Review and improve ADR-013 for inconsistencies**
   - Fixed dates and references
   - Improved structure and clarity

5. **Comprehensive review of ADR-013 with Optimum analysis**
   - Analyzed Optimum source code
   - Discovered config.json requirement
   - Created revised ADR with "Always Copy" approach

6. **Replace original ADR-013 with revised version**
   - Consolidated to single authoritative version
   - Updated to reflect Optimum's actual requirements

7. **Clean up duplicate documentation files**
   - Removed all duplicate ADR versions
   - Consolidated experiment docs

8. **Create feasibility demo showing Optimum requirements**
   - Created `config_only_demo.ipynb` - optimal approach using AutoConfig directly
   - Created `optimum_feasibility_demo.ipynb` - proves Optimum requires config.json
   - Created validation scripts in `experiments/` directory
   - All demos successfully validate the "Always Copy Configuration" approach

## 🔄 In Progress Tasks

None currently active.

## 📋 Pending Tasks

### Implementation Tasks

8. **Update HTP exporter to include config copying**
   - Add `export_with_config()` function
   - Integrate config copying into export pipeline
   - Handle both Hub and local models

9. **Update CLI to use new export_with_config function**
   - Modify CLI export command
   - Add `--no-config` flag for legacy behavior
   - Update help documentation

10. **Add validation tests for Optimum compatibility**
    - Test config copying for different model types
    - Verify Optimum can load exported models
    - Add integration tests with ORTModel classes

### Documentation Tasks

11. **Update references to ADR-013 in other documents**
    - Update main README
    - Update HTP documentation
    - Check for any stale references

12. **Create inference documentation guide**
    - Step-by-step inference tutorial
    - Common use cases and patterns
    - Troubleshooting guide

### Example & Demo Tasks

13. **Create example scripts for different model types**
    - NLP models (BERT, GPT-2, T5)
    - Vision models (ViT, ResNet)
    - Multimodal models

14. **Add performance benchmarking examples**
    - PyTorch vs ONNX Runtime comparison
    - Latency and throughput metrics
    - Memory usage analysis

15. **Create integration examples (FastAPI, Gradio)**
    - REST API server with FastAPI
    - Interactive demo with Gradio
    - Batch processing examples

## 🎯 Next Priority

**Immediate next step**: Create feasibility demo notebook showing:
1. Export BERT-tiny with config files
2. Load with Optimum ORTModel
3. Run inference successfully

This will validate our revised approach before implementing the full solution.

## 📅 Timeline Estimate

- **Week 1**: Implementation tasks (8-10)
- **Week 2**: Documentation and examples (11-15)
- **Testing**: Continuous throughout

## 📝 Notes

- The "Always Copy Configuration" approach has been validated as the most pragmatic solution
- Optimum requires config.json to be physically present in the model directory
- The 2-5KB overhead for config files is negligible (< 0.01% of model size)
- Future optimization with custom loaders is possible but not immediately necessary