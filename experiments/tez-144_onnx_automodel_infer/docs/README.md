# TEZ-144 Documentation Index

This folder contains documentation for the ONNX AutoModel Inference experiment.

## Document Structure

### Architecture & Design (Read First)

1. **[high_level_design.md](./high_level_design.md)** 📐
   - System architecture with mermaid diagrams
   - Core loops and function flows
   - Error handling and caching strategies
   - Performance optimizations

2. **[implementation_guide.md](./implementation_guide.md)** 🔧
   - Detailed implementation with code examples
   - Config resolution implementation
   - AutoModelForONNX class details
   - Testing and deployment patterns

### Reference Documents

3. **[optimum_config_generation_progress.md](./optimum_config_generation_progress.md)** 📚
   - Historical: UniversalOnnxConfig implementation
   - Superseded by ADR-013 decision
   - Kept for edge case reference

### User Guides (To Be Created)

4. **inference_guide.md** (TODO)
   - Main user guide for ONNX inference
   - Step-by-step tutorials
   - Best practices

5. **performance_benchmarks.md** (TODO)
   - Performance comparisons
   - Optimization techniques
   - Benchmarking methodology

6. **troubleshooting.md** (TODO)
   - Common issues and solutions
   - FAQ
   - Debug techniques

## Key Architectural References

These are the main architectural decision records that guide this implementation:

- **[ADR-012: Version Management Strategy](../../../docs/adr/ADR-012-version-management-strategy.md)**
  - Version management and tracking strategy
  
- **[ADR-013: ONNX Config for Optimum Compatibility](../../../docs/adr/ADR-013-onnx-config-for-optimum-compatibility.md)**
  - The architectural decision for configuration handling
  - Smart hybrid approach for Hub vs Local models  
  - Includes decision that UniversalOnnxConfig is NOT needed

## Quick Summary

### What We Use (ADR-013 Approach)
✅ Automatic config handling for HF Hub models  
✅ Config.json copying for local models  
✅ Smart detection of model source  
✅ Zero manual configuration  

### What We DON'T Use
❌ UniversalOnnxConfig (except rare edge cases)  
❌ Manual config generation  
❌ Complex configuration logic  

## Navigation

- [Back to Experiment README](../README.md)
- [Main Project README](../../../README.md)
- [ADR Directory](../../../docs/adr/)