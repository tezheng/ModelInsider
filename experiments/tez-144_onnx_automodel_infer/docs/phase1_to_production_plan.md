# TEZ-144: Phase 1 to Production Migration Plan

## Executive Summary

This document outlines the migration plan for moving the ONNX AutoProcessor implementation from experimental (`experiments/tez-144_onnx_automodel_infer/`) to production (`modelexport/onnx_processors/`).

**Current Status**: Phase 1 Complete (Experimental)  
**Target**: Production-Ready Integration  
**Timeline**: 2-3 weeks  
**Risk Level**: Medium (due to integration complexity)

## Phase 1 Achievements

### ✅ Completed Items
1. **Core Implementation**
   - ONNXAutoProcessor factory class with `from_model()` API
   - Support for 5 modalities (text, image, audio, video, multimodal)
   - Fixed-shape optimization for 40x+ performance improvement
   - Protocol-based type system with comprehensive validation

2. **Testing Infrastructure**
   - 38 comprehensive tests (100% passing)
   - Performance benchmarking framework
   - Mock-based testing strategy

3. **Documentation**
   - High-level design document
   - Processor-specific design documents
   - Test design documentation
   - API documentation

4. **Architecture**
   - Clean factory pattern implementation
   - ONNX-first design (removed `from_pretrained()`)
   - Modular processor architecture
   - Comprehensive error handling

## Critical Gaps (From Review)

### 🚨 Must Fix Before Production

1. **Missing Real-World Examples**
   - No working examples with actual ONNX models
   - Performance claims unverified with real benchmarks
   - Integration patterns not demonstrated

2. **Unverified Performance Claims**
   - 40x speedup not empirically validated
   - No comparison with PyTorch baselines
   - Missing ONNX Runtime benchmarks

3. **Mock-Heavy Testing**
   - Tests rely on mocks instead of real models
   - Integration with actual HuggingFace models untested
   - Edge cases with real ONNX models unknown

4. **Enhanced Pipeline Integration**
   - Universal `data_processor` parameter not implemented
   - Pipeline compatibility layer missing
   - Integration with modelexport main codebase unclear

## Production Migration Plan

### Phase 1.5: Cleanup and Deprecation

#### Files to Remove (Not Needed in Production)
- 13 experimental test scripts (replaced by comprehensive test suite)
- Legacy `src/onnx_tokenizer.py` (already removed)
- One-time analysis scripts
- Temporary notebooks and outputs
- Duplicate model files in notebooks/

#### Files to Archive (Keep for Reference)
- Research notebooks showing validation process
- Investigation results and findings
- Progress tracking documents
- Original design explorations

### Phase 2: Validation & Integration (Week 1)

#### 2.1 Real Model Validation
```bash
# Location: experiments/tez-144_onnx_automodel_infer/validation/
```

**Tasks:**
1. Export 5 reference models to ONNX:
   - BERT (prajjwal1/bert-tiny)
   - ResNet (microsoft/resnet-50)
   - Whisper (openai/whisper-tiny)
   - CLIP (openai/clip-vit-base-patch32)
   - GPT2 (gpt2)

2. Create real benchmarks:
   ```python
   validation/
   ├── benchmarks/
   │   ├── performance_comparison.py  # PyTorch vs ONNX
   │   ├── memory_profiling.py        # Memory usage analysis
   │   └── throughput_testing.py      # Batch processing speeds
   ├── models/
   │   └── [exported ONNX models]
   └── results/
       └── benchmark_results.json
   ```

3. Validate processor detection:
   - Test with non-standard tensor names
   - Verify fallback mechanisms
   - Document edge cases

**Success Criteria:**
- [ ] 40x speedup verified on at least 3 models
- [ ] All 5 reference models process correctly
- [ ] Memory usage within acceptable bounds
- [ ] Edge cases documented and handled

#### 2.2 Working Examples
```bash
# Location: experiments/tez-144_onnx_automodel_infer/examples/
```

**Create Examples:**
```python
examples/
├── basic/
│   ├── text_classification.py
│   ├── image_classification.py
│   ├── audio_transcription.py
│   └── multimodal_clip.py
├── advanced/
│   ├── batch_processing.py
│   ├── performance_optimization.py
│   └── custom_processors.py
└── integration/
    ├── with_optimum.py
    ├── with_modelexport.py
    └── with_pipelines.py
```

**Success Criteria:**
- [ ] Each example runs successfully
- [ ] Clear documentation in each file
- [ ] Performance metrics included
- [ ] Integration patterns demonstrated

### Phase 3: Production Refactoring (Week 2)

#### 3.1 Code Migration Structure

```bash
# Production code in modelexport:
modelexport/
└── inference/                  # New inference module
    ├── __init__.py
    ├── onnx_auto_processor.py # Main factory class
    ├── auto_model_loader.py   # AutoModel-like interface
    ├── types.py               # Type definitions (from onnx_processor_types.py)
    ├── pipeline.py            # Enhanced pipeline with data_processor
    ├── processors/            # Individual processor implementations
    │   ├── __init__.py
    │   ├── base.py
    │   ├── text.py
    │   ├── image.py
    │   ├── audio.py
    │   ├── video.py
    │   └── multimodal.py
    └── config/                # Configuration utilities
        ├── __init__.py
        ├── universal_config.py
        ├── task_detector.py
        └── shape_inference.py

# Tests at project root:
tests/
└── inference/                 # Inference tests
    ├── conftest.py
    ├── test_utils.py
    ├── test_onnx_auto_processor.py
    └── fixtures/
        └── models/

# Examples at project root:
examples/
└── inference/                 # Inference examples
    ├── README.md
    ├── basic_onnx_inference.py
    └── multimodal_inference.py

# Documentation at project root:
docs/
└── inference/                 # Inference documentation
    ├── README.md
    ├── architecture.md
    ├── testing_guide.md
    └── user_guide.md
```

#### 3.2 Integration with ModelExport

**Integration Points:**

1. **CLI Integration**

   ```python
   # modelexport/cli.py
   @cli.command()
   @click.option('--with-inference', is_flag=True, help='Setup inference pipeline')
   def export(model_name, output_path, with_inference):
       """Export model with optional inference setup"""
       if with_inference:
           from modelexport.inference import ONNXAutoProcessor
           # Export ONNX model
           # Create optimized processor for inference
   ```

2. **Export Pipeline Integration**

   ```python
   # modelexport/export.py
   def export_with_inference_setup(model_name, output_path):
       """Export model and setup inference pipeline"""
       from modelexport.inference import ONNXAutoProcessor
       # Export ONNX model
       # Create processor with fixed shapes
       # Save inference configuration
   ```

3. **Configuration Management**

   ```python
   # modelexport/config.py
   class InferenceConfig:
       """Configuration for ONNX inference"""
       batch_size: int = 1
       max_sequence_length: int = 128
       optimization_level: str = "aggressive"
       use_fixed_shapes: bool = True  # For ONNX optimization
   ```

#### 3.3 Testing Strategy

**Test Migration:**
1. Convert mock-based tests to use real models
2. Add integration tests with modelexport
3. Create performance regression tests
4. Add CI/CD pipeline tests

**Test Structure:**

```python
tests/test_inference/
├── unit/
│   ├── test_metadata_extraction.py
│   ├── test_shape_enforcement.py
│   └── test_processor_creation.py
├── integration/
│   ├── test_with_optimum.py
│   ├── test_with_modelexport.py
│   └── test_end_to_end.py
└── performance/
    ├── test_benchmarks.py
    ├── test_memory_usage.py
    └── test_throughput.py
```

### Phase 4: Production Deployment (Week 3)

#### 4.1 Documentation Update

**Documentation Tasks:**
1. Update main README with ONNX processor section
2. Create user guide for ONNX processors
3. Add API reference documentation
4. Create migration guide for existing users

**Documentation Structure:**

```markdown
docs/
├── inference/
│   ├── quickstart.md
│   ├── user_guide.md
│   ├── api_reference.md
│   ├── performance.md
│   └── troubleshooting.md
└── migration/
    └── inference_setup.md
```

#### 4.2 Release Preparation

**Release Checklist:**
- [ ] All tests passing with real models
- [ ] Performance benchmarks documented
- [ ] Examples working and documented
- [ ] Integration with modelexport complete
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] PR created and reviewed

#### 4.3 Rollout Strategy

**Phased Rollout:**
1. **Alpha Release** (Internal Testing)
   - Deploy to internal users
   - Gather feedback
   - Fix critical issues

2. **Beta Release** (Limited External)
   - Release to selected users
   - Monitor performance
   - Address feedback

3. **Production Release**
   - Full release with documentation
   - Announce in release notes
   - Monitor for issues

## Risk Mitigation

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Performance claims invalid | Medium | High | Validate early with real benchmarks |
| Integration breaks existing code | Low | High | Comprehensive integration testing |
| Edge cases in real models | High | Medium | Test with diverse model architectures |
| Memory issues at scale | Medium | Medium | Memory profiling and optimization |

### Mitigation Strategies

1. **Early Validation**
   - Start with performance validation immediately
   - If 40x speedup not achievable, adjust claims
   - Document realistic performance improvements

2. **Incremental Integration**
   - Integrate one component at a time
   - Maintain backward compatibility
   - Feature flag new functionality

3. **Comprehensive Testing**
   - Test with at least 10 different model architectures
   - Include edge cases and error conditions
   - Performance regression testing

## Success Metrics

### Quantitative Metrics
- [ ] 40x speedup verified on 3+ models
- [ ] 100% test coverage with real models
- [ ] <100ms processor creation time
- [ ] <10MB memory overhead per processor
- [ ] 5+ working examples

### Qualitative Metrics
- [ ] Clear and comprehensive documentation
- [ ] Intuitive API design
- [ ] Seamless integration with modelexport
- [ ] Positive user feedback
- [ ] No critical bugs in production

## Timeline

### Week 1 (Validation & Examples)
- Days 1-2: Export reference models and set up validation
- Days 3-4: Implement and run benchmarks
- Days 5-7: Create working examples

### Week 2 (Production Refactoring)
- Days 1-2: Migrate code to production structure
- Days 3-4: Integrate with modelexport
- Days 5-7: Update and run tests

### Week 3 (Deployment)
- Days 1-2: Documentation update
- Days 3-4: Alpha release and testing
- Days 5-7: Beta release and monitoring

## Conclusion

The ONNX AutoProcessor implementation is architecturally sound and feature-complete but requires validation with real models and integration work before production deployment. This plan provides a structured approach to address the gaps identified in the review and ensure a successful production release.

**Next Steps:**
1. Validate performance claims with real ONNX models
2. Create working examples demonstrating usage
3. Begin code migration to production structure
4. Update documentation for production release

**Estimated Completion**: 3 weeks from start date

---

# Complete File Migration Table

**Complete Inventory**: This table accounts for ALL files in the experimental directory tree, including hidden files, cache directories, temporary files, and external test suites.

**Last Updated**: 2025-08-13 | **Total Files Inventoried**: 230+

## Legend

- **KEEP**: Move to production location
- **ARCHIVE**: Keep for reference in docs/archive/inference/
- **DELETE**: Remove, not needed
- **REFACTOR**: Update before moving

## Source Code Files (`src/`)

| File | Action | Destination | Notes |
|------|--------|-------------|-------|
| `src/__init__.py` | KEEP | `modelexport/inference/__init__.py` | Update imports |
| `src/onnx_auto_processor.py` | KEEP | `modelexport/inference/onnx_auto_processor.py` | Core factory class |
| `src/auto_model_loader.py` | KEEP | `modelexport/inference/auto_model_loader.py` | AutoModel interface |
| `src/onnx_processor_types.py` | KEEP | `modelexport/inference/types.py` | Rename to types.py |
| `src/enhanced_pipeline.py` | KEEP | `modelexport/inference/pipeline.py` | Rename to pipeline.py |
| `src/inference_utils.py` | KEEP | `modelexport/inference/utils.py` | Useful inference utilities |
| `src/processors/__init__.py` | KEEP | `modelexport/inference/processors/__init__.py` | Processor exports |
| `src/processors/base.py` | KEEP | `modelexport/inference/processors/base.py` | Base processor class |
| `src/processors/text.py` | KEEP | `modelexport/inference/processors/text.py` | Text processor |
| `src/processors/image.py` | KEEP | `modelexport/inference/processors/image.py` | Image processor |
| `src/processors/audio.py` | KEEP | `modelexport/inference/processors/audio.py` | Audio processor |
| `src/processors/video.py` | KEEP | `modelexport/inference/processors/video.py` | Video processor |
| `src/processors/multimodal.py` | KEEP | `modelexport/inference/processors/multimodal.py` | Multimodal processor |
| `src/onnx_config/__init__.py` | KEEP | `modelexport/inference/config/__init__.py` | Config module |
| `src/onnx_config/universal_config.py` | KEEP | `modelexport/inference/config/universal_config.py` | Universal config |
| `src/onnx_config/task_detector.py` | KEEP | `modelexport/inference/config/task_detector.py` | Task detection |
| `src/onnx_config/shape_inference.py` | KEEP | `modelexport/inference/config/shape_inference.py` | Shape inference |
| `src/onnx_config/input_generator.py` | KEEP | `modelexport/inference/config/input_generator.py` | Input generation |
| `src/onnx_config/patterns.py` | KEEP | `modelexport/inference/config/patterns.py` | Pattern matching |

## Test Files (`tests/`)

| File | Action | Destination | Notes |
|------|--------|-------------|-------|
| `tests/__init__.py` | KEEP | `tests/inference/__init__.py` | Test module init |
| `tests/conftest.py` | KEEP | `tests/inference/conftest.py` | Pytest fixtures |
| `tests/test_utils.py` | KEEP | `tests/inference/test_utils.py` | Test utilities |
| `tests/test_onnx_auto_processor.py` | KEEP | `tests/inference/test_onnx_auto_processor.py` | Main test suite |
| `tests/README.md` | ARCHIVE | `docs/archive/inference/test_implementation.md` | Historical reference |
| `tests/TEST_SUITE_SUMMARY.md` | ARCHIVE | `docs/archive/inference/test_summary.md` | Test documentation |

## Example Files (`examples/`)

| File | Action | Destination | Notes |
|------|--------|-------------|-------|
| `examples/README.md` | REFACTOR | `examples/inference/README.md` | Update paths |
| `examples/onnx_inference_example.py` | REFACTOR | `examples/inference/basic_onnx_inference.py` | Update imports |
| `examples/multimodal_example.py` | REFACTOR | `examples/inference/multimodal_inference.py` | Update imports |
| `examples/test_onnx_inference.py` | KEEP | `examples/inference/test_onnx_inference.py` | Verification script |
| `examples/ONNX_AUTO_PROCESSOR_EXAMPLES.md` | ARCHIVE | `docs/archive/inference/examples_design.md` | Design notes |

## Documentation (`docs/`)

| File | Action | Destination | Notes |
|------|--------|-------------|-------|
| `docs/high_level_design.md` | REFACTOR | `docs/inference/architecture.md` | Update to production |
| `docs/onnx_auto_processor_design.md` | REFACTOR | `docs/inference/processor_design.md` | Production design |
| `docs/onnx_auto_processor_test_design.md` | REFACTOR | `docs/inference/testing_guide.md` | Testing documentation |
| `docs/processor_metadata_requirements.md` | ARCHIVE | `docs/archive/inference/metadata_research.md` | Research notes |
| `docs/phase1_to_production_plan.md` | KEEP | `docs/inference/migration_plan.md` | Migration guide |
| `docs/inference_module_benefits.md` | ARCHIVE | `docs/archive/inference/naming_rationale.md` | Decision record |
| `docs/MIGRATION_FINAL.md` | KEEP | `docs/inference/migration_guide.md` | Final migration plan |
| `docs/FILE_MIGRATION_TABLE.md` | KEEP | This file - for reference | Migration tracking |
| `docs/TODO.md` | DELETE | - | Tasks completed |
| `docs/optimum_config_generation_progress.md` | ARCHIVE | `docs/archive/inference/config_progress.md` | Historical progress |
| `docs/ENHANCED_PIPELINE_SOLUTION.md` | ARCHIVE | `docs/archive/inference/pipeline_design.md` | Design exploration |
| `docs/FIXED_SHAPE_SOLUTION.md` | ARCHIVE | `docs/archive/inference/shape_optimization.md` | Research findings |
| `docs/design/` | ARCHIVE | `docs/archive/inference/design/` | All design explorations |

## Root Level Test Scripts

### ⚠️ IMPORTANT: Test Coverage Analysis
After detailed review, these experimental tests contain **critical production validation** not covered in the main test suite:

**Critical Gaps in Main Test Suite:**
1. **End-to-End Workflows**: No testing of complete export→config→deployment cycle
2. **Real ONNX Export**: Main suite uses mocks, doesn't test actual ONNX export
3. **Optimum Integration**: Missing validation with real ORTModel classes
4. **Shape Handling**: Mock-based tests don't validate real shape constraints
5. **Pipeline Integration**: Limited testing of transformers pipeline integration

| File | Action | Destination | Notes |
|------|--------|-------------|-------|
| `final_validation_test.py` | REFACTOR | `tests/inference/integration/test_optimum_validation.py` | **CRITICAL**: End-to-end workflow validation |
| `test_clean_onnx_optimum.py` | REFACTOR | `tests/inference/integration/test_clean_export.py` | **CRITICAL**: Complete export→config→validation |
| `test_universal_config.py` | REFACTOR | `tests/inference/integration/test_onnx_export.py` | **HIGH**: ONNX export workflow tests |
| `test_existing_onnx.py` | REFACTOR | `tests/inference/integration/test_config_addition.py` | **HIGH**: Config file addition workflow |
| `test_ort_pipeline.py` | REFACTOR | `tests/inference/integration/test_ort_pipeline.py` | **HIGH**: ORTModel pipeline integration |
| `test_fixed_shape_tokenizer.py` | REFACTOR | `tests/inference/test_shape_handling.py` | **HIGH**: Real shape constraint tests |
| `test_enhanced_pipeline.py` | REFACTOR | `tests/inference/test_enhanced_pipeline.py` | **HIGH**: Enhanced pipeline API tests |
| `test_auto_shape_detection.py` | REFACTOR | `tests/inference/test_shape_detection.py` | **HIGH**: Auto shape inference tests |
| `test_bert_model_ort.py` | REFACTOR | `tests/inference/test_model_compatibility.py` | **MEDIUM**: Model-specific tests |
| `test_export_simple.py` | REFACTOR | `tests/inference/test_basic_export.py` | **MEDIUM**: Basic export validation |
| `test_pipeline_with_fixed_tokenizer.py` | REFACTOR | `tests/inference/test_pipeline_integration.py` | **MEDIUM**: Pipeline parameter tests |
| `test_processor_as_tokenizer.py` | DELETE | - | Can be merged into pipeline tests |
| `test_simple_auto_tokenizer.py` | DELETE | - | Basic usage, covered elsewhere |
| `ortmodel_analysis.py` | DELETE | - | One-time analysis |

## Root Level Documentation

| File | Action | Destination | Notes |
|------|--------|-------------|-------|
| `README.md` | REFACTOR | `docs/inference/README.md` | Main documentation |
| `PROJECT_OVERVIEW.md` | ARCHIVE | `docs/archive/inference/project_overview.md` | Historical context |
| `FINAL_VALIDATION_SUMMARY.md` | ARCHIVE | `docs/archive/inference/validation_summary.md` | Validation results |
| `INVESTIGATION_RESULTS.md` | ARCHIVE | `docs/archive/inference/investigation_results.md` | Research findings |
| `ONNX_AUTO_PROCESSOR_TEST_REPORT.md` | ARCHIVE | `docs/archive/inference/test_report.md` | Test implementation |

## Notebooks (`notebooks/`)

| File | Action | Destination | Notes |
|------|--------|-------------|-------|
| `notebooks/README.md` | REFACTOR | `docs/archive/inference/notebooks/README.md` | Notebook documentation |
| `notebooks/BERT_ONNX_README.md` | ARCHIVE | `docs/archive/inference/notebooks/bert_onnx_demo_guide.md` | Production demo documentation |
| `notebooks/-.md` | DELETE | - | Empty/broken file |
| `notebooks/config_only_demo_executed.ipynb` | ARCHIVE | `docs/archive/inference/notebooks/config_validation.ipynb` | Research value |
| `notebooks/optimum_feasibility_demo_executed.ipynb` | ARCHIVE | `docs/archive/inference/notebooks/optimum_research.ipynb` | Core research |
| `notebooks/optimum_feasibility_demo.ipynb` | DELETE | - | Duplicate (unexecuted version) |
| `notebooks/onnx_pipeline_integration_demo.ipynb` | ARCHIVE | `docs/archive/inference/notebooks/pipeline_integration.ipynb` | Integration research |
| `notebooks/optimum_infer_onnx_bert.ipynb` | DELETE | - | Example covered elsewhere |
| `notebooks/understanding_onnxconfig.ipynb` | DELETE | - | Research complete |
| `notebooks/models/` | DELETE | - | Duplicate test models directory |
| `notebooks/models/**/*` | DELETE | - | All duplicate model files |
| `notebooks/.ipynb_checkpoints/` | DELETE | - | Jupyter checkpoint directory |
| `notebooks/.ipynb_checkpoints/**/*` | DELETE | - | All checkpoint files |

## Models (`models/`)

| File/Folder | Action | Destination | Notes |
|------|--------|-------------|-------|
| `models/bert-tiny-optimum/` | KEEP | `tests/inference/fixtures/models/bert-tiny-optimum/` | Test model |
| `models/bert-tiny-optimum/config.json` | KEEP | `tests/inference/fixtures/models/bert-tiny-optimum/config.json` | Model config |
| `models/bert-tiny-optimum/model.onnx` | KEEP | `tests/inference/fixtures/models/bert-tiny-optimum/model.onnx` | ONNX model |
| `models/bert-tiny-optimum/model_htp_metadata.json` | KEEP | `tests/inference/fixtures/models/bert-tiny-optimum/model_htp_metadata.json` | HTP metadata |
| `models/bert-tiny-optimum/special_tokens_map.json` | KEEP | `tests/inference/fixtures/models/bert-tiny-optimum/special_tokens_map.json` | Tokenizer tokens |
| `models/bert-tiny-optimum/tokenizer.json` | KEEP | `tests/inference/fixtures/models/bert-tiny-optimum/tokenizer.json` | Tokenizer |
| `models/bert-tiny-optimum/tokenizer_config.json` | KEEP | `tests/inference/fixtures/models/bert-tiny-optimum/tokenizer_config.json` | Tokenizer config |
| `models/bert-tiny-optimum/vocab.txt` | KEEP | `tests/inference/fixtures/models/bert-tiny-optimum/vocab.txt` | Vocabulary |
| `models/bert.onnx` | DELETE | - | Duplicate test artifact |
| `models/bert_hierarchical_graph.graphml` | DELETE | - | Graph visualization, not needed |
| `models/bert_hierarchical_graph.onnxdata` | DELETE | - | Graph data, not needed |
| `models/bert_htp_export_report.md` | DELETE | - | Test report |
| `models/bert_htp_metadata.json` | DELETE | - | Duplicate metadata |
| `models/config-only-test/` | DELETE | - | Test artifacts directory |
| `models/config-only-test/**/*` | DELETE | - | All config-only test files (various models) |

## Experiments Folder

| File/Folder | Action | Destination | Notes |
|------|--------|-------------|-------|
| `experiments/README.md` | ARCHIVE | `docs/archive/inference/experiments_readme.md` | Research notes |
| `experiments/test_config_only_copy.py` | DELETE | - | Test script |
| `experiments/test_optimum_requirement.py` | DELETE | - | Test script |
| `experiments/tez-144_onnx_automodel_infer/` | DELETE | - | Duplicate nested folder |
| `experiments/tez-144_onnx_automodel_infer/models/` | DELETE | - | Empty models directory |

## Temporary Files

| File/Folder | Action | Destination | Notes |
|------|--------|-------------|-------|
| `temp/` | DELETE | - | Temporary files directory |
| `temp/bert-tiny_htp_export_report.md` | DELETE | - | Test export report |
| `temp/bert-tiny_htp_metadata.json` | DELETE | - | Test metadata |
| `temp/quick_performance_test.py` | DELETE | - | Performance test script |
| `temp/*.ipynb` | DELETE | - | All temporary notebooks (21 files) |
| `temp/temp/` | DELETE | - | Nested temp directory |
| `temp/temp/bert-dynamic-export/` | DELETE | - | Dynamic export test artifacts |
| `temp/temp/bert-dynamic-export/**/*` | DELETE | - | Model files and HF cache |
| `src/__pycache__/` | DELETE | - | Python cache |
| `src/processors/__pycache__/` | DELETE | - | Python cache |
| `src/onnx_config/__pycache__/` | DELETE | - | Python cache |
| `tests/__pycache__/` | DELETE | - | Python cache |
| `tests/.pytest_cache/` | DELETE | - | Pytest cache |
| `notebooks/.ipynb_checkpoints/` | DELETE | - | Jupyter checkpoints |

## External Experimental Tests (Outside Project)

| File/Folder | Action | Destination | Notes |
|------|--------|-------------|-------|
| `../tests/onnx_infer/` | REFACTOR | `tests/inference/experimental/` | External test suite |
| `../tests/onnx_infer/README.md` | REFACTOR | `tests/inference/experimental/README.md` | Test documentation |
| `../tests/onnx_infer/TEST_SUMMARY.md` | ARCHIVE | `docs/archive/inference/experimental_test_summary.md` | Test results |
| `../tests/onnx_infer/conftest.py` | REFACTOR | `tests/inference/experimental/conftest.py` | Pytest config |
| `../tests/onnx_infer/pytest.ini` | REFACTOR | `tests/inference/experimental/pytest.ini` | Pytest settings |
| `../tests/onnx_infer/test_auto_model_loader.py` | REFACTOR | `tests/inference/experimental/test_auto_model_loader.py` | AutoModel tests |
| `../tests/onnx_infer/test_enhanced_pipeline.py` | REFACTOR | `tests/inference/experimental/test_enhanced_pipeline.py` | Pipeline tests |
| `../tests/onnx_infer/test_onnx_tokenizer.py` | REFACTOR | `tests/inference/experimental/test_onnx_tokenizer.py` | Tokenizer tests |
| `../tests/onnx_infer/test_optimum_onnx_integration.py` | REFACTOR | `tests/inference/experimental/test_optimum_integration.py` | Optimum integration |
| `../tests/onnx_infer/test_pipeline_tasks.py` | REFACTOR | `tests/inference/experimental/test_pipeline_tasks.py` | Task pipeline tests |
| `../tests/onnx_infer/test_runner.py` | REFACTOR | `tests/inference/experimental/test_runner.py` | Test runner |
| `../tests/onnx_infer/test_sanity.py` | REFACTOR | `tests/inference/experimental/test_sanity.py` | Sanity tests |
| `../tests/onnx_infer/test_smoke.py` | REFACTOR | `tests/inference/experimental/test_smoke.py` | Smoke tests |

## Final Migration Summary

| Action | Count | Description |
|--------|-------|-------------|
| **KEEP** | 34 | Core production files to migrate |
| **REFACTOR** | 32 | Files needing updates before migration (11 more test files to extract) |
| **ARCHIVE** | 25 | Research/historical value files |
| **DELETE** | 139+ | Deprecated, duplicate, or temporary files (11 test files moved to REFACTOR) |

### Import Path Updates Required

All files marked as KEEP or REFACTOR will need import updates:

| Old Import | New Import |
|------------|------------|
| `from src.onnx_auto_processor import` | `from modelexport.inference import` |
| `from src.processors.text import` | `from modelexport.inference.processors.text import` |
| `from src.onnx_processor_types import` | `from modelexport.inference.types import` |
| `from src.enhanced_pipeline import` | `from modelexport.inference.pipeline import` |
| `sys.path.append('../src')` | Remove (use proper imports) |