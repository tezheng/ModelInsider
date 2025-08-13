# TEZ-144 File Migration Table

**Complete Inventory**: This table accounts for ALL files in the experimental directory tree, including hidden files, cache directories, temporary files, and external test suites.

**Last Updated**: 2025-08-13 | **Total Files Inventoried**: 230+

## Legend
- **KEEP**: Move to production location
- **ARCHIVE**: Keep for reference in docs/research/
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
| `src/inference_utils.py` | DELETE | - | Functionality integrated elsewhere |
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
| `tests/README.md` | ARCHIVE | `docs/research/test_implementation.md` | Historical reference |
| `tests/TEST_SUITE_SUMMARY.md` | ARCHIVE | `docs/research/test_summary.md` | Test documentation |

## Example Files (`examples/`)

| File | Action | Destination | Notes |
|------|--------|-------------|-------|
| `examples/README.md` | REFACTOR | `examples/inference/README.md` | Update paths |
| `examples/onnx_inference_example.py` | REFACTOR | `examples/inference/basic_onnx_inference.py` | Update imports |
| `examples/multimodal_example.py` | REFACTOR | `examples/inference/multimodal_inference.py` | Update imports |
| `examples/test_onnx_inference.py` | KEEP | `examples/inference/test_onnx_inference.py` | Verification script |
| `examples/ONNX_AUTO_PROCESSOR_EXAMPLES.md` | ARCHIVE | `docs/research/examples_design.md` | Design notes |

## Documentation (`docs/`)

| File | Action | Destination | Notes |
|------|--------|-------------|-------|
| `docs/high_level_design.md` | REFACTOR | `docs/inference/architecture.md` | Update to production |
| `docs/onnx_auto_processor_design.md` | REFACTOR | `docs/inference/processor_design.md` | Production design |
| `docs/onnx_auto_processor_test_design.md` | REFACTOR | `docs/inference/testing_guide.md` | Testing documentation |
| `docs/processor_metadata_requirements.md` | ARCHIVE | `docs/research/metadata_research.md` | Research notes |
| `docs/phase1_to_production_plan.md` | KEEP | `docs/inference/migration_plan.md` | Migration guide |
| `docs/inference_module_benefits.md` | ARCHIVE | `docs/research/naming_rationale.md` | Decision record |
| `docs/MIGRATION_FINAL.md` | KEEP | `docs/inference/migration_guide.md` | Final migration plan |
| `docs/FILE_MIGRATION_TABLE.md` | KEEP | This file - for reference | Migration tracking |
| `docs/TODO.md` | DELETE | - | Tasks completed |
| `docs/optimum_config_generation_progress.md` | ARCHIVE | `docs/research/config_progress.md` | Historical progress |
| `docs/ENHANCED_PIPELINE_SOLUTION.md` | ARCHIVE | `docs/research/pipeline_design.md` | Design exploration |
| `docs/FIXED_SHAPE_SOLUTION.md` | ARCHIVE | `docs/research/shape_optimization.md` | Research findings |
| `docs/design/` | ARCHIVE | `docs/research/design/` | All design explorations |

## Root Level Test Scripts

| File | Action | Destination | Notes |
|------|--------|-------------|-------|
| `final_validation_test.py` | REFACTOR | `tests/inference/integration/test_optimum_validation.py` | Convert to pytest |
| `test_clean_onnx_optimum.py` | REFACTOR | `tests/inference/integration/test_clean_export.py` | Convert to pytest |
| `test_bert_model_ort.py` | DELETE | - | Covered by main tests |
| `test_export_simple.py` | DELETE | - | Basic test, covered |
| `test_fixed_shape_tokenizer.py` | DELETE | - | Covered in processor tests |
| `test_ort_pipeline.py` | DELETE | - | Covered in main suite |
| `test_pipeline_with_fixed_tokenizer.py` | DELETE | - | Duplicate functionality |
| `test_processor_as_tokenizer.py` | DELETE | - | Covered in main suite |
| `test_simple_auto_tokenizer.py` | DELETE | - | Basic, covered |
| `test_auto_shape_detection.py` | DELETE | - | Covered in processor tests |
| `test_enhanced_pipeline.py` | DELETE | - | Integrated in main suite |
| `test_existing_onnx.py` | DELETE | - | Legacy validation |
| `test_universal_config.py` | DELETE | - | Config tests integrated |
| `ortmodel_analysis.py` | DELETE | - | One-time analysis |

## Root Level Documentation

| File | Action | Destination | Notes |
|------|--------|-------------|-------|
| `README.md` | REFACTOR | `docs/inference/README.md` | Main documentation |
| `PROJECT_OVERVIEW.md` | ARCHIVE | `docs/research/project_overview.md` | Historical context |
| `FINAL_VALIDATION_SUMMARY.md` | ARCHIVE | `docs/research/validation_summary.md` | Validation results |
| `INVESTIGATION_RESULTS.md` | ARCHIVE | `docs/research/investigation_results.md` | Research findings |
| `ONNX_AUTO_PROCESSOR_TEST_REPORT.md` | ARCHIVE | `docs/research/test_report.md` | Test implementation |

## Notebooks (`notebooks/`)

| File | Action | Destination | Notes |
|------|--------|-------------|-------|
| `notebooks/README.md` | REFACTOR | `docs/research/notebooks/README.md` | Notebook documentation |
| `notebooks/BERT_ONNX_README.md` | ARCHIVE | `docs/research/notebooks/bert_onnx_demo_guide.md` | Production demo documentation |
| `notebooks/-.md` | DELETE | - | Empty/broken file |
| `notebooks/config_only_demo_executed.ipynb` | ARCHIVE | `docs/research/notebooks/config_validation.ipynb` | Research value |
| `notebooks/optimum_feasibility_demo_executed.ipynb` | ARCHIVE | `docs/research/notebooks/optimum_research.ipynb` | Core research |
| `notebooks/optimum_feasibility_demo.ipynb` | DELETE | - | Duplicate (unexecuted version) |
| `notebooks/onnx_pipeline_integration_demo.ipynb` | ARCHIVE | `docs/research/notebooks/pipeline_integration.ipynb` | Integration research |
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
| `experiments/README.md` | ARCHIVE | `docs/research/experiments_readme.md` | Research notes |
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
| `../tests/onnx_infer/TEST_SUMMARY.md` | ARCHIVE | `docs/research/experimental_test_summary.md` | Test results |
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

## Summary Statistics

| Action | Count | Description |
|--------|-------|-------------|
| **KEEP** | 33 | Core production files to migrate |
| **REFACTOR** | 21 | Files needing updates before migration |
| **ARCHIVE** | 25 | Research/historical value files |
| **DELETE** | 150+ | Deprecated, duplicate, or temporary files |

### Detailed Counts by Category

| Category | KEEP | REFACTOR | ARCHIVE | DELETE |
|----------|------|----------|---------|--------|
| **Source Code** | 19 | 0 | 0 | 1 |
| **Tests** | 4 | 13 | 2 | 1 |
| **Examples** | 1 | 3 | 1 | 0 |
| **Documentation** | 1 | 2 | 13 | 1 |
| **Models** | 8 | 0 | 0 | 9 |
| **Notebooks** | 0 | 1 | 5 | 10 |
| **Root Level** | 0 | 1 | 4 | 17 |
| **Experimental** | 0 | 0 | 2 | 5 |
| **Temporary** | 0 | 0 | 0 | 100+ |
| **External Tests** | 0 | 13 | 1 | 0 |

## Migration Priority

### Phase 1: Critical Files (KEEP)
1. Core source files in `src/`
2. Main test suite files
3. Essential documentation

### Phase 2: Refactor Files
1. Update import paths in examples and documentation
2. Refactor external experimental tests
3. Remove experimental references
4. Convert to production format

### Phase 3: Archive Research
1. Move research notebooks to `docs/research/`
2. Archive design explorations and reports
3. Keep historical documentation and test summaries

### Phase 4: Cleanup
1. Delete deprecated test scripts (17 root-level files)
2. Remove temporary files and directories
3. Clean up duplicate models and cache files
4. Remove Jupyter checkpoints and Python cache

## Import Path Updates Required

All files marked as KEEP or REFACTOR will need import updates:

| Old Import | New Import |
|------------|------------|
| `from src.onnx_auto_processor import` | `from modelexport.inference import` |
| `from src.processors.text import` | `from modelexport.inference.processors.text import` |
| `from src.onnx_processor_types import` | `from modelexport.inference.types import` |
| `from src.enhanced_pipeline import` | `from modelexport.inference.pipeline import` |
| `sys.path.append('../src')` | Remove (use proper imports) |

## Notes

1. **Test Models**: The `bert-tiny-optimum` model should be kept as a test fixture
2. **Documentation**: Consolidate related docs during migration
3. **Examples**: All examples need import path updates
4. **Research**: Archive valuable research for future reference
5. **Cleanup**: Delete all experimental scripts that are covered by the main test suite