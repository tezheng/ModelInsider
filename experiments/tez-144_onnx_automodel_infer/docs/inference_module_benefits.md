# Why `modelexport/inference/` is the Right Choice

## Naming Rationale

The `modelexport/inference/` structure is superior to `modelexport/onnx_processors/` for several reasons:

### 1. **Broader Scope**
- **inference/** captures the full purpose: enabling high-performance inference
- Not limited to just "processors" - includes pipelines, optimizations, and integrations
- Future-proof for adding non-processor inference capabilities

### 2. **Clear Intent**
- Immediately communicates that this module is about **using** exported models
- Natural complement to the export functionality
- Aligns with user mental model: "export → inference"

### 3. **Better Organization**
```
modelexport/
├── export/         # Model export functionality
├── inference/      # Model inference functionality (NEW)
├── conversion/     # Format conversion utilities
└── optimization/   # Model optimization tools
```

This creates a clean, logical structure where each module has a clear purpose.

### 4. **API Clarity**
```python
# Clear and intuitive imports
from modelexport.inference import ONNXAutoProcessor
from modelexport.inference import pipeline
from modelexport.inference.processors import TextProcessor

# vs. the more narrow:
from modelexport.onnx_processors import ONNXAutoProcessor  # Limited scope
```

### 5. **Extensibility**
The `inference/` module can naturally grow to include:
- Runtime optimization strategies
- Batching and streaming utilities
- Model serving components
- Performance monitoring tools
- Multiple backend support (ONNX Runtime, TensorRT, etc.)

## Module Structure

```
modelexport/inference/
├── __init__.py                 # Main exports
├── onnx_auto_processor.py      # Core factory class
├── pipeline.py                 # Enhanced pipeline with data_processor
├── types.py                    # Type definitions and protocols
├── processors/                 # Processor implementations
│   ├── base.py                # Base processor class
│   ├── text.py                # Text processing (tokenization)
│   ├── image.py               # Image processing
│   ├── audio.py               # Audio processing
│   ├── video.py               # Video processing
│   └── multimodal.py          # Multimodal processing
├── optimizations/              # Runtime optimizations
│   ├── fixed_shape.py         # Fixed-shape optimization
│   ├── batching.py            # Dynamic batching
│   └── caching.py             # Result caching
├── backends/                   # Backend integrations
│   ├── onnxruntime.py         # ONNX Runtime backend
│   ├── optimum.py             # HuggingFace Optimum
│   └── tensorrt.py            # TensorRT backend (future)
└── utils/                      # Utility functions
    ├── metadata.py            # ONNX metadata extraction
    ├── validation.py          # Input/output validation
    └── benchmarking.py        # Performance measurement
```

## Integration Benefits

### 1. **Natural CLI Flow**
```bash
# Export and setup inference in one command
modelexport export bert-base-uncased model.onnx --with-inference

# Or separately
modelexport export bert-base-uncased model.onnx
modelexport inference setup model.onnx
```

### 2. **Clear Documentation Path**
- `docs/export/` - How to export models
- `docs/inference/` - How to use exported models
- `docs/optimization/` - How to optimize models

### 3. **Testing Organization**
```
tests/
├── test_export/       # Export functionality tests
├── test_inference/    # Inference functionality tests
└── test_integration/  # End-to-end tests
```

## User Benefits

1. **Intuitive Discovery**: Users looking for inference capabilities naturally look in `inference/`
2. **Complete Solution**: Not just processors, but a full inference toolkit
3. **Clear Separation**: Export vs. inference responsibilities are clearly separated
4. **Future Growth**: Room for adding serving, monitoring, and optimization features

## Migration Path

From: `experiments/tez-144_onnx_automodel_infer/`
To: `modelexport/inference/`

This creates a production-ready inference module that complements the existing export functionality perfectly.

## Conclusion

The `modelexport/inference/` structure:
- ✅ Better represents the module's purpose
- ✅ Provides clearer API organization  
- ✅ Allows for natural future expansion
- ✅ Creates intuitive user experience
- ✅ Maintains clean separation of concerns

This naming choice sets up the project for long-term success by creating a clear, extensible structure that users will find intuitive and developers will find maintainable.