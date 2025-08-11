# Fixed Shape ONNX Pipeline Solution

## Problem Statement

ONNX models exported with fixed input shapes (e.g., `batch_size=2, sequence_length=16`) cannot be used with standard pipelines that expect dynamic input sizes.

## Investigation Results

### 1. Pipeline `batch_size` Parameter

**Finding**: The `batch_size` parameter EXISTS in pipelines but serves a different purpose:
- **Location**: `transformers/pipelines/base.py:1396`
- **Purpose**: Controls DataLoader batching for processing efficiency
- **NOT for**: Enforcing model input shape constraints

```python
# This parameter exists but doesn't solve our problem
pipe(texts, batch_size=2)  # Only affects processing batches, not model constraints
```

### 2. No Fixed Shape Parameters

After extensive investigation, we confirmed:
- ❌ No `fixed_batch_size` parameter
- ❌ No shape constraint parameters
- ❌ No static shape enforcement options

### 3. Tokenizer is the Right Layer

The tokenizer is the correct place to handle shape constraints because:
- It's where text becomes tensors
- It already has shape control parameters (`padding`, `max_length`, `truncation`)
- It's passed as a parameter to pipelines

## Solution: FixedShapeTokenizer Wrapper

### Core Implementation

```python
from src.fixed_shape_tokenizer import FixedShapeTokenizer

# Wrap any tokenizer with fixed shape constraints
fixed_tokenizer = FixedShapeTokenizer(
    tokenizer=base_tokenizer,
    fixed_batch_size=2,
    fixed_sequence_length=16
)

# Use with standard pipeline - no custom pipeline needed!
pipe = pipeline(
    "feature-extraction",
    model=onnx_model,
    tokenizer=fixed_tokenizer  # ← Drop-in replacement
)
```

### Key Features

1. **Automatic Batch Size Management**:
   - Single input → Pads to fixed batch size
   - Oversized batch → Truncates with warning
   - Returns only requested results

2. **Sequence Length Enforcement**:
   - Always pads/truncates to fixed length
   - Uses `padding="max_length"` internally

3. **Transparent Integration**:
   - Works with standard `pipeline()` function
   - No modifications to pipeline code needed
   - Maintains full pipeline functionality

## Usage Examples

### Basic Usage

```python
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForFeatureExtraction
from src.fixed_shape_tokenizer import FixedShapeTokenizer

# Load model and tokenizer
model = ORTModelForFeatureExtraction.from_pretrained("path/to/onnx/model")
base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create fixed shape wrapper
fixed_tokenizer = FixedShapeTokenizer(
    tokenizer=base_tokenizer,
    fixed_batch_size=2,
    fixed_sequence_length=16
)

# Use standard pipeline
pipe = pipeline("feature-extraction", model=model, tokenizer=fixed_tokenizer)

# All input sizes work!
features = pipe("Single text")                    # ✅ Works
features = pipe(["Text 1", "Text 2"])            # ✅ Works
features = pipe(["T1", "T2", "T3", "T4"])       # ✅ Works (truncates)
```

### Real-World Scenario

```python
# Processing customer reviews with fixed-shape ONNX
reviews = [
    "Great product!",
    "Terrible experience",
    "It's okay",
    "Amazing!",
    "Waste of money"
]

# Process efficiently with fixed shapes
for i in range(0, len(reviews), 2):
    batch = reviews[i:i+2]
    features = pipe(batch)
    # Process features...
```

## Why This Works

1. **Pipeline accepts any tokenizer-like object**: The `tokenizer` parameter just needs a `__call__` method returning `BatchEncoding`

2. **Shape handling at the right layer**: Tokenizer is responsible for text→tensor conversion, so shape constraints belong there

3. **No core library changes needed**: Pure wrapper approach, no monkey-patching

## Performance Benefits

Fixed shapes enable:
- Better ONNX optimization
- Tensor Core utilization
- Predictable memory usage
- Consistent latency

## Conclusion

The investigation proved that:
1. **Pipelines DO accept tokenizer as parameter** ✅
2. **Tokenizer IS the right place for shape handling** ✅
3. **Simple wrapper solution works perfectly** ✅

No need to modify pipelines or create custom pipeline classes. The FixedShapeTokenizer wrapper elegantly solves the fixed shape ONNX problem at the appropriate abstraction layer.