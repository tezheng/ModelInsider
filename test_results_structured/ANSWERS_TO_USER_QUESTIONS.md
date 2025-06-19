# Answers to Critical User Questions

## 🎯 Your Questions and My Findings

### 1. **"You haven't asked me about the I/O shape of the model, how did you make the comparison?"**

**You're absolutely right!** I made assumptions without asking:

#### ❌ What I Assumed:
```python
# I blindly used this for all tests:
inputs = tokenizer("Hello world", return_tensors='pt')
# Shape: input_ids=[1, 8], attention_mask=[1, 8], token_type_ids=[1, 8]
```

#### ❓ What I Should Have Asked:
1. **Input shapes to test**: What sequence lengths? Batch sizes?
2. **Dynamic vs fixed shapes**: Should I test with dynamic axes?
3. **Edge cases**: Empty inputs? Maximum sequence length?
4. **Batch variations**: Single vs multi-batch inputs?

#### ✅ What I Found Through Testing:
```python
# Different I/O shapes I tested:
'single_token': input_ids=[1, 3]      # "Hi" -> [CLS] Hi [SEP]
'medium_sequence': input_ids=[1, 11]  # Longer text
'batch_different_lengths': input_ids=[3, 7]  # Batch with padding
```

**Result**: All shape variations worked, but I should have asked about your specific requirements.

---

### 2. **"Which ONNX file is exported with unmodified torch.export? Which ONNX is the tagged ONNX? Which test cases are related with comparison of these two?"**

#### 📄 **File Classification**:

**Unmodified ONNX (Baseline)**:
- ✅ `baseline_unmodified.onnx` (304 nodes, 0 hierarchy attributes)
- ✅ `hierarchy_baseline.onnx` (from test_baseline_comparison.py)

**Tagged ONNX (With Hierarchy)**:
- ✅ `cli_test.onnx` (186 nodes, 90 with hierarchy attributes)
- ✅ `bounded_test.onnx` (186 nodes, 90 with hierarchy attributes)
- ✅ `hierarchy_tagged.onnx` (from test_baseline_comparison.py)

#### 🧪 **Test Cases for Comparison**:

**Direct Comparison Tests**:
1. `test_export_parameters_identical` - Uses identical export parameters
2. `test_functional_equivalence` - Tests I/O compatibility
3. `test_hierarchy_preservation_vs_baseline` - Validates hierarchy doesn't break functionality

**Files Preserved for Inspection**:
```
test_results_structured/exports/
├── baseline_unmodified.onnx      # ← Standard torch.onnx.export
├── cli_test.onnx                 # ← Our HierarchyExporter
├── bounded_test.onnx             # ← Bounded propagation test
└── *_hierarchy.json              # ← Sidecar metadata
```

---

## 🚨 **Critical Issue Discovered**

### **ONNX Validation Failure**:
```
ValidationError: Unrecognized attribute: hierarchy_tags for operator Constant
```

**What this means**:
- ✅ Our hierarchy export works
- ❌ ONNX's standard checker rejects custom attributes
- ❓ **Question for you**: Should we use ONNX extensions or custom metadata instead?

---

## 📊 **Key Findings from Comparison**

### **Node Count Differences**:
```
Baseline (unmodified): 304 nodes
Tagged (our export):   186 nodes
Difference:            -118 nodes
```

**Possible reasons**:
1. **Different optimization settings** in our exporter
2. **Constant folding differences**
3. **Our export uses different default parameters**

#### ❓ **Questions for You**:
1. **Should the node counts be identical?**
2. **Are different optimizations acceptable as long as functionality is preserved?**
3. **Should we match PyTorch's exact export parameters?**

### **I/O Compatibility**: ✅ **PASSED**
```
Inputs:  baseline=3, tagged=3  ✅
Outputs: baseline=2, tagged=2  ✅
```

### **Hierarchy Preservation**: ✅ **PASSED**
```
Baseline hierarchy attrs: 0    ✅ (expected)
Tagged hierarchy attrs:   90   ✅ (working)
```

---

## 🎯 **Test Coverage Gaps I Identified**

### **What I Should Have Asked About**:

1. **I/O Shape Requirements**:
   - What sequence lengths to test?
   - Batch size variations?
   - Dynamic vs fixed dimensions?

2. **Comparison Methodology**:
   - Should functional outputs be identical?
   - Is numerical precision comparison needed?
   - How to handle optimization differences?

3. **ONNX Compatibility**:
   - Should custom attributes validate with ONNX checker?
   - Are ONNX extensions preferred over custom attributes?

4. **Export Parameter Alignment**:
   - Should we use identical export parameters?
   - How to handle optimization differences?

---

## 📋 **Test Files Reference**

### **Baseline vs Tagged Comparison**:
- `tests/test_baseline_comparison.py` - Comprehensive comparison tests
- `test_results_structured/reports/model_comparison.json` - Detailed analysis

### **Preserved Artifacts**:
```
test_results_structured/
├── exports/
│   ├── baseline_unmodified.onnx     # Standard export
│   ├── cli_test.onnx                # Tagged export  
│   └── *_hierarchy.json             # Metadata
├── reports/
│   ├── model_comparison.json        # Node count analysis
│   └── baseline_vs_tagged_comparison.json  # Detailed comparison
└── COMPREHENSIVE_TEST_SUMMARY.md    # Full results
```

---

## ❓ **Questions Back to You**

1. **I/O Shape Testing**: What specific input shapes/configurations should I test?

2. **ONNX Validation**: How should I handle custom attributes that don't validate with ONNX checker?

3. **Node Count Differences**: Is it acceptable that our export has different optimization (186 vs 304 nodes)?

4. **Comparison Criteria**: What constitutes "equivalent" models - functional similarity or identical structure?

5. **Export Parameters**: Should I ensure identical export parameters between baseline and tagged versions?

Your questions revealed important gaps in my testing methodology. Thank you for pointing these out!