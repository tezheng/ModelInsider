# Iteration 1: Research and Planning

**Goal**: Research optimum project and plan FX implementation

## Findings
- Optimum uses model patching but no hierarchy preservation
- FX patterns available: node.target for module paths, graph_module.get_submodule()
- ONNX metadata can be added via node attributes
- No existing FX-to-ONNX hierarchy preservation found

## Key Insights
- Our approach is novel - optimum doesn't preserve hierarchy
- Can use FX node annotation patterns from optimum/fx/optimization
- Need custom FX→ONNX mapping with hierarchy preservation
- Must handle torch.nn filtering during FX analysis

## Next Steps
1. Create core FX hierarchy exporter
2. Implement module filtering with exceptions
3. Build FX→ONNX mapping with tag preservation