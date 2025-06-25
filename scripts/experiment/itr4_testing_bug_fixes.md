# Iteration 4: Testing and Bug Fixes

**Goal**: Run tests and fix identified issues

## Fixes Implemented
- ✅ Added Linear to torch.nn exceptions (was missing)
- ✅ Fixed hierarchy path building (removed double filtering)
- ✅ Improved BERT tracing with concrete_args
- ✅ Fixed filtering test to check actual hierarchy paths vs module types

## Test Results
- ✅ Simple Model: Now working! 2 hierarchy nodes detected correctly
- ✅ torch.nn Filtering: Working correctly (4 hierarchy paths created)
- ✅ FX Graph Analysis: Working (42.9% coverage, proper mapping)
- ❌ BERT Model: FX tracing limitation (control flow issue)

## Critical Discovery - FX Limitation
BERT models use dynamic control flow (`if use_sdpa_attention_masks and attention_mask.dim() == 2:`) 
which is incompatible with FX symbolic tracing. This is a fundamental limitation, not a bug.

## FX Tracing Errors
1. "symbolically traced variables cannot be used as inputs to control flow"
2. Complex parameter validation in transformers models
3. Dynamic behavior that FX cannot trace symbolically

## Next Steps
1. Implement graceful fallback for untraceable models
2. Focus on simpler models where FX excels
3. Document FX limitations clearly
4. Consider hybrid approach: FX for simple models, HTP for complex ones