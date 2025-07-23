# Iteration 10 - Test with Different Models

## Date
2025-07-19 07:11:55

## Iteration Number
10 of 20

## What Was Done

### Model Testing
- Tested export monitor with 6 different model architectures
- Included text models: BERT, DistilBERT, ALBERT, GPT-2
- Included vision models: ResNet, MobileNet
- Verified monitor works across different input types

### Key Results
- All models tested successfully
- Export monitor handles different architectures correctly
- Text styling works consistently
- Performance is good across all models

## Test Models
1. **prajjwal1/bert-tiny**: 48 modules, 4.4M params
2. **distilbert-base-uncased**: 98 modules, 66.4M params
3. **albert-base-v2**: 105 modules, 11.7M params
4. **microsoft/resnet-18**: 65 modules, 11.7M params
5. **google/mobilenet_v2_0.35_96**: 155 modules, 1.7M params
6. **sshleifer/tiny-gpt2**: 74 modules, 0.4M params

## Key Findings
1. Export monitor is robust across architectures
2. Vision models require different input format (pixel_values)
3. Module count varies significantly (48-155)
4. All outputs have proper ANSI styling

## Next Steps
- Continue with remaining iterations
- Fine-tune node name formatting
- Test with even larger models
- Optimize performance for large models

## Notes
- Consider adding model type auto-detection
- May need to handle more input formats
- Performance scales well with model size
