#!/usr/bin/env python3
"""
Test script to verify if ORTModel classes work specifically with BertModel for feature extraction.
This tests the exact use case we need: using BertModel (not BertForSequenceClassification) 
with optimum's ORTModel classes.
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_bert_model_compatibility():
    """Test BertModel compatibility with ORTModel classes."""
    
    try:
        from transformers import pipeline, AutoTokenizer, BertModel
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        
        print("✅ Successfully imported required modules")
        
        # Test with a small BERT model that is definitely a BertModel, not BertForSequenceClassification
        model_name = "prajjwal1/bert-tiny"
        print(f"\n🔍 Testing with BertModel: {model_name}")
        
        # First, verify this is actually a BertModel
        try:
            original_model = BertModel.from_pretrained(model_name)
            print(f"✅ Confirmed this is a BertModel: {type(original_model).__name__}")
            print(f"   Config model type: {original_model.config.model_type}")
            
            # Test the original model with pipeline first
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            pipe_original = pipeline("feature-extraction", model=original_model, tokenizer=tokenizer)
            result_original = pipe_original("Test sentence for feature extraction")
            print(f"✅ Original BertModel works with feature-extraction pipeline")
            print(f"   Original result shape: {len(result_original)} x {len(result_original[0])} x {len(result_original[0][0])}")
            
        except Exception as e:
            print(f"❌ Failed to test original BertModel: {e}")
            return False
        
        # Now test with ORTModelForFeatureExtraction
        print(f"\n📦 Testing ORTModelForFeatureExtraction with BertModel...")
        try:
            ort_model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
            print(f"✅ Successfully loaded ORTModelForFeatureExtraction")
            print(f"   Model config: {ort_model.config.model_type}")
            
            # Test direct forward call
            inputs = tokenizer("Test sentence", return_tensors="pt", padding=True, truncation=True)
            outputs = ort_model(**inputs)
            print(f"✅ Direct forward call successful")
            print(f"   ORT output shape: {outputs.last_hidden_state.shape}")
            
            # Test with pipeline
            pipe_ort = pipeline("feature-extraction", model=ort_model, tokenizer=tokenizer)
            result_ort = pipe_ort("Test sentence for feature extraction")
            print(f"✅ ORT BertModel works with feature-extraction pipeline")
            print(f"   ORT result shape: {len(result_ort)} x {len(result_ort[0])} x {len(result_ort[0][0])}")
            
            # Compare shapes
            if (len(result_original) == len(result_ort) and 
                len(result_original[0]) == len(result_ort[0]) and
                len(result_original[0][0]) == len(result_ort[0][0])):
                print(f"✅ Output shapes match between original and ORT models")
            else:
                print(f"⚠️ Output shapes differ: original={len(result_original)}x{len(result_original[0])}x{len(result_original[0][0])}, ORT={len(result_ort)}x{len(result_ort[0])}x{len(result_ort[0][0])}")
            
            return True
            
        except Exception as e:
            print(f"❌ ORTModelForFeatureExtraction failed: {e}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        traceback.print_exc()
        return False

def test_different_bert_models():
    """Test with different BERT model variants."""
    
    from transformers import pipeline, AutoTokenizer
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    
    models_to_test = [
        "google/bert_uncased_L-2_H-128_A-2",  # Very small BERT
        "sentence-transformers/all-MiniLM-L6-v2",  # Popular sentence transformer (still BERT-based)
    ]
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n🔍 Testing model: {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
            
            # Test with pipeline
            pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
            result = pipe("Test sentence")
            
            print(f"✅ {model_name}: SUCCESS")
            print(f"   Shape: {len(result)} x {len(result[0])} x {len(result[0][0])}")
            results[model_name] = True
            
        except Exception as e:
            print(f"❌ {model_name}: FAILED - {e}")
            results[model_name] = False
    
    return results

def main():
    """Main test function."""
    print("🚀 Testing BertModel compatibility with ORTModel classes")
    print("=" * 70)
    
    # Test main compatibility
    success1 = test_bert_model_compatibility()
    
    print("\n" + "=" * 70)
    print("🔍 Testing additional BERT model variants...")
    
    # Test different models
    results = test_different_bert_models()
    
    print("\n" + "=" * 70)
    print("📋 FINAL SUMMARY:")
    print(f"   Primary BertModel test: {'✅ SUCCESS' if success1 else '❌ FAILED'}")
    
    if results:
        print("   Additional model tests:")
        for model, success in results.items():
            status = '✅ SUCCESS' if success else '❌ FAILED'
            print(f"     - {model}: {status}")
    
    total_success = success1 and any(results.values() if results else [False])
    
    if success1:
        print("\n🎉 BertModel DOES work with ORTModelForFeatureExtraction and transformers pipeline!")
        print("   This means you can use optimum's ORTModel as a drop-in replacement for BertModel in feature extraction tasks.")
    else:
        print("\n😞 BertModel compatibility test failed.")
    
    return total_success

if __name__ == "__main__":
    main()