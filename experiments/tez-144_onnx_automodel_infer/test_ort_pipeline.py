#!/usr/bin/env python3
"""
Test script to verify if ORTModel classes work with transformers pipeline for feature extraction.
Specifically testing whether ORTModelForFeatureExtraction can work with BertModel (not BertForSequenceClassification).
"""

import sys
import traceback
from pathlib import Path

# Add project root to path to use optimum from virtual environment
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_ortmodel_with_pipeline():
    """Test ORTModel classes with transformers pipeline."""
    
    try:
        from transformers import pipeline, AutoTokenizer
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        
        print("✅ Successfully imported required modules")
        
        # Test with a small BERT model
        model_name = "prajjwal1/bert-tiny"
        print(f"\n🔍 Testing with model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ Loaded tokenizer: {type(tokenizer).__name__}")
        
        # Try to load the model for feature extraction
        print("\n📦 Loading ORTModelForFeatureExtraction...")
        try:
            model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
            print(f"✅ Successfully loaded model: {type(model).__name__}")
            print(f"   Auto model class: {model.auto_model_class}")
            print(f"   Model type: {model.model_type}")
        except Exception as e:
            print(f"❌ Failed to load ORTModelForFeatureExtraction: {e}")
            return False
        
        # Test the forward method directly
        print("\n🧪 Testing direct forward method...")
        try:
            inputs = tokenizer("Hello world", return_tensors="pt", padding=True, truncation=True)
            print(f"   Input keys: {list(inputs.keys())}")
            
            outputs = model(**inputs)
            print(f"✅ Direct forward call successful")
            print(f"   Output type: {type(outputs)}")
            print(f"   Output keys: {list(outputs.keys()) if hasattr(outputs, 'keys') else 'no keys'}")
            if hasattr(outputs, 'last_hidden_state'):
                print(f"   Last hidden state shape: {outputs.last_hidden_state.shape}")
        except Exception as e:
            print(f"❌ Direct forward call failed: {e}")
            traceback.print_exc()
            return False
        
        # Test with transformers pipeline
        print("\n🔗 Testing with transformers pipeline...")
        try:
            pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
            print(f"✅ Pipeline created successfully: {type(pipe).__name__}")
            
            # Test pipeline inference
            text = "This is a test sentence"
            result = pipe(text)
            print(f"✅ Pipeline inference successful")
            print(f"   Result type: {type(result)}")
            print(f"   Result shape: {len(result)} x {len(result[0])} x {len(result[0][0])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline creation/inference failed: {e}")
            traceback.print_exc()
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False

def test_other_ortmodels():
    """Test other ORTModel classes that might work for feature extraction."""
    
    from optimum.onnxruntime import ORTModelForCustomTasks
    
    print("\n🔍 Testing ORTModelForCustomTasks...")
    
    try:
        model_name = "prajjwal1/bert-tiny"
        
        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Try ORTModelForCustomTasks
        model = ORTModelForCustomTasks.from_pretrained(model_name, export=True)
        print(f"✅ Successfully loaded ORTModelForCustomTasks")
        
        # Test direct inference
        inputs = tokenizer("Hello world", return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        print(f"✅ Direct inference successful")
        print(f"   Output type: {type(outputs)}")
        print(f"   Available keys: {list(outputs.keys()) if hasattr(outputs, 'keys') else 'no keys'}")
        
        # Test with pipeline if possible
        try:
            from transformers import pipeline
            pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
            result = pipe("Test sentence")
            print(f"✅ ORTModelForCustomTasks works with pipeline")
            return True
        except Exception as e:
            print(f"⚠️ ORTModelForCustomTasks doesn't work with pipeline: {e}")
            return False
            
    except Exception as e:
        print(f"❌ ORTModelForCustomTasks failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Testing ORTModel classes with transformers pipeline")
    print("=" * 60)
    
    # Test main feature extraction model
    success1 = test_ortmodel_with_pipeline()
    
    print("\n" + "=" * 60)
    
    # Test custom tasks model  
    success2 = test_other_ortmodels()
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY:")
    print(f"   ORTModelForFeatureExtraction: {'✅ SUCCESS' if success1 else '❌ FAILED'}")
    print(f"   ORTModelForCustomTasks: {'✅ SUCCESS' if success2 else '❌ FAILED'}")
    
    if success1 or success2:
        print("\n🎉 At least one ORTModel class works with pipeline for feature extraction!")
    else:
        print("\n😞 No ORTModel classes work with pipeline for feature extraction.")
    
    return success1 or success2

if __name__ == "__main__":
    main()