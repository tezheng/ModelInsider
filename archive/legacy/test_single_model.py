#!/usr/bin/env python3
"""
Single Model Tester - Test one model at a time to avoid timeouts
"""

import sys

from test_universal_hierarchy import UniversalHierarchyTester


def test_bert():
    """Test BERT model"""
    print("Testing BERT model...")
    tester = UniversalHierarchyTester()
    
    def load_bert():
        from transformers import AutoModel
        return AutoModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
    
    result = tester.test_model(
        'google/bert_uncased_L-2_H-128_A-2',
        load_bert,
        'transformer'
    )
    
    print(f"BERT Result: {result['status']}")
    if result['status'] == 'SUCCESS':
        print(f"  Tag coverage: {result['metrics'].get('tag_coverage', 0):.1%}")
        print(f"  Modules: {result['metrics'].get('total_modules', 0)}")
        print(f"  Operations: {result['metrics'].get('operations_found', 0)}")
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")
    
    return result


def test_resnet():
    """Test ResNet model"""
    print("\nTesting ResNet model...")
    tester = UniversalHierarchyTester()
    
    def load_resnet():
        import torchvision.models as models
        return models.resnet18(pretrained=False)
    
    result = tester.test_model(
        'resnet18',
        load_resnet,
        'vision'
    )
    
    print(f"ResNet Result: {result['status']}")
    if result['status'] == 'SUCCESS':
        print(f"  Tag coverage: {result['metrics'].get('tag_coverage', 0):.1%}")
        print(f"  Modules: {result['metrics'].get('total_modules', 0)}")
        print(f"  Operations: {result['metrics'].get('operations_found', 0)}")
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")
    
    return result


def test_vit():
    """Test ViT model"""
    print("\nTesting ViT model...")
    tester = UniversalHierarchyTester()
    
    def load_vit():
        from transformers import AutoModel
        return AutoModel.from_pretrained('google/vit-base-patch16-224')
    
    result = tester.test_model(
        'google/vit-base-patch16-224',
        load_vit,
        'vision_transformer'
    )
    
    print(f"ViT Result: {result['status']}")
    if result['status'] == 'SUCCESS':
        print(f"  Tag coverage: {result['metrics'].get('tag_coverage', 0):.1%}")
        print(f"  Modules: {result['metrics'].get('total_modules', 0)}")
        print(f"  Operations: {result['metrics'].get('operations_found', 0)}")
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")
    
    return result


def main():
    """Test models individually"""
    if len(sys.argv) > 1:
        model_name = sys.argv[1].lower()
        if model_name == 'bert':
            test_bert()
        elif model_name == 'resnet':
            test_resnet()
        elif model_name == 'vit':
            test_vit()
        else:
            print(f"Unknown model: {model_name}")
            print("Available models: bert, resnet, vit")
    else:
        # Test all models sequentially
        results = []
        
        print("Testing all models sequentially...")
        
        # Test BERT
        results.append(test_bert())
        
        # Test ResNet
        results.append(test_resnet())
        
        # Test ViT
        results.append(test_vit())
        
        # Summary
        print(f"\n{'='*50}")
        print("FINAL SUMMARY")
        print(f"{'='*50}")
        
        successful = sum(1 for r in results if r['status'] == 'SUCCESS')
        print(f"Successful tests: {successful}/3")
        
        for result in results:
            status_emoji = "✅" if result['status'] == 'SUCCESS' else "❌"
            print(f"{status_emoji} {result['model_name']}: {result['status']}")


if __name__ == "__main__":
    main()