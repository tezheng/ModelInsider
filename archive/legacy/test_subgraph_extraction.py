#!/usr/bin/env python3
"""
Test Subgraph Extraction
Validate that extracted subgraphs produce equivalent results
"""

import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
import torch.nn as nn
from enhanced_dag_extractor import EnhancedDAGExtractor
from input_generator import UniversalInputGenerator
from onnx_subgraph_extractor import ONNXSubgraphExtractor


class SubgraphValidator:
    """Validate that extracted subgraphs work correctly"""
    
    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = Path(temp_dir)
        self.extraction_dir = self.temp_dir / "subgraph_extraction"
        self.extraction_dir.mkdir(parents=True, exist_ok=True)
    
    def test_full_workflow(self, model_name: str = "resnet18"):
        """Test the complete workflow: model -> enhanced ONNX -> subgraph extraction"""
        print(f"\n{'='*80}")
        print(f"TESTING COMPLETE SUBGRAPH EXTRACTION WORKFLOW: {model_name.upper()}")
        print(f"{'='*80}")
        
        try:
            # Step 1: Create enhanced ONNX model with 100% tagging
            print("\n🔧 STEP 1: Creating enhanced ONNX model...")
            enhanced_onnx_path, original_model, inputs = self._create_enhanced_model(model_name)
            
            # Step 2: Extract subgraphs for each module
            print("\n📦 STEP 2: Extracting subgraphs...")
            extracted_models = self._extract_all_subgraphs(enhanced_onnx_path)
            
            # Step 3: Validate extracted subgraphs
            print("\n✅ STEP 3: Validating extracted subgraphs...")
            validation_results = self._validate_extractions(extracted_models, original_model, inputs)
            
            # Step 4: Summary
            self._print_summary(model_name, validation_results)
            
            return validation_results
            
        except Exception as e:
            print(f"❌ Workflow failed: {e}")
            traceback.print_exc()
            return None
    
    def _create_enhanced_model(self, model_name: str) -> tuple:
        """Create enhanced ONNX model with 100% node tagging"""
        
        # Load model
        if model_name == "resnet18":
            import torchvision.models as models
            model = models.resnet18(pretrained=False)
        else:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_name)
        
        # Generate inputs
        generator = UniversalInputGenerator()
        inputs = generator.generate_inputs(model, model_name)
        
        # Create enhanced extractor
        extractor = EnhancedDAGExtractor()
        
        # Run full analysis
        hierarchy = extractor.analyze_model_structure(model)
        trace = extractor.trace_execution_with_hooks(model, inputs)
        params = extractor.create_parameter_mapping(model)
        
        # Export to ONNX
        model_safe_name = model_name.replace('/', '_')
        onnx_path = self.extraction_dir / f"{model_safe_name}_enhanced.onnx"
        onnx_model = extractor.export_and_analyze_onnx(model, inputs, str(onnx_path))
        
        # Get the path to the model with tags
        enhanced_path = str(onnx_path).replace('.onnx', '_with_tags.onnx')
        
        print(f"✅ Enhanced ONNX model created: {enhanced_path}")
        return enhanced_path, model, inputs
    
    def _extract_all_subgraphs(self, onnx_model_path: str) -> dict[str, str]:
        """Extract subgraphs for all modules"""
        
        extractor = ONNXSubgraphExtractor(onnx_model_path)
        available_modules = extractor.list_available_modules()
        
        print(f"Found {len(available_modules)} modules to extract")
        
        # Extract all modules
        extraction_dir = self.extraction_dir / "extracted"
        results = extractor.extract_multiple_modules(
            list(available_modules.keys()), 
            str(extraction_dir)
        )
        
        # Return paths to successfully extracted models
        extracted_models = {}
        for module_tag, result in results.items():
            if result['status'] == 'SUCCESS':
                extracted_models[module_tag] = result['output_path']
        
        print(f"✅ Successfully extracted {len(extracted_models)} subgraphs")
        return extracted_models
    
    def _validate_extractions(self, extracted_models: dict[str, str], 
                            original_model: nn.Module, inputs: dict) -> dict[str, Any]:
        """Validate that extracted subgraphs are valid ONNX models"""
        
        validation_results = {
            'total_extracted': len(extracted_models),
            'valid_models': 0,
            'loadable_models': 0,
            'runnable_models': 0,
            'results': {}
        }
        
        for module_tag, onnx_path in extracted_models.items():
            print(f"\n🔍 Validating: {module_tag}")
            
            result = {
                'module_tag': module_tag,
                'onnx_path': onnx_path,
                'is_valid': False,
                'is_loadable': False,
                'is_runnable': False,
                'error': None
            }
            
            try:
                # Test 1: ONNX model validity
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                result['is_valid'] = True
                validation_results['valid_models'] += 1
                print(f"   ✅ Valid ONNX model")
                
                # Test 2: ONNX Runtime loadability
                session = onnxruntime.InferenceSession(onnx_path)
                result['is_loadable'] = True
                validation_results['loadable_models'] += 1
                print(f"   ✅ Loadable in ONNX Runtime")
                
                # Test 3: Try to run with dummy inputs
                try:
                    # Generate appropriate inputs for the subgraph
                    input_info = session.get_inputs()
                    dummy_inputs = {}
                    
                    for inp in input_info:
                        shape = inp.shape
                        # Replace dynamic dimensions with concrete values
                        concrete_shape = []
                        for dim in shape:
                            if isinstance(dim, str) or dim < 0:
                                concrete_shape.append(1)  # Use 1 for dynamic dims
                            else:
                                concrete_shape.append(dim)
                        
                        # Generate dummy data
                        if inp.type == 'tensor(float)':
                            dummy_inputs[inp.name] = np.random.randn(*concrete_shape).astype(np.float32)
                        elif inp.type == 'tensor(int64)':
                            dummy_inputs[inp.name] = np.random.randint(0, 100, concrete_shape).astype(np.int64)
                        else:
                            dummy_inputs[inp.name] = np.random.randn(*concrete_shape).astype(np.float32)
                    
                    # Run inference
                    outputs = session.run(None, dummy_inputs)
                    result['is_runnable'] = True
                    validation_results['runnable_models'] += 1
                    result['output_shapes'] = [output.shape for output in outputs]
                    print(f"   ✅ Runnable (outputs: {len(outputs)})")
                    
                except Exception as e:
                    result['run_error'] = str(e)
                    print(f"   ⚠️  Not runnable: {e}")
                
            except Exception as e:
                result['error'] = str(e)
                print(f"   ❌ Validation failed: {e}")
            
            validation_results['results'][module_tag] = result
        
        return validation_results
    
    def _print_summary(self, model_name: str, validation_results: dict[str, Any]):
        """Print validation summary"""
        
        print(f"\n{'='*80}")
        print(f"SUBGRAPH EXTRACTION SUMMARY: {model_name.upper()}")
        print(f"{'='*80}")
        
        total = validation_results['total_extracted']
        valid = validation_results['valid_models']
        loadable = validation_results['loadable_models']
        runnable = validation_results['runnable_models']
        
        print(f"📊 OVERALL RESULTS:")
        print(f"   Total extracted: {total}")
        print(f"   Valid ONNX models: {valid}/{total} ({valid/total:.1%})")
        print(f"   Loadable models: {loadable}/{total} ({loadable/total:.1%})")
        print(f"   Runnable models: {runnable}/{total} ({runnable/total:.1%})")
        
        print(f"\n📋 DETAILED RESULTS:")
        for module_tag, result in validation_results['results'].items():
            status_icons = []
            if result['is_valid']:
                status_icons.append("✅")
            if result['is_loadable']:
                status_icons.append("🔄")
            if result['is_runnable']:
                status_icons.append("▶️")
            
            status = " ".join(status_icons) if status_icons else "❌"
            print(f"   {status} {module_tag}")
            
            if result.get('error'):
                print(f"       Error: {result['error']}")
        
        # Save detailed results
        results_path = self.extraction_dir / f"{model_name.replace('/', '_')}_validation_results.json"
        with open(results_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {results_path}")
        
        # Overall assessment
        success_rate = runnable / total if total > 0 else 0
        if success_rate >= 0.8:
            print(f"\n🎉 EXCELLENT: {success_rate:.1%} of extracted subgraphs are fully functional!")
        elif success_rate >= 0.6:
            print(f"\n👍 GOOD: {success_rate:.1%} of extracted subgraphs are functional")
        else:
            print(f"\n⚠️  NEEDS IMPROVEMENT: Only {success_rate:.1%} of extracted subgraphs are functional")


def main():
    """Test subgraph extraction on multiple models"""
    
    validator = SubgraphValidator()
    
    models_to_test = [
        "resnet18",
        # "google/bert_uncased_L-2_H-128_A-2"  # Add BERT if needed
    ]
    
    all_results = {}
    
    for model_name in models_to_test:
        try:
            results = validator.test_full_workflow(model_name)
            all_results[model_name] = results
        except Exception as e:
            print(f"❌ Failed to test {model_name}: {e}")
            all_results[model_name] = None
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY ACROSS ALL MODELS")
    print(f"{'='*80}")
    
    for model_name, results in all_results.items():
        if results:
            success_rate = results['runnable_models'] / results['total_extracted']
            print(f"{model_name:30} {success_rate:.1%} success rate ({results['runnable_models']}/{results['total_extracted']})")
        else:
            print(f"{model_name:30} FAILED")


if __name__ == "__main__":
    main()