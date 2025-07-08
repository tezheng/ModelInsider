#!/usr/bin/env python3
"""
Universal Hierarchy Testing Script
Test DAG extraction and ONNX export for multiple model architectures
"""

import torch
import torch.nn as nn
import onnx
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from transformers import AutoModel
import torchvision.models as models
import traceback

from dag_extractor import DAGExtractor
from input_generator import UniversalInputGenerator


class UniversalHierarchyTester:
    """Test hierarchy extraction across different model architectures"""
    
    def __init__(self, output_dir: str = "temp"):
        self.output_dir = Path(output_dir)
        self.test_outputs_dir = self.output_dir / "test_outputs"
        self.onnx_models_dir = self.output_dir / "onnx_models"
        self.test_data_dir = self.output_dir / "test_data"
        
        # Create directories
        for dir_path in [self.test_outputs_dir, self.onnx_models_dir, self.test_data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.input_generator = UniversalInputGenerator()
        self.test_results = {}
    
    def test_model(self, model_name: str, model_loader, model_type: str) -> Dict[str, Any]:
        """Test a single model with the universal hierarchy extractor"""
        print(f"\n{'='*60}")
        print(f"TESTING {model_name.upper()} ({model_type})")
        print(f"{'='*60}")
        
        try:
            # Step 1: Load model
            print(f"Loading model: {model_name}")
            model = model_loader()
            model.eval()
            
            # Step 2: Generate inputs
            print("Generating inputs...")
            inputs = self.input_generator.generate_inputs(model, model_name)
            print(f"Inputs generated: {list(inputs.keys())}")
            
            # Step 3: Test forward pass
            print("Testing forward pass...")
            with torch.no_grad():
                if len(inputs) == 1 and list(inputs.keys())[0] not in ['input_ids', 'pixel_values']:
                    output = model(list(inputs.values())[0])
                else:
                    output = model(**inputs)
            print("Forward pass successful ✓")
            
            # Step 4: Initialize DAG extractor
            print("Initializing DAG extractor...")
            extractor = DAGExtractor()
            
            # Step 5: Analyze model structure
            print("Analyzing model structure...")
            hierarchy = extractor.analyze_model_structure(model)
            print(f"Found {len(hierarchy)} modules ✓")
            
            # Step 6: Trace execution
            print("Tracing execution...")
            trace = extractor.trace_execution_with_hooks(model, inputs)
            print(f"Traced {len(trace)} executions ✓")
            
            # Step 7: Create parameter mapping
            print("Creating parameter mapping...")
            params = extractor.create_parameter_mapping(model)
            print(f"Mapped {len(params)} parameters ✓")
            
            # Step 8: Export to ONNX and analyze
            print("Exporting to ONNX and analyzing...")
            onnx_output_path = self.onnx_models_dir / f"{model_name.replace('/', '_')}.onnx"
            onnx_model = extractor.export_and_analyze_onnx(model, inputs, str(onnx_output_path))
            print(f"ONNX export successful ✓")
            
            # Step 9: Generate DAGs for all modules
            print("Generating module DAGs...")
            all_dags = extractor.generate_all_module_dags()
            print(f"Generated DAGs for {len(all_dags)} modules ✓")
            
            # Step 10: Save test data (static JSON files)
            print("Saving test data...")
            model_safe_name = model_name.replace('/', '_')
            
            # Save operation metadata
            metadata_file = self.test_outputs_dir / f"{model_safe_name}_operation_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(extractor.operation_metadata, f, indent=2)
            
            # Save module DAGs
            dags_file = self.test_outputs_dir / f"{model_safe_name}_module_dags.json"
            with open(dags_file, "w") as f:
                json.dump(all_dags, f, indent=2)
            
            # Save model hierarchy
            hierarchy_file = self.test_outputs_dir / f"{model_safe_name}_hierarchy.json"
            with open(hierarchy_file, "w") as f:
                json.dump(hierarchy, f, indent=2)
            
            # Save inputs used
            inputs_file = self.test_data_dir / f"{model_safe_name}_inputs.json"
            # Convert tensors to lists for JSON serialization
            inputs_serializable = {}
            for key, tensor in inputs.items():
                inputs_serializable[key] = {
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'data': tensor.tolist()
                }
            with open(inputs_file, "w") as f:
                json.dump(inputs_serializable, f, indent=2)
            
            print(f"Test data saved ✓")
            
            # Step 11: Verify ONNX model has tags
            print("Verifying ONNX model tags...")
            tag_verification = self.verify_onnx_tags(onnx_output_path, extractor.operation_metadata)
            print(f"Tag verification: {tag_verification['summary']}")
            
            # Compile results
            results = {
                'model_name': model_name,
                'model_type': model_type,
                'status': 'SUCCESS',
                'metrics': {
                    'total_modules': len(hierarchy),
                    'execution_traces': len(trace),
                    'parameters_mapped': len(params),
                    'operations_found': len(extractor.operation_metadata),
                    'modules_with_dags': len(all_dags),
                    'nodes_tagged': tag_verification['nodes_tagged'],
                    'total_nodes': tag_verification['total_nodes'],
                    'tag_coverage': tag_verification['tag_coverage']
                },
                'files': {
                    'onnx_model': str(onnx_output_path),
                    'onnx_with_tags': str(onnx_output_path).replace('.onnx', '_with_tags.onnx'),
                    'operation_metadata': str(metadata_file),
                    'module_dags': str(dags_file),
                    'hierarchy': str(hierarchy_file),
                    'inputs': str(inputs_file)
                },
                'tag_verification': tag_verification
            }
            
            print(f"\n✅ {model_name} test COMPLETED successfully!")
            return results
            
        except Exception as e:
            error_msg = f"Error testing {model_name}: {str(e)}"
            print(f"\n❌ {error_msg}")
            traceback.print_exc()
            
            return {
                'model_name': model_name,
                'model_type': model_type,
                'status': 'FAILED',
                'error': error_msg,
                'metrics': {},
                'files': {},
                'tag_verification': {}
            }
    
    def verify_onnx_tags(self, onnx_path: Path, operation_metadata: Dict) -> Dict[str, Any]:
        """Verify that ONNX model contains hierarchy tags"""
        try:
            # Load both models (original and enhanced)
            enhanced_path = str(onnx_path).replace('.onnx', '_with_tags.onnx')
            
            if not os.path.exists(enhanced_path):
                return {
                    'status': 'FAILED',
                    'error': 'Enhanced ONNX model with tags not found',
                    'nodes_tagged': 0,
                    'total_nodes': 0,
                    'tag_coverage': 0.0,
                    'summary': 'No enhanced model found'
                }
            
            # Load enhanced model
            onnx_model = onnx.load(enhanced_path)
            
            nodes_with_tags = 0
            total_nodes = len(onnx_model.graph.node)
            tagged_nodes = []
            untagged_nodes = []
            
            for node in onnx_model.graph.node:
                has_source_module = False
                has_hierarchy_tags = False
                
                for attr in node.attribute:
                    if attr.name == "source_module":
                        has_source_module = True
                    elif attr.name == "hierarchy_tags":
                        has_hierarchy_tags = True
                
                if has_source_module or has_hierarchy_tags:
                    nodes_with_tags += 1
                    tagged_nodes.append({
                        'name': node.name,
                        'op_type': node.op_type,
                        'has_source_module': has_source_module,
                        'has_hierarchy_tags': has_hierarchy_tags
                    })
                else:
                    untagged_nodes.append({
                        'name': node.name,
                        'op_type': node.op_type
                    })
            
            tag_coverage = nodes_with_tags / total_nodes if total_nodes > 0 else 0.0
            
            # Check metadata
            hierarchy_metadata = None
            operation_metadata_found = None
            
            for prop in onnx_model.metadata_props:
                if prop.key == "module_hierarchy":
                    hierarchy_metadata = json.loads(prop.value)
                elif prop.key == "operation_metadata":
                    operation_metadata_found = json.loads(prop.value)
            
            return {
                'status': 'SUCCESS',
                'nodes_tagged': nodes_with_tags,
                'total_nodes': total_nodes,
                'tag_coverage': tag_coverage,
                'tagged_nodes': tagged_nodes[:5],  # First 5 for inspection
                'untagged_nodes': untagged_nodes[:5],  # First 5 for inspection
                'has_hierarchy_metadata': hierarchy_metadata is not None,
                'has_operation_metadata': operation_metadata_found is not None,
                'summary': f"{nodes_with_tags}/{total_nodes} nodes tagged ({tag_coverage:.1%})"
            }
            
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'nodes_tagged': 0,
                'total_nodes': 0,
                'tag_coverage': 0.0,
                'summary': f'Verification failed: {str(e)}'
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run tests for all supported model architectures"""
        print("Starting Universal Hierarchy Testing...")
        print(f"Output directory: {self.output_dir}")
        
        # Define test models
        test_models = {
            'bert': {
                'name': 'google/bert_uncased_L-2_H-128_A-2',
                'type': 'transformer',
                'loader': lambda: AutoModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
            },
            'resnet': {
                'name': 'resnet18',
                'type': 'vision',
                'loader': lambda: models.resnet18(pretrained=False)
            },
            'vit': {
                'name': 'google/vit-base-patch16-224',
                'type': 'vision_transformer',
                'loader': lambda: AutoModel.from_pretrained('google/vit-base-patch16-224')
            }
        }
        
        all_results = {}
        successful_tests = 0
        
        # Test each model
        for model_key, model_info in test_models.items():
            try:
                result = self.test_model(
                    model_info['name'],
                    model_info['loader'],
                    model_info['type']
                )
                all_results[model_key] = result
                
                if result['status'] == 'SUCCESS':
                    successful_tests += 1
                    
            except Exception as e:
                print(f"Failed to test {model_key}: {e}")
                all_results[model_key] = {
                    'model_name': model_info['name'],
                    'model_type': model_info['type'],
                    'status': 'FAILED',
                    'error': str(e)
                }
        
        # Create summary
        summary = {
            'total_models_tested': len(test_models),
            'successful_tests': successful_tests,
            'failed_tests': len(test_models) - successful_tests,
            'test_results': all_results,
            'output_directory': str(self.output_dir)
        }
        
        # Save comprehensive results
        results_file = self.test_outputs_dir / "comprehensive_test_results.json"
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("TESTING SUMMARY")
        print(f"{'='*60}")
        print(f"Total models tested: {summary['total_models_tested']}")
        print(f"Successful tests: {summary['successful_tests']}")
        print(f"Failed tests: {summary['failed_tests']}")
        print(f"Results saved to: {results_file}")
        
        # Show individual results
        for model_key, result in all_results.items():
            status_emoji = "✅" if result['status'] == 'SUCCESS' else "❌"
            print(f"\n{status_emoji} {model_key.upper()}: {result['status']}")
            if result['status'] == 'SUCCESS' and 'metrics' in result:
                metrics = result['metrics']
                print(f"   Modules: {metrics.get('total_modules', 0)}")
                print(f"   Operations: {metrics.get('operations_found', 0)}")
                print(f"   Tag coverage: {metrics.get('tag_coverage', 0):.1%}")
        
        return summary
    
    def compare_static_data_with_onnx(self, model_name: str) -> Dict[str, Any]:
        """Compare static JSON data with ONNX model tags"""
        model_safe_name = model_name.replace('/', '_')
        
        # Load static data
        metadata_file = self.test_outputs_dir / f"{model_safe_name}_operation_metadata.json"
        dags_file = self.test_outputs_dir / f"{model_safe_name}_module_dags.json"
        
        if not metadata_file.exists() or not dags_file.exists():
            return {'status': 'FAILED', 'error': 'Static data files not found'}
        
        with open(metadata_file) as f:
            static_metadata = json.load(f)
        
        with open(dags_file) as f:
            static_dags = json.load(f)
        
        # Load ONNX model with tags
        onnx_path = self.onnx_models_dir / f"{model_safe_name}_with_tags.onnx"
        if not onnx_path.exists():
            return {'status': 'FAILED', 'error': 'ONNX model with tags not found'}
        
        onnx_model = onnx.load(onnx_path)
        
        # Extract tags from ONNX model
        onnx_tags = {}
        for node in onnx_model.graph.node:
            node_tags = []
            for attr in node.attribute:
                if attr.name == "source_module":
                    node_tags.append(attr.s.decode('utf-8'))
                elif attr.name == "hierarchy_tags":
                    node_tags.extend([tag.decode('utf-8') for tag in attr.strings])
            
            if node_tags:
                onnx_tags[node.name] = node_tags
        
        # Compare
        comparison = {
            'static_operations': len(static_metadata),
            'onnx_tagged_nodes': len(onnx_tags),
            'matching_operations': 0,
            'tag_consistency': True,
            'differences': []
        }
        
        for op_name, op_data in static_metadata.items():
            static_tags = op_data.get('tags', [])
            onnx_op_tags = onnx_tags.get(op_name, [])
            
            if set(static_tags) == set(onnx_op_tags):
                comparison['matching_operations'] += 1
            else:
                comparison['tag_consistency'] = False
                comparison['differences'].append({
                    'operation': op_name,
                    'static_tags': static_tags,
                    'onnx_tags': onnx_op_tags
                })
        
        comparison['consistency_rate'] = comparison['matching_operations'] / len(static_metadata) if static_metadata else 0
        
        return {'status': 'SUCCESS', 'comparison': comparison}


def main():
    """Main function to run comprehensive testing"""
    print("Universal Hierarchy-Preserving ONNX Export Testing")
    print("=" * 60)
    
    # Initialize tester
    tester = UniversalHierarchyTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # If any tests were successful, run comparison
    successful_models = [k for k, v in results['test_results'].items() if v['status'] == 'SUCCESS']
    
    if successful_models:
        print(f"\n{'='*60}")
        print("VERIFYING STATIC DATA VS ONNX CONSISTENCY")
        print(f"{'='*60}")
        
        for model_key in successful_models:
            model_name = results['test_results'][model_key]['model_name']
            print(f"\nVerifying {model_key.upper()}...")
            comparison = tester.compare_static_data_with_onnx(model_name)
            
            if comparison['status'] == 'SUCCESS':
                comp_data = comparison['comparison']
                print(f"✅ Consistency: {comp_data['consistency_rate']:.1%}")
                print(f"   Static operations: {comp_data['static_operations']}")
                print(f"   ONNX tagged nodes: {comp_data['onnx_tagged_nodes']}")
                print(f"   Matching operations: {comp_data['matching_operations']}")
                
                if not comp_data['tag_consistency']:
                    print(f"⚠️  Found {len(comp_data['differences'])} tag differences")
            else:
                print(f"❌ Verification failed: {comparison['error']}")
    
    return results


if __name__ == "__main__":
    main()