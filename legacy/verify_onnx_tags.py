#!/usr/bin/env python3
"""
ONNX Tag Verification Script
Verify that ONNX models contain the expected hierarchy tags and compare with static data
"""

import onnx
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse


class ONNXTagVerifier:
    """Verify ONNX model tags against static test data"""
    
    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = Path(temp_dir)
        self.test_outputs_dir = self.temp_dir / "test_outputs"
        self.onnx_models_dir = self.temp_dir / "onnx_models"
    
    def verify_model_tags(self, model_name: str, verbose: bool = True) -> Dict[str, Any]:
        """Verify tags for a specific model"""
        if verbose:
            print(f"\n{'='*50}")
            print(f"VERIFYING {model_name.upper()}")
            print(f"{'='*50}")
        
        model_safe_name = model_name.replace('/', '_')
        
        # Check if required files exist
        onnx_path = self.onnx_models_dir / f"{model_safe_name}_with_tags.onnx"
        metadata_path = self.test_outputs_dir / f"{model_safe_name}_operation_metadata.json"
        
        if not onnx_path.exists():
            return {
                'status': 'FAILED',
                'error': f'ONNX model not found: {onnx_path}',
                'model_name': model_name
            }
        
        if not metadata_path.exists():
            return {
                'status': 'FAILED', 
                'error': f'Static metadata not found: {metadata_path}',
                'model_name': model_name
            }
        
        try:
            # Load static metadata
            with open(metadata_path) as f:
                static_metadata = json.load(f)
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Extract verification results
            verification = self._extract_and_compare_tags(onnx_model, static_metadata, verbose)
            verification['model_name'] = model_name
            verification['files'] = {
                'onnx_model': str(onnx_path),
                'static_metadata': str(metadata_path)
            }
            
            return verification
            
        except Exception as e:
            return {
                'status': 'FAILED',
                'error': f'Verification failed: {str(e)}',
                'model_name': model_name
            }
    
    def _extract_and_compare_tags(self, onnx_model, static_metadata: Dict, verbose: bool) -> Dict[str, Any]:
        """Extract tags from ONNX and compare with static data"""
        
        # Extract tags from ONNX nodes
        onnx_node_tags = {}
        nodes_with_tags = 0
        total_nodes = len(onnx_model.graph.node)
        
        for node in onnx_model.graph.node:
            node_tags = []
            for attr in node.attribute:
                if attr.name == "source_module":
                    node_tags.append(attr.s.decode('utf-8'))
                elif attr.name == "hierarchy_tags":
                    node_tags.extend([tag.decode('utf-8') for tag in attr.strings])
            
            if node_tags:
                onnx_node_tags[node.name] = node_tags
                nodes_with_tags += 1
        
        # Extract metadata from ONNX model properties
        onnx_metadata = {}
        hierarchy_metadata = None
        operation_metadata = None
        
        for prop in onnx_model.metadata_props:
            if prop.key == "module_hierarchy":
                hierarchy_metadata = json.loads(prop.value)
            elif prop.key == "operation_metadata":
                operation_metadata = json.loads(prop.value)
        
        # Compare with static data
        comparison_results = self._compare_tags(onnx_node_tags, static_metadata, verbose)
        
        # Calculate coverage metrics
        tag_coverage = nodes_with_tags / total_nodes if total_nodes > 0 else 0.0
        
        results = {
            'status': 'SUCCESS',
            'onnx_analysis': {
                'total_nodes': total_nodes,
                'nodes_with_tags': nodes_with_tags,
                'tag_coverage': tag_coverage,
                'has_hierarchy_metadata': hierarchy_metadata is not None,
                'has_operation_metadata': operation_metadata is not None
            },
            'static_analysis': {
                'total_operations': len(static_metadata),
                'operations_with_tags': len([op for op in static_metadata.values() if op.get('tags')])
            },
            'comparison': comparison_results,
            'sample_tagged_nodes': list(onnx_node_tags.items())[:5] if onnx_node_tags else []
        }
        
        if verbose:
            self._print_verification_results(results)
        
        return results
    
    def _compare_tags(self, onnx_tags: Dict, static_metadata: Dict, verbose: bool) -> Dict[str, Any]:
        """Compare ONNX tags with static metadata"""
        
        matching_operations = 0
        total_comparable = 0
        differences = []
        
        # Find operations that exist in both
        for op_name, op_data in static_metadata.items():
            static_tags = set(op_data.get('tags', []))
            
            if op_name in onnx_tags:
                total_comparable += 1
                onnx_op_tags = set(onnx_tags[op_name])
                
                if static_tags == onnx_op_tags:
                    matching_operations += 1
                else:
                    differences.append({
                        'operation': op_name,
                        'static_tags': list(static_tags),
                        'onnx_tags': list(onnx_op_tags),
                        'missing_in_onnx': list(static_tags - onnx_op_tags),
                        'extra_in_onnx': list(onnx_op_tags - static_tags)
                    })
        
        # Find ONNX operations not in static data
        onnx_only_ops = []
        for onnx_op in onnx_tags:
            if onnx_op not in static_metadata:
                onnx_only_ops.append({
                    'operation': onnx_op,
                    'onnx_tags': onnx_tags[onnx_op]
                })
        
        # Find static operations not in ONNX
        static_only_ops = []
        for static_op in static_metadata:
            if static_op not in onnx_tags:
                static_only_ops.append({
                    'operation': static_op,
                    'static_tags': static_metadata[static_op].get('tags', [])
                })
        
        consistency_rate = matching_operations / total_comparable if total_comparable > 0 else 0.0
        
        return {
            'total_comparable': total_comparable,
            'matching_operations': matching_operations,
            'consistency_rate': consistency_rate,
            'tag_differences': differences[:10],  # Limit to first 10 for readability
            'onnx_only_operations': onnx_only_ops[:5],  # Limit to first 5
            'static_only_operations': static_only_ops[:5],  # Limit to first 5
            'total_differences': len(differences),
            'total_onnx_only': len(onnx_only_ops),
            'total_static_only': len(static_only_ops)
        }
    
    def _print_verification_results(self, results: Dict):
        """Print detailed verification results"""
        onnx_analysis = results['onnx_analysis']
        static_analysis = results['static_analysis']
        comparison = results['comparison']
        
        print(f"\n📊 ONNX MODEL ANALYSIS:")
        print(f"   Total nodes: {onnx_analysis['total_nodes']}")
        print(f"   Nodes with tags: {onnx_analysis['nodes_with_tags']}")
        print(f"   Tag coverage: {onnx_analysis['tag_coverage']:.1%}")
        print(f"   Has hierarchy metadata: {'✅' if onnx_analysis['has_hierarchy_metadata'] else '❌'}")
        print(f"   Has operation metadata: {'✅' if onnx_analysis['has_operation_metadata'] else '❌'}")
        
        print(f"\n📁 STATIC DATA ANALYSIS:")
        print(f"   Total operations: {static_analysis['total_operations']}")
        print(f"   Operations with tags: {static_analysis['operations_with_tags']}")
        
        print(f"\n🔍 COMPARISON RESULTS:")
        print(f"   Comparable operations: {comparison['total_comparable']}")
        print(f"   Matching operations: {comparison['matching_operations']}")
        print(f"   Consistency rate: {comparison['consistency_rate']:.1%}")
        print(f"   Tag differences: {comparison['total_differences']}")
        print(f"   ONNX-only operations: {comparison['total_onnx_only']}")
        print(f"   Static-only operations: {comparison['total_static_only']}")
        
        # Show sample tagged nodes
        if results['sample_tagged_nodes']:
            print(f"\n🏷️  SAMPLE TAGGED NODES:")
            for i, (node_name, tags) in enumerate(results['sample_tagged_nodes']):
                print(f"   {i+1}. {node_name}: {tags}")
        
        # Show some differences if any
        if comparison['tag_differences']:
            print(f"\n⚠️  SAMPLE TAG DIFFERENCES:")
            for i, diff in enumerate(comparison['tag_differences'][:3]):
                print(f"   {i+1}. {diff['operation']}:")
                print(f"      Static: {diff['static_tags']}")
                print(f"      ONNX:   {diff['onnx_tags']}")
        
        # Overall status
        if comparison['consistency_rate'] > 0.9:
            print(f"\n✅ VERIFICATION PASSED (consistency: {comparison['consistency_rate']:.1%})")
        elif comparison['consistency_rate'] > 0.7:
            print(f"\n⚠️  VERIFICATION WARNING (consistency: {comparison['consistency_rate']:.1%})")
        else:
            print(f"\n❌ VERIFICATION FAILED (consistency: {comparison['consistency_rate']:.1%})")
    
    def verify_all_models(self, verbose: bool = True) -> Dict[str, Any]:
        """Verify all available models in the temp directory"""
        if verbose:
            print("VERIFYING ALL MODELS")
            print("=" * 60)
        
        # Find all available models by looking for ONNX files
        onnx_files = list(self.onnx_models_dir.glob("*_with_tags.onnx"))
        
        if not onnx_files:
            return {
                'status': 'NO_MODELS',
                'message': 'No ONNX models with tags found',
                'models_verified': {}
            }
        
        results = {}
        successful_verifications = 0
        
        for onnx_file in onnx_files:
            # Extract model name from filename
            model_safe_name = onnx_file.stem.replace('_with_tags', '')
            model_name = model_safe_name.replace('_', '/')  # Convert back from safe filename
            
            verification_result = self.verify_model_tags(model_name, verbose)
            results[model_name] = verification_result
            
            if verification_result['status'] == 'SUCCESS':
                successful_verifications += 1
        
        # Summary
        summary = {
            'status': 'SUCCESS',
            'total_models': len(onnx_files),
            'successful_verifications': successful_verifications,
            'failed_verifications': len(onnx_files) - successful_verifications,
            'models_verified': results
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print("VERIFICATION SUMMARY")
            print(f"{'='*60}")
            print(f"Total models: {summary['total_models']}")
            print(f"Successful: {summary['successful_verifications']}")
            print(f"Failed: {summary['failed_verifications']}")
            
            for model_name, result in results.items():
                status_emoji = "✅" if result['status'] == 'SUCCESS' else "❌"
                print(f"\n{status_emoji} {model_name}")
                if result['status'] == 'SUCCESS' and 'comparison' in result:
                    consistency = result['comparison']['consistency_rate']
                    print(f"   Consistency: {consistency:.1%}")
        
        return summary


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Verify ONNX model hierarchy tags")
    parser.add_argument('--model', type=str, help='Specific model to verify (e.g., google/bert_uncased_L-2_H-128_A-2)')
    parser.add_argument('--all', action='store_true', help='Verify all available models')
    parser.add_argument('--temp-dir', type=str, default='temp', help='Temporary directory path')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    verifier = ONNXTagVerifier(args.temp_dir)
    verbose = not args.quiet
    
    if args.model:
        result = verifier.verify_model_tags(args.model, verbose)
        if result['status'] != 'SUCCESS':
            print(f"❌ Verification failed: {result['error']}")
            return 1
    elif args.all:
        result = verifier.verify_all_models(verbose)
        if result['failed_verifications'] > 0:
            print(f"⚠️  {result['failed_verifications']} verifications failed")
            return 1
    else:
        # Default: verify all models
        result = verifier.verify_all_models(verbose)
        if result['status'] == 'NO_MODELS':
            print("No models found to verify. Run test_universal_hierarchy.py first.")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())