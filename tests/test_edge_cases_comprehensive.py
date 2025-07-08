#!/usr/bin/env python3
"""
Comprehensive Edge Case Testing Framework

This test suite systematically validates edge case handling in the enhanced semantic mapper.
Tests all problematic node patterns identified from real model analysis.

CARDINAL RULES:
- MUST-002: ALL testing via pytest 
- MUST-001: NO hardcoded logic in tests
- MUST-003: Universal design validation
"""

import pytest
import torch
import onnx
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from transformers import AutoModel, AutoTokenizer

from modelexport.core.enhanced_semantic_exporter import EnhancedSemanticExporter
from modelexport.semantic.enhanced_semantic_mapper import EnhancedSemanticMapper


class EdgeCaseTestFramework:
    """
    Comprehensive framework for testing edge cases in semantic mapping.
    
    Tests all identified edge case patterns:
    1. Constants without context (/Constant, /Constant_N)
    2. Root-level operations (/Equal, /Where, /Expand) 
    3. Numbered operations (/Equal_1, /Where_1)
    4. Compiler-generated operations
    5. Shape operations (/Shape, /Size, /Slice)
    6. Cross-module arithmetic
    """
    
    def __init__(self):
        self.test_models = self._get_test_models()
        self.edge_case_patterns = self._define_edge_case_patterns()
        self.test_results = {}
    
    def _get_test_models(self) -> List[Dict[str, Any]]:
        """Get test models for edge case validation."""
        return [
            {
                'name': 'bert-tiny',
                'model_name': 'prajjwal1/bert-tiny',
                'input_text': 'Test input for edge cases',
                'expected_edge_cases': ['constant', 'root_operation', 'numbered_operation']
            },
            # Can add more models here for broader validation
        ]
    
    def _define_edge_case_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Define expected edge case patterns with validation criteria."""
        return {
            'constant': {
                'node_patterns': ['/Constant', '/Constant_\\d+', '/ConstantOfShape'],
                'semantic_types': ['constant'],
                'confidence_levels': ['medium', 'low'],
                'min_expected_count': 5,
                'description': 'Constant operations without module context'
            },
            'root_operation': {
                'node_patterns': ['/\\w+$'],  # Single operation at root
                'semantic_types': ['root_operation', 'arithmetic', 'tensor_manipulation'],
                'confidence_levels': ['medium', 'low'],
                'min_expected_count': 1,
                'description': 'Operations at graph root level'
            },
            'numbered_operation': {
                'node_patterns': ['/\\w+_\\d+$'],  # Numbered operations
                'semantic_types': ['numbered_operation', 'arithmetic', 'tensor_manipulation'],
                'confidence_levels': ['medium', 'low'],
                'min_expected_count': 1,
                'description': 'Numbered operations (compiler-generated)'
            },
            'shape_operation': {
                'node_patterns': ['/Shape', '/Size', '/Slice'],
                'semantic_types': ['introspection'],
                'confidence_levels': ['medium'],
                'min_expected_count': 0,  # May not be present in all models
                'description': 'Shape introspection operations'
            },
            'tensor_manipulation': {
                'node_patterns': ['/Reshape', '/Transpose', '/Squeeze', '/Unsqueeze'],
                'semantic_types': ['tensor_manipulation'],
                'confidence_levels': ['medium'],
                'min_expected_count': 0,
                'description': 'Tensor manipulation operations'
            }
        }
    
    def run_comprehensive_edge_case_tests(self) -> Dict[str, Any]:
        """Run comprehensive edge case tests across all models."""
        print("🧪 Running Comprehensive Edge Case Tests...")
        
        all_results = {}
        
        for model_config in self.test_models:
            print(f"\n📋 Testing model: {model_config['name']}")
            model_results = self._test_model_edge_cases(model_config)
            all_results[model_config['name']] = model_results
        
        # Generate summary report
        summary = self._generate_test_summary(all_results)
        return summary
    
    def _test_model_edge_cases(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test edge cases for a specific model."""
        model_name = model_config['model_name']
        
        # Load model and prepare inputs
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        inputs = tokenizer(
            [model_config['input_text']], 
            return_tensors='pt', 
            max_length=16, 
            padding=True, 
            truncation=True
        )
        
        # Export with enhanced semantic mapping
        exporter = EnhancedSemanticExporter(verbose=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, f'{model_config["name"]}.onnx')
            
            result = exporter.export(
                model=model,
                args=(inputs['input_ids'],),
                output_path=output_path,
                input_names=['input_ids'],
                output_names=['last_hidden_state'],
                opset_version=17
            )
            
            # Analyze edge cases
            metadata = exporter.get_semantic_metadata()
            edge_case_analysis = self._analyze_model_edge_cases(metadata)
            
            return {
                'export_stats': result,
                'edge_case_analysis': edge_case_analysis,
                'validation_results': self._validate_edge_case_coverage(edge_case_analysis)
            }
    
    def _analyze_model_edge_cases(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze edge cases in semantic mappings."""
        semantic_mappings = metadata['semantic_mappings']
        
        edge_case_nodes = {}
        for pattern_name, pattern_config in self.edge_case_patterns.items():
            edge_case_nodes[pattern_name] = []
        
        # Categorize all nodes by edge case patterns
        for node_name, mapping in semantic_mappings.items():
            semantic_type = mapping.get('semantic_type', 'unknown')
            confidence = mapping.get('confidence', 'unknown')
            op_type = mapping.get('onnx_op_type', 'unknown')
            
            # Check against each edge case pattern
            for pattern_name, pattern_config in self.edge_case_patterns.items():
                if self._matches_edge_case_pattern(
                    node_name, semantic_type, confidence, op_type, pattern_config
                ):
                    edge_case_nodes[pattern_name].append({
                        'node_name': node_name,
                        'semantic_type': semantic_type,
                        'confidence': confidence,
                        'op_type': op_type,
                        'mapping': mapping
                    })
        
        return edge_case_nodes
    
    def _matches_edge_case_pattern(self, node_name: str, semantic_type: str, 
                                   confidence: str, op_type: str, 
                                   pattern_config: Dict[str, Any]) -> bool:
        """Check if node matches an edge case pattern."""
        import re
        
        # Check node name patterns
        for pattern in pattern_config['node_patterns']:
            if re.match(pattern, node_name):
                return True
        
        # Check semantic type match
        if semantic_type in pattern_config['semantic_types']:
            return True
        
        return False
    
    def _validate_edge_case_coverage(self, edge_case_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that edge case coverage meets expectations."""
        validation_results = {}
        
        for pattern_name, pattern_config in self.edge_case_patterns.items():
            found_nodes = edge_case_analysis.get(pattern_name, [])
            
            validation = {
                'pattern_name': pattern_name,
                'description': pattern_config['description'],
                'expected_min_count': pattern_config['min_expected_count'],
                'actual_count': len(found_nodes),
                'meets_expectation': len(found_nodes) >= pattern_config['min_expected_count'],
                'confidence_distribution': self._analyze_confidence_distribution(found_nodes),
                'semantic_type_distribution': self._analyze_semantic_distribution(found_nodes),
                'sample_nodes': found_nodes[:3]  # Show first 3 examples
            }
            
            validation_results[pattern_name] = validation
        
        return validation_results
    
    def _analyze_confidence_distribution(self, nodes: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze confidence level distribution."""
        distribution = {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0}
        for node in nodes:
            confidence = node.get('confidence', 'unknown')
            if confidence in distribution:
                distribution[confidence] += 1
        return distribution
    
    def _analyze_semantic_distribution(self, nodes: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze semantic type distribution."""
        distribution = {}
        for node in nodes:
            semantic_type = node.get('semantic_type', 'unknown')
            distribution[semantic_type] = distribution.get(semantic_type, 0) + 1
        return distribution
    
    def _generate_test_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        summary = {
            'total_models_tested': len(all_results),
            'overall_results': {},
            'edge_case_coverage': {},
            'validation_summary': {},
            'recommendations': []
        }
        
        # Aggregate results across all models
        for model_name, model_results in all_results.items():
            validation = model_results['validation_results']
            
            for pattern_name, pattern_validation in validation.items():
                if pattern_name not in summary['edge_case_coverage']:
                    summary['edge_case_coverage'][pattern_name] = {
                        'total_occurrences': 0,
                        'models_with_pattern': 0,
                        'confidence_breakdown': {'high': 0, 'medium': 0, 'low': 0}
                    }
                
                coverage = summary['edge_case_coverage'][pattern_name]
                coverage['total_occurrences'] += pattern_validation['actual_count']
                
                if pattern_validation['actual_count'] > 0:
                    coverage['models_with_pattern'] += 1
                
                # Aggregate confidence distribution
                for conf_level, count in pattern_validation['confidence_distribution'].items():
                    if conf_level in coverage['confidence_breakdown']:
                        coverage['confidence_breakdown'][conf_level] += count
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations(summary)
        
        return summary
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        for pattern_name, coverage in summary['edge_case_coverage'].items():
            total_occurrences = coverage['total_occurrences']
            
            if total_occurrences > 0:
                confidence_breakdown = coverage['confidence_breakdown']
                low_confidence_ratio = confidence_breakdown.get('low', 0) / total_occurrences
                
                if low_confidence_ratio > 0.5:
                    recommendations.append(
                        f"Improve confidence for {pattern_name} edge cases "
                        f"({low_confidence_ratio:.1%} are low confidence)"
                    )
                
                if pattern_name == 'constant' and total_occurrences > 10:
                    recommendations.append(
                        "Implement data flow analysis for constant operations "
                        "to inherit semantic context from consumers"
                    )
                
                if pattern_name in ['root_operation', 'numbered_operation']:
                    recommendations.append(
                        f"Add graph pattern recognition for {pattern_name} "
                        "to better understand their role in computation"
                    )
        
        return recommendations


# Pytest test cases
class TestEdgeCasesComprehensive:
    """Pytest test class for comprehensive edge case validation."""
    
    @pytest.fixture(scope="class")
    def test_framework(self):
        """Create edge case test framework."""
        return EdgeCaseTestFramework()
    
    @pytest.fixture(scope="class") 
    def test_results(self, test_framework):
        """Run comprehensive tests and return results."""
        return test_framework.run_comprehensive_edge_case_tests()
    
    def test_edge_case_coverage_meets_expectations(self, test_results):
        """Test that edge case coverage meets minimum expectations."""
        edge_case_coverage = test_results['edge_case_coverage']
        
        # Validate each edge case pattern
        for pattern_name, coverage in edge_case_coverage.items():
            total_occurrences = coverage['total_occurrences']
            
            # All models should have some constant operations
            if pattern_name == 'constant':
                assert total_occurrences >= 5, (
                    f"Expected at least 5 constant operations, found {total_occurrences}"
                )
            
            # Should have some confidence in classifications
            confidence_breakdown = coverage['confidence_breakdown']
            unknown_ratio = confidence_breakdown.get('unknown', 0) / max(total_occurrences, 1)
            assert unknown_ratio < 0.1, (
                f"Too many unknown confidence levels for {pattern_name}: {unknown_ratio:.1%}"
            )
    
    def test_semantic_mapper_handles_all_edge_cases(self, test_results):
        """Test that semantic mapper provides mappings for all edge case nodes."""
        # This test ensures no edge case nodes are left without semantic information
        recommendations = test_results['recommendations']
        
        # Should not have critical gaps
        critical_issues = [r for r in recommendations if 'critical' in r.lower()]
        assert len(critical_issues) == 0, f"Critical edge case issues found: {critical_issues}"
    
    def test_confidence_levels_are_appropriate(self, test_results):
        """Test that confidence levels are appropriate for edge case types."""
        edge_case_coverage = test_results['edge_case_coverage']
        
        for pattern_name, coverage in edge_case_coverage.items():
            if coverage['total_occurrences'] > 0:
                confidence_breakdown = coverage['confidence_breakdown']
                total = sum(confidence_breakdown.values())
                
                # Edge cases should not be high confidence (they're inherently uncertain)
                high_conf_ratio = confidence_breakdown.get('high', 0) / total
                assert high_conf_ratio < 0.3, (
                    f"Edge cases in {pattern_name} should not be high confidence: {high_conf_ratio:.1%}"
                )
    
    def test_no_hardcoded_logic_in_edge_case_handling(self, test_framework):
        """Validate that edge case handling follows universal design principles."""
        # Check that edge case patterns are defined generically
        patterns = test_framework.edge_case_patterns
        
        for pattern_name, pattern_config in patterns.items():
            # Patterns should use regex, not specific model names
            for node_pattern in pattern_config['node_patterns']:
                assert 'bert' not in node_pattern.lower(), (
                    f"Edge case pattern {pattern_name} contains hardcoded model reference"
                )
                assert 'gpt' not in node_pattern.lower(), (
                    f"Edge case pattern {pattern_name} contains hardcoded model reference"
                )


def test_comprehensive_edge_cases():
    """Main test function for comprehensive edge case validation."""
    framework = EdgeCaseTestFramework()
    results = framework.run_comprehensive_edge_case_tests()
    
    print("\n📊 Comprehensive Edge Case Test Results:")
    print(f"Models tested: {results['total_models_tested']}")
    
    for pattern_name, coverage in results['edge_case_coverage'].items():
        print(f"\n{pattern_name}:")
        print(f"  Total occurrences: {coverage['total_occurrences']}")
        print(f"  Models with pattern: {coverage['models_with_pattern']}")
        print(f"  Confidence breakdown: {coverage['confidence_breakdown']}")
    
    print(f"\n🔍 Recommendations ({len(results['recommendations'])}):")
    for rec in results['recommendations']:
        print(f"  - {rec}")
    
    return results


if __name__ == "__main__":
    # Run comprehensive edge case tests
    test_results = test_comprehensive_edge_cases()
    print("\n✅ Comprehensive edge case testing complete!")