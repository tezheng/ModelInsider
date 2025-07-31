#!/usr/bin/env python3
"""
Integration prototype: Advanced Context Resolution in HierarchyExporter

This module demonstrates how to integrate the advanced context resolver
into the main HierarchyExporter for production deployment.
"""

import json

from advanced_context_resolver import AdvancedContextResolver, ContaminationCase


class EnhancedHierarchyExporter:
    """Enhanced HierarchyExporter with advanced context resolution."""
    
    def __init__(self, strategy='htp', enable_advanced_resolution=True):
        self.strategy = strategy
        self.enable_advanced_resolution = enable_advanced_resolution
        self.advanced_resolver = AdvancedContextResolver() if enable_advanced_resolution else None
        
        # Performance tracking
        self.resolution_stats = {
            'contamination_detected': 0,
            'contamination_resolved': 0,
            'resolution_confidence': 0.0,
            'patterns_detected': {},
            'processing_time': 0.0
        }
    
    def export_with_advanced_resolution(self, model, example_inputs, output_path, **kwargs):
        """Export with advanced context resolution post-processing."""
        
        print(f"🚀 Enhanced HTP Export with Advanced Context Resolution")
        
        # Step 1: Standard HTP export (using built-in tracking)
        print(f"📊 Step 1: Standard HTP export...")
        
        # Simulate standard export results (would be actual HierarchyExporter call)
        standard_result = self._simulate_standard_export(model, example_inputs, output_path, **kwargs)
        
        # Step 2: Load hierarchy metadata for analysis
        hierarchy_file = output_path.replace('.onnx', '_hierarchy.json')
        
        if self.enable_advanced_resolution and self.advanced_resolver:
            print(f"🔬 Step 2: Advanced context resolution...")
            
            # Load hierarchy data
            with open(hierarchy_file) as f:
                hierarchy_data = json.load(f)
            
            # Detect contamination cases
            contamination_cases = self._detect_contamination_cases(hierarchy_data)
            
            if contamination_cases:
                print(f"   Detected {len(contamination_cases)} contamination cases")
                
                # Apply advanced resolution
                resolution_results = self.advanced_resolver.resolve_contamination_cases(
                    contamination_cases, None, hierarchy_data
                )
                
                # Apply resolutions to hierarchy data
                enhanced_hierarchy = self._apply_resolutions(hierarchy_data, resolution_results)
                
                # Save enhanced hierarchy
                enhanced_hierarchy_file = output_path.replace('.onnx', '_enhanced_hierarchy.json')
                with open(enhanced_hierarchy_file, 'w') as f:
                    json.dump(enhanced_hierarchy, f, indent=2)
                
                # Update stats
                self._update_resolution_stats(resolution_results)
                
                print(f"   Resolved {len(resolution_results['resolved_cases'])} cases")
                print(f"   Enhanced hierarchy saved to: {enhanced_hierarchy_file}")
                
                return {
                    **standard_result,
                    'advanced_resolution': True,
                    'contamination_analysis': resolution_results,
                    'enhanced_hierarchy_file': enhanced_hierarchy_file,
                    'resolution_stats': self.resolution_stats
                }
            else:
                print(f"   No contamination detected - hierarchy is clean! ✨")
                return {
                    **standard_result,
                    'advanced_resolution': True,
                    'contamination_analysis': {'message': 'No contamination detected'},
                    'resolution_stats': self.resolution_stats
                }
        else:
            print(f"   Advanced resolution disabled")
            return standard_result
    
    def _simulate_standard_export(self, model, example_inputs, output_path, **kwargs):
        """Simulate standard HTP export (placeholder)."""
        
        # In real implementation, this would call the actual HierarchyExporter
        # For simulation, create a hierarchy file with known contamination
        
        simulated_hierarchy = {
            "version": "1.0",
            "format": "modelexport_hierarchy_htp_builtin", 
            "exporter": {"strategy": "htp_builtin"},
            "summary": {
                "total_operations": 25,
                "tagged_operations": 23,
                "builtin_tracking_enabled": True
            },
            "tag_statistics": {
                "/TestModel/Layer0": 8,
                "/TestModel/Layer1": 15
            },
            "node_tags": {
                "/layer.0/attention/Add": {
                    "op_type": "Add",
                    "tags": ["/TestModel/Layer1/Attention"]  # Contamination case
                },
                "/layer.1/intermediate/MatMul": {
                    "op_type": "MatMul", 
                    "tags": ["/TestModel/Layer0/Intermediate"]  # Contamination case
                },
                "/layer.0/linear/MatMul": {
                    "op_type": "MatMul",
                    "tags": ["/TestModel/Layer0/Linear"]  # Clean case
                },
                "/layer.1/output/Add": {
                    "op_type": "Add",
                    "tags": ["/TestModel/Layer1/Output"]  # Clean case
                }
            }
        }
        
        # Save simulated hierarchy
        hierarchy_file = output_path.replace('.onnx', '_hierarchy.json')
        with open(hierarchy_file, 'w') as f:
            json.dump(simulated_hierarchy, f, indent=2)
        
        return {
            "output_path": output_path,
            "strategy": "htp_builtin",
            "tagged_operations": 23,
            "hierarchy_file": hierarchy_file
        }
    
    def _detect_contamination_cases(self, hierarchy_data: dict) -> list[ContaminationCase]:
        """Detect contamination cases from hierarchy data."""
        
        contamination_cases = []
        node_tags = hierarchy_data.get('node_tags', {})
        
        for node_name, node_info in node_tags.items():
            tags = node_info.get('tags', [])
            op_type = node_info.get('op_type', 'Unknown')
            
            # Simple contamination detection: layer.X operations with LayerY tags
            node_layer = None
            if '/layer.0/' in node_name:
                node_layer = '0'
            elif '/layer.1/' in node_name:
                node_layer = '1'
            
            if node_layer is None:
                continue
            
            # Check for wrong layer tags
            contaminated = False
            for tag in tags:
                tag_layer = None
                if 'Layer0' in tag:
                    tag_layer = '0'
                elif 'Layer1' in tag:
                    tag_layer = '1'
                
                if tag_layer and tag_layer != node_layer:
                    contaminated = True
                    break
            
            if contaminated:
                expected_context = f"/TestModel/Layer{node_layer}"
                case = ContaminationCase(
                    node_name=node_name,
                    expected_context=expected_context,
                    actual_contexts=tags,
                    operation_type=op_type
                )
                contamination_cases.append(case)
        
        return contamination_cases
    
    def _apply_resolutions(self, hierarchy_data: dict, resolution_results: dict) -> dict:
        """Apply advanced resolutions to hierarchy data."""
        
        enhanced_hierarchy = hierarchy_data.copy()
        
        # Track resolution metadata
        enhanced_hierarchy['advanced_resolution'] = {
            'enabled': True,
            'total_contamination_cases': resolution_results['total_cases'],
            'resolved_cases': len(resolution_results['resolved_cases']),
            'resolution_rate': len(resolution_results['resolved_cases']) / resolution_results['total_cases'] if resolution_results['total_cases'] > 0 else 0,
            'strategies_used': list({res['strategy'] for res in resolution_results['resolved_cases']}),
            'confidence_average': sum(res['resolution'].confidence for res in resolution_results['resolved_cases']) / len(resolution_results['resolved_cases']) if resolution_results['resolved_cases'] else 0
        }
        
        # Apply actual resolutions to node tags
        for resolved in resolution_results['resolved_cases']:
            case = resolved['case']
            resolution = resolved['resolution']
            
            node_name = case.node_name
            if node_name in enhanced_hierarchy['node_tags']:
                # Update with resolved context assignment
                enhanced_hierarchy['node_tags'][node_name]['tags'] = [resolution.primary_context]
                if resolution.auxiliary_contexts:
                    enhanced_hierarchy['node_tags'][node_name]['auxiliary_tags'] = resolution.auxiliary_contexts
                
                enhanced_hierarchy['node_tags'][node_name]['resolution_info'] = {
                    'strategy': resolved['strategy'],
                    'confidence': resolution.confidence,
                    'assignment_type': resolution.assignment_type,
                    'reasoning': resolution.reasoning
                }
        
        # Update tag statistics
        new_tag_stats = {}
        for node_info in enhanced_hierarchy['node_tags'].values():
            for tag in node_info.get('tags', []):
                new_tag_stats[tag] = new_tag_stats.get(tag, 0) + 1
        
        enhanced_hierarchy['tag_statistics'] = new_tag_stats
        enhanced_hierarchy['summary']['contamination_resolved'] = len(resolution_results['resolved_cases'])
        
        return enhanced_hierarchy
    
    def _update_resolution_stats(self, resolution_results: dict):
        """Update resolution statistics."""
        
        self.resolution_stats['contamination_detected'] = resolution_results['total_cases']
        self.resolution_stats['contamination_resolved'] = len(resolution_results['resolved_cases'])
        
        if resolution_results['resolved_cases']:
            confidences = [res['resolution'].confidence for res in resolution_results['resolved_cases']]
            self.resolution_stats['resolution_confidence'] = sum(confidences) / len(confidences)
        
        # Track patterns detected
        strategy_counts = {}
        for resolved in resolution_results['resolved_cases']:
            strategy = resolved['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        self.resolution_stats['patterns_detected'] = strategy_counts
    
    def validate_enhanced_export(self, enhanced_hierarchy_file: str):
        """Validate the enhanced export results."""
        
        print(f"\n🔍 ENHANCED EXPORT VALIDATION")
        print("="*40)
        
        with open(enhanced_hierarchy_file) as f:
            enhanced_data = json.load(f)
        
        resolution_info = enhanced_data.get('advanced_resolution', {})
        
        print(f"Advanced Resolution Results:")
        print(f"  Contamination detected: {resolution_info.get('total_contamination_cases', 0)}")
        print(f"  Cases resolved: {resolution_info.get('resolved_cases', 0)}")
        print(f"  Resolution rate: {resolution_info.get('resolution_rate', 0)*100:.1f}%")
        print(f"  Average confidence: {resolution_info.get('confidence_average', 0):.3f}")
        print(f"  Strategies used: {resolution_info.get('strategies_used', [])}")
        
        # Check for remaining contamination
        remaining_contamination = self._detect_contamination_cases(enhanced_data)
        print(f"  Remaining contamination: {len(remaining_contamination)} cases")
        
        if len(remaining_contamination) == 0:
            print(f"  ✅ All contamination resolved successfully!")
        else:
            print(f"  ⚠️  {len(remaining_contamination)} cases still need attention")
        
        return len(remaining_contamination) == 0


def demo_enhanced_export():
    """Demonstrate enhanced export with advanced resolution."""
    
    print("🎯 ENHANCED HIERARCHY EXPORT DEMONSTRATION")
    print("="*60)
    
    # Create enhanced exporter
    exporter = EnhancedHierarchyExporter(
        strategy='htp',
        enable_advanced_resolution=True
    )
    
    # Simulate model and inputs (for demo)
    class DemoModel:
        pass
    
    model = DemoModel()
    example_inputs = "demo_inputs"
    output_path = "temp/enhanced_demo.onnx"
    
    # Run enhanced export
    result = exporter.export_with_advanced_resolution(
        model, example_inputs, output_path
    )
    
    print(f"\n📊 EXPORT RESULTS")
    print("="*30)
    print(f"Output path: {result['output_path']}")
    print(f"Strategy: {result['strategy']}")
    print(f"Tagged operations: {result['tagged_operations']}")
    print(f"Advanced resolution: {result['advanced_resolution']}")
    
    if 'enhanced_hierarchy_file' in result:
        print(f"Enhanced hierarchy: {result['enhanced_hierarchy_file']}")
        
        # Validate results
        is_clean = exporter.validate_enhanced_export(result['enhanced_hierarchy_file'])
        
        if is_clean:
            print(f"\n🎉 BREAKTHROUGH: Perfect hierarchy achieved!")
        else:
            print(f"\n🔧 Minor refinements needed for complete resolution")
    
    print(f"\nResolution Statistics:")
    stats = result.get('resolution_stats', {})
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demo_enhanced_export()