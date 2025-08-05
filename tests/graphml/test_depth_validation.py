"""
Test cases for GraphML depth validation with configurable limits.

This test suite validates that the GraphMLDepthValidator correctly:
- Detects deep hierarchies in both nested graphs and hierarchy tags
- Issues warnings at the configured warning threshold
- Fails validation when maximum depth is exceeded
- Provides meaningful metrics and error messages

Linear Task: TEZ-133 (Code Quality Improvements)
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from modelexport.graphml.constants import GRAPHML_CONST, GRAPHML_V13_KEYS
from modelexport.graphml.exceptions import GraphMLDepthError
from modelexport.graphml.validators import GraphMLDepthValidator, ValidationStatus

# Add timeout decorator for hanging tests
pytestmark = pytest.mark.timeout(30)


class TestGraphMLDepthValidation:
    """Test cases for GraphML depth validation."""
    
    @pytest.fixture
    def create_graphml_with_depth(self, tmp_path):
        """Helper to create GraphML files with specific depth."""
        
        def _create(graph_depth: int = 0, tag_depth: int = 0):
            """Create GraphML with specified graph nesting and hierarchy tag depth."""
            # Create root element
            root = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
            
            # Add key definitions
            for key_id, info in GRAPHML_V13_KEYS.items():
                key_elem = ET.SubElement(root, "key", 
                    id=key_id,
                    **{"for": info["for"], "attr.name": info["attr.name"], "attr.type": info["attr.type"]}
                )
            
            # Create nested graph structure
            current_parent = root
            for i in range(graph_depth + 1):
                graph = ET.SubElement(current_parent, "graph", 
                    id=f"graph_{i}", 
                    edgedefault="directed"
                )
                
                # Add a node with hierarchy tag
                node = ET.SubElement(graph, "node", id=f"node_{i}")
                
                # Add hierarchy tag with specified depth
                if tag_depth > 0:
                    tag_path = "/".join([f"Level{j}" for j in range(tag_depth + 1)])
                    tag_data = ET.SubElement(node, "data", key="n1")
                    tag_data.text = tag_path
                
                # Add op_type
                op_type = ET.SubElement(node, "data", key="n0")
                op_type.text = "TestOp"
                
                current_parent = graph
            
            # Save to file
            file_path = tmp_path / "test_depth.graphml"
            tree = ET.ElementTree(root)
            tree.write(file_path, encoding="utf-8", xml_declaration=True)
            
            return str(file_path)
            
        return _create
    
    def test_shallow_hierarchy_passes(self, create_graphml_with_depth):
        """Test that shallow hierarchies pass validation."""
        # Create GraphML with depth well below limits
        graphml_file = create_graphml_with_depth(graph_depth=5, tag_depth=10)
        
        validator = GraphMLDepthValidator()
        result = validator.validate(graphml_file)
        
        assert result.status == ValidationStatus.PASS
        assert "Depth validation passed" in result.message
        assert result.metrics['max_graph_depth'] == 5
        assert result.metrics['max_tag_depth'] == 10
    
    @pytest.mark.timeout(10)
    def test_warning_at_threshold(self, create_graphml_with_depth):
        """Test that warnings are issued at the warning threshold."""
        # Create GraphML with depth at warning threshold
        warn_depth = GRAPHML_CONST.WARN_GRAPH_DEPTH
        graphml_file = create_graphml_with_depth(
            graph_depth=warn_depth + 1, 
            tag_depth=warn_depth + 2
        )
        
        validator = GraphMLDepthValidator()
        result = validator.validate(graphml_file)
        
        assert result.status == ValidationStatus.WARNING
        assert "Deep hierarchies found" in result.message
        assert result.details['total_warnings'] > 0
        assert len(result.details['warnings']) > 0
    
    def test_failure_at_max_depth(self, create_graphml_with_depth):
        """Test that validation fails when max depth is exceeded."""
        # Create GraphML exceeding max depth
        max_depth = GRAPHML_CONST.MAX_GRAPH_DEPTH
        graphml_file = create_graphml_with_depth(graph_depth=max_depth + 1)
        
        validator = GraphMLDepthValidator()
        result = validator.validate(graphml_file)
        
        assert result.status == ValidationStatus.FAIL
        assert "exceeds maximum" in result.message
        assert result.error_code == "DEPTH_001"
    
    def test_custom_depth_limits(self, create_graphml_with_depth):
        """Test validation with custom depth limits."""
        # Create GraphML with moderate depth
        graphml_file = create_graphml_with_depth(graph_depth=15, tag_depth=20)
        
        # Test with strict custom limits
        validator = GraphMLDepthValidator(max_depth=10, warn_depth=5)
        result = validator.validate(graphml_file)
        
        assert result.status == ValidationStatus.FAIL
        assert result.metrics['configured_max_depth'] == 10
        assert result.metrics['configured_warn_depth'] == 5
    
    def test_hierarchy_tag_depth_detection(self, create_graphml_with_depth):
        """Test that hierarchy tag depth is correctly detected."""
        # Create GraphML with deep hierarchy tags but shallow graph nesting
        graphml_file = create_graphml_with_depth(graph_depth=2, tag_depth=60)
        
        validator = GraphMLDepthValidator()
        result = validator.validate(graphml_file)
        
        assert result.status == ValidationStatus.WARNING
        assert result.metrics['max_tag_depth'] == 60
        assert result.metrics['max_graph_depth'] == 2
    
    def test_metrics_collection(self, create_graphml_with_depth):
        """Test that depth metrics are correctly collected."""
        graphml_file = create_graphml_with_depth(graph_depth=10, tag_depth=15)
        
        validator = GraphMLDepthValidator(max_depth=100, warn_depth=50)
        result = validator.validate(graphml_file)
        
        # Check all expected metrics
        assert 'max_graph_depth' in result.metrics
        assert 'max_tag_depth' in result.metrics
        assert 'deep_paths_count' in result.metrics
        assert 'configured_max_depth' in result.metrics
        assert 'configured_warn_depth' in result.metrics
        
        assert result.metrics['max_graph_depth'] == 10
        assert result.metrics['max_tag_depth'] == 15
        assert result.metrics['configured_max_depth'] == 100
        assert result.metrics['configured_warn_depth'] == 50
    
    def test_empty_graphml(self, tmp_path):
        """Test validation of empty GraphML file."""
        # Create minimal valid GraphML
        root = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
        graph = ET.SubElement(root, "graph", id="G", edgedefault="directed")
        
        file_path = tmp_path / "empty.graphml"
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding="utf-8", xml_declaration=True)
        
        validator = GraphMLDepthValidator()
        result = validator.validate(str(file_path))
        
        assert result.status == ValidationStatus.PASS
        assert result.metrics['max_graph_depth'] == 0
        assert result.metrics['max_tag_depth'] == 0
    
    def test_integration_with_v13_validator(self, create_graphml_with_depth):
        """Test depth validation as part of the complete validation pipeline."""
        from modelexport.graphml.validators import GraphMLV13Validator
        
        # Create GraphML with warning-level depth
        graphml_file = create_graphml_with_depth(
            graph_depth=GRAPHML_CONST.WARN_GRAPH_DEPTH + 5
        )
        
        # Add required metadata to make it valid
        tree = ET.parse(graphml_file)
        root = tree.getroot()
        graph = root.find(".//{http://graphml.graphdrawing.org/xmlns}graph")
        
        # Add format version
        version_data = ET.SubElement(graph, "data", key="meta2")
        version_data.text = "1.3"
        
        # Add export timestamp
        timestamp_data = ET.SubElement(graph, "data", key="meta3")
        timestamp_data.text = "2025-08-05T12:00:00"
        
        # Add parameter strategy
        param_data = ET.SubElement(graph, "data", key="param0")
        param_data.text = "embedded"
        
        tree.write(graphml_file)
        
        # Validate with complete validator
        validator = GraphMLV13Validator()
        results = validator.validate_all(graphml_file)
        
        # Find depth validation result
        depth_results = [r for r in results if r.layer == "Depth"]
        assert len(depth_results) == 1
        assert depth_results[0].status == ValidationStatus.WARNING
    
    def test_performance_with_deep_hierarchy(self, create_graphml_with_depth):
        """Test that validation completes quickly even with deep hierarchies."""
        import time
        
        # Create GraphML with depth at warning level
        graphml_file = create_graphml_with_depth(
            graph_depth=GRAPHML_CONST.WARN_GRAPH_DEPTH - 5,
            tag_depth=GRAPHML_CONST.WARN_GRAPH_DEPTH - 5
        )
        
        validator = GraphMLDepthValidator()
        
        # Time the validation
        start_time = time.time()
        result = validator.validate(graphml_file)
        duration = time.time() - start_time
        
        assert result.status == ValidationStatus.PASS
        # Validation should complete quickly even with deep hierarchies
        assert duration < 0.1  # Less than 100ms


class TestGraphMLDepthError:
    """Test cases for GraphMLDepthError exception."""
    
    def test_depth_error_creation(self):
        """Test that GraphMLDepthError is created with correct details."""
        error = GraphMLDepthError(
            current_depth=150,
            max_depth=100,
            path="/Model/Layer1/Layer2/..."
        )
        
        assert "150" in str(error)
        assert "100" in str(error)
        assert "/Model/Layer1/Layer2/..." in str(error)
        assert error.details['current_depth'] == 150
        assert error.details['max_depth'] == 100
        assert error.details['path'] == "/Model/Layer1/Layer2/..."
    
    def test_depth_error_inheritance(self):
        """Test that GraphMLDepthError inherits correctly."""
        from modelexport.graphml.exceptions import GraphMLValidationError, GraphMLError
        
        error = GraphMLDepthError(100, 50)
        
        assert isinstance(error, GraphMLDepthError)
        assert isinstance(error, GraphMLValidationError)
        assert isinstance(error, GraphMLError)
        assert isinstance(error, Exception)