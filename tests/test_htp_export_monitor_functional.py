#!/usr/bin/env python3
"""
Minimal functional tests for HTP Export Monitor.

Tests core functionality without relying on internal implementation details.
"""

from pathlib import Path

import pytest

from modelexport.strategies.htp.export_monitor import HTPExportMonitor


class TestHTPExportMonitorFunctional:
    """Test HTP Export Monitor core functionality."""
    
    def test_monitor_creation(self, tmp_path):
        """Test that monitor can be created and initialized."""
        output_path = str(tmp_path / "test.onnx")
        
        monitor = HTPExportMonitor(
            output_path=output_path,
            model_name="test-model",
            verbose=True
        )
        
        assert monitor.output_path == output_path
        assert monitor.model_name == "test-model"
        assert monitor.verbose is True
    
    def test_context_manager(self, tmp_path):
        """Test monitor works as context manager."""
        output_path = str(tmp_path / "test.onnx")
        
        with HTPExportMonitor(output_path=output_path) as monitor:
            assert monitor is not None
        
        # Check metadata file was created
        metadata_path = output_path.replace('.onnx', '_htp_metadata.json')
        assert Path(metadata_path).exists()
    
    def test_basic_functionality(self, tmp_path):
        """Test basic monitor functionality without implementation details."""
        output_path = str(tmp_path / "test.onnx")
        
        # Just test that we can create and use the monitor
        monitor = HTPExportMonitor(output_path=output_path, verbose=False)
        
        # Should not raise
        with monitor:
            pass
        
        # Check basic data access  
        assert hasattr(monitor, 'data')
        assert hasattr(monitor, 'output_path')
        assert hasattr(monitor, 'model_name')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])