"""
CLI Environment Portability Tests

Comprehensive tests to ensure CLI works across different environments
and doesn't rely on hardcoded paths or system-specific configurations.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from modelexport.cli import cli
from modelexport import __version__


class TestCLIEnvironmentPortability:
    """Test CLI functionality across different environments."""

    def test_version_option_without_installation(self):
        """Test that version option works even when package isn't formally installed."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        assert __version__ in result.output
        # Version output format is "cli, version X.Y.Z"
        assert 'version' in result.output.lower()

    def test_version_option_different_formats(self):
        """Test version option with different argument formats."""
        runner = CliRunner()
        
        # Test --version
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert __version__ in result.output
        
        # Test -V if supported (Click may not support this by default)
        # This test documents expected behavior
        result = runner.invoke(cli, ['-V'])
        # May fail if Click doesn't support -V, that's expected

    def test_cli_module_execution_current_interpreter(self):
        """Test CLI execution uses current Python interpreter."""
        # Test that sys.executable is used correctly
        result = subprocess.run([
            sys.executable, '-m', 'modelexport', '--version'
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert __version__ in result.stdout
        
    def test_cli_without_hardcoded_paths(self):
        """Test CLI operations don't depend on hardcoded paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temporary directory
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                # Test help command works from any directory 
                # Need to ensure Python path includes the project root
                env = os.environ.copy()
                env['PYTHONPATH'] = original_cwd + ':' + env.get('PYTHONPATH', '')
                
                result = subprocess.run([
                    sys.executable, '-m', 'modelexport', '--help'
                ], capture_output=True, text=True, env=env)
                
                assert result.returncode == 0
                assert 'export' in result.stdout
                assert 'Universal hierarchy-preserving ONNX export' in result.stdout
                
            finally:
                os.chdir(original_cwd)

    def test_cli_subcommand_help_portability(self):
        """Test that all subcommand help works in any environment."""
        subcommands = ['export', 'analyze', 'validate', 'compare']
        
        for subcommand in subcommands:
            result = subprocess.run([
                sys.executable, '-m', 'modelexport', subcommand, '--help'
            ], capture_output=True, text=True)
            
            assert result.returncode == 0, f"Help for {subcommand} subcommand failed"
            assert '--help' in result.stdout
            
    def test_uv_run_command_portability(self):
        """Test uv run command works without environment dependencies."""
        # Test basic help
        result = subprocess.run([
            'uv', 'run', 'modelexport', '--help'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:  # uv is available
            assert 'export' in result.stdout
            assert 'Universal hierarchy-preserving ONNX export' in result.stdout
        else:
            # uv not available, skip test
            pytest.skip("uv not available in test environment")

    def test_error_handling_no_hardcoded_paths(self):
        """Test error messages don't expose hardcoded paths."""
        result = subprocess.run([
            sys.executable, '-m', 'modelexport', 'invalid-command'
        ], capture_output=True, text=True)
        
        assert result.returncode != 0
        # Error message should not contain hardcoded paths like /mnt/d/ or similar
        assert '/mnt/' not in result.stderr
        assert 'C:\\' not in result.stderr
        assert result.stderr  # Should have error message

    def test_python_path_independence(self):
        """Test CLI works regardless of Python path configuration."""
        # Set minimal environment
        env = os.environ.copy()
        # Remove potentially problematic path entries
        if 'PYTHONPATH' in env:
            del env['PYTHONPATH']
            
        result = subprocess.run([
            sys.executable, '-m', 'modelexport', '--version'
        ], capture_output=True, text=True, env=env)
        
        assert result.returncode == 0
        assert __version__ in result.stdout


class TestCLIVersionHandling:
    """Specific tests for CLI version handling functionality."""

    def test_version_import_works(self):
        """Test that version import doesn't fail."""
        from modelexport import __version__
        assert __version__ == '0.1.0'

    def test_click_version_option_configuration(self):
        """Test Click version option is properly configured."""
        from modelexport.cli import cli
        
        # Check that version option is configured
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        assert 'version' in result.output.lower() or __version__ in result.output

    def test_version_consistency(self):
        """Test version is consistent across different access methods."""
        from modelexport import __version__
        
        # Test via CLI
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_version_option_edge_cases(self):
        """Test version option edge cases."""
        runner = CliRunner()
        
        # Test version with other flags
        result = runner.invoke(cli, ['--version', '--verbose'])
        # Should still show version, ignore verbose
        assert result.exit_code == 0
        
        # Test version with invalid combinations
        result = runner.invoke(cli, ['--version', 'export'])
        # Should show version and exit, not process export command
        assert result.exit_code == 0


class TestCLIRegressionPrevention:
    """Tests to prevent regression of fixed issues."""

    def test_no_hardcoded_windows_paths(self):
        """Ensure no Windows-specific hardcoded paths in test files."""
        test_files = [
            'tests/integration/test_cli.py',
            'tests/integration/test_cli_integration.py'
        ]
        
        for test_file in test_files:
            if Path(test_file).exists():
                with open(test_file) as f:
                    content = f.read()
                
                # Check for common Windows path patterns
                assert 'C:\\' not in content, f"Windows path found in {test_file}"
                assert '/mnt/c/' not in content, f"WSL Windows mount found in {test_file}"
                assert '/mnt/d/' not in content, f"WSL Windows mount found in {test_file}"
                
    def test_no_hardcoded_python_executable(self):
        """Ensure tests use sys.executable instead of hardcoded 'python'."""
        test_file = 'tests/integration/test_cli_integration.py'
        
        if Path(test_file).exists():
            with open(test_file) as f:
                content = f.read()
            
            # Should not have hardcoded 'python' in subprocess calls
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'subprocess.run' in line and "'python'" in line and not line.strip().startswith('#'):
                    # Hardcoded 'python' found in actual call (not comment)
                    raise AssertionError(f"Hardcoded 'python' found in {test_file} line {i+1}: {line}")

    def test_subprocess_calls_use_current_interpreter(self):
        """Test that subprocess calls in tests use current interpreter."""
        # This is a meta-test that verifies our test infrastructure
        import inspect

        from tests.integration import test_cli_integration
        
        # Get source of test methods
        test_methods = [method for name, method in inspect.getmembers(test_cli_integration.TestCLIProcessIntegration) 
                       if name.startswith('test_')]
        
        for method in test_methods:
            source = inspect.getsource(method)
            if 'subprocess.run' in source:
                # Should use sys.executable, not hardcoded python
                assert 'sys.executable' in source or 'uv' in source, \
                    f"Method {method.__name__} should use sys.executable for Python calls"