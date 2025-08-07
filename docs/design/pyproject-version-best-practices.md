# Best Practices for Version Management with pyproject.toml

## Executive Summary

Modern Python projects need a single source of truth for version information. With `pyproject.toml` as the standard, there are three main approaches, each with trade-offs.

## The Three Approaches

### 1. Static Version in pyproject.toml (Recommended for Most Projects)

**pyproject.toml:**
```toml
[project]
name = "modelexport"
version = "0.1.0"  # Single source of truth
```

**modelexport/__init__.py:**
```python
"""Package docstring."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("modelexport")
except PackageNotFoundError:
    # Development mode - package not installed
    __version__ = "dev"
```

**Pros:**
- ✅ Simple and standard
- ✅ Works with all build backends
- ✅ Version visible in pyproject.toml
- ✅ No import-time file I/O

**Cons:**
- ❌ Doesn't work without installation (even editable)
- ❌ Need to reinstall after version changes
- ❌ Shows "dev" in development mode

### 2. Dynamic Version from Source Code

**pyproject.toml:**
```toml
[project]
name = "modelexport"
dynamic = ["version"]

[tool.setuptools.dynamic]
version = {attr = "modelexport._version.__version__"}
```

**modelexport/_version.py:**
```python
"""Version information."""
__version__ = "0.1.0"  # Single source of truth
```

**modelexport/__init__.py:**
```python
"""Package docstring."""
from ._version import __version__
```

**Pros:**
- ✅ Works without installation
- ✅ Version always available
- ✅ Simple import pattern

**Cons:**
- ❌ Requires setuptools (or similar)
- ❌ Version not visible in pyproject.toml
- ❌ Extra file just for version

### 3. Git-Based Dynamic Version (setuptools-scm)

**pyproject.toml:**
```toml
[build-system]
requires = ["setuptools>=64", "setuptools-scm>=8"]

[project]
name = "modelexport"
dynamic = ["version"]

[tool.setuptools_scm]
version_file = "modelexport/_version.py"
```

**Pros:**
- ✅ Automatic versioning from git tags
- ✅ No manual version updates
- ✅ Includes commit info in dev versions

**Cons:**
- ❌ Requires git
- ❌ More complex setup
- ❌ Can't work without git history

## The Hybrid Approach (Best of Both Worlds)

For maximum compatibility and developer experience:

**pyproject.toml:**
```toml
[project]
name = "modelexport"
version = "0.1.0"  # Canonical version
```

**modelexport/__init__.py:**
```python
"""Universal hierarchy-preserving ONNX export."""

def _get_version() -> str:
    """Get version from package metadata or pyproject.toml."""
    # Try 1: Get from installed package metadata
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("modelexport")
        except PackageNotFoundError:
            pass  # Not installed, try next method
    except ImportError:
        pass  # Python < 3.8, try next method
    
    # Try 2: Read from pyproject.toml in development
    try:
        from pathlib import Path
        
        # Handle different Python versions for TOML parsing
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib  # Fallback for older Python
            except ImportError:
                tomllib = None
        
        if tomllib:
            root = Path(__file__).resolve().parent.parent
            pyproject_path = root / "pyproject.toml"
            
            if pyproject_path.exists():
                with open(pyproject_path, "rb") as f:
                    data = tomllib.load(f)
                    version = data.get("project", {}).get("version")
                    if version:
                        return f"{version}+dev"  # Mark as development
    except Exception:
        pass  # Any error in reading pyproject.toml
    
    # Try 3: Git describe for development
    try:
        import subprocess
        result = subprocess.run(
            ["git", "describe", "--tags", "--dirty", "--always"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent
        )
        return result.stdout.strip()
    except Exception:
        pass  # Git not available or not in repo
    
    # Final fallback
    return "unknown"

__version__ = _get_version()
```

## Edge Cases and Solutions

### 1. Editable Installs

```bash
# Modern editable install (PEP 660)
pip install -e .

# After changing version in pyproject.toml
pip install -e . --force-reinstall
```

**Note:** With editable installs, `importlib.metadata` should work, but you need to reinstall after version changes.

### 2. Multiple Python Versions

```python
# Support Python 3.8+
try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Add tomli to dev dependencies

# Or for minimal dependencies
try:
    import tomllib
except ImportError:
    # Simple regex fallback
    import re
    from pathlib import Path
    
    pyproject_text = Path("pyproject.toml").read_text()
    match = re.search(r'version\s*=\s*"([^"]+)"', pyproject_text)
    version = match.group(1) if match else "unknown"
```

### 3. Namespace Packages

For namespace packages, ensure each sub-package has its own version:

```python
# modelexport/graphml/__init__.py
__version__ = "1.3.0"  # GraphML format version

# modelexport/strategies/htp/__init__.py  
__version__ = "1.0.0"  # HTP strategy version
```

## Real-World Examples

### NumPy
```python
# Uses a generated version file during build
from numpy.version import version as __version__
```

### Requests
```python
# Hardcoded in source
__version__ = "2.31.0"
```

### FastAPI
```python
# Uses importlib.metadata with fallback
try:
    from importlib.metadata import version
    __version__ = version("fastapi")
except Exception:
    __version__ = "0.0.0"
```

### Pytest
```python
# Dynamic from setuptools_scm
from _pytest._version import version as __version__
```

## Recommendations by Use Case

### For Libraries
Use **Static Version in pyproject.toml** with importlib.metadata:
- Clear version in pyproject.toml
- Standard approach
- Works with all package managers

### For Applications
Use **Git-based versioning** with setuptools-scm:
- Automatic version from tags
- Includes commit info
- Great for CI/CD

### For Development-Heavy Projects
Use **Hybrid Approach**:
- Always shows correct version
- Works in all scenarios
- Good developer experience

### For Simple Projects
Use **Dynamic from Source**:
- Simple and always works
- No external dependencies
- Easy to understand

## Implementation Checklist

- [ ] Choose versioning strategy based on project type
- [ ] Add version to pyproject.toml or source file
- [ ] Implement __version__ in __init__.py
- [ ] Handle PackageNotFoundError gracefully
- [ ] Test in both installed and development modes
- [ ] Document version bumping process
- [ ] Consider CI/CD integration
- [ ] Add version to CLI (`--version` flag)
- [ ] Include version in logs/reports
- [ ] Test with editable installs

## Common Pitfalls

1. **Don't use `except Exception`** - Be specific about exceptions
2. **Don't hardcode fallback versions** - They become outdated
3. **Don't forget editable installs** - Test `pip install -e .`
4. **Don't ignore Python version compatibility** - tomllib is 3.11+
5. **Don't mix approaches** - Choose one and stick with it

## Testing Version Management

```python
# tests/test_version.py
import pytest
import sys
from pathlib import Path

def test_version_accessible():
    """Test that version is accessible."""
    import modelexport
    assert hasattr(modelexport, "__version__")
    assert modelexport.__version__ != "unknown"

def test_version_format():
    """Test version follows semver."""
    import modelexport
    import re
    
    # Match semver with optional dev suffix
    pattern = r'^\d+\.\d+\.\d+([+-].+)?$|^dev$|^\d+\.\d+\.\d+-\d+-g[0-9a-f]+$'
    assert re.match(pattern, modelexport.__version__)

def test_version_consistency():
    """Test version matches pyproject.toml."""
    import tomllib
    import modelexport
    
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    
    pyproject_version = data["project"]["version"]
    
    # In development, version might have +dev suffix
    assert (modelexport.__version__ == pyproject_version or
            modelexport.__version__.startswith(f"{pyproject_version}+"))
```

## Summary

The best practice for 2024 is:

1. **Keep version in pyproject.toml** as the canonical source
2. **Use importlib.metadata** to read it at runtime
3. **Provide fallbacks** for development mode
4. **Be explicit** about exception handling
5. **Test thoroughly** in different installation modes

The hybrid approach provides the best developer experience while maintaining compatibility with modern Python packaging standards.