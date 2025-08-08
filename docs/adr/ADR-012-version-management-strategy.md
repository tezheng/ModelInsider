# ADR-012: Version Management Strategy

| Status | Date | Decision Maker(s) | Consulted | Informed |
|--------|------|-------------------|-----------|----------|
| Accepted | 2025-08-07 | Development Team | Architecture Team | Engineering Team |

## Context and Problem Statement

The modelexport project currently has a convoluted version management system with multiple architectural issues that compromise maintainability and create circular dependencies. The existing system conflates different versioning concerns and implements unnecessary complexity for what should be simple version string management.

**Current Issues:**
1. **Circular Dependencies**: GraphML module imports its version from the parent module, creating import cycles
2. **Overengineering**: Complex Version classes and validation functions for simple version strings that could be handled with standard Python conventions
3. **Conflated Concerns**: Package version, format versions, and strategy versions are mixed together without clear separation
4. **Hardcoded Duplicates**: Same version defined in multiple places (`version.py`, `constants.py`, module-specific `_version.py` files)
5. **No Clear Ownership**: Unclear which module owns which version, leading to maintenance confusion

**System Requirements:**
The system has three distinct versioning needs that should be managed independently:
- **Package version**: Software release version (modelexport package) - follows software release cycles
- **Format versions**: GraphML schema version (data format specification) - changes only when schema evolves
- **Strategy versions**: HTP algorithm version (export strategy implementation) - changes when algorithm logic changes

## Decision Drivers

- **Maintainability**: Eliminate circular dependencies and reduce complexity
- **Python Convention Compliance**: Follow standard `__version__` patterns used throughout Python ecosystem
- **Clear Separation of Concerns**: Each version type should have a clear owner and purpose
- **Backward Compatibility**: Maintain ability to check spec compatibility across versions
- **Development Velocity**: Simplify version bumping and release processes
- **Debugging Support**: Provide full traceability for troubleshooting

## Considered Options

1. **Centralized Version Management** (current approach)
2. **Distributed Module-Owned Versions** (proposed)
3. **Hybrid Approach with Central Registry**

## Decision Outcome

**Chosen option**: **Hybrid Approach with pyproject.toml Integration** - Package version uses dynamic reading from both importlib.metadata and pyproject.toml, while specification versions (GraphML, HTP) remain simple hardcoded values in their respective modules.

### Rationale

This approach provides the best of both worlds: robust package version management that works in both installed and development environments, while keeping specification versions simple and module-owned. The hybrid approach eliminates circular dependencies, follows Python best practices, and provides excellent development workflow support including editable installs.

### Consequences

**Positive:**
- **Robust package versioning**: Works seamlessly in installed packages, development environments, and editable installs
- **Development workflow optimized**: Automatic ".dev0" suffix for development versions, proper version detection
- **Python 3.11+ optimized**: Uses native tomllib for efficient pyproject.toml parsing (project requires Python 3.12+)
- **Clear separation of concerns**: Package version (dynamic) vs specification versions (hardcoded)
- **Python convention compliance**: Uses standard `__version__` pattern with enhanced fallback mechanisms
- **No circular dependencies**: Package version isolated from specification modules
- **Simple specification management**: GraphML and HTP versions remain simple hardcoded strings
- **Excellent debugging support**: Version source traceability and development indicators

**Negative:**
- **Implementation complexity**: Package version logic is more sophisticated than simple hardcoding
- **Python version dependency**: tomllib requires Python 3.11+ (acceptable since project requires 3.12+)
- **Fallback chain complexity**: Multiple fallback mechanisms require careful testing

**Neutral:**
- **Hybrid approach**: Package version is dynamic, spec versions are static (clear architectural decision)
- **Development indicators**: ".dev0" suffix clearly identifies development environments

## Implementation Notes

### Version Format Strategy
- **Package Version**: Dynamic reading from pyproject.toml with development indicators
- **Specification Versions**: Semantic versioning with specification/implementation split
  - **Format**: `MAJOR.MINOR.PATCH` (e.g., "1.3.2")
  - **MAJOR.MINOR**: Specification version (schema/format/algorithm changes)
  - **PATCH**: Implementation version (bug fixes, optimizations, no spec changes)

### Hybrid Implementation Approach

#### Package Version (Dynamic) - Simplified Approach
```python
# modelexport/__init__.py (version logic integrated)

# ... existing imports ...

# Package version with hybrid pyproject.toml integration
def _get_version() -> str:
    """
    Get package version using hybrid approach:
    1. Try importlib.metadata (installed package)
    2. Try pyproject.toml (development/editable install)  
    3. Fallback to development version
    """
    try:
        from importlib.metadata import version, PackageNotFoundError
        return version("modelexport")
    except (PackageNotFoundError, ImportError):
        pass
    
    # Try reading from pyproject.toml (development environment)
    try:
        import tomllib
        from pathlib import Path
        
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                pyproject_data = tomllib.load(f)
            base_version = pyproject_data["project"]["version"]
            return f"{base_version}.dev0"
    except Exception:
        pass
    
    # Final fallback
    return "0.1.0.dev0"

__version__ = _get_version()

# ... rest of package exports ...
```

#### Specification Versions (Simple Hardcoded)
```python
# modelexport/graphml/__init__.py
__version__ = "1.3.0"  # GraphML format version
__spec_version__ = ".".join(__version__.split(".")[:2])  # "1.3"

# modelexport/strategies/htp/__init__.py
__version__ = "1.0.0"  # HTP strategy version
__spec_version__ = ".".join(__version__.split(".")[:2])  # "1.0"
```

### Development Workflow Considerations

#### Version Detection in Different Environments
- **Installed Package**: `importlib.metadata.version()` returns clean version (e.g., "0.1.0")
- **Editable Install**: `importlib.metadata.version()` may return version with local identifiers
- **Development Source**: `pyproject.toml` reading with ".dev0" suffix (e.g., "0.1.0.dev0")

#### Development Version Suffix Convention
- **".dev0" Suffix**: Indicates development/source environment (PEP 440 compliant)
- **Clear Distinction**: Easy to identify non-release versions in logs and debugging
- **Tool Compatibility**: Better support in pip, setuptools, and other packaging tools

#### Build and Release Process
- **pyproject.toml**: Single source of truth for release version
- **Automated CI/CD**: Version bumping updates only pyproject.toml
- **Git Tags**: Should match pyproject.toml version for consistency
- **Development Builds**: Automatically get ".dev0" suffix when not installed

#### Error Handling Strategy
- **Silent Fallbacks**: Default behavior is silent fallback to prevent log noise
- **Optional Verbose Mode**: Enable via `MODELEXPORT_VERBOSE=1` environment variable for debugging
- **Library Behavior**: As a library, avoid polluting application logs by default
- **Development Support**: Verbose mode provides detailed version detection debugging

### Implementation Phases
1. **Phase 1**: Implement hybrid `_get_version()` function directly in `__init__.py` with tomllib integration
2. **Phase 2**: Remove existing `_version.py` if present and consolidate version logic
3. **Phase 3**: Add specification versions to GraphML and HTP modules
4. **Phase 4**: Update dependent code (CLI, metadata writers, tests) to use new version imports
5. **Phase 5**: Clean up legacy version system (remove old `version.py` complexity)
6. **Phase 6**: Documentation and testing (README, contribution guidelines, test coverage)

## Validation/Confirmation

- All existing tests pass with hybrid version implementation
- No circular import errors detected in test suite
- Version detection works correctly in all environments:
  - Installed packages (clean version from metadata)
  - Editable installs (version with local identifiers)
  - Development source (version with ".dev0" suffix)
- CLI commands correctly display all version information with proper environment indicators
- Metadata generation includes proper version information with source traceability
- pyproject.toml parsing works correctly with tomllib (Python 3.12+ requirement satisfied)
- Fallback mechanisms handle all edge cases gracefully
- Development workflow provides clear version differentiation

## Detailed Analysis of Options

### Option 1: Centralized Version Management (Current)
- **Description**: Single `version.py` module manages all versions with complex validation
- **Pros**: 
  - Single source of truth
  - Centralized validation logic
- **Cons**: 
  - Creates circular dependencies
  - Overengineered for simple version strings
  - Conflates different version concerns
  - Difficult to maintain and extend
- **Technical Impact**: High complexity, circular imports, maintenance overhead

### Option 2: Distributed Module-Owned Versions 
- **Description**: Each module owns its version in `__init__.py` following Python conventions
- **Pros**:
  - Eliminates circular dependencies
  - Follows Python standards
  - Clear separation of concerns
  - Simple to understand and maintain
- **Cons**:
  - Package version management complexity
  - Development environment detection challenges
  - Multiple version locations
- **Technical Impact**: Low complexity for specs, but package version needs enhancement

### Option 3: Hybrid Approach with pyproject.toml Integration (Chosen)
- **Description**: Dynamic package version with robust fallback mechanisms, simple hardcoded specification versions
- **Pros**:
  - Robust package version management across all environments
  - Simple specification version management
  - Excellent development workflow support
  - Python 3.11+ optimized with tomllib
  - Clear architectural separation
  - Development environment indicators
- **Cons**:
  - Package version logic complexity
  - Requires Python 3.11+ for tomllib (satisfied by 3.12+ requirement)
  - Multiple fallback mechanisms to test
- **Technical Impact**: Medium complexity for package version, low complexity for specs, excellent user experience

## Related Decisions

- ADR-010: ONNX to GraphML Format Specification (defines GraphML versioning requirements)
- ADR-009: GraphML Converter Architecture (impacts version handling in conversion pipeline)

## More Information

- [Semantic Versioning 2.0.0](https://semver.org/) - Versioning specification
- [PEP 396 -- Module Version Numbers](https://www.python.org/dev/peps/pep-0396/) - Python version standards
- [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/single-sourcing-package-version/) - Best practices
- Version Bumping Guidelines documented in project README
- Migration scripts available in `temp/` directory

---
*Last updated: 2025-08-07*
*Updated: 2025-08-07 - Clarified implementation details: simplified approach in __init__.py, .dev0 suffix for PEP 440 compliance, silent fallbacks with optional verbose logging*
*Next review: 2025-11-07*