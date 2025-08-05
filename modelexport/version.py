"""
Centralized version management for modelexport.

This module provides a single source of truth for all version numbers used
throughout the project, following Python best practices.
"""

from typing import NamedTuple


class Version(NamedTuple):
    """Version information container."""
    major: int
    minor: int
    patch: int
    
    def __str__(self) -> str:
        """Return version as string (e.g., '1.3.0')."""
        return f"{self.major}.{self.minor}.{self.patch}"
    
    @property
    def short(self) -> str:
        """Return short version without patch (e.g., '1.3')."""
        return f"{self.major}.{self.minor}"
    
    @classmethod
    def from_string(cls, version_str: str) -> "Version":
        """Create Version from string like '1.3.0' or '1.3'."""
        parts = version_str.split('.')
        if len(parts) == 2:
            parts.append('0')  # Add patch version if missing
        elif len(parts) != 3:
            raise ValueError(f"Invalid version string: {version_str}")
        
        try:
            return cls(*map(int, parts))
        except ValueError:
            raise ValueError(f"Invalid version string: {version_str}")


# Project version (from pyproject.toml)
__version__ = "0.1.0"

# GraphML format version - export as string for direct import
GRAPHML_VERSION = "1.3"

# HTP metadata version - export as string for direct import
HTP_VERSION = "1.0"

# Version objects for internal use
_PROJECT_VERSION = Version(0, 1, 0)
_GRAPHML_VERSION = Version(1, 3, 0)
_HTP_VERSION = Version(1, 0, 0)

# Legacy versions for compatibility checking
GRAPHML_LEGACY_VERSIONS = {
    "1.1": Version(1, 1, 0),
    "1.2": Version(1, 2, 0),
}


def get_project_version() -> str:
    """Get the project version string (deprecated - use __version__ directly)."""
    return __version__


def get_graphml_version() -> str:
    """Get the GraphML format version string (deprecated - use GRAPHML_VERSION directly)."""
    return GRAPHML_VERSION


def get_htp_version() -> str:
    """Get the HTP metadata version string (deprecated - use HTP_VERSION directly)."""
    return HTP_VERSION


def is_graphml_version_supported(version_str: str) -> bool:
    """Check if a GraphML version is supported."""
    try:
        version = Version.from_string(version_str)
        # Currently only support exact version match
        return version.major == _GRAPHML_VERSION.major and version.minor == _GRAPHML_VERSION.minor
    except ValueError:
        return False


def get_all_graphml_versions() -> list[str]:
    """Get all known GraphML versions (current + legacy)."""
    versions = [GRAPHML_VERSION]
    versions.extend(GRAPHML_LEGACY_VERSIONS.keys())
    return sorted(versions, reverse=True)


def validate_version_consistency() -> bool:
    """Validate that all version references are consistent."""
    issues = []
    
    # Check that GRAPHML_FORMAT_VERSION matches our version
    from .graphml.constants import GRAPHML_FORMAT_VERSION
    if GRAPHML_FORMAT_VERSION != GRAPHML_VERSION:
        issues.append(f"GRAPHML_FORMAT_VERSION mismatch: {GRAPHML_FORMAT_VERSION} != {GRAPHML_VERSION}")
    
    # Check that constants.GRAPHML_VERSION matches
    from .constants import GRAPHML_VERSION as CONST_GRAPHML_VERSION
    if CONST_GRAPHML_VERSION != GRAPHML_VERSION:
        issues.append(f"constants.GRAPHML_VERSION mismatch: {CONST_GRAPHML_VERSION} != {GRAPHML_VERSION}")
    
    if issues:
        for issue in issues:
            print(f"Version consistency error: {issue}")
        return False
    
    return True