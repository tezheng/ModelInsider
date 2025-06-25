"""
Usage-based Strategy (Legacy)

This is the original tagging strategy that captures hierarchy based on module
usage during execution. Simpler approach but less accurate than modern strategies.

Key Features:
- Simple hook-based approach
- Basic tag propagation
- Legacy compatibility
- Lower coverage than FX/HTP

Note: This strategy is maintained for backward compatibility and baseline comparisons.
Consider using FX or HTP strategies for new applications.
"""

from .usage_based_exporter import UsageBasedExporter

__all__ = ["UsageBasedExporter"]