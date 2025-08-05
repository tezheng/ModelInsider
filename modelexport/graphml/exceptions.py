"""
GraphML-specific exception hierarchy for consistent error handling.

This module provides a structured exception hierarchy for all GraphML operations,
ensuring consistent error handling and meaningful error messages throughout the
GraphML conversion system.

Linear Task: TEZ-133 (Code Quality Improvements)
"""

from typing import Any, Dict, Optional


class GraphMLError(Exception):
    """
    Base exception for all GraphML operations.
    
    This exception class provides structured error handling with additional
    context information to help debugging and error recovery.
    
    Attributes:
        message: Human-readable error message
        details: Dictionary containing additional error context
        error_code: Optional error code for programmatic handling
    """
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ):
        """
        Initialize GraphML error with context.
        
        Args:
            message: Primary error message
            details: Additional context information
            error_code: Optional error code for categorization
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.error_code = error_code
        
    def __str__(self) -> str:
        """Format error with details for display."""
        if self.error_code:
            base = f"[{self.error_code}] {self.message}"
        else:
            base = self.message
            
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{base} ({detail_str})"
        return base


class GraphMLValidationError(GraphMLError):
    """
    Raised when GraphML validation fails.
    
    This includes schema validation, semantic validation, format validation,
    and any other validation-related errors.
    """
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)


class GraphMLConversionError(GraphMLError):
    """
    Raised when conversion between formats fails.
    
    This includes ONNX to GraphML conversion, GraphML to ONNX conversion,
    and any format transformation errors.
    """
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="CONVERSION_ERROR", **kwargs)


class GraphMLParameterError(GraphMLError):
    """
    Raised when parameter handling fails.
    
    This includes parameter loading, saving, strategy errors,
    and checksum validation failures.
    """
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="PARAMETER_ERROR", **kwargs)


class GraphMLDepthError(GraphMLValidationError):
    """
    Raised when graph depth exceeds configured limits.
    
    This is a specific validation error for hierarchy depth violations
    to prevent stack overflow and performance degradation.
    """
    
    def __init__(
        self, 
        current_depth: int, 
        max_depth: int, 
        path: str = "",
        **kwargs
    ):
        message = f"Graph depth {current_depth} at '{path}' exceeds maximum {max_depth}"
        details = kwargs.get("details", {})
        details.update({
            "current_depth": current_depth,
            "max_depth": max_depth,
            "path": path
        })
        super().__init__(message, details=details)


class GraphMLSchemaError(GraphMLValidationError):
    """
    Raised when GraphML structure violates schema requirements.
    
    This includes missing required keys, invalid key types,
    and structural violations.
    """
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="SCHEMA_ERROR", **kwargs)


class GraphMLSemanticError(GraphMLValidationError):
    """
    Raised when GraphML content violates semantic rules.
    
    This includes logical inconsistencies, invalid references,
    and semantic constraint violations.
    """
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="SEMANTIC_ERROR", **kwargs)


class GraphMLIOError(GraphMLError):
    """
    Raised when file I/O operations fail.
    
    This includes file not found, permission errors,
    and disk space issues.
    """
    
    def __init__(self, message: str, path: str = "", **kwargs):
        details = kwargs.get("details", {})
        details["path"] = path
        super().__init__(message, details=details, error_code="IO_ERROR")


class GraphMLTimeoutError(GraphMLError):
    """
    Raised when operations exceed configured timeouts.
    
    This helps prevent hanging operations and provides
    predictable behavior for long-running conversions.
    """
    
    def __init__(
        self, 
        operation: str, 
        timeout_seconds: float, 
        elapsed_seconds: float,
        **kwargs
    ):
        message = (
            f"Operation '{operation}' timed out after {elapsed_seconds:.1f}s "
            f"(timeout: {timeout_seconds}s)"
        )
        details = kwargs.get("details", {})
        details.update({
            "operation": operation,
            "timeout_seconds": timeout_seconds,
            "elapsed_seconds": elapsed_seconds
        })
        super().__init__(message, details=details, error_code="TIMEOUT_ERROR")


class GraphMLMemoryError(GraphMLError):
    """
    Raised when operations exceed memory limits.
    
    This helps prevent out-of-memory crashes and provides
    guidance for handling large models.
    """
    
    def __init__(
        self, 
        operation: str,
        memory_used_mb: float,
        memory_limit_mb: float,
        **kwargs
    ):
        message = (
            f"Operation '{operation}' exceeded memory limit: "
            f"{memory_used_mb:.1f}MB used, {memory_limit_mb:.1f}MB allowed"
        )
        details = kwargs.get("details", {})
        details.update({
            "operation": operation,
            "memory_used_mb": memory_used_mb,
            "memory_limit_mb": memory_limit_mb
        })
        super().__init__(message, details=details, error_code="MEMORY_ERROR")


class GraphMLNotImplementedError(GraphMLError):
    """
    Raised when attempting to use unimplemented features.
    
    This provides clear feedback about feature availability
    and potential workarounds.
    """
    
    def __init__(self, feature: str, workaround: str = "", **kwargs):
        message = f"Feature not implemented: {feature}"
        if workaround:
            message += f". Workaround: {workaround}"
        details = kwargs.get("details", {})
        details["feature"] = feature
        if workaround:
            details["workaround"] = workaround
        super().__init__(message, details=details, error_code="NOT_IMPLEMENTED")


class GraphMLSecurityError(GraphMLError):
    """
    Raised when security violations are detected.
    
    This includes XML injection attempts, path traversal,
    and other security-related issues.
    """
    
    def __init__(self, message: str, threat_type: str = "", **kwargs):
        details = kwargs.get("details", {})
        if threat_type:
            details["threat_type"] = threat_type
        super().__init__(message, details=details, error_code="SECURITY_ERROR")


# Error code constants for programmatic handling
ERROR_CODES = {
    "VALIDATION_ERROR": GraphMLValidationError,
    "CONVERSION_ERROR": GraphMLConversionError,
    "PARAMETER_ERROR": GraphMLParameterError,
    "SCHEMA_ERROR": GraphMLSchemaError,
    "SEMANTIC_ERROR": GraphMLSemanticError,
    "IO_ERROR": GraphMLIOError,
    "TIMEOUT_ERROR": GraphMLTimeoutError,
    "MEMORY_ERROR": GraphMLMemoryError,
    "NOT_IMPLEMENTED": GraphMLNotImplementedError,
    "SECURITY_ERROR": GraphMLSecurityError,
}


def raise_for_error_code(
    error_code: str, 
    message: str, 
    **kwargs
) -> None:
    """
    Raise appropriate exception based on error code.
    
    Args:
        error_code: Error code from ERROR_CODES
        message: Error message
        **kwargs: Additional arguments for the exception
        
    Raises:
        Appropriate GraphMLError subclass
    """
    error_class = ERROR_CODES.get(error_code, GraphMLError)
    raise error_class(message, **kwargs)