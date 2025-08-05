"""
Structured logging configuration for GraphML operations.

This module provides structured logging with automatic operation tracking,
timing, and error context for all GraphML operations. It uses structlog
for structured output and provides decorators for easy integration.

Linear Task: TEZ-133 (Code Quality Improvements)
"""

import logging
import os
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    
from ..version import GRAPHML_VERSION


# Default log level from environment or INFO
DEFAULT_LOG_LEVEL = os.getenv("GRAPHML_LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("GRAPHML_LOG_FILE", None)
STRUCTURED_LOGGING = os.getenv("GRAPHML_STRUCTURED_LOGS", "false").lower() == "true"


def setup_graphml_logging(
    level: str = DEFAULT_LOG_LEVEL,
    structured: bool = STRUCTURED_LOGGING,
    log_file: Optional[str] = LOG_FILE
) -> None:
    """
    Configure logging for GraphML module with structured output.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Whether to use structured JSON logging
        log_file: Optional log file path
    """
    if STRUCTLOG_AVAILABLE and structured:
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        
        if structured:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    # Configure stdlib logging
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def get_logger(name: str) -> Any:
    """
    Get a logger for GraphML operations.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance (structlog if available, stdlib otherwise)
    """
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name).bind(
            module="graphml",
            version=GRAPHML_VERSION
        )
    else:
        return logging.getLogger(name)


def log_operation(
    operation: str, 
    include_args: bool = False,
    include_result: bool = False,
    log_level: str = "INFO"
) -> Callable:
    """
    Decorator for automatic operation logging with timing.
    
    This decorator automatically logs operation start, completion, duration,
    and any errors that occur. It's designed to provide consistent logging
    across all GraphML operations.
    
    Args:
        operation: Name of the operation being logged
        include_args: Whether to include function arguments in logs
        include_result: Whether to include function result in logs
        log_level: Log level for successful operations
        
    Returns:
        Decorated function with automatic logging
        
    Example:
        @log_operation("graphml_conversion", include_args=True)
        def convert_to_graphml(model_path: str) -> str:
            # Function implementation
            return output_path
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = get_logger(func.__module__)
            
            # Prepare log context
            log_context = {"operation": operation}
            
            # Add function arguments if requested
            if include_args:
                # Skip 'self' for methods
                func_args = args[1:] if args and hasattr(args[0], '__class__') else args
                if func_args:
                    log_context["args"] = str(func_args)[:200]  # Truncate long args
                if kwargs:
                    log_context["kwargs"] = str(kwargs)[:200]
                    
            # Log operation start
            log_method = getattr(logger, log_level.lower(), logger.info)
            log_method(f"{operation}_started", **log_context)
            
            start_time = time.perf_counter()
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Calculate duration
                duration = time.perf_counter() - start_time
                log_context["duration_ms"] = round(duration * 1000, 2)
                
                # Add result if requested
                if include_result:
                    result_str = str(result)[:200]  # Truncate long results
                    log_context["result"] = result_str
                
                # Log successful completion
                log_method(f"{operation}_completed", **log_context)
                
                return result
                
            except Exception as e:
                # Calculate duration even for errors
                duration = time.perf_counter() - start_time
                log_context.update({
                    "duration_ms": round(duration * 1000, 2),
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                
                # Log error with full context
                logger.error(
                    f"{operation}_failed",
                    exc_info=True,
                    **log_context
                )
                
                # Re-raise the exception
                raise
                
        return wrapper
    return decorator


def log_validation(validation_type: str) -> Callable:
    """
    Specialized decorator for validation operations.
    
    Args:
        validation_type: Type of validation (schema, semantic, round_trip)
        
    Returns:
        Decorated function with validation-specific logging
    """
    return log_operation(
        f"validation_{validation_type}",
        include_args=True,
        include_result=True,
        log_level="DEBUG"
    )


def log_conversion(
    source_format: str,
    target_format: str
) -> Callable:
    """
    Specialized decorator for conversion operations.
    
    Args:
        source_format: Source format (e.g., "onnx")
        target_format: Target format (e.g., "graphml")
        
    Returns:
        Decorated function with conversion-specific logging
    """
    return log_operation(
        f"convert_{source_format}_to_{target_format}",
        include_args=True,
        log_level="INFO"
    )


class LogContext:
    """
    Context manager for adding temporary logging context.
    
    This allows adding context that will be included in all logs
    within the context block.
    
    Example:
        with LogContext(model_name="bert", size_mb=500):
            # All logs within this block will include model_name and size_mb
            converter.convert(model_path)
    """
    
    def __init__(self, **context):
        """Initialize with context key-value pairs."""
        self.context = context
        self.logger = None
        self.original_context = None
        
    def __enter__(self):
        """Add context to logger."""
        if STRUCTLOG_AVAILABLE:
            self.logger = structlog.get_context()
            self.original_context = dict(self.logger)
            self.logger.update(self.context)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original context."""
        if STRUCTLOG_AVAILABLE and self.logger:
            self.logger.clear()
            if self.original_context:
                self.logger.update(self.original_context)


# Initialize logging on module import
setup_graphml_logging()


# Module-level logger for internal use
logger = get_logger(__name__)


# Log GraphML module initialization
logger.info(
    "graphml_module_initialized",
    version=GRAPHML_VERSION,
    structured_logging=STRUCTLOG_AVAILABLE,
    log_level=DEFAULT_LOG_LEVEL
)