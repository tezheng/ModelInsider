"""
Constants used throughout the modelexport package.
"""

# Common model names for testing and demos
DEFAULT_TEST_MODEL = "prajjwal1/bert-tiny"

# File extensions
ONNX_EXTENSION = ".onnx"
GRAPHML_EXTENSION = ".graphml"
METADATA_EXTENSION = ".json"

# GraphML namespace and schema
GRAPHML_NAMESPACE = "http://graphml.graphdrawing.org/xmlns"
GRAPHML_SCHEMA_LOCATION = "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"

# Version information
from .version import GRAPHML_VERSION
SUPPORTED_ONNX_OPSET = 17

# Default configuration values
DEFAULT_BATCH_SIZE = 1
DEFAULT_SEQUENCE_LENGTH = 16
DEFAULT_TIMEOUT = 300  # 5 minutes
DEFAULT_MAX_RETRIES = 3

# Performance thresholds
DEFAULT_PERFORMANCE_THRESHOLD = 0.05  # 5% file size difference tolerance
DEFAULT_GRAPHML_OVERHEAD_THRESHOLD = 0.50  # 50% overhead threshold

# Error messages
ERROR_FILE_NOT_FOUND = "File not found: {path}"
ERROR_INVALID_FORMAT = "Invalid format version: {version}. Expected: {expected}"
ERROR_CONVERSION_FAILED = "Conversion failed: {reason}"
ERROR_VALIDATION_FAILED = "Validation failed: {reason}"