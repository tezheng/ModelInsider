"""
Shared test utilities for GraphML tests.
"""

import xml.etree.ElementTree as ET


def get_graphml_content(converter_output):
    """
    Helper function to get GraphML content from converter output.

    Args:
        converter_output: Output from converter.convert() - either string (flat) or dict (hierarchical)

    Returns:
        tuple: (graphml_content_as_string, root_element)
    """
    if isinstance(converter_output, dict):
        # Hierarchical mode - read from file
        graphml_path = converter_output["graphml"]
        root = ET.parse(graphml_path).getroot()
        with open(graphml_path) as f:
            content = f.read()
        return content, root
    else:
        # Flat mode - parse string directly
        root = ET.fromstring(converter_output)
        return converter_output, root
