"""Entry point for the modelexport package when run as a module."""

import os
import sys

# Add the parent directory to sys.path to enable absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelexport.cli import cli

if __name__ == '__main__':
    cli()