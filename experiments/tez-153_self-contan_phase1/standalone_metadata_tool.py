#!/usr/bin/env python3
"""
Standalone Metadata Embedding Tool for ONNX Models.

This tool can add metadata to ANY ONNX model, regardless of how it was exported.
It operates as a completely isolated post-processing step.

Usage:
    # Add metadata to existing ONNX model
    python standalone_metadata_tool.py embed model.onnx bert-base-uncased
    
    # Read metadata from ONNX model
    python standalone_metadata_tool.py read model.onnx
    
    # Validate metadata in ONNX model
    python standalone_metadata_tool.py validate model.onnx
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.tez_153_onnx_infer_2.metadata_utils import (
    MetadataManager,
    MetadataDiscovery,
    ONNXMetadataEmbedder,
    ONNXMetadataReader,
    MetadataValidator
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class StandaloneMetadataTool:
    """
    Standalone tool for embedding metadata into ANY ONNX model.
    
    This tool is completely independent of the export method used.
    It can work with ONNX models from:
    - HTP Exporter
    - Optimum
    - torch.onnx.export
    - ONNX Runtime Training
    - Any other ONNX exporter
    """
    
    def __init__(self):
        self.discovery = MetadataDiscovery()
        self.embedder = ONNXMetadataEmbedder()
        self.reader = ONNXMetadataReader()
        self.validator = MetadataValidator()
    
    def embed_metadata(
        self,
        onnx_path: str,
        model_name: str,
        output_path: Optional[str] = None,
        compress: bool = True
    ) -> Dict[str, Any]:
        """
        Embed metadata into an existing ONNX model.
        Preserves complete configurations without filtering.
        
        Args:
            onnx_path: Path to existing ONNX model (from any exporter)
            model_name: HuggingFace model name or local path
            output_path: Output path (if None, overwrites input)
            compress: Whether to compress large metadata
            
        Returns:
            Embedded metadata dictionary with complete configs
        """
        logger.info(f"Processing ONNX model: {onnx_path}")
        logger.info(f"Loading HuggingFace model: {model_name}")
        
        # Verify ONNX file exists
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
        
        try:
            # Load HuggingFace model and processor for metadata discovery
            from transformers import AutoModel, AutoTokenizer, AutoProcessor
            
            # Load model
            model = AutoModel.from_pretrained(model_name)
            logger.info(f"Loaded model: {model.__class__.__name__}")
            
            # Try to load processor
            processor = None
            try:
                processor = AutoProcessor.from_pretrained(model_name)
                logger.info(f"Loaded processor: {processor.__class__.__name__}")
            except:
                try:
                    processor = AutoTokenizer.from_pretrained(model_name)
                    logger.info(f"Loaded tokenizer: {processor.__class__.__name__}")
                except:
                    logger.warning("No processor/tokenizer found")
            
            # Discover and aggregate complete metadata
            logger.info("Discovering complete metadata (no filtering)...")
            metadata = self.discovery.discover_from_hf(model, processor)
            
            # Log metadata summary
            self._log_metadata_summary(metadata)
            
            # Validate metadata
            logger.info("Validating metadata...")
            self.validator.validate(metadata)
            
            # Embed into ONNX
            output = output_path or onnx_path
            logger.info(f"Embedding metadata into ONNX...")
            
            if compress:
                self.embedder.embed_with_compression(onnx_path, metadata, output)
            else:
                self.embedder.embed(onnx_path, metadata, output)
            
            # Calculate size overhead
            original_size = Path(onnx_path).stat().st_size
            new_size = Path(output).stat().st_size
            overhead_kb = (new_size - original_size) / 1024
            
            logger.info(f"✅ Successfully embedded metadata")
            logger.info(f"   Output: {output}")
            logger.info(f"   Metadata overhead: {overhead_kb:.2f} KB")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to embed metadata: {e}")
            raise
    
    def read_metadata(self, onnx_path: str) -> Dict[str, Any]:
        """
        Read metadata from an ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            Metadata dictionary
        """
        logger.info(f"Reading metadata from: {onnx_path}")
        
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
        
        metadata = self.reader.read(onnx_path)
        
        if not metadata:
            logger.warning("No metadata found in ONNX model")
            return {}
        
        logger.info(f"Found {len(metadata)} metadata entries")
        return metadata
    
    def validate_metadata(self, onnx_path: str) -> bool:
        """
        Validate metadata in an ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            True if valid
        """
        logger.info(f"Validating metadata in: {onnx_path}")
        
        metadata = self.read_metadata(onnx_path)
        
        if not metadata:
            logger.error("No metadata to validate")
            return False
        
        try:
            self.validator.validate(metadata)
            logger.info("✅ Metadata is valid")
            return True
        except Exception as e:
            logger.error(f"❌ Metadata validation failed: {e}")
            return False
    
    def _log_metadata_summary(self, metadata: Dict[str, Any]):
        """Log a summary of discovered metadata."""
        logger.info("Metadata summary:")
        
        # Task type
        if 'hf_pipeline_task' in metadata:
            logger.info(f"  Task: {metadata['hf_pipeline_task']}")
        
        # Model config
        if 'hf_model_config' in metadata:
            config = metadata['hf_model_config']
            if isinstance(config, dict):
                logger.info(f"  Model type: {config.get('model_type', 'unknown')}")
                if 'num_labels' in config:
                    logger.info(f"  Num labels: {config['num_labels']}")
        
        # Feature engineering
        if 'feature_engineering.type' in metadata:
            logger.info(f"  Feature engineering: {metadata['feature_engineering.type']}")
        
        # Size estimate
        metadata_json = json.dumps(metadata, separators=(',', ':'))
        size_kb = len(metadata_json) / 1024
        logger.info(f"  Metadata size: {size_kb:.2f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Metadata Tool for ONNX Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Embed metadata into existing ONNX model
  %(prog)s embed model.onnx bert-base-uncased
  %(prog)s embed model.onnx prajjwal1/bert-tiny
  %(prog)s embed optimum_export.onnx gpt2 --output gpt2_self_contained.onnx
  
  # Read metadata from ONNX model
  %(prog)s read model.onnx
  %(prog)s read model.onnx --json
  
  # Validate metadata in ONNX model
  %(prog)s validate model.onnx
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Embed metadata into ONNX model')
    embed_parser.add_argument('onnx_path', help='Path to ONNX model')
    embed_parser.add_argument('model_name', help='HuggingFace model name or path')
    embed_parser.add_argument('--output', '-o', help='Output path (default: overwrite)')
    embed_parser.add_argument('--no-compress', action='store_true',
                            help='Disable compression')
    
    # Read command
    read_parser = subparsers.add_parser('read', help='Read metadata from ONNX model')
    read_parser.add_argument('onnx_path', help='Path to ONNX model')
    read_parser.add_argument('--json', action='store_true',
                            help='Output as JSON')
    read_parser.add_argument('--key', help='Read specific metadata key')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate metadata in ONNX model')
    validate_parser.add_argument('onnx_path', help='Path to ONNX model')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Create tool instance
    tool = StandaloneMetadataTool()
    
    try:
        if args.command == 'embed':
            metadata = tool.embed_metadata(
                onnx_path=args.onnx_path,
                model_name=args.model_name,
                output_path=args.output,
                compress=not args.no_compress
            )
            return 0
            
        elif args.command == 'read':
            metadata = tool.read_metadata(args.onnx_path)
            
            if args.key:
                # Read specific key
                if args.key in metadata:
                    value = metadata[args.key]
                    if args.json and isinstance(value, (dict, list)):
                        print(json.dumps(value, indent=2))
                    else:
                        print(value)
                else:
                    logger.error(f"Key '{args.key}' not found")
                    return 1
            else:
                # Read all metadata
                if args.json:
                    print(json.dumps(metadata, indent=2))
                else:
                    for key, value in metadata.items():
                        if isinstance(value, (dict, list)):
                            print(f"{key}: {json.dumps(value, separators=(',', ':'))[:100]}...")
                        else:
                            print(f"{key}: {value}")
            return 0
            
        elif args.command == 'validate':
            valid = tool.validate_metadata(args.onnx_path)
            return 0 if valid else 1
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())