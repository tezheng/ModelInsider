#!/usr/bin/env python3
"""
Capture baseline outputs from original HTP exporter implementation.
This script captures console output WITH Rich formatting/colors preserved.
"""

import json
import subprocess
from pathlib import Path

from rich.console import Console


def capture_original_outputs():
    """Capture outputs from the original HTP implementation."""
    
    # Create baseline directory
    baseline_dir = Path("temp/baseline")
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Restoring original HTP exporter implementation...")
    
    # First, stash current changes
    subprocess.run(["git", "stash", "--include-untracked"], check=True)
    
    try:
        # Checkout the version before export monitor was added
        # Looking at the commits, we want a version before the recent changes
        subprocess.run(["git", "checkout", "2abd924"], check=True)
        
        print("✅ Original implementation restored")
        print("🎯 Running export to capture baseline...")
        
        # Run the export and capture output
        cmd = [
            "uv", "run", "modelexport", "export",
            "--model", "prajjwal1/bert-tiny",
            "--output", str(baseline_dir / "model.onnx"),
            "--strategy", "htp",
            "--verbose",
            "--with-report"
        ]
        
        # Capture console output with colors
        console = Console(record=True, width=120, force_terminal=True)
        
        # Run the command and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Capture output line by line to preserve formatting
        full_output = []
        for line in process.stdout:
            full_output.append(line.rstrip())
            console.print(line.rstrip(), style="")
        
        process.wait()
        
        if process.returncode == 0:
            print("\n✅ Export completed successfully!")
            
            # Save console output with ANSI codes
            console_with_colors = console.export_text(styles=True)
            with open(baseline_dir / "console_output_with_colors.txt", "w") as f:
                f.write(console_with_colors)
            
            # Save console output without colors (plain text)
            console_plain = console.export_text(styles=False)
            with open(baseline_dir / "console_output_plain.txt", "w") as f:
                f.write(console_plain)
            
            # Save raw output
            with open(baseline_dir / "console_output_raw.txt", "w") as f:
                f.write("\n".join(full_output))
            
            # Save console as HTML (best for preserving colors)
            console_html = console.export_html()
            with open(baseline_dir / "console_output.html", "w") as f:
                f.write(console_html)
            
            # Copy other outputs
            print("\n📁 Copying output files...")
            
            # Check what files were created
            onnx_file = baseline_dir / "model.onnx"
            metadata_file = baseline_dir / "model_htp_metadata.json"
            report_file = baseline_dir / "model_htp_export_report.txt"
            
            if metadata_file.exists():
                print(f"✅ Metadata found: {metadata_file}")
                # Pretty print the metadata
                with open(metadata_file) as f:
                    metadata = json.load(f)
                with open(baseline_dir / "metadata_pretty.json", "w") as f:
                    json.dump(metadata, f, indent=2)
            else:
                print("❌ Metadata file not found")
            
            if report_file.exists():
                print(f"✅ Report found: {report_file}")
            else:
                print("❌ Report file not found")
            
            print("\n📊 Baseline files saved to:", baseline_dir)
            print("  - console_output_with_colors.txt (ANSI codes)")
            print("  - console_output_plain.txt (no colors)")
            print("  - console_output_raw.txt (raw output)")
            print("  - console_output.html (best for viewing colors)")
            print("  - model_htp_metadata.json")
            print("  - model_htp_export_report.txt")
            
        else:
            print(f"\n❌ Export failed with code {process.returncode}")
            
    finally:
        print("\n🔄 Restoring current implementation...")
        subprocess.run(["git", "checkout", "-"], check=True)
        subprocess.run(["git", "stash", "pop"], check=True)
        print("✅ Current implementation restored")

if __name__ == "__main__":
    capture_original_outputs()