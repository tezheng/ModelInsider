#!/usr/bin/env python3
"""
Runner script to test constant handling fix
"""

import sys
import subprocess
from pathlib import Path

def run_bert_test():
    """Try to run the BERT test with various Python environments"""
    
    test_script = "bert_self_attention_test.py"
    
    # Try different python commands
    python_commands = [
        "python3",
        "python", 
        "/usr/bin/python3",
        "uv run python3",
        "uv run python"
    ]
    
    for cmd in python_commands:
        print(f"Trying: {cmd} {test_script}")
        try:
            result = subprocess.run(
                f"{cmd} {test_script}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Test completed successfully!")
                print("STDOUT:")
                print(result.stdout)
                return True
            else:
                print(f"❌ Failed with return code {result.returncode}")
                if "ModuleNotFoundError" in result.stderr:
                    print(f"Missing modules: {result.stderr}")
                    continue
                else:
                    print("STDERR:")
                    print(result.stderr)
                    return False
                    
        except subprocess.TimeoutExpired:
            print("⏱️  Test timed out")
            return False
        except Exception as e:
            print(f"❌ Error running command: {e}")
            continue
    
    print("❌ Could not run test with any Python environment")
    return False

if __name__ == "__main__":
    success = run_bert_test()
    sys.exit(0 if success else 1)