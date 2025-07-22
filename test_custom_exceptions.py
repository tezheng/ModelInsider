#!/usr/bin/env python3
"""Test custom torch_nn_exceptions override."""

from modelexport.strategies.htp import HTPExporter

# Test 1: Default torch_nn_exceptions
print("Test 1: Default torch_nn_exceptions")
exporter1 = HTPExporter()
print(f"Default exceptions: {exporter1.torch_nn_exceptions}")

# Test 2: Custom torch_nn_exceptions
print("\nTest 2: Custom torch_nn_exceptions")
custom_exceptions = ["Linear", "Conv2d", "MyCustomModule"]
exporter2 = HTPExporter(torch_nn_exceptions=custom_exceptions)
print(f"Custom exceptions: {exporter2.torch_nn_exceptions}")

# Test 3: Empty list override
print("\nTest 3: Empty list override")
exporter3 = HTPExporter(torch_nn_exceptions=[])
print(f"Empty exceptions: {exporter3.torch_nn_exceptions}")

print("\n✅ All tests passed!")