"""Test Rich console behavior with paths"""
from rich.console import Console
from io import StringIO

# Test 1: Simple format
buffer1 = StringIO()
console1 = Console(file=buffer1, force_terminal=True)
console1.print("  1. /BertModel/BertEmbeddings: 3 nodes")
output1 = buffer1.getvalue()

# Test 2: With bold cyan
buffer2 = StringIO()
console2 = Console(file=buffer2, force_terminal=True)
console2.print("  1. /BertModel/BertEmbeddings: [bold cyan]3[/bold cyan] nodes")
output2 = buffer2.getvalue()

# Test 3: Check if Rich is auto-highlighting paths
buffer3 = StringIO()
console3 = Console(file=buffer3, force_terminal=True, highlight=False)  # Disable highlighting
console3.print("  1. /BertModel/BertEmbeddings: [bold cyan]3[/bold cyan] nodes")
output3 = buffer3.getvalue()

print("Test 1 - Simple format:")
print(repr(output1))
print("\nTest 2 - With bold cyan:")
print(repr(output2))
print("\nTest 3 - With highlighting disabled:")
print(repr(output3))

# Check for magenta
print("\nContains magenta in test 1:", '\x1b[35m' in output1)
print("Contains magenta in test 2:", '\x1b[35m' in output2)
print("Contains magenta in test 3:", '\x1b[35m' in output3)