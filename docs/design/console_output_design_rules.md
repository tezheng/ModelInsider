# Console Output Design Rules

This document captures the design rules for console output in the HTP Export Monitor system, based on implementation requirements and user feedback.

## Core Principles

1. **Rich Library Usage**: Use Rich library exclusively for all console output - no exceptions
2. **Consistent Styling**: Apply consistent color and text styling across all output
3. **Readability**: Ensure output is clear and well-structured with proper hierarchy

## Color Scheme

### Primary Colors
- **Cyan (`bold cyan`)**: Numbers, counts, percentages, and quantitative values
- **Green (`bold green`)**: Success states, enabled features, and configuration values
- **Red (`bold red`)**: Error states, disabled features, and warnings
- **Bold**: Class names, module names, and important labels
- **Dim/Gray (`dim`)**: Secondary information like paths and descriptions
- **Bold Magenta (`bold magenta`)**: Highlighted context after colons (file paths, model names)

### Specific Usage

#### Configuration Values
All configuration values should be displayed in **green** to indicate they are active settings:
```python
# Correct
self.console.print(f"   • Opset version: [bold green]{opset}[/bold green]")
self.console.print(f"   • Model type: [bold green]{model_type}[/bold green]")
self.console.print(f"   • Detected task: [bold green]{task}[/bold green]")

# Timestamp values
self.console.print(f"📅 Export Time: [bold green]{timestamp}[/bold green]")

# Input specifications
self.console.print(f"   • {name}: shape=[bold green]{shape}[/bold green], dtype=[bold green]{dtype}[/bold green]")

# Also correct for boolean values
self.console.print(f"   • Constant folding: [bold green]True[/bold green]")
```

#### Numeric Values
All numeric values (counts, percentages, sizes) should be displayed in **cyan** including units/symbols:
```python
# Correct - number and unit/symbol both in cyan
self.console.print(f"Coverage: {self._bright_cyan(f'{coverage:.1f}%')}")
self.console.print(f"Tagged nodes: {self._bright_cyan(str(tagged))}/{self._bright_cyan(str(total))}")
self.console.print(f"Total time: {self._bright_cyan(f'{time:.2f}s')}")
self.console.print(f"Model size: {self._bright_cyan(f'{size:.2f}MB')}")
self.console.print(f"Parameters: {self._bright_cyan(f'{params}M')}")
```

**Best Practice**: Keep number and unit together as a single highlighted unit. Use utility functions for consistency.

#### Context Values (After Colons)
Model names, file paths, and other contextual information after colons should be displayed in **bold magenta**:
```python
# Correct
self.console.print(f"🔄 Loading model and exporting: {self._bright_magenta(model_name)}")
self.console.print(f"🎯 Export target: {self._bright_magenta(output_path)}")
self.console.print(f"   • ONNX model: {self._bright_magenta(file_path)}")
```

#### Hierarchy Trees
In hierarchy displays, follow these rules:
1. **Class names**: Bold (shown first)
2. **Paths**: Dim/gray (shown after colon)
3. **Node counts**: Cyan
4. **Operation counts**: Cyan
5. **ONNX operation names**: Bold (show simple name only)

Example:
```python
# Module hierarchy
node_text = Text()
node_text.append(f"{class_name}", style="bold")        # BertAttention
node_text.append(": ", style="")                       # :
node_text.append(child_path, style="dim")              # encoder.layer.0.attention
node_text.append(" (", style="")                       # (
node_text.append(str(count), style="bold cyan")        # 39
node_text.append(" nodes)", style="")                  # nodes)

# ONNX operations (show only actual nodes, not paths)
# Multiple operations of same type:
op_text.append("Add", style="bold")                    # Add
op_text.append(" (", style="")                         # (
op_text.append("2", style="bold cyan")                 # 2
op_text.append(" ops)", style="")                      # ops)

# Single operation (show simple name):
op_text.append("LayerNormalization_0", style="bold")   # LayerNormalization_0
op_text.append(": ", style="")                         # :
op_text.append("/embeddings/LayerNorm/LayerNormalization_0", style="dim")  # Full path
```

## Utility Functions

### Style Helpers
To maintain consistency and avoid hardcoded Rich markup, use these utility functions:
```python
@staticmethod
def _bright_cyan(text: str) -> str:
    """Format text in bright cyan."""
    return f"[bold cyan]{text}[/bold cyan]"

@staticmethod
def _bright_green(text: str) -> str:
    """Format text in bright green."""
    return f"[bold green]{text}[/bold green]"

@staticmethod
def _bright_red(text: str) -> str:
    """Format text in bright red."""
    return f"[bold red]{text}[/bold red]"

@staticmethod
def _bright_magenta(text: str) -> str:
    """Format text in bright magenta."""
    return f"[bold magenta]{text}[/bold magenta]"

@staticmethod
def _bright_yellow(text: str) -> str:
    """Format text in bright yellow."""
    return f"[bold yellow]{text}[/bold yellow]"

@staticmethod
def _dim(text: str) -> str:
    """Format text in dim style."""
    return f"[dim]{text}[/dim]"

@staticmethod
def _bold(text: str) -> str:
    """Format text in bold."""
    return f"[bold]{text}[/bold]"
```

Use these functions instead of hardcoding Rich markup for better maintainability.

## Console Configuration

### Required Settings
```python
console = Console(
    width=120,                    # Wide enough for trees
    force_terminal=True,          # Force ANSI colors
    legacy_windows=False,         # Modern terminal support
    highlight=False               # CRITICAL: Disable auto-highlighting
)
```

**Important**: Always set `highlight=False` to prevent Rich from automatically applying syntax highlighting to paths and other content.

## Text Formatting Rules

### Headers and Steps
- Step numbers: `STEP {n}/6:` format
- Step titles: Bold and uppercase
- Icons: Use appropriate emoji for each step

### Lists and Bullets
- Use `•` for bullet points
- Indent with 3 spaces for sub-items
- Apply appropriate styling to values

### Success/Status Indicators
- ✅ for success/completion
- ⚠️ for warnings
- ❌ for errors
- 📊 for statistics
- 🌳 for trees/hierarchies

## Hierarchy Display Rules

### Module Hierarchy
Show the module structure with proper indentation and tree characters:
```
BertModel (136 nodes)
├── BertEmbeddings: embeddings (8 nodes)
│   ├── Add (2 ops)
│   ├── Constant (2 ops)
│   ├── LayerNormalization_0: /embeddings/LayerNorm/LayerNormalization_0
│   ├── Gather_0: /embeddings/position_embeddings/Gather_0
│   ├── Gather_1: /embeddings/token_type_embeddings/Gather_1
│   └── Gather_2: /embeddings/word_embeddings/Gather_2
└── BertEncoder: encoder (106 nodes)
```

### Truncation Rules
Both Module Hierarchy and Complete HF Hierarchy with ONNX Nodes should be truncated to 30 lines maximum:
- If total lines ≤ 30: Display all lines
- If total lines > 30: Display first 30 lines and add truncation message
- Truncation message format: `... showing first 30 lines (truncated for console)`

**Important Rules for ONNX Operations**:
1. Show only actual ONNX operation nodes (not module paths)
2. Use simple names (just the operation name with suffix, e.g., `Add_0`, `LayerNormalization_0`)
3. For single operations, show `simple_name: full_path` format
4. For multiple operations of same type, show `OpType (N ops)` format
5. Operation names should be bold, full paths should be dim

### ONNX Operations
Group operations by type under their parent modules:
- Multiple operations: Show count in parentheses
- Single operation: Show the operation name
- Use tree structure to show containment

## Common Pitfalls to Avoid

1. **Don't use inline markup in tree nodes** - Create Text objects with proper styling
2. **Don't forget to disable highlighting** - Set `highlight=False` in Console
3. **Don't hardcode ANSI codes** - Use Rich's style system
4. **Don't mix styling approaches** - Be consistent throughout

## Output Files Section

The output files section should always show:
1. **ONNX model**: Always shown (the primary output)
2. **Metadata**: Always shown (JSON metadata is always written)
3. **Report**: Only shown when `--enable-report` flag is used

Example:
```
📁 Output files:
   • ONNX model: temp/bert.onnx
   • Metadata: temp/bert_htp_metadata.json
   • Report: temp/bert_htp_export_report.txt  # Only if --enable-report
```

**Important**: Metadata file is ALWAYS written regardless of report settings.

## Array Display Rules

When displaying arrays of string values (like input/output names), apply green styling to individual array items:
```python
# Format array items with green color
names = ['input_ids', 'attention_mask', 'token_type_ids']
formatted = "[" + ", ".join(f"[bold green]'{name}'[/bold green]" for name in names) + "]"
self.console.print(f"📥 Input names: {formatted}")
# Output: 📥 Input names: ['input_ids', 'attention_mask', 'token_type_ids']
#         where each name is displayed in green
```

### Empty Array Handling
When arrays are empty or not detected, display a warning with yellow styling:
```python
# When output names are empty or not available
self.console.print("⚠️  Output names: [bold yellow]Not detected[/bold yellow] (model may not have named outputs)")
```

This provides better visual separation and consistency with other green-styled configuration values, while also clearly indicating when expected data is missing.

## Testing Guidelines

When testing console output:
1. Capture output with a test console that has the same settings
2. Check for presence of ANSI codes for specific styles
3. Test both content and styling separately
4. Use `repr()` to inspect raw output including escape codes

Example test patterns:
```python
# Check for dim style
has_dim = "\\x1b[2m" in repr(output)

# Check for cyan
has_cyan = "\\x1b[36m" in repr(output) or "\\x1b[1;36m" in repr(output)

# Check for green
has_green = "\\x1b[32m" in repr(output) or "\\x1b[1;32m" in repr(output)
```