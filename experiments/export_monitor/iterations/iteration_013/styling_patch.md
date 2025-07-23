# Patch for export_monitor.py to fix text styling

## Key Changes Required:

1. **Console initialization**:
   ```python
   # OLD:
   self.console = console or Console(width=80)
   
   # NEW:
   self.console = console or Console(width=80, force_terminal=True)
   ```

2. **Replace all print() with self.console.print()**:
   - Every `print(...)` becomes `self.console.print(...)`
   
3. **Add styled text for numbers and special formatting**:
   - Use Text objects for complex styling
   - Numbers in bold cyan: `style="bold cyan"`
   - Parentheses in bold: `style="bold"`

4. **Example fix for model loaded message**:
   ```python
   # OLD:
   print(f"✅ Model loaded: {data.model_class} ({data.total_modules} modules, {data.total_parameters/1e6:.1f}M parameters)")
   
   # NEW:
   text = Text("✅ Model loaded: ")
   text.append(data.model_class)
   text.append(" ")
   text.append("(", style="bold")
   text.append(str(data.total_modules), style="bold cyan")
   text.append(" modules, ", style="bold")
   text.append(f"{data.total_parameters/1e6:.1f}", style="bold cyan")
   text.append("M parameters", style="bold")
   text.append(")", style="bold")
   self.console.print(text)
   ```

5. **Strategy line formatting**:
   ```python
   # Match baseline: HTP [1m([0mHierarchy-Preserving[1m)[0m
   strategy_text = Text("⚙️ Strategy: HTP ")
   strategy_text.append("(", style="bold")
   strategy_text.append("Hierarchy-Preserving", style="normal")
   strategy_text.append(")", style="bold")
   self.console.print(strategy_text)
   ```

## Files to update:
- modelexport/strategies/htp/export_monitor.py

## Testing:
After applying changes, the console output should contain ANSI escape codes like:
- [1;36m for bold cyan numbers
- [1m and [0m for bold text
