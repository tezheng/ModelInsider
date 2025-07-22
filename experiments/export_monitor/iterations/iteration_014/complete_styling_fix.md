# Complete Text Styling Fix

## Implementation Strategy

1. **Create styled text builder methods**:
   ```python
   def _style_number(self, num: Any) -> Text:
       return Text(str(num), style="bold cyan")
   
   def _style_bold(self, text: str) -> Text:
       return Text(text, style="bold")
   ```

2. **Fix all numeric outputs**:
   - Step numbers: 1/8 -> styled
   - Module counts: 48 -> styled
   - Parameter counts: 4.4M -> styled
   - Tensor dimensions: [2, 16] -> styled

3. **Fix special formatting**:
   - Strategy line: HTP (Hierarchy-Preserving)
   - Parentheses: always bold
   - Tensor shapes: bold brackets

4. **Test thoroughly**:
   - Capture console output
   - Verify ANSI codes present
   - Compare with baseline
