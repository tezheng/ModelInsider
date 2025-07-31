"""
Optimized Export Monitor - Iteration 18
Performance improvements and memory optimizations.
"""

import json
from functools import lru_cache
from typing import Any


class OptimizedHTPExportMonitor:
    """Performance-optimized export monitor."""
    
    def __init__(self):
        # Pre-allocate buffers
        self._console_buffer = []
        self._tree_cache = {}
        
    @lru_cache(maxsize=256)
    def _style_number(self, num: Any) -> str:
        """Cached number styling."""
        return f"[bold cyan]{num}[/bold cyan]"
    
    @lru_cache(maxsize=128)
    def _style_bold(self, text: str) -> str:
        """Cached bold styling."""
        return f"[bold]{text}[/bold]"
    
    def _build_tree_optimized(self, hierarchy: dict[str, Any], max_depth: int = 10) -> list[str]:
        """Build tree with depth limit and caching."""
        if not hierarchy:
            return ["  (empty)"]
        
        # Check cache
        cache_key = tuple(sorted(hierarchy.keys()))
        if cache_key in self._tree_cache:
            return self._tree_cache[cache_key]
        
        lines = []
        
        def _add_node(path: str, depth: int = 0):
            if depth >= max_depth:
                lines.append(f"{' ' * (depth * 2)}... (truncated)")
                return
            
            # Process node
            indent = " " * (depth * 2)
            lines.append(f"{indent}└─ {path}")
            
            # Find children efficiently
            prefix = f"{path}."
            children = [p for p in hierarchy if p.startswith(prefix) and p.count('.') == path.count('.') + 1]
            
            for child in children[:20]:  # Limit children display
                _add_node(child, depth + 1)
        
        # Add root nodes
        roots = [p for p in hierarchy if '.' not in p]
        for root in roots[:30]:  # Limit root display
            _add_node(root)
        
        # Cache result
        self._tree_cache[cache_key] = lines
        return lines
    
    def write_json_streaming(self, data: dict[str, Any], filepath: Path) -> None:
        """Stream JSON writing for large data."""
        with open(filepath, 'w') as f:
            f.write('{\n')
            
            items = list(data.items())
            for i, (key, value) in enumerate(items):
                if i > 0:
                    f.write(',\n')
                
                # Stream write based on value type
                if isinstance(value, list | dict) and len(str(value)) > 10000:
                    # Large value - stream write
                    f.write(f'  "{key}": ')
                    json.dump(value, f, separators=(',', ':'))
                else:
                    # Small value - normal write
                    f.write(f'  "{key}": {json.dumps(value, separators=(",", ":"))}')
            
            f.write('\n}')
