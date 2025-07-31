#!/bin/bash
# Hook: post-edit-python-ruff
# Purpose: Automatically run ruff linting after editing Python files
# Trigger: After Edit or MultiEdit tools modify .py files

# Check if the edited file is a Python file
if [[ "$CLAUDE_EDITED_FILE" == *.py ]]; then
    echo "🔍 Running ruff on $CLAUDE_EDITED_FILE..."
    
    # Run ruff check
    if command -v ruff &> /dev/null; then
        ruff check "$CLAUDE_EDITED_FILE"
        RUFF_EXIT_CODE=$?
        
        if [ $RUFF_EXIT_CODE -eq 0 ]; then
            echo "✅ Ruff check passed!"
        else
            echo "⚠️  Ruff found issues. Consider running: ruff check --fix $CLAUDE_EDITED_FILE"
        fi
    else
        # If ruff is not in PATH, try with uv run
        if command -v uv &> /dev/null; then
            uv run ruff check "$CLAUDE_EDITED_FILE"
            RUFF_EXIT_CODE=$?
            
            if [ $RUFF_EXIT_CODE -eq 0 ]; then
                echo "✅ Ruff check passed!"
            else
                echo "⚠️  Ruff found issues. Consider running: uv run ruff check --fix $CLAUDE_EDITED_FILE"
            fi
        else
            echo "❌ Ruff not found. Please install ruff or ensure uv is available."
        fi
    fi
fi