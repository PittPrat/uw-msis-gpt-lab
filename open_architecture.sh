#!/bin/bash
# Quick script to open the architecture visualization

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HTML_FILE="$SCRIPT_DIR/instructions/architecture_visualization.html"

if [ -f "$HTML_FILE" ]; then
    echo "Opening architecture visualization..."
    open "$HTML_FILE"
    echo "✓ File opened in your default browser!"
else
    echo "❌ Error: File not found at $HTML_FILE"
    exit 1
fi


