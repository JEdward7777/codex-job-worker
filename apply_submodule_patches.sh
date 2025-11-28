#!/bin/bash

# Script to apply local patches to submodules after cloning
# This is necessary when we need to fix issues in upstream submodules that we don't control

set -e  # Exit on error

echo "Applying patches to submodules..."

# Check if the finetune-hf-vits submodule exists
if [ ! -d "finetune-hf-vits" ]; then
    echo "Error: finetune-hf-vits submodule directory not found."
    echo "Please run 'git submodule update --init --recursive' first."
    exit 1
fi

# Apply the patch to fix outdated send_example_telemetry import
echo "Applying patch to finetune-hf-vits..."
cd finetune-hf-vits

# Check if patch has already been applied
if git diff --quiet HEAD; then
    echo "Applying finetune-hf-vits.patch..."
    git apply ../finetune-hf-vits.patch
    echo "✓ Patch applied successfully to finetune-hf-vits"
else
    echo "⚠ Submodule already has local changes. Skipping patch application."
    echo "  If you need to reapply the patch, reset the submodule first with:"
    echo "  git submodule update --init --force"
fi

cd ..

echo ""
echo "All patches applied successfully!"
echo ""
echo "Note: These patches fix compatibility issues with newer library versions:"
echo "  - Removes deprecated send_example_telemetry from transformers"
echo "  - Fixes matplotlib canvas.tostring_rgb() -> canvas.buffer_rgba()"