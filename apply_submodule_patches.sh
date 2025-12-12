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
    echo "⚠ finetune-hf-vits already has local changes. Skipping patch application."
    echo "  If you need to reapply the patch, reset the submodule first with:"
    echo "  cd finetune-hf-vits && git reset --hard HEAD && cd .."
fi

cd ..

# Check if the StableTTS submodule exists
if [ ! -d "StableTTS" ]; then
    echo "Error: StableTTS submodule directory not found."
    echo "Please run 'git submodule update --init --recursive' first."
    exit 1
fi

# Apply the patch to make Japanese support optional
echo "Applying patch to StableTTS..."
cd StableTTS

# Check if patch has already been applied
if git diff --quiet HEAD; then
    echo "Applying StableTTS.patch..."
    git apply ../StableTTS.patch
    echo "✓ Patch applied successfully to StableTTS"
else
    echo "⚠ StableTTS already has local changes. Skipping patch application."
    echo "  If you need to reapply the patch, reset the submodule first with:"
    echo "  cd StableTTS && git reset --hard HEAD && cd .."
fi

cd ..

echo ""
echo "All patches applied successfully!"
echo ""
echo "Note: These patches fix compatibility issues:"
echo "  - finetune-hf-vits: Removes deprecated send_example_telemetry from transformers"
echo "  - finetune-hf-vits: Fixes matplotlib canvas.tostring_rgb() -> canvas.buffer_rgba()"
echo "  - StableTTS: Makes Japanese (pyopenjtalk) support optional for English/Chinese-only training"