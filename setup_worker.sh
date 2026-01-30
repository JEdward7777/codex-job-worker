#!/bin/bash
# Setup script for codex-job-worker
# This script clones the repository, installs dependencies, and applies patches.
# Designed to be run in a SkyPilot setup phase or manually on a fresh machine.

set -e  # Exit on error

echo "=========================================="
echo "Setting up codex-job-worker"
echo "=========================================="

# Configuration
REPO_URL="${REPO_URL:-https://github.com/JEdward7777/codex-job-worker.git}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/codex-job-worker}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# Check if already installed
if [ -d "$INSTALL_DIR" ] && [ -f "$INSTALL_DIR/worker_entry.py" ]; then
    echo "Worker already installed at $INSTALL_DIR"
    echo "Updating repository..."
    cd "$INSTALL_DIR"
    git pull origin main || true
    git submodule update --init --recursive
else
    echo "Cloning repository..."
    git clone --recurse-submodules "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
fi

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Install Python dependencies
echo "Installing Python dependencies..."
cd "$INSTALL_DIR"
uv sync

# Apply patches to submodules
echo "Applying patches to submodules..."
./apply_submodule_patches.sh

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To run the worker:"
echo "  cd $INSTALL_DIR"
echo "  uv run python worker_entry.py --token YOUR_TOKEN --worker-id YOUR_WORKER_ID"
echo ""
echo "Or with environment variables:"
echo "  export GITLAB_TOKEN=your_token"
echo "  export WORKER_ID=your_worker_id"
echo "  uv run python worker_entry.py"
