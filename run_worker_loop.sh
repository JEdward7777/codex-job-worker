#!/bin/bash
# Wrapper script that runs worker_entry.py in a loop.
# After each run, if the worker exits with code 22 (EXIT_CODE_UPDATE_RESTART),
# the script does a git pull + setup and restarts the worker.
# Any other exit code causes this script to exit with that same code.
#
# Usage:
#   bash run_worker_loop.sh [worker_entry.py arguments...]
#
# All arguments are forwarded to worker_entry.py.  The --enable-return-update
# flag is automatically added so the Python process exits after each job
# (or idle sleep) instead of looping internally.

set -e
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"

EXIT_CODE_UPDATE_RESTART=22

while true; do
    echo "=== Starting worker ==="
    # Run worker with --enable-return-update so it exits back to us.
    # +e so we can capture the exit code without the script dying.
    set +e
    uv run python worker_entry.py --enable-return-update "$@"
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -ne $EXIT_CODE_UPDATE_RESTART ]; then
        echo "Worker exited with code $EXIT_CODE. Done."
        exit $EXIT_CODE
    fi

    echo "Worker requested restart (exit $EXIT_CODE_UPDATE_RESTART)."
    echo "=== Updating worker code ==="
    git pull origin main || true
    git submodule update --init --recursive
    bash setup_worker.sh
done
