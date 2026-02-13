#!/bin/bash
# ===========================================================================
# run_worker_loop.sh — Self-updating worker loop
# ===========================================================================
#
# This script runs worker_entry.py, and when the worker exits with code 22
# (EXIT_CODE_UPDATE_RESTART), it does a git pull to pick up code changes
# and then restarts the worker.  Any other exit code stops the loop.
#
# All arguments are forwarded to worker_entry.py.  The --enable-return-update
# flag is automatically added so the Python process exits after each job
# (or idle sleep) instead of looping internally.
#
# IMPORTANT — Why this script is structured the way it is:
#
#   Bash reads scripts incrementally, not all at once.  If `git pull`
#   modifies THIS VERY FILE while bash is in the middle of executing it,
#   the results are undefined — bash might skip lines, re-execute lines,
#   or read partial commands.
#
#   We defend against this with TWO complementary techniques:
#
#   1. FUNCTION WRAPPING — The entire script body is inside main().
#      Bash parses the full function definition into memory before
#      executing any of it.  Once main() is running, changes to the
#      file on disk don't affect the in-memory copy.
#
#   2. exec SELF-RESTART — Instead of using a `while` loop (which would
#      keep running the OLD in-memory code forever), we use
#      `exec bash "$0" "$@"` after git pull.  `exec` replaces the
#      current process with a fresh read of the (possibly updated)
#      script from disk.  This is the Unix execve(2) syscall — it does
#      NOT create a new stack frame or child process, so there is ZERO
#      risk of stack overflow no matter how many iterations we go
#      through.  The PID stays the same; the old process image is
#      simply replaced.
#
#   Together these mean:
#     - The script is safe even if git pull changes it mid-execution
#     - Each restart picks up the latest version of this script
#     - No stack growth, no resource leaks, runs indefinitely
#
# ===========================================================================

main() {
    export PATH="$HOME/.local/bin:$PATH"
    cd "$(dirname "$0")"

    EXIT_CODE_UPDATE_RESTART=22

    echo "=== Starting worker ==="

    # Run worker with --enable-return-update so it exits back to us.
    # We disable `set -e` around this call so we can capture the exit
    # code without the script dying on a non-zero return.
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

    # Re-execute ourselves with the (possibly updated) version of this
    # script.  exec replaces the current process — no stack growth, no
    # child process, no return.  See the header comment for details.
    echo "=== Restarting with updated code ==="
    exec bash "$0" "$@"
}

# Invoke main with all original arguments.  By the time bash reaches
# this line, the entire main() function above has been parsed into
# memory, so even if git pull changed this file on disk, we're safe.
main "$@"
