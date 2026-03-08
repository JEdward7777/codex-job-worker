#!/bin/bash
# ===========================================================================
# run_monitor_cron.sh — Cron-friendly monitor entry point with git pull
# ===========================================================================
#
# This script is designed to be called by cron every N minutes.
# Each invocation:
#   1. Pulls the latest code from git (so monitor updates are automatic)
#   2. Runs a single monitoring cycle (monitor.py run)
#   3. Exits — cron handles the scheduling
#
# If the script crashes, cron simply runs it again next cycle.
# No long-running daemon to babysit.
#
# IMPORTANT — Why this script is structured the way it is:
#
#   Bash reads scripts incrementally, not all at once.  If `git pull`
#   modifies THIS VERY FILE while bash is in the middle of executing it,
#   the results are undefined — bash might skip lines, re-execute lines,
#   or read partial commands.
#
#   We defend against this with FUNCTION WRAPPING — The entire script
#   body is inside main().  Bash parses the full function definition
#   into memory before executing any of it.  Once main() is running,
#   changes to the file on disk don't affect the in-memory copy.
#
#   This means the script is safe even if git pull changes it
#   mid-execution.
#
# Cron example (every 10 minutes):
#   */10 * * * * cd /path/to/launcher_project && bash run_monitor_cron.sh >> /var/log/monitor_cron.log 2>&1
#
# Or with a lock file and timeout to prevent overlapping/hung runs:
#   */10 * * * * flock -n /tmp/monitor.lock -c 'cd /path/to/launcher_project && timeout 7200 bash run_monitor_cron.sh' >> /var/log/monitor_cron.log 2>&1
#
# First-time setup on the monitor VM:
#   1. Clone the repo:
#      git clone --recurse-submodules https://github.com/JEdward7777/codex-job-worker.git
#      cd codex-job-worker/launcher_project
#
#   2. Install uv:
#      curl -LsSf https://astral.sh/uv/install.sh | sh
#
#   3. Install dependencies:
#      uv sync
#
#   4. Configure secrets:
#      cp .env.template .env
#      nano .env  # fill in GITLAB_TOKEN
#
#   5. Configure SkyPilot:
#      uv run sky check  # follow prompts to set up cloud credentials
#
#   6. Test with dry run:
#      bash run_monitor_cron.sh --dry-run
#
#   7. Add to cron:
#      crontab -e
#      # Add: */10 * * * * flock -n /tmp/monitor.lock -c 'cd /path/to/codex-job-worker/launcher_project && timeout 7200 bash run_monitor_cron.sh' >> /var/log/monitor_cron.log 2>&1
#
# ===========================================================================

main() {
    # Ensure we're in the script's directory (launcher_project/)
    cd "$(dirname "$0")"

    # --- Set up environment for cron ---
    # Cron runs with a minimal environment (just HOME, LOGNAME, PATH=/usr/bin:/bin).
    # We need the user's full environment for:
    #   - uv, sky, git on PATH
    #   - Cloud credential env vars (e.g., VAST_API_KEY)
    #   - SSH agent for SkyPilot VM access
    #   - Any other tools SkyPilot depends on (rsync, ssh, etc.)
    #
    # Source the user's profile to get the full interactive environment.
    # The -l flag on bash would do this, but we use explicit sourcing for clarity.
    if [ -f "$HOME/.bashrc" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.bashrc"
    elif [ -f "$HOME/.profile" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.profile"
    fi

    # Ensure uv and locally-installed tools are on PATH (may already be set
    # by .bashrc, but this guarantees it even if .bashrc is minimal)
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    # --- Git pull to pick up code updates ---
    # Go to the repo root (parent of launcher_project/)
    REPO_ROOT="$(cd .. && pwd)"

    echo "$(date -Iseconds) [INFO] Updating code from git..."
    (
        cd "$REPO_ROOT"
        git pull origin main 2>&1 || echo "$(date -Iseconds) [WARN] git pull failed (non-fatal)"
        git submodule update --init --recursive 2>&1 || true
    )

    # --- Sync dependencies (in case pyproject.toml changed) ---
    echo "$(date -Iseconds) [INFO] Syncing dependencies..."
    uv sync 2>&1 || echo "$(date -Iseconds) [WARN] uv sync failed (non-fatal)"

    # --- Run a single monitoring cycle ---
    echo "$(date -Iseconds) [INFO] Running monitor cycle..."
    uv run python monitor.py run "$@"
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "$(date -Iseconds) [ERROR] Monitor exited with code $EXIT_CODE"
    else
        echo "$(date -Iseconds) [INFO] Monitor cycle complete"
    fi

    exit $EXIT_CODE
}

# Invoke main with all original arguments.
# The function wrapping ensures the entire script is parsed into memory
# before execution, making it safe even if git pull modifies this file.
main "$@"
