#!/bin/bash
# Usage: bash slurm/launch_tournament.sh --rounds 500
#    or: bash slurm/launch_tournament.sh 500

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

ROUNDS=500
if [ "${1:-}" = "--rounds" ] && [ -n "${2:-}" ]; then
    ROUNDS="$2"
elif [ -n "${1:-}" ]; then
    ROUNDS="$1"
fi

python3 "$PROJECT_ROOT/slurm/generate_commands.py" --rounds "$ROUNDS"
bash "$PROJECT_ROOT/slurm/submit_array_job.sh" "$PROJECT_ROOT/slurm/commands.txt"
