#!/bin/bash
# Wrapper: one array task per line in the command file.
# Usage (from project root): bash slurm/submit_array_job.sh slurm/commands.txt
# Adapted from ~/projects/standby_run_array.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_command_file>"
    exit 1
fi

COMMAND_FILE="$1"

if [ ! -f "$COMMAND_FILE" ]; then
    echo "Error: Command file not found at '$COMMAND_FILE'."
    exit 1
fi

NUM_COMMANDS=$(wc -l < "$COMMAND_FILE" | tr -d ' ')

if [ "$NUM_COMMANDS" -eq 0 ]; then
    echo "Error: Command file is empty."
    exit 1
fi

JOB_NAME=$(basename "$COMMAND_FILE" .txt)

echo "Submitting array job for $COMMAND_FILE with $NUM_COMMANDS tasks."

mkdir -p "$PROJECT_ROOT/slurm_output"

if [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ -f "$HOME/.hf_token" ]; then
    export HUGGING_FACE_HUB_TOKEN=$(cat "$HOME/.hf_token")
fi

sbatch \
    --job-name="${JOB_NAME}" \
    --output="$PROJECT_ROOT/slurm_output/${JOB_NAME}_%A_%a.out" \
    --error="$PROJECT_ROOT/slurm_output/${JOB_NAME}_%A_%a.err" \
    --array=1-"${NUM_COMMANDS}" \
    --export=ALL,COMMAND_FILE="$COMMAND_FILE" \
    "$SCRIPT_DIR/array_job_template.sh"
