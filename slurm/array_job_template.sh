#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#
# Adapted from ~/projects/array_job_template_standby.sh for rl_tournament (cd + project root).

# Initialize conda (required for non-interactive batch jobs)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
else
    echo "Error: conda not found. Install conda or edit array_job_template.sh with your conda path."
    exit 1
fi

# Switch to project root (parent of slurm/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Activate env with torch, transformers, etc. (override via RL_TOURNAMENT_CONDA_ENV)
CONDA_ENV="${RL_TOURNAMENT_CONDA_ENV:-base}"
conda activate "$CONDA_ENV"

# Fix GLIBCXX / PIL on some nodes
type module 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
for mod in gcc/11 gcc/12 gcc/13 gcc; do
    module load "$mod" 2>/dev/null && break
done
if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export MKL_THREADING_LAYER=GNU

if [ -z "$HUGGING_FACE_HUB_TOKEN" ] && [ -f "$HOME/.hf_token" ]; then
    export HUGGING_FACE_HUB_TOKEN=$(cat "$HOME/.hf_token")
fi

if [ -n "$SCRATCH" ] && [ -w "$SCRATCH" ]; then
    export HF_HOME="$SCRATCH/.cache/huggingface"
elif [ -d "/scratch/$USER" ] && [ -w "/scratch/$USER" ]; then
    export HF_HOME="/scratch/$USER/.cache/huggingface"
elif [ -n "$TMPDIR" ] && [ -w "$TMPDIR" ]; then
    export HF_HOME="$TMPDIR/hf_cache"
else
    export HF_HOME="$HOME/.cache/huggingface"
fi
export TRANSFORMERS_CACHE=$HF_HOME/hub
mkdir -p "$TRANSFORMERS_CACHE"

if [ -z "$COMMAND_FILE" ]; then
    echo "Error: COMMAND_FILE environment variable not set."
    exit 1
fi

COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$COMMAND_FILE")

if [ -z "$COMMAND" ]; then
    echo "Error: No command found for task ID $SLURM_ARRAY_TASK_ID in $COMMAND_FILE"
    exit 1
fi

echo "Running command for task $SLURM_ARRAY_TASK_ID: $COMMAND"

eval "$COMMAND"
EXIT=$?
if [ $EXIT -ne 0 ]; then
    echo "Command exited with code $EXIT" >&2
fi
exit $EXIT
