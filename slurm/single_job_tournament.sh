#!/bin/bash
# One SLURM task: full tournament in a single process (correct for RL + checkpoints).
# Usage: sbatch slurm/single_job_tournament.sh
# Optional: TOURNAMENT_ROUNDS=500 sbatch ...

#SBATCH --job-name=rl_tournament_full
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=slurm_output/full_%j.out
#SBATCH --error=slurm_output/full_%j.err

# Do not use `set -u` here — conda activate can trip on unset vars in non-interactive jobs.

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
else
    echo "Error: conda not found."
    exit 1
fi

# sbatch may run a copy under /var/spool/slurmd — do not derive root from BASH_SOURCE.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT_ROOT" || exit 1

CONDA_ENV="${RL_TOURNAMENT_CONDA_ENV:-base}"
conda activate "$CONDA_ENV"

type module 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
for mod in gcc/11 gcc/12 gcc/13 gcc; do
    module load "$mod" 2>/dev/null && break
done
if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export MKL_THREADING_LAYER=GNU

if [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ] && [ -f "$HOME/.hf_token" ]; then
    export HUGGING_FACE_HUB_TOKEN=$(cat "$HOME/.hf_token")
fi

if [ -n "${SCRATCH:-}" ] && [ -w "$SCRATCH" ]; then
    export HF_HOME="$SCRATCH/.cache/huggingface"
elif [ -d "/scratch/$USER" ] && [ -w "/scratch/$USER" ]; then
    export HF_HOME="/scratch/$USER/.cache/huggingface"
elif [ -n "${TMPDIR:-}" ] && [ -w "$TMPDIR" ]; then
    export HF_HOME="$TMPDIR/hf_cache"
else
    export HF_HOME="$HOME/.cache/huggingface"
fi
export TRANSFORMERS_CACHE=$HF_HOME/hub
mkdir -p "$TRANSFORMERS_CACHE"

mkdir -p "$PROJECT_ROOT/results"
ROUNDS="${TOURNAMENT_ROUNDS:-500}"
# Log path: override with TOURNAMENT_LOG if $HOME quota is tight (compute nodes may not share /scratch).
LOG_FILE="${TOURNAMENT_LOG:-$PROJECT_ROOT/results/experiment_run.log}"

echo "=== single_job_tournament: rounds=$ROUNDS quantize=1 job=${SLURM_JOB_ID:-local} log=$LOG_FILE ==="
# LoRA checkpoints every round: keep on node /tmp so $HOME quota is not exhausted.
export RL_TOURNAMENT_CHECKPOINT_DIR="${RL_TOURNAMENT_CHECKPOINT_DIR:-/tmp/${USER}_rl_ckpt_${SLURM_JOB_ID:-local}}"
mkdir -p "$RL_TOURNAMENT_CHECKPOINT_DIR"
echo "Checkpoints: $RL_TOURNAMENT_CHECKPOINT_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python3 "$PROJECT_ROOT/main.py" --rounds "$ROUNDS" --quantize 2>&1 | tee "$LOG_FILE"
EXIT=${PIPESTATUS[0]}
echo "python exit code: $EXIT"
exit "$EXIT"
