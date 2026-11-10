#!/usr/bin/env bash
#SBATCH --job-name=fwm_prepare
#SBATCH --array=0-4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
#SBATCH --output=logs/prepare_%A_%a.out
# Stage 1 (one array task per seed): collect 4 x 300 episodes and train every model (nominal, fine-tuned
# per shifted condition and budget, and the union models for T5). About 20-30 min per seed on one core.
# Account, partition, time limit and Python module are passed by pipelines/slurm/submit.sh from environment
# variables (see README, "Full run on a SLURM cluster"); the script itself names no cluster.
set -euo pipefail
cd "${FWM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ -n "${FWM_PYTHON_MODULE:-}" ] && command -v module >/dev/null; then module load "$FWM_PYTHON_MODULE"; fi
# dependencies are normally installed once from the login node (README); this is only a fallback
python -c "import highway_env, torch, matplotlib" 2>/dev/null || \
  pip install --user --quiet -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
export FWM_THREADS=${FWM_THREADS:-1} OMP_NUM_THREADS=${FWM_THREADS:-1} MKL_NUM_THREADS=${FWM_THREADS:-1}
mkdir -p logs
python scripts/01_prepare.py --seed "$SLURM_ARRAY_TASK_ID" --collect-episodes "${FWM_COLLECT_EPISODES:-300}" \
  --steps "${FWM_STEPS:-5000}" --finetune-steps "${FWM_FT_STEPS:-2000}" \
  --budgets "${FWM_BUDGETS_CSV:-0,100,1000,10000}"
