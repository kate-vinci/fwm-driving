#!/usr/bin/env bash
#SBATCH --job-name=fwm_evaluate
#SBATCH --array=0-519%120
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
#SBATCH --output=logs/evaluate_%A_%a.out
# Stage 2: one evaluation variant per array task (5 seeds x 104 variants = 520 tasks by default; the
# array range is overridden by pipelines/slurm/submit.sh from the length of JOBS). Each task runs 100 closed-loop
# episodes with per-decision oracle scoring, ~45-75 min on one core; T4 federation tasks run the contract
# check on and off and take about twice as long. Reads data/ and models/ written by prepare.sbatch.
#   bash pipelines/slurm/evaluate.sbatch --list     prints the task list and the task count
set -euo pipefail
cd "${FWM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source pipelines/slurm/jobs.sh
if [ "${1:-}" = "--list" ]; then printf '%s\n' "${JOBS[@]}"; echo "${#JOBS[@]} tasks"; exit 0; fi
if [ -n "${FWM_PYTHON_MODULE:-}" ] && command -v module >/dev/null; then module load "$FWM_PYTHON_MODULE"; fi
python -c "import highway_env, torch, matplotlib" 2>/dev/null || \
  pip install --user --quiet -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
export FWM_THREADS=${FWM_THREADS:-1} OMP_NUM_THREADS=${FWM_THREADS:-1} MKL_NUM_THREADS=${FWM_THREADS:-1}
echo "task $SLURM_ARRAY_TASK_ID: ${JOBS[$SLURM_ARRAY_TASK_ID]}"
# shellcheck disable=SC2086
python scripts/04_run.py ${JOBS[$SLURM_ARRAY_TASK_ID]} --episodes "${FWM_EPISODES:-100}" --collect-episodes "${FWM_COLLECT_EPISODES:-300}"
