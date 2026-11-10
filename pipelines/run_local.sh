#!/usr/bin/env bash
# Run the whole study on one machine without SLURM: stage 1 (prepare, one process per seed), then the
# evaluation task list of pipelines/slurm/jobs.sh with N tasks in parallel, then aggregation.
#
#   bash run_local.sh [N_PARALLEL]            default: number of CPU cores
#
# The same environment variables as the SLURM scripts apply (FWM_SEEDS, FWM_BUDGETS, FWM_STALE,
# FWM_EPISODES, FWM_COLLECT_EPISODES, FWM_STEPS, FWM_FT_STEPS, FWM_THREADS). Two shortcuts for a laptop:
#   FWM_NO_ORACLE=1   skip the per-decision oracle scoring for every system except the oracle itself
#                     (no regret column; about 30x cheaper)
#   FWM_EPISODES=50   half the episodes per cell
# Tasks whose result file already exists are skipped, so the script can be interrupted and resumed.
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-$( (nproc 2>/dev/null || sysctl -n hw.ncpu) )}
export FWM_THREADS=${FWM_THREADS:-1} OMP_NUM_THREADS=${FWM_THREADS:-1} MKL_NUM_THREADS=${FWM_THREADS:-1}
SEEDS=${FWM_SEEDS:-"0 1 2 3 4"}
mkdir -p logs

echo "[local] stage 1: prepare ${SEEDS} (${N} in parallel)"
for seed in $SEEDS; do echo "$seed"; done | xargs -P "$N" -I{} sh -c \
  'python scripts/01_prepare.py --seed {} --collect-episodes "${FWM_COLLECT_EPISODES:-300}" --steps "${FWM_STEPS:-5000}" \
     --finetune-steps "${FWM_FT_STEPS:-2000}" --budgets "${FWM_BUDGETS_CSV:-0,100,1000,10000}" > logs/prep_{}.out 2>&1 \
   && echo "[local] prepare seed {} done"'

source pipelines/slurm/jobs.sh
echo "[local] stage 2: ${#JOBS[@]} evaluation tasks (${N} in parallel)"
run_one() {
  args="$1"
  extra=""
  case "$args" in *"--system oracle"*) ;; *) [ "${FWM_NO_ORACLE:-0}" = "1" ] && extra="--no-oracle-scores" ;; esac
  # result file name = results/<test>/<system>_<seed><tag>.jsonl ; skip when present
  test=$(echo "$args" | sed -E 's/.*--test ([A-Z0-9]+).*/\1/'); sys=$(echo "$args" | sed -E 's/.*--system ([a-z_0-9]+).*/\1/')
  seed=$(echo "$args" | sed -E 's/.*--seed ([0-9]+).*/\1/'); tag=$(echo "$args" | sed -nE 's/.*--tag (_[a-z0-9]+).*/\1/p')
  out="results/$test/${sys}_${seed}${tag}.jsonl"
  if [ -s "$out" ]; then echo "[local] skip $out"; return 0; fi
  # shellcheck disable=SC2086
  python scripts/04_run.py $args $extra --episodes "${FWM_EPISODES:-100}" > "logs/eval_${test}_${sys}_${seed}${tag}.out" 2>&1 \
    && echo "[local] done $out" || echo "[local] FAILED $args (see logs/)"
}
export -f run_one
printf '%s\n' "${JOBS[@]}" | xargs -P "$N" -I{} bash -c 'run_one "$@"' _ {}

echo "[local] stage 3: aggregate"
python scripts/05_aggregate.py
echo "[local] all done"
