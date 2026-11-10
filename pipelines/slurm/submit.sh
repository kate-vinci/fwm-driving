#!/usr/bin/env bash
# Submit stage 1 (prepare, one task per seed), stage 2 (evaluation array, dependency on stage 1) and the
# aggregation (dependency on stage 2). Every cluster-specific setting is an environment variable:
#   FWM_ACCOUNT        SLURM account (sbatch -A); leave unset if the cluster needs none
#   FWM_PARTITION      SLURM partition (sbatch -p); leave unset for the cluster default
#   FWM_PYTHON_MODULE  environment module that provides Python >= 3.10, e.g. python/3.11; unset = none
#   FWM_TIME_PREPARE   time limit of a prepare task      (default 02:00:00)
#   FWM_TIME_EVAL      time limit of an evaluation task  (default 04:00:00)
#   FWM_TIME_AGG       time limit of the aggregation     (default 00:30:00)
#   FWM_MAX_CONCURRENT array throttle                    (default 120)
#   FWM_ROOT           repository directory              (default: current directory)
# Further knobs read by the scripts: FWM_SEEDS, FWM_BUDGETS, FWM_STALE, FWM_EPISODES, FWM_COLLECT_EPISODES,
# FWM_STEPS, FWM_FT_STEPS, FWM_BUDGETS_CSV, FWM_THREADS.
set -euo pipefail
export FWM_ROOT=${FWM_ROOT:-$(pwd)}
cd "$FWM_ROOT"; mkdir -p logs
OPTS=()
[ -n "${FWM_ACCOUNT:-}" ] && OPTS+=(-A "$FWM_ACCOUNT")
[ -n "${FWM_PARTITION:-}" ] && OPTS+=(-p "$FWM_PARTITION")
source pipelines/slurm/jobs.sh; N=${#JOBS[@]}
SEED_ARRAY=$(echo ${FWM_SEEDS:-"0 1 2 3 4"} | tr ' ' ',')
J1=$(sbatch --parsable "${OPTS[@]}" --export=ALL --time="${FWM_TIME_PREPARE:-02:00:00}" --array="$SEED_ARRAY" pipelines/slurm/prepare.sbatch)
J2=$(sbatch --parsable "${OPTS[@]}" --export=ALL --time="${FWM_TIME_EVAL:-04:00:00}" --dependency=afterok:"$J1" \
     --array=0-$((N-1))%"${FWM_MAX_CONCURRENT:-120}" pipelines/slurm/evaluate.sbatch)
LOAD=""; [ -n "${FWM_PYTHON_MODULE:-}" ] && LOAD="module load $FWM_PYTHON_MODULE && "
J3=$(sbatch --parsable "${OPTS[@]}" --export=ALL --time="${FWM_TIME_AGG:-00:30:00}" --mem=4G --dependency=afterany:"$J2" \
     --job-name=fwm_aggregate --output=logs/aggregate_%j.out --wrap "cd $FWM_ROOT && ${LOAD}python scripts/05_aggregate.py")
echo "submitted prepare=$J1 evaluate=$J2 ($N tasks) aggregate=$J3"
