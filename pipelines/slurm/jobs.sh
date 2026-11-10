#!/usr/bin/env bash
# Enumerates every evaluation task (one variant per task) as run.py argument strings.
# Usage: source pipelines/slurm/jobs.sh; echo ${#JOBS[@]}; echo "${JOBS[$i]}"
# Per seed: T1 11 + horizon 10, T2 21, T3 27, T4 24, T5 32, T6 24 = 149; five seeds = 745 tasks.
# FWM_CORE=1 restricts the list to the original study (T1-T5, eight systems: 104 tasks per seed).
SEEDS=${FWM_SEEDS:-"0 1 2 3 4"}
BUDGETS=${FWM_BUDGETS:-"0 100 1000 10000"}
STALE=${FWM_STALE:-"0 1 2 4"}
FAULTS=${FWM_FAULTS:-"stale frame unit missing"}
HORIZONS=${FWM_HORIZONS:-"2 5"}                                          # T1 sweep (3 is the study's horizon)
ALL="oracle random federation federation3 fed_learned_ego monolith monolith_width monolith_safe"
LEARNED="fed_learned_ego monolith monolith_width monolith_safe"          # adapted with shifted data
ADAPTED="federation federation3 $LEARNED"                                # federations adapt the traffic member
EXTRA="conventional federation_cv monolith_aego"                         # comparator and attribution systems
JOBS=()
for seed in $SEEDS; do
  for s in $ALL; do JOBS+=("--test T1 --system $s --seed $seed"); done
  for s in oracle random federation federation3; do JOBS+=("--test T2 --system $s --seed $seed"); done
  for s in $LEARNED; do for b in $BUDGETS; do
    JOBS+=("--test T2 --system $s --seed $seed --budgets $b --tag _b$b"); done; done
  for t in T3 T5; do
    for s in oracle random; do JOBS+=("--test $t --system $s --seed $seed"); done
    for s in $ADAPTED; do for b in $BUDGETS; do
      JOBS+=("--test $t --system $s --seed $seed --budgets $b --tag _b$b"); done; done
  done
  for s in $ADAPTED; do for k in $STALE; do
    JOBS+=("--test T4 --system $s --seed $seed --stale $k --tag _k$k"); done; done
  [ "${FWM_CORE:-0}" = "1" ] && continue
  # --- complements: conventional comparator, attribution, horizon sweep, interface faults
  for s in $EXTRA; do JOBS+=("--test T1 --system $s --seed $seed"); done
  for s in oracle federation federation3 monolith monolith_safe; do for h in $HORIZONS; do
    JOBS+=("--test T1 --system $s --seed $seed --horizon $h --tag _h$h"); done; done
  for t in T2 T3; do JOBS+=("--test $t --system conventional --seed $seed"); done
  JOBS+=("--test T5 --system conventional --seed $seed")
  JOBS+=("--test T5 --system federation_cv --seed $seed")
  for b in $BUDGETS; do JOBS+=("--test T5 --system monolith_aego --seed $seed --budgets $b --tag _b$b"); done
  for s in federation federation3 monolith monolith_safe; do for f in $FAULTS; do
    JOBS+=("--test T6 --system $s --seed $seed --faults $f --tag _f$f"); done; done
  for s in federation_cv fed_learned_ego; do for f in $FAULTS; do
    JOBS+=("--test T6 --system $s --seed $seed --faults $f --tag _f$f"); done; done
done
