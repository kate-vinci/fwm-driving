#!/usr/bin/env bash
# Reduced end-to-end smoke test (< 8 min on one CPU core): 20 collection episodes per condition,
# 500 training steps, 1-2 evaluation episodes, 1 seed, budgets 0 and 100, all six tests and every
# system. Writes into smoke_data/, smoke_models/ and smoke_results/<test>/{table.md,table.tex,figure.png}.
set -euo pipefail
cd "$(dirname "$0")/.."
export FWM_THREADS=${FWM_THREADS:-1} FWM_MIN_TRANSITIONS=0
D=smoke_data; M=smoke_models; R=smoke_results
COMMON="--data-dir $D --model-dir $M --out $R --seed 0 --episodes 2 --steps 500 --finetune-steps 200 --budgets 0,100 --stale 0,2 --collect-episodes 20"
python scripts/01_prepare.py --seed 0 --collect-episodes 20 --steps 500 --finetune-steps 200 --budgets 0,100 --data-dir $D --model-dir $M --out $R
for s in federation federation3 monolith monolith_safe oracle; do python scripts/04_run.py --test T1 --system $s $COMMON --episodes 1; done   # with oracle scoring (regret)
for s in random fed_learned_ego monolith_width conventional federation_cv monolith_aego; do python scripts/04_run.py --test T1 --system $s $COMMON --no-oracle-scores; done
for s in federation monolith; do python scripts/04_run.py --test T1 --system $s $COMMON --no-oracle-scores --horizon 5 --tag _h5; done
for s in federation federation3 fed_learned_ego monolith monolith_safe; do python scripts/04_run.py --test T2 --system $s $COMMON --no-oracle-scores; done
for s in federation federation3 monolith monolith_safe; do python scripts/04_run.py --test T3 --system $s $COMMON --no-oracle-scores; done
for s in federation federation3 monolith monolith_safe; do python scripts/04_run.py --test T4 --system $s $COMMON --no-oracle-scores; done
for s in federation federation3 fed_learned_ego monolith monolith_safe conventional federation_cv monolith_aego; do python scripts/04_run.py --test T5 --system $s $COMMON --no-oracle-scores; done
for s in federation federation3 federation_cv monolith monolith_safe; do python scripts/04_run.py --test T6 --system $s $COMMON --no-oracle-scores; done
python scripts/05_aggregate.py --results $R --draws 500
echo "smoke test OK"
