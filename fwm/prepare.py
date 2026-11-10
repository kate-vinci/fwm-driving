"""Stage 1: collect data for the four conditions and train every model a seed needs (so that the
evaluation array tasks only read cached files).   python prepare.py --seed 0

The "combined" data (T5) is collected for the analytic-ego RMSE check only; no model is trained on it.
The "union" models (T5 monoliths) are fine-tuned on both single-condition shifted subsets together.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from fwm.common import EGO_PARAMS, SAFETY_PARAMS
from fwm.train import get_model, analytic_ego_rmse, load_data
from fwm.run import param_table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--collect-episodes", type=int, default=300)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--finetune-steps", type=int, default=2000)
    p.add_argument("--budgets", type=lambda s: [int(x) for x in s.split(",")], default=[0, 100, 1000, 10000])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    t0 = time.time()
    log = {"seed": args.seed, "params": param_table(), "n_transitions": {},
           "ego_params": {c: repr(p) for c, p in EGO_PARAMS.items()},
           "safety_params": {c: repr(p) for c, p in SAFETY_PARAMS.items()}}
    for cond in ["nominal", "ego_shift", "traffic_shift", "combined"]:
        S, _, _ = load_data(cond, args.seed, args.data_dir, args.collect_episodes)
        log["n_transitions"][cond] = int(len(S))
    log["ego_rmse"] = {"nominal_member_on_nominal": analytic_ego_rmse("nominal", args.seed, args, "nominal"),
                       "nominal_member_on_ego_shift": analytic_ego_rmse("ego_shift", args.seed, args, "nominal"),
                       "updated_member_on_ego_shift": analytic_ego_rmse("ego_shift", args.seed, args, "ego_shift"),
                       "nominal_member_on_combined": analytic_ego_rmse("combined", args.seed, args, "nominal"),
                       "updated_member_on_combined": analytic_ego_rmse("combined", args.seed, args, "combined")}
    for kind in ["agents", "ego", "mono", "mono_w"]:
        get_model(kind, "nominal", 0, args.seed, args, log)
        for cond in ["ego_shift", "traffic_shift"] + (["union"] if kind.startswith("mono") else []):
            for b in args.budgets:
                get_model(kind, cond, b, args.seed, args, log)
    log["wall_clock_s"] = time.time() - t0
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"prepare_{args.seed}.json"), "w") as f:
        json.dump(log, f, indent=1, default=float)
    print(json.dumps({k: log[k] for k in ["params", "n_transitions", "ego_rmse"]}, indent=1, default=float))
    print(f"[prepare] seed {args.seed} done in {log['wall_clock_s']:.0f}s")


if __name__ == "__main__":
    main()
