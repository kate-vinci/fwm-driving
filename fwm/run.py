"""Closed-loop evaluation.  python run.py --test T1 --system federation --seed 0

Tests:  T1 nominal | T2 ego shift | T3 traffic shift | T4 coupler stress (stale agents message)
        T5 withheld combination (ego shift and traffic shift together; no learned model has seen it)
        T6 interface faults (stale / wrong frame / wrong units / missing message during a 10 s window)
Systems: oracle random federation federation3 fed_learned_ego monolith monolith_width monolith_safe
         conventional (IDM + MOBIL rules, no model, no planner)
         federation_cv (analytic ego + constant-velocity traffic member: the learned member removed)
         monolith_aego (the monolith's traffic prediction with the analytic ego member substituted)
Writes results/<test>/<system>_<seed>.jsonl, one line per episode (variants: budget / stale / contract / fault).
--horizon H changes the planning horizon (T1 sweep); the record carries "horizon".
"""
from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np
from fwm.common import EGO_PARAMS, SAFETY_PARAMS, EGO_SHIFT_CONDITIONS, make_env
from fwm.members import (AnalyticEgo, LearnedEgo, AgentsMember, SafetyMember, ConstantVelocityMember, Federation,
                     Monolith, MonolithAnalyticEgo, Conventional, Trivial, FAULTS, matched_hidden, new_mlp,
                     set_horizon)
from fwm import members
from fwm.planner import run_episode
from fwm.train import get_model, analytic_ego_rmse

TEST_CONDITION = {"T1": "nominal", "T2": "ego_shift", "T3": "traffic_shift", "T4": "nominal", "T5": "combined",
                  "T6": "nominal"}
SYSTEMS = ["oracle", "random", "federation", "federation3", "fed_learned_ego", "monolith", "monolith_width",
           "monolith_safe", "conventional", "federation_cv", "monolith_aego"]
FEDERATIONS = ("federation", "federation3")                       # analytic ego (+ safety) members
LEARNED = ("fed_learned_ego", "monolith", "monolith_width", "monolith_safe", "monolith_aego")   # adapted with shifted data
NO_VARIANTS = ("oracle", "random", "conventional", "federation_cv")   # nothing to adapt, nothing to corrupt


def param_table():
    agents, ego, mono_w = new_mlp("agents", 0).n_params, new_mlp("ego", 0).n_params, new_mlp("mono", 0).n_params
    mono = new_mlp("mono", 0, matched_hidden()).n_params
    return {"federation": agents, "federation_agents_member": agents, "federation_ego_member": 0,
            "federation3": agents, "federation3_safety_member": 0,
            "fed_learned_ego": agents + ego, "monolith": mono, "monolith_hidden": matched_hidden(),
            "monolith_width": mono_w, "monolith_width_hidden": 128, "monolith_safe": mono,
            "conventional": 0, "federation_cv": 0, "monolith_aego": mono}


def build(system, test, seed, budget, stale, contract, args, log, fault=None):
    """Instantiate the system under test for one variant."""
    sp = args.stale_prob
    cond = TEST_CONDITION[test]
    if system in ("oracle", "random"):
        return Trivial(system)
    if system == "conventional":
        return Conventional()
    if system == "federation_cv":
        return Federation(AnalyticEgo(EGO_PARAMS[cond]), ConstantVelocityMember(), contract=contract, delay=stale,
                          stale_prob=sp, name=system, fault=fault)
    if system == "monolith_aego":
        mono_cond = "union" if test == "T5" else cond
        return MonolithAnalyticEgo(get_model("mono", mono_cond, budget, seed, args, log), AnalyticEgo(EGO_PARAMS[cond]),
                                   delay=stale, stale_prob=sp, name=system, fault=fault)
    if system in FEDERATIONS:
        # analytic ego (and safety) members: parameters set to the condition's values (zero data); the
        # agents member is fine-tuned on traffic_shift data under the traffic shift (T3) and under the
        # withheld combination (T5)
        agents_cond, agents_budget = ("traffic_shift", budget) if test in ("T3", "T5") else ("nominal", 0)
        safety = SafetyMember(SAFETY_PARAMS[cond]) if system == "federation3" else None
        return Federation(AnalyticEgo(EGO_PARAMS[cond]), AgentsMember(get_model("agents", agents_cond, agents_budget, seed, args, log)),
                          contract=contract, delay=stale, stale_prob=sp, name=system, safety=safety, fault=fault)
    if system == "fed_learned_ego":
        # T5: each learned member is fine-tuned on the shifted data of its own domain
        ego_cond, agents_cond = ("ego_shift", "traffic_shift") if test == "T5" else (cond, cond)
        ego = LearnedEgo(get_model("ego", ego_cond, budget, seed, args, log))
        agents = AgentsMember(get_model("agents", agents_cond, budget, seed, args, log))
        return Federation(ego, agents, contract=contract, delay=stale, stale_prob=sp, name="fed_learned_ego", fault=fault)
    # monoliths; T5: fine-tuned on the union of both shifted data sets (2 x budget transitions)
    kind = "mono_w" if system == "monolith_width" else "mono"
    mono_cond = "union" if test == "T5" else cond
    safety = SafetyMember(SAFETY_PARAMS[cond]) if system == "monolith_safe" else None
    return Monolith(get_model(kind, mono_cond, budget, seed, args, log), delay=stale, stale_prob=sp, name=system,
                    safety=safety, fault=fault)


def variants(test, system, args):
    """List of (budget, stale, contract, fault) variants a system runs under a test."""
    learned = system in LEARNED
    if system in NO_VARIANTS and test != "T6":
        return [(0, 0, True, None)]
    if test == "T2":
        return [(b, 0, True, None) for b in args.budgets] if learned else [(0, 0, True, None)]
    if test in ("T3", "T5"):
        return [(b, 0, True, None) for b in args.budgets] if (learned or system in FEDERATIONS) else [(0, 0, True, None)]
    if test == "T4":
        if system in FEDERATIONS or system == "fed_learned_ego":
            return [(0, k, c, None) for k in args.stale for c in (True, False)]
        return [(0, k, True, None) for k in args.stale] if learned else [(0, 0, True, None)]
    if test == "T6":
        if system in FEDERATIONS or system in ("fed_learned_ego", "federation_cv"):
            return [(0, 0, c, f) for f in args.faults for c in (True, False)]
        return [(0, 0, True, f) for f in args.faults] if learned else [(0, 0, True, None)]
    return [(0, 0, True, None)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", required=True, choices=list(TEST_CONDITION))
    p.add_argument("--system", required=True, choices=SYSTEMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--steps", type=int, default=5000, help="training steps for nominal models")
    p.add_argument("--finetune-steps", type=int, default=2000)
    p.add_argument("--budgets", type=lambda s: [int(x) for x in s.split(",")], default=[0, 100, 1000, 10000])
    p.add_argument("--stale", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2, 4])
    p.add_argument("--stale-prob", type=float, default=1.0, help="T4: probability a decision's message is delayed")
    p.add_argument("--faults", type=lambda s: [x for x in s.split(",") if x], default=list(FAULTS),
                   help="T6: fault classes to inject (stale,frame,unit,missing)")
    p.add_argument("--horizon", type=int, default=members.H, help="planning horizon in decisions (default 3)")
    p.add_argument("--collect-episodes", type=int, default=300, help="episodes to collect if data is missing")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--out", default="results")
    p.add_argument("--no-oracle-scores", action="store_true", help="skip per-decision oracle scoring (no regret)")
    p.add_argument("--tag", default="", help="suffix for the output file name (one variant per SLURM task)")
    args = p.parse_args()

    set_horizon(args.horizon)
    cond = TEST_CONDITION[args.test]
    out_dir = os.path.join(args.out, args.test)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.system}_{args.seed}{args.tag}.jsonl")
    log = {"params": param_table(), "ego_params": repr(EGO_PARAMS[cond]), "safety_params": repr(SAFETY_PARAMS[cond])}
    if args.system == "federation":
        log["ego_rmse_nominal_member_on_" + cond] = analytic_ego_rmse(cond, args.seed, args, "nominal")
        if cond in EGO_SHIFT_CONDITIONS:
            log["ego_rmse_updated_member_on_" + cond] = analytic_ego_rmse(cond, args.seed, args, cond)
    env = make_env(cond)
    t_start = time.time()
    with open(out_path, "w") as f:
        for budget, stale, contract, fault in variants(args.test, args.system, args):
            system = build(args.system, args.test, args.seed, budget, stale, contract, args, log, fault=fault)
            t0 = time.time()
            for ep in range(args.episodes):
                rng = np.random.default_rng([args.seed, ep, 99])
                rec = run_episode(env, system, 1_000_000 + args.seed * 10_000 + ep, cond, rng,
                                  oracle_scoring=not args.no_oracle_scores)
                rec.update(test=args.test, system=args.system, seed=args.seed, episode=ep, condition=cond,
                           budget=budget, stale=stale, stale_prob=args.stale_prob, contract=contract,
                           fault=fault, horizon=args.horizon, n_params=system.n_params)
                f.write(json.dumps(rec) + "\n")
                f.flush()
            print(f"[run] {args.test} {args.system} seed={args.seed} budget={budget} stale={stale} "
                  f"contract={contract} fault={fault} H={args.horizon}: {args.episodes} episodes in {time.time() - t0:.1f}s")
    log["wall_clock_s"] = time.time() - t_start
    with open(os.path.join(out_dir, f"{args.system}_{args.seed}{args.tag}_meta.json"), "w") as f:
        json.dump(log, f, indent=1, default=float)
    print(f"[run] wrote {out_path} ({log['wall_clock_s']:.0f}s total)")


if __name__ == "__main__":
    main()
