"""Composition-error budget (reference substitution), the driving analogue of a member-by-member error budget.

python error_budget.py --seed 0 [--episodes 10] [--driver federation]

The driver system drives closed loop in the nominal condition. At every decision, the executed chunk is rolled
out (a) by the true environment (a deep copy, rows of the neighbours tracked), (b) by every predictor below, and
the predicted joint futures are compared with the true one at the planning horizon:

  federation      ego member (+) traffic member through the coupler
  true_ego        the true ego future substituted for the ego member (what the traffic member and the coupler
                  contribute to the error on their own)
  true_traffic    the true neighbour motion substituted for the traffic member (the ego member's share)
  cv_traffic      the traffic member replaced by constant velocity (zero parameters)
  monolith        the parameter-matched monolith
  monolith_aego   the monolith's traffic prediction with the analytic ego member

Errors: ego position error and neighbour position RMSE (present rows) at the horizon, in metres; verdict
disagreement: predicted collision of the chunk against the true one, and predicted safety-margin violation
against the true one. Writes results/error_budget/<driver>_<seed>.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
import numpy as np
from fwm.common import (EGO_PARAMS, SAFETY_PARAMS, COLLISION_PENALTY, DURATION, POLICY_FREQ, make_env, reset_env,
                    observe, split, join, predicted_collision)
from fwm.members import (AnalyticEgo, AgentsMember, SafetyMember, ConstantVelocityMember, Federation, Monolith,
                     MonolithAnalyticEgo, rel_others, H)
from fwm.planner import make_chunks, score_chunk_detail
from fwm.train import get_model


def true_rollout(env, chunk):
    """True joint future of the executed chunk with the neighbour rows tracked (same vehicles, same order)."""
    e = copy.deepcopy(env)
    _, tracked = observe(e)
    states, crashed = [], False
    for a in chunk:
        _, _, _, _, info = e.step(a)
        s, _ = observe(e, tracked)
        states.append(s)
        if info["crashed"]:
            crashed = True
            break
    return states, crashed


def composed(state, chunk, ego_pred, traffic_pred):
    """The coupler's composition with pluggable ego and traffic predictors (the Federation.rollout arithmetic)."""
    ego, oth = split(state)
    rel = rel_others(ego, oth)
    states = []
    for i, a in enumerate(chunk):
        ego_next = ego_pred(ego, a, i)
        d = traffic_pred(ego, rel, i)
        rel_next = rel.copy()
        rel_next[:, 1:6] += d
        rel_next[:, 1] -= ego_next[0] - ego[0]
        rel_next[:, 2] -= ego_next[1] - ego[1]
        others_abs = rel_next.copy()
        others_abs[:, 1] += ego_next[0]
        others_abs[:, 2] += ego_next[1]
        others_abs *= (rel[:, :1] > 0.5)
        states.append(join(ego_next, others_abs))
        ego, rel = ego_next, rel_next
    return states


def errors(pred, true, crashed, safety):
    """Errors of a predicted joint future against the true one at the last common pre-impact step; the verdicts
    use the whole chunk (the true collision is the environment's)."""
    n = min(len(pred), len(true) - (1 if crashed else 0))
    if n > 0:
        pe, te = split(pred[n - 1]), split(true[n - 1])
        present = (te[1][:, 0] > 0.5) & (pe[1][:, 0] > 0.5)
        ego_err = float(np.hypot(*(pe[0][:2] - te[0][:2])))
        oth_err = float(np.sqrt(((pe[1][present, 1:3] - te[1][present, 1:3]) ** 2).sum(1).mean())) if present.any() else np.nan
    else:
        ego_err = oth_err = np.nan
    pc = any(predicted_collision(s, prev_state=p) for p, s in zip([None] + pred[:-1], pred))
    tc = crashed
    pv, tv = safety.violation(pred) > 0, safety.violation(true) > 0
    return {"ego_err": ego_err, "oth_err": oth_err, "collision_disagree": float(pc != tc),
            "unsafe_accept": float(tc and not pc), "margin_disagree": float(pv != tv), "steps": n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--driver", default="federation", choices=["federation", "federation3", "monolith"])
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--finetune-steps", type=int, default=2000)
    p.add_argument("--collect-episodes", type=int, default=300)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    seed, cond, log = args.seed, "nominal", {}
    ego_m = AnalyticEgo(EGO_PARAMS[cond])
    agents = AgentsMember(get_model("agents", cond, 0, seed, args, log))
    mono = get_model("mono", cond, 0, seed, args, log)
    safety = SafetyMember(SAFETY_PARAMS[cond])
    drivers = {"federation": Federation(ego_m, agents), "federation3": Federation(ego_m, agents, safety=safety),
               "monolith": Monolith(mono)}
    driver = drivers[args.driver]
    predictors = {
        "federation": Federation(ego_m, agents), "cv_traffic": Federation(ego_m, ConstantVelocityMember()),
        "monolith": Monolith(mono), "monolith_aego": MonolithAnalyticEgo(mono, ego_m)}
    env = make_env(cond)
    acc = {k: [] for k in list(predictors) + ["true_ego", "true_traffic"]}
    t0, n_dec = time.time(), 0
    for ep in range(args.episodes):
        ep_seed = 2_000_000 + seed * 10_000 + ep
        state, _ = reset_env(env, ep_seed, cond)
        rng = np.random.default_rng([seed, ep, 77])
        for sys_ in list(predictors.values()) + [driver]:
            sys_.begin_episode(ep_seed)
        for t in range(DURATION * POLICY_FREQ):
            for sys_ in list(predictors.values()) + [driver]:
                sys_.observe(state, t)
            chunks = make_chunks(rng)
            scores = [score_chunk_detail(driver, state, c)[0] for c in chunks]
            chunk = chunks[int(np.argmax(scores))]
            true, crashed = true_rollout(env, chunk)
            if len(true) == 0:
                break
            tr_ego = lambda ego, a, i: split(true[i])[0]
            prev_oth = [split(state)[1]] + [split(s)[1] for s in true[:-1]]
            tr_traffic = lambda ego, rel, i: ((split(true[i])[1][:, 1:6] - prev_oth[i][:, 1:6])
                                              * (split(true[i])[1][:, :1] > 0.5) * (prev_oth[i][:, :1] > 0.5))
            futures = {k: s.rollout(state, chunk) for k, s in predictors.items()}
            sub = chunk[:len(true)]                      # the true rollout stops at a collision
            futures["true_ego"] = composed(state, sub, tr_ego, lambda ego, rel, i: agents.predict(ego, rel))
            futures["true_traffic"] = composed(state, sub, lambda ego, a, i: ego_m.predict(ego, a), tr_traffic)
            for k, f in futures.items():
                acc[k].append(errors(f, true, crashed, safety))
            n_dec += 1
            _, _, term, trunc, info = env.step(chunk[0])
            state, _ = observe(env)
            if term or trunc:
                break
    out = {"seed": seed, "driver": args.driver, "episodes": args.episodes, "decisions": n_dec, "horizon": H,
           "wall_clock_s": time.time() - t0, "predictors": {}}
    for k, rows in acc.items():
        out["predictors"][k] = {m: float(np.nanmean([r[m] for r in rows])) for m in rows[0]} if rows else {}
        out["predictors"][k]["oth_err_rms"] = float(np.sqrt(np.nanmean([r["oth_err"] ** 2 for r in rows]))) if rows else None
    os.makedirs(os.path.join(args.out, "error_budget"), exist_ok=True)
    path = os.path.join(args.out, "error_budget", f"{args.driver}_{seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out["predictors"].items():
        print(f"[error_budget] {k:14s} ego {v['ego_err']:.3f} m  others {v['oth_err']:.3f} m  "
              f"collision disagree {v['collision_disagree']:.3f}  unsafe accept {v['unsafe_accept']:.3f}  "
              f"margin disagree {v['margin_disagree']:.3f}")
    print(f"[error_budget] {n_dec} decisions in {out['wall_clock_s']:.0f}s -> {path}")


if __name__ == "__main__":
    main()
