"""Collect transitions with an epsilon-greedy heuristic behaviour policy.

python collect.py --episodes 300 --seed 0 --condition nominal
-> data/<condition>_<seed>.npz with arrays S (n,49), A (n,), S2 (n,49), crashed (n,), episode (n,), speed (n,)
Row identity of the other vehicles is preserved between S and S2 (same vehicles, one step later).
"""
from __future__ import annotations

import argparse
import os
import time
import numpy as np
from fwm.common import (make_env, reset_env, observe, split, FASTER, SLOWER, LANE_LEFT, LANE_RIGHT, IDLE,
                    LANE_W, DURATION, POLICY_FREQ)

EPS = 0.3          # epsilon-greedy exploration
AHEAD_M = 30.0     # brake if a vehicle in the ego lane is closer than this (20 m crashes too often)
P_LANE_CHANGE = 0.03


def heuristic(state, rng):
    ego, others = split(state)
    ahead = [o for o in others if o[0] > 0.5 and abs(o[2] - ego[1]) < LANE_W / 2 and 0 < o[1] - ego[0] < AHEAD_M]
    if rng.random() < P_LANE_CHANGE:
        return int(rng.choice([LANE_LEFT, LANE_RIGHT]))
    return SLOWER if ahead else FASTER


def collect(episodes, seed, condition, out_dir="data", verbose=True, min_transitions=0):
    """Log `episodes` episodes and, if min_transitions > 0, keep logging until that many usable (non-crash)
    transitions exist, so that a data budget is a real subset of the logged data in every condition."""
    env = make_env(condition)
    rng = np.random.default_rng([seed, 12345])
    S, A, S2, C, E, V = [], [], [], [], [], []
    t0 = time.time()
    ep = -1
    while True:
        ep += 1
        if ep >= episodes and (len(C) - sum(C)) >= min_transitions:
            break
        state, tracked = reset_env(env, seed * 10_000 + ep, condition)
        for t in range(DURATION * POLICY_FREQ):
            a = int(rng.integers(5)) if rng.random() < EPS else heuristic(state, rng)
            _, _, term, trunc, info = env.step(a)
            nxt, _ = observe(env, tracked)
            S.append(state); A.append(a); S2.append(nxt); C.append(bool(info["crashed"]))
            E.append(ep); V.append(float(info["speed"]))
            state, tracked = observe(env)
            if term or trunc:
                break
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{condition}_{seed}.npz")
    np.savez_compressed(path, S=np.array(S), A=np.array(A), S2=np.array(S2), crashed=np.array(C),
                        episode=np.array(E), speed=np.array(V))
    if verbose:
        print(f"[collect] {condition} seed={seed}: {len(S)} transitions ({len(C) - sum(C)} usable), {ep} episodes, "
              f"{np.mean(C):.3f} crash rate/transition, {time.time() - t0:.1f}s -> {path}")
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    # "combined" (T5) is collected only for the analytic-ego RMSE check and optional evaluation data; no
    # learned model is ever trained on it
    p.add_argument("--condition", default="nominal", choices=["nominal", "ego_shift", "traffic_shift", "combined"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--min-transitions", type=int, default=0)
    a = p.parse_args()
    collect(a.episodes, a.seed, a.condition, a.data_dir, min_transitions=a.min_transitions)
