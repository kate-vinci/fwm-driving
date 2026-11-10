"""Chunked model-predictive planner shared by all systems, oracle scoring, and the episode loop."""
from __future__ import annotations

import copy
import time
import numpy as np
from fwm.common import (N_ACTIONS, SLOWER, COLLISION_PENALTY, DURATION, POLICY_FREQ, reset_env, observe,
                    reward_model)
from fwm import members
N_RANDOM_CHUNKS = 4
SLOWER_CHUNK = SLOWER   # index of the constant-SLOWER chunk (chunks 0..4 are the constant actions)


def make_chunks(rng):
    H = members.H
    chunks = [[a] * H for a in range(N_ACTIONS)]
    chunks += [[int(x) for x in rng.integers(0, N_ACTIONS, size=H)] for _ in range(N_RANDOM_CHUNKS)]
    return chunks


def score_chunk_detail(system, state, chunk):
    """(score, safety violation) of a chunk. Score = sum of predicted rewards; large penalty and stop at
    the first predicted collision. A chunk whose predicted joint future violates the safety margin is
    scored like an unexecutable one, -COLLISION_PENALTY minus the violation, so that among several
    violating chunks the least violating wins. Score None = abstain (contract violation)."""
    states = system.rollout(state, chunk)
    if states is None:
        return None, 0.0
    pen = system.safety_penalty(states)
    if pen > 0:
        return -COLLISION_PENALTY - pen, pen
    total, prev = 0.0, state
    for s in states:
        r, collided = reward_model(s, prev_state=prev)
        total += r
        if collided:
            return total - COLLISION_PENALTY, pen
        prev = s
    return total, pen


def score_chunk(system, state, chunk):
    """Score of a chunk (see score_chunk_detail); None = abstain."""
    return score_chunk_detail(system, state, chunk)[0]


def oracle_scores(env, chunks):
    """True score of each chunk, obtained by rolling out a deep copy of the environment."""
    scores = []
    for chunk in chunks:
        e = copy.deepcopy(env)
        total = 0.0
        for a in chunk:
            _, r, term, _, _ = e.step(a)
            total += r
            if e.unwrapped.vehicle.crashed:
                total -= COLLISION_PENALTY
                break
        scores.append(float(total))
    return scores


def run_episode(env, system, ep_seed, condition, rng, oracle_scoring=True):
    state, _ = reset_env(env, ep_seed, condition)
    system.begin_episode(ep_seed)
    log = {"chosen": [], "regret": [], "latency_ms": [], "abstained": [], "true_scores": [], "pred_scores": [],
           "safety_violation": []}
    speeds, ret, crashed = [], 0.0, False
    for t in range(DURATION * POLICY_FREQ):
        system.observe(state, t)
        chunks = make_chunks(rng)
        t0 = time.perf_counter()
        true = oracle_scores(env, chunks) if ((oracle_scoring and system.kind != "rule") or system.kind == "oracle") else None
        t_oracle = (time.perf_counter() - t0) * 1000.0
        abstained, pred, viol = False, None, 0.0
        t0 = time.perf_counter()
        if system.kind == "oracle":
            choice = int(np.argmax(true))
        elif system.kind == "random":
            choice = int(rng.integers(len(chunks)))
        elif system.kind == "rule":                   # conventional driver: one action from the rules, no chunk
            choice = None
            rule_action = int(system.act(state))
        else:
            scored = [score_chunk_detail(system, state, c) for c in chunks]
            pred = [sc for sc, _ in scored]
            if pred[0] is None:                       # contract violation -> conservative fallback
                abstained, choice = True, SLOWER_CHUNK
            else:
                choice = int(np.argmax(pred))
                viol = float(scored[choice][1])       # margin violation of the executed chunk
        latency = t_oracle if system.kind == "oracle" else (time.perf_counter() - t0) * 1000.0
        action = chunks[choice][0] if choice is not None else rule_action
        _, r, term, trunc, info = env.step(action)
        state, _ = observe(env)
        ret += r
        speeds.append(float(info["speed"]))
        crashed = bool(info["crashed"])
        log["chosen"].append(choice)
        log["regret"].append(float(max(true) - true[choice]) if (true is not None and choice is not None) else None)
        log["latency_ms"].append(latency)
        log["abstained"].append(abstained)
        log["true_scores"].append(true)
        log["pred_scores"].append(None if pred is None else [None if p is None else float(p) for p in pred])
        log["safety_violation"].append(viol)
        if term or trunc:
            break
    regrets = [x for x in log["regret"] if x is not None]
    return {"success": not crashed, "crashed": crashed, "steps": len(speeds), "return": float(ret),
            "mean_speed": float(np.mean(speeds)), "regret_mean": float(np.mean(regrets)) if regrets else None,
            "latency_ms_mean": float(np.mean(log["latency_ms"])),
            "abstentions": int(sum(log["abstained"])), "contract_violations": int(system.violations),
            "safety_vetoes": int(sum(v > 0 for v in log["safety_violation"])),
            "safety_violation_mean": float(np.mean(log["safety_violation"])),
            "decisions": log}
