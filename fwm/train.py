"""Training / fine-tuning with on-disk caching, plus the analytic-ego RMSE check on logged data."""
from __future__ import annotations

import os
import time
import numpy as np
from fwm.common import EGO_PARAMS, ego_step, split
from fwm.members import build_xy, new_mlp, matched_hidden
from fwm.nn_backend import MLP
from fwm.collect import collect


BUDGET_CONDITIONS = ("ego_shift", "traffic_shift")   # conditions whose data is handed out by budget
MIN_TRANSITIONS = int(os.environ.get("FWM_MIN_TRANSITIONS", 10_500))   # usable transitions each must hold (budget 10,000)


def load_data(condition, seed, data_dir, collect_episodes=300):
    """Logged transitions of a condition, crash steps excluded. The two budget conditions are logged until
    they hold MIN_TRANSITIONS usable transitions (aggressive traffic ends episodes early), so that the largest
    budget is a genuine subset; a file logged with fewer is re-logged."""
    path = os.path.join(data_dir, f"{condition}_{seed}.npz")
    need = MIN_TRANSITIONS if condition in BUDGET_CONDITIONS else 0
    if os.path.exists(path):
        d = np.load(path)
        if int((~d["crashed"]).sum()) < need:
            print(f"[train] {path} holds {int((~d['crashed']).sum())} usable transitions < {need} -> re-logging")
            os.remove(path)
    if not os.path.exists(path):
        print(f"[train] {path} missing -> collecting {collect_episodes} episodes (at least {need} usable transitions)")
        collect(collect_episodes, seed, condition, data_dir, min_transitions=need)
    d = np.load(path)
    keep = ~d["crashed"]                       # a crashing step has post-impact dynamics: exclude
    return d["S"][keep], d["A"][keep], d["S2"][keep]


UNION = ("ego_shift", "traffic_shift")   # condition "union": both shifted data sets, `budget` transitions of each


def shifted_subset(condition, budget, seed, args):
    """The first `budget` transitions of the seeded permutation of a shifted condition's data."""
    S, A, S2 = load_data(condition, seed, args.data_dir, args.collect_episodes)
    idx = np.random.default_rng([seed, 7]).permutation(len(S))[:budget]
    return S[idx], A[idx], S2[idx]


def get_model(kind, condition, budget, seed, args, log):
    """kind in {agents, ego, mono, mono_w}. Nominal models are trained on nominal data; for a shifted
    condition the nominal model is fine-tuned on the first `budget` shifted transitions (budget 0 =
    nominal model, no adaptation). Condition "union" (T5) fine-tunes on the first `budget` ego_shift
    transitions and the first `budget` traffic_shift transitions together (2 x budget, the same subsets as
    the single-condition fine-tunes). Cached under args.model_dir."""
    os.makedirs(args.model_dir, exist_ok=True)
    arch = "mono" if kind.startswith("mono") else kind
    hidden = matched_hidden() if kind == "mono" else 128
    nom_path = os.path.join(args.model_dir, f"{kind}_nominal_s{seed}.pt")
    if os.path.exists(nom_path):
        model = MLP.load(nom_path)
    else:
        S, A, S2 = load_data("nominal", seed, args.data_dir, args.collect_episodes)
        X, Y, M = build_xy(arch, S, A, S2)
        t0 = time.time()
        model = new_mlp(arch, seed, hidden).fit(X, Y, M, steps=args.steps, seed=seed)
        model.save(nom_path)
        log.setdefault("training", {})[f"{kind}_nominal"] = dict(model.train_log, secs=time.time() - t0)
        print(f"[train] {kind} nominal: n={len(X)} params={model.n_params} loss={model.train_log['final_loss']:.4f}"
              f" ({time.time() - t0:.1f}s)")
    if condition == "nominal" or budget == 0:
        return model
    ft_path = os.path.join(args.model_dir, f"{kind}_{condition}_b{budget}_s{seed}.pt")
    if os.path.exists(ft_path):
        return MLP.load(ft_path)
    if condition == "union":
        parts = [shifted_subset(c, budget, seed, args) for c in UNION]
        S, A, S2 = (np.concatenate([p[i] for p in parts]) for i in range(3))
    else:
        S, A, S2 = shifted_subset(condition, budget, seed, args)
    X, Y, M = build_xy(arch, S, A, S2)
    t0 = time.time()
    model.fit(X, Y, M, steps=args.finetune_steps, seed=seed, reset_norm=False)
    model.save(ft_path)
    log.setdefault("training", {})[f"{kind}_{condition}_b{budget}"] = dict(model.train_log, secs=time.time() - t0)
    print(f"[train] {kind} fine-tuned on {condition} budget={budget} (n={len(X)}) "
          f"loss={model.train_log['final_loss']:.4f} ({time.time() - t0:.1f}s)")
    return model


def analytic_ego_rmse(condition, seed, args, params_name="nominal"):
    """One-step RMSE of the analytic ego member (with EGO_PARAMS[params_name]) on logged transitions."""
    S, A, S2 = load_data(condition, seed, args.data_dir, args.collect_episodes)
    err = np.array([ego_step(split(s)[0], a, EGO_PARAMS[params_name])[:5] - split(s2)[0][:5]
                    for s, a, s2 in zip(S, A, S2)])
    rmse = np.sqrt((err ** 2).mean(0))
    return {"x": rmse[0], "y": rmse[1], "vx": rmse[2], "vy": rmse[3], "heading": rmse[4], "n": len(err)}
