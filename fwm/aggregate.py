"""Aggregate results/<test>/*.jsonl into markdown + LaTeX tables and a figure per test.

python aggregate.py [--results results] [--draws 4000]
Differences "system - monolith" are paired by (seed, episode) with a paired bootstrap 95% interval.
"veto" is the fraction of decisions whose executed chunk violated the safety margin (systems with a
safety member only).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
import warnings
import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")

METRICS = ["success", "mean_speed", "return", "regret_mean", "latency_ms_mean", "latency_p50", "latency_p95", "deadline_miss",
           "abstain_rate", "safety_veto_rate", "safety_violation_mean"]
DEADLINE_MS = 100.0   # per-decision budget for the deadline-miss rate (CPU runs only; see README)
COLORS = {"federation": "#2a78d6", "federation3": "#7a4fd6", "fed_learned_ego": "#1baf7a", "monolith": "#eb6834",
          "monolith_width": "#eda100", "monolith_safe": "#b8860b", "oracle": "#52514e", "random": "#898781",
          "conventional": "#2f7f6f", "federation_cv": "#79a9e6", "monolith_aego": "#f0a58a"}
LABELS = {"federation": "Federation (analytic ego)", "federation3": "Federation (analytic ego + safety)",
          "fed_learned_ego": "Federation (learned ego)", "monolith": "Monolith (=params)",
          "monolith_width": "Monolith (=width)", "monolith_safe": "Monolith (=params) + safety",
          "oracle": "Oracle", "random": "Random", "conventional": "Conventional (IDM+MOBIL, no model)",
          "federation_cv": "Federation, constant-velocity traffic", "monolith_aego": "Monolith + analytic ego"}
BUDGET_TESTS = {"T2": "T2 ego shift", "T3": "T3 traffic shift", "T5": "T5 withheld combination"}
FAULT_ORDER = ["stale", "frame", "unit", "missing"]


def load(test_dir):
    groups = defaultdict(dict)   # (system, budget, stale, contract) -> {(seed, ep): metrics}
    for path in glob.glob(os.path.join(test_dir, "*.jsonl")):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                stale = r["stale"] if r.get("stale_prob", 1.0) >= 1.0 else f"{r['stale']}@p{r['stale_prob']}"
                key = (r["system"], r["budget"], stale, r["contract"], r.get("fault"), r.get("horizon", 3))
                lat = np.asarray(r["decisions"]["latency_ms"], float) if "decisions" in r else np.array([r["latency_ms_mean"]])
                groups[key][(r["seed"], r["episode"])] = {
                    "success": float(r["success"]), "mean_speed": r["mean_speed"], "return": r["return"],
                    "regret_mean": r["regret_mean"] if r["regret_mean"] is not None else np.nan,
                    "latency_ms_mean": r["latency_ms_mean"], "latency_p50": float(np.median(lat)),
                    "latency_p95": float(np.percentile(lat, 95)), "deadline_miss": float((lat > DEADLINE_MS).mean()),
                    "abstain_rate": r["abstentions"] / max(r["steps"], 1),
                    "safety_veto_rate": r.get("safety_vetoes", 0) / max(r["steps"], 1),
                    "safety_violation_mean": r.get("safety_violation_mean", 0.0),
                    "n_params": r["n_params"]}
    return groups


def paired_boot(a, b, draws, rng):
    d = np.asarray(a) - np.asarray(b)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(d), size=(draws, len(d)))
    means = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(groups, draws, rng):
    rows = []
    n_variants = defaultdict(set)
    for (s, b, k, c, f, h) in groups:
        n_variants[s].add((b, k, c, f, h))
    for key in sorted(groups, key=lambda t: (t[0], t[1], str(t[2]), not t[3], str(t[4]), t[5])):
        s, b, k, c, f, h = key
        eps = groups[key]
        row = {"system": s, "budget": b, "stale": k, "contract": c, "fault": f, "horizon": h, "n": len(eps),
               "n_params": int(next(iter(eps.values()))["n_params"])}
        for m in METRICS:
            row[m] = float(np.nanmean([e[m] for e in eps.values()]))
        # paired differences against the parameter-matched monolith (same staleness, fault and horizon; same
        # budget if this system has several budgets, else against every monolith budget)
        refs = [r for r in groups if r[0] == "monolith" and r[2] == k and r[3] and r[4] == f and r[5] == h
                and (r[1] == b or len({v[0] for v in n_variants[s]}) == 1)]
        row["diffs"] = []
        for ref in refs:
            common = sorted(set(eps) & set(groups[ref]))
            if s == "monolith" or not common:
                continue
            d = {"ref_budget": ref[1], "n_pairs": len(common)}
            for m in ["success", "regret_mean", "mean_speed", "return"]:
                d[m] = paired_boot([eps[i][m] for i in common], [groups[ref][i][m] for i in common], draws, rng)
            row["diffs"].append(d)
        rows.append(row)
    return rows


def fmt_ci(t):
    return "" if np.isnan(t[0]) else f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def markdown(test, rows):
    out = [f"## {test}", "", "| system | budget | stale | contract | fault | H | n | params | success | speed [m/s] | regret | "
           "latency p50/p95 [ms] | miss>100ms | abstain | veto | Δsuccess vs monolith [95% CI] | Δregret vs monolith [95% CI] |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        diffs = r["diffs"] or [{}]
        for i, d in enumerate(diffs):
            tag = f" (vs b={d['ref_budget']})" if d and len(diffs) > 1 else ""
            base = (f"| {r['system']} | {r['budget']} | {r['stale']} | {'on' if r['contract'] else 'off'} | {r['fault'] or '-'} | "
                    f"{r['horizon']} | {r['n']} | {r['n_params']} | {r['success']:.3f} | {r['mean_speed']:.2f} | {r['regret_mean']:.3f} | "
                    f"{r['latency_p50']:.1f}/{r['latency_p95']:.1f} | {r['deadline_miss']:.3f} | {r['abstain_rate']:.3f} | "
                    f"{r['safety_veto_rate']:.3f} |") if i == 0 else "| | | | | | | | | | | | | | | |"
            out.append(base + (f" {fmt_ci(d['success'])}{tag} | {fmt_ci(d['regret_mean'])}{tag} |" if d else " | |"))
    return "\n".join(out) + "\n"


def latex(test, rows):
    out = [r"\begin{table}[t]\centering\small", r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
           r"System & Budget & Stale & Contract & Fault/H & Params & Success & Speed & Regret & $\Delta$Success vs mono \\", r"\midrule"]
    for r in rows:
        d = r["diffs"][0] if r["diffs"] else None
        ci = fmt_ci(d["success"]).replace("[", "$[").replace("]", "]$") if d else "--"
        reg = "--" if np.isnan(r["regret_mean"]) else f"{r['regret_mean']:.2f}"
        fh = (r["fault"] or "-") + ("" if r["horizon"] == 3 else f"/H{r['horizon']}")
        out.append(f"{LABELS.get(r['system'], r['system'])} & {r['budget']} & {r['stale']} & {'on' if r['contract'] else 'off'} & {fh} & "
                   f"{r['n_params']} & {r['success']:.2f} & {r['mean_speed']:.1f} & {reg} & {ci} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", f"\\caption{{{test}: closed-loop results (n={rows[0]['n'] if rows else 0} episodes per cell).}}",
            r"\end{table}"]
    return "\n".join(out).replace("_", r"\_") + "\n"


def figure(test, rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = defaultdict(list)
    for r in rows:
        if r["horizon"] == 3:
            by[(r["system"], r["contract"])].append(r)
    if test == "T6":
        fig, ax = plt.subplots(figsize=(7, 3.2))
        systems = [s for s in ["federation", "federation3", "federation_cv", "fed_learned_ego", "monolith", "monolith_safe"]
                   if any(k[0] == s for k in by)]
        faults = [f for f in FAULT_ORDER if any(r["fault"] == f for r in rows)]
        w = 0.8 / max(1, 2 * len(systems))
        for i, s in enumerate(systems):
            for j, c in enumerate((True, False)):
                rs = {r["fault"]: r for r in by.get((s, c), [])}
                if not rs:
                    continue
                xs = [k + (2 * i + j - len(systems) + 0.5) * w for k in range(len(faults))]
                ax.bar(xs, [rs[f]["success"] if f in rs else 0 for f in faults], width=w, color=COLORS[s],
                       alpha=1.0 if c else 0.45, label=LABELS[s] + ("" if s.startswith("mono") else (" / contract on" if c else " / contract off")))
        ax.set_xticks(range(len(faults)), [f"{f} message" for f in faults])
        ax.set_ylabel("success rate"); ax.set_title("T6 interface faults (10 s window)", loc="left")
        axes = [ax]
    elif test in BUDGET_TESTS:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        budgets = sorted({r["budget"] for r in rows})
        xs = {b: i for i, b in enumerate(budgets)}
        for (s, c), rs in sorted(by.items()):
            rs = sorted(rs, key=lambda r: r["budget"])
            if len(rs) == 1:
                ax.axhline(rs[0]["success"], color=COLORS[s], lw=2, ls="--" if s in ("oracle", "random") else "-", label=LABELS[s])
            else:
                ax.plot([xs[r["budget"]] for r in rs], [r["success"] for r in rs], "-o", color=COLORS[s], lw=2, ms=6, label=LABELS[s])
        ax.set_xticks(range(len(budgets)), [str(b) for b in budgets])
        ax.set_xlabel("shifted-data budget (transitions)")
        ax.set_ylabel("success rate")
        ax.set_title(BUDGET_TESTS[test], loc="left")
        axes = [ax]
    elif test == "T4":
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))
        for (s, c), rs in sorted(by.items()):
            rs = sorted(rs, key=lambda r: str(r["stale"]))
            if s in ("oracle", "random"):
                continue
            ks = [int(str(r["stale"]).split("@")[0]) for r in rs]
            lab = LABELS[s] + ("" if s.startswith("mono") else (" / contract on" if c else " / contract off"))
            ls = "-" if c else ":"
            ax.plot(ks, [r["success"] for r in rs], ls, marker="o", color=COLORS[s], lw=2, ms=6, label=lab)
            ax2.plot(ks, [r["abstain_rate"] for r in rs], ls, marker="o", color=COLORS[s], lw=2, ms=6, label=lab)
        for a in (ax, ax2):
            a.set_xticks(sorted({int(str(r["stale"]).split("@")[0]) for r in rows}))
        ax.set_xlabel("message staleness k (decisions)"); ax.set_ylabel("success rate")
        ax2.set_xlabel("message staleness k (decisions)"); ax2.set_ylabel("abstention rate")
        ax.set_title("T4 coupler stress", loc="left")
        axes = [ax, ax2]
    else:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        rs = sorted([r for r in rows if r["horizon"] == 3], key=lambda r: -r["success"])
        ax.bar([LABELS[r["system"]] for r in rs], [r["success"] for r in rs], color=[COLORS[r["system"]] for r in rs], width=0.6)
        ax.set_ylabel("success rate"); ax.set_title("T1 nominal", loc="left"); ax.tick_params(axis="x", rotation=30)
        axes = [ax]
    for a in axes:
        a.set_ylim(0, 1.02); a.grid(axis="y", color="#e1e0d9", lw=0.8); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.spines["left"].set_color("#c3c2b7"); a.spines["bottom"].set_color("#c3c2b7")
    if axes[-1].get_legend_handles_labels()[0]:
        axes[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--draws", type=int, default=4000)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    for test in ["T1", "T2", "T3", "T4", "T5", "T6"]:
        d = os.path.join(args.results, test)
        if not os.path.isdir(d):
            continue
        groups = load(d)
        if not groups:
            continue
        rows = summarise(groups, args.draws, rng)
        md, tex = markdown(test, rows), latex(test, rows)
        open(os.path.join(d, "table.md"), "w").write(md)
        open(os.path.join(d, "table.tex"), "w").write(tex)
        figure(test, rows, os.path.join(d, "figure.png"))
        print(md)


if __name__ == "__main__":
    main()
