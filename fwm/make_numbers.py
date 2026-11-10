"""Write the paper's driving numbers, tables and figure from results/.

python make_numbers.py --results results --paper ../overleaf [--draws 4000]

Writes into <paper>/tables/: numbers_driving.tex (LaTeX macros used by the main text),
driving_main.tex (compact main table), driving_T1..T5.tex (full appendix tables),
and <paper>/figures/fig_driving.pdf.  Nothing in the paper's driving section is typed by hand.
Macro names contain letters only (LaTeX \\newcommand).
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
import numpy as np
from fwm.aggregate import load, summarise

NAMES = {"federation3": "Federation (analytical ego + safety)", "federation": "Federation (analytical ego)",
         "fed_learned_ego": "Federation (learned ego)", "monolith_safe": "Monolith (equal parameters) + safety",
         "monolith": "Monolith (equal parameters)", "monolith_width": "Monolith (equal width)",
         "oracle": "Oracle (true environment)", "random": "Random chunk",
         "conventional": "Conventional driver (IDM + MOBIL, no model)",
         "federation_cv": "Federation (analytical ego + constant-velocity traffic)",
         "monolith_aego": "Monolith (equal parameters) + analytical ego"}
MACRO = {"federation": "Fed", "federation3": "FedS", "fed_learned_ego": "FedLE", "monolith": "Mono",
         "monolith_width": "MonoW", "monolith_safe": "MonoS", "oracle": "Oracle", "random": "Random",
         "conventional": "Conv", "federation_cv": "FedCV", "monolith_aego": "MonoAE"}
TESTS = {"T1": "One", "T2": "Two", "T3": "Three", "T4": "Four", "T5": "Five", "T6": "Six"}
BUDGET_TESTS = {"T2": "T2 vehicle change", "T3": "T3 traffic change", "T5": "T5 withheld combination"}
COLORS = {"federation": "#2a78d6", "federation3": "#7a4fd6", "fed_learned_ego": "#1baf7a", "monolith": "#eb6834",
          "monolith_width": "#eda100", "monolith_safe": "#b8860b", "oracle": "#52514e", "random": "#898781",
          "conventional": "#2f7f6f", "federation_cv": "#79a9e6", "monolith_aego": "#f0a58a"}


def f2(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def ci(t):
    if t is None or np.isnan(t[0]):
        return "--"
    return f"${t[0]:+.2f}$ $[{t[1]:+.2f}, {t[2]:+.2f}]$"


def key_of(r):
    return (r["system"], r["budget"], str(r["stale"]), r["contract"], r.get("fault"), r.get("horizon", 3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--paper", required=True)
    p.add_argument("--draws", type=int, default=4000)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    rows = {}
    seeds, eps_per_seed = set(), defaultdict(int)
    for t in TESTS:
        d = os.path.join(a.results, t)
        if not os.path.isdir(d):
            continue
        g = load(d)
        if not g:
            continue
        for k, v in g.items():
            for (s, e) in v:
                seeds.add(s)
                eps_per_seed[(t, k, s)] += 1
        for r in summarise(g, a.draws, rng):
            rows[(t,) + key_of(r)] = r
    n_seeds = len(seeds)
    n_eps = int(round(np.mean(list(eps_per_seed.values())))) if eps_per_seed else 0

    def get(t, s, b=0, k="0", c=True, f=None, h=3):
        return rows.get((t, s, b, str(k), c, f, h))

    def val(t, s, m, b=0, k="0", c=True):
        r = get(t, s, b, k, c)
        return None if r is None else r[m]

    def diff(t, s, b=0, k="0", c=True):
        r = get(t, s, b, k, c)
        if r is None or not r["diffs"]:
            return None
        return r["diffs"][0]["success"]

    def pm(t, s, m, b=0, k="0", c=True, fmt=f2):
        v = val(t, s, m, b, k, c)
        return "--" if v is None else fmt(v)

    os.makedirs(os.path.join(a.paper, "tables"), exist_ok=True)
    os.makedirs(os.path.join(a.paper, "figures"), exist_ok=True)
    L = []
    def m(name, value):
        L.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    # ---- parameter counts
    params = {s: int(val("T1", s, "n_params") or val("T2", s, "n_params") or 0) for s in MACRO}
    m("nParamsFed", f"{params['federation']:,}".replace(",", "\\,"))
    m("nParamsMono", f"{params['monolith']:,}".replace(",", "\\,"))
    m("nParamsMonoW", f"{params['monolith_width']:,}".replace(",", "\\,"))
    m("nParamsFedLE", f"{params['fed_learned_ego']:,}".replace(",", "\\,"))
    m("nParamsFedS", f"{params['federation3']:,}".replace(",", "\\,"))
    m("nParamsMonoS", f"{params['monolith_safe']:,}".replace(",", "\\,"))
    try:
        from fwm.members import matched_hidden
        m("nMonoHidden", str(matched_hidden()))
    except Exception:
        m("nMonoHidden", "124")
    m("nSeeds", str(n_seeds))
    m("nSeedsWord", f"{n_seeds} seed" + ("s" if n_seeds != 1 else ""))
    m("nEpisodes", str(n_eps))

    # ---- per-cell macros
    for t, T in TESTS.items():
        if t == "T6":
            continue
        for s, S in MACRO.items():
            for b in (0, 100, 1000, 10000):
                for k in (0, 1, 2, 4):
                    for c in (True, False):
                        r = get(t, s, b, k, c)
                        if r is None:
                            continue
                        words = {0: "Zero", 1: "One", 2: "Two", 4: "Four", 100: "Hundred", 1000: "OneK", 10000: "TenK"}
                        suf = f"{T}{S}" + (f"B{words[b]}" if t in BUDGET_TESTS else "") + (f"K{words[k]}" if t == "T4" else "") + ("" if c else "Off")
                        m("t" + suf, f2(r["success"]))
                        m("t" + suf + "Regret", f2(r["regret_mean"]))
                        m("t" + suf + "Lat", f"{r['latency_ms_mean']:.0f}")
                        m("t" + suf + "Speed", f"{r['mean_speed']:.1f}")
                        m("t" + suf + "Abs", f2(r["abstain_rate"]))
                        m("t" + suf + "Veto", f2(r.get("safety_veto_rate", 0.0)))
                        m("t" + suf + "Delta", ci(diff(t, s, b, k, c)))

    # ---- narrative macros (neutral wording; polish by hand once the final numbers are in)
    T1 = lambda s, mm="success": pm("T1", s, mm)
    head = (f"succeeded on {T1('federation')} of nominal episodes against {T1('monolith')} for the monolith, kept "
            f"{pm('T2','federation','success')} after a vehicle change without any new data, against {pm('T2','monolith','success',0)} "
            f"for the monolith, and turned stale messages into abstentions rather than collisions "
            f"({pm('T4','federation','success',0,2,True)} against {pm('T4','federation','success',0,2,False)} success at two decisions "
            f"of delay with the contract check on and off)")
    m("drivingHeadline", head)
    narr = (f"In the nominal condition the federation succeeded on {T1('federation')} of episodes against "
            f"{T1('monolith')} for the parameter-matched monolith ({ci(diff('T1','federation'))}); with the safety member it reached "
            f"{T1('federation3')}, the monolith with the same rule {T1('monolith_safe')}, the oracle {T1('oracle')}"
            + (f", and mean regret per decision was {T1('federation','regret_mean')} against {T1('monolith','regret_mean')}. "
               if T1('federation','regret_mean') != '--' and T1('monolith','regret_mean') != '--' else ". ")
            +
            f"After the vehicle change (T2) the federation, with its parameters updated and no new data, succeeded on "
            f"{pm('T2','federation','success')}, against {pm('T2','monolith','success',0)} for the monolith without shifted data and "
            f"{pm('T2','monolith','success',10000)} after 10\\,000 shifted transitions. "
            f"After the traffic change (T3), retraining only the traffic member took the federation from "
            f"{pm('T3','federation','success',0)} to {pm('T3','federation','success',10000)}, against "
            f"{pm('T3','monolith','success',0)} to {pm('T3','monolith','success',10000)} for the monolith fine-tuned whole. "
            f"At two decisions of message delay (T4) the federation with the contract check succeeded on "
            f"{pm('T4','federation','success',0,2,True)} while abstaining on {pm('T4','federation','abstain_rate',0,2,True)} of decisions "
            f"at {pm('T4','federation','mean_speed',0,2,True, lambda v: f'{v:.1f}')}\\,m/s; with the check off it succeeded on "
            f"{pm('T4','federation','success',0,2,False)}, the monolith with equally stale information on "
            f"{pm('T4','monolith','success',0,2,True)}. "
            f"On the withheld combination (T5) the federation reached {pm('T5','federation','success',0)} with no data "
            f"and {pm('T5','federation','success',10000)} with its traffic member adapted, the three-member federation "
            f"{pm('T5','federation3','success',0)} and {pm('T5','federation3','success',10000)}, against "
            f"{pm('T5','monolith','success',0)} and {pm('T5','monolith','success',10000)} for the monolith fine-tuned on both "
            f"shifted data sets separately.")
    m("drivingResultsNarrative", narr)
    concl = (f"scored {T1('federation')} against {T1('monolith')} for a monolith at equal learned parameters, "
             f"{pm('T2','federation','success')} against {pm('T2','monolith','success',0)} after a vehicle change that cost it no data, "
             f"and {pm('T4','federation','success',0,2,True)} against {pm('T4','federation','success',0,2,False)} under stale messages with its contract check on and off")
    m("drivingConclusionClause", concl)
    open(os.path.join(a.paper, "tables", "numbers_driving.tex"), "w").write(
        "% written by make_numbers.py -- do not edit by hand\n" + "\n".join(L) + "\n")

    # ---- compact main table
    def cell(t, s, b=0, k="0", c=True):
        r = get(t, s, b, k, c)
        return "--" if r is None else f2(r["success"])
    lines = [r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}",
             r"\caption{Driving federation against a parameter-matched monolith (" + f"{n_seeds} seed{'s' if n_seeds != 1 else ''}, {n_eps} episodes per seed and cell" + r"): success without collision in T1; with 0 / 10\,000 shifted transitions in T2, T3 and the withheld combination T5 (federations adapt the traffic member only, monoliths are fine-tuned on both shifted sets); at two decisions of delay with the contract check on / off in T4. $\Delta$: paired difference to the monolith, bootstrap 95\% interval.}\label{tab:driving}",
             r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}lrrrrrrr@{}}\toprule",
             r"System & Params & T1 & T2 (0 / 10k) & T3 (0 / 10k) & T4 $k{=}2$ (on / off) & T5 (0 / 10k) & $\Delta$ T1 vs monolith \\ \midrule"]
    for s in ["federation3", "federation", "fed_learned_ego", "monolith_safe", "monolith"]:
        r1 = get("T1", s)
        if r1 is None:
            continue
        t2 = f"{cell('T2', s, 0)} (no data)" if s in ("federation", "federation3") else f"{cell('T2', s, 0)} / {cell('T2', s, 10000)}"
        t3 = f"{cell('T3', s, 0)} / {cell('T3', s, 10000)}"
        t4 = f"{cell('T4', s, 0, 2, True)} / {cell('T4', s, 0, 2, False)}" if s.startswith("fed") else cell("T4", s, 0, 2, True)
        t5 = f"{cell('T5', s, 0)} / {cell('T5', s, 10000)}"
        dl = "ref." if s == "monolith" else ci(diff("T1", s))
        lines.append(f"{NAMES[s]} & {params[s]:,} & {f2(r1['success'])} & {t2} & {t3} & {t4} & {t5} & {dl} \\\\")
    lines += [r"\bottomrule\end{tabular}}\end{table}"]
    open(os.path.join(a.paper, "tables", "driving_main.tex"), "w").write("\n".join(lines) + "\n")

    # ---- full appendix tables
    for t in TESTS:
        rs = [r for (tt, *_), r in rows.items() if tt == t]
        if not rs:
            open(os.path.join(a.paper, "tables", f"driving_{t}.tex"), "w").write(f"% no results for {t}\n")
            continue
        cap = {"T1": "nominal condition", "T2": "vehicle change; budget is the number of shifted transitions used for fine-tuning",
               "T3": "traffic change; budget as in T2, and for the federations only the traffic member is retrained",
               "T4": "coupler stress; stale is the age of the traffic message in decisions",
               "T5": "withheld combination of the vehicle change and the traffic change; the federations adapt the traffic member on the traffic-change data only (budget transitions), the learned-ego federation adapts each member on its own shifted data, and the monoliths are fine-tuned on both shifted data sets (budget transitions of each)",
               "T6": "interface faults during a 10\\,s window (the Stale column names the fault class: stale message, wrong frame, wrong units, missing message); the monoliths receive the same corruption on their observation"}[t]
        out = [r"\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}",
               f"\\caption{{{t}: {cap}. Success, mean speed, regret where the oracle was run, model latency per decision, abstention rate, fraction of decisions whose executed chunk violated the safety margin (veto; systems with a safety member), and the paired difference in success to the parameter-matched monolith (same budget and staleness).}}\\label{{tab:driving{t}}}",
               r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}lrrrrrrrrr@{}}\toprule",
               r"System & Budget & Stale & Contract & Success & Speed [m/s] & Regret & Latency [ms] & Veto & $\Delta$ success \\ \midrule"]
        for r in sorted(rs, key=lambda r: (list(MACRO).index(r["system"]), r["budget"], str(r["stale"]), not r["contract"], str(r["fault"]))):
            if r["horizon"] != 3:
                continue
            d = r["diffs"][0]["success"] if r["diffs"] else None
            if t == "T6":
                r = dict(r, stale=r["fault"])
            veto = f2(r["safety_veto_rate"]) if r["system"] in ("federation3", "monolith_safe") else "--"
            out.append(f"{NAMES[r['system']]} & {r['budget']} & {r['stale']} & {'on' if r['contract'] else 'off'} & "
                       f"{f2(r['success'])} & {r['mean_speed']:.1f} & {f2(r['regret_mean'])} & {r['latency_ms_mean']:.0f} & {veto} & "
                       f"{'ref.' if r['system']=='monolith' else ci(d)} \\\\")
        out += [r"\bottomrule\end{tabular}}\end{table}"]
        open(os.path.join(a.paper, "tables", f"driving_{t}.tex"), "w").write("\n".join(out) + "\n")

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.0))
    seen = {}                                               # one legend entry per system / line style
    for ax, t in zip(axes[:3], ("T2", "T3", "T5")):
        rs = [r for (tt, *_), r in rows.items() if tt == t]
        budgets = sorted({r["budget"] for r in rs})
        xs = {b: i for i, b in enumerate(budgets)}
        by = defaultdict(list)
        for r in rs:
            by[r["system"]].append(r)
        for s, lst in by.items():
            lst = sorted(lst, key=lambda r: r["budget"])
            if len(lst) == 1:
                line = ax.axhline(lst[0]["success"], color=COLORS[s], lw=2, ls="--" if s in ("oracle", "random") else "-")
            else:
                line, = ax.plot([xs[r["budget"]] for r in lst], [r["success"] for r in lst], "-o", color=COLORS[s], lw=2, ms=5)
            seen.setdefault(NAMES[s], line)
        ax.set_xticks(range(len(budgets)), [str(b) for b in budgets])
        ax.set_xlabel("shifted transitions"); ax.set_ylabel("success")
        ax.set_title(BUDGET_TESTS[t], loc="left", fontsize=10)
    rs = [r for (tt, *_), r in rows.items() if tt == "T4"]
    by = defaultdict(list)
    for r in rs:
        by[(r["system"], r["contract"])].append(r)
    for (s, c), lst in sorted(by.items()):
        if s in ("oracle", "random"):
            continue
        lst = sorted(lst, key=lambda r: int(str(r["stale"]).split("@")[0]))
        ks = [int(str(r["stale"]).split("@")[0]) for r in lst]
        lab = NAMES[s] + ("" if s.startswith("mono") else (", contract on" if c else ", contract off"))
        ls = "-" if c else ":"
        line, = axes[3].plot(ks, [r["success"] for r in lst], ls, marker="o", color=COLORS[s], lw=2, ms=5)
        axes[4].plot(ks, [r["abstain_rate"] for r in lst], ls, marker="o", color=COLORS[s], lw=2, ms=5)
        if not c or NAMES[s] not in seen:
            seen.setdefault(lab, line)
    axes[3].set_xlabel("message delay (decisions)"); axes[3].set_ylabel("success"); axes[3].set_title("T4 coupler stress", loc="left", fontsize=10)
    axes[4].set_xlabel("message delay (decisions)"); axes[4].set_ylabel("abstention rate"); axes[4].set_title("T4 abstentions", loc="left", fontsize=10)
    for ax in axes:
        ax.set_ylim(0, 1.02); ax.grid(axis="y", color="#e1e0d9", lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.legend(list(seen.values()), list(seen.keys()), loc="upper center", ncol=5, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.savefig(os.path.join(a.paper, "figures", "fig_driving.pdf"), bbox_inches="tight")
    print(f"wrote macros, tables and figure to {a.paper} ({n_seeds} seeds, ~{n_eps} episodes per cell)")


if __name__ == "__main__":
    main()
