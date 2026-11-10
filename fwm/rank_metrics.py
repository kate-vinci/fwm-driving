"""Decision-level agreement between each system's predicted chunk scores and the oracle's true scores,
from the per-decision logs in results/<test>/*.jsonl (runs with oracle scoring on).

python rank_metrics.py [--results results] [--tests T1,T2,T3,T5]

Per system and variant (budget / stale / contract / fault / horizon), over all decisions where both score
vectors exist:
  spearman          rank correlation between predicted and true scores of the nine chunks (mean over decisions)
  top1_agree        the chunk the system chose is the oracle's best (ties counted as agreement)
  infeasible_prec   of the chunks the system predicts as colliding (score < -50), the fraction that truly collide
  infeasible_rec    of the chunks that truly collide, the fraction the system predicts as colliding
  unsafe_accept     fraction of chunks predicted safe that truly collide (false clearance)
  chosen_collides   fraction of decisions whose executed chunk truly collides
Writes results/<test>/rank.json and rank.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
import numpy as np

THRESH = -50.0   # a chunk score below this carries the collision (or safety) penalty


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _rank(x):
    x = np.asarray(x, float)
    order = x.argsort()
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ranks for ties
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--tests", default="T1,T2,T3,T4,T5,T6")
    args = p.parse_args()
    for test in args.tests.split(","):
        d = os.path.join(args.results, test)
        if not os.path.isdir(d):
            continue
        acc = defaultdict(lambda: defaultdict(list))
        for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if r["system"] in ("oracle", "random", "conventional"):
                        continue
                    key = (r["system"], r["budget"], r["stale"], r["contract"], r.get("fault"), r.get("horizon", 3))
                    dec = r["decisions"]
                    for true, pred, chosen in zip(dec["true_scores"], dec["pred_scores"], dec["chosen"]):
                        if true is None or pred is None or any(x is None for x in pred) or chosen is None:
                            continue
                        true, pred = np.asarray(true, float), np.asarray(pred, float)
                        a = acc[key]
                        a["spearman"].append(spearman(pred, true))
                        a["top1_agree"].append(float(true[chosen] >= true.max() - 1e-9))
                        pi, ti = pred < THRESH, true < THRESH
                        a["infeasible_prec"].append(float(ti[pi].mean()) if pi.any() else np.nan)
                        a["infeasible_rec"].append(float(pi[ti].mean()) if ti.any() else np.nan)
                        a["unsafe_accept"].append(float(ti[~pi].mean()) if (~pi).any() else np.nan)
                        a["chosen_collides"].append(float(ti[chosen]))
        rows = []
        for key in sorted(acc, key=lambda t: (t[0], t[1], str(t[2]), not t[3], str(t[4]), t[5])):
            s, b, k, c, f, h = key
            row = {"system": s, "budget": b, "stale": k, "contract": c, "fault": f, "horizon": h,
                   "decisions": len(acc[key]["spearman"])}
            for m, v in acc[key].items():
                row[m] = float(np.nanmean(v))
            rows.append(row)
        if not rows:
            continue
        with open(os.path.join(d, "rank.json"), "w") as f:
            json.dump(rows, f, indent=1)
        md = [f"## {test}: decision-level agreement with the oracle", "",
              "| system | budget | stale | contract | fault | H | decisions | Spearman | top-1 agree | infeasible precision | "
              "infeasible recall | unsafe accept | chosen collides |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            md.append(f"| {r['system']} | {r['budget']} | {r['stale']} | {'on' if r['contract'] else 'off'} | {r['fault'] or '-'} | "
                      f"{r['horizon']} | {r['decisions']} | {r['spearman']:.3f} | {r['top1_agree']:.3f} | {r['infeasible_prec']:.3f} | "
                      f"{r['infeasible_rec']:.3f} | {r['unsafe_accept']:.3f} | {r['chosen_collides']:.3f} |")
        open(os.path.join(d, "rank.md"), "w").write("\n".join(md) + "\n")
        print("\n".join(md))


if __name__ == "__main__":
    main()
