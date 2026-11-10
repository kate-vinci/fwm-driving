"""Draw media/hero_federation.png: one real state of the simulation with the federation's predicted futures.

python media/make_hero.py [--model-dir models]   (needs a trained nominal traffic member, e.g. after prepare.py)
"""
import os, sys, copy, argparse, numpy as np
_ap = argparse.ArgumentParser(); _ap.add_argument("--model-dir", default="models"); MODEL_DIR = _ap.parse_args().model_dir
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from fwm.common import (make_env, reset_env, observe, split, LANE_W, N_LANES, VEH_LENGTH, VEH_WIDTH, EGO_PARAMS,
                    SAFETY_PARAMS, FASTER, IDLE)
from fwm.members import AnalyticEgo, AgentsMember, SafetyMember, Federation
from fwm.nn_backend import MLP

FED, MONO, SAFE, INK, GREY, ROAD, LINE = "#2a78d6", "#eb6834", "#b8860b", "#1d1c19", "#9a9a9a", "#2e2f33", "#d9d9d9"
env = make_env("nominal")
state, _ = reset_env(env, 1_000_011, "nominal")
for _ in range(8):
    env.step(IDLE); state, _ = observe(env)
ego, oth = split(state)
fed = Federation(AnalyticEgo(EGO_PARAMS["nominal"]), AgentsMember(MLP.load(os.path.join(MODEL_DIR, "agents_nominal_s0.pt"))),
                 safety=SafetyMember(SAFETY_PARAMS["nominal"]))
fed.begin_episode(0); fed.observe(state, 0)
chunk = [FASTER, FASTER, FASTER]
states = fed.rollout(state, chunk)

fig, ax = plt.subplots(figsize=(14, 4.2), dpi=160)
x0 = ego[0]
xmin, xmax = x0 - 45, x0 + 115
ax.set_xlim(xmin, xmax); ax.set_ylim(-LANE_W/2 - 0.8, (N_LANES - 0.5) * LANE_W + 0.8)
ax.set_facecolor(ROAD); fig.patch.set_facecolor("white")
for i in range(N_LANES + 1):
    y = -LANE_W/2 + i * LANE_W
    if i in (0, N_LANES): ax.plot([xmin, xmax], [y, y], color=LINE, lw=2)
    else: ax.plot([xmin, xmax], [y, y], color=LINE, lw=1, ls=(0, (6, 6)))

def car(x, y, h, color, alpha=1.0, lw=1.2, fill=True, z=3):
    t = matplotlib.transforms.Affine2D().rotate_around(x, y, h) + ax.transData
    r = FancyBboxPatch((x - VEH_LENGTH/2, y - VEH_WIDTH/2), VEH_LENGTH, VEH_WIDTH, boxstyle="round,pad=0,rounding_size=0.5",
                       fc=color if fill else "none", ec=color, lw=lw, alpha=alpha, transform=t, zorder=z)
    ax.add_patch(r)

for o in oth:
    if o[0] > 0.5: car(o[1], o[2], o[5], GREY)
car(ego[0], ego[1], ego[4], FED, z=5)
# predicted futures
for k, s in enumerate(states):
    e, o2 = split(s); a = 0.75 - 0.2 * k
    car(e[0], e[1], e[4], FED, alpha=a, fill=False, lw=1.6, z=4)
    for o in o2:
        if o[0] > 0.5: car(o[1], o[2], o[5], "#e8e8e8", alpha=a * 0.9, fill=False, lw=1.2, z=4)
# safety margin from the last predicted ego state to its lead
e_last, o_last = split(states[-1]); sm = fed.safety
lead, gap = None, np.inf
for o in o_last:
    if o[0] > 0.5 and abs(o[2] - e_last[1]) < LANE_W/2 and o[1] > e_last[0]:
        g = o[1] - e_last[0] - VEH_LENGTH
        if g < gap: lead, gap = o, g
dmin = sm.d_min(float(e_last[2]), float(lead[3])) if lead is not None else 0
xf = e_last[0] + VEH_LENGTH/2
ax.add_patch(Rectangle((xf, e_last[1] - VEH_WIDTH/2 - 0.35), max(dmin, 0.5), VEH_WIDTH + 0.7, fc=SAFE, ec="none", alpha=0.28, zorder=2))
ax.annotate("", xy=(xf + max(dmin, 0.5), e_last[1] - LANE_W*0.52), xytext=(xf, e_last[1] - LANE_W*0.52),
            arrowprops=dict(arrowstyle="<->", color=SAFE, lw=1.4))
ax.text(xf + max(dmin, 0.5)/2, e_last[1] - LANE_W*0.95, f"safety member (analytic, 0 parameters): $d_{{\\min}}$ = {dmin:.1f} m to the predicted lead", color=SAFE, ha="center", va="top", fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc=ROAD, ec="none", alpha=0.9))
# labels
ax.annotate("ego member (analytic, 0 parameters)\npredicted ego positions, 3 decisions", xy=(states[1][0], ego[1] - VEH_WIDTH/2), xytext=(x0 - 40, -LANE_W/2 + 0.9),
            color=FED, fontsize=9, fontweight="bold", arrowprops=dict(arrowstyle="-", color=FED, lw=0.9), va="bottom")
lab_o = next(o for o in oth if o[0] > 0.5 and o[1] > ego[0] + 10 and o[2] > 2 * LANE_W)
ax.annotate("traffic member (learned, 27 939 parameters)\npredicted neighbour positions", xy=(lab_o[1], lab_o[2] + VEH_WIDTH/2), xytext=(x0 - 40, (N_LANES - 0.5) * LANE_W - 2.2),
            color="#f2f2f2", fontsize=9, fontweight="bold", arrowprops=dict(arrowstyle="-", color="#f2f2f2", lw=0.9), va="top")
ax.text(xmin + 2, (N_LANES - 0.5) * LANE_W - 0.3, "coupler: traffic message {timestamp, frame, horizon, validity} → ego frame → joint future", color="#f2f2f2", fontsize=8.5, va="top", alpha=0.95)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title("One decision in highway-env: the federation predicts a candidate chunk (FASTER × 3), then the safety member checks the joint future",
             fontsize=10.5, loc="left", color=INK, pad=8)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hero_federation.png"), dpi=160, facecolor="white")
print(f"wrote media/hero_federation.png (d_min = {dmin:.1f} m, gap = {gap:.1f} m)")
