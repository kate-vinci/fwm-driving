"""Draw media/episode_federation3.gif: one closed-loop episode of the three-member federation, one frame
per decision, with the predicted joint future of the executed chunk and the safety margin.

python media/make_episode_gif.py [--model-dir models] [--seed 1000011] [--fps 6] [--decisions 60]
(needs a trained nominal traffic member, e.g. after prepare.py)
"""
import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image
from fwm.common import (make_env, reset_env, observe, split, LANE_W, N_LANES, VEH_LENGTH, VEH_WIDTH, EGO_PARAMS,
                    SAFETY_PARAMS, ACTIONS)
from fwm.members import AnalyticEgo, AgentsMember, SafetyMember, Federation
from fwm.nn_backend import MLP
from fwm.planner import make_chunks, score_chunk_detail

FED, SAFE, INK, GREY, ROAD, LINE = "#2a78d6", "#b8860b", "#1d1c19", "#9a9a9a", "#2e2f33", "#d9d9d9"

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default="models")
ap.add_argument("--seed", type=int, default=1000011)
ap.add_argument("--fps", type=int, default=6)
ap.add_argument("--decisions", type=int, default=60)
ap.add_argument("--width", type=int, default=1100)
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))

env = make_env("nominal")
state, _ = reset_env(env, a.seed, "nominal")
fed = Federation(AnalyticEgo(EGO_PARAMS["nominal"]),
                 AgentsMember(MLP.load(os.path.join(a.model_dir, "agents_nominal_s0.pt"))),
                 safety=SafetyMember(SAFETY_PARAMS["nominal"]), name="federation3")
fed.begin_episode(0)
rng = np.random.default_rng([a.seed, 99])


def car(ax, x, y, h, color, alpha=1.0, lw=1.2, fill=True, z=3):
    t = matplotlib.transforms.Affine2D().rotate_around(x, y, h) + ax.transData
    ax.add_patch(FancyBboxPatch((x - VEH_LENGTH / 2, y - VEH_WIDTH / 2), VEH_LENGTH, VEH_WIDTH,
                                boxstyle="round,pad=0,rounding_size=0.5", fc=color if fill else "none",
                                ec=color, lw=lw, alpha=alpha, transform=t, zorder=z))


def draw(state, states_pred, chunk, t, speed, vetoed):
    ego, oth = split(state)
    fig, ax = plt.subplots(figsize=(11, 3.3), dpi=100)
    x0 = ego[0]; xmin, xmax = x0 - 45, x0 + 115
    ax.set_xlim(xmin, xmax); ax.set_ylim(-LANE_W / 2 - 0.8, (N_LANES - 0.5) * LANE_W + 0.8)
    ax.set_facecolor(ROAD); fig.patch.set_facecolor("white")
    for i in range(N_LANES + 1):
        y = -LANE_W / 2 + i * LANE_W
        ax.plot([xmin, xmax], [y, y], color=LINE, lw=2 if i in (0, N_LANES) else 1,
                ls="-" if i in (0, N_LANES) else (0, (6, 6)))
    for o in oth:
        if o[0] > 0.5: car(ax, o[1], o[2], o[5], GREY)
    car(ax, ego[0], ego[1], ego[4], FED, z=5)
    if states_pred:
        for k, s in enumerate(states_pred):
            e, o2 = split(s); al = 0.75 - 0.2 * k
            car(ax, e[0], e[1], e[4], FED, alpha=al, fill=False, lw=1.6, z=4)
            for o in o2:
                if o[0] > 0.5: car(ax, o[1], o[2], o[5], "#e8e8e8", alpha=al * 0.9, fill=False, lw=1.1, z=4)
        e_last, o_last = split(states_pred[-1])
        lead, gap = None, np.inf
        for o in o_last:
            if o[0] > 0.5 and abs(o[2] - e_last[1]) < LANE_W / 2 and o[1] > e_last[0]:
                g = o[1] - e_last[0] - VEH_LENGTH
                if g < gap: lead, gap = o, g
        if lead is not None:
            dmin = fed.safety.d_min(float(e_last[2]), float(lead[3]))
            xf = e_last[0] + VEH_LENGTH / 2
            ax.add_patch(Rectangle((xf, e_last[1] - VEH_WIDTH / 2 - 0.35), max(dmin, 0.5), VEH_WIDTH + 0.7,
                                   fc=SAFE, ec="none", alpha=0.28, zorder=2))
            below = e_last[1] > LANE_W      # label under the band, or above it in the bottom lane
            ax.text(xf + max(dmin, 0.5) / 2, e_last[1] - LANE_W * 0.95 if below else e_last[1] + LANE_W * 0.95,
                    f"$d_{{\\min}}$ = {dmin:.0f} m, gap {gap:.0f} m",
                    color=SAFE, ha="center", va="top" if below else "bottom", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc=ROAD, ec="none", alpha=0.9))
    ax.text(xmin + 2, (N_LANES - 0.5) * LANE_W - 0.3,
            f"decision {t + 1:2d}/{a.decisions} · executed chunk: {' · '.join(ACTIONS[c] for c in chunk)} · "
            f"speed {speed:.1f} m/s" + ("  · least-violating chunk" if vetoed else ""),
            color="#f2f2f2", fontsize=9, va="top", family="monospace")
    ax.text(xmin + 2, -LANE_W / 2 + 0.6, "federation3: ego member (blue) · traffic member (grey outlines) · safety member (gold)",
            color="#f2f2f2", fontsize=8.5, va="bottom", alpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return Image.fromarray(img)


frames, crashed = [], False
for t in range(a.decisions):
    fed.observe(state, t)
    chunks = make_chunks(rng)
    scored = [score_chunk_detail(fed, state, c) for c in chunks]
    scores = [s[0] for s in scored]
    choice = int(np.argmax(scores))
    chunk = chunks[choice]
    pred = fed.rollout(state, chunk)
    vetoed = scored[choice][1] > 0
    ego = split(state)[0]
    frames.append(draw(state, pred, chunk, t, float(np.hypot(ego[2], ego[3])), vetoed))
    _, r, term, trunc, info = env.step(chunk[0])
    state, _ = observe(env)
    if info["crashed"]:
        crashed = True
        frames.append(draw(state, None, chunk, t, float(info["speed"]), False))
        break
    if term or trunc:
        break

w = a.width; h = int(frames[0].height * w / frames[0].width)
frames = [f.resize((w, h), Image.LANCZOS).quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
out = os.path.join(here, "episode_federation3.gif")
frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
print(f"wrote {out}: {len(frames)} frames, crashed={crashed}, {os.path.getsize(out) / 1e6:.1f} MB")
