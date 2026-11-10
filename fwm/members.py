"""World-model members, the coupler (with typed message contract), and the monolith.

Every system exposes:  begin_episode(), observe(state, t), rollout(state, chunk) -> list[state] | None
(None = the coupler refused a message that violates its contract -> the planner abstains), and
safety_penalty(states) -> float (0 unless the system carries a safety-margin member).
"""
from __future__ import annotations

import numpy as np
from fwm.common import (EGO_DIM, K_OTHERS, OTH_DIM, N_ACTIONS, N_LANES, LANE_W, VEH_LENGTH, FRAMES, DT_SIM,
                    TARGET_SPEEDS, FASTER, SLOWER, IDLE, LANE_LEFT, LANE_RIGHT, EgoParams, SafetyParams,
                    ego_step, next_targets, split, join)

DT_DECISION = FRAMES * DT_SIM   # 0.5 s
from fwm.nn_backend import MLP, hidden_for_params

H = 3  # planning horizon (decisions); run.py --horizon changes it through set_horizon()


def set_horizon(h):
    global H
    H = int(h)


# ------------------------------------------------------------------ feature / target builders
def onehot(a):
    v = np.zeros(N_ACTIONS)
    v[int(a)] = 1.0
    return v


def rel_others(ego, others):
    """others expressed relative to the ego position (velocities and headings stay absolute)."""
    r = others.copy()
    r[:, 1] -= ego[0]
    r[:, 2] -= ego[1]
    return r


def others_feat(ego, rel):
    """(K,7): relative row + the vehicle's lateral offset from its nearest lane centre (road geometry
    is known to every system; it disambiguates a lane change in progress from lane keeping)."""
    off = (ego[1] + rel[:, 2] + LANE_W / 2) % LANE_W - LANE_W / 2
    return np.concatenate([rel, (off * (rel[:, 0] > 0.5))[:, None]], axis=1)


def ego_phys(ego):
    return np.array([ego[1], ego[2], ego[3], ego[4], ego[5], ego[6]])      # y, vx, vy, h, v_target, lane_target


def agents_x(ego, rel):
    return np.concatenate([[ego[2], ego[3], ego[4], ego[1]], others_feat(ego, rel).reshape(-1)])   # 4 + K*7 = 53


def ego_x(ego, a):
    return np.concatenate([ego_phys(ego), onehot(a)])                           # 6 + 5 = 11


def mono_x(ego, rel, a):
    return np.concatenate([ego_phys(ego), others_feat(ego, rel).reshape(-1), onehot(a)])   # 6 + K*7 + 5 = 60


def others_delta(others, others_next):
    """world-frame deltas (K,5) of [x,y,vx,vy,heading] and a (K,5) presence mask"""
    d = others_next[:, 1:6] - others[:, 1:6]
    m = np.repeat((others[:, :1] > 0.5) & (others_next[:, :1] > 0.5), 5, axis=1).astype(float)
    return d * m, m


def build_xy(kind, S, A, S2):
    """Training arrays for kind in {agents, ego, mono} from transition arrays S, A, S2."""
    X, Y, M = [], [], []
    for s, a, s2 in zip(S, A, S2):
        ego, oth = split(s)
        ego2, oth2 = split(s2)
        rel = rel_others(ego, oth)
        d_oth, m_oth = others_delta(oth, oth2)
        d_ego = ego2[:5] - ego[:5]
        if kind == "agents":
            X.append(agents_x(ego, rel)); Y.append(d_oth.reshape(-1)); M.append(m_oth.reshape(-1))
        elif kind == "ego":
            X.append(ego_x(ego, a)); Y.append(d_ego); M.append(np.ones(5))
        else:
            X.append(mono_x(ego, rel, a)); Y.append(np.concatenate([d_ego, d_oth.reshape(-1)]))
            M.append(np.concatenate([np.ones(5), m_oth.reshape(-1)]))
    return np.array(X), np.array(Y), np.array(M)


OF = OTH_DIM + 1   # features per other vehicle
DIMS = {"agents": (4 + K_OTHERS * OF, K_OTHERS * 5), "ego": (6 + N_ACTIONS, 5),
        "mono": (6 + K_OTHERS * OF + N_ACTIONS, 5 + K_OTHERS * 5)}


def new_mlp(kind, seed, hidden=128):
    i, o = DIMS[kind]
    return MLP(i, o, hidden=hidden, seed=seed)


def matched_hidden():
    """hidden width of the monolith that matches the federation's learned parameter count (agents only)."""
    return hidden_for_params(new_mlp("agents", 0).n_params, *DIMS["mono"])


# ------------------------------------------------------------------ members
class AnalyticEgo:
    """Zero-parameter ego member: highway-env's own MDPVehicle controller + bicycle kinematics."""
    n_params = 0

    def __init__(self, params: EgoParams):
        self.params = params

    def predict(self, ego, a):
        return ego_step(ego, a, self.params)


class LearnedEgo:
    def __init__(self, mlp: MLP):
        self.mlp, self.n_params = mlp, mlp.n_params

    def predict(self, ego, a):
        nxt = ego.copy()
        nxt[:5] += self.mlp.predict(ego_x(ego, a))
        nxt[5], nxt[6] = next_targets(ego, a)
        return nxt


class AgentsMember:
    """Learned member for the other vehicles; works in the ego-relative frame, outputs world deltas."""
    def __init__(self, mlp: MLP):
        self.mlp, self.n_params = mlp, mlp.n_params

    def predict(self, ego, rel):
        d = self.mlp.predict(agents_x(ego, rel)).reshape(K_OTHERS, 5)
        return d * (rel[:, :1] > 0.5)


class ConstantVelocityMember:
    """Zero-parameter traffic member: every neighbour keeps its velocity and heading for one decision.
    Replacing the learned traffic member by this one measures what the learned member contributes."""
    n_params = 0

    def predict(self, ego, rel):
        d = np.zeros((K_OTHERS, 5))
        d[:, 0] = rel[:, 3] * DT_DECISION
        d[:, 1] = rel[:, 4] * DT_DECISION
        return d * (rel[:, :1] > 0.5)


class SafetyMember:
    """Zero-parameter longitudinal safety-margin member: an RSS-style minimum following distance
    evaluated on the joint predicted future (ego from the ego member, traffic from the traffic member,
    assembled by the coupler; or the monolith's own joint prediction).

    For a predicted state, the lead vehicle is the present other vehicle in the ego's lane
    (|y_other - y_ego| < LANE_W / 2) that is ahead (x_other > x_ego) with the smallest gap,
    gap = x_other - x_ego - VEH_LENGTH. With v_r the ego forward speed and v_f the lead's forward
    speed, the minimum safe distance is

        d_min = max(0, v_r rho + a_accel rho^2 / 2 + (v_r + rho a_accel)^2 / (2 a_brake_min)
                       - v_f^2 / (2 a_brake_front)),

    the distance the ego needs if it accelerates at a_accel for the response time rho and then brakes at
    a_brake_min while the lead brakes at a_brake_front. The violation at a state is max(0, d_min - gap);
    violation(states) is the largest violation over the predicted states (0 without a lead vehicle).
    """
    # Shalev-Shwartz, Shammah, Shashua, On a formal model of safe and scalable self-driving cars,
    # arXiv:1708.06374, 2017 (safe longitudinal distance).
    n_params = 0

    def __init__(self, params: SafetyParams):
        self.params = params

    def d_min(self, v_r, v_f):
        p = self.params
        d = (v_r * p.rho + 0.5 * p.a_accel * p.rho ** 2 + (v_r + p.rho * p.a_accel) ** 2 / (2 * p.a_brake_min)
             - v_f ** 2 / (2 * p.a_brake_front))
        return max(0.0, float(d))

    def violation_at(self, state):
        ego, others = split(state)
        v_r = float(ego[2])                               # forward speed (vx) on the straight road
        lead, gap = None, np.inf
        for o in others:
            if o[0] > 0.5 and abs(o[2] - ego[1]) < LANE_W / 2 and o[1] > ego[0]:
                g = o[1] - ego[0] - VEH_LENGTH
                if g < gap:
                    lead, gap = o, g
        if lead is None:
            return 0.0
        v_f = float(lead[3])
        return max(0.0, self.d_min(v_r, v_f) - gap)

    def violation(self, states):
        return max((self.violation_at(s) for s in states), default=0.0)


class Message(dict):
    """Typed member message: {timestamp, frame, horizon, validity, units, payload}."""
    def valid_at(self, t, horizon):
        return (self["timestamp"] == t and self["frame"] == ("ego", t)
                and self["horizon"] >= horizon and self["validity"][0] <= t < self["validity"][1]
                and self["units"] == "SI")


FAULTS = ("stale", "frame", "unit", "missing")
FAULT_WINDOW = (10, 30)   # decisions [10, 30): a 10 s fault window inside the 30 s episode (T6)


def in_window(t, window):
    return window is not None and window[0] <= t < window[1]


# ------------------------------------------------------------------ systems
class SafetyMixin:
    """Optional safety-margin member shared by the systems: safety_penalty(states) is the margin violation
    of a predicted joint future (0 without a safety member)."""
    safety = None

    def safety_penalty(self, states):
        return 0.0 if self.safety is None else self.safety.violation(states)


class Federation(SafetyMixin):
    """EGO (+) AGENTS composed through an explicit coupler, with an optional SAFETY member that reads the
    assembled joint future.

    delay k (T4): the agents member computes its message from the observation k decisions old.
    contract: whether the coupler checks the typed contract (timestamp / frame / horizon / validity / units).
    fault (T6): one of FAULTS, injected on every decision inside fault_window:
        stale    the message is computed from the observation two decisions old
        frame    the payload is expressed in world coordinates and the message says so
        unit     the payload's velocities are in km/h and the message says so
        missing  no message arrives; with the contract off the coupler silently reuses the last message it
                 accepted (which grows older for the length of the window)
    """
    kind = "model"

    def __init__(self, ego_member, agents_member, contract=True, delay=0, stale_prob=1.0, name="federation",
                 safety=None, fault=None, fault_window=FAULT_WINDOW):
        self.ego_member, self.agents_member, self.safety = ego_member, agents_member, safety
        self.contract, self.delay, self.stale_prob, self.name = contract, int(delay), stale_prob, name
        self.fault, self.fault_window = fault, fault_window
        self.n_params = ego_member.n_params + agents_member.n_params + (0 if safety is None else safety.n_params)
        self.buffer, self.t, self.abstentions, self.violations = [], 0, 0, 0
        self.last_accepted = None

    def begin_episode(self, ep_seed=0):
        self.buffer, self.t, self.abstentions, self.violations = [], 0, 0, 0
        self.last_accepted = None
        self.rng = np.random.default_rng([ep_seed, 4242])

    def observe(self, state, t):
        self.buffer.append((t, state))
        self.t = t
        self.delayed_now = self.delay if self.rng.random() < self.stale_prob else 0
        self.fault_now = self.fault if in_window(t, self.fault_window) else None
        if self.fault_now == "stale":
            self.delayed_now = 2

    def agents_message(self):
        """The traffic member's message for the current decision (None when it is missing)."""
        if self.fault_now == "missing":
            return None
        ts, src = self.buffer[max(len(self.buffer) - 1 - self.delayed_now, 0)]
        ego_s, oth_s = split(src)
        msg = Message(timestamp=ts, frame=("ego", ts), horizon=H, validity=(ts, ts + 1), units="SI",
                      payload=rel_others(ego_s, oth_s))
        if self.fault_now == "frame":                   # world coordinates, declared as such
            msg["frame"] = ("world", ts)
            msg["payload"] = oth_s.copy()
        elif self.fault_now == "unit":                  # km/h, declared as such
            msg["units"] = "km/h"
            msg["payload"] = msg["payload"].copy()
            msg["payload"][:, 3:5] *= 3.6
        return msg

    def rollout(self, state, chunk):
        ego, _ = split(state)
        msg = self.agents_message()
        if msg is None or not msg.valid_at(self.t, len(chunk)):
            self.violations += 1
            if self.contract:                       # contract ON: refuse the message -> planner abstains
                self.abstentions += 1
                return None
            if msg is None:                         # contract OFF, missing message: silently reuse the last one
                msg = self.last_accepted
                if msg is None:
                    return None
        else:
            self.last_accepted = msg
        rel = msg["payload"]                        # coupler assumes the frame is the current ego frame, SI units
        states = []
        for a in chunk:
            ego_next = self.ego_member.predict(ego, a)
            d = self.agents_member.predict(ego, rel)
            rel_next = rel.copy()
            rel_next[:, 1:6] += d
            rel_next[:, 1] -= ego_next[0] - ego[0]  # re-express in the predicted next ego frame
            rel_next[:, 2] -= ego_next[1] - ego[1]
            others_abs = rel_next.copy()
            others_abs[:, 1] += ego_next[0]
            others_abs[:, 2] += ego_next[1]
            others_abs *= (rel[:, :1] > 0.5)
            states.append(join(ego_next, others_abs))
            ego, rel = ego_next, rel_next
        return states


class Monolith(SafetyMixin):
    """Single MLP: full observation + one-hot action -> full next-observation delta. The optional safety
    member applies the same analytical margin rule to the monolith's own predicted joint states, so that a
    comparison with the three-member federation isolates the prediction source rather than the rule.

    delay / fault: the monolith has no internal message to check, so the same corruption is applied to the
    neighbour rows of its observation (T4: rows k decisions old; T6: stale rows, rows in ego-relative instead
    of absolute coordinates, velocities in km/h, or rows missing and the last ones silently reused)."""
    kind = "model"

    def __init__(self, mlp: MLP, delay=0, stale_prob=1.0, name="monolith", safety=None, fault=None,
                 fault_window=FAULT_WINDOW):
        self.mlp, self.delay, self.stale_prob, self.name, self.safety = mlp, int(delay), stale_prob, name, safety
        self.fault, self.fault_window = fault, fault_window
        self.n_params = mlp.n_params + (0 if safety is None else safety.n_params)
        self.buffer, self.abstentions, self.violations = [], 0, 0
        self.last_rows = None

    def begin_episode(self, ep_seed=0):
        self.buffer, self.abstentions, self.violations = [], 0, 0
        self.last_rows = None
        self.rng = np.random.default_rng([ep_seed, 4242])   # same staleness pattern as the federation

    def observe(self, state, t):
        self.buffer.append((t, state))
        self.delayed_now = self.delay if self.rng.random() < self.stale_prob else 0
        self.fault_now = self.fault if in_window(t, self.fault_window) else None
        if self.fault_now == "stale":
            self.delayed_now = 2

    def neighbour_rows(self, state):
        """The neighbour rows the monolith sees for this decision (corrupted under T4 / T6)."""
        ego, oth = split(state)
        if self.delayed_now:                        # equally stale information about the others
            _, src = self.buffer[max(len(self.buffer) - 1 - self.delayed_now, 0)]
            oth = split(src)[1]
        if self.fault_now == "frame":
            oth = rel_others(ego, oth)
        elif self.fault_now == "unit":
            oth = oth.copy()
            oth[:, 3:5] *= 3.6
        elif self.fault_now == "missing":
            if self.last_rows is not None:
                oth = self.last_rows
        else:
            self.last_rows = oth
        return ego, oth

    def rollout(self, state, chunk):
        ego, oth = self.neighbour_rows(state)
        states = []
        for a in chunk:
            rel = rel_others(ego, oth)
            out = self.mlp.predict(mono_x(ego, rel, a))
            ego_next = self.predict_ego(ego, a, out)
            oth_next = oth.copy()
            oth_next[:, 1:6] += out[5:].reshape(K_OTHERS, 5) * (oth[:, :1] > 0.5)
            states.append(join(ego_next, oth_next))
            ego, oth = ego_next, oth_next
        return states

    def predict_ego(self, ego, a, out):
        ego_next = ego.copy()
        ego_next[:5] += out[:5]
        ego_next[5], ego_next[6] = next_targets(ego, a)
        return ego_next


class MonolithAnalyticEgo(Monolith):
    """The monolith's own traffic prediction with the analytic ego member substituted for its ego rows
    (the monolith network still receives the ego state as input and is trained jointly). Against the
    federation it isolates what separate training in the ego frame adds beyond the analytic member itself;
    against the monolith it isolates what the analytic member adds on its own."""
    def __init__(self, mlp: MLP, ego_member, **kw):
        super().__init__(mlp, **kw)
        self.ego_member = ego_member

    def predict_ego(self, ego, a, out):
        return self.ego_member.predict(ego, a)


class Conventional(SafetyMixin):
    """Conventional driver without a world model or planner: IDM longitudinal rule and MOBIL lane choice
    (the rules highway-env's own vehicles follow, politeness 0) applied to the current observation and
    mapped to the meta-actions. Zero parameters, no prediction, no message to check."""
    kind = "rule"
    n_params = 0
    V0 = float(TARGET_SPEEDS[-1])                      # desired speed
    COMFORT_ACC_MAX, COMFORT_ACC_MIN = 3.0, -5.0       # IDM a, b (highway-env IDMVehicle)
    DISTANCE_WANTED, TIME_WANTED, DELTA = 5.0 + VEH_LENGTH, 1.5, 4.0
    MAX_BRAKING_IMPOSED, MIN_ACC_GAIN = 2.0, 0.2       # MOBIL
    DEAD_BAND = 0.5                                     # [m/s^2] IDLE when |a| is below it

    def __init__(self, name="conventional"):
        self.name = name
        self.abstentions = self.violations = 0

    def begin_episode(self, ep_seed=0): pass
    def observe(self, state, t): pass
    def rollout(self, state, chunk): return []

    def idm(self, v, lead_gap, lead_v):
        a = self.COMFORT_ACC_MAX * (1 - (max(v, 0) / self.V0) ** self.DELTA)
        if lead_gap is not None:
            d_star = (self.DISTANCE_WANTED + v * self.TIME_WANTED
                      + v * (v - lead_v) / (2 * np.sqrt(self.COMFORT_ACC_MAX * -self.COMFORT_ACC_MIN)))
            a -= self.COMFORT_ACC_MAX * (d_star / max(lead_gap + VEH_LENGTH, 1e-3)) ** 2
        return float(a)

    @staticmethod
    def neighbours(x, lane_y, others):
        """(leader, follower) in the lane at lane_y as (gap, vx) tuples, bumper to bumper; None if absent."""
        lead, foll, g_lead, g_foll = None, None, np.inf, np.inf
        for o in others:
            if o[0] < 0.5 or abs(o[2] - lane_y) >= LANE_W / 2:
                continue
            g = o[1] - x - VEH_LENGTH
            if o[1] > x and g < g_lead:
                lead, g_lead = (g, float(o[3])), g
            elif o[1] <= x and -g - 2 * VEH_LENGTH < g_foll:
                foll, g_foll = (x - o[1] - VEH_LENGTH, float(o[3])), -g - 2 * VEH_LENGTH
        return lead, foll

    def act(self, state):
        ego, others = split(state)
        x, y, v = float(ego[0]), float(ego[1]), float(ego[2])
        lane = int(round(y / LANE_W))
        lead, _ = self.neighbours(x, lane * LANE_W, others)
        a_keep = self.idm(v, *(lead if lead else (None, 0.0)))
        best, best_gain = None, self.MIN_ACC_GAIN
        for action, target in ((LANE_LEFT, lane - 1), (LANE_RIGHT, lane + 1)):
            if not 0 <= target < N_LANES or abs(y - lane * LANE_W) > 0.5:     # finish a change first
                continue
            new_lead, new_foll = self.neighbours(x, target * LANE_W, others)
            a_self = self.idm(v, *(new_lead if new_lead else (None, 0.0)))
            if a_self < -self.MAX_BRAKING_IMPOSED:
                continue
            if new_foll is not None and self.idm(new_foll[1], new_foll[0], v) < -self.MAX_BRAKING_IMPOSED:
                continue
            if a_self - a_keep > best_gain:
                best, best_gain = action, a_self - a_keep
        if best is not None:
            return best
        if a_keep < -self.DEAD_BAND:
            return SLOWER
        if a_keep > self.DEAD_BAND and ego[5] < self.V0:
            return FASTER
        return IDLE


class Trivial(SafetyMixin):
    """Oracle (scores chunks with a deep copy of the true env) and Random (random chunk)."""
    n_params = 0

    def __init__(self, kind):
        self.kind, self.name = kind, kind
        self.abstentions = self.violations = 0

    def begin_episode(self, ep_seed=0): pass
    def observe(self, state, t): pass
    def rollout(self, state, chunk): return []
