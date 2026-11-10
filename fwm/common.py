"""Shared definitions: environment config, conditions, state layout, ego-shift vehicle, ego and
safety-margin parameters, analytic ego step, reward model.

State layout used by every system (flat vector of length STATE_DIM = 7 + 7*6 = 49):
  ego    = [x, y, vx, vy, heading, v_target, lane_target]   (v_target / lane_target are the ego's
           own command set-points, i.e. the MDPVehicle's target_speed and target lane index)
  others = K_OTHERS rows of [presence, x, y, vx, vy, heading] in ABSOLUTE coordinates
"""
from __future__ import annotations

import copy
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401  (registers highway-v0)
from highway_env import utils as hw_utils
from highway_env.vehicle.controller import MDPVehicle

ACTIONS = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER = range(5)
N_ACTIONS = 5
N_LANES, LANE_W = 4, 4.0
SIM_FREQ, POLICY_FREQ = 10, 2
DT_SIM, FRAMES = 1.0 / SIM_FREQ, SIM_FREQ // POLICY_FREQ      # 0.1 s x 5 frames = one 0.5 s decision
DURATION = 30                                                  # [s] -> 60 decisions per episode
K_OTHERS = 7
EGO_DIM, OTH_DIM = 7, 6
STATE_DIM = EGO_DIM + K_OTHERS * OTH_DIM
VEH_LENGTH, VEH_WIDTH = 5.0, 2.0
TARGET_SPEEDS = np.array([20.0, 25.0, 30.0])
REWARD_W = {"collision": -1.0, "right_lane": 0.1, "high_speed": 0.4, "speed_range": (20.0, 30.0)}
COLLISION_PENALTY = 100.0   # planner penalty for a predicted (or oracle-true) collision in a chunk

ENV_CONFIG = {
    "observation": {"type": "Kinematics", "absolute": True, "normalize": False,
                    "vehicles_count": K_OTHERS + 1, "see_behind": True,
                    "features": ["presence", "x", "y", "vx", "vy", "heading"]},
    "action": {"type": "DiscreteMetaAction"},
    "policy_frequency": POLICY_FREQ, "simulation_frequency": SIM_FREQ,
    "duration": DURATION, "vehicles_density": 1.0, "lanes_count": N_LANES,
    "vehicles_count": 20, "ego_spacing": 2, "normalize_reward": True,
}
TRAFFIC_SHIFT = {"other_vehicles_type": "highway_env.vehicle.behavior.AggressiveVehicle",
                 "vehicles_density": 1.5}
CONDITIONS = {
    "nominal":       {},
    "ego_shift":     {},                                   # shifted ego vehicle installed by reset_env()
    "traffic_shift": dict(TRAFFIC_SHIFT),
    "combined":      dict(TRAFFIC_SHIFT),                  # traffic shift AND the shifted ego vehicle (T5)
}
EGO_SHIFT_CONDITIONS = ("ego_shift", "combined")           # conditions that install ShiftedMDPVehicle


class EgoParams:
    """Parameters of the ego longitudinal controller that the analytic member exposes:
    acceleration gain kp_a and the acceleration clip [-acc_brake_max, acc_max]."""
    def __init__(self, kp_a=MDPVehicle.KP_A, acc_max=np.inf, acc_brake_max=np.inf):
        self.kp_a, self.acc_max, self.acc_brake_max = float(kp_a), float(acc_max), float(acc_brake_max)

    def __repr__(self):
        return f"EgoParams(kp_a={self.kp_a:.3f}, acc_max={self.acc_max}, acc_brake_max={self.acc_brake_max})"


class ShiftedMDPVehicle(MDPVehicle):
    """T2 ego shift (a loaded vehicle): half the acceleration gain, sluggish acceleration and reduced
    but not crippled braking. The acceleration is clipped to [-ACC_BRAKE_MAX, +ACC_MAX]."""
    KP_A = 0.5 * MDPVehicle.KP_A
    ACC_MAX = 1.5         # [m/s^2]
    ACC_BRAKE_MAX = 4.0   # [m/s^2]

    def act(self, action=None):
        super().act(action)
        self.action["acceleration"] = float(np.clip(self.action["acceleration"], -self.ACC_BRAKE_MAX, self.ACC_MAX))


EGO_PARAMS = {"nominal": EgoParams(), "traffic_shift": EgoParams(),
              "ego_shift": EgoParams(ShiftedMDPVehicle.KP_A, ShiftedMDPVehicle.ACC_MAX, ShiftedMDPVehicle.ACC_BRAKE_MAX)}
EGO_PARAMS["combined"] = EGO_PARAMS["ego_shift"]

IDM_ACC_MAX = 6.0            # highway-env IDMVehicle.ACC_MAX: the hardest braking of a leading vehicle
SPEED_STEP = float(TARGET_SPEEDS[1] - TARGET_SPEEDS[0])   # one target-speed step (5 m/s)


class SafetyParams:
    """Parameters of the longitudinal safety-margin member (RSS-style minimum following distance),
    derived from the ego controller parameters.

    rho           response time: one decision (0.5 s)
    a_accel       largest ego acceleration during the response time: kp_a times one target-speed step,
                  bounded by the acceleration clip
    a_brake_min   braking the ego is guaranteed to deliver: kp_a times one speed step, bounded by the
                  braking clip
    a_brake_front hardest braking of a leading vehicle (IDM ACC_MAX)
    """
    def __init__(self, ego: EgoParams, rho=FRAMES * DT_SIM, a_brake_front=IDM_ACC_MAX):
        self.rho = float(rho)
        self.a_accel = float(min(ego.kp_a * SPEED_STEP, ego.acc_max))
        self.a_brake_min = float(min(ego.kp_a * SPEED_STEP, ego.acc_brake_max))
        self.a_brake_front = float(a_brake_front)

    def __repr__(self):
        return (f"SafetyParams(rho={self.rho}, a_accel={self.a_accel:.3f}, a_brake_min={self.a_brake_min:.3f}, "
                f"a_brake_front={self.a_brake_front})")


SAFETY_PARAMS = {cond: SafetyParams(p) for cond, p in EGO_PARAMS.items()}


def make_env(condition: str):
    cfg = copy.deepcopy(ENV_CONFIG)
    cfg.update(CONDITIONS[condition])
    return gym.make("highway-v0", config=cfg)


def reset_env(env, seed: int, condition: str):
    env.reset(seed=int(seed))
    if condition in EGO_SHIFT_CONDITIONS:
        u = env.unwrapped
        old = u.vehicle
        new = ShiftedMDPVehicle(u.road, old.position.copy(), old.heading, old.speed,
                                target_speeds=TARGET_SPEEDS)
        u.road.vehicles[u.road.vehicles.index(old)] = new
        u.vehicle = new
    return observe(env)


def observe(env, tracked=None):
    """Build the state vector. If `tracked` (list of vehicle objects) is given, use those vehicles
    (in that order) so that row identity is preserved across one step; otherwise select the
    K_OTHERS nearest vehicles exactly like highway-env's KinematicObservation does."""
    u = env.unwrapped
    ego = u.vehicle
    if tracked is None:
        tracked = u.road.close_objects_to(ego, u.PERCEPTION_DISTANCE, count=K_OTHERS,
                                          see_behind=True, sort=True, vehicles_only=True)
    s = np.zeros(STATE_DIM)
    s[:EGO_DIM] = [ego.position[0], ego.position[1], ego.velocity[0], ego.velocity[1],
                   ego.heading, ego.target_speed, ego.target_lane_index[2]]
    for i, v in enumerate(tracked[:K_OTHERS]):
        s[EGO_DIM + i * OTH_DIM: EGO_DIM + (i + 1) * OTH_DIM] = [
            1.0, v.position[0], v.position[1], v.velocity[0], v.velocity[1], v.heading]
    return s, list(tracked[:K_OTHERS])


def split(s):
    """-> ego (7,), others (K,6)"""
    return s[:EGO_DIM], s[EGO_DIM:].reshape(K_OTHERS, OTH_DIM)


def join(ego, others):
    return np.concatenate([ego, others.reshape(-1)])


# ----------------------------------------------------------------------------------------------
# Analytic ego model: re-implementation of MDPVehicle.act + ControlledVehicle steering/speed
# control + Vehicle.step (kinematic bicycle) on a straight 4-lane road, for one decision step.
# ----------------------------------------------------------------------------------------------
def speed_to_index(speed):
    x = (speed - TARGET_SPEEDS[0]) / (TARGET_SPEEDS[-1] - TARGET_SPEEDS[0])
    return int(np.clip(np.round(x * (TARGET_SPEEDS.size - 1)), 0, TARGET_SPEEDS.size - 1))


def next_targets(ego, action):
    """Command set-point update (the discrete meta-action semantics of MDPVehicle)."""
    x, y, vx, vy, h, vt, lt = ego
    speed = float(np.hypot(vx, vy))
    if action == FASTER:
        vt = TARGET_SPEEDS[min(speed_to_index(speed) + 1, TARGET_SPEEDS.size - 1)]
    elif action == SLOWER:
        vt = TARGET_SPEEDS[max(speed_to_index(speed) - 1, 0)]
    elif action == LANE_LEFT:
        lt = max(int(lt) - 1, 0)
    elif action == LANE_RIGHT:
        lt = min(int(lt) + 1, N_LANES - 1)
    return float(vt), float(lt)


def ego_step(ego, action, p: EgoParams):
    """One decision step (FRAMES sim frames) of the ego under meta-action `action`."""
    V = MDPVehicle
    x, y, vx, vy, h, _, _ = [float(v) for v in ego]
    speed = float(np.hypot(vx, vy))
    vt, lt = next_targets(ego, action)
    y_lane = lt * LANE_W
    for _ in range(FRAMES):
        # steering_control (straight lane: heading 0, lateral coordinate = y - y_lane)
        lat_cmd = -V.KP_LATERAL * (y - y_lane)
        heading_cmd = np.arcsin(np.clip(lat_cmd / hw_utils.not_zero(speed), -1, 1))
        heading_ref = np.clip(heading_cmd, -np.pi / 4, np.pi / 4)
        heading_rate = V.KP_HEADING * hw_utils.wrap_to_pi(heading_ref - h)
        slip = np.arcsin(np.clip(VEH_LENGTH / 2 / hw_utils.not_zero(speed) * heading_rate, -1, 1))
        steer = float(np.clip(np.arctan(2 * np.tan(slip)), -V.MAX_STEERING_ANGLE, V.MAX_STEERING_ANGLE))
        acc = float(np.clip(p.kp_a * (vt - speed), -p.acc_brake_max, p.acc_max))
        # Vehicle.step
        beta = np.arctan(0.5 * np.tan(steer))
        x += speed * np.cos(h + beta) * DT_SIM
        y += speed * np.sin(h + beta) * DT_SIM
        h += speed * np.sin(beta) / (VEH_LENGTH / 2) * DT_SIM
        speed += acc * DT_SIM
    return np.array([x, y, speed * np.cos(h), speed * np.sin(h), h, vt, lt])


# ----------------------------------------------------------------------------------------------
# Reward model (mirrors HighwayEnv._reward/_rewards) evaluated on a predicted state.
# ----------------------------------------------------------------------------------------------
def predicted_collision(state, prev_state=None, n_sub=FRAMES):
    """Ego rectangle vs. other rectangles. With `prev_state`, the motion between the two predicted
    states is swept at the simulation-frame resolution (linear interpolation), because the env checks
    collisions every frame and a lateral crossing can start and end within one decision."""
    if prev_state is None:
        return _collision_at(state)
    return any(_collision_at(prev_state + (state - prev_state) * j / n_sub) for j in range(1, n_sub + 1))


def _collision_at(state):
    ego, others = split(state)
    r_ego = (ego[:2], VEH_LENGTH, VEH_WIDTH, ego[4])
    for o in others:
        if o[0] > 0.5 and abs(o[1] - ego[0]) < 2 * VEH_LENGTH and abs(o[2] - ego[1]) < 2 * VEH_WIDTH:
            if hw_utils.rotated_rectangles_intersect(r_ego, (o[1:3], VEH_LENGTH, VEH_WIDTH, o[5])):
                return True
    return False


def reward_model(state, collided=None, prev_state=None):
    ego, _ = split(state)
    if collided is None:
        collided = predicted_collision(state, prev_state)
    forward_speed = np.hypot(ego[2], ego[3]) * np.cos(ego[4])
    lo, hi = REWARD_W["speed_range"]
    r = (REWARD_W["collision"] * float(collided)
         + REWARD_W["right_lane"] * ego[6] / (N_LANES - 1)
         + REWARD_W["high_speed"] * float(np.clip((forward_speed - lo) / (hi - lo), 0, 1)))
    r = (r - REWARD_W["collision"]) / (REWARD_W["high_speed"] + REWARD_W["right_lane"] - REWARD_W["collision"])
    on_road = -LANE_W / 2 <= ego[1] <= (N_LANES - 1) * LANE_W + LANE_W / 2
    return float(r * on_road), bool(collided)
