# Systems, tests and records

Specification of the data rule, the message contract, the comparator systems, the interface-fault test, the horizon
sweep and the episode records, as implemented in `fwm/`. The task list is `pipelines/slurm/jobs.sh` (745 tasks;
`FWM_CORE=1` restricts it to the 520 tasks of T1–T5 with the eight original systems). Reference records for every
system and test are in `results_ref/` (seed 0, two episodes per task, oracle scoring on); a reimplementation is
considered equivalent when it reproduces `chosen`, `abstained`, `safety_violation` and `success` per decision and the
chunk scores to 1e-9.

## 1. Data rule (`train.py`, `collect.py`)

`load_data()` requires `MIN_TRANSITIONS = 10,500` usable (non-crash) transitions for the two budget conditions
`ego_shift` and `traffic_shift`; `collect()` keeps logging episodes after the 300th until that holds. Nominal and
combined logs are unchanged (300 episodes). Consequence: the budget 10,000 is a genuine subset of the logged data in T2, T3 and T5; the `_b<budget>` subsets
are drawn from one seeded permutation of the whole log.

## 2. Message contract (`members.py`)

`Message` carries `timestamp`, `frame`, `horizon`, `validity`, `units` (`"SI"`) and `payload`; `valid_at()` checks
the first five against the current decision. `Federation.last_accepted` keeps the last message that passed the check.

## 3. Comparator and attribution systems

| name | class | what it is | parameters |
|---|---|---|---|
| `conventional` | `Conventional` | IDM longitudinal rule + MOBIL lane choice (politeness 0, braking imposed <= 2 m/s^2, gain >= 0.2 m/s^2, no change while |y - lane centre| > 0.5 m) on the current observation, mapped to meta-actions: lane change if MOBIL accepts one, else SLOWER if a < -0.5, FASTER if a > +0.5 and target speed < 30, else IDLE. No model, no planner, no message. `kind = "rule"`: the episode loop calls `act(state)`, draws the nine chunks anyway (rng streams unchanged), records `chosen = None`, regret `None`, and skips oracle scoring. | 0 |
| `federation_cv` | `Federation(AnalyticEgo, ConstantVelocityMember)` | the learned traffic member replaced by constant velocity for one decision (`d = [vx, vy, 0, 0, 0] * 0.5 s`); same coupler and contract | 0 |
| `monolith_aego` | `MonolithAnalyticEgo` | the parameter-matched monolith's traffic prediction with the analytic ego member substituted for the ego rows (the network still receives the ego state; models shared with `monolith`, including `union` fine-tunes in T5) | 28,064 |

IDM constants: a = 3, b = 5, d0 = 10 m (5 + vehicle length), T = 1.5 s, delta = 4, v0 = 30 m/s. Leader / follower
search: present rows in the lane (|y - lane_y| < 2 m), gap bumper to bumper (x difference minus 5 m).

## 4. T6, interface faults (`members.py`, `run.py`)

Condition nominal, fault injected at every decision with 10 <= t < 30 (a 10 s window), nothing outside it.
Variants per system: `fault in {stale, frame, unit, missing}`; federations also run `contract off`.

| fault | federation message | monolith observation (no check exists) |
|---|---|---|
| stale | computed from the observation two decisions old | neighbour rows two decisions old |
| frame | payload in world coordinates, `frame = ("world", t)` | neighbour rows made ego-relative (x - x_ego, y - y_ego) |
| unit | payload velocities x 3.6, `units = "km/h"` | neighbour velocities x 3.6 |
| missing | no message. Contract on: abstain (SLOWER). Contract off: silently reuse `last_accepted` (the message from decision 9 for the whole window) | neighbour rows of the last observation before the window reused |

Contract on: a failed check counts one `contract_violation` and one abstention, the planner executes the SLOWER chunk.
Contract off: the payload is used as if valid; `violations` still counts the failed check.
Outside the window, every system is exactly its T1 self (T1 and T6 share the episode seeds).

## 5. Horizon sweep (`run.py --horizon`, `members.set_horizon`)

`H` is set per run (default 3). It changes the chunk length everywhere: `make_chunks` (five constant chunks of H
actions plus four random chunks of H draws from the same rng stream), the message's `horizon` field, the rollouts,
the oracle. Tasks `--horizon 2` and `--horizon 5` with tag `_h2` / `_h5` in T1 for oracle, federation, federation3,
monolith, monolith_safe; the record carries `"horizon"`.

## 6. Records

Every episode record carries `"fault"` (null outside T6) and `"horizon"`. `chosen` can be `null`
(conventional). `aggregate.py` groups by (system, budget, stale, contract, fault, horizon) and adds per-decision
latency p50 / p95 and the fraction of decisions above 100 ms from `decisions.latency_ms` (one task per core; the latency column of a batched run is not a per-decision latency).

## 7. Complementary analyses

- `error_budget.py --seed s`: reference-substitution error budget on 10 closed-loop episodes per seed (true ego /
  true traffic substituted into the coupler; needs deep copies of the environment).
- `rank_metrics.py`: decision-level agreement with the oracle from the per-decision logs of any run (GPU records
  included, since scores are logged).
- A CPU latency run (`run.py --test T1 --no-oracle-scores --episodes 20`, every system, one core) gives the cost table.

## 8. Task counts

Per seed: T1 11 + horizon 10, T2 21, T3 27, T4 24, T5 32, T6 24 = 149; five seeds = 745 tasks, each independent.
