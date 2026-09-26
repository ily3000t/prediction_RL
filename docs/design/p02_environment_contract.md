# P2 environment and execution contract

## Frozen semantics

The adapter delegates to the supplied ContinuousJerkEnv and the P1 legacy
500-step wrapper. No edits to upstream reward, physics, traffic generation,
DDPG, ST, XML or configs. No prediction features, shield, action snapping or
wall-clock-dependent observation replacement.

The public P2 interface is AuditedJerkEnv within upstream_session, under
src/prediction_rl/envs/upstream.py. This is not automatically wired into DDPG.
reset() returns (observation, reset_info); step() returns the legacy four values
(observation, original reward, done, info). It is not a Gymnasium five-value API.
Observation casting matches original ALL's declared float32 space. Values must
be finite and have the correct shape; the wrapper does not clip observations.
Valid actions are one-element finite vectors within upstream jerk bounds,
cast to the original action dtype. Invalid actions raise and invalidate the
adapter, rather than being clipped, replaced or retried.

## Run seeds and continuous training episodes

Original main.py seeds Python, NumPy, torch and SUMO once per run.
The original environment does NOT restart SUMO between reset calls.
Background traffic and control.delay persist across episodes.

upstream_session sets the explicit run seed and initializes control.delay=0,
equivalent to a new upstream process. reset() never reseeds or clears traffic.
Different independent simulator seeds should run in separate worker processes.
The same seed at the tenth streaming episode is not the same scene as the first.

Only one owned default TraCI connection is supported per process. Concurrent
sessions and foreign generic module names are rejected. Use separate processes,
not threads. Sessions restore CWD, path, settings, Python/NumPy/torch RNG state
and control.delay at exit. Do not embed this initialization in an already
running policy-training process: create its dedicated worker first.

## Action audit

Every step delegates exactly once and records:

- requested_jerk: policy request at original action dtype.
- commanded_speed_reconstructed: output of the unchanged upstream speed helper
  applied to measured pre-step speed/acceleration. This is a RECONSTRUCTION,
  not a network interception or a direct SUMO measurement.
- measured_post_step_speed / acceleration: actual available SUMO observations.
- measured_jerk: acceleration difference over the original decision period,
  only when ego remains present and exactly one control period has elapsed.
- upstream_reward_projected_jerk and invalid_action_reward: upstream reward
  accounting values, not substituted for measured execution.
- simulation_elapsed_s and full before/after traffic snapshots.
- termination_reason plus the original TimeLimit.truncated field.

SUMO speedMode=22 and all other execution constraints remain original.
Requested jerk, reconstructed command and realized motion need not coincide;
the adapter records their distinction without enforcing a new controller.

## Termination boundaries

Use the original order: upstream collision, upstream arrival, internal time
limit, then registered wrapper limit. The upstream internal time-limit branch
removes ego and calls control.step() a second time. Therefore its terminal
transition advances 0.4 seconds with the default 0.2-second period. Preserve it.

After arrival/removal, measured ego values are null, NOT invented zero values.
Returned terminal observation/reward remains the original upstream result.
The internal timeout can produce TimeLimit.truncated=false because upstream
already returned done=true. Record the separate reason; do not reinterpret
done or change ALL bootstrapping silently.

The upstream collision detector itself is retained. Unit tests cover reason
precedence, but the current real-SUMO audit does not intentionally create a
collision. It is not a collision detector validation or safety guarantee.

## Validation and next phase

P2 compares exact observations at policy dtype, rewards, flags, original info,
full traffic snapshots and control.delay between reference and adapter.
It also checks independent-process repeatability and continuing traffic reset.
No tolerance is used in trajectory equality. A small floating-point tolerance
only checks elapsed time equals the specified period at the timeout boundary.

P3 must save/restore SUMO and Python-side state, including control.delay,
random states, previous acceleration/state, episode counters and histories.
Saving SUMO state alone is insufficient. Keep branch intervention definitions
separate from continuing-episode reset semantics. No snapshot/branch API has
been implemented or accepted in P2.
