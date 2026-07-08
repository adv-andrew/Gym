<!-- SPDX-License-Identifier: Apache-2.0 -->
# OpenAir Congestion Resource Server

5G RAN congestion control on OpenAirInterface (OAI). Multi-turn gymnasium-style environment: each step the model observes rolling 5-second cell and UE telemetry and issues exactly one network-control tool call (7 actuators plus `noop`) to relieve congestion. Reward is a dense per-step KPI total; the shared `gymnasium_agent` drives `/reset` + `/step` and sums step rewards into the episode return.

**The `openair_congestion` telco env package lives in a separate repository and is not on PyPI.** It must be installed in the Gym venv before the server starts (see Setup). The default replay backend runs standalone -- no 5G stack, no GPU, no KPI exporter.

## Tools

Tool schemas ride in each task row's `responses_create_params.tools`; the canonical JSON-Schema per tool is `openair_congestion.tools.TOOL_SCHEMA_BY_NAME` in the env package. Argument ranges are validated by the env guardrail: out-of-range actions are rejected and penalized, never crash.

| Tool | Required arguments | Effect |
|------|--------------------|--------|
| `set_scheduler_policy` | `cell_id`, `policy` in {PF, RR, MaxCI} | Per-cell MAC scheduler |
| `set_prb_cap` | `cell_id`, `target` in {ue, slice}, `target_id`, `max_prb` | Cap PRBs for a UE or slice |
| `set_mcs_bounds` | `cell_id`, `mcs_min`, `mcs_max`, `target_bler` | Bound link adaptation |
| `set_qos_weights` | `cell_id`, `weights` (object) | Per-5QI scheduler weights |
| `set_admission_policy` | `cell_id`, `accept_threshold_pct`, `slice_reservation` (object) | RRC admission control |
| `set_handover_trigger` | `cell_id`, `a3_offset_db` (-24..24), `ttt_ms` (enum) | A3 handover trigger |
| `set_ul_power_control` | `cell_id`, `p0_dbm` (-126..23), `alpha` (enum) | UL fractional power control |
| `noop` | (none) | Take no action this step |

## Backends

`app.py` codes against a three-method `Backend` contract (`reset`/`step`/`close`, defined in `backends.py`); which driver serves episodes is pure configuration.

| Backend | Runs | Description |
|---------|------|-------------|
| `replay` (default) | offline | Deterministic `ReplayEnv` from the env package. The KPI trajectory is pre-baked at reset from the task row's `seed`/`difficulty`/`regime_mix`; `step()` applies a deterministic synthetic action-effect model, the guardrail, and the reward. Identical inputs produce identical episodes. |
| `dataset_replay` | offline | Replays a provided dataset file (`dataset_backend.py`), in either format below. Actions are observational pass-through: they never change the next recorded KPI row. The only action-dependent signal comes from format/guardrail validity and configured action/rejection terms, so this backend supports protocol-validity learning and data-path tests, not a claim that the policy learned to relieve congestion. |
| `oai_collector` | online | Live OAI 5G stack via a Prometheus-style KPI exporter. Stub: the constructor raises `NotImplementedError` until the lab wiring lands. The config knobs (`kpi_url`, `oai_pool_size`, `step_dt_s`, `steady_state_s`, `scenario_mode`) are already forwarded by `select_backend`. |

Select with the `backend:` field in `configs/openair_congestion.yaml`. The `OPENAIR_CONGESTION_BACKEND` environment variable overrides it for local development.

Episode slots are finite (`pool_size`). Four guards keep slots from leaking: a repeated `/reset` on the same session closes the previous episode first; every requested task budget is capped by `agent_max_steps`; terminal `/step` responses close automatically; and a trainer that exits early can call cookie-scoped, idempotent `POST /close`. When the pool is exhausted, `reset()` also reaps episodes no live session owns before failing. Session state is process-local, so the standalone launcher deliberately uses exactly one Uvicorn worker.

## Setup

Install the telco env package (editable) into the same venv that runs NeMo-Gym:

```bash
pip install -e /path/to/openair-rl-gym/env/nemo_gym/envs/openair_congestion
```

`requirements.txt` lists this editable install; adjust the path to the local checkout. The replay path's only runtime dependencies are `pydantic` and `numpy`. The optional `congestion_gen` scenario generator is not required -- without it the replay backend uses a built-in 2-cell/4-UE fallback scenario. A missing install fails at server startup with an actionable `ImportError`.

## Dataset Formats

The `dataset_replay` backend accepts two JSONL formats, detected from the first row: a row carrying `tool_sent` or `reward_measurements` selects the GRPO rollout-trace parser, anything else the KPI-snapshot parser. The format must be consistent across the file. Every row is validated at server boot; malformed data fails with the file, line number, episode, and field name -- never mid-training. Each episode needs at least two rows (each step scores a prev/curr observation pair).

### KPI snapshots

One JSON object per line; one line is one timestep of one episode. Rows sharing an `episode_id` form one episode.

| Field | Required | Notes |
|-------|----------|-------|
| `cells[]` | yes | At least one cell |
| `cells[].prb_util_dl_p50` | yes | 0-1 |
| `cells[].ues[]` | yes | Non-empty |
| `cells[].ues[].delivered_mbps` | yes | Core measured KPI |
| `episode_id` | no | `episode` accepted as an alias; rows without one collapse into a single episode in file order |
| `step` | no | Ordering within an episode; wins over `t_s`, which wins over file order |
| `t_s` | no | Timestamp in seconds |
| `kpi_source_mode` | no | Default `replay` stamps synthesized fields as synthetic; set e.g. `runner_snapshot` for lab-measured data |

Optional per cell: `prb_util_dl_p99`, `prb_util_ul_p50`, `sched_latency_ms_p99`, `rrc_connected_ues`, `prach_collision_rate`, `fairness_jain`, `sla_violations_last_window`. Optional per UE: `offered_mbps`, `bler`, `sinr_db`, `mcs_mean`, `buffer_occupancy_kb`, `pdb_violations`, `5qi` (alias `qos_5qi`). Provided values pass through unchanged (clamped to schema bounds); missing values are synthesized with the same heuristics the live env uses (`env.py::_build_observation`), so a sparse dataset still yields the full training shape.

A minimal row:

```json
{"episode_id": "run_a", "step": 0,
 "cells": [{"cell_id": 0, "prb_util_dl_p50": 0.55,
   "ues": [{"ue_id": 0, "offered_mbps": 20.0, "delivered_mbps": 18.0,
            "bler": 0.05, "sinr_db": 12.0}]}]}
```

CSV is accepted through the adapter in `_rows_from_csv` (one line per UE per timestep, regrouped into the nested row shape). Rewards for snapshot data are the standard `rewards.compute_breakdown(prev_obs, curr_obs, action, rejected=...)` over the provided pair -- no new reward math. A snapshot fixture is checked in at `data/fixtures/sample_provided.jsonl`.

### GRPO rollout traces

One JSON object per RL step, as emitted by GRPO training runs. Only the observation-bearing fields are consumed:

| Column | Required | Read as |
|--------|----------|---------|
| `reward_measurements` | yes | Aggregate KPI state (the dict `rewards.compute_breakdown` emits), distributed over the inferred runner topology. `aggregate_delivered_mbps` and `n_ues` are required; the other read keys -- `mean_jain_fairness`, `sla_violations`, `prb_pressure`, `access_pressure`, `buffer_pressure`, `requested_service_mbps` -- default to their uncongested values when absent |
| `reward_measurements.cell_capacity_mbps_total` | no | Keeps that transition's reward normalizer at the recorded scale; absent on a row, the configured `cell_capacity_mbps` applies to the transition ending at that row |
| `episode_id` | no | Episode grouping; if a merged trace reuses an ID across `iter` values, keys become `iter_N::episode_id` so iterations cannot interleave |
| `step` | no | Ordering within an episode; falls back to `t_s`, then file order |
| `iter` | no | GRPO iteration identity; used to disambiguate repeated episode IDs |
| `scenario_mode` | no | Known runner topology (`t1_runner` = 2 cells, `t2_runner` = 3); validated against accepted recorded action targets |
| `kpi_source` | no | Provenance stamp (default `replay`) |

Actions during replay come from the policy being trained. Recorded accepted `tool_sent` metadata is used only to infer/validate cell topology, and row 0's accepted action seeds the logical guardrail history because row 0 is already a post-action snapshot. That action is exposed in the initial observation's `L` row, so a first-turn identical-repeat rejection never depends on hidden row-0 state. Aggregate throughput, UE/SLA counts, pressure, mean fairness, and fairness-deficit measurements are distributed deterministically across the inferred cells; local UE IDs restart at zero in each cell. This is a versioned reconstructed proxy (`aggregate_trace_multicell_proxy_v1`), not recovery of the original nested observation: per-UE service accounting, 5QI mix, radio fields, and buffer distribution are unavailable. Reward equivalence is therefore not guaranteed for richer reward versions, and Run A absolute returns must not be compared with the source trace's returns.

A trace with **N action/reward rows yields N-1 replay transitions** because it does not contain the observation that preceded row 0. Row 0 becomes the initial observation and its accepted action becomes initial guardrail history; its recorded reward is not replayed.

### Replaying a provided dataset

In `configs/openair_congestion.yaml`, under the resources server:

```yaml
backend: dataset_replay
dataset_path: data/dataset/provided.jsonl
cell_capacity_mbps: 60.0   # reward throughput normalizer
reward_profile: openair_v2_measured
reward_weights: {w_sla: 0.0, w_sla_level: 0.0, w_buffer: 0.0, w_action: 0.0}
```

Episode selection per rollout uses this precedence: `task_params.scenario_id` exact-matches a dataset key; otherwise a non-negative `dataset_index` selects `keys[index % num_episodes]`; otherwise the seed is the backward-compatible fallback. GRPO trainers should give every replica in one group the same `dataset_index` and advance it by group, avoiding accidental low-coverage cycles from seed modulo arithmetic. The shipped `data/example.jsonl` pins replay-backend scenario names and pairs with the default backend only.

## End-to-End Rollout

### 1. Start the Gym servers

`ng_run` must also load a `responses_api_models` config named `policy_model` -- the model server the agent config references. Supply your own and load it alongside the env config:

```bash
ng_run "+config_paths=[resources_servers/openair_congestion/configs/openair_congestion.yaml,<your_policy_model>.yaml]"
```

### 2. Collect rollouts

```bash
ng_collect_rollouts \
  +agent_name=openair_congestion_gymnasium_agent \
  +input_jsonl_fpath=resources_servers/openair_congestion/data/example.jsonl \
  +output_jsonl_fpath=results/openair_congestion_rollouts.jsonl \
  +num_samples_in_parallel=5
```

Add `+limit=1` for a quick single-episode test.

### Standalone server for a local trainer

No model server or API key is needed when a local trainer owns generation. The standalone launcher serves only the resource API and hardcodes one worker:

```bash
python -m resources_servers.openair_congestion.serve \
  --backend dataset_replay \
  --dataset-path /absolute/path/to/train.jsonl \
  --reward-profile openair_v2_measured \
  --observation-render resource_compact_pipe_v1 \
  --max-steps 12 --pool-size 64 --port 9110
```

Use `--backend replay --port 9111` for an action-responsive synthetic comparison. Three observation contracts are explicit: `verbose_v1`; `resource_compact_pipe_v1`, the truthful T/C/U/L/A resource form and standalone default; and strict `t2_compact_pipe_v2`, which delegates to the qualified telco renderer and includes P/D rows. Aggregate traces do not contain the capacity/candidate state needed for truthful P/D rows, so dataset replay must use the resource form. Selecting strict T2 fails at server startup when the installed telco package lacks that renderer; use it only with a real T2 observation source. The catalog YAML retains the verbose form. The `/reset` and `/step` responses expose backend/dynamics semantics, the selected render, effective reward weights, and (for datasets) SHA-256 identity, row/episode counts, reconstruction schema/topology/assumptions, transition capacity, and dataset key/index.

Malformed, missing, or multiple tool calls consume one transition and receive the backend's ordinary guardrail-rejection penalty. This closes the shortcut where a policy could skip a negative recorded KPI transition by emitting no valid call.

For `dataset_replay`, report return improvement only together with parse-invalid, rejection, noop, and accepted-nonnoop rates. A falling rejection rate demonstrates learning this reconstructed server's tool-validity contract; it does **not** demonstrate that actions improved recorded KPIs. The action-responsive `replay` backend is the separate decision-learning experiment.

### Scripted client demo

`client.py` drives one full episode over the real HTTP surface -- POST `/reset`, then a `/step` loop -- using a scripted congestion-relief heuristic instead of an LLM, printing each step's tool call, guardrail verdict, and reward, then the episode return. With a gym up (step 1) it connects to the served instance; without one it boots a local in-process server on the replay backend, so it runs fully offline:

```bash
python resources_servers/openair_congestion/client.py
```

### Run tests

```bash
ng_test +entrypoint=resources_servers/openair_congestion

# Or directly (the env package must be importable -- see Setup):
pytest resources_servers/openair_congestion/tests -q
```

`tests/test_reward_correctness.py` is the reward oracle: on fixed replay seeds, scripted targeted congestion relief must out-return `noop`, which must out-return random valid play, which must out-return always-rejected catastrophic play. This ordering is intentional under the v3 zero-sum action model: an action passing the guardrail is not automatically beneficial, and arbitrary controls should lose to deliberately standing pat. Per step, clearing an SLA violation, dropping PRB below the pressure threshold, or draining buffers never scores lower, and a rejected action always scores below the same step accepted. All assertions are relative orderings, never absolute thresholds, so a reward renormalization does not invalidate them.

## Verification

Episode return is the undiscounted sum of per-step reward totals. Each step's reward comes from the env's `rewards.compute_breakdown(prev_obs, curr_obs, action, rejected=...)` -- a weighted mix of SLA-violation, throughput, and fairness deltas plus congestion-pressure level terms -- and passes through the backend layer unchanged. Guardrail-rejected actions, including turns with no tool call, consume a transition, keep the episode alive until its normal horizon, and earn the rejection penalty. Identical `seed` plus action sequence yields an identical observation sequence on the offline backends.

## License

Code is Apache-2.0. The default replay backend produces synthetic benchmark telemetry. `dataset_replay` preserves the provenance supplied by its input dataset; operators are responsible for the dataset's privacy, licensing, and measurement claims.
