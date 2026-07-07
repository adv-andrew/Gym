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
| `dataset_replay` | offline | Replays a provided dataset file (`dataset_backend.py`), in either format below. Actions are pass-through -- the data is pre-recorded -- but the guardrail still runs and rejections still cost the standard penalty. |
| `oai_collector` | online | Live OAI 5G stack via a Prometheus-style KPI exporter. Stub: the constructor raises `NotImplementedError` until the lab wiring lands. The config knobs (`kpi_url`, `oai_pool_size`, `step_dt_s`, `steady_state_s`, `scenario_mode`) are already forwarded by `select_backend`. |

Select with the `backend:` field in `configs/openair_congestion.yaml`. The `OPENAIR_CONGESTION_BACKEND` environment variable overrides it for local development.

Episode slots are finite (`pool_size`). Three guards keep slots from leaking: a repeated `/reset` on the same session closes the previous episode first; task rows lacking `max_steps` fall back to a budget capped at the agent's turn budget (`agent_max_steps` in the yaml must not exceed the gymnasium agent's `max_steps`), so the server always emits a terminal step and `close_session()` runs; and when the pool is exhausted, `reset()` reaps episodes orphaned by crashed rollouts before failing.

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
| `reward_measurements` | yes | Aggregate KPI state (the dict `rewards.compute_breakdown` emits), reconstructed into a single-cell observation. `aggregate_delivered_mbps` and `n_ues` are required; the other read keys -- `mean_jain_fairness`, `sla_violations`, `prb_pressure`, `access_pressure`, `buffer_pressure`, `requested_service_mbps` -- default to their uncongested values when absent |
| `reward_measurements.cell_capacity_mbps_total` | no | Keeps the reward's throughput normalizer at the recorded scale; absent, the `cell_capacity_mbps` config knob applies |
| `episode_id` | no | Episode grouping; rows without one collapse into a single episode in file order |
| `step` | no | Ordering within an episode; falls back to `t_s`, then file order |
| `kpi_source` | no | Provenance stamp (default `replay`) |

Everything else a trace row carries -- `tool_sent`, `reward`, `reward_terms`, `rejected`, `guardrail_accepted`, `rejection_reason`, `seed`, `scenario_mode`, `iter`, `group_id`, `episode_return`, `episode_advantage`, `raw_text`, `actuator` -- is ignored on replay: actions come from the policy being trained, and rewards and guardrail outcomes are recomputed over the reconstructed observation pairs. (`tool_sent` still matters for format detection: its presence on the first row selects the trace parser.) Reconstruction is exact at the aggregate level -- delivered throughput, mean Jain fairness, PRB/access/buffer pressure, SLA count -- and synthesized below it; replaying a trace's recorded action and guardrail outcome over a reconstructed pair reproduces the recorded per-step reward. A trace fixture is checked in at `data/fixtures/sample_trace.jsonl`.

### Replaying a provided dataset

In `configs/openair_congestion.yaml`, under the resources server:

```yaml
backend: dataset_replay
dataset_path: data/dataset/provided.jsonl
cell_capacity_mbps: 60.0   # reward throughput normalizer
```

Episode selection per rollout: `task_params.scenario_id` exact-matches a dataset episode key (an unknown id raises, listing the available keys); without a `scenario_id`, the pick is deterministic by seed (`keys[seed % num_episodes]`). Task rows for `dataset_replay` must therefore omit `scenario_id` or set it to a dataset episode key -- the shipped `data/example.jsonl` pins replay-backend scenario names (`prb_exhaustion`, `bursty`, ...) and pairs with the default backend only.

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

`tests/test_reward_correctness.py` is the reward oracle: on fixed replay seeds, scripted congestion relief must out-return random valid play, which must out-return `noop`, which must out-return always-rejected catastrophic play; per step, clearing an SLA violation, dropping PRB below the pressure threshold, or draining buffers never scores lower, and a rejected action always scores below the same step accepted. All assertions are relative orderings, never absolute thresholds, so a reward renormalization does not invalidate them.

## Verification

Episode return is the undiscounted sum of per-step reward totals. Each step's reward comes from the env's `rewards.compute_breakdown(prev_obs, curr_obs, action, rejected=...)` -- a weighted mix of SLA-violation, throughput, and fairness deltas plus congestion-pressure level terms -- and passes through the backend layer unchanged. Guardrail-rejected actions keep the episode alive and earn the rejection penalty; turns with no tool call return 0.0 without advancing the env. Identical `seed` plus action sequence yields an identical observation sequence on the offline backends.

## License

Code is Apache-2.0. All telemetry produced by the offline backends is synthetic benchmark data, not measured OAI/FlexRIC KPM.
