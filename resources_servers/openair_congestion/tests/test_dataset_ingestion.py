# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Loader and backend tests for the dataset ingestion path
# (dataset_backend.py), covering both the KPI-snapshot and the GRPO
# rollout-trace formats. Fully offline: fixtures live under data/fixtures/.
# Requires the cross-repo telco env package (see README Setup); tests are
# skipped, not failed, if the package is missing.
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


openair = pytest.importorskip(
    "openair_congestion",
    reason="telco env package 'openair_congestion' not installed; see README Setup",
)

from openair_congestion.schemas import Observation, ToolCall  # noqa: E402

from resources_servers.openair_congestion.app import _to_resource_compact_pipe_v1  # noqa: E402
from resources_servers.openair_congestion.backends import select_backend  # noqa: E402
from resources_servers.openair_congestion.dataset_backend import (  # noqa: E402
    DATASET_DYNAMICS_MODE,
    DatasetReplayBackend,
    load_provided_dataset,
)


FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"
SNAPSHOT_FIXTURE = FIXTURES / "sample_provided.jsonl"
TRACE_FIXTURE = FIXTURES / "sample_trace.jsonl"

NOOP = ToolCall(name="noop", arguments={})


def _make_backend(**overrides) -> DatasetReplayBackend:
    kwargs = dict(dataset_path=str(SNAPSHOT_FIXTURE), pool_size=4, max_steps_default=60)
    kwargs.update(overrides)
    return DatasetReplayBackend(**kwargs)


class TestSnapshotLoader:
    def test_fixture_parses_into_two_valid_episodes(self):
        episodes = load_provided_dataset(SNAPSHOT_FIXTURE)
        assert sorted(episodes) == ["lab_run_a", "lab_run_b"]
        assert len(episodes["lab_run_a"].observations) == 4
        assert len(episodes["lab_run_b"].observations) == 3
        for source in episodes.values():
            for obs in source.observations:
                assert isinstance(obs, Observation)  # fully schema-validated

    def test_missing_optional_fields_are_synthesized(self):
        # lab_run_a rows only carry the required fields; everything else is
        # derived with the env's own heuristics (env.py::_build_observation).
        obs = load_provided_dataset(SNAPSHOT_FIXTURE)["lab_run_a"].observations[0]
        cell = obs.cells[0]
        assert cell.prb_util_dl_p99 == pytest.approx(
            max(cell.prb_util_dl_p50, min(1.0, cell.prb_util_dl_p50 * 1.15 + 0.02))
        )
        assert cell.prb_util_ul_p50 == pytest.approx(cell.prb_util_dl_p50 * 0.4)
        assert cell.sched_latency_ms_p99 == pytest.approx(5.0 + 20.0 * cell.prb_util_dl_p99)
        assert cell.rrc_connected_ues == len(cell.ues) == 2
        ue1 = cell.ues[1]  # offered 30, delivered 14, sinr 6
        assert ue1.mcs_mean == pytest.approx((6.0 + 5.0) * 1.2)
        assert ue1.buffer_occupancy_kb == pytest.approx((30.0 - 14.0) * 50.0)
        assert ue1.pdb_violations == 1  # buffer 800 kB > 500 kB
        assert ue1.qos_5qi == 9  # default
        assert 0.0 <= cell.fairness_jain <= 1.0
        assert cell.sla_violations_last_window == 1

    def test_provided_fields_pass_through_unchanged(self):
        # lab_run_b row 0 cell 0 provides the full KPI set; no synthesis.
        obs = load_provided_dataset(SNAPSHOT_FIXTURE)["lab_run_b"].observations[0]
        cell = obs.cells[0]
        assert cell.prb_util_dl_p99 == pytest.approx(0.52)
        assert cell.fairness_jain == pytest.approx(0.93)
        assert cell.ues[0].mcs_mean == pytest.approx(20.0)
        assert obs.global_.difficulty == pytest.approx(0.7)

    def test_malformed_row_fails_fast_with_line_number(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        rows = [
            {"episode_id": "e", "cells": [{"prb_util_dl_p50": 0.5, "ues": [{"delivered_mbps": 1.0}]}]},
            {"episode_id": "e", "cells": [{"prb_util_dl_p50": 0.5, "ues": [{"delivered_mbps": "not_a_number"}]}]},
        ]
        bad.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        with pytest.raises(ValueError, match="bad.jsonl:2"):
            load_provided_dataset(bad)

    def test_row_missing_required_field_names_it(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        row = {"episode_id": "e", "cells": [{"ues": [{"delivered_mbps": 1.0}]}]}
        bad.write_text(json.dumps(row) + "\n")
        with pytest.raises(ValueError, match="prb_util_dl_p50"):
            load_provided_dataset(bad)

    def test_single_row_episode_is_rejected(self, tmp_path):
        # One observation cannot form a (prev, curr) reward pair.
        short = tmp_path / "short.jsonl"
        row = {"episode_id": "e", "cells": [{"prb_util_dl_p50": 0.5, "ues": [{"delivered_mbps": 1.0}]}]}
        short.write_text(json.dumps(row) + "\n")
        with pytest.raises(ValueError, match="need >= 2"):
            load_provided_dataset(short)

    def test_missing_file_error_is_actionable(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="dataset_path"):
            load_provided_dataset(tmp_path / "nope.jsonl")

    def test_wrong_scalar_type_fails_fast_with_line_number(self, tmp_path):
        # A list where a number belongs raises TypeError inside float();
        # the loader must still wrap it with file:line context.
        bad = tmp_path / "bad.jsonl"
        rows = [
            {"episode_id": "e", "cells": [{"prb_util_dl_p50": 0.5, "ues": [{"delivered_mbps": 1.0}]}]},
            {"episode_id": "e", "cells": [{"prb_util_dl_p50": 0.5, "ues": [{"delivered_mbps": [1, 2]}]}]},
        ]
        bad.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        with pytest.raises(ValueError, match="bad.jsonl:2"):
            load_provided_dataset(bad)

    def test_long_episode_key_is_accepted(self, tmp_path):
        # Placeholder episode_id is clamped ('src_' + key[:56]) so long run
        # names don't trip the schema's episode_id max_length=64 at boot.
        key = "run_" + "x" * 100
        path = tmp_path / "long.jsonl"
        row = {"episode_id": key, "cells": [{"prb_util_dl_p50": 0.5, "ues": [{"delivered_mbps": 1.0}]}]}
        path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
        episodes = load_provided_dataset(path)
        assert list(episodes) == [key]  # full key preserved for scenario_id matching
        assert len(episodes[key].observations[0].episode_id) <= 64

    def test_csv_adapter_point_round_trips(self, tmp_path):
        # Flat shape, one line per UE per timestep (see _rows_from_csv).
        csv_path = tmp_path / "provided.csv"
        csv_path.write_text(
            "episode_id,step,t_s,cell_id,prb_util_dl_p50,ue_id,offered_mbps,delivered_mbps,bler,sinr_db\n"
            "e0,0,0.0,0,0.5,0,10,9,0.05,12\n"
            "e0,0,0.0,0,0.5,1,10,8,0.05,10\n"
            "e0,1,5.0,0,0.6,0,10,9.5,0.04,12\n"
            "e0,1,5.0,0,0.6,1,10,8.5,0.05,10\n"
        )
        episodes = load_provided_dataset(csv_path)
        assert list(episodes) == ["e0"]
        assert len(episodes["e0"].observations) == 2
        assert len(episodes["e0"].observations[0].cells[0].ues) == 2


class TestTraceLoader:
    def test_trace_fixture_is_detected_and_parsed(self):
        episodes = load_provided_dataset(TRACE_FIXTURE)
        assert sorted(episodes) == ["ep_000341", "ep_000342"]
        assert len(episodes["ep_000341"].observations) == 3
        assert len(episodes["ep_000342"].observations) == 3
        for source in episodes.values():
            for obs in source.observations:
                assert isinstance(obs, Observation)

    def test_aggregates_reconstruct_recorded_measurements(self):
        # Step 1 of ep_000341 carries the full measurement set; the
        # reconstructed observation must reproduce every
        # aggregate that trace row recorded.
        rows = [json.loads(line) for line in TRACE_FIXTURE.open()]
        recorded = next(r for r in rows if r["episode_id"] == "ep_000341" and r["step"] == 1)["reward_measurements"]
        obs = load_provided_dataset(TRACE_FIXTURE)["ep_000341"].observations[1]
        cell = obs.cells[0]
        assert len(obs.cells) == 1
        assert len(cell.ues) == int(recorded["n_ues"])
        assert sum(ue.delivered_mbps for ue in cell.ues) == pytest.approx(recorded["aggregate_delivered_mbps"])
        assert cell.fairness_jain == pytest.approx(recorded["mean_jain_fairness"])
        assert cell.sla_violations_last_window == int(recorded["sla_violations"])
        assert cell.prb_util_dl_p99 == pytest.approx(0.85 + 0.15 * recorded["prb_pressure"])
        assert cell.prach_collision_rate == pytest.approx(0.05 + 0.45 * recorded["access_pressure"])

    def test_recomputed_reward_measurements_match_the_trace(self):
        # Re-running compute_breakdown over a reconstructed (prev, curr) pair
        # must reproduce the trace's aggregate-level measurements. Per-UE
        # quantities (elastic Jain fairness) are flattened and excluded.
        from openair_congestion import rewards

        rows = [json.loads(line) for line in TRACE_FIXTURE.open()]
        recorded = next(r for r in rows if r["episode_id"] == "ep_000341" and r["step"] == 1)["reward_measurements"]
        episodes = load_provided_dataset(TRACE_FIXTURE)
        breakdown = rewards.compute_breakdown(
            prev_obs=episodes["ep_000341"].observations[0],
            curr_obs=episodes["ep_000341"].observations[1],
            action=NOOP,
            rejected=False,
        )
        recomputed = breakdown["measurements"]
        for key in (
            "sla_violations",
            "aggregate_delivered_mbps",
            "mean_jain_fairness",
            "prb_pressure",
            "access_pressure",
            "fairness_deficit",
            "buffer_pressure",
            "n_ues",
        ):
            assert recomputed[key] == pytest.approx(recorded[key]), key

    def test_sparse_trace_row_defaults_to_uncongested(self):
        # Step 0 of ep_000342 carries only the required measurement keys:
        # pressures default to 0, fairness passes through.
        obs = load_provided_dataset(TRACE_FIXTURE)["ep_000342"].observations[0]
        cell = obs.cells[0]
        assert cell.prb_util_dl_p99 == pytest.approx(0.85)  # zero pressure
        assert cell.prach_collision_rate == 0.0
        assert cell.sla_violations_last_window == 0
        assert cell.fairness_jain == pytest.approx(0.95)
        assert all(ue.buffer_occupancy_kb == 0.0 for ue in cell.ues)

    def test_trace_row_missing_measurements_fails_fast(self, tmp_path):
        bad = tmp_path / "trace.jsonl"
        rows = [
            {
                "episode_id": "e",
                "step": 0,
                "tool_sent": {"name": "noop", "arguments": {}},
                "reward_measurements": {"aggregate_delivered_mbps": 10.0, "n_ues": 2},
            },
            {"episode_id": "e", "step": 1, "tool_sent": {"name": "noop", "arguments": {}}},
        ]
        bad.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        with pytest.raises(ValueError, match="reward_measurements"):
            load_provided_dataset(bad)

    def test_trace_row_missing_required_key_names_it(self, tmp_path):
        bad = tmp_path / "trace.jsonl"
        row = {"episode_id": "e", "step": 0, "reward_measurements": {"aggregate_delivered_mbps": 10.0}}
        bad.write_text(json.dumps(row) + "\n")
        with pytest.raises(ValueError, match="n_ues"):
            load_provided_dataset(bad)

    def test_recorded_cell_capacity_is_loaded_per_observation(self):
        episodes = load_provided_dataset(TRACE_FIXTURE)
        assert episodes["ep_000341"].cell_capacity_mbps_by_observation == (
            120.0,
            120.0,
            120.0,
        )
        assert episodes["ep_000342"].cell_capacity_mbps_by_observation == (
            None,
            120.0,
            120.0,
        )
        # Snapshot data records no capacity; the config knob applies.
        assert load_provided_dataset(SNAPSHOT_FIXTURE)["lab_run_a"].cell_capacity_mbps_by_observation == (
            None,
            None,
            None,
            None,
        )

    def test_replaying_recorded_actions_reproduces_recorded_rewards(self):
        # With the recorded action, guardrail outcome, and
        # cell_capacity_mbps_total, compute_breakdown over reconstructed
        # pairs reproduces the trace's per-step reward exactly.
        from openair_congestion import rewards

        rows = [json.loads(line) for line in TRACE_FIXTURE.open()]
        episodes = load_provided_dataset(TRACE_FIXTURE)
        for key in ("ep_000341", "ep_000342"):
            obs = episodes[key].observations
            for step in (1, 2):
                row = next(r for r in rows if r["episode_id"] == key and r["step"] == step)
                breakdown = rewards.compute_breakdown(
                    prev_obs=obs[step - 1],
                    curr_obs=obs[step],
                    action=ToolCall(**row["tool_sent"]),
                    rejected=row["rejected"],
                    cell_capacity_mbps=episodes[key].cell_capacity_mbps_by_observation[step],
                )
                assert breakdown["total"] == pytest.approx(row["reward"]), (key, step)

    def test_backend_step_uses_recorded_capacity(self):
        # ep_000341's rows record cell_capacity_mbps_total=120; replaying the
        # recorded (accepted) actions through the backend, with the config
        # knob left at its 60.0 default, reproduces the recorded rewards.
        rows = [json.loads(line) for line in TRACE_FIXTURE.open()]
        backend = _make_backend(dataset_path=str(TRACE_FIXTURE))
        _, meta = backend.reset({"scenario_id": "ep_000341"})
        for step in (1, 2):
            row = next(r for r in rows if r["episode_id"] == "ep_000341" and r["step"] == step)
            _, reward, _, info = backend.step(meta.episode_id, ToolCall(**row["tool_sent"]))
            assert info["guardrail_accepted"] is True
            assert info["cell_capacity_mbps"] == pytest.approx(120.0)
            assert info["cell_capacity_mbps_per_cell"] == pytest.approx(120.0)
            assert info["cell_capacity_mbps_total"] == pytest.approx(120.0)
            assert info["cell_capacity_source"] == "recorded_transition_total"
            assert reward == pytest.approx(row["reward"])
        backend.close(meta.episode_id)

    def test_backend_replays_trace_episodes(self):
        backend = _make_backend(dataset_path=str(TRACE_FIXTURE))
        first_obs, meta = backend.reset({"scenario_id": "ep_000342"})
        assert meta.scenario_id == "ep_000342"
        assert meta.max_steps == 2  # 3 trace rows -> 2 steps
        assert len(first_obs.cells[0].ues) == 6
        for expected_idx in (1, 2):
            obs, reward, done, info = backend.step(meta.episode_id, NOOP)
            assert math.isfinite(reward)
            assert info["step_idx"] == expected_idx
            assert info["dynamics_mode"] == DATASET_DYNAMICS_MODE
            assert info["action_affects_observation"] is False
            assert done is (expected_idx == 2)
        backend.close(meta.episode_id)

    def test_reused_episode_ids_are_namespaced_by_iteration(self, tmp_path):
        path = tmp_path / "merged_trace.jsonl"

        def row(iteration: int, step: int, delivered: float) -> dict:
            return {
                "iter": iteration,
                "episode_id": "shared",
                "step": step,
                "tool_sent": {"name": "noop", "arguments": {}},
                "reward_measurements": {
                    "aggregate_delivered_mbps": delivered,
                    "n_ues": 1,
                    "cell_capacity_mbps_total": 100.0,
                },
            }

        rows = [row(0, 0, 10.0), row(1, 0, 20.0), row(0, 1, 11.0), row(1, 1, 21.0)]
        path.write_text("\n".join(json.dumps(item) for item in rows) + "\n")
        episodes = load_provided_dataset(path)
        assert sorted(episodes) == ["iter_0::shared", "iter_1::shared"]
        assert episodes["iter_0::shared"].observations[1].cells[0].ues[0].delivered_mbps == 11.0
        assert episodes["iter_1::shared"].observations[1].cells[0].ues[0].delivered_mbps == 21.0

    def test_duplicate_step_within_iteration_is_rejected(self, tmp_path):
        path = tmp_path / "duplicate_trace.jsonl"
        row = {
            "iter": 0,
            "episode_id": "shared",
            "step": 0,
            "tool_sent": {"name": "noop", "arguments": {}},
            "reward_measurements": {"aggregate_delivered_mbps": 10.0, "n_ues": 1},
        }
        path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
        with pytest.raises(ValueError, match="duplicate step"):
            load_provided_dataset(path)

    def test_transition_uses_capacity_from_destination_row(self, tmp_path):
        path = tmp_path / "changing_capacity.jsonl"
        rows = []
        for step, capacity in enumerate((100.0, 200.0, 400.0)):
            rows.append(
                {
                    "iter": 0,
                    "episode_id": "changing",
                    "step": step,
                    "tool_sent": {"name": "noop", "arguments": {}},
                    "reward_measurements": {
                        "aggregate_delivered_mbps": 10.0 + step,
                        "n_ues": 1,
                        "cell_capacity_mbps_total": capacity,
                    },
                }
            )
        path.write_text("\n".join(json.dumps(item) for item in rows) + "\n")
        backend = _make_backend(
            dataset_path=str(path),
            reward_profile="openair_v2_measured",
            reward_weights={"w_sla": 0.0, "w_sla_level": 0.0, "w_buffer": 0.0, "w_action": 0.0},
        )
        _, meta = backend.reset({"scenario_id": "changing"})
        _, _, _, first_info = backend.step(meta.episode_id, NOOP)
        _, _, _, second_info = backend.step(meta.episode_id, NOOP)
        assert first_info["cell_capacity_mbps"] == pytest.approx(200.0)
        assert second_info["cell_capacity_mbps"] == pytest.approx(400.0)
        assert first_info["reward_profile"] == "openair_v2_measured"
        assert first_info["reward_weights"]["w_sla"] == 0.0
        assert first_info["action_affects_observation"] is False
        backend.close(meta.episode_id)

    def test_nonpositive_or_nonfinite_recorded_capacity_is_rejected(self, tmp_path):
        path = tmp_path / "bad_capacity.jsonl"
        rows = [
            {
                "episode_id": "e",
                "step": step,
                "tool_sent": {"name": "noop", "arguments": {}},
                "reward_measurements": {
                    "aggregate_delivered_mbps": 10.0,
                    "n_ues": 1,
                    "cell_capacity_mbps_total": capacity,
                },
            }
            for step, capacity in enumerate((100.0, 0.0))
        ]
        path.write_text("\n".join(json.dumps(item) for item in rows) + "\n")
        with pytest.raises(ValueError, match="finite and positive"):
            load_provided_dataset(path)

    def test_t1_trace_preserves_topology_initial_history_and_receipts(self, tmp_path):
        path = tmp_path / "amparo_like.jsonl"

        def row(step: int, action: dict, accepted: bool, delivered: float) -> dict:
            return {
                "iter": 0,
                "episode_id": "amparo",
                "step": step,
                "scenario_mode": "t1_runner",
                "kpi_source": "runner_snapshot",
                "tool_sent": action,
                "guardrail_accepted": accepted,
                "rejected": not accepted,
                "reward_measurements": {
                    "aggregate_delivered_mbps": delivered,
                    "n_ues": 4.0,
                    "mean_jain_fairness": 0.7264269868637496,
                    "fairness_deficit": 0.18663647693240693,
                    "prb_pressure": 0.5466413043478261,
                    "access_pressure": 0.0,
                    "buffer_pressure": 0.6207570070635575,
                    "sla_violations": 1.0,
                    "cell_capacity_mbps_total": 150.0,
                },
            }

        cell0 = {
            "name": "set_prb_cap",
            "arguments": {"cell_id": 0, "target": "ue", "target_id": 0, "max_prb": 10},
        }
        cell1 = {
            "name": "set_prb_cap",
            "arguments": {"cell_id": 1, "target": "ue", "target_id": 1, "max_prb": 10},
        }
        rows = [
            row(0, cell0, True, 22.753),
            row(1, cell1, True, 8.582),
            row(2, cell0, False, 8.582),
            row(3, cell0, True, 8.582),
        ]
        path.write_text("\n".join(json.dumps(item) for item in rows) + "\n")

        source = load_provided_dataset(path)["amparo"]
        first = source.observations[0]
        assert first.global_.n_cells == 2
        assert first.global_.n_ues_total == 4
        assert [len(cell.ues) for cell in first.cells] == [2, 2]
        assert sum(ue.delivered_mbps for cell in first.cells for ue in cell.ues) == pytest.approx(22.753)
        assert sum(cell.sla_violations_last_window for cell in first.cells) == 1
        assert sum(cell.fairness_jain for cell in first.cells) / 2 == pytest.approx(0.7264269868637496)
        assert sum(max(0.0, 0.8 - cell.fairness_jain) / 0.8 for cell in first.cells) / 2 == pytest.approx(
            0.18663647693240693
        )

        backend = _make_backend(
            dataset_path=str(path),
            reward_profile="openair_v2_measured",
            reward_weights={"w_sla": 0.0, "w_sla_level": 0.0, "w_buffer": 0.0, "w_action": 0.0},
        )
        initial_obs, meta = backend.reset({"scenario_id": "amparo"})
        assert initial_obs.agent_aux.last_action.model_dump() == cell0
        assert initial_obs.agent_aux.last_rejection is None
        compact = _to_resource_compact_pipe_v1(initial_obs)
        assert 'L|set_prb_cap|{"cell_id":0,"max_prb":10,"target":"ue","target_id":0}|none' in compact

        # The seeded row-0 history is now observable before it can reject the
        # same first-turn action.
        _, _, _, repeated_info = backend.step(meta.episode_id, ToolCall(**cell0))
        assert repeated_info["guardrail_accepted"] is False
        assert "identical-action repeat" in repeated_info["rejection_reason"]
        backend.close(meta.episode_id)

        _, meta = backend.reset({"scenario_id": "amparo"})
        reset_receipt = backend.episode_receipt_info(meta.episode_id)
        assert reset_receipt["reconstruction_schema"] == "aggregate_trace_multicell_proxy_v1"
        assert reset_receipt["reconstruction_n_cells"] == 2
        assert reset_receipt["reconstruction_ue_counts"] == [2, 2]
        assert reset_receipt["dataset_row_count"] == 4
        assert reset_receipt["dataset_episode_count"] == 1
        assert reset_receipt["dataset_identity"] == f"sha256:{reset_receipt['dataset_sha256']}"

        observed_acceptance = []
        for expected in rows[1:]:
            _, _, _, info = backend.step(meta.episode_id, ToolCall(**expected["tool_sent"]))
            observed_acceptance.append(info["guardrail_accepted"])
            assert info["cell_capacity_mbps_per_cell"] == pytest.approx(75.0)
            assert info["cell_capacity_mbps_total"] == pytest.approx(150.0)
            assert info["reconstruction_n_cells"] == 2
        assert observed_acceptance == [True, False, True]
        backend.close(meta.episode_id)

    def test_accepted_action_conflicting_with_scenario_topology_fails(self, tmp_path):
        path = tmp_path / "conflict.jsonl"
        rows = [
            {
                "episode_id": "e",
                "step": step,
                "scenario_mode": "t1_runner",
                "tool_sent": {
                    "name": "set_scheduler_policy",
                    "arguments": {"cell_id": 2, "policy": "PF"},
                },
                "guardrail_accepted": True,
                "reward_measurements": {"aggregate_delivered_mbps": 10.0, "n_ues": 4},
            }
            for step in range(2)
        ]
        path.write_text("\n".join(json.dumps(item) for item in rows) + "\n")
        with pytest.raises(ValueError, match="requires 3 cells.*declares 2"):
            load_provided_dataset(path)


class TestDatasetReplayBackend:
    def test_reset_serves_provided_first_observation(self):
        backend = _make_backend()
        first_obs, meta = backend.reset({"scenario_id": "lab_run_a", "seed": 7})
        assert meta.scenario_id == "lab_run_a"
        assert meta.seed == 7
        assert meta.max_steps == 3  # 4 provided observations -> 3 steps
        assert first_obs.episode_id == meta.episode_id
        # Values come from the dataset, not seed-driven synthesis.
        assert first_obs.cells[0].prb_util_dl_p50 == pytest.approx(0.55)
        assert first_obs.cells[0].ues[1].delivered_mbps == pytest.approx(14.0)
        backend.close(meta.episode_id)

    def test_seed_maps_deterministically_when_no_scenario_id(self):
        backend = _make_backend()
        _, meta0 = backend.reset({"seed": 0})
        _, meta1 = backend.reset({"seed": 1})
        _, meta2 = backend.reset({"seed": 2})
        assert meta0.scenario_id == "lab_run_a"  # sorted keys[0 % 2]
        assert meta1.scenario_id == "lab_run_b"  # sorted keys[1 % 2]
        assert meta2.scenario_id == "lab_run_a"  # wraps
        for meta in (meta0, meta1, meta2):
            backend.close(meta.episode_id)

    def test_dataset_index_cycles_independently_of_seed(self):
        backend = _make_backend()
        _, meta0 = backend.reset({"seed": 999, "dataset_index": 0})
        _, meta1 = backend.reset({"seed": 999, "dataset_index": 1})
        _, meta2 = backend.reset({"seed": 999, "dataset_index": 2})
        assert meta0.scenario_id == "lab_run_a"
        assert meta1.scenario_id == "lab_run_b"
        assert meta2.scenario_id == "lab_run_a"
        for meta in (meta0, meta1, meta2):
            backend.close(meta.episode_id)

    @pytest.mark.parametrize("dataset_index", [-1, True, 1.5, "abc", float("inf"), float("nan")])
    def test_invalid_dataset_index_is_rejected(self, dataset_index):
        backend = _make_backend()
        with pytest.raises(ValueError, match="dataset_index"):
            backend.reset({"dataset_index": dataset_index})

    def test_step_replays_provided_data_and_computes_reward(self):
        backend = _make_backend()
        _, meta = backend.reset({"scenario_id": "lab_run_a"})
        for expected_idx, expected_p50 in ((1, 0.70), (2, 0.92), (3, 0.80)):
            obs, reward, done, info = backend.step(meta.episode_id, NOOP)
            assert math.isfinite(reward)
            assert info["step_idx"] == expected_idx
            assert info["kpi_source"] == "dataset_replay"
            assert info["dynamics_mode"] == DATASET_DYNAMICS_MODE
            assert info["guardrail_accepted"] is True
            # Pass-through: KPIs are the recorded row, untouched by the action.
            assert obs.cells[0].prb_util_dl_p50 == pytest.approx(expected_p50)
            # agent_aux is stamped like the other backends.
            assert obs.agent_aux.step_idx == expected_idx
            assert obs.agent_aux.last_action.name == "noop"
            assert obs.agent_aux.last_reward == pytest.approx(reward)
            assert done is (expected_idx == 3)
        summary = backend.close(meta.episode_id)
        assert summary == {"ok": True, "n_steps": 3}

    def test_reward_breakdown_matches_rewards_module(self):
        # The reward must be rewards.compute_breakdown over the served
        # (prev, curr) pair, nothing else.
        from openair_congestion import rewards

        backend = _make_backend()
        episodes = load_provided_dataset(SNAPSHOT_FIXTURE)
        first_obs, meta = backend.reset({"scenario_id": "lab_run_a"})
        _, reward, _, info = backend.step(meta.episode_id, NOOP)
        expected = rewards.compute_breakdown(
            prev_obs=episodes["lab_run_a"].observations[0],
            curr_obs=episodes["lab_run_a"].observations[1],
            action=NOOP,
            rejected=False,
            cell_capacity_mbps=60.0,
        )
        assert reward == pytest.approx(float(expected["total"]))
        assert info["reward_terms"]["total"] == pytest.approx(float(expected["total"]))
        backend.close(meta.episode_id)

    def test_out_of_range_action_rejected_not_crashed(self):
        # Guardrail semantics survive: cell_id 3 does not exist in a 1-cell
        # episode -> rejected with the standard penalty, env intact.
        backend = _make_backend()
        _, meta = backend.reset({"scenario_id": "lab_run_a"})
        obs, reward, done, info = backend.step(
            meta.episode_id,
            ToolCall(name="set_scheduler_policy", arguments={"cell_id": 3, "policy": "PF"}),
        )
        assert info["guardrail_accepted"] is False
        assert info["rejection_reason"]
        assert math.isfinite(reward)
        assert done is False
        backend.close(meta.episode_id)

    def test_pool_exhaustion_reaps_orphans_then_raises(self):
        backend = _make_backend(pool_size=2)
        _, meta_a = backend.reset({"seed": 0})
        _, meta_b = backend.reset({"seed": 1})
        # Pool full; meta_a is not in live_episode_ids -> reaped, reset works.
        _, meta_c = backend.reset({"seed": 2}, live_episode_ids={meta_b.episode_id})
        assert meta_c.episode_id != meta_a.episode_id
        with pytest.raises(KeyError):
            backend.step(meta_a.episode_id, NOOP)  # reaped
        # Both slots live now -> exhausted.
        with pytest.raises(RuntimeError, match="pool exhausted"):
            backend.reset(
                {"seed": 3},
                live_episode_ids={meta_b.episode_id, meta_c.episode_id},
            )

    def test_task_max_steps_clamped_to_provided_length(self):
        backend = _make_backend()
        _, meta = backend.reset({"scenario_id": "lab_run_b", "max_steps": 50})
        assert meta.max_steps == 2  # 3 provided observations -> 2 steps max
        backend.close(meta.episode_id)

    def test_unknown_scenario_id_lists_available(self):
        backend = _make_backend()
        with pytest.raises(KeyError, match="lab_run_a"):
            backend.reset({"scenario_id": "does_not_exist"})


class TestSelectBackend:
    def test_config_only_switch(self, monkeypatch):
        monkeypatch.delenv("OPENAIR_CONGESTION_BACKEND", raising=False)
        config = SimpleNamespace(
            backend="dataset_replay",
            dataset_path=str(SNAPSHOT_FIXTURE),
            pool_size=4,
            max_steps_default=60,
            cell_capacity_mbps=60.0,
        )
        backend = select_backend(config)
        assert isinstance(backend, DatasetReplayBackend)
        assert backend.dataset_path == SNAPSHOT_FIXTURE

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("OPENAIR_CONGESTION_BACKEND", "dataset_replay")
        config = SimpleNamespace(backend="replay", dataset_path=str(SNAPSHOT_FIXTURE))
        assert isinstance(select_backend(config), DatasetReplayBackend)
