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

"""Dataset-replay backend: serve a recorded KPI dataset through the Backend
contract.

``ReplayBackend`` synthesizes trajectories from seeds; this backend replays a
dataset file instead. Same reset/step/close contract, so switching is
config-only (``backend: dataset_replay``).

Two JSONL row formats are accepted and auto-detected from the first row (see
the README for the column contract of each):

- KPI snapshots: one row per timestep with nested ``cells[]`` / ``ues[]``
  telemetry. Missing optional KPI fields are synthesized with the same
  heuristics the live env uses (``openair_congestion/env.py``), so a sparse
  dataset still yields the stable training shape.
- GRPO rollout traces: one row per policy step carrying a
  ``reward_measurements`` dict (the aggregates emitted by
  ``rewards.compute_breakdown``). Each trace row is reconstructed over an
  inferred multi-cell topology whose aggregate KPIs — delivered throughput, mean
  Jain fairness, PRB/access/buffer pressure, SLA violation count — reproduce
  the recorded measurements, and a recorded ``cell_capacity_mbps_total``
  keeps each transition's reward normalizer at the recorded scale. Per-UE
  structure is not recoverable from aggregates: throughput is spread evenly
  across ``n_ues`` identical UEs, so per-UE quantities (elastic Jain
  fairness, individual buffers, 5QI mix) are flattened.

Actions are pass-through in both formats: the data is pre-recorded, so
``step()`` advances a pointer and does not mutate KPIs. The guardrail still
runs (rejected actions earn the same penalty semantics as ReplayEnv) and the
reward is computed over the served observation pair via the unchanged
``rewards.compute_breakdown(prev_obs, curr_obs, action, rejected=...)``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import threading
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

# backends guards the cross-repo 'openair_congestion' import; keep it ahead of
# the telco imports so a missing install fails with the pip hint.
from resources_servers.openair_congestion.backends import Backend, validate_reward_profile


# isort: split
from openair_congestion import guardrail as _guardrail
from openair_congestion import rewards as _rewards
from openair_congestion.schemas import (
    AgentAux,
    EpisodeMeta,
    LastActionEcho,
    Observation,
    ToolCall,
)


# Stamped into step() info["dynamics_mode"] so trainers can tell recorded-data
# rollouts apart from ReplayEnv's synthetic action-effect model.
DATASET_DYNAMICS_MODE = "provided_data_passthrough_v1"

# Aggregate-only GRPO traces do not carry the original nested observation.
# This versioned label makes that lossy reconstruction explicit in receipts.
TRACE_RECONSTRUCTION_SCHEMA = "aggregate_trace_multicell_proxy_v1"
NATIVE_SNAPSHOT_SCHEMA = "native_snapshot_v1"

# Known runner layouts. Accepted action metadata is the primary topology
# evidence; this mapping validates it and supplies a deterministic fallback for
# episodes whose recorded policy happened not to address every cell.
_SCENARIO_N_CELLS = {
    "t1_runner": 2,
    "t2_runner": 3,
}

# Schema bound on total UEs per observation (tools.MAX_UES).
_MAX_UES = 24


@dataclass(frozen=True)
class TraceTopology:
    """Auditable topology inferred for one aggregate trace episode."""

    n_cells: int
    scenario_mode: Optional[str]
    source: str


def _recorded_action_accepted(row: dict[str, Any]) -> Optional[bool]:
    accepted = row.get("guardrail_accepted")
    if accepted is not None:
        if not isinstance(accepted, bool):
            raise ValueError(f"guardrail_accepted must be bool, got {accepted!r}")
        return accepted
    rejected = row.get("rejected")
    if rejected is not None:
        if not isinstance(rejected, bool):
            raise ValueError(f"rejected must be bool, got {rejected!r}")
        return not rejected
    return None


def _recorded_tool_call(row: dict[str, Any]) -> Optional[ToolCall]:
    raw = row.get("tool_sent")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"tool_sent must be an object, got {type(raw).__name__}")
    return ToolCall.model_validate(raw)


def _trace_n_ues(row: dict[str, Any]) -> int:
    measurements = row.get("reward_measurements")
    if not isinstance(measurements, dict) or "n_ues" not in measurements:
        raise ValueError("trace row reward_measurements is missing required key 'n_ues'")
    value = measurements["n_ues"]
    if isinstance(value, bool):
        raise ValueError(f"reward_measurements.n_ues must be an integer, got {value!r}")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"reward_measurements.n_ues must be an integer, got {value!r}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric <= 0 or numeric > _MAX_UES:
        raise ValueError(f"reward_measurements.n_ues must be an integer in [1,{_MAX_UES}], got {value!r}")
    return int(numeric)


def _ue_counts(n_ues: int, n_cells: int) -> tuple[int, ...]:
    if n_ues < n_cells:
        raise ValueError(f"cannot reconstruct {n_cells} non-empty cells from only {n_ues} aggregate UEs")
    quotient, remainder = divmod(n_ues, n_cells)
    return tuple(quotient + (1 if cell_id < remainder else 0) for cell_id in range(n_cells))


def _infer_trace_topology(rows: list[dict[str, Any]]) -> TraceTopology:
    """Infer topology from accepted actions, checked against runner metadata."""
    scenario_modes = {
        str(row["scenario_mode"]).strip().lower() for row in rows if row.get("scenario_mode") not in (None, "")
    }
    if len(scenario_modes) > 1:
        raise ValueError(f"trace episode has conflicting scenario_mode values: {sorted(scenario_modes)}")
    scenario_mode = next(iter(scenario_modes), None)
    mapped_n_cells = _SCENARIO_N_CELLS.get(scenario_mode or "")

    max_accepted_cell_id = -1
    parsed_actions: list[tuple[dict[str, Any], Optional[ToolCall], Optional[bool]]] = []
    for row in rows:
        action = _recorded_tool_call(row)
        accepted = _recorded_action_accepted(row)
        parsed_actions.append((row, action, accepted))
        if not accepted or action is None or action.name == "noop":
            continue
        cell_id = action.arguments.get("cell_id")
        if isinstance(cell_id, bool) or not isinstance(cell_id, int) or cell_id < 0:
            raise ValueError(f"accepted recorded action has invalid cell_id={cell_id!r}")
        max_accepted_cell_id = max(max_accepted_cell_id, cell_id)

    action_n_cells = max_accepted_cell_id + 1 if max_accepted_cell_id >= 0 else None
    if mapped_n_cells is not None and action_n_cells is not None and action_n_cells > mapped_n_cells:
        raise ValueError(
            f"accepted action metadata requires {action_n_cells} cells but "
            f"scenario_mode={scenario_mode!r} declares {mapped_n_cells}"
        )
    n_cells = mapped_n_cells or action_n_cells or 1
    source_parts = []
    if action_n_cells is not None:
        source_parts.append("accepted_tool_metadata")
    if mapped_n_cells is not None:
        source_parts.append(f"scenario_mode:{scenario_mode}")
    if not source_parts:
        source_parts.append("single_cell_fallback")

    # Validate every accepted UE target against the deterministic per-row
    # distribution. Rejected actions are intentionally not topology evidence.
    for row, action, accepted in parsed_actions:
        counts = _ue_counts(_trace_n_ues(row), n_cells)
        if not accepted or action is None or action.name != "set_prb_cap":
            continue
        args = action.arguments
        cell_id = args.get("cell_id")
        if not isinstance(cell_id, int) or isinstance(cell_id, bool) or not 0 <= cell_id < n_cells:
            raise ValueError(
                f"accepted set_prb_cap cell_id={cell_id!r} conflicts with inferred {n_cells}-cell topology"
            )
        if args.get("target") == "ue":
            target_id = args.get("target_id")
            if not isinstance(target_id, int) or isinstance(target_id, bool) or not 0 <= target_id < counts[cell_id]:
                raise ValueError(
                    f"accepted set_prb_cap target_id={target_id!r} conflicts with "
                    f"cell {cell_id} UE count {counts[cell_id]}"
                )

    return TraceTopology(
        n_cells=n_cells,
        scenario_mode=scenario_mode,
        source="+".join(source_parts),
    )


# --- KPI-snapshot rows -> Observation ----------------------------------------
#
# Field-synthesis heuristics below mirror the live env's observation builder
# (openair_congestion/env.py::_build_observation) so a sparse dataset row
# produces the same derived KPIs the env would produce.


def _jain(values: list[float]) -> float:
    """Jain fairness index (same as env.py::_jain)."""
    if not values or all(v <= 0.0 for v in values):
        return 1.0
    s = sum(values)
    n = len(values)
    sq = sum(v * v for v in values)
    return float((s * s) / max(1e-9, n * sq))


def _num(raw: dict[str, Any], key: str, default: float) -> float:
    value = raw.get(key)
    if value is None:
        return float(default)
    return float(value)


def _parse_ue(raw: dict[str, Any], ue_idx: int) -> dict[str, Any]:
    """Parse one UE record; synthesize any missing optional field.

    ``delivered_mbps`` is required; everything else falls back to the env's
    defaults (sinr=10.0, bler=0.0) or derivation heuristics.
    """
    if "delivered_mbps" not in raw:
        raise ValueError(
            f"dataset UE record #{ue_idx} is missing required field 'delivered_mbps'; got keys {sorted(raw)}"
        )
    delivered = max(0.0, float(raw["delivered_mbps"]))
    offered = max(0.0, _num(raw, "offered_mbps", max(delivered, 1.0)))
    sinr = min(40.0, max(-20.0, _num(raw, "sinr_db", 10.0)))
    bler = min(1.0, max(0.0, _num(raw, "bler", 0.0)))
    # env.py heuristics: mcs from SINR, backlog from offered - delivered,
    # PDB violation when backlog exceeds 500 kB.
    mcs_mean = _num(raw, "mcs_mean", max(0.0, min(27.0, (max(sinr, -10.0) + 5.0) * 1.2)))
    buffer_kb = max(0.0, _num(raw, "buffer_occupancy_kb", max(0.0, (offered - delivered) * 50.0)))
    pdb = int(_num(raw, "pdb_violations", 1 if buffer_kb > 500.0 else 0))
    return {
        "ue_id": int(raw.get("ue_id", ue_idx)),
        "offered_mbps": offered,
        "delivered_mbps": delivered,
        "bler": bler,
        "mcs_mean": max(0.0, min(27.0, mcs_mean)),
        "sinr_db": sinr,
        "buffer_occupancy_kb": buffer_kb,
        "pdb_violations": pdb,
        # Both the JSON alias '5qi' and the field name 'qos_5qi' are accepted.
        "5qi": int(raw.get("5qi", raw.get("qos_5qi", 9))),
    }


def _parse_cell(raw: dict[str, Any], cell_idx: int) -> dict[str, Any]:
    """Parse one cell record; synthesize any missing optional field.

    ``prb_util_dl_p50`` and a non-empty ``ues`` list are required; everything
    else falls back to the env's derivation heuristics.
    """
    if "prb_util_dl_p50" not in raw:
        raise ValueError(
            f"dataset cell record #{cell_idx} is missing required field 'prb_util_dl_p50'; got keys {sorted(raw)}"
        )
    ues_raw = raw.get("ues") or []
    if not ues_raw:
        raise ValueError(f"dataset cell record #{cell_idx} has no 'ues' entries")
    ues = [_parse_ue(ue, i) for i, ue in enumerate(ues_raw)]

    p50 = min(1.0, max(0.0, float(raw["prb_util_dl_p50"])))
    # Heuristics mirror env.py exactly (see schemas.KPI_PROVENANCE_V1 notes).
    p99 = _num(raw, "prb_util_dl_p99", max(p50, min(1.0, p50 * 1.15 + 0.02)))
    p99 = max(p50, min(1.0, p99))  # schema invariant: p99 >= p50
    ul_p50 = min(1.0, max(0.0, _num(raw, "prb_util_ul_p50", min(1.0, p50 * 0.4))))
    sched_latency = max(0.0, _num(raw, "sched_latency_ms_p99", 5.0 + 20.0 * p99))
    n_ues = int(_num(raw, "rrc_connected_ues", len(ues)))
    n_ues = max(0, min(_MAX_UES, n_ues))
    prach = _num(
        raw,
        "prach_collision_rate",
        0.0 if n_ues < 8 else min(0.5, 0.01 * (n_ues - 8) ** 2),
    )
    fairness = _num(raw, "fairness_jain", _jain([u["delivered_mbps"] for u in ues]))
    sla = int(
        _num(
            raw,
            "sla_violations_last_window",
            sum(1 for u in ues if u["pdb_violations"] > 0),
        )
    )
    return {
        "cell_id": int(raw.get("cell_id", cell_idx)),
        "prb_util_dl_p50": p50,
        "prb_util_dl_p99": p99,
        "prb_util_ul_p50": ul_p50,
        "sched_latency_ms_p99": sched_latency,
        "rrc_connected_ues": n_ues,
        "prach_collision_rate": min(1.0, max(0.0, prach)),
        "fairness_jain": min(1.0, max(0.0, fairness)),
        "sla_violations_last_window": max(0, sla),
        "ues": ues,
    }


def row_to_observation(
    row: dict[str, Any],
    *,
    step_idx: int,
    episode_id: str,
) -> Observation:
    """Validate one dataset row into a frozen, Pydantic-valid Observation.

    Raises ``ValueError`` or ``TypeError`` with a field-level message; the
    loader wraps either with the row's line number.
    """
    cells_raw = row.get("cells") or []
    if not cells_raw:
        raise ValueError(f"dataset row has no 'cells' entries; got keys {sorted(row)}")
    cells = [_parse_cell(c, i) for i, c in enumerate(cells_raw)]

    global_raw = row.get("global") or {}
    payload: dict[str, Any] = {
        "t_s": max(0.0, _num(row, "t_s", float(step_idx))),
        "episode_id": episode_id,
        "cells": cells,
        "global": {
            "n_cells": int(global_raw.get("n_cells", len(cells))),
            "n_ues_total": int(global_raw.get("n_ues_total", sum(len(c["ues"]) for c in cells))),
            "difficulty": float(global_raw.get("difficulty", 0.5)),
            "regime_mix": global_raw.get("regime_mix") or {},
            "tier": global_raw.get("tier", "replay"),
        },
        # Default 'replay' keeps kpi_provenance honest (fields stamped
        # 'synthetic'). Lab-measured rows should say so via kpi_source_mode
        # (e.g. 'runner_snapshot'); the schema's provenance auto-fill handles
        # the rest.
        "kpi_source_mode": str(row.get("kpi_source_mode", "replay")),
    }
    return Observation.model_validate(payload)


# --- GRPO trace rows -> snapshot rows -----------------------------------------


def is_trace_row(row: dict[str, Any]) -> bool:
    """A row carrying ``tool_sent`` or ``reward_measurements`` is a trace row."""
    return "tool_sent" in row or "reward_measurements" in row


def _fairness_by_cell(measurements: dict[str, Any], n_cells: int) -> tuple[float, ...]:
    """Reconstruct cell fairness while preserving mean and deficit aggregates."""
    mean = _num(measurements, "mean_jain_fairness", 1.0)
    if not math.isfinite(mean) or not 0.0 <= mean <= 1.0:
        raise ValueError(f"mean_jain_fairness must be finite and in [0,1], got {mean!r}")
    raw_deficit = measurements.get("fairness_deficit")
    if raw_deficit is None:
        return (mean,) * n_cells
    deficit = float(raw_deficit)
    if not math.isfinite(deficit) or not 0.0 <= deficit <= 1.0:
        raise ValueError(f"fairness_deficit must be finite and in [0,1], got {raw_deficit!r}")

    target = 0.8
    total_fairness = n_cells * mean
    total_shortfall = n_cells * target * deficit
    tolerance = 1e-8
    for below_count in range(n_cells + 1):
        above_count = n_cells - below_count
        below_sum = below_count * target - total_shortfall
        above_sum = total_fairness - below_sum
        if below_count == 0:
            if abs(below_sum) > tolerance:
                continue
            below_values: list[float] = []
        elif not -tolerance <= below_sum <= below_count * target + tolerance:
            continue
        else:
            below_values = [min(target, max(0.0, below_sum / below_count))] * below_count
        if above_count == 0:
            if abs(above_sum) > tolerance:
                continue
            above_values: list[float] = []
        elif not above_count * target - tolerance <= above_sum <= above_count + tolerance:
            continue
        else:
            above_values = [min(1.0, max(target, above_sum / above_count))] * above_count
        values = tuple(below_values + above_values)
        rebuilt_mean = sum(values) / n_cells
        rebuilt_deficit = sum(max(0.0, target - value) / target for value in values) / n_cells
        if abs(rebuilt_mean - mean) <= 1e-7 and abs(rebuilt_deficit - deficit) <= 1e-7:
            return values
    raise ValueError(
        "mean_jain_fairness and fairness_deficit cannot be represented by "
        f"{n_cells} reconstructed cells: mean={mean}, deficit={deficit}"
    )


def trace_row_to_snapshot(
    row: dict[str, Any],
    *,
    topology: Optional[TraceTopology] = None,
) -> dict[str, Any]:
    """Rebuild one GRPO trace row into the nested KPI-snapshot row shape.

    Reads only the aggregates that ``rewards.compute_breakdown`` emits into
    ``reward_measurements``; requires ``aggregate_delivered_mbps`` and
    ``n_ues``, everything else defaults to its uncongested value. Pressure
    measurements are inverted back to the KPI that produced them:

        prb_pressure    -> prb_util_dl_p99      = 0.85 + 0.15 * pressure
        access_pressure -> prach_collision_rate = 0.05 + 0.45 * pressure
        buffer_pressure -> buffer_occupancy_kb  = (pressure + 0.7) * 1024 (if > 0)

    The result distributes ``n_ues`` over an inferred runner topology; re-running
    ``compute_breakdown`` over reconstructed pairs reproduces the recorded
    aggregate measurements, but per-UE detail (elastic Jain fairness, the
    real buffer distribution) is lost. A recorded ``cell_capacity_mbps_total``
    is carried through so the reward's throughput normalizer keeps the
    recorded scale. The backend converts that total to the per-cell value
    ``compute_breakdown`` expects after topology reconstruction.
    """
    measurements = row.get("reward_measurements")
    if not isinstance(measurements, dict):
        raise ValueError(f"trace row is missing the 'reward_measurements' object; got keys {sorted(row)}")
    for key in ("aggregate_delivered_mbps", "n_ues"):
        if key not in measurements:
            raise ValueError(
                f"trace row reward_measurements is missing required key {key!r}; got keys {sorted(measurements)}"
            )

    if topology is None:
        topology = _infer_trace_topology([row])
    n_cells = topology.n_cells
    n_ues = _trace_n_ues(row)
    ue_counts = _ue_counts(n_ues, n_cells)
    delivered_total = max(0.0, _num(measurements, "aggregate_delivered_mbps", 0.0))
    delivered = delivered_total / n_ues
    offered = max(delivered, _num(measurements, "requested_service_mbps", delivered * n_ues) / n_ues)

    prb_pressure = max(0.0, _num(measurements, "prb_pressure", 0.0))
    p99 = min(1.0, 0.85 + 0.15 * prb_pressure)
    p50 = max(0.0, (p99 - 0.02) / 1.15)

    access_pressure = max(0.0, _num(measurements, "access_pressure", 0.0))
    prach = min(1.0, 0.05 + 0.45 * access_pressure) if access_pressure > 0.0 else 0.0

    buffer_pressure = max(0.0, _num(measurements, "buffer_pressure", 0.0))
    buffer_kb = (buffer_pressure + 0.7) * 1024.0 if buffer_pressure > 0.0 else 0.0

    fairness_by_cell = _fairness_by_cell(measurements, n_cells)
    sla = max(0, int(round(_num(measurements, "sla_violations", 0.0))))
    sla_quotient, sla_remainder = divmod(sla, n_cells)
    sla_by_cell = tuple(sla_quotient + (1 if cell_id < sla_remainder else 0) for cell_id in range(n_cells))

    cells = []
    for cell_id, (cell_n_ues, cell_fairness, cell_sla) in enumerate(zip(ue_counts, fairness_by_cell, sla_by_cell)):
        ues = [
            {
                # UE ids are local to each reconstructed cell. For Amparo's
                # t1_runner trace this yields cells 0/1 with local ids 0/1.
                "ue_id": ue_id,
                "offered_mbps": offered,
                "delivered_mbps": delivered,
                "buffer_occupancy_kb": buffer_kb,
                "pdb_violations": 1 if ue_id < min(cell_sla, cell_n_ues) else 0,
            }
            for ue_id in range(cell_n_ues)
        ]
        cells.append(
            {
                "cell_id": cell_id,
                "prb_util_dl_p50": p50,
                "prb_util_dl_p99": p99,
                "prach_collision_rate": prach,
                "rrc_connected_ues": cell_n_ues,
                "fairness_jain": cell_fairness,
                "sla_violations_last_window": cell_sla,
                "ues": ues,
            }
        )
    snapshot: dict[str, Any] = {
        "episode_id": row.get("episode_id"),
        "step": row.get("step"),
        # Retain the GRPO iteration as ingestion-only identity metadata. It
        # is not part of Observation, but it prevents repeated episode IDs
        # from different iterations being silently interleaved.
        "_source_iter": row.get("iter"),
        "_source_tool_sent": row.get("tool_sent"),
        "_source_action_accepted": _recorded_action_accepted(row),
        "_source_scenario_mode": topology.scenario_mode,
        "_reconstruction_schema": TRACE_RECONSTRUCTION_SCHEMA,
        "_reconstruction_topology_source": topology.source,
        "_reconstruction_n_cells": n_cells,
        "_reconstruction_ue_counts": list(ue_counts),
        "kpi_source_mode": str(row.get("kpi_source", "replay")),
        "cell_capacity_mbps_total": measurements.get("cell_capacity_mbps_total"),
        "cells": cells,
        "_lineno": row.get("_lineno", "?"),
    }
    return snapshot


# --- File loading (JSONL first; CSV adapter point) ---------------------------


def _rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: row must be a JSON object")
            row["_lineno"] = lineno
            rows.append(row)
    return rows


def _rows_from_csv(path: Path) -> list[dict[str, Any]]:
    """CSV adapter for the KPI-snapshot format.

    Assumed flat shape (one line per UE per timestep), regrouped into the
    nested JSONL row shape::

        episode_id, step, t_s, cell_id, prb_util_dl_p50, ue_id,
        offered_mbps, delivered_mbps, bler, sinr_db

    If a provided CSV differs, rewrite only this function so it returns the
    same nested row dicts as ``_rows_from_jsonl``.
    """
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for lineno, rec in enumerate(csv.DictReader(fh), start=2):
            episode = str(rec.get("episode_id") or "episode_0")
            step = int(float(rec.get("step") or 0))
            row = grouped.setdefault(
                (episode, step),
                {"episode_id": episode, "step": step, "cells": [], "_lineno": lineno},
            )
            if rec.get("t_s"):
                row["t_s"] = float(rec["t_s"])
            cell_id = int(float(rec.get("cell_id") or 0))
            cell = next((c for c in row["cells"] if c["cell_id"] == cell_id), None)
            if cell is None:
                cell = {"cell_id": cell_id, "ues": []}
                row["cells"].append(cell)
            if rec.get("prb_util_dl_p50"):
                cell["prb_util_dl_p50"] = float(rec["prb_util_dl_p50"])
            ue: dict[str, Any] = {"ue_id": int(float(rec.get("ue_id") or 0))}
            for key in ("offered_mbps", "delivered_mbps", "bler", "sinr_db"):
                if rec.get(key):
                    ue[key] = float(rec[key])
            cell["ues"].append(ue)
    # Deterministic order: by (episode, step).
    return [grouped[key] for key in sorted(grouped)]


@dataclass(frozen=True)
class EpisodeSource:
    """One recorded episode: validated observations plus recorded reward context."""

    observations: list[Observation]
    # One value per observation. A trace row's capacity describes the reward
    # transition ending at that row, so step(prev=i, curr=i+1) consumes entry
    # i+1. None means the backend's configured default applies for that step.
    cell_capacity_mbps_by_observation: tuple[Optional[float], ...]
    reconstruction_schema: str = NATIVE_SNAPSHOT_SCHEMA
    reconstruction_topology_source: str = "native_dataset_rows"
    source_scenario_mode: Optional[str] = None
    initial_history_action: Optional[ToolCall] = None
    initial_history_action_accepted: bool = False


def _order_value(path: Path, row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{row.get('_lineno', '?')}: non-numeric {key!r} value {row[key]!r}") from exc


def _source_iteration(path: Path, row: dict[str, Any]) -> Optional[int]:
    value = row.get("_source_iter", row.get("iter"))
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{path}:{row.get('_lineno', '?')}: trace iter must be an integer, not bool")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{row.get('_lineno', '?')}: non-numeric trace iter {value!r}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{path}:{row.get('_lineno', '?')}: trace iter must be a finite integer, got {value!r}")
    return int(numeric)


def _capacity_value(path: Path, row: dict[str, Any]) -> Optional[float]:
    value = row.get("cell_capacity_mbps_total")
    if value is None:
        return None
    try:
        capacity = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{row.get('_lineno', '?')}: invalid cell_capacity_mbps_total {value!r}") from exc
    if not math.isfinite(capacity) or capacity <= 0.0:
        raise ValueError(
            f"{path}:{row.get('_lineno', '?')}: cell_capacity_mbps_total must be finite and positive, got {value!r}"
        )
    return capacity


def load_provided_dataset(path: str | Path) -> dict[str, EpisodeSource]:
    """Load and validate a dataset into per-episode observation trajectories.

    Returns ``{episode_key: EpisodeSource}`` with observations in timestep
    order. The format (KPI snapshot vs. GRPO trace) is detected from the
    first row and must be consistent across the file. Every row is fully
    validated here so a malformed dataset fails at boot with a line-numbered
    error, never mid-training.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"dataset file not found: {path}. Point the config's 'dataset_path' "
            "at the dataset JSONL (see README, Dataset Formats)."
        )
    if path.suffix.lower() in {".jsonl", ".json", ".ndjson"}:
        rows = _rows_from_jsonl(path)
    elif path.suffix.lower() == ".csv":
        rows = _rows_from_csv(path)
    else:
        raise ValueError(
            f"unsupported dataset extension {path.suffix!r}; expected .jsonl "
            "(preferred) or .csv (adapter in _rows_from_csv)"
        )

    trace_format = bool(rows and is_trace_row(rows[0]))
    if trace_format:
        # Infer topology per source episode before flattening rows into the
        # snapshot representation. Iteration is part of the identity so merged
        # traces cannot lend topology evidence across independent episodes.
        trace_groups: dict[tuple[str, Optional[int]], list[dict[str, Any]]] = {}
        for row in rows:
            base_key = str(row.get("episode_id") or row.get("episode") or "episode_0")
            identity = (base_key, _source_iteration(path, row))
            trace_groups.setdefault(identity, []).append(row)
        topologies: dict[tuple[str, Optional[int]], TraceTopology] = {}
        for identity, group in trace_groups.items():
            try:
                topologies[identity] = _infer_trace_topology(group)
            except (TypeError, ValueError) as exc:
                first = group[0]
                raise ValueError(f"{path}:{first.get('_lineno', '?')} (trace episode {identity!r}): {exc}") from exc
        converted = []
        for row in rows:
            base_key = str(row.get("episode_id") or row.get("episode") or "episode_0")
            identity = (base_key, _source_iteration(path, row))
            try:
                converted.append(trace_row_to_snapshot(row, topology=topologies[identity]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{row.get('_lineno', '?')}: {exc}") from exc
        rows = converted

    # Group rows into episodes by 'episode_id' (or 'episode'); a dataset
    # without one becomes a single episode in file order. Preserve legacy keys
    # when an id occurs in only one iteration, but namespace it when a merged
    # trace reuses that id across iterations.
    iterations_by_episode: dict[str, set[Optional[int]]] = {}
    for row in rows:
        base_key = str(row.get("episode_id") or row.get("episode") or "episode_0")
        iterations_by_episode.setdefault(base_key, set()).add(_source_iteration(path, row))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        base_key = str(row.get("episode_id") or row.get("episode") or "episode_0")
        iterations = iterations_by_episode[base_key]
        if len(iterations) > 1:
            iteration = _source_iteration(path, row)
            if iteration is None:
                raise ValueError(
                    f"{path}:{row.get('_lineno', '?')}: episode {base_key!r} spans multiple "
                    "iterations but one row has no iter field"
                )
            key = f"iter_{iteration}::{base_key}"
        else:
            key = base_key
        grouped.setdefault(key, []).append(row)

    episodes: dict[str, EpisodeSource] = {}
    for key, group in grouped.items():
        # Order within an episode: explicit 'step' field wins, then 't_s',
        # then file order (stable sort keeps ties in file order).
        if all(r.get("step") is not None for r in group):
            group.sort(key=lambda r: _order_value(path, r, "step"))
            ordered_steps = [_order_value(path, row, "step") for row in group]
            if len(ordered_steps) != len(set(ordered_steps)):
                raise ValueError(f"episode {key!r} contains duplicate step coordinates: {ordered_steps}")
        elif all(r.get("t_s") is not None for r in group):
            group.sort(key=lambda r: _order_value(path, r, "t_s"))
        obs_list: list[Observation] = []
        for step_idx, row in enumerate(group):
            try:
                # Placeholder id, re-stamped at reset() via model_copy.
                # key[:56] keeps 'src_' + key within the schema's episode_id
                # max_length=64 for long run names.
                obs_list.append(row_to_observation(row, step_idx=step_idx, episode_id=f"src_{key[:56]}"))
            # ValueError covers pydantic ValidationError (a subclass) and
            # float('bad'); TypeError covers structurally wrong scalar types
            # like "delivered_mbps": [1, 2] or "t_s": {} hitting float().
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{row.get('_lineno', '?')} (episode {key!r}, step {step_idx}): {exc}"
                ) from exc
        if len(obs_list) < 2:
            raise ValueError(
                f"episode {key!r} has only {len(obs_list)} row(s); need >= 2 "
                "observations per episode (each step consumes an obs pair)"
            )
        capacities = tuple(_capacity_value(path, row) for row in group)
        first_row = group[0]
        initial_action = None
        initial_action_accepted = False
        if trace_format:
            raw_initial_action = first_row.get("_source_tool_sent")
            if raw_initial_action is not None:
                initial_action = ToolCall.model_validate(raw_initial_action)
            initial_action_accepted = bool(first_row.get("_source_action_accepted"))
        episodes[key] = EpisodeSource(
            observations=obs_list,
            cell_capacity_mbps_by_observation=capacities,
            reconstruction_schema=str(first_row.get("_reconstruction_schema", NATIVE_SNAPSHOT_SCHEMA)),
            reconstruction_topology_source=str(
                first_row.get("_reconstruction_topology_source", "native_dataset_rows")
            ),
            source_scenario_mode=(
                str(first_row["_source_scenario_mode"]) if first_row.get("_source_scenario_mode") is not None else None
            ),
            initial_history_action=initial_action,
            initial_history_action_accepted=initial_action_accepted,
        )
    if not episodes:
        raise ValueError(f"dataset file {path} contains no rows")
    return episodes


# --- The backend --------------------------------------------------------------


@dataclass
class DatasetEpisode:
    """One live replay over a recorded trajectory (internal bookkeeping)."""

    episode_id: str
    meta: EpisodeMeta
    trajectory: list[Observation]
    cell_capacity_mbps_by_observation: tuple[Optional[float], ...]
    source_key: str
    source_index: int
    reconstruction_schema: str
    reconstruction_topology_source: str
    source_scenario_mode: Optional[str]
    step_idx: int = 0
    closed: bool = False
    history: list[Any] = field(default_factory=list)  # guardrail.HistoryEntry
    lock: threading.RLock = field(default_factory=threading.RLock)


class DatasetReplayBackend(Backend):
    """Replay recorded observations through the Backend contract.

    Offline and deterministic, like ReplayBackend, but the trajectory comes
    from an ingested dataset file instead of seed-driven synthesis. Actions
    never mutate the KPIs (the data is pre-recorded); they still pass the
    guardrail and still earn the standard reward via
    ``rewards.compute_breakdown`` over the served (prev_obs, curr_obs) pair.
    """

    backend_name = "dataset_replay"
    dynamics_mode = DATASET_DYNAMICS_MODE
    action_affects_observation = False

    def __init__(
        self,
        *,
        dataset_path: str = "data/dataset/provided.jsonl",
        pool_size: int = 32,
        max_steps_default: int = 60,
        cell_capacity_mbps: float = 60.0,
        reward_profile: str = "openair_v1",
        reward_weights: Optional[dict[str, float]] = None,
    ) -> None:
        """
        Args:
            dataset_path: Dataset file (.jsonl preferred, .csv via the
                adapter). Loaded and validated eagerly so bad data fails at
                server boot.
            pool_size: Max concurrent live episodes (same semantics as
                ReplayBackend's pool).
            max_steps_default: Step budget for task rows lacking max_steps;
                always clamped to the recorded trajectory's length.
            cell_capacity_mbps: Normalizer for the reward's throughput-delta
                term. ReplayEnv gets this from its scenario fingerprint; a
                recorded dataset has no fingerprint, so it is a config knob
                (compute_breakdown's own default is 60.0). Trace episodes
                that record cell_capacity_mbps_total override it per transition.
            reward_profile: Auditable label for the configured reward weights.
            reward_weights: Per-field overrides on rewards.DEFAULT_WEIGHTS.
                Must match the profile the dataset was recorded under, or
                recomputed rewards drift from the recorded ones (the
                openair_v2_measured runs zero w_sla, w_sla_level, w_buffer
                and w_action).
        """
        self.dataset_path = Path(dataset_path)
        self.pool_size = int(pool_size)
        self.max_steps_default = int(max_steps_default)
        self.cell_capacity_mbps = float(cell_capacity_mbps)
        self.reward_profile = str(reward_profile)
        validate_reward_profile(self.reward_profile, reward_weights)
        self.reward_weights = (
            replace(_rewards.DEFAULT_WEIGHTS, **reward_weights) if reward_weights else _rewards.DEFAULT_WEIGHTS
        )
        self.reward_weights_dict = asdict(self.reward_weights)

        # episode_key -> validated source trajectory (shared read-only across
        # episodes; per-episode copies get their own episode_id stamps).
        self._sources: dict[str, EpisodeSource] = load_provided_dataset(self.dataset_path)
        self._keys: list[str] = sorted(self._sources)
        digest = hashlib.sha256()
        with self.dataset_path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        self.dataset_sha256 = digest.hexdigest()
        self.dataset_identity = f"sha256:{self.dataset_sha256}"
        self.dataset_row_count = sum(len(source.observations) for source in self._sources.values())
        self.dataset_episode_count = len(self._sources)

        self._lock = threading.Lock()
        self._episodes: dict[str, DatasetEpisode] = {}

    def receipt_info(self) -> dict[str, Any]:
        return {
            **super().receipt_info(),
            "reward_weights": dict(self.reward_weights_dict),
            "dataset_identity": self.dataset_identity,
            "dataset_sha256": self.dataset_sha256,
            "dataset_row_count": self.dataset_row_count,
            "dataset_episode_count": self.dataset_episode_count,
        }

    @staticmethod
    def _reconstruction_receipt(episode: DatasetEpisode, observation: Observation) -> dict[str, Any]:
        return {
            "reconstruction_schema": episode.reconstruction_schema,
            "reconstruction_topology_source": episode.reconstruction_topology_source,
            "reconstruction_n_cells": observation.global_.n_cells,
            "reconstruction_ue_counts": [len(cell.ues) for cell in observation.cells],
            "reconstruction_source_scenario_mode": episode.source_scenario_mode,
            "reconstruction_assumptions": (
                [
                    "aggregate_kpis_distributed_across_inferred_cells",
                    "per_ue_radio_and_service_detail_not_recoverable",
                    "recorded_actions_do_not_drive_replayed_observations",
                ]
                if episode.reconstruction_schema == TRACE_RECONSTRUCTION_SCHEMA
                else []
            ),
        }

    def episode_receipt_info(self, episode_id: str) -> dict[str, Any]:
        with self._lock:
            episode = self._episodes.get(episode_id)
        if episode is None:
            raise KeyError(f"unknown episode_id {episode_id!r}")
        return {
            **self.receipt_info(),
            "dataset_episode_key": episode.source_key,
            "dataset_index": episode.source_index,
            **self._reconstruction_receipt(episode, episode.trajectory[episode.step_idx]),
        }

    # --- episode selection ----------------------------------------------------

    def _select_key(self, task_params: dict[str, Any]) -> str:
        """Map task_params onto one recorded episode.

        An explicit 'scenario_id' must match a dataset episode key exactly.
        Otherwise an explicit non-negative ``dataset_index`` selects by
        modulo; the seed is the backward-compatible final fallback.
        """
        scenario_id = task_params.get("scenario_id")
        if scenario_id is not None:
            key = str(scenario_id)
            if key not in self._sources:
                raise KeyError(f"scenario_id {key!r} not in dataset; available: {self._keys}")
            return key
        dataset_index = task_params.get("dataset_index")
        if dataset_index is not None:
            if isinstance(dataset_index, bool):
                raise ValueError("dataset_index must be a non-negative integer, not bool")
            try:
                numeric_index = float(dataset_index)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"dataset_index must be a non-negative integer, got {dataset_index!r}") from exc
            if not math.isfinite(numeric_index) or numeric_index < 0 or not numeric_index.is_integer():
                raise ValueError(f"dataset_index must be a non-negative integer, got {dataset_index!r}")
            index = int(numeric_index)
            return self._keys[index % len(self._keys)]
        seed = int(task_params.get("seed", 0))
        return self._keys[seed % len(self._keys)]

    # --- Backend contract -------------------------------------------------------

    def reset(
        self, task_params: dict[str, Any], *, live_episode_ids: Optional[set[str]] = None
    ) -> tuple[Observation, EpisodeMeta]:
        # Selection is read-only over immutable source data; safe outside
        # the lock (and lets a bad scenario_id fail before touching the pool).
        key = self._select_key(task_params)
        source = self._sources[key]
        source_index = self._keys.index(key)

        # Hold the lock across the whole check-reap-build-insert sequence so
        # concurrent resets cannot both pass the capacity check and overshoot
        # pool_size.
        with self._lock:
            if len(self._episodes) >= self.pool_size:
                # Reap episodes no live session owns (crashed rollouts), same
                # leak-safety rule as ReplayBackend.
                live = live_episode_ids or set()
                for eid in [e for e in self._episodes if e not in live]:
                    self._episodes.pop(eid, None)
            if len(self._episodes) >= self.pool_size:
                raise RuntimeError(
                    f"dataset episode pool exhausted ({self.pool_size} live); close episodes or raise pool_size"
                )

            episode_id = f"ds_{uuid.uuid4().hex[:12]}"

            # Re-stamp observations with the real episode id (frozen models:
            # model_copy(update=...), same pattern ReplayEnv uses at reset).
            trajectory = [obs.model_copy(update={"episode_id": episode_id}) for obs in source.observations]

            # A trajectory of N observations supports N-1 (prev, curr) steps.
            budget = int(task_params.get("max_steps") or self.max_steps_default)
            max_steps = max(1, min(len(trajectory) - 1, budget))

            first_obs = trajectory[0]
            meta = EpisodeMeta(
                episode_id=episode_id,
                seed=int(task_params.get("seed", 0)),
                difficulty=first_obs.global_.difficulty,
                regime_mix=first_obs.global_.regime_mix,
                tier=first_obs.global_.tier,
                scenario_id=key,
                max_steps=max_steps,
            )
            episode = DatasetEpisode(
                episode_id=episode_id,
                meta=meta,
                trajectory=trajectory,
                cell_capacity_mbps_by_observation=source.cell_capacity_mbps_by_observation,
                source_key=key,
                source_index=source_index,
                reconstruction_schema=source.reconstruction_schema,
                reconstruction_topology_source=source.reconstruction_topology_source,
                source_scenario_mode=source.source_scenario_mode,
            )
            if source.initial_history_action is not None and source.initial_history_action_accepted:
                # Trace row 0 is already a post-action observation. Preserve
                # that accepted action at logical t=0 so the first replayed
                # turn (row 0 -> row 1, logical t=1) sees the same rate-limit
                # history as the recorded episode.
                episode.history.append(
                    _guardrail.HistoryEntry(
                        action=source.initial_history_action,
                        t_s=0.0,
                    )
                )
            self._episodes[episode_id] = episode
        return first_obs, meta

    def step(self, episode_id: str, tool_call: ToolCall) -> tuple[Observation, float, bool, dict[str, Any]]:
        with self._lock:
            episode = self._episodes.get(episode_id)
        if episode is None:
            raise KeyError(f"unknown episode_id {episode_id!r}")
        with episode.lock:
            if episode.closed:
                raise RuntimeError(f"episode {episode_id!r} is closed")

            prev_obs = episode.trajectory[episode.step_idx]
            logical_now_s = float(episode.step_idx + 1)

            # Same guardrail as ReplayEnv.step, fed from the observation
            # itself (a recorded dataset has no scenario fingerprint).
            gr = _guardrail.check(
                tool_call,
                history=episode.history,
                n_cells=max(1, prev_obs.global_.n_cells),
                n_ues=max(1, prev_obs.global_.n_ues_total),
                n_ues_by_cell={c.cell_id: len(c.ues) for c in prev_obs.cells},
                now_s=logical_now_s,
            )
            rejected = not gr.accepted

            # Pass-through dynamics: the next observation is the recorded
            # data, unmodified by the action.
            next_idx = min(episode.step_idx + 1, len(episode.trajectory) - 1)
            new_obs = episode.trajectory[next_idx]

            # A trace row records the normalizer for the transition ending at
            # that observation. Missing values use the configured fallback.
            recorded_capacity_total = episode.cell_capacity_mbps_by_observation[next_idx]
            n_cells = max(1, new_obs.global_.n_cells)
            # compute_breakdown multiplies its per-cell normalizer by n_cells.
            # Trace rows record the aggregate total, so divide before passing it
            # through. A configured fallback remains explicitly per-cell.
            capacity_per_cell = (
                recorded_capacity_total / n_cells if recorded_capacity_total is not None else self.cell_capacity_mbps
            )
            capacity_total = capacity_per_cell * n_cells
            reward_breakdown = _rewards.compute_breakdown(
                prev_obs=prev_obs,
                curr_obs=new_obs,
                action=tool_call,
                rejected=rejected,
                cell_capacity_mbps=capacity_per_cell,
                weights=self.reward_weights,
            )
            reward = float(reward_breakdown["total"])
            if not math.isfinite(reward):
                raise RuntimeError(f"non-finite dataset reward for episode {episode.source_key!r} step {next_idx}")

            # Commit the state transition only after reward computation and
            # validation succeeds.
            episode.step_idx = next_idx

            if not rejected:
                episode.history.append(_guardrail.HistoryEntry(action=tool_call, t_s=logical_now_s))
                if len(episode.history) > 64:
                    episode.history = episode.history[-32:]

            # Stamp agent_aux so renderer / SFT / GRPO see the same shape as
            # the other backends.
            aux = AgentAux(
                last_action=LastActionEcho(name=tool_call.name, arguments=tool_call.arguments),
                last_reward=reward,
                last_rejection=gr.reason,
                step_idx=episode.step_idx,
            )
            new_obs = new_obs.model_copy(update={"agent_aux": aux})
            episode.trajectory[next_idx] = new_obs  # idempotent on re-read

            done = episode.step_idx >= episode.meta.max_steps
            info = {
                "guardrail_accepted": gr.accepted,
                "rejection_reason": gr.reason,
                "step_idx": episode.step_idx,
                "kpi_source": "dataset_replay",
                "dynamics_mode": DATASET_DYNAMICS_MODE,
                "action_affects_observation": False,
                "reward_profile": self.reward_profile,
                "reward_weights": dict(self.reward_weights_dict),
                # Backward-compatible field now explicitly means the effective
                # per-cell value passed to compute_breakdown.
                "cell_capacity_mbps": capacity_per_cell,
                "cell_capacity_mbps_per_cell": capacity_per_cell,
                "cell_capacity_mbps_total": capacity_total,
                "cell_capacity_source": (
                    "recorded_transition_total"
                    if recorded_capacity_total is not None
                    else "configured_per_cell_default"
                ),
                "dataset_episode_key": episode.source_key,
                "dataset_index": episode.source_index,
                "dataset_identity": self.dataset_identity,
                "dataset_sha256": self.dataset_sha256,
                "dataset_row_count": self.dataset_row_count,
                "dataset_episode_count": self.dataset_episode_count,
                **self._reconstruction_receipt(episode, new_obs),
                "reward_measurements": reward_breakdown["measurements"],
                "reward_terms": reward_breakdown["terms"],
            }
            return new_obs, reward, done, info

    def close(self, episode_id: str) -> dict[str, Any]:
        with self._lock:
            episode = self._episodes.pop(episode_id, None)
        if episode is None:
            raise KeyError(f"unknown episode_id {episode_id!r}")
        episode.closed = True
        return {"ok": True, "n_steps": episode.step_idx}


__all__ = [
    "DATASET_DYNAMICS_MODE",
    "DatasetEpisode",
    "DatasetReplayBackend",
    "EpisodeSource",
    "is_trace_row",
    "load_provided_dataset",
    "row_to_observation",
    "trace_row_to_snapshot",
]
