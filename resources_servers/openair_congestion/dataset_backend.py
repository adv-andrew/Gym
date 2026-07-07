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
  ``rewards.compute_breakdown``). Each trace row is reconstructed into a
  single-cell observation whose aggregate KPIs — delivered throughput, mean
  Jain fairness, PRB/access/buffer pressure, SLA violation count — reproduce
  the recorded measurements, and a recorded ``cell_capacity_mbps_total``
  keeps the reward's throughput normalizer at the recorded scale. Per-UE
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
import json
import math
import threading
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

# backends guards the cross-repo 'openair_congestion' import; keep it ahead of
# the telco imports so a missing install fails with the pip hint.
from resources_servers.openair_congestion.backends import Backend


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

# Schema bound on total UEs per observation (tools.MAX_UES).
_MAX_UES = 24


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
            f"dataset UE record #{ue_idx} is missing required field "
            f"'delivered_mbps'; got keys {sorted(raw)}"
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
            f"dataset cell record #{cell_idx} is missing required field "
            f"'prb_util_dl_p50'; got keys {sorted(raw)}"
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
    sla = int(_num(
        raw,
        "sla_violations_last_window",
        sum(1 for u in ues if u["pdb_violations"] > 0),
    ))
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
            "n_ues_total": int(
                global_raw.get("n_ues_total", sum(len(c["ues"]) for c in cells))
            ),
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


def trace_row_to_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    """Rebuild one GRPO trace row into the nested KPI-snapshot row shape.

    Reads only the aggregates that ``rewards.compute_breakdown`` emits into
    ``reward_measurements``; requires ``aggregate_delivered_mbps`` and
    ``n_ues``, everything else defaults to its uncongested value. Pressure
    measurements are inverted back to the KPI that produced them:

        prb_pressure    -> prb_util_dl_p99      = 0.85 + 0.15 * pressure
        access_pressure -> prach_collision_rate = 0.05 + 0.45 * pressure
        buffer_pressure -> buffer_occupancy_kb  = (pressure + 0.7) * 1024 (if > 0)

    The result is one cell with ``n_ues`` identical UEs; re-running
    ``compute_breakdown`` over reconstructed pairs reproduces the recorded
    aggregate measurements, but per-UE detail (elastic Jain fairness, the
    real buffer distribution) is lost. A recorded ``cell_capacity_mbps_total``
    is carried through so the reward's throughput normalizer keeps the
    recorded scale (the reconstruction is single-cell, so the total is the
    per-cell value ``compute_breakdown`` expects).
    """
    measurements = row.get("reward_measurements")
    if not isinstance(measurements, dict):
        raise ValueError(
            "trace row is missing the 'reward_measurements' object; got keys "
            f"{sorted(row)}"
        )
    for key in ("aggregate_delivered_mbps", "n_ues"):
        if key not in measurements:
            raise ValueError(
                f"trace row reward_measurements is missing required key {key!r}; "
                f"got keys {sorted(measurements)}"
            )

    n_ues = max(1, min(_MAX_UES, int(_num(measurements, "n_ues", 1))))
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

    fairness = min(1.0, max(0.0, _num(measurements, "mean_jain_fairness", 1.0)))
    sla = max(0, int(round(_num(measurements, "sla_violations", 0.0))))

    ues = [
        {
            "ue_id": i,
            "offered_mbps": offered,
            "delivered_mbps": delivered,
            "buffer_occupancy_kb": buffer_kb,
            # Per-UE PDB flags are aggregate bookkeeping: the first `sla` UEs
            # carry the violation so cell and UE counts stay consistent.
            "pdb_violations": 1 if i < sla else 0,
        }
        for i in range(n_ues)
    ]
    snapshot: dict[str, Any] = {
        "episode_id": row.get("episode_id"),
        "step": row.get("step"),
        "kpi_source_mode": str(row.get("kpi_source", "replay")),
        "cell_capacity_mbps_total": measurements.get("cell_capacity_mbps_total"),
        "cells": [
            {
                "cell_id": 0,
                "prb_util_dl_p50": p50,
                "prb_util_dl_p99": p99,
                "prach_collision_rate": prach,
                "rrc_connected_ues": n_ues,
                "fairness_jain": fairness,
                "sla_violations_last_window": sla,
                "ues": ues,
            }
        ],
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
    # cell_capacity_mbps_total recorded by trace rows; None for snapshot data
    # (the backend's cell_capacity_mbps config knob applies).
    cell_capacity_mbps: Optional[float] = None


def _order_value(path: Path, row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}:{row.get('_lineno', '?')}: non-numeric {key!r} value {row[key]!r}"
        ) from exc


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

    if rows and is_trace_row(rows[0]):
        converted = []
        for row in rows:
            try:
                converted.append(trace_row_to_snapshot(row))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{row.get('_lineno', '?')}: {exc}") from exc
        rows = converted

    # Group rows into episodes by 'episode_id' (or 'episode'); a dataset
    # without one becomes a single episode in file order.
    # TODO(dataset-schema): trace episode ids are assumed unique across GRPO
    # iterations; compose the key with 'iter' if provided traces reuse them.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("episode_id") or row.get("episode") or "episode_0")
        grouped.setdefault(key, []).append(row)

    episodes: dict[str, EpisodeSource] = {}
    for key, group in grouped.items():
        # Order within an episode: explicit 'step' field wins, then 't_s',
        # then file order (stable sort keeps ties in file order).
        if all(r.get("step") is not None for r in group):
            group.sort(key=lambda r: _order_value(path, r, "step"))
        elif all(r.get("t_s") is not None for r in group):
            group.sort(key=lambda r: _order_value(path, r, "t_s"))
        obs_list: list[Observation] = []
        for step_idx, row in enumerate(group):
            try:
                # Placeholder id, re-stamped at reset() via model_copy.
                # key[:56] keeps 'src_' + key within the schema's episode_id
                # max_length=64 for long run names.
                obs_list.append(
                    row_to_observation(
                        row, step_idx=step_idx, episode_id=f"src_{key[:56]}"
                    )
                )
            # ValueError covers pydantic ValidationError (a subclass) and
            # float('bad'); TypeError covers structurally wrong scalar types
            # like "delivered_mbps": [1, 2] or "t_s": {} hitting float().
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{row.get('_lineno', '?')} (episode {key!r}, "
                    f"step {step_idx}): {exc}"
                ) from exc
        if len(obs_list) < 2:
            raise ValueError(
                f"episode {key!r} has only {len(obs_list)} row(s); need >= 2 "
                "observations per episode (each step consumes an obs pair)"
            )
        capacity = next(
            (
                float(r["cell_capacity_mbps_total"])
                for r in group
                if r.get("cell_capacity_mbps_total") is not None
            ),
            None,
        )
        episodes[key] = EpisodeSource(observations=obs_list, cell_capacity_mbps=capacity)
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
    # Recorded reward normalizer for trace episodes; None -> the config knob.
    cell_capacity_mbps: Optional[float] = None
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

    def __init__(
        self,
        *,
        dataset_path: str = "data/dataset/provided.jsonl",
        pool_size: int = 32,
        max_steps_default: int = 60,
        cell_capacity_mbps: float = 60.0,
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
                that record cell_capacity_mbps_total override it per episode.
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
        self.reward_weights = (
            replace(_rewards.DEFAULT_WEIGHTS, **reward_weights)
            if reward_weights
            else _rewards.DEFAULT_WEIGHTS
        )

        # episode_key -> validated source trajectory (shared read-only across
        # episodes; per-episode copies get their own episode_id stamps).
        self._sources: dict[str, EpisodeSource] = load_provided_dataset(self.dataset_path)
        self._keys: list[str] = sorted(self._sources)

        self._lock = threading.Lock()
        self._episodes: dict[str, DatasetEpisode] = {}

    # --- episode selection ----------------------------------------------------

    def _select_key(self, task_params: dict[str, Any]) -> str:
        """Map task_params onto one recorded episode.

        An explicit 'scenario_id' must match a dataset episode key exactly;
        otherwise the seed picks deterministically: keys[seed % n].
        """
        scenario_id = task_params.get("scenario_id")
        if scenario_id is not None:
            key = str(scenario_id)
            if key not in self._sources:
                raise KeyError(
                    f"scenario_id {key!r} not in dataset; available: {self._keys}"
                )
            return key
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
                    f"dataset episode pool exhausted ({self.pool_size} live); "
                    "close episodes or raise pool_size"
                )

            episode_id = f"ds_{uuid.uuid4().hex[:12]}"

            # Re-stamp observations with the real episode id (frozen models:
            # model_copy(update=...), same pattern ReplayEnv uses at reset).
            trajectory = [
                obs.model_copy(update={"episode_id": episode_id})
                for obs in source.observations
            ]

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
                cell_capacity_mbps=source.cell_capacity_mbps,
            )
            self._episodes[episode_id] = episode
        return first_obs, meta

    def step(
        self, episode_id: str, tool_call: ToolCall
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        with self._lock:
            episode = self._episodes.get(episode_id)
        if episode is None:
            raise KeyError(f"unknown episode_id {episode_id!r}")
        with episode.lock:
            if episode.closed:
                raise RuntimeError(f"episode {episode_id!r} is closed")

            prev_obs = episode.trajectory[episode.step_idx]
            logical_now_s = float(episode.step_idx)

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
            episode.step_idx = next_idx

            # Trace episodes carry their recorded capacity (single-cell
            # reconstruction, so the recorded total is the per-cell value).
            capacity = (
                episode.cell_capacity_mbps
                if episode.cell_capacity_mbps is not None
                else self.cell_capacity_mbps
            )
            reward_breakdown = _rewards.compute_breakdown(
                prev_obs=prev_obs,
                curr_obs=new_obs,
                action=tool_call,
                rejected=rejected,
                cell_capacity_mbps=capacity,
                weights=self.reward_weights,
            )
            reward = float(reward_breakdown["total"])

            if not rejected:
                episode.history.append(
                    _guardrail.HistoryEntry(action=tool_call, t_s=logical_now_s)
                )
                if len(episode.history) > 64:
                    episode.history = episode.history[-32:]

            # Stamp agent_aux so renderer / SFT / GRPO see the same shape as
            # the other backends.
            aux = AgentAux(
                last_action=LastActionEcho(
                    name=tool_call.name, arguments=tool_call.arguments
                ),
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
                "reward_measurements": reward_breakdown["measurements"],
                "reward_terms": reward_breakdown["terms"],
            }
            assert math.isfinite(reward), "reward must be finite"
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
