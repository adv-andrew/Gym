# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite resource-server action support for RunB2.

``resource_candidate_pipe_v1`` is deliberately independent of the connected
T2 policy-feature contract.  It derives only from the observation currently
served by this resource server, the backend's effective per-cell capacity,
and the recent accepted-action window.  In particular, it preserves the
actual UE ids present in ``U`` rows instead of assuming that ids are dense or
position-based.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Iterable, Mapping

from openair_congestion import guardrail as _guardrail
from openair_congestion.schemas import Observation, ToolCall


RESOURCE_CANDIDATE_CONTRACT = "resource_candidate_pipe_v1"
RESOURCE_GUARDRAIL_PROBE = "resource_candidate_guardrail_probe_v1"
RESOURCE_MAX_PRB = 273
RESOURCE_MIN_PRB = 200


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _milli(value: float) -> int:
    return int((Decimal(str(float(value))) * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))


def resource_action_key(action: ToolCall) -> str:
    """Return a type-preserving identity for a candidate or recent action.

    Tuple/dict equality is unsafe for an authenticated action boundary because
    Python considers ``False == 0`` and ``True == 1``.  Canonical JSON keeps
    booleans, integers, floats, missing fields, and extra fields distinct.
    """

    return _canonical_json(action.model_dump(by_alias=True))


def validate_resource_candidate_guardrail_contract() -> dict[str, object]:
    """Probe the installed guardrail assumptions before serving candidates.

    The finite menu promises that every emitted member is syntactically
    admissible.  Bind its PRB endpoints and two-step identical-repeat model to
    the actually imported telco guardrail at server boot, so a future package
    change fails before the first rollout instead of crashing mid-episode.
    """

    cap = ToolCall(
        name="set_prb_cap",
        arguments={
            "cell_id": 0,
            "target": "ue",
            "target_id": 0,
            "max_prb": RESOURCE_MIN_PRB,
        },
    )
    release = ToolCall(
        name="set_prb_cap",
        arguments={**cap.arguments, "max_prb": RESOURCE_MAX_PRB},
    )
    different = ToolCall(
        name="set_prb_cap",
        arguments={**cap.arguments, "max_prb": RESOURCE_MIN_PRB + 1},
    )
    common = {"n_cells": 1, "n_ues": 1, "n_ues_by_cell": {0: 1}}
    history = [_guardrail.HistoryEntry(action=cap, t_s=0.0)]
    checks = {
        "minimum_prb_accepted": _guardrail.check(cap, now_s=0.0, **common).accepted,
        "release_prb_accepted": _guardrail.check(release, now_s=0.0, **common).accepted,
        "different_setpoint_not_blanket_cooled_down": _guardrail.check(
            different, history=history, now_s=1.0, **common
        ).accepted,
        "identical_repeat_rejected_at_step_one": not _guardrail.check(
            cap, history=history, now_s=1.0, **common
        ).accepted,
        "identical_repeat_rejected_at_step_two": not _guardrail.check(
            cap, history=history, now_s=2.0, **common
        ).accepted,
        "identical_repeat_reaccepted_at_step_three": _guardrail.check(
            cap, history=history, now_s=3.0, **common
        ).accepted,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            "installed guardrail is incompatible with "
            f"{RESOURCE_CANDIDATE_CONTRACT}: {failed}"
        )
    return {
        "schema_version": RESOURCE_GUARDRAIL_PROBE,
        "candidate_contract": RESOURCE_CANDIDATE_CONTRACT,
        "minimum_prb": RESOURCE_MIN_PRB,
        "release_prb": RESOURCE_MAX_PRB,
        "repeat_window_steps": 2,
        "checks": checks,
        "pass": True,
    }


@dataclass(frozen=True)
class ResourceCandidateSupport:
    """Canonical support payload shared by text and response receipts."""

    capacity_milli_mbps_by_cell: tuple[tuple[int, int], ...]
    actions: tuple[ToolCall, ...]
    contract: str = RESOURCE_CANDIDATE_CONTRACT

    def __post_init__(self) -> None:
        if self.contract != RESOURCE_CANDIDATE_CONTRACT:
            raise ValueError(f"unsupported candidate contract {self.contract!r}")
        if not self.capacity_milli_mbps_by_cell:
            raise ValueError("candidate support requires at least one cell capacity")
        capacity_ids = [cell_id for cell_id, _ in self.capacity_milli_mbps_by_cell]
        if capacity_ids != sorted(set(capacity_ids)):
            raise ValueError("candidate cell capacities must have unique sorted cell ids")
        if any(cell_id < 0 or capacity <= 0 for cell_id, capacity in self.capacity_milli_mbps_by_cell):
            raise ValueError("candidate cell ids must be nonnegative and capacities must be positive")
        capacity_cell_ids = set(capacity_ids)
        if not self.actions or self.actions[0].name != "noop" or self.actions[0].arguments != {}:
            raise ValueError("candidate support must begin with noop({})")
        keys = [resource_action_key(action) for action in self.actions]
        if len(keys) != len(set(keys)):
            raise ValueError("candidate support contains duplicate actions")
        for action in self.actions[1:]:
            arguments = action.arguments
            if action.name != "set_prb_cap" or set(arguments) != {
                "cell_id",
                "target",
                "target_id",
                "max_prb",
            }:
                raise ValueError("non-noop candidates must be exact set_prb_cap calls")
            if arguments["target"] != "ue":
                raise ValueError("resource candidates may target only U-row UEs")
            for key in ("cell_id", "target_id", "max_prb"):
                value = arguments[key]
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(f"candidate {key} must be an integer")
            if arguments["cell_id"] < 0 or arguments["target_id"] < 0:
                raise ValueError("candidate cell_id and target_id must be nonnegative")
            if arguments["cell_id"] not in capacity_cell_ids:
                raise ValueError(
                    "candidate cell_id must have an authenticated capacity row"
                )
            if not RESOURCE_MIN_PRB <= arguments["max_prb"] <= RESOURCE_MAX_PRB:
                raise ValueError(f"candidate max_prb must be in [{RESOURCE_MIN_PRB},{RESOURCE_MAX_PRB}]")

    @property
    def payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "capacity_milli_mbps_by_cell": [
                {
                    "cell_id": cell_id,
                    "capacity_milli_mbps": capacity,
                }
                for cell_id, capacity in self.capacity_milli_mbps_by_cell
            ],
            "actions": [action.model_dump(by_alias=True) for action in self.actions],
        }

    @property
    def support_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload).encode()).hexdigest()

    def receipt_fields(self) -> dict[str, object]:
        actions = [action.model_dump(by_alias=True) for action in self.actions]
        capacities = list(self.payload["capacity_milli_mbps_by_cell"])
        return {
            "candidate_contract": self.contract,
            "candidate_actions": actions,
            "candidate_support_sha256": self.support_sha256,
            "candidate_capacity_milli_mbps_by_cell": capacities,
            # Compatibility aliases for the existing trainer trace schema.
            "visible_action_mask": self.contract,
            "visible_action_candidates": actions,
            "visible_action_support_sha256": self.support_sha256,
            "visible_action_supported": True,
        }

    def render_rows(self) -> list[str]:
        rows = [f"RCP|{self.contract}|{self.support_sha256}|{len(self.actions)}"]
        rows.extend(f"RCC|{cell_id}|{capacity}" for cell_id, capacity in self.capacity_milli_mbps_by_cell)
        rows.extend(
            f"RCA|{index}|{_canonical_json(action.model_dump(by_alias=True))}"
            for index, action in enumerate(self.actions)
        )
        rows.append("A|choose_exactly_one_resource_candidate")
        return rows


def _requested_milli(ue: object) -> int:
    value = getattr(ue, "requested_mbps", None)
    if value is None:
        value = getattr(ue, "offered_mbps")
    return _milli(float(value))


def _admitted_milli(ue: object) -> int:
    value = getattr(ue, "admitted_mbps", None)
    if value is None:
        value = getattr(ue, "offered_mbps")
    return _milli(float(value))


def derive_resource_candidate_support(
    observation: Observation,
    *,
    capacity_mbps_by_cell: Mapping[int, float],
    excluded_actions: Iterable[ToolCall] = (),
) -> ResourceCandidateSupport:
    """Derive noop plus one canonical release/cap opportunity per cell.

    A release is offered only after visible requested load has recovered to
    capacity.  Under overload, a cap is computed from the exact visible
    requested/admitted totals and the backend-reported capacity.  Equal
    active setpoints and relaxing an active cap are never candidates.
    Recently accepted identical actions are skipped so every emitted non-noop
    remains valid under the environment's two-step rate-limit window.
    """

    observation_cell_ids = sorted(cell.cell_id for cell in observation.cells)
    supplied_ids = sorted(capacity_mbps_by_cell)
    if supplied_ids != observation_cell_ids:
        raise ValueError(
            "candidate capacity cells must exactly match observation cells: "
            f"capacity={supplied_ids}, observation={observation_cell_ids}"
        )
    capacities = tuple((cell_id, _milli(float(capacity_mbps_by_cell[cell_id]))) for cell_id in observation_cell_ids)
    if any(capacity <= 0 for _, capacity in capacities):
        raise ValueError("candidate capacities must be finite positive values")
    capacity_lookup = dict(capacities)
    excluded = {resource_action_key(action) for action in excluded_actions if action.name != "noop"}

    releases: list[ToolCall] = []
    caps: list[ToolCall] = []
    for cell in sorted(observation.cells, key=lambda item: item.cell_id):
        capacity = capacity_lookup[cell.cell_id]
        requested_total = sum(_requested_milli(ue) for ue in cell.ues)
        admitted_total = sum(_admitted_milli(ue) for ue in cell.ues)

        active = sorted(
            (ue for ue in cell.ues if ue.prb_cap_max_prb is not None and int(ue.prb_cap_max_prb) < RESOURCE_MAX_PRB),
            key=lambda ue: ue.ue_id,
        )
        if requested_total <= capacity:
            for ue in active:
                action = ToolCall(
                    name="set_prb_cap",
                    arguments={
                        "cell_id": cell.cell_id,
                        "target": "ue",
                        "target_id": ue.ue_id,
                        "max_prb": RESOURCE_MAX_PRB,
                    },
                )
                if resource_action_key(action) not in excluded:
                    releases.append(action)
                    break
            continue

        if admitted_total <= capacity:
            continue
        ranked = sorted(
            cell.ues,
            key=lambda ue: (
                _milli(float(ue.delivered_mbps)),
                _milli(float(ue.buffer_occupancy_kb)),
                -ue.ue_id,
            ),
            reverse=True,
        )
        for ue in ranked:
            requested = _requested_milli(ue)
            admitted = _admitted_milli(ue)
            if requested <= 0:
                continue
            available = max(0, capacity - (admitted_total - admitted))
            max_prb = RESOURCE_MAX_PRB * available // requested
            max_prb = max(RESOURCE_MIN_PRB, min(RESOURCE_MAX_PRB, max_prb))
            current = int(ue.prb_cap_max_prb) if ue.prb_cap_max_prb is not None else RESOURCE_MAX_PRB
            if max_prb >= current or max_prb >= RESOURCE_MAX_PRB:
                continue
            action = ToolCall(
                name="set_prb_cap",
                arguments={
                    "cell_id": cell.cell_id,
                    "target": "ue",
                    "target_id": ue.ue_id,
                    "max_prb": max_prb,
                },
            )
            if resource_action_key(action) in excluded:
                continue
            caps.append(action)
            break

    return ResourceCandidateSupport(
        capacity_milli_mbps_by_cell=capacities,
        actions=(ToolCall(name="noop", arguments={}), *releases, *caps),
    )


def parse_resource_candidate_support(text: str) -> ResourceCandidateSupport:
    """Strictly parse and independently authenticate candidate text rows."""

    headers = [line.split("|", 3) for line in text.splitlines() if line.startswith("RCP|")]
    if len(headers) != 1 or len(headers[0]) != 4:
        raise ValueError("candidate text must contain exactly one four-field RCP row")
    _, contract, declared_sha, count_raw = headers[0]
    if contract != RESOURCE_CANDIDATE_CONTRACT:
        raise ValueError(f"unexpected candidate contract {contract!r}")
    try:
        declared_count = int(count_raw)
    except ValueError as exc:
        raise ValueError("RCP candidate count must be an integer") from exc
    if declared_count <= 0 or str(declared_count) != count_raw:
        raise ValueError("RCP candidate count must be a canonical positive integer")

    capacity_rows: list[tuple[int, int]] = []
    action_rows: list[tuple[int, ToolCall]] = []
    for line in text.splitlines():
        if line.startswith("RCC|"):
            parts = line.split("|")
            if len(parts) != 3:
                raise ValueError("RCC rows must have exactly three fields")
            try:
                cell_id = int(parts[1])
                capacity = int(parts[2])
            except ValueError as exc:
                raise ValueError("RCC fields must be integers") from exc
            if str(cell_id) != parts[1] or str(capacity) != parts[2]:
                raise ValueError("RCC integers must use canonical decimal spelling")
            capacity_rows.append((cell_id, capacity))
        elif line.startswith("RCA|"):
            parts = line.split("|", 2)
            if len(parts) != 3:
                raise ValueError("RCA rows must have exactly three fields")
            try:
                index = int(parts[1])
                payload = json.loads(parts[2])
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError("RCA index and payload must be valid") from exc
            if parts[1] != str(index):
                raise ValueError("RCA index must use canonical decimal spelling")
            action = ToolCall.model_validate(payload)
            if parts[2] != _canonical_json(action.model_dump(by_alias=True)):
                raise ValueError("RCA payload must use canonical JSON")
            action_rows.append((index, action))

    if [index for index, _ in action_rows] != list(range(len(action_rows))):
        raise ValueError("RCA indices must be unique, contiguous, and ordered from zero")
    if declared_count != len(action_rows):
        raise ValueError("RCP candidate count does not match RCA rows")
    support = ResourceCandidateSupport(
        capacity_milli_mbps_by_cell=tuple(capacity_rows),
        actions=tuple(action for _, action in action_rows),
        contract=contract,
    )
    if support.support_sha256 != declared_sha:
        raise ValueError("candidate support hash mismatch")
    return support


__all__ = [
    "RESOURCE_CANDIDATE_CONTRACT",
    "RESOURCE_GUARDRAIL_PROBE",
    "RESOURCE_MAX_PRB",
    "RESOURCE_MIN_PRB",
    "ResourceCandidateSupport",
    "derive_resource_candidate_support",
    "parse_resource_candidate_support",
    "resource_action_key",
    "validate_resource_candidate_guardrail_contract",
]
