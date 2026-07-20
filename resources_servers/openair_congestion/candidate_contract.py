# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Server-authored finite support for the RunB2 V10 replay protocol.

The ordinary OpenAir resource server exposes the full heterogeneous tool
surface.  The constrained RunB2 decoder, however, has a deliberately smaller
and independently-auditable action language: ``noop`` plus the observation
derived T2 UE PRB-cap actions.  This module owns the translation at the
server boundary.  A client must consume the rendered support; it must never
invent, append, or reorder candidates locally.

This is intentionally *not* a generic full-tool action schema.  Calling it
from a non-T2 or non-synthetic replay path is an error rather than an excuse
to make a broader policy claim than the decoder can support.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from openair_congestion import render
from openair_congestion.schemas import Observation, ToolCall
from openair_congestion.t2_action_mask import derive_t2_visible_action_set_from_text


RESOURCE_CANDIDATE_CONTRACT = "resource_candidate_pipe_v1"
RUNB2_V10_PROTOCOL_MODE = "runb2_v10_t2_prb_v1"
RUNB2_V10_ACTION_EFFECT_CONTRACT_SCHEMA = "openair_runb2_v10_action_effect_source_contract_v1"
RUNB2_V10_ACTION_SCOPE = "t2_prb_only_synthetic_replay_v1"
RUNB2_V10_OBSERVATION_RENDER = RESOURCE_CANDIDATE_CONTRACT
RUNB2_V10_REWARD_PROFILE = "openair_v1"
RESOURCE_ACTION_ARGUMENT_CONTRACT = "resource_candidate_ue_prb_200_273_v1"
# V10 deliberately has no caller-selectable capacity normalizer.  The replay
# environment's T2 generator contract fixes it at 250 Mbps per cell; carrying a
# second knob in the resource server would let the visible candidate contract
# disagree with the causal reward/dynamics source.
RUNB2_V10_CELL_CAPACITY_MBPS = 250.0
RUNB2_V10_CELL_CAPACITY_MILLI_MBPS = 250_000
RUNB2_V10_CAPACITY_UNIT = "milli_mbps"
RUNB2_V10_MAX_STEPS_HARD_CAP = 16

# This line is intentionally visible in every V10 observation.  Its digest
# binds the server's exact policy/reward/capacity/code contract to the RCP
# support rendered alongside it.  The payload itself travels in response
# provenance; placing it verbatim in the prompt would create needless token
# pressure and duplicate a server-authored receipt.
RUNB2_V10_VISIBLE_BINDING_SCHEMA = "openair_runb2_v10_server_observation_binding_v3"
RUNB2_V10_VISIBLE_BINDING_PREFIX = "V10B"
RUNB2_V10_RUNTIME_MANIFEST_SCHEMA = "openair_runb2_v10_runtime_manifest_v2"
RUNB2_V10_LAUNCH_CONTRACT_SCHEMA = "openair_runb2_v10_launch_contract_v1"

_VISIBLE_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "protocol_mode",
        "action_scope",
        "observation_render",
        "candidate_contract",
        "candidate_action_argument_contract",
        "environment_contract",
        "reward_contract",
        "capacity_contract",
        "runtime_manifest",
        "launch_contract",
        "candidate_support_sha256",
        "candidate_capacity_milli_mbps_by_cell",
        "observation_without_binding_sha256",
    }
)
_RUNTIME_MANIFEST_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "sha256",
    }
)
_LAUNCH_CONTRACT_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "system_prompt_sha256",
        "task_manifest_sha256",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CandidateContractError(ValueError):
    """A support, action, or replay-only protocol invariant failed."""


def canonical_json(value: Any) -> str:
    """Return the stable JSON form used in support hashes and RCP rows."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _reject_json_constant(value: str) -> None:
    """Reject non-JSON numeric spellings such as ``NaN`` and ``Infinity``."""

    raise CandidateContractError(f"JSON contains unsupported non-finite constant {value!r}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting ambiguous duplicate keys.

    Python's default decoder silently retains only the final duplicate.  That
    is unacceptable at a policy/action boundary because a visibly valid call
    could otherwise be interpreted differently by another consumer.
    """

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateContractError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def parse_strict_json(value: str, *, label: str) -> Any:
    """Decode JSON without duplicate keys or non-finite pseudo-numbers."""

    if not isinstance(value, str):
        raise CandidateContractError(f"{label} must be JSON text")
    try:
        return json.loads(
            value,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, CandidateContractError) as exc:
        raise CandidateContractError(f"{label} is invalid JSON") from exc


def canonical_json_sha256(value: Any) -> str:
    """Digest a canonical payload using the V10B trailing-LF convention."""

    return hashlib.sha256((canonical_json(value) + "\n").encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    """Hash exact rendered UTF-8 text without normalizing its line endings."""

    if not isinstance(value, str):
        raise CandidateContractError("rendered observation must be text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_action(action: ToolCall | Mapping[str, Any]) -> dict[str, Any]:
    """Validate one exact V10 PRB-only action and return its two-field form."""

    try:
        parsed = action if isinstance(action, ToolCall) else ToolCall.model_validate(dict(action))
    except Exception as exc:  # noqa: BLE001 - isolate Pydantic at this boundary.
        raise CandidateContractError(f"invalid resource candidate action: {exc}") from exc
    normalized = {
        "name": parsed.name,
        "arguments": dict(parsed.arguments),
    }
    _validate_action(normalized)
    return normalized


def _validate_action(action: Mapping[str, Any]) -> None:
    if set(action) != {"name", "arguments"}:
        raise CandidateContractError("candidate action must contain exactly name and arguments")
    name = action["name"]
    arguments = action["arguments"]
    if name == "noop":
        if arguments != {}:
            raise CandidateContractError("noop candidate arguments must be exactly {}")
        return
    if name != "set_prb_cap":
        raise CandidateContractError(f"V10 T2 support does not enable {name!r}")
    if not isinstance(arguments, Mapping) or set(arguments) != {
        "cell_id",
        "target",
        "target_id",
        "max_prb",
    }:
        raise CandidateContractError(
            "set_prb_cap candidate arguments must contain exactly cell_id, target, target_id, and max_prb"
        )
    if arguments["target"] != "ue":
        raise CandidateContractError("set_prb_cap candidate target must be 'ue'")
    for field_name in ("cell_id", "target_id"):
        value = arguments[field_name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CandidateContractError(f"set_prb_cap candidate {field_name} must be a nonnegative integer")
    max_prb = arguments["max_prb"]
    if not isinstance(max_prb, int) or isinstance(max_prb, bool) or not 200 <= max_prb <= 273:
        raise CandidateContractError("set_prb_cap candidate max_prb must be an integer in [200,273]")


def action_key(action: ToolCall | Mapping[str, Any]) -> str:
    """Return the stable membership identity for a candidate action."""

    return canonical_json(canonical_action(action))


def canonical_support_payload(
    *,
    actions: Iterable[ToolCall | Mapping[str, Any]],
    capacity_milli_mbps_by_cell: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return one fully validated, ordering-sensitive support payload."""

    normalized_actions = [canonical_action(action) for action in actions]
    if not normalized_actions:
        raise CandidateContractError("resource candidate support must not be empty")
    if normalized_actions[0] != {"name": "noop", "arguments": {}}:
        raise CandidateContractError("resource candidate support must place noop first")
    encoded_actions = [canonical_json(action) for action in normalized_actions]
    if len(encoded_actions) != len(set(encoded_actions)):
        raise CandidateContractError("resource candidate support contains duplicate actions")

    capacities: list[dict[str, int]] = []
    for index, raw in enumerate(capacity_milli_mbps_by_cell):
        if not isinstance(raw, Mapping) or set(raw) != {
            "cell_id",
            "capacity_milli_mbps",
        }:
            raise CandidateContractError(f"capacity row {index} must contain cell_id and capacity_milli_mbps")
        cell_id = raw["cell_id"]
        capacity = raw["capacity_milli_mbps"]
        if (
            not isinstance(cell_id, int)
            or isinstance(cell_id, bool)
            or cell_id < 0
            or not isinstance(capacity, int)
            or isinstance(capacity, bool)
            or capacity <= 0
        ):
            raise CandidateContractError(f"capacity row {index} contains invalid integer values")
        capacities.append({"cell_id": cell_id, "capacity_milli_mbps": capacity})
    if not capacities:
        raise CandidateContractError("resource candidate support requires cell capacities")
    if capacities != sorted(capacities, key=lambda row: row["cell_id"]):
        raise CandidateContractError("resource candidate capacity rows must be cell-sorted")
    if len({row["cell_id"] for row in capacities}) != len(capacities):
        raise CandidateContractError("resource candidate capacity rows contain duplicate cells")
    known_cells = {row["cell_id"] for row in capacities}
    unknown_cells = sorted(
        int(action["arguments"]["cell_id"])
        for action in normalized_actions
        if action["name"] == "set_prb_cap" and action["arguments"]["cell_id"] not in known_cells
    )
    if unknown_cells:
        raise CandidateContractError(f"resource candidates reference cells without capacity rows: {unknown_cells}")
    return {
        "contract": RESOURCE_CANDIDATE_CONTRACT,
        "capacity_milli_mbps_by_cell": capacities,
        "actions": normalized_actions,
    }


def support_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a canonical support payload."""

    return hashlib.sha256(canonical_json(dict(payload)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateSupport:
    """The exact visible V10 support for one observation."""

    policy_text: str
    observation_text: str
    actions: tuple[dict[str, Any], ...]
    capacity_milli_mbps_by_cell: tuple[dict[str, int], ...]
    support_sha256: str
    visible_binding_schema: str | None = None
    visible_binding_sha256: str | None = None
    visible_binding_payload: Mapping[str, Any] | None = None

    @property
    def action_keys(self) -> frozenset[str]:
        return frozenset(canonical_json(action) for action in self.actions)

    @property
    def payload(self) -> dict[str, Any]:
        return canonical_support_payload(
            actions=self.actions,
            capacity_milli_mbps_by_cell=self.capacity_milli_mbps_by_cell,
        )

    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "candidate_contract": RESOURCE_CANDIDATE_CONTRACT,
            "candidate_action_argument_contract": RESOURCE_ACTION_ARGUMENT_CONTRACT,
            # Return a deep canonical copy.  A shallow ``dict(action)`` leaves
            # nested ``arguments`` mutable outside the server's support state.
            "candidate_actions": [json.loads(canonical_json(action)) for action in self.actions],
            "candidate_support_sha256": self.support_sha256,
            "visible_action_support_sha256": self.support_sha256,
            "candidate_capacity_milli_mbps_by_cell": [dict(row) for row in self.capacity_milli_mbps_by_cell],
            "observation_render": RUNB2_V10_OBSERVATION_RENDER,
        }
        if self.visible_binding_payload is not None:
            result["server_observation_binding"] = {
                "observation_sha256": text_sha256(self.observation_text),
                "candidate_support_sha256": self.support_sha256,
                "binding_payload": _canonical_mapping_copy(
                    self.visible_binding_payload,
                    label="visible binding payload",
                ),
            }
        return result


def _canonical_mapping_copy(value: Any, *, label: str) -> dict[str, Any]:
    """Return a JSON-only canonical mapping or fail before rendering evidence."""

    if not isinstance(value, Mapping):
        raise CandidateContractError(f"{label} must be an object")
    try:
        copied = json.loads(canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise CandidateContractError(f"{label} must be canonical JSON") from exc
    if not isinstance(copied, dict):  # Defensive: ``dict(value)`` above is a mapping.
        raise CandidateContractError(f"{label} must be an object")
    return copied


def _require_exact_mapping(value: Any, *, label: str, keys: frozenset[str]) -> dict[str, Any]:
    mapping = _canonical_mapping_copy(value, label=label)
    if set(mapping) != keys:
        missing = sorted(keys - set(mapping))
        extra = sorted(set(mapping) - keys)
        raise CandidateContractError(f"{label} fields differ from the V10 contract (missing={missing}, extra={extra})")
    return mapping


def _require_finite_positive_float(value: Any, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise CandidateContractError(f"{label} must be a finite positive number")
    return float(value)


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CandidateContractError(f"{label} must be a lowercase SHA-256")
    return value


def build_visible_binding_payload(
    *,
    environment_contract: Mapping[str, Any],
    reward_weights: Mapping[str, Any],
    runtime_manifest: Mapping[str, Any],
    launch_contract: Mapping[str, Any],
    support: CandidateSupport,
) -> dict[str, Any]:
    """Build the exact V10B payload for one already-rendered support.

    This function owns the payload shape shared by the HTTP server and source
    adapter.  Keeping it next to the RCP/RCC/RCA renderer prevents a caller
    from adding a second, subtly different support/capacity serialization.
    """

    payload = {
        "schema_version": RUNB2_V10_VISIBLE_BINDING_SCHEMA,
        "protocol_mode": RUNB2_V10_PROTOCOL_MODE,
        "action_scope": RUNB2_V10_ACTION_SCOPE,
        "observation_render": RUNB2_V10_OBSERVATION_RENDER,
        "candidate_contract": RESOURCE_CANDIDATE_CONTRACT,
        "candidate_action_argument_contract": RESOURCE_ACTION_ARGUMENT_CONTRACT,
        "environment_contract": _canonical_mapping_copy(
            environment_contract,
            label="environment_contract",
        ),
        "reward_contract": {
            "reward_profile": RUNB2_V10_REWARD_PROFILE,
            "reward_weights": _canonical_mapping_copy(
                reward_weights,
                label="reward_weights",
            ),
        },
        "capacity_contract": {
            "unit": RUNB2_V10_CAPACITY_UNIT,
            "candidate_cell_capacity_mbps": RUNB2_V10_CELL_CAPACITY_MBPS,
        },
        "runtime_manifest": _canonical_mapping_copy(
            runtime_manifest,
            label="runtime_manifest",
        ),
        "launch_contract": _canonical_mapping_copy(
            launch_contract,
            label="launch_contract",
        ),
        "candidate_support_sha256": support.support_sha256,
        "candidate_capacity_milli_mbps_by_cell": [dict(row) for row in support.capacity_milli_mbps_by_cell],
        # Hash the exact prompt bytes before the final V10B line is appended.
        # This makes the V10B digest commit to the policy-visible observation
        # while avoiding a circular hash over the line that carries that digest.
        "observation_without_binding_sha256": text_sha256(support.observation_text),
    }
    return validate_visible_binding_payload(payload, support=support)


def validate_visible_binding_payload(value: Any, *, support: CandidateSupport) -> dict[str, Any]:
    """Fail closed unless a V10B payload exactly matches its RCP support."""

    payload = _require_exact_mapping(
        value,
        label="V10 visible binding payload",
        keys=_VISIBLE_BINDING_KEYS,
    )
    expected_static = {
        "schema_version": RUNB2_V10_VISIBLE_BINDING_SCHEMA,
        "protocol_mode": RUNB2_V10_PROTOCOL_MODE,
        "action_scope": RUNB2_V10_ACTION_SCOPE,
        "observation_render": RUNB2_V10_OBSERVATION_RENDER,
        "candidate_contract": RESOURCE_CANDIDATE_CONTRACT,
        "candidate_action_argument_contract": RESOURCE_ACTION_ARGUMENT_CONTRACT,
    }
    for key, expected in expected_static.items():
        if payload[key] != expected:
            raise CandidateContractError(f"V10 visible binding {key} does not match the server contract")

    environment = _require_exact_mapping(
        payload["environment_contract"],
        label="V10 visible binding environment_contract",
        keys=frozenset(
            {
                "schema_version",
                "backend",
                "dynamics_mode",
                "action_affects_observation",
                "candidate_contract",
            }
        ),
    )
    if (
        environment["schema_version"] != RUNB2_V10_ACTION_EFFECT_CONTRACT_SCHEMA
        or environment["backend"] != "replay"
        or environment["action_affects_observation"] is not True
        or environment["candidate_contract"] != RESOURCE_CANDIDATE_CONTRACT
        or not isinstance(environment["dynamics_mode"], str)
        or not environment["dynamics_mode"]
    ):
        raise CandidateContractError("V10 visible binding environment_contract does not match replay")

    reward = _require_exact_mapping(
        payload["reward_contract"],
        label="V10 visible binding reward_contract",
        keys=frozenset({"reward_profile", "reward_weights"}),
    )
    if reward["reward_profile"] != RUNB2_V10_REWARD_PROFILE:
        raise CandidateContractError("V10 visible binding reward_profile is wrong")
    expected_weight_keys = frozenset(
        {
            "w_sla",
            "w_tput",
            "w_fair",
            "w_buffer",
            "w_sla_level",
            "w_prb_level",
            "w_access_level",
            "w_fair_level",
            "w_action",
            "w_reject",
        }
    )
    weights = _require_exact_mapping(
        reward["reward_weights"],
        label="V10 visible binding reward_weights",
        keys=expected_weight_keys,
    )
    for key, raw in weights.items():
        if not isinstance(raw, (int, float)) or isinstance(raw, bool) or not math.isfinite(float(raw)):
            raise CandidateContractError(f"V10 visible binding reward weight {key} must be finite")

    capacity = _require_exact_mapping(
        payload["capacity_contract"],
        label="V10 visible binding capacity_contract",
        keys=frozenset({"unit", "candidate_cell_capacity_mbps"}),
    )
    if capacity["unit"] != RUNB2_V10_CAPACITY_UNIT:
        raise CandidateContractError("V10 visible binding capacity unit is wrong")
    if (
        _require_finite_positive_float(
            capacity["candidate_cell_capacity_mbps"],
            label="V10 visible binding candidate_cell_capacity_mbps",
        )
        != RUNB2_V10_CELL_CAPACITY_MBPS
    ):
        raise CandidateContractError("V10 visible binding capacity is not pinned to replay")

    runtime_manifest = _require_exact_mapping(
        payload["runtime_manifest"],
        label="V10 visible binding runtime_manifest",
        keys=_RUNTIME_MANIFEST_BINDING_KEYS,
    )
    if runtime_manifest["schema_version"] != RUNB2_V10_RUNTIME_MANIFEST_SCHEMA:
        raise CandidateContractError("V10 visible binding runtime manifest schema is wrong")
    _require_sha256(runtime_manifest["sha256"], label="V10 visible binding runtime manifest SHA")

    launch_contract = _require_exact_mapping(
        payload["launch_contract"],
        label="V10 visible binding launch_contract",
        keys=_LAUNCH_CONTRACT_BINDING_KEYS,
    )
    if launch_contract["schema_version"] != RUNB2_V10_LAUNCH_CONTRACT_SCHEMA:
        raise CandidateContractError("V10 visible binding launch contract schema is wrong")
    _require_sha256(
        launch_contract["system_prompt_sha256"],
        label="V10 visible binding system prompt SHA",
    )
    _require_sha256(
        launch_contract["task_manifest_sha256"],
        label="V10 visible binding task manifest SHA",
    )

    if payload["candidate_support_sha256"] != support.support_sha256:
        raise CandidateContractError("V10 visible binding support SHA does not match RCP")
    if payload["observation_without_binding_sha256"] != text_sha256(support.observation_text):
        raise CandidateContractError("V10 visible binding observation SHA does not match pre-binding text")
    capacity_rows = payload["candidate_capacity_milli_mbps_by_cell"]
    expected_rows = [dict(row) for row in support.capacity_milli_mbps_by_cell]
    if capacity_rows != expected_rows:
        raise CandidateContractError("V10 visible binding capacity rows do not match RCP support")
    if any(row["capacity_milli_mbps"] != RUNB2_V10_CELL_CAPACITY_MILLI_MBPS for row in expected_rows):
        raise CandidateContractError("V10 visible binding capacity rows are not pinned to 250000 milli_mbps")
    # Revalidate rows through the same strict support serializer, including
    # sort/order/type checks, without giving the payload an alternate action
    # source of truth.
    canonical_support_payload(
        actions=support.actions,
        capacity_milli_mbps_by_cell=capacity_rows,
    )
    return payload


def attach_visible_binding(support: CandidateSupport, *, binding_payload: Mapping[str, Any]) -> CandidateSupport:
    """Append the exact V10B line after a validated RCP/RCC/RCA block."""

    if support.visible_binding_schema is not None or support.visible_binding_sha256 is not None:
        raise CandidateContractError("V10 candidate support already has a visible binding")
    if any(line.startswith(f"{RUNB2_V10_VISIBLE_BINDING_PREFIX}|") for line in support.observation_text.splitlines()):
        raise CandidateContractError("pre-binding V10 candidate support must not already contain a V10B row")
    payload = validate_visible_binding_payload(binding_payload, support=support)
    digest = canonical_json_sha256(payload)
    line = f"{RUNB2_V10_VISIBLE_BINDING_PREFIX}|{RUNB2_V10_VISIBLE_BINDING_SCHEMA}|{digest}"
    observation_text = f"{support.observation_text}\n{line}"
    # The V10B line is a terminal attestation, not an optional metadata row.
    # Keep this assertion next to the renderer so a later formatting edit
    # cannot quietly move/duplicate it.
    rendered_lines = observation_text.split("\n")
    binding_rows = [
        index for index, row in enumerate(rendered_lines) if row.startswith(f"{RUNB2_V10_VISIBLE_BINDING_PREFIX}|")
    ]
    if binding_rows != [len(rendered_lines) - 1]:
        raise CandidateContractError("V10B row must occur exactly once as the final line")
    return CandidateSupport(
        policy_text=support.policy_text,
        observation_text=observation_text,
        actions=support.actions,
        capacity_milli_mbps_by_cell=support.capacity_milli_mbps_by_cell,
        support_sha256=support.support_sha256,
        visible_binding_schema=RUNB2_V10_VISIBLE_BINDING_SCHEMA,
        visible_binding_sha256=digest,
        visible_binding_payload=payload,
    )


def build_t2_prb_support(observation: Observation) -> CandidateSupport:
    """Render one T2 policy prompt and its server-authored candidate block.

    ``derive_t2_visible_action_set_from_text`` intentionally reconstructs the
    finite support from the policy-visible compact text, so a server-side
    source of truth cannot silently depend on hidden scenario labels.
    """

    if observation.global_.tier.upper() != "T2":
        raise CandidateContractError("V10 candidate support requires a T2 observation")
    policy_text = render.to_compact_user_text(
        observation,
        capacity_milli_mbps=RUNB2_V10_CELL_CAPACITY_MILLI_MBPS,
    )
    visible = derive_t2_visible_action_set_from_text(
        policy_text,
        capacity_milli_mbps=RUNB2_V10_CELL_CAPACITY_MILLI_MBPS,
    )
    actions = tuple(canonical_action(action) for action in visible.actions)
    capacities = tuple(
        {
            "cell_id": int(cell.cell_id),
            "capacity_milli_mbps": RUNB2_V10_CELL_CAPACITY_MILLI_MBPS,
        }
        for cell in sorted(observation.cells, key=lambda item: item.cell_id)
    )
    payload = canonical_support_payload(
        actions=actions,
        capacity_milli_mbps_by_cell=capacities,
    )
    digest = support_sha256(payload)
    lines = [
        policy_text,
        f"RCP|{RESOURCE_CANDIDATE_CONTRACT}|{digest}|{len(actions)}",
        *(f"RCC|{row['cell_id']}|{row['capacity_milli_mbps']}" for row in capacities),
        *(f"RCA|{index}|{canonical_json(action)}" for index, action in enumerate(actions)),
    ]
    return CandidateSupport(
        policy_text=policy_text,
        observation_text="\n".join(lines),
        actions=actions,
        capacity_milli_mbps_by_cell=capacities,
        support_sha256=digest,
    )


def parse_rendered_support(observation_text: str) -> CandidateSupport:
    """Independently parse/rehash a rendered RCP/RCC/RCA block for tests.

    This is deliberately strict and only used for verification.  The server
    never consumes its own parsed output to decide what support to enforce.
    """

    if not isinstance(observation_text, str) or not observation_text:
        raise CandidateContractError("candidate observation must be non-empty text")
    lines = observation_text.splitlines()
    rcp_indices = [index for index, line in enumerate(lines) if line.startswith("RCP|")]
    if len(rcp_indices) != 1:
        raise CandidateContractError("observation must contain exactly one RCP row")
    rcp_index = rcp_indices[0]
    header = lines[rcp_index].split("|")
    if len(header) != 4 or header[1] != RESOURCE_CANDIDATE_CONTRACT:
        raise CandidateContractError("RCP row has the wrong candidate contract")
    if _SHA256_RE.fullmatch(header[2]) is None:
        raise CandidateContractError("RCP support SHA-256 must be lowercase hexadecimal")
    try:
        count = int(header[3])
    except ValueError as exc:
        raise CandidateContractError("RCP candidate count must be an integer") from exc
    if count < 1 or str(count) != header[3]:
        raise CandidateContractError("RCP candidate count must be canonical and positive")
    binding_indices = [
        index for index, line in enumerate(lines) if line.startswith(f"{RUNB2_V10_VISIBLE_BINDING_PREFIX}|")
    ]
    if len(binding_indices) > 1:
        raise CandidateContractError("observation must contain at most one V10B row")
    binding_schema: str | None = None
    binding_sha256: str | None = None
    contract_end = len(lines)
    if binding_indices:
        binding_index = binding_indices[0]
        if binding_index != len(lines) - 1 or binding_index <= rcp_index:
            raise CandidateContractError("V10B row must be the final contract row")
        fields = lines[binding_index].split("|")
        if (
            len(fields) != 3
            or fields[0] != RUNB2_V10_VISIBLE_BINDING_PREFIX
            or fields[1] != RUNB2_V10_VISIBLE_BINDING_SCHEMA
            or _SHA256_RE.fullmatch(fields[2]) is None
        ):
            raise CandidateContractError("V10B row has the wrong binding contract")
        binding_schema = fields[1]
        binding_sha256 = fields[2]
        contract_end = binding_index

    capacities: list[dict[str, int]] = []
    action_rows: list[str] = []
    saw_action = False
    for line in lines[rcp_index + 1 : contract_end]:
        if line.startswith("RCC|"):
            if saw_action:
                raise CandidateContractError("RCC rows must precede all RCA rows")
            fields = line.split("|")
            capacity_index = len(capacities)
            if len(fields) != 3:
                raise CandidateContractError(f"RCC row {capacity_index} has invalid field count")
            try:
                cell_id, capacity = int(fields[1]), int(fields[2])
            except ValueError as exc:
                raise CandidateContractError(f"RCC row {capacity_index} must contain integers") from exc
            if str(cell_id) != fields[1] or str(capacity) != fields[2]:
                raise CandidateContractError(f"RCC row {capacity_index} is not canonical")
            capacities.append({"cell_id": cell_id, "capacity_milli_mbps": capacity})
        elif line.startswith("RCA|"):
            saw_action = True
            action_rows.append(line)
        else:
            raise CandidateContractError("unexpected row inside rendered RCP contract")

    actions: list[dict[str, Any]] = []
    if len(action_rows) != count:
        raise CandidateContractError("RCP count does not equal the number of RCA rows")
    for index, line in enumerate(action_rows):
        fields = line.split("|", 2)
        if len(fields) != 3 or fields[1] != str(index):
            raise CandidateContractError("RCA rows must be contiguous and ordered")
        try:
            action = canonical_action(parse_strict_json(fields[2], label=f"RCA row {index}"))
        except CandidateContractError as exc:
            raise CandidateContractError(f"RCA row {index} is invalid") from exc
        if fields[2] != canonical_json(action):
            raise CandidateContractError(f"RCA row {index} is not canonical")
        actions.append(action)
    payload = canonical_support_payload(
        actions=actions,
        capacity_milli_mbps_by_cell=capacities,
    )
    digest = support_sha256(payload)
    if digest != header[2]:
        raise CandidateContractError("RCP support SHA does not match its visible rows")
    return CandidateSupport(
        policy_text="\n".join(lines[:rcp_index]),
        observation_text=observation_text,
        actions=tuple(actions),
        capacity_milli_mbps_by_cell=tuple(capacities),
        support_sha256=digest,
        visible_binding_schema=binding_schema,
        visible_binding_sha256=binding_sha256,
    )


def is_supported_action(action: ToolCall | Mapping[str, Any], support: CandidateSupport) -> bool:
    """Return membership without converting malformed calls into a noop."""

    try:
        return action_key(action) in support.action_keys
    except CandidateContractError:
        return False


__all__ = [
    "CandidateContractError",
    "CandidateSupport",
    "RESOURCE_ACTION_ARGUMENT_CONTRACT",
    "RESOURCE_CANDIDATE_CONTRACT",
    "RUNB2_V10_ACTION_EFFECT_CONTRACT_SCHEMA",
    "RUNB2_V10_ACTION_SCOPE",
    "RUNB2_V10_CAPACITY_UNIT",
    "RUNB2_V10_CELL_CAPACITY_MBPS",
    "RUNB2_V10_CELL_CAPACITY_MILLI_MBPS",
    "RUNB2_V10_MAX_STEPS_HARD_CAP",
    "RUNB2_V10_LAUNCH_CONTRACT_SCHEMA",
    "RUNB2_V10_OBSERVATION_RENDER",
    "RUNB2_V10_PROTOCOL_MODE",
    "RUNB2_V10_REWARD_PROFILE",
    "RUNB2_V10_RUNTIME_MANIFEST_SCHEMA",
    "RUNB2_V10_VISIBLE_BINDING_PREFIX",
    "RUNB2_V10_VISIBLE_BINDING_SCHEMA",
    "action_key",
    "attach_visible_binding",
    "build_t2_prb_support",
    "build_visible_binding_payload",
    "canonical_action",
    "canonical_json",
    "canonical_json_sha256",
    "parse_strict_json",
    "canonical_support_payload",
    "is_supported_action",
    "parse_rendered_support",
    "support_sha256",
    "text_sha256",
    "validate_visible_binding_payload",
]
