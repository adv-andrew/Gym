# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


pytest.importorskip(
    "openair_congestion",
    reason="telco env package 'openair_congestion' not installed; see README Setup",
)

from openair_congestion.schemas import Observation, ToolCall  # noqa: E402

from nemo_gym.openai_utils import (  # noqa: E402
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
)
from nemo_gym.server_utils import ServerClient  # noqa: E402
from resources_servers.openair_congestion.app import (  # noqa: E402
    OpenAirCongestionEnv,
    OpenAirCongestionResourcesServerConfig,
    _to_resource_candidate_pipe_v1,
)
from resources_servers.openair_congestion.backends import (  # noqa: E402
    NAMED_REWARD_PROFILE_OVERRIDES,
)
from resources_servers.openair_congestion.candidate_contract import (  # noqa: E402
    RESOURCE_CANDIDATE_CONTRACT,
    ResourceCandidateSupport,
    derive_resource_candidate_support,
    parse_resource_candidate_support,
    validate_resource_candidate_guardrail_contract,
)
from resources_servers.openair_congestion.dataset_backend import (  # noqa: E402
    load_provided_dataset,
)


_RESPONSE_KWARGS = dict(
    id="r",
    created_at=0.0,
    model="m",
    object="response",
    parallel_tool_calls=True,
    tool_choice="auto",
    tools=[],
)


def _tool_response(action: ToolCall) -> NeMoGymResponse:
    return NeMoGymResponse(
        output=[
            NeMoGymResponseFunctionToolCall(
                arguments=json.dumps(action.arguments),
                call_id="call_0",
                name=action.name,
                type="function_call",
                id="fc_0",
                status="completed",
            )
        ],
        **_RESPONSE_KWARGS,
    )


def _sparse_dataset(path: Path, *, cell_id: int = 0) -> Path:
    rows = []
    for step in range(3):
        rows.append(
            {
                "episode_id": "sparse_ids",
                "step": step,
                "t_s": float(step),
                "cells": [
                    {
                        "cell_id": cell_id,
                        "prb_util_dl_p50": 0.95,
                        "ues": [
                            {
                                "ue_id": 7,
                                "offered_mbps": 40.0,
                                "delivered_mbps": 35.0,
                                "buffer_occupancy_kb": 700.0,
                            },
                            {
                                "ue_id": 19,
                                "offered_mbps": 40.0,
                                "delivered_mbps": 25.0,
                                "buffer_occupancy_kb": 900.0,
                            },
                        ],
                    }
                ],
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _make_dataset_env(path: Path) -> OpenAirCongestionEnv:
    config = OpenAirCongestionResourcesServerConfig(
        host="",
        port=0,
        entrypoint="",
        name="openair_congestion",
        backend="dataset_replay",
        dataset_path=str(path),
        cell_capacity_mbps=73.25,
        max_steps_default=2,
        agent_max_steps=2,
        reward_profile="dataset_validity_v1",
        reward_weights=NAMED_REWARD_PROFILE_OVERRIDES["dataset_validity_v1"],
        observation_render=RESOURCE_CANDIDATE_CONTRACT,
    )
    return OpenAirCongestionEnv(
        config=config,
        server_client=MagicMock(spec=ServerClient),
    )


def _with_policy_fields(
    observation: Observation,
    *,
    requested: tuple[float, ...],
    admitted: tuple[float, ...],
    caps: tuple[int | None, ...],
) -> Observation:
    payload = observation.model_dump(by_alias=True)
    for ue, req, adm, cap in zip(
        payload["cells"][0]["ues"],
        requested,
        admitted,
        caps,
        strict=True,
    ):
        ue["requested_mbps"] = req
        ue["admitted_mbps"] = adm
        ue["prb_cap_max_prb"] = cap
    return Observation.model_validate(payload)


class TestResourceCandidateContract:
    def test_installed_guardrail_contract_is_bound_at_boot(self):
        receipt = validate_resource_candidate_guardrail_contract()
        assert receipt["pass"] is True
        assert receipt["repeat_window_steps"] == 2
        assert all(receipt["checks"].values())

    @pytest.mark.asyncio
    async def test_replay_support_binds_installed_episode_capacity(self):
        config = OpenAirCongestionResourcesServerConfig(
            host="",
            port=0,
            entrypoint="",
            name="openair_congestion",
            backend="replay",
            max_steps_default=2,
            agent_max_steps=2,
            observation_render=RESOURCE_CANDIDATE_CONTRACT,
        )
        env = OpenAirCongestionEnv(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )
        text, info = await env.reset(
            {
                "seed": 811,
                "difficulty": 0.9,
                "regime_mix": {"prb_exhaustion": 1.0},
                "tier": "replay",
                "max_steps": 2,
            },
            session_id="sid",
        )
        support = parse_resource_candidate_support(text)
        episode = env.backend._env._episodes[info["episode_id"]]
        expected_milli = round(episode.fingerprint.cell_capacity_mbps * 1000)

        assert support.capacity_milli_mbps_by_cell
        assert {capacity for _, capacity in support.capacity_milli_mbps_by_cell} == {expected_milli}

    def test_text_round_trip_authenticates_capacity_actions_and_hash(self, tmp_path):
        source = load_provided_dataset(_sparse_dataset(tmp_path / "sparse.jsonl"))
        observation = source["sparse_ids"].observations[0]
        support = derive_resource_candidate_support(
            observation,
            capacity_mbps_by_cell={0: 73.25},
        )
        text = _to_resource_candidate_pipe_v1(observation, support)
        parsed = parse_resource_candidate_support(text)

        assert parsed == support
        assert "RCC|0|73250" in text
        assert f"RCP|{RESOURCE_CANDIDATE_CONTRACT}|{support.support_sha256}|" in text
        assert support.payload == {
            "contract": RESOURCE_CANDIDATE_CONTRACT,
            "capacity_milli_mbps_by_cell": [{"cell_id": 0, "capacity_milli_mbps": 73250}],
            "actions": [action.model_dump(by_alias=True) for action in support.actions],
        }

    def test_parser_rejects_noncanonical_rows_and_action_without_capacity(self, tmp_path):
        observation = load_provided_dataset(_sparse_dataset(tmp_path / "sparse.jsonl"))["sparse_ids"].observations[0]
        support = derive_resource_candidate_support(
            observation,
            capacity_mbps_by_cell={0: 73.25},
        )
        text = _to_resource_candidate_pipe_v1(observation, support)
        with pytest.raises(ValueError, match="canonical JSON"):
            parse_resource_candidate_support(
                text.replace('RCA|0|{"arguments":{},"name":"noop"}', 'RCA|0|{ "arguments":{},"name":"noop"}')
            )
        with pytest.raises(ValueError, match="canonical decimal"):
            parse_resource_candidate_support(text.replace("RCA|0|", "RCA|00|"))
        with pytest.raises(ValueError, match="capacity row"):
            ResourceCandidateSupport(
                capacity_milli_mbps_by_cell=((0, 73250),),
                actions=(
                    ToolCall(name="noop", arguments={}),
                    ToolCall(
                        name="set_prb_cap",
                        arguments={
                            "cell_id": 2,
                            "target": "ue",
                            "target_id": 7,
                            "max_prb": 200,
                        },
                    ),
                ),
            )

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("delivered_mbps", None, "missing required"),
            ("delivered_mbps", float("inf"), "finite nonnegative"),
            ("offered_mbps", float("nan"), "finite nonnegative"),
            ("requested_mbps", float("inf"), "finite nonnegative"),
            ("admitted_mbps", float("nan"), "finite nonnegative"),
            ("buffer_occupancy_kb", float("inf"), "finite nonnegative"),
            ("prb_cap_max_prb", True, "integer in"),
            ("prb_cap_max_prb", 240.9, "integer in"),
        ],
    )
    def test_dataset_rejects_invalid_candidate_state_at_ingestion(
        self, tmp_path, field, value, message
    ):
        path = _sparse_dataset(tmp_path / "invalid.jsonl")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["cells"][0]["ues"][0][field] = value
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        with pytest.raises(ValueError, match=message):
            load_provided_dataset(path)

    def test_arbitrary_visible_ue_ids_are_preserved(self, tmp_path):
        observation = load_provided_dataset(_sparse_dataset(tmp_path / "sparse.jsonl"))["sparse_ids"].observations[0]
        support = derive_resource_candidate_support(
            observation,
            capacity_mbps_by_cell={0: 60.0},
        )
        target_ids = {action.arguments["target_id"] for action in support.actions if action.name == "set_prb_cap"}
        assert target_ids
        assert target_ids <= {7, 19}
        assert 0 not in target_ids and 1 not in target_ids

    def test_redundant_setpoint_and_recent_repeat_are_not_emitted(self, tmp_path):
        observation = load_provided_dataset(_sparse_dataset(tmp_path / "sparse.jsonl"))["sparse_ids"].observations[0]
        overloaded = _with_policy_fields(
            observation,
            requested=(40.0, 40.0),
            admitted=(40.0, 40.0),
            caps=(200, None),
        )
        support = derive_resource_candidate_support(
            overloaded,
            capacity_mbps_by_cell={0: 60.0},
        )
        assert all(
            not (
                action.name == "set_prb_cap"
                and action.arguments["target_id"] == 7
                and action.arguments["max_prb"] == 200
            )
            for action in support.actions
        )

        first_action = next(action for action in support.actions if action.name != "noop")
        repeated = derive_resource_candidate_support(
            overloaded,
            capacity_mbps_by_cell={0: 60.0},
            excluded_actions=[first_action],
        )
        assert first_action not in repeated.actions

        recovered = _with_policy_fields(
            observation,
            requested=(20.0, 20.0),
            admitted=(20.0, 20.0),
            caps=(240, 250),
        )
        releases = derive_resource_candidate_support(
            recovered,
            capacity_mbps_by_cell={0: 60.0},
        )
        release = next(action for action in releases.actions if action.name != "noop")
        assert release.arguments == {
            "cell_id": 0,
            "target": "ue",
            "target_id": 7,
            "max_prb": 273,
        }
        assert all(action.arguments.get("max_prb") not in {240, 250} for action in releases.actions[1:])

    @pytest.mark.asyncio
    async def test_emitted_sparse_id_candidate_is_accepted_and_receipted(self, tmp_path):
        env = _make_dataset_env(_sparse_dataset(tmp_path / "sparse.jsonl"))
        text, reset_info = await env.reset(
            {"scenario_id": "sparse_ids", "max_steps": 2},
            session_id="sid",
        )
        support = parse_resource_candidate_support(text)
        candidate = next(action for action in support.actions if action.name != "noop")

        next_text, reward, terminated, truncated, step_info = await env.step(
            _tool_response(candidate),
            {},
            session_id="sid",
        )

        assert reward == 0.0
        assert step_info["guardrail_accepted"] is True
        assert step_info["rejection_reason"] is None
        assert step_info["submitted_candidate_support_sha256"] == support.support_sha256
        assert step_info["submitted_candidate_supported"] is True
        assert reset_info["candidate_support_sha256"] == support.support_sha256
        assert reset_info["candidate_actions"] == [action.model_dump(by_alias=True) for action in support.actions]
        assert terminated is False and truncated is False
        next_support = parse_resource_candidate_support(next_text)
        assert step_info["candidate_support_sha256"] == next_support.support_sha256
        assert candidate not in next_support.actions

    @pytest.mark.asyncio
    async def test_emitted_nonzero_cell_id_candidate_is_accepted(self, tmp_path):
        env = _make_dataset_env(
            _sparse_dataset(tmp_path / "sparse-cell.jsonl", cell_id=2)
        )
        text, _ = await env.reset(
            {"scenario_id": "sparse_ids", "max_steps": 2},
            session_id="sid",
        )
        support = parse_resource_candidate_support(text)
        candidate = next(action for action in support.actions if action.name != "noop")

        _, reward, _, _, info = await env.step(
            _tool_response(candidate),
            {},
            session_id="sid",
        )

        assert candidate.arguments["cell_id"] == 2
        assert reward == 0.0
        assert info["submitted_candidate_supported"] is True
        assert info["guardrail_accepted"] is True

    @pytest.mark.asyncio
    async def test_validity_profile_reward_is_exact_reject_only(self, tmp_path):
        env = _make_dataset_env(_sparse_dataset(tmp_path / "sparse.jsonl"))
        await env.reset(
            {"scenario_id": "sparse_ids", "max_steps": 2},
            session_id="sid",
        )
        # This is a schema-valid tool call but not a member of the finite
        # support; the server consumes one rejected transition.
        outside_support = ToolCall(
            name="set_prb_cap",
            arguments={
                "cell_id": 0,
                "target": "ue",
                "target_id": 19,
                "max_prb": 273,
            },
        )
        _, reward, _, _, info = await env.step(
            _tool_response(outside_support),
            {},
            session_id="sid",
        )

        assert reward == -0.5
        assert info["reward_profile"] == "dataset_validity_v1"
        assert info["guardrail_accepted"] is False
        assert info["error"] == "candidate_support_violation"
        assert info["reward_terms"]["reject"] == -0.5
        assert info["reward_terms"]["total"] == -0.5
        assert all(value == 0.0 for name, value in info["reward_terms"].items() if name not in {"reject", "total"})
        assert info["reward_weights"] == NAMED_REWARD_PROFILE_OVERRIDES["dataset_validity_v1"]

    @pytest.mark.asyncio
    async def test_boolean_ids_cannot_alias_advertised_integer_candidate(self, tmp_path):
        env = _make_dataset_env(_sparse_dataset(tmp_path / "sparse.jsonl"))
        text, _ = await env.reset(
            {"scenario_id": "sparse_ids", "max_steps": 2},
            session_id="sid",
        )
        support = parse_resource_candidate_support(text)
        candidate = next(action for action in support.actions if action.name != "noop")
        malformed = ToolCall(
            name=candidate.name,
            arguments={
                **candidate.arguments,
                "cell_id": False,
                "target_id": False,
            },
        )

        _, reward, _, _, info = await env.step(
            _tool_response(malformed),
            {},
            session_id="sid",
        )

        assert reward == -0.5
        assert info["submitted_candidate_supported"] is False
        assert info["guardrail_accepted"] is False
        assert info["error"] == "candidate_support_violation"
