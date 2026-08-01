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
# Unit tests for the openair_congestion gymnasium-style resources_server,
# modeled on resources_servers/blackjack/tests/test_app.py (direct
# reset()/step() calls with a mock ServerClient).
#
# Requires the cross-repo telco env package (see README Setup). Tests are
# skipped, not failed, if the package is missing.
import json
import math
from unittest.mock import MagicMock

import httpx
import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient


openair = pytest.importorskip(
    "openair_congestion",
    reason="telco env package 'openair_congestion' not installed; see README Setup",
)

from resources_servers.openair_congestion import app as app_source  # noqa: E402
from resources_servers.openair_congestion import candidate_contract  # noqa: E402
from resources_servers.openair_congestion.app import (  # noqa: E402
    OpenAirCongestionEnv,
    OpenAirCongestionResourcesServerConfig,
    V10ProtocolError,
)
from resources_servers.openair_congestion.backends import (  # noqa: E402
    OAICollectorBackend,
    ReplayBackend,
    V10FixedReplayBackend,
    select_backend,
)


def _make_env(**config_overrides) -> OpenAirCongestionEnv:
    config_values = {"host": "", "port": 0, "entrypoint": "", "name": ""}
    config_values.update(config_overrides)
    config = OpenAirCongestionResourcesServerConfig(**config_values)
    return OpenAirCongestionEnv(config=config, server_client=MagicMock(spec=ServerClient))


_RESPONSE_KWARGS = dict(
    id="r",
    created_at=0.0,
    model="m",
    object="response",
    parallel_tool_calls=True,
    tool_choice="auto",
    tools=[],
)


def _text_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        output=[
            NeMoGymResponseOutputMessage(
                id="msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        **_RESPONSE_KWARGS,
    )


def _tool_response(name: str, arguments: dict) -> NeMoGymResponse:
    return NeMoGymResponse(
        output=[
            NeMoGymResponseFunctionToolCall(
                arguments=json.dumps(arguments),
                call_id="call_0",
                name=name,
                type="function_call",
                id="fc_0",
                status="completed",
            )
        ],
        **_RESPONSE_KWARGS,
    )


def _raw_tool_response(name: str, arguments: str) -> NeMoGymResponse:
    """Build one deliberately malformed function call for protocol tests."""

    return NeMoGymResponse(
        output=[
            NeMoGymResponseFunctionToolCall(
                arguments=arguments,
                call_id="call_0",
                name=name,
                type="function_call",
                id="fc_0",
                status="completed",
            )
        ],
        **_RESPONSE_KWARGS,
    )


def _multi_tool_response(*actions: tuple[str, dict]) -> NeMoGymResponse:
    return NeMoGymResponse(
        output=[
            NeMoGymResponseFunctionToolCall(
                arguments=json.dumps(arguments),
                call_id=f"call_{index}",
                name=name,
                type="function_call",
                id=f"fc_{index}",
                status="completed",
            )
            for index, (name, arguments) in enumerate(actions)
        ],
        **_RESPONSE_KWARGS,
    )


_TASK_METADATA = {
    "seed": 7001,
    "difficulty": 0.6,
    "regime_mix": {"prb_exhaustion": 1.0},
    "scenario_id": "prb_exhaustion",
    "tier": "replay",
    "max_steps": 16,
}

_V10_TASK_METADATA = {
    **_TASK_METADATA,
    "tier": "T2",
    "max_steps": 4,
}
_V10_ACTION_TASK_METADATA = {
    **_V10_TASK_METADATA,
    "difficulty": 0.8,
}
_V10_TEST_SESSION_SECRET = "test-only-v10-session-secret-2v0yQdG2w8Tn0Hkx6cA7"
_V10_TEST_SYSTEM_PROMPT_SHA256 = "1" * 64
_V10_TEST_TASK_MANIFEST_SHA256 = "2" * 64
_V10_SCENARIO_SOURCE = "trainer_worktree_congestion_gen_sampler_v1"
_V10_CONGESTION_GEN_SOURCE_PATHS = {
    "congestion_gen.package": "services/congestion-gen/congestion_gen/__init__.py",
    "congestion_gen.materializer": ("services/congestion-gen/congestion_gen/materializer.py"),
    "congestion_gen.sampler": "services/congestion-gen/congestion_gen/sampler.py",
    "congestion_gen.schemas": "services/congestion-gen/congestion_gen/schemas.py",
    "congestion_gen.validate": "services/congestion-gen/congestion_gen/validate.py",
}
_V10_EXPECTED_RUNTIME_SOURCE_IDS = frozenset(
    {
        "resource_server.app",
        "resource_server.backends",
        "resource_server.candidate_contract",
        "gymnasium.package",
        "gymnasium.base",
        "nemo_gym.package",
        "nemo_gym.package_info",
        "nemo_gym.cli.package",
        "nemo_gym.cli.compat",
        "nemo_gym.base_resources_server",
        "nemo_gym.server_utils",
        "nemo_gym.openai_utils",
        "nemo_gym.config_types",
        "nemo_gym.global_config",
        "nemo_gym.reward_profile",
        "nemo_gym.profiling",
        "openair_congestion.package",
        "openair_congestion.env",
        "openair_congestion.guardrail",
        "openair_congestion.kpi_client",
        "openair_congestion.render",
        "openair_congestion.replay_env",
        "openair_congestion.rewards",
        "openair_congestion.schemas",
        "openair_congestion.t2_action_mask",
        "openair_congestion.t2_candidate_sampler",
        "openair_congestion.t2_policy_features",
        "openair_congestion.tools",
        "openair_congestion.v10_fixed_replay",
        "congestion_gen.package",
        "congestion_gen.materializer",
        "congestion_gen.sampler",
        "congestion_gen.schemas",
        "congestion_gen.validate",
    }
)
_V10_EXPECTED_REWARD_WEIGHTS = {
    "w_sla": 1.0,
    "w_tput": 2.0,
    "w_fair": 5.0,
    "w_buffer": 0.15,
    "w_sla_level": 0.8,
    "w_prb_level": 0.4,
    "w_access_level": 0.3,
    "w_fair_level": 0.35,
    "w_action": 0.0,
    "w_reject": 0.5,
}
_ZERO_FORCED_V10_SERVICE_ACCOUNTING = {
    "requested_service_mbps": 10.0,
    "admitted_service_mbps": 8.0,
    "delivered_service_mbps": 6.0,
    "forced_terminated_service_mbps": 0.0,
    "cumulative_forced_terminated_service_mbps": 0.0,
    "forced_termination_events": 0.0,
    "step_forced_terminated_service_mbps": 0.0,
    "step_forced_termination_events": 0.0,
    "unadmitted_service_mbps": 2.0,
    "undelivered_admitted_service_mbps": 2.0,
}
_VALID_V10_SERVICE_ACCOUNTING = dict(_ZERO_FORCED_V10_SERVICE_ACCOUNTING)
_PREVIOUS_V10_SERVICE_ACCOUNTING = dict(_ZERO_FORCED_V10_SERVICE_ACCOUNTING)


class TestReset:
    @pytest.mark.asyncio
    async def test_reset_populates_state_and_renders_kpis(self):
        env = _make_env()
        obs, info = await env.reset(dict(_TASK_METADATA), session_id="sid")
        assert "sid" in env.session_state
        state = env.session_state["sid"]
        assert state["episode_id"] == info["episode_id"]
        assert state["cumulative_reward"] == 0.0
        assert state["n_steps"] == 0
        assert "5G RAN telemetry" in obs  # render.to_user_text output
        assert info["seed"] == 7001
        assert info["scenario_id"] == "prb_exhaustion"

    @pytest.mark.asyncio
    async def test_sessions_are_isolated(self):
        env = _make_env()
        _, info_a = await env.reset(dict(_TASK_METADATA), session_id="a")
        _, info_b = await env.reset(dict(_TASK_METADATA, seed=7002), session_id="b")
        assert info_a["episode_id"] != info_b["episode_id"]
        assert env.session_state["a"]["episode_id"] != env.session_state["b"]["episode_id"]

    @pytest.mark.asyncio
    async def test_re_reset_same_session_closes_old_episode(self):
        # A client retry POSTing /reset twice with the same session cookie
        # must not leak the first episode's backend pool slot.
        env = _make_env()
        _, info_old = await env.reset(dict(_TASK_METADATA), session_id="sid")
        _, info_new = await env.reset(dict(_TASK_METADATA), session_id="sid")
        assert info_new["episode_id"] != info_old["episode_id"]
        assert env.session_state["sid"]["episode_id"] == info_new["episode_id"]
        # The old episode was closed during the second reset: closing it again
        # must raise KeyError (unknown episode_id) inside the backend.
        with pytest.raises(KeyError):
            env.backend.close(info_old["episode_id"])

    @pytest.mark.asyncio
    async def test_orphaned_episode_reaped_when_pool_exhausted(self):
        # A rollout that dies between /reset and its terminal /step leaks a
        # pool slot; the reaper must reclaim it on the next pool-exhausted
        # reset instead of failing every new rollout until server restart.
        env = _make_env(pool_size=1)
        _, info_dead = await env.reset(dict(_TASK_METADATA), session_id="dead")
        # Simulate the crash: the session vanishes without close_session().
        del env.session_state["dead"]
        # Pool (size 1) is exhausted, but the orphaned episode has no live
        # session, so ReplayBackend reaps it and the retry succeeds.
        _, info_new = await env.reset(dict(_TASK_METADATA, seed=7002), session_id="new")
        assert info_new["episode_id"] != info_dead["episode_id"]
        assert env.session_state["new"]["episode_id"] == info_new["episode_id"]

    @pytest.mark.asyncio
    async def test_missing_max_steps_falls_back_to_agent_budget(self):
        # Rows lacking max_steps must NOT fall back to the env default (60):
        # the agent truncates client-side at agent_max_steps (16), and a
        # larger server budget would strand the episode slot.
        env = _make_env()
        metadata = {k: v for k, v in _TASK_METADATA.items() if k != "max_steps"}
        await env.reset(metadata, session_id="sid")
        assert env.session_state["sid"]["max_agent_steps"] == 16


class TestStep:
    @pytest.mark.asyncio
    async def test_noop_step_returns_finite_reward_and_tool_output(self):
        env = _make_env()
        await env.reset(dict(_TASK_METADATA), session_id="sid")
        obs, reward, term, trunc, info = await env.step(_tool_response("noop", {}), {}, session_id="sid")
        assert math.isfinite(reward)
        assert term is False
        assert trunc is False
        assert "5G RAN telemetry" in obs
        assert info["guardrail_accepted"] is True
        # The applied call gets a matching function_call_output for the agent.
        assert info["tool_outputs"][0]["call_id"] == "call_0"
        assert env.session_state["sid"]["n_steps"] == 1

    @pytest.mark.asyncio
    async def test_out_of_range_action_is_rejected_not_crashed(self):
        env = _make_env()
        await env.reset(dict(_TASK_METADATA), session_id="sid")
        # cell_id=99 is in-schema-type but out of range: the env guardrail
        # rejects and applies its own penalty; the server must not raise.
        obs, reward, term, trunc, info = await env.step(
            _tool_response("set_scheduler_policy", {"cell_id": 99, "policy": "PF"}), {}, session_id="sid"
        )
        assert math.isfinite(reward)
        assert info["guardrail_accepted"] is False
        assert info["rejection_reason"]
        assert term is False and trunc is False

    @pytest.mark.asyncio
    async def test_unknown_tool_name_rejected_without_env_step(self):
        env = _make_env()
        await env.reset(dict(_TASK_METADATA), session_id="sid")
        obs, reward, term, trunc, info = await env.step(_tool_response("open_pod_bay_doors", {}), {}, session_id="sid")
        assert reward == 0.0
        assert info["error"] == "invalid_tool_call"
        assert env.session_state["sid"]["n_steps"] == 0  # env did NOT advance

    @pytest.mark.asyncio
    async def test_no_tool_call_returns_zero_reward_nudge(self):
        env = _make_env()
        await env.reset(dict(_TASK_METADATA), session_id="sid")
        obs, reward, term, trunc, info = await env.step(
            _text_response("Hmm, the PRBs look full."), {}, session_id="sid"
        )
        assert reward == 0.0
        assert term is False and trunc is False
        assert "tool call" in obs
        assert env.session_state["sid"]["n_steps"] == 0

    @pytest.mark.asyncio
    async def test_reward_accumulates_per_step_like_blackjack(self):
        # The server returns PER-STEP rewards (the agent sums them); the
        # session's cumulative bookkeeping must equal that sum.
        env = _make_env()
        await env.reset(dict(_TASK_METADATA), session_id="sid")
        total = 0.0
        for _ in range(3):
            _, reward, term, trunc, _ = await env.step(_tool_response("noop", {}), {}, session_id="sid")
            total += reward
            assert not term and not trunc
        assert env.session_state["sid"]["cumulative_reward"] == pytest.approx(total)
        assert env.session_state["sid"]["n_steps"] == 3

    @pytest.mark.asyncio
    async def test_episode_terminates_at_env_max_steps_and_session_closes(self):
        env = _make_env()
        await env.reset(dict(_TASK_METADATA, max_steps=3), session_id="sid")
        term = trunc = False
        for _ in range(3):
            _, _, term, trunc, _ = await env.step(_tool_response("noop", {}), {}, session_id="sid")
        assert term or trunc  # episode ended within the 3-step budget
        # Mirror the framework: /step calls close_session on terminated/truncated.
        await env.close_session("sid")
        assert "sid" not in env.session_state

    @pytest.mark.asyncio
    async def test_step_without_reset_truncates_gracefully(self):
        env = _make_env()
        obs, reward, term, trunc, info = await env.step(_tool_response("noop", {}), {}, session_id="ghost")
        assert reward == 0.0
        assert trunc is True
        assert info["error"] == "no_active_episode"


class TestV10ConstrainedProtocol:
    """The V10 path is deliberately narrower than the generic resource server."""

    def test_public_contract_exports_both_exact_reward_vectors(self):
        assert "runb2_v10_reward_coefficients" in candidate_contract.__all__
        assert "runb2_v10_reward_weights" in candidate_contract.__all__

    @staticmethod
    def _env() -> OpenAirCongestionEnv:
        return _make_env(
            protocol_mode=candidate_contract.RUNB2_V10_PROTOCOL_MODE,
            replay_scenario_source=_V10_SCENARIO_SOURCE,
            cell_capacity_mbps=250.0,
            v10_session_secret=_V10_TEST_SESSION_SECRET,
            v10_system_prompt_sha256=_V10_TEST_SYSTEM_PROMPT_SHA256,
            v10_task_manifest_sha256=_V10_TEST_TASK_MANIFEST_SHA256,
        )

    @pytest.mark.asyncio
    async def test_reset_renders_and_metadata_bind_one_authoritative_support(self):
        env = self._env()
        observation, info = await env.reset(dict(_V10_TASK_METADATA), session_id="sid")
        parsed = candidate_contract.parse_rendered_support(observation)
        assert parsed.support_sha256 == info["candidate_support_sha256"]
        assert parsed.actions == tuple(info["candidate_actions"])
        assert parsed.policy_text.splitlines()[1] == "P|250.000|0|1"
        assert info["visible_action_support_sha256"] == parsed.support_sha256
        assert info["environment_contract"] == {
            "schema_version": candidate_contract.RUNB2_V10_ACTION_EFFECT_CONTRACT_SCHEMA,
            "backend": "replay",
            "dynamics_mode": "synthetic_action_effect_v3_zero_sum_prb273",
            "action_affects_observation": True,
            "candidate_contract": candidate_contract.RESOURCE_CANDIDATE_CONTRACT,
        }
        assert info["action_scope"] == candidate_contract.RUNB2_V10_ACTION_SCOPE
        assert info["reward_profile"] == "openair_t2_v3"
        assert info["reward_version"] == "openair_t2_v3"
        assert info["reward_weights"] == _V10_EXPECTED_REWARD_WEIGHTS
        assert info["tier"] == "T2"
        assert parsed.visible_binding_schema == (candidate_contract.RUNB2_V10_VISIBLE_BINDING_SCHEMA)
        assert parsed.visible_binding_sha256 is not None
        binding = info["server_observation_binding"]
        assert set(binding) == {
            "observation_sha256",
            "candidate_support_sha256",
            "binding_payload",
        }
        assert binding["observation_sha256"] == candidate_contract.text_sha256(observation)
        assert binding["candidate_support_sha256"] == parsed.support_sha256
        assert set(binding["binding_payload"]) == {
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
        assert binding["binding_payload"]["candidate_support_sha256"] == parsed.support_sha256
        assert binding["binding_payload"]["observation_without_binding_sha256"] == (
            candidate_contract.text_sha256(observation.rsplit("\n", 1)[0])
        )
        assert binding["binding_payload"]["capacity_contract"] == {
            "unit": "milli_mbps",
            "candidate_cell_capacity_mbps": 250.0,
        }
        assert binding["binding_payload"]["reward_contract"] == {
            "reward_profile": "openair_t2_v3",
            "reward_version": "openair_t2_v3",
            "reward_weights": info["reward_weights"],
            "reward_coefficients": {
                "service_denial": 1.0,
                "delivery_gap": 1.25,
                "elastic_fairness": 0.25,
                "sla": 2.0,
                "forced_event": 5.0,
                "forced_ratio": 2.0,
                "action": 0.005,
            },
        }
        assert binding["binding_payload"]["runtime_manifest"] == {
            "schema_version": candidate_contract.RUNB2_V10_RUNTIME_MANIFEST_SCHEMA,
            "sha256": info["server_runtime_manifest_sha256"],
        }
        assert binding["binding_payload"]["launch_contract"] == {
            "schema_version": candidate_contract.RUNB2_V10_LAUNCH_CONTRACT_SCHEMA,
            "system_prompt_sha256": _V10_TEST_SYSTEM_PROMPT_SHA256,
            "task_manifest_sha256": _V10_TEST_TASK_MANIFEST_SHA256,
        }
        assert info["v10_launch_contract"] == binding["binding_payload"]["launch_contract"]
        assert info["server_runtime_manifest_sha256"] == candidate_contract.canonical_json_sha256(
            info["server_runtime_manifest"]
        )
        runtime_manifest = info["server_runtime_manifest"]
        assert runtime_manifest["schema_version"] == "openair_runb2_v10_runtime_manifest_v4"
        runtime_source_files = runtime_manifest["source_files"]
        assert len(runtime_source_files) == 34
        assert set(runtime_source_files) == _V10_EXPECTED_RUNTIME_SOURCE_IDS
        public_config = runtime_manifest["effective_public_config"]
        assert public_config["scenario_source"] == _V10_SCENARIO_SOURCE
        assert public_config["replay_scenario_source"] == _V10_SCENARIO_SOURCE
        assert public_config["cell_capacity_mbps"] == 250.0
        assert public_config["reward_profile"] == "openair_t2_v3"
        assert public_config["reward_version"] == "openair_t2_v3"
        assert public_config["reward_coefficients"] == {
            "service_denial": 1.0,
            "delivery_gap": 1.25,
            "elastic_fairness": 0.25,
            "sla": 2.0,
            "forced_event": 5.0,
            "forced_ratio": 2.0,
            "action": 0.005,
        }
        assert public_config["dynamic_congestion_gen_importable"] is True
        assert public_config["dynamic_congestion_gen_configured"] is True
        assert public_config["dynamic_congestion_gen_used"] is True
        assert set(public_config["congestion_gen_source_files"]) == set(_V10_CONGESTION_GEN_SOURCE_PATHS)
        for source_id, relative_path in _V10_CONGESTION_GEN_SOURCE_PATHS.items():
            source_record = public_config["congestion_gen_source_files"][source_id]
            assert source_record["relative_path"] == relative_path
            assert source_record["sha256"] == runtime_source_files[source_id]
        assert _V10_TEST_SESSION_SECRET not in candidate_contract.canonical_json(info["server_runtime_manifest"])
        task_receipt = info["task_budget_receipt"]
        assert task_receipt["schema_version"] == "openair_runb2_v10_task_budget_receipt_v1"
        assert task_receipt["task_params"] == {
            "seed": 7001,
            "difficulty": 0.6,
            "regime_mix": {"prb_exhaustion": 1.0},
            "scenario_id": "prb_exhaustion",
            "tier": "T2",
            "max_steps": 4,
        }
        assert task_receipt["requested_max_steps"] == 4
        assert task_receipt["runtime_manifest_sha256"] == info["server_runtime_manifest_sha256"]
        assert task_receipt["launch_contract"] == info["v10_launch_contract"]
        assert info["task_budget_receipt_sha256"] == candidate_contract.canonical_json_sha256(task_receipt)
        reset_receipt = info["reset_receipt"]
        assert reset_receipt["schema_version"] == "openair_runb2_v10_reset_receipt_v1"
        assert reset_receipt["task_budget_receipt_sha256"] == info["task_budget_receipt_sha256"]
        assert reset_receipt["initial_observation_sha256"] == candidate_contract.text_sha256(observation)
        assert reset_receipt["initial_candidate_support_sha256"] == parsed.support_sha256
        assert reset_receipt["initial_binding_payload_sha256"] == candidate_contract.canonical_json_sha256(
            binding["binding_payload"]
        )
        assert info["reset_receipt_sha256"] == candidate_contract.canonical_json_sha256(reset_receipt)
        assert binding["binding_payload"]["candidate_capacity_milli_mbps_by_cell"] == [
            {"cell_id": row["cell_id"], "capacity_milli_mbps": 250_000}
            for row in binding["binding_payload"]["candidate_capacity_milli_mbps_by_cell"]
        ]
        assert parsed.visible_binding_sha256 == candidate_contract.canonical_json_sha256(binding["binding_payload"])
        assert observation.rsplit("\n", 1)[1] == (
            f"V10B|{candidate_contract.RUNB2_V10_VISIBLE_BINDING_SCHEMA}|{parsed.visible_binding_sha256}"
        )
        assert sum(line.startswith("V10B|") for line in observation.split("\n")) == 1

        tampered = observation.replace("RCA|0|", "RCA|1|", 1)
        with pytest.raises(candidate_contract.CandidateContractError):
            candidate_contract.parse_rendered_support(tampered)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("weight_name", tuple(_V10_EXPECTED_REWARD_WEIGHTS))
    async def test_v10_visible_binding_rejects_any_reward_weight_numeric_drift(
        self,
        weight_name,
    ):
        env = self._env()
        observation, info = await env.reset(dict(_V10_TASK_METADATA), session_id="sid")
        support = candidate_contract.parse_rendered_support(observation.rsplit("\n", 1)[0])
        payload = json.loads(candidate_contract.canonical_json(info["server_observation_binding"]["binding_payload"]))
        payload["reward_contract"]["reward_weights"][weight_name] += 0.125

        with pytest.raises(
            candidate_contract.CandidateContractError,
            match=rf"reward weight {weight_name} is wrong",
        ):
            candidate_contract.validate_visible_binding_payload(
                payload,
                support=support,
            )

    @pytest.mark.asyncio
    async def test_supported_call_has_pre_step_attestation_and_complete_reward_arithmetic(self):
        env = self._env()
        observation, reset_info = await env.reset(dict(_V10_TASK_METADATA), session_id="sid")
        support = candidate_contract.parse_rendered_support(observation)
        action = support.actions[0]
        next_observation, reward, terminated, truncated, info = await env.step(
            _tool_response(action["name"], action["arguments"]), {}, session_id="sid"
        )
        assert not terminated and not truncated
        assert info["protocol_rejection"] is False
        assert info["submitted_candidate_supported"] is True
        assert info["submitted_candidate_support_sha256"] == reset_info["candidate_support_sha256"]
        assert info["submitted_action"] == action
        assert info["action"] == action
        assert info["scalar_reward"] == pytest.approx(reward)
        assert info["reward_terms"]["total"] == pytest.approx(reward)
        assert sum(value for key, value in info["reward_terms"].items() if key != "total") == pytest.approx(reward)
        assert all(math.isfinite(value) for value in info["reward_measurements"].values())
        assert info["reward_profile"] == "openair_t2_v3"
        assert info["reward_version"] == "openair_t2_v3"
        assert set(info["service_accounting"]) == {
            "requested_service_mbps",
            "admitted_service_mbps",
            "delivered_service_mbps",
            "forced_terminated_service_mbps",
            "cumulative_forced_terminated_service_mbps",
            "forced_termination_events",
            "step_forced_terminated_service_mbps",
            "step_forced_termination_events",
            "unadmitted_service_mbps",
            "undelivered_admitted_service_mbps",
        }
        assert all(value >= 0.0 and math.isfinite(value) for value in info["service_accounting"].values())
        ledger = info["service_accounting"]
        assert ledger["requested_service_mbps"] >= ledger["admitted_service_mbps"]
        assert ledger["admitted_service_mbps"] >= ledger["delivered_service_mbps"]
        assert ledger["unadmitted_service_mbps"] == pytest.approx(
            ledger["requested_service_mbps"] - ledger["admitted_service_mbps"]
        )
        assert ledger["undelivered_admitted_service_mbps"] == pytest.approx(
            ledger["admitted_service_mbps"] - ledger["delivered_service_mbps"]
        )
        assert ledger["step_forced_terminated_service_mbps"] == pytest.approx(
            ledger["cumulative_forced_terminated_service_mbps"]
        )
        assert ledger["step_forced_termination_events"] == pytest.approx(ledger["forced_termination_events"])
        assert env.session_state["sid"]["service_accounting"] == ledger
        for key, value in info["service_accounting"].items():
            assert info["reward_measurements"][key] == pytest.approx(value)
        assert info["service_accounting_sha256"] == candidate_contract.canonical_json_sha256(
            info["service_accounting"]
        )
        assert info["training_eligible"] is True
        assert info["rollout_usable"] is True
        assert info["training_usable"] is True
        receipt = info["server_step_receipt"]
        assert set(receipt) == {
            "action",
            "submitted_action",
            "scalar_reward",
            "reward_measurements",
            "reward_measurements_sha256",
            "reward_terms",
            "reward_terms_sha256",
            "guardrail_accepted",
            "protocol_rejection",
            "submitted_candidate_contract",
            "submitted_candidate_support_sha256",
            "submitted_candidate_supported",
            "rejection_reason",
            "error",
            "kpi_source",
            "dynamics_mode",
            "reward_profile",
            "reward_version",
            "reward_weights",
            "reward_coefficients",
            "service_accounting",
            "service_accounting_sha256",
            "terminated",
            "truncated",
            "training_usable",
            "runtime_manifest_sha256",
            "launch_contract",
            "task_budget_receipt_sha256",
            "reset_receipt_sha256",
            "transition_binding_sha256",
        }
        assert receipt["action"] == action == receipt["submitted_action"]
        assert receipt["scalar_reward"] == pytest.approx(reward)
        assert receipt["reward_measurements"] == info["reward_measurements"]
        assert receipt["reward_terms"] == info["reward_terms"]
        assert receipt["reward_profile"] == info["reward_profile"] == "openair_t2_v3"
        assert receipt["reward_version"] == info["reward_version"] == "openair_t2_v3"
        assert receipt["reward_coefficients"] == info["reward_coefficients"]
        assert receipt["service_accounting"] == info["service_accounting"]
        assert receipt["service_accounting_sha256"] == candidate_contract.canonical_json_sha256(
            receipt["service_accounting"]
        )
        assert receipt["reward_measurements_sha256"] == candidate_contract.canonical_json_sha256(
            receipt["reward_measurements"]
        )
        assert receipt["reward_terms_sha256"] == candidate_contract.canonical_json_sha256(receipt["reward_terms"])
        assert receipt["reward_weights"] == info["reward_weights"]
        assert receipt["dynamics_mode"] == info["dynamics_mode"]
        assert receipt["submitted_candidate_support_sha256"] == reset_info["candidate_support_sha256"]
        assert receipt["training_usable"] is True
        assert receipt["runtime_manifest_sha256"] == reset_info["server_runtime_manifest_sha256"]
        assert receipt["launch_contract"] == reset_info["v10_launch_contract"]
        assert receipt["task_budget_receipt_sha256"] == reset_info["task_budget_receipt_sha256"]
        assert receipt["reset_receipt_sha256"] == reset_info["reset_receipt_sha256"]
        assert info["server_step_receipt_sha256"] == candidate_contract.canonical_json_sha256(receipt)
        assert next_observation is not None
        assert (
            candidate_contract.parse_rendered_support(next_observation).support_sha256
            == info["candidate_support_sha256"]
        )
        assert info["server_observation_binding"]["observation_sha256"] == (
            candidate_contract.text_sha256(next_observation)
        )
        transition = info["server_transition_binding"]
        assert set(transition) == {
            "schema_version",
            "action",
            "runtime_manifest_sha256",
            "launch_contract",
            "task_budget_receipt_sha256",
            "reset_receipt_sha256",
            "pre_observation_sha256",
            "pre_candidate_support_sha256",
            "pre_binding_payload_sha256",
            "pre_cell_count",
            "pre_capacity_milli_mbps_total",
            "post_observation",
            "post_observation_sha256",
            "post_candidate_support_sha256",
            "post_cell_count",
            "post_capacity_milli_mbps_total",
            "post_binding_payload",
        }
        assert transition["schema_version"] == "openair_runb2_v10_transition_binding_v2"
        assert transition["action"] == action
        assert transition["runtime_manifest_sha256"] == reset_info["server_runtime_manifest_sha256"]
        assert transition["launch_contract"] == reset_info["v10_launch_contract"]
        assert transition["task_budget_receipt_sha256"] == reset_info["task_budget_receipt_sha256"]
        assert transition["reset_receipt_sha256"] == reset_info["reset_receipt_sha256"]
        assert info["server_transition_binding_sha256"] == candidate_contract.canonical_json_sha256(transition)
        assert receipt["transition_binding_sha256"] == info["server_transition_binding_sha256"]
        assert transition["pre_observation_sha256"] == candidate_contract.text_sha256(observation)
        assert transition["pre_candidate_support_sha256"] == reset_info["candidate_support_sha256"]
        assert transition["post_observation"] == next_observation
        assert transition["post_observation_sha256"] == candidate_contract.text_sha256(next_observation)
        assert transition["post_binding_payload"] == info["server_observation_binding"]["binding_payload"]
        assert transition["post_binding_payload"][
            "observation_without_binding_sha256"
        ] == candidate_contract.text_sha256(next_observation.rsplit("\n", 1)[0])
        assert transition["pre_cell_count"] == transition["post_cell_count"]
        assert (
            transition["pre_capacity_milli_mbps_total"]
            == transition["post_capacity_milli_mbps_total"]
            == 250_000 * transition["post_cell_count"]
        )
        assert info["reward_measurements"]["cell_capacity_mbps_total"] == pytest.approx(
            250.0 * transition["post_cell_count"]
        )

    @pytest.mark.asyncio
    async def test_v10_server_local_formula_rejects_coupled_shared_reward_corruption(
        self,
        monkeypatch,
    ):
        """A corrupted producer and shared helper cannot redefine the objective together."""

        env = self._env()
        observation, _ = await env.reset(dict(_V10_TASK_METADATA), session_id="sid")
        support = candidate_contract.parse_rendered_support(observation)
        original_step = env.backend.step
        coupled_breakdown: dict[str, object] = {}

        def corrupted_step(*args, **kwargs):
            next_obs, reward, done, step_info = original_step(*args, **kwargs)
            step_info = dict(step_info)
            terms = dict(step_info["reward_terms"])
            terms["service_denial"] += 0.125
            terms["total"] += 0.125
            measurements = dict(step_info["reward_measurements"])
            step_info["reward_terms"] = terms
            coupled_breakdown.update(
                {
                    "measurements": measurements,
                    "terms": terms,
                    "total": reward + 0.125,
                }
            )
            return next_obs, reward + 0.125, done, step_info

        def coupled_shared_compute_breakdown(*_args, **_kwargs):
            return coupled_breakdown

        monkeypatch.setattr(env.backend, "step", corrupted_step)
        monkeypatch.setattr(
            app_source,
            "compute_breakdown",
            coupled_shared_compute_breakdown,
        )
        next_observation, reward, terminated, truncated, info = await env.step(
            _tool_response(support.actions[0]["name"], support.actions[0]["arguments"]),
            {},
            session_id="sid",
        )
        assert next_observation is None
        assert reward == 0.0 and terminated is False and truncated is True
        assert info["protocol_rejection"] is True
        assert info["rejection_reason"].startswith("server_contract_failure:")
        assert info["environment_transition_discarded"] is True
        assert info["training_usable"] is False
        assert "sid" not in env.session_state

    @pytest.mark.asyncio
    async def test_v10_server_local_ledger_rejects_coupled_service_arithmetic_corruption(
        self,
        monkeypatch,
    ):
        env = self._env()
        observation, _ = await env.reset(dict(_V10_TASK_METADATA), session_id="sid")
        support = candidate_contract.parse_rendered_support(observation)
        original_step = env.backend.step
        coupled_breakdown: dict[str, object] = {}

        def corrupted_step(*args, **kwargs):
            next_obs, _reward, done, step_info = original_step(*args, **kwargs)
            step_info = dict(step_info)
            ledger = dict(step_info["service_accounting"])
            ledger["requested_service_mbps"] += 10.0
            ledger["unadmitted_service_mbps"] += 10.0
            measurements = dict(step_info["reward_measurements"])
            measurements.update(ledger)
            terms = dict(step_info["reward_terms"])
            terms["service_denial"] = -(ledger["unadmitted_service_mbps"] / ledger["requested_service_mbps"])
            terms["delivery_gap"] = -1.25 * (
                ledger["undelivered_admitted_service_mbps"] / ledger["requested_service_mbps"]
            )
            terms["total"] = sum(value for key, value in terms.items() if key != "total")
            step_info["service_accounting"] = ledger
            step_info["reward_measurements"] = measurements
            step_info["reward_terms"] = terms
            coupled_breakdown.update(
                {
                    "measurements": measurements,
                    "terms": terms,
                    "total": terms["total"],
                }
            )
            return next_obs, terms["total"], done, step_info

        monkeypatch.setattr(env.backend, "step", corrupted_step)
        monkeypatch.setattr(
            app_source,
            "compute_breakdown",
            lambda *_args, **_kwargs: coupled_breakdown,
        )
        next_observation, reward, terminated, truncated, info = await env.step(
            _tool_response(support.actions[0]["name"], support.actions[0]["arguments"]),
            {},
            session_id="sid",
        )
        assert next_observation is None
        assert reward == 0.0 and terminated is False and truncated is True
        assert info["protocol_rejection"] is True
        assert info["rejection_reason"].startswith("server_contract_failure:")
        assert info["environment_transition_discarded"] is True
        assert info["training_usable"] is False
        assert "sid" not in env.session_state

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_reward_version", [None, "openair_v1", "openair_t2_v2"])
    async def test_v10_quarantines_missing_or_wrong_step_reward_version(
        self,
        monkeypatch,
        bad_reward_version,
    ):
        env = self._env()
        observation, _ = await env.reset(dict(_V10_TASK_METADATA), session_id="sid")
        support = candidate_contract.parse_rendered_support(observation)
        original_step = env.backend.step

        def corrupted_step(*args, **kwargs):
            next_obs, reward, done, step_info = original_step(*args, **kwargs)
            step_info = dict(step_info)
            if bad_reward_version is None:
                step_info.pop("reward_version", None)
            else:
                step_info["reward_version"] = bad_reward_version
            return next_obs, reward, done, step_info

        monkeypatch.setattr(env.backend, "step", corrupted_step)
        next_observation, reward, terminated, truncated, info = await env.step(
            _tool_response(support.actions[0]["name"], support.actions[0]["arguments"]),
            {},
            session_id="sid",
        )
        assert next_observation is None
        assert reward == 0.0 and terminated is False and truncated is True
        assert info["protocol_rejection"] is True
        assert info["rejection_reason"].startswith("server_contract_failure:")
        assert info["environment_transition_discarded"] is True
        assert info["training_usable"] is False
        assert "sid" not in env.session_state

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "corrupt_service_accounting",
        [
            lambda _mapping: None,
            lambda mapping: {**mapping, "unexpected": 0.0},
            lambda mapping: {key: value for key, value in mapping.items() if key != "requested_service_mbps"},
            lambda mapping: {**mapping, "requested_service_mbps": float("nan")},
            lambda mapping: {**mapping, "requested_service_mbps": -1.0},
        ],
        ids=("missing", "extra-key", "missing-key", "nonfinite", "negative"),
    )
    async def test_v10_quarantines_invalid_exact_service_accounting(
        self,
        monkeypatch,
        corrupt_service_accounting,
    ):
        env = self._env()
        observation, _ = await env.reset(dict(_V10_TASK_METADATA), session_id="sid")
        support = candidate_contract.parse_rendered_support(observation)
        original_step = env.backend.step

        def corrupted_step(*args, **kwargs):
            next_obs, reward, done, step_info = original_step(*args, **kwargs)
            step_info = dict(step_info)
            step_info["service_accounting"] = corrupt_service_accounting(dict(step_info["service_accounting"]))
            return next_obs, reward, done, step_info

        monkeypatch.setattr(env.backend, "step", corrupted_step)
        next_observation, reward, terminated, truncated, info = await env.step(
            _tool_response(support.actions[0]["name"], support.actions[0]["arguments"]),
            {},
            session_id="sid",
        )
        assert next_observation is None
        assert reward == 0.0 and terminated is False and truncated is True
        assert info["protocol_rejection"] is True
        assert info["rejection_reason"].startswith("server_contract_failure:")
        assert info["environment_transition_discarded"] is True
        assert info["training_usable"] is False
        assert "sid" not in env.session_state

    @pytest.mark.parametrize(
        "mutated",
        [
            {
                **_ZERO_FORCED_V10_SERVICE_ACCOUNTING,
                "forced_terminated_service_mbps": 1.0,
            },
            {
                **_ZERO_FORCED_V10_SERVICE_ACCOUNTING,
                "cumulative_forced_terminated_service_mbps": 1.0,
            },
            {
                **_ZERO_FORCED_V10_SERVICE_ACCOUNTING,
                "forced_termination_events": 1.0,
            },
            {
                **_ZERO_FORCED_V10_SERVICE_ACCOUNTING,
                "step_forced_terminated_service_mbps": 1.0,
            },
            {
                **_ZERO_FORCED_V10_SERVICE_ACCOUNTING,
                "step_forced_termination_events": 1.0,
            },
            {
                **_ZERO_FORCED_V10_SERVICE_ACCOUNTING,
                "forced_terminated_service_mbps": 1.0,
                "cumulative_forced_terminated_service_mbps": 1.0,
                "forced_termination_events": 1.0,
                "step_forced_terminated_service_mbps": 1.0,
                "step_forced_termination_events": 1.0,
            },
        ],
        ids=(
            "forced-service",
            "cumulative-forced-service",
            "forced-events",
            "step-forced-service",
            "step-forced-events",
            "coupled-valid-looking-forced-ledger",
        ),
    )
    def test_v10_prb_only_scope_rejects_every_nonzero_forced_service_field(
        self,
        mutated,
    ):
        with pytest.raises(
            V10ProtocolError,
            match="PRB-only V10 service_accounting forced fields must be exactly zero",
        ):
            OpenAirCongestionEnv._validated_v10_service_accounting(
                dict(mutated),
                previous=None,
            )

    @pytest.mark.asyncio
    async def test_v10_two_step_quarantines_coupled_corruption_against_carried_ledger(
        self,
        monkeypatch,
    ):
        env = self._env()
        observation, _ = await env.reset(dict(_V10_TASK_METADATA), session_id="sid")
        support = candidate_contract.parse_rendered_support(observation)
        first_observation, _, terminated, truncated, _ = await env.step(
            _tool_response(support.actions[0]["name"], support.actions[0]["arguments"]),
            {},
            session_id="sid",
        )
        assert first_observation is not None
        assert not terminated and not truncated
        first_ledger = dict(env.session_state["sid"]["service_accounting"])
        assert all(
            first_ledger[key] == 0.0
            for key in (
                "forced_terminated_service_mbps",
                "cumulative_forced_terminated_service_mbps",
                "forced_termination_events",
                "step_forced_terminated_service_mbps",
                "step_forced_termination_events",
            )
        )

        # Model a corrupted carried ledger plus a producer/shared-helper pair
        # that advances every coupled counter consistently.  A keys-only or
        # monotonicity-only check would accept this fabricated forced event.
        env.session_state["sid"]["service_accounting"].update(
            {
                "forced_terminated_service_mbps": 1.0,
                "cumulative_forced_terminated_service_mbps": 1.0,
                "forced_termination_events": 1.0,
                "step_forced_terminated_service_mbps": 1.0,
                "step_forced_termination_events": 1.0,
            }
        )
        second_support = candidate_contract.parse_rendered_support(first_observation)
        original_step = env.backend.step
        coupled_breakdown: dict[str, object] = {}

        def corrupted_second_step(*args, **kwargs):
            next_obs, _reward, done, step_info = original_step(*args, **kwargs)
            step_info = dict(step_info)
            ledger = dict(step_info["service_accounting"])
            ledger.update(
                {
                    "forced_terminated_service_mbps": 1.0,
                    "cumulative_forced_terminated_service_mbps": 2.0,
                    "forced_termination_events": 2.0,
                    "step_forced_terminated_service_mbps": 1.0,
                    "step_forced_termination_events": 1.0,
                }
            )
            measurements = dict(step_info["reward_measurements"])
            measurements.update(ledger)
            terms = dict(step_info["reward_terms"])
            requested = measurements["requested_service_mbps"]
            forced_ratio = min(1.0, ledger["step_forced_terminated_service_mbps"] / requested)
            terms["forced_termination"] = -(5.0 + 2.0 * forced_ratio)
            terms["total"] = sum(value for key, value in terms.items() if key != "total")
            step_info["service_accounting"] = ledger
            step_info["reward_measurements"] = measurements
            step_info["reward_terms"] = terms
            coupled_breakdown.update(
                {
                    "measurements": measurements,
                    "terms": terms,
                    "total": terms["total"],
                }
            )
            return next_obs, terms["total"], done, step_info

        monkeypatch.setattr(env.backend, "step", corrupted_second_step)
        monkeypatch.setattr(
            app_source,
            "compute_breakdown",
            lambda *_args, **_kwargs: coupled_breakdown,
        )
        next_observation, reward, terminated, truncated, info = await env.step(
            _tool_response(
                second_support.actions[0]["name"],
                second_support.actions[0]["arguments"],
            ),
            {},
            session_id="sid",
        )
        assert next_observation is None
        assert reward == 0.0 and terminated is False and truncated is True
        assert info["protocol_rejection"] is True
        assert info["rejection_reason"].startswith("server_contract_failure:")
        assert info["environment_transition_discarded"] is True
        assert info["training_usable"] is False
        assert "sid" not in env.session_state

    @pytest.mark.asyncio
    async def test_v10_two_step_quarantines_corrupted_carried_session_ledger(
        self,
    ):
        env = self._env()
        observation, _ = await env.reset(dict(_V10_TASK_METADATA), session_id="sid")
        support = candidate_contract.parse_rendered_support(observation)
        first_observation, _, terminated, truncated, _ = await env.step(
            _tool_response(support.actions[0]["name"], support.actions[0]["arguments"]),
            {},
            session_id="sid",
        )
        assert first_observation is not None
        assert not terminated and not truncated

        # Corrupt only the ledger carried by the server between transitions.
        # The second backend result remains honest, so quarantine proves the
        # previous session ledger is actually revalidated at integration time.
        env.session_state["sid"]["service_accounting"]["requested_service_mbps"] += 1.0
        second_support = candidate_contract.parse_rendered_support(first_observation)
        next_observation, reward, terminated, truncated, info = await env.step(
            _tool_response(
                second_support.actions[0]["name"],
                second_support.actions[0]["arguments"],
            ),
            {},
            session_id="sid",
        )
        assert next_observation is None
        assert reward == 0.0 and terminated is False and truncated is True
        assert info["protocol_rejection"] is True
        assert info["rejection_reason"].startswith("server_contract_failure:")
        assert info["environment_transition_discarded"] is True
        assert info["training_usable"] is False
        assert "sid" not in env.session_state

    def test_v10_accepts_a_conserving_monotonic_service_ledger(self):
        assert (
            OpenAirCongestionEnv._validated_v10_service_accounting(
                dict(_VALID_V10_SERVICE_ACCOUNTING),
                previous=dict(_PREVIOUS_V10_SERVICE_ACCOUNTING),
            )
            == _VALID_V10_SERVICE_ACCOUNTING
        )

    @pytest.mark.parametrize(
        ("mutated", "previous"),
        [
            (
                {**_VALID_V10_SERVICE_ACCOUNTING, "admitted_service_mbps": 11.0},
                _PREVIOUS_V10_SERVICE_ACCOUNTING,
            ),
            (
                {**_VALID_V10_SERVICE_ACCOUNTING, "delivered_service_mbps": 9.0},
                _PREVIOUS_V10_SERVICE_ACCOUNTING,
            ),
            (
                {**_VALID_V10_SERVICE_ACCOUNTING, "unadmitted_service_mbps": 1.0},
                _PREVIOUS_V10_SERVICE_ACCOUNTING,
            ),
            (
                {
                    **_VALID_V10_SERVICE_ACCOUNTING,
                    "undelivered_admitted_service_mbps": 1.0,
                },
                _PREVIOUS_V10_SERVICE_ACCOUNTING,
            ),
            (
                {
                    **_VALID_V10_SERVICE_ACCOUNTING,
                    "forced_terminated_service_mbps": 6.0,
                },
                _PREVIOUS_V10_SERVICE_ACCOUNTING,
            ),
            (
                {
                    **_VALID_V10_SERVICE_ACCOUNTING,
                    "step_forced_terminated_service_mbps": 6.0,
                },
                _PREVIOUS_V10_SERVICE_ACCOUNTING,
            ),
            (
                {**_VALID_V10_SERVICE_ACCOUNTING, "forced_termination_events": 3.5},
                _PREVIOUS_V10_SERVICE_ACCOUNTING,
            ),
            (
                {
                    **_VALID_V10_SERVICE_ACCOUNTING,
                    "step_forced_termination_events": 1.5,
                },
                _PREVIOUS_V10_SERVICE_ACCOUNTING,
            ),
            (
                {
                    **_VALID_V10_SERVICE_ACCOUNTING,
                    "cumulative_forced_terminated_service_mbps": 2.0,
                    "step_forced_terminated_service_mbps": 0.0,
                },
                _PREVIOUS_V10_SERVICE_ACCOUNTING,
            ),
            (
                {
                    **_VALID_V10_SERVICE_ACCOUNTING,
                    "forced_termination_events": 1.0,
                    "step_forced_termination_events": 0.0,
                },
                _PREVIOUS_V10_SERVICE_ACCOUNTING,
            ),
            (
                {
                    **_VALID_V10_SERVICE_ACCOUNTING,
                    "step_forced_terminated_service_mbps": 1.0,
                },
                _PREVIOUS_V10_SERVICE_ACCOUNTING,
            ),
        ],
        ids=(
            "admitted-exceeds-requested",
            "delivered-exceeds-admitted",
            "unadmitted-does-not-conserve",
            "undelivered-does-not-conserve",
            "forced-exceeds-cumulative",
            "step-forced-exceeds-cumulative",
            "fractional-cumulative-event-count",
            "fractional-step-event-count",
            "cumulative-service-decreases",
            "cumulative-event-count-decreases",
            "step-service-does-not-match-delta",
        ),
    )
    def test_v10_rejects_nonconserving_or_nonmonotonic_service_ledgers(
        self,
        mutated,
        previous,
    ):
        with pytest.raises(V10ProtocolError, match="service_accounting"):
            OpenAirCongestionEnv._validated_v10_service_accounting(
                dict(mutated),
                previous=None if previous is None else dict(previous),
            )

    @pytest.mark.asyncio
    async def test_v10_quarantines_service_accounting_that_disagrees_with_reward_measurements(
        self,
        monkeypatch,
    ):
        env = self._env()
        observation, _ = await env.reset(dict(_V10_TASK_METADATA), session_id="sid")
        support = candidate_contract.parse_rendered_support(observation)
        original_step = env.backend.step

        def corrupted_step(*args, **kwargs):
            next_obs, reward, done, step_info = original_step(*args, **kwargs)
            step_info = dict(step_info)
            service_accounting = dict(step_info["service_accounting"])
            service_accounting["requested_service_mbps"] += 1.0
            step_info["service_accounting"] = service_accounting
            return next_obs, reward, done, step_info

        monkeypatch.setattr(env.backend, "step", corrupted_step)
        next_observation, reward, terminated, truncated, info = await env.step(
            _tool_response(support.actions[0]["name"], support.actions[0]["arguments"]),
            {},
            session_id="sid",
        )
        assert next_observation is None
        assert reward == 0.0 and terminated is False and truncated is True
        assert info["protocol_rejection"] is True
        assert info["rejection_reason"].startswith("server_contract_failure:")
        assert info["environment_transition_discarded"] is True
        assert info["training_usable"] is False
        assert "sid" not in env.session_state

    @pytest.mark.asyncio
    async def test_v10_terminal_quarantines_every_invalid_call_shape(self):
        env = self._env()
        cases = (
            (
                "no_function_call",
                _text_response("I will decide later."),
                "exactly_one_function_call_required",
            ),
            (
                "multiple_function_calls",
                _multi_tool_response(("noop", {}), ("noop", {})),
                "multiple_function_calls_forbidden",
            ),
            (
                "malformed_json",
                _raw_tool_response("noop", "{not-json"),
                "invalid_candidate:",
            ),
            (
                "duplicate_json_key",
                _raw_tool_response("noop", '{"x":1,"x":2}'),
                "invalid_candidate:",
            ),
            (
                "nonfinite_json_constant",
                _raw_tool_response("noop", '{"x":NaN}'),
                "invalid_candidate:",
            ),
            (
                "out_of_support",
                _tool_response(
                    "set_prb_cap",
                    {"cell_id": 0, "target": "ue", "target_id": 999, "max_prb": 200},
                ),
                "candidate_not_in_pre_step_support",
            ),
        )
        for session_id, response, reason in cases:
            await env.reset(dict(_V10_TASK_METADATA), session_id=session_id)
            observation, reward, terminated, truncated, info = await env.step(
                response,
                {},
                session_id=session_id,
            )
            assert observation is None
            assert reward == 0.0 and terminated is False and truncated is True
            assert info["protocol_rejection"] is True
            assert info["terminal_quarantine"] is True
            assert info["training_eligible"] is False
            assert info["rollout_usable"] is False
            assert info["training_usable"] is False
            assert info["reward_measurements"] is None
            assert info["reward_terms"] is None
            assert info["rejection_reason"].startswith(reason)
            assert session_id not in env.session_state

    @pytest.mark.asyncio
    async def test_v10_supported_prb_cap_changes_synthetic_kpis_relative_to_noop(self):
        # This is the causal claim V10 is allowed to make: on the same replay
        # state, an accepted *visible* PRB cap changes measured synthetic KPI
        # values.  Comparing the opaque rendered text alone would be weaker,
        # because its L-row also echoes the chosen action.
        acting = self._env()
        baseline = self._env()
        _, acting_info = await acting.reset(
            dict(_V10_ACTION_TASK_METADATA),
            session_id="acting",
        )
        await baseline.reset(
            dict(_V10_ACTION_TASK_METADATA),
            session_id="baseline",
        )
        action = next(
            item
            for item in acting_info["candidate_actions"]
            if item["name"] == "set_prb_cap" and item["arguments"]["max_prb"] < 273
        )
        _, _, _, _, acted = await acting.step(
            _tool_response(action["name"], action["arguments"]), {}, session_id="acting"
        )
        _, _, _, _, nooped = await baseline.step(_tool_response("noop", {}), {}, session_id="baseline")
        assert acted["guardrail_accepted"] is True
        assert acted["prb_cap_dynamics"]
        assert acted["reward_measurements"]["aggregate_delivered_mbps"] != pytest.approx(
            nooped["reward_measurements"]["aggregate_delivered_mbps"]
        )

    @pytest.mark.asyncio
    async def test_v10_reset_metadata_is_a_deep_copy_not_authoritative_state(self):
        env = self._env()
        _, info = await env.reset(
            dict(_V10_ACTION_TASK_METADATA),
            session_id="sid",
        )
        index = next(i for i, action in enumerate(info["candidate_actions"]) if action["name"] == "set_prb_cap")
        original = dict(env.session_state["sid"]["candidate_support"].actions[index]["arguments"])
        info["candidate_actions"][index]["arguments"]["max_prb"] = 1
        assert env.session_state["sid"]["candidate_support"].actions[index]["arguments"] == original

    @pytest.mark.asyncio
    async def test_v10_rejects_ambiguous_or_incomplete_task_row_before_opening(self):
        env = self._env()
        with pytest.raises(RuntimeError, match="exact deterministic schema"):
            await env.reset(dict(_V10_TASK_METADATA, unrelated="not-consumed"), session_id="extra")
        with pytest.raises(RuntimeError, match="exact deterministic schema"):
            incomplete = dict(_V10_TASK_METADATA)
            del incomplete["regime_mix"]
            await env.reset(incomplete, session_id="missing")
        assert not env.session_state

    @pytest.mark.asyncio
    async def test_v10_runtime_manifest_drift_fails_closed(self, monkeypatch):
        env = self._env()
        await env.reset(dict(_V10_TASK_METADATA), session_id="sid")
        original_payload = env._v10_runtime_manifest_payload

        def drifted_payload():
            payload = original_payload()
            payload["dependency_versions"] = dict(payload["dependency_versions"])
            payload["dependency_versions"]["python"] = "injected-drift"
            return payload

        monkeypatch.setattr(env, "_v10_runtime_manifest_payload", drifted_payload)
        with pytest.raises(RuntimeError, match="runtime source/config drifted"):
            env._v10_static_info(env.session_state["sid"]["candidate_support"])

    @pytest.mark.asyncio
    async def test_v10_runtime_manifest_drift_during_step_releases_slot(self, monkeypatch):
        env = self._env()
        observation, info = await env.reset(dict(_V10_TASK_METADATA), session_id="sid")
        support = candidate_contract.parse_rendered_support(observation)
        closed: list[str] = []
        original_close = env.backend.close

        def tracked_close(episode_id):
            closed.append(episode_id)
            return original_close(episode_id)

        original_payload = env._v10_runtime_manifest_payload

        def drifted_payload():
            payload = original_payload()
            payload["dependency_versions"] = dict(payload["dependency_versions"])
            payload["dependency_versions"]["python"] = "injected-drift"
            return payload

        monkeypatch.setattr(env.backend, "close", tracked_close)
        monkeypatch.setattr(env, "_v10_runtime_manifest_payload", drifted_payload)
        with pytest.raises(RuntimeError, match="runtime source/config drifted"):
            await env.step(
                _tool_response(support.actions[0]["name"], support.actions[0]["arguments"]),
                {},
                session_id="sid",
            )

        assert "sid" not in env.session_state
        assert closed == [info["episode_id"]]

    @pytest.mark.asyncio
    async def test_v10_requires_t2_and_replay_at_configuration_boundary(self, monkeypatch):
        env = self._env()
        with pytest.raises(RuntimeError, match="tier='T2'"):
            await env.reset(dict(_TASK_METADATA), session_id="sid")

        monkeypatch.setattr(
            "resources_servers.openair_congestion.app.select_backend",
            lambda _config: object(),
        )
        with pytest.raises(RuntimeError, match="requires backend='replay'"):
            _make_env(
                protocol_mode=candidate_contract.RUNB2_V10_PROTOCOL_MODE,
                replay_scenario_source=_V10_SCENARIO_SOURCE,
                cell_capacity_mbps=250.0,
                v10_session_secret=_V10_TEST_SESSION_SECRET,
                v10_system_prompt_sha256=_V10_TEST_SYSTEM_PROMPT_SHA256,
                v10_task_manifest_sha256=_V10_TEST_TASK_MANIFEST_SHA256,
            )

    @pytest.mark.parametrize(
        ("overrides", "error"),
        (
            ({"v10_session_secret": None}, "requires an injected high-entropy"),
            ({"v10_session_secret": "too-short"}, "requires an injected high-entropy"),
            ({"v10_system_prompt_sha256": None}, "requires a pinned lowercase SHA-256"),
            ({"v10_system_prompt_sha256": "A" * 64}, "requires a pinned lowercase SHA-256"),
            ({"v10_task_manifest_sha256": None}, "requires a pinned lowercase SHA-256"),
            ({"v10_task_manifest_sha256": "B" * 64}, "requires a pinned lowercase SHA-256"),
            ({"backend": "dataset_replay"}, "requires config backend='replay'"),
            ({"num_workers": 2}, "requires exactly one FastAPI worker"),
            ({"host": "0.0.0.0"}, "requires an in-process test host or a loopback host"),
            ({"entrypoint": "serve.py"}, "requires the bound app.py entrypoint"),
            ({"replay_scenario_source": "auto"}, "'auto' is forbidden"),
            ({"cell_capacity_mbps": 60.0}, "fixes cell_capacity_mbps at 250.0"),
            ({"candidate_cell_capacity_mbps": 250.0}, "was removed for V10"),
            ({"v10_max_steps": 17}, "v10_max_steps must be a positive integer"),
            ({"v10_max_steps": 5, "agent_max_steps": 4}, "v10_max_steps must be"),
        ),
    )
    def test_v10_refuses_unsafe_launch_configuration(self, overrides, error):
        config = {
            "protocol_mode": candidate_contract.RUNB2_V10_PROTOCOL_MODE,
            "replay_scenario_source": _V10_SCENARIO_SOURCE,
            "cell_capacity_mbps": 250.0,
            "v10_session_secret": _V10_TEST_SESSION_SECRET,
            "v10_system_prompt_sha256": _V10_TEST_SYSTEM_PROMPT_SHA256,
            "v10_task_manifest_sha256": _V10_TEST_TASK_MANIFEST_SHA256,
            **overrides,
        }
        with pytest.raises(RuntimeError, match=error):
            _make_env(**config)

    def test_v10_refuses_nonreplay_environment_backend_override(self, monkeypatch):
        monkeypatch.setenv("OPENAIR_CONGESTION_BACKEND", "dataset_replay")
        with pytest.raises(RuntimeError, match="OPENAIR_CONGESTION_BACKEND"):
            self._env()

    def test_v10_attests_exact_generator_as_configured_and_used(self):
        env = self._env()
        config = env._v10_runtime_manifest["effective_public_config"]
        assert config["dynamic_congestion_gen_importable"] is True
        assert config["dynamic_congestion_gen_configured"] is True
        assert config["dynamic_congestion_gen_used"] is True
        assert config["scenario_source"] == _V10_SCENARIO_SOURCE

    def test_v10_refuses_reward_coefficient_drift_at_startup(self, monkeypatch):
        monkeypatch.setattr(
            "resources_servers.openair_congestion.app.T2_V3_ACTION_WEIGHT",
            0.123,
        )
        with pytest.raises(RuntimeError, match="reviewed reward coefficients"):
            self._env()

    @pytest.mark.asyncio
    async def test_v10_requires_explicit_bounded_task_max_steps(self):
        env = self._env()
        missing_budget = dict(_V10_TASK_METADATA)
        del missing_budget["max_steps"]
        with pytest.raises(RuntimeError, match="exact deterministic schema"):
            await env.reset(missing_budget, session_id="missing")
        assert "missing" not in env.session_state

        with pytest.raises(RuntimeError, match="exceeds the configured V10 launch bound"):
            await env.reset(
                dict(_V10_TASK_METADATA, max_steps=17),
                session_id="oversized",
            )
        assert "oversized" not in env.session_state

    @pytest.mark.asyncio
    async def test_v10_terminal_budget_eagerly_releases_the_session(self):
        env = self._env()
        observation, _ = await env.reset(
            dict(_V10_TASK_METADATA, max_steps=1),
            session_id="sid",
        )
        support = candidate_contract.parse_rendered_support(observation)
        _, _, terminated, truncated, info = await env.step(
            _tool_response(support.actions[0]["name"], support.actions[0]["arguments"]),
            {},
            session_id="sid",
        )
        assert terminated or truncated
        assert info["training_usable"] is False
        assert "sid" not in env.session_state

    @pytest.mark.asyncio
    async def test_v10_contract_failure_discards_transition_and_releases_slot(self, monkeypatch):
        env = self._env()
        observation, _ = await env.reset(dict(_V10_TASK_METADATA), session_id="sid")
        support = candidate_contract.parse_rendered_support(observation)
        original_step = env.backend.step

        def corrupted_step(*args, **kwargs):
            next_obs, reward, done, step_info = original_step(*args, **kwargs)
            step_info = dict(step_info)
            terms = dict(step_info["reward_terms"])
            terms["total"] += 0.01
            step_info["reward_terms"] = terms
            return next_obs, reward, done, step_info

        monkeypatch.setattr(env.backend, "step", corrupted_step)
        observation, reward, terminated, truncated, info = await env.step(
            _tool_response(support.actions[0]["name"], support.actions[0]["arguments"]),
            {},
            session_id="sid",
        )
        assert observation is None
        assert reward == 0.0 and terminated is False and truncated is True
        assert info["protocol_rejection"] is True
        assert info["environment_transition_discarded"] is True
        assert info["training_usable"] is False
        assert "sid" not in env.session_state

    @pytest.mark.asyncio
    async def test_v10_backend_failure_returns_quarantine_and_releases_slot(self, monkeypatch):
        env = self._env()
        observation, _ = await env.reset(dict(_V10_TASK_METADATA), session_id="sid")
        support = candidate_contract.parse_rendered_support(observation)

        def failed_step(*_args, **_kwargs):
            raise RuntimeError("injected backend failure")

        monkeypatch.setattr(env.backend, "step", failed_step)
        observation, reward, terminated, truncated, info = await env.step(
            _tool_response(support.actions[0]["name"], support.actions[0]["arguments"]),
            {},
            session_id="sid",
        )
        assert observation is None
        assert reward == 0.0 and terminated is False and truncated is True
        assert info["rejection_reason"] == "backend_step_failure"
        assert info["environment_transition_discarded"] is True
        assert "sid" not in env.session_state

    @pytest.mark.asyncio
    async def test_v10_reset_contract_failure_closes_the_opened_episode(self, monkeypatch):
        env = self._env()
        closed: list[str] = []
        original_close = env.backend.close

        def tracked_close(episode_id):
            closed.append(episode_id)
            return original_close(episode_id)

        def failed_support(_self, _observation):
            raise RuntimeError("injected support binding failure")

        monkeypatch.setattr(env.backend, "close", tracked_close)
        monkeypatch.setattr(OpenAirCongestionEnv, "_v10_support_for_observation", failed_support)
        with pytest.raises(RuntimeError, match="injected support binding failure"):
            await env.reset(dict(_V10_TASK_METADATA), session_id="sid")
        assert closed
        assert "sid" not in env.session_state


class TestRoutes:
    def test_gymnasium_routes_registered(self):
        env = _make_env()
        routes = {r.path for r in env.setup_webserver().routes}
        assert {"/reset", "/step", "/close", "/aggregate_metrics"}.issubset(routes)


def _http_client(app) -> httpx.AsyncClient:
    # In-process ASGI transport (as in aviary's tests): real /reset and /step
    # requests through routing, request parsing, and the session middleware,
    # no live socket. Each AsyncClient keeps its own cookie jar, so each
    # client is one session.
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


def _reset_body(**overrides) -> dict:
    # EnvResetRequest: responses_create_params plus task-row extras.
    body = {"responses_create_params": {"input": []}, **_TASK_METADATA}
    body.update(overrides)
    return body


def _step_body(name: str, arguments: dict) -> dict:
    # EnvStepRequest: responses_create_params plus the model's response.
    return {"responses_create_params": {"input": []}, "response": _tool_response(name, arguments).model_dump()}


class TestHTTPSurface:
    @pytest.mark.asyncio
    async def test_interleaved_sessions_step_independently_over_http(self):
        # Two clients (= two session cookies) with interleaved /step calls:
        # each must advance only its own episode, with no state bleed.
        env = _make_env()
        app = env.setup_webserver()
        async with _http_client(app) as client_a, _http_client(app) as client_b:
            episode_a = (await client_a.post("/reset", json=_reset_body())).json()["info"]["episode_id"]
            episode_b = (await client_b.post("/reset", json=_reset_body(seed=7002))).json()["info"]["episode_id"]
            assert episode_a != episode_b
            assert len(env.session_state) == 2
            expected = {id(client_a): episode_a, id(client_b): episode_b}
            steps_taken = {id(client_a): 0, id(client_b): 0}
            for client in (client_a, client_b, client_a, client_b, client_a):
                response = await client.post("/step", json=_step_body("noop", {}))
                assert response.status_code == 200
                info = response.json()["info"]
                steps_taken[id(client)] += 1
                assert info["episode_id"] == expected[id(client)]
                assert info["n_steps"] == steps_taken[id(client)]

    @pytest.mark.asyncio
    async def test_same_task_row_twice_yields_identical_episode_over_http(self):
        # Offline determinism at the HTTP surface: the same task row and
        # action sequence must reproduce the observation and reward sequence
        # exactly (fresh session each run; episode_ids differ, so info is
        # excluded from the comparison).
        env = _make_env()
        app = env.setup_webserver()
        actions = [
            ("set_ul_power_control", {"cell_id": 0, "p0_dbm": -90, "alpha": 0.8}),
            ("noop", {}),
            ("set_prb_cap", {"cell_id": 0, "target": "ue", "target_id": 0, "max_prb": 120}),
        ]

        async def run_episode() -> list:
            async with _http_client(app) as client:
                trace = [(await client.post("/reset", json=_reset_body())).json()["observation"]]
                for name, arguments in actions:
                    body = (await client.post("/step", json=_step_body(name, arguments))).json()
                    trace.append((body["observation"], body["reward"], body["terminated"], body["truncated"]))
                return trace

        assert await run_episode() == await run_episode()

    @pytest.mark.asyncio
    async def test_pool_exhaustion_reaps_orphans_over_http(self):
        # HTTP counterpart of the in-process reaper test: with a live session
        # holding the only slot, a second /reset fails pool-exhausted; once
        # that session dies without close_session() (crashed rollout), the
        # reaper reclaims its slot and the retry succeeds.
        env = _make_env(pool_size=1)
        app = env.setup_webserver()
        async with _http_client(app) as client_dead, _http_client(app) as client_new:
            episode_dead = (await client_dead.post("/reset", json=_reset_body())).json()["info"]["episode_id"]
            # The server registers no exception middleware, so the pool-exhausted
            # RuntimeError tunnels through the in-process ASGI transport; a
            # client on a real socket would see a 500 instead.
            with pytest.raises(RuntimeError, match="pool exhausted"):
                await client_new.post("/reset", json=_reset_body(seed=7002))
            # Simulate the crash: the session vanishes without close_session().
            del env.session_state[next(iter(env.session_state))]
            response = await client_new.post("/reset", json=_reset_body(seed=7002))
            assert response.status_code == 200
            info = response.json()["info"]
            assert info["episode_id"] != episode_dead
            assert env.session_state[next(iter(env.session_state))]["episode_id"] == info["episode_id"]

    @pytest.mark.asyncio
    async def test_explicit_close_is_cookie_scoped_and_idempotent(self):
        env = _make_env(pool_size=1)
        app = env.setup_webserver()
        async with _http_client(app) as client:
            reset = await client.post("/reset", json=_reset_body())
            assert reset.status_code == 200
            first = await client.post("/close", json={})
            assert first.status_code == 200
            assert first.json()["ok"] is True
            assert first.json()["already_closed"] is False
            assert first.json()["summary"]["ok"] is True
            assert not env.session_state
            second = await client.post("/close", json={})
            assert second.status_code == 200
            assert second.json()["already_closed"] is True

    @pytest.mark.asyncio
    async def test_v10_http_cookie_contract_binds_the_pre_step_support(self):
        env = _make_env(
            protocol_mode=candidate_contract.RUNB2_V10_PROTOCOL_MODE,
            replay_scenario_source=_V10_SCENARIO_SOURCE,
            cell_capacity_mbps=250.0,
            v10_session_secret=_V10_TEST_SESSION_SECRET,
            v10_system_prompt_sha256=_V10_TEST_SYSTEM_PROMPT_SHA256,
            v10_task_manifest_sha256=_V10_TEST_TASK_MANIFEST_SHA256,
        )
        app = env.setup_webserver()
        async with _http_client(app) as client:
            reset = await client.post("/reset", json=_reset_body(tier="T2", max_steps=4))
            assert reset.status_code == 200
            reset_body = reset.json()
            support = candidate_contract.parse_rendered_support(reset_body["observation"])
            step = await client.post(
                "/step",
                json=_step_body(support.actions[0]["name"], support.actions[0]["arguments"]),
            )
            assert step.status_code == 200
            step_body = step.json()
            assert step_body["info"]["submitted_candidate_support_sha256"] == support.support_sha256
            assert step_body["info"]["submitted_candidate_supported"] is True
            assert step_body["info"]["protocol_rejection"] is False
            assert step_body["info"]["training_usable"] is True
            set_cookie = reset.headers["set-cookie"]
            assert "openair_v10_session=" in set_cookie
            assert _V10_TEST_SESSION_SECRET not in set_cookie
            assert "httponly" in set_cookie.lower()
            assert "samesite=strict" in set_cookie.lower()


class TestBackends:
    def test_select_backend_defaults_to_replay(self, monkeypatch):
        monkeypatch.delenv("OPENAIR_CONGESTION_BACKEND", raising=False)
        config = OpenAirCongestionResourcesServerConfig(host="", port=0, entrypoint="", name="")
        backend = select_backend(config)
        assert isinstance(backend, ReplayBackend)
        assert not isinstance(backend, V10FixedReplayBackend)

    def test_select_backend_uses_v10_generator_replay_only_when_explicit(self, monkeypatch):
        monkeypatch.delenv("OPENAIR_CONGESTION_BACKEND", raising=False)
        config = OpenAirCongestionResourcesServerConfig(
            host="",
            port=0,
            entrypoint="",
            name="",
            replay_scenario_source=_V10_SCENARIO_SOURCE,
        )
        assert isinstance(select_backend(config), V10FixedReplayBackend)

    def test_select_backend_rejects_unknown_replay_source(self, monkeypatch):
        monkeypatch.delenv("OPENAIR_CONGESTION_BACKEND", raising=False)
        config = OpenAirCongestionResourcesServerConfig(
            host="",
            port=0,
            entrypoint="",
            name="",
            replay_scenario_source="ambient_plugin",
        )
        with pytest.raises(ValueError, match="unknown replay_scenario_source"):
            select_backend(config)

    def test_select_backend_rejects_unknown_name(self, monkeypatch):
        monkeypatch.delenv("OPENAIR_CONGESTION_BACKEND", raising=False)
        config = OpenAirCongestionResourcesServerConfig(
            host="", port=0, entrypoint="", name="", backend="flexric_dreams"
        )
        with pytest.raises(ValueError, match="unknown backend"):
            select_backend(config)

    def test_oai_collector_is_a_stub_until_lab_access(self):
        with pytest.raises(NotImplementedError, match="lab access"):
            OAICollectorBackend()
