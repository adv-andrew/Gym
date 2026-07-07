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

from resources_servers.openair_congestion.app import (  # noqa: E402
    OpenAirCongestionEnv,
    OpenAirCongestionResourcesServerConfig,
)
from resources_servers.openair_congestion.backends import (  # noqa: E402
    OAICollectorBackend,
    ReplayBackend,
    select_backend,
)


def _make_env(**config_overrides) -> OpenAirCongestionEnv:
    config = OpenAirCongestionResourcesServerConfig(host="", port=0, entrypoint="", name="", **config_overrides)
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


_TASK_METADATA = {
    "seed": 7001,
    "difficulty": 0.6,
    "regime_mix": {"prb_exhaustion": 1.0},
    "scenario_id": "prb_exhaustion",
    "tier": "replay",
    "max_steps": 16,
}


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


class TestRoutes:
    def test_gymnasium_routes_registered(self):
        env = _make_env()
        routes = {r.path for r in env.setup_webserver().routes}
        assert {"/reset", "/step", "/aggregate_metrics"}.issubset(routes)


class TestBackends:
    def test_select_backend_defaults_to_replay(self, monkeypatch):
        monkeypatch.delenv("OPENAIR_CONGESTION_BACKEND", raising=False)
        config = OpenAirCongestionResourcesServerConfig(host="", port=0, entrypoint="", name="")
        assert isinstance(select_backend(config), ReplayBackend)

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
