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
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import ValidationError

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

from openair_congestion.replay_env import action_effect_version  # noqa: E402

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


def _multi_tool_response() -> NeMoGymResponse:
    first = _tool_response("noop", {}).output[0]
    second = NeMoGymResponseFunctionToolCall(
        arguments=json.dumps({}),
        call_id="call_1",
        name="noop",
        type="function_call",
        id="fc_1",
        status="completed",
    )
    return NeMoGymResponse(output=[first, second], **_RESPONSE_KWARGS)


_TASK_METADATA = {
    "seed": 7001,
    "difficulty": 0.6,
    "regime_mix": {"prb_exhaustion": 1.0},
    "scenario_id": "prb_exhaustion",
    "tier": "replay",
    "max_steps": 16,
}
_DATASET_FIXTURE = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "sample_provided.jsonl"


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

    @pytest.mark.asyncio
    async def test_explicit_task_budget_is_capped_to_agent_budget(self):
        env = _make_env(agent_max_steps=16, max_steps_default=60)
        _, info = await env.reset(dict(_TASK_METADATA, max_steps=60), session_id="sid")
        assert env.session_state["sid"]["max_agent_steps"] == 16
        assert info["max_steps"] == 16

    @pytest.mark.parametrize("max_steps", [0, 1.5, True, float("inf"), float("nan")])
    @pytest.mark.asyncio
    async def test_invalid_task_budget_fails_before_allocating_episode(self, max_steps):
        env = _make_env()
        with pytest.raises(ValueError, match="positive integer"):
            await env.reset(dict(_TASK_METADATA, max_steps=max_steps), session_id="sid")
        assert env.session_state == {}

    @pytest.mark.asyncio
    async def test_reset_exposes_backend_semantics(self):
        env = _make_env()
        _, info = await env.reset(dict(_TASK_METADATA), session_id="sid")
        assert info["backend"] == "replay"
        assert info["dynamics_mode"] == action_effect_version()
        assert info["action_affects_observation"] is True
        assert info["reward_profile"] == "env_default"
        assert info["reward_weights"]["w_reject"] > 0.0
        assert info["observation_render"] == "verbose_v1"

    def test_replay_backend_fails_closed_on_inconsistent_version_exports(self, monkeypatch):
        import resources_servers.openair_congestion.backends as backends_module

        monkeypatch.setattr(
            backends_module._replay_env,
            "action_effect_version",
            lambda: f"{backends_module._replay_env.ACTION_EFFECT_VERSION}_stale",
        )
        with pytest.raises(RuntimeError, match="action-effect receipt mismatch"):
            ReplayBackend()

    def test_replay_backend_fails_closed_on_missing_version_export(self, monkeypatch):
        import resources_servers.openair_congestion.backends as backends_module

        monkeypatch.delattr(backends_module._replay_env, "ACTION_EFFECT_VERSION")
        with pytest.raises(RuntimeError, match="ACTION_EFFECT_VERSION"):
            ReplayBackend()

    def test_replay_backend_fails_closed_on_missing_version_reporter(self, monkeypatch):
        import resources_servers.openair_congestion.backends as backends_module

        monkeypatch.delattr(backends_module._replay_env, "action_effect_version")
        with pytest.raises(RuntimeError, match=r"callable action_effect_version\(\)"):
            ReplayBackend()

    @pytest.mark.asyncio
    async def test_resource_compact_observation_render_is_selected_and_stamped(self):
        env = _make_env(observation_render="resource_compact_pipe_v1")
        obs, info = await env.reset(dict(_TASK_METADATA), session_id="sid")
        assert obs.startswith("T|")
        assert "A|one_tool_call_or_noop" in obs
        assert info["observation_render"] == "resource_compact_pipe_v1"

    @pytest.mark.asyncio
    async def test_strict_t2_render_delegates_to_telco_renderer(self, monkeypatch):
        import resources_servers.openair_congestion.app as app_module

        monkeypatch.setattr(app_module, "_load_t2_compact_renderer", lambda: lambda _: "strict-t2")
        env = _make_env(observation_render="t2_compact_pipe_v2")
        obs, info = await env.reset(dict(_TASK_METADATA), session_id="sid")
        assert obs == "strict-t2"
        assert info["observation_render"] == "t2_compact_pipe_v2"


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
        assert info["dynamics_mode"] == action_effect_version()
        # The applied call gets a matching function_call_output for the agent.
        assert info["tool_outputs"][0]["call_id"] == "call_0"
        assert env.session_state["sid"]["n_steps"] == 1

    @pytest.mark.asyncio
    async def test_replay_backend_rejects_missing_runtime_dynamics_receipt(self, monkeypatch):
        env = _make_env()
        await env.reset(dict(_TASK_METADATA), session_id="sid")
        backend = env.backend
        assert isinstance(backend, ReplayBackend)
        original_step = backend._env.step

        def step_without_receipt(*args, **kwargs):
            observation, reward, terminated, info = original_step(*args, **kwargs)
            info = dict(info)
            info.pop("dynamics_mode", None)
            return observation, reward, terminated, info

        monkeypatch.setattr(backend._env, "step", step_without_receipt)
        with pytest.raises(RuntimeError, match="step dynamics_mode receipt is missing"):
            await env.step(_tool_response("noop", {}), {}, session_id="sid")

    @pytest.mark.asyncio
    async def test_app_rejects_runtime_receipt_conflict_instead_of_overwriting(self, monkeypatch):
        env = _make_env()
        await env.reset(dict(_TASK_METADATA), session_id="sid")
        backend = env.backend
        assert isinstance(backend, ReplayBackend)

        # Bypass ReplayBackend.step's own check to prove the app-level merge is
        # independently fail-closed and cannot mask a real runtime receipt.
        def unvalidated_step(episode_id, tool_call):
            observation, reward, terminated, info = backend._env.step(episode_id, tool_call)
            info = dict(info)
            info["dynamics_mode"] = f"{action_effect_version()}_contradiction"
            return observation, reward, terminated, info

        monkeypatch.setattr(backend, "step", unvalidated_step)
        with pytest.raises(RuntimeError, match="backend runtime receipt conflict"):
            await env.step(_tool_response("noop", {}), {}, session_id="sid")

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
    async def test_unknown_tool_name_consumes_penalized_transition(self):
        env = _make_env()
        await env.reset(dict(_TASK_METADATA), session_id="sid")
        obs, reward, term, trunc, info = await env.step(_tool_response("open_pod_bay_doors", {}), {}, session_id="sid")
        assert math.isfinite(reward)
        assert info["error"] == "invalid_tool_call"
        assert info["guardrail_accepted"] is False
        assert info["reward_terms"]["reject"] < 0.0
        assert env.session_state["sid"]["n_steps"] == 1
        assert "Last action: invalid_tool_call" in obs

    @pytest.mark.asyncio
    async def test_no_tool_call_consumes_penalized_transition(self):
        env = _make_env()
        await env.reset(dict(_TASK_METADATA), session_id="sid")
        obs, reward, term, trunc, info = await env.step(
            _text_response("Hmm, the PRBs look full."), {}, session_id="sid"
        )
        assert math.isfinite(reward)
        assert term is False and trunc is False
        assert info["error"] == "no_tool_call"
        assert info["guardrail_accepted"] is False
        assert info["reward_terms"]["reject"] < 0.0
        assert info["backend"] == "replay"
        assert info["action_affects_observation"] is True
        assert env.session_state["sid"]["n_steps"] == 1
        assert "Last action: no_tool_call" in obs

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_reject_whole_turn_with_matching_outputs(self):
        env = _make_env()
        await env.reset(dict(_TASK_METADATA), session_id="sid")
        _, reward, _, _, info = await env.step(_multi_tool_response(), {}, session_id="sid")
        assert math.isfinite(reward)
        assert info["error"] == "multiple_tool_calls"
        assert info["guardrail_accepted"] is False
        assert info["reward_terms"]["reject"] < 0.0
        assert [item["call_id"] for item in info["tool_outputs"]] == ["call_0", "call_1"]
        assert env.session_state["sid"]["n_steps"] == 1

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
    async def test_dataset_index_is_forwarded_and_semantics_are_exposed(self):
        env = _make_env(backend="dataset_replay", dataset_path=str(_DATASET_FIXTURE))
        async with _http_client(env.setup_webserver()) as client:
            response = await client.post(
                "/reset",
                json=_reset_body(scenario_id=None, dataset_index=1),
            )
            assert response.status_code == 200
            info = response.json()["info"]
            assert info["scenario_id"] == "lab_run_b"
            assert info["backend"] == "dataset_replay"
            assert info["dynamics_mode"] == "provided_data_passthrough_v1"
            assert info["action_affects_observation"] is False
            assert info["dataset_identity"].startswith("sha256:")
            assert info["dataset_sha256"] in info["dataset_identity"]
            assert info["dataset_row_count"] == 7
            assert info["dataset_episode_count"] == 2
            assert info["reward_weights"]["w_reject"] > 0.0
            assert info["reconstruction_schema"] == "native_snapshot_v1"

    @pytest.mark.asyncio
    async def test_close_is_cookie_scoped_and_idempotent(self):
        env = _make_env(pool_size=1)
        app = env.setup_webserver()
        async with _http_client(app) as owner, _http_client(app) as stranger:
            reset = await owner.post("/reset", json=_reset_body())
            episode_id = reset.json()["info"]["episode_id"]
            assert len(env.session_state) == 1

            # A different cookie cannot name or release the owner's episode.
            response = await stranger.post("/close", json={"episode_id": episode_id})
            assert response.status_code == 200
            assert response.json() == {"ok": True, "closed": False}
            assert len(env.session_state) == 1

            response = await owner.post("/close", json={})
            assert response.json() == {"ok": True, "closed": True}
            assert env.session_state == {}

            # Repeating close is a successful no-op and the pool slot is free.
            response = await owner.post("/close", json={})
            assert response.json() == {"ok": True, "closed": False}
            reset_again = await stranger.post("/reset", json=_reset_body(seed=7002))
            assert reset_again.status_code == 200

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


class TestBackends:
    def test_select_backend_defaults_to_replay(self, monkeypatch):
        monkeypatch.delenv("OPENAIR_CONGESTION_BACKEND", raising=False)
        config = OpenAirCongestionResourcesServerConfig(host="", port=0, entrypoint="", name="")
        assert isinstance(select_backend(config), ReplayBackend)

    def test_select_backend_rejects_unknown_name(self, monkeypatch):
        monkeypatch.delenv("OPENAIR_CONGESTION_BACKEND", raising=False)
        with pytest.raises(ValidationError, match="backend"):
            OpenAirCongestionResourcesServerConfig(host="", port=0, entrypoint="", name="", backend="flexric_dreams")

    def test_oai_collector_is_a_stub_until_lab_access(self):
        with pytest.raises(NotImplementedError, match="lab access"):
            OAICollectorBackend()


class TestConfigValidation:
    def test_reward_weights_are_explicit_and_forwarded(self):
        config = OpenAirCongestionResourcesServerConfig(
            host="",
            port=0,
            entrypoint="",
            name="",
            backend="dataset_replay",
            dataset_path=str(_DATASET_FIXTURE),
            reward_profile="openair_v2_measured",
            reward_weights={
                "w_sla": 0.0,
                "w_sla_level": 0.0,
                "w_buffer": 0.0,
                "w_action": 0.0,
            },
        )
        backend = select_backend(config)
        assert backend.reward_profile == "openair_v2_measured"
        assert backend.reward_weights_dict["w_sla"] == 0.0
        assert backend.reward_weights_dict["w_reject"] > 0.0

    @pytest.mark.parametrize(
        "overrides",
        [
            {"reward_weights": {"w_typo": 1.0}},
            {"reward_weights": {"w_reject": -1.0}},
            {"cell_capacity_mbps": float("inf")},
            {"pool_size": 0},
            {"misspelled_reward_profile": "v2"},
            {"reward_profile": "openair_v2_measured"},
            {"reward_profile": "openair_v1", "reward_weights": {"w_sla": 0.0}},
            {"reward_profile": "custom"},
            {
                "backend": "replay",
                "reward_profile": "openair_v2_measured",
                "reward_weights": {
                    "w_sla": 0.0,
                    "w_sla_level": 0.0,
                    "w_buffer": 0.0,
                    "w_action": 0.0,
                },
            },
            {
                "backend": "dataset_replay",
                "observation_render": "t2_compact_pipe_v2",
            },
        ],
    )
    def test_bad_or_unknown_config_fails_at_startup(self, overrides):
        with pytest.raises(ValidationError):
            OpenAirCongestionResourcesServerConfig(host="", port=0, entrypoint="", name="", **overrides)
