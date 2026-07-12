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
# The capability sweep is the pre-training environment check: scripted
# anchors of known quality must land in their known order over the real HTTP
# surface, and the report must expose the profile a reviewer needs. The LLM
# path runs end-to-end against a stub chat-completions server, so request
# shape, parsing, and failure accounting are covered without a network.
import asyncio
import json
import random

import pytest
from aiohttp import web


pytest.importorskip(
    "openair_congestion",
    reason="telco env package 'openair_congestion' not installed; see README Setup",
)

from resources_servers.openair_congestion.client import _free_port  # noqa: E402
from resources_servers.openair_congestion.model_sweep import (  # noqa: E402
    _ANCHOR_ORDER,
    ModelSpec,
    _make_random_valid,
    _parse_tool_call,
    _parse_topology,
    _sweep,
)


def test_anchor_sweep_orders_policies_and_reports_profile():
    report = asyncio.run(_sweep([(42, 0.9), (555, 0.95)], repeats=1, specs=[]))

    assert report["episodes_per_policy"] == 2
    by_policy = {row["policy"]: row for row in report["profile"]}
    assert set(by_policy) == set(_ANCHOR_ORDER)
    assert all(row["episodes"] == 2 for row in report["profile"])

    # The known quality ladder must hold on the deterministic replay tasks,
    # and the report must say so itself.
    assert report["anchor_ordering_ok"] is True
    means = [by_policy[label]["mean_return"] for label in _ANCHOR_ORDER]
    assert means == sorted(means, reverse=True)

    # Catastrophic play is rejected every step; valid play never is.
    assert by_policy["anchor:catastrophic"]["rejection_rate"] == 1.0
    assert by_policy["anchor:random-valid"]["rejection_rate"] == 0.0
    assert by_policy["anchor:noop"]["noop_rate"] == 1.0


def test_random_valid_never_repeats_within_rate_limit_window():
    policy = _make_random_valid()
    observation = (
        "- Cell 0: DL PRB util p50=34%, p99=41%; 0 SLA violation(s) in last 5s.\n"
        "    UE 0 (5QI 9): offered 8.0 Mbps.\n"
        "    UE 1 (5QI 9): offered 8.0 Mbps.\n"
        "- Cell 1: DL PRB util p50=24%, p99=30%; 0 SLA violation(s) in last 5s.\n"
        "    UE 0 (5QI 9): offered 8.0 Mbps.\n"
    )
    rng = random.Random(0)
    keys = [json.dumps(policy(observation, i, rng), sort_keys=True) for i in range(64)]
    # The guardrail rejects an identical action within its 2 s window (two
    # logical steps); the policy's dedupe must keep adjacent pairs distinct.
    assert all(keys[i] != keys[i - 1] for i in range(1, len(keys)))
    assert all(keys[i] != keys[i - 2] for i in range(2, len(keys)))


def test_parse_topology_reads_cells_and_ues():
    observation = "- Cell 0: stuff\n    UE 0 (5QI 9): x\n    UE 1 (5QI 9): x\n- Cell 1: stuff\n    UE 0 (5QI 9): x\n"
    assert _parse_topology(observation) == {0: [0, 1], 1: [0]}
    # Unparseable text degrades to a safe single-cell fallback, never a crash.
    assert _parse_topology("") == {0: [0]}


def test_llm_policies_end_to_end_against_mock_endpoint():
    # A stub chat-completions server proves the whole LLM path with no
    # network and no key. Four model personas cover the failure taxonomy:
    # "cooperative" calls noop natively; "rambler" replies prose only (parse
    # failures, stepped as noop); "hallucinator" calls a tool that does not
    # exist (invalid calls -- the env scores 0.0, and without the counter it
    # would look spotless); "dead" 500s every request (episodes dropped as
    # infra errors). "dead" runs FIRST so the test also proves a failing
    # model cannot starve the replay pool for the models after it.
    seen_payloads: list[dict] = []

    async def chat_completions(request: web.Request) -> web.Response:
        payload = await request.json()
        seen_payloads.append(payload)
        if payload["model"] == "dead":
            return web.json_response({"error": "upstream unavailable"}, status=500)
        if payload["model"] == "cooperative":
            message = {"tool_calls": [{"function": {"name": "noop", "arguments": "{}"}}], "content": None}
        elif payload["model"] == "hallucinator":
            message = {"tool_calls": [{"function": {"name": "restart_gnb", "arguments": "{}"}}], "content": None}
        else:
            message = {"content": "The network seems fine; monitoring further."}
        return web.json_response({"choices": [{"message": message}]})

    async def run() -> dict:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat_completions)
        runner = web.AppRunner(app)
        await runner.setup()
        port = _free_port()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()
        try:
            base_url = f"http://127.0.0.1:{port}/v1"
            specs = [
                ModelSpec(label="dead", model="dead", base_url=base_url),
                ModelSpec(label="cooperative", model="cooperative", base_url=base_url),
                ModelSpec(label="hallucinator", model="hallucinator", base_url=base_url),
                ModelSpec(label="rambler", model="rambler", base_url=base_url),
            ]
            return await _sweep([(42, 0.9)], repeats=1, specs=specs)
        finally:
            await runner.cleanup()

    report = asyncio.run(run())
    by_policy = {row["policy"]: row for row in report["profile"]}

    dead = by_policy["model:dead"]
    assert dead["episodes"] == 0
    assert dead["infra_errors"] == 1
    assert dead["mean_return"] is None and dead["vs_relief"] is None

    # The models after the dead one still complete: its drained episode did
    # not leak a replay-pool slot or bleed errors into their rows.
    cooperative = by_policy["model:cooperative"]
    assert cooperative["episodes"] == 1
    assert cooperative["parse_failures"] == 0
    assert cooperative["noop_rate"] == 1.0
    assert cooperative["infra_errors"] == 0

    hallucinator = by_policy["model:hallucinator"]
    assert hallucinator["episodes"] == 1
    assert hallucinator["invalid_calls"] > 0
    assert hallucinator["parse_failures"] == 0  # the call parses; the env refuses it

    rambler = by_policy["model:rambler"]
    assert rambler["episodes"] == 1
    assert rambler["parse_failures"] > 0
    assert rambler["noop_rate"] == 1.0  # unparseable turns are stepped as noop

    # LLM rows carry the reward-ceiling reference; noop-playing models must
    # match the noop anchor's return on the same paired task.
    assert cooperative["mean_return"] == by_policy["anchor:noop"]["mean_return"]
    assert cooperative["vs_relief"] is not None

    # The request carried the row's own messages plus the observation.
    payload = seen_payloads[0]
    assert payload["messages"][0]["role"] == "system" and payload["messages"][0]["content"]
    assert payload["messages"][-1]["role"] == "user" and "Cell 0" in payload["messages"][-1]["content"]
    assert any(t["function"]["name"] == "set_prb_cap" for t in payload["tools"])


def test_sweep_rejects_bad_config():
    with pytest.raises(ValueError, match="repeats"):
        asyncio.run(_sweep([(42, 0.9)], repeats=0, specs=[]))
    dup = [
        ModelSpec(label="same", model="a", base_url="http://127.0.0.1:1/v1"),
        ModelSpec(label="same", model="b", base_url="http://127.0.0.1:1/v1"),
    ]
    with pytest.raises(ValueError, match="duplicate model labels"):
        asyncio.run(_sweep([(42, 0.9)], repeats=1, specs=dup))


def test_sweep_refuses_named_but_unset_api_key_env(monkeypatch):
    monkeypatch.delenv("SWEEP_TEST_MISSING_KEY", raising=False)
    spec = ModelSpec(label="x", model="x", base_url="http://127.0.0.1:1/v1", api_key_env="SWEEP_TEST_MISSING_KEY")
    with pytest.raises(ValueError, match="SWEEP_TEST_MISSING_KEY"):
        asyncio.run(_sweep([(42, 0.9)], repeats=1, specs=[spec]))


def test_parse_tool_call_handles_tool_calls_content_json_and_garbage():
    native = {
        "tool_calls": [{"function": {"name": "noop", "arguments": "{}"}}],
        "content": None,
    }
    assert _parse_tool_call(native) == {"name": "noop", "arguments": {}}

    content = {"content": 'Sure: {"name": "set_scheduler_policy", "arguments": {"cell_id": 0, "policy": "PF"}}'}
    assert _parse_tool_call(content) == {
        "name": "set_scheduler_policy",
        "arguments": {"cell_id": 0, "policy": "PF"},
    }

    assert _parse_tool_call({"content": "I would consider the network first."}) is None
    assert _parse_tool_call({"tool_calls": [{"function": {"name": "noop", "arguments": "{not json"}}]}) is None
