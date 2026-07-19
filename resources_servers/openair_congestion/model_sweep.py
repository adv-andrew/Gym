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

"""Capability sweep: profile how the environment attributes reward across policies.

Environment sanity check, run before any training: drive seeded episodes over
the real HTTP surface for a ladder of scripted anchor policies of known
quality (congestion relief > noop > random-valid > catastrophic) and,
optionally, any number of OpenAI-compatible chat-completions models. A sound
environment must (a) rank the scripted anchors in their known order and
(b) rank LLM policies consistently with their general capability -- if a
frontier model lands no higher than a small one, the reward attribution is
suspect, not the models.

Anchors need no model server or API key; the sweep runs fully offline on the
replay backend. LLM policies are described in a JSON file (see --models):

    [{"label": "gpt-5.6", "model": "gpt-5.6", "base_url": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY", "temperature": 0.2}]

Each model receives the task row's own messages (system prompt + task
prompt), the current rendered observation as the latest user message, and the
row's tool schemas converted to chat-completions format. Single-turn on
purpose: each step stands alone, so models are compared on state-reading, not
context management. A reply without a parseable tool call is counted as a
parse failure and stepped as noop, a parseable call the env rejects as an
unknown tool is counted as invalid, and a dead endpoint drops the episode as
an infra error -- the episode is drained server-side either way, so one
model's failures never starve the pool for the next.

Usage:
    python resources_servers/openair_congestion/model_sweep.py
    python resources_servers/openair_congestion/model_sweep.py --models sweep_models.json --out sweep.json
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import math
import os
import random
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import aiohttp

from resources_servers.openair_congestion.client import (
    _load_task_row,
    _start_local_server,
    _tool_response,
    choose_action,
)


# Same fixed (seed, difficulty) ladder as tests/test_reward_correctness.py: the
# replay backend is deterministic, so anchor orderings are stable across runs.
_DEFAULT_TASKS = ((42, 0.9), (123, 0.5), (555, 0.95), (888, 0.95))
_SCHEDULERS = ("PF", "RR", "MaxCI")


# --- Policies ----------------------------------------------------------------
# A policy maps (rendered observation, step_idx, rng) -> tool call dict, plus
# an async LLM variant. Anchors mirror the reward-oracle test policies but
# read the rendered text, exactly like an LLM would.


def _noop(observation: str, step_idx: int, rng: random.Random) -> dict[str, Any]:
    return {"name": "noop", "arguments": {}}


def _catastrophic(observation: str, step_idx: int, rng: random.Random) -> dict[str, Any]:
    # max_prb=0 starves the target and is guardrail-rejected every step.
    return {"name": "set_prb_cap", "arguments": {"cell_id": 0, "target": "ue", "target_id": 0, "max_prb": 0}}


_CELL_HEADER = re.compile(r"- Cell (\d+):")
_UE_LINE = re.compile(r"UE (\d+) \(")


def _parse_topology(observation: str) -> dict[int, list[int]]:
    # cell_id -> UE ids, read from the rendered per-cell / per-UE lines --
    # the same text an LLM policy sees.
    cells: dict[int, list[int]] = {}
    current: int | None = None
    for line in (observation or "").splitlines():
        header = _CELL_HEADER.search(line)
        if header:
            current = int(header.group(1))
            cells[current] = []
        elif current is not None:
            ue = _UE_LINE.search(line)
            if ue:
                cells[current].append(int(ue.group(1)))
    return cells or {0: [0]}


def _make_random_valid() -> Callable[[str, int, random.Random], dict[str, Any]]:
    # Mirrors the reward-oracle test policy (tests/test_reward_correctness.py):
    # uniform over the five guardrail-valid tool families with the same argument
    # ranges.  "valid" means accepted by the guardrail, not beneficial: under
    # the V3 persistent zero-sum PRB dynamics this deliberately unguided policy
    # is expected to score below standing pat on the fixed ladder tasks.
    # deduplicated against the last two actions so the identical-action rate
    # limit never fires -- but reading the rendered text, like an LLM would.
    recent: list[str] = []

    def _sample(observation: str, rng: random.Random) -> dict[str, Any]:
        cells = _parse_topology(observation)
        cell_id = rng.choice(sorted(cells))
        choice = rng.randrange(5)
        if choice == 0:
            return {
                "name": "set_scheduler_policy",
                "arguments": {"cell_id": cell_id, "policy": rng.choice(_SCHEDULERS)},
            }
        if choice == 1:
            ue_id = rng.choice(cells[cell_id] or [0])
            return {
                "name": "set_prb_cap",
                "arguments": {
                    "cell_id": cell_id,
                    "target": "ue",
                    "target_id": ue_id,
                    "max_prb": rng.randrange(10, 273),
                },
            }
        if choice == 2:
            return {
                "name": "set_mcs_bounds",
                "arguments": {"cell_id": cell_id, "mcs_min": 0, "mcs_max": rng.randrange(5, 28), "target_bler": 0.1},
            }
        if choice == 3:
            return {
                "name": "set_admission_policy",
                "arguments": {
                    "cell_id": cell_id,
                    "accept_threshold_pct": rng.randrange(10, 100),
                    "slice_reservation": {},
                },
            }
        return {
            "name": "set_ul_power_control",
            "arguments": {
                "cell_id": cell_id,
                "p0_dbm": rng.randrange(-120, 20),
                "alpha": rng.choice([0.4, 0.7, 0.8, 1.0]),
            },
        }

    def policy(observation: str, step_idx: int, rng: random.Random) -> dict[str, Any]:
        action = _sample(observation, rng)
        for _ in range(20):
            key = json.dumps(action, sort_keys=True)
            if key not in recent:
                break
            action = _sample(observation, rng)
        recent.append(json.dumps(action, sort_keys=True))
        del recent[:-2]
        return action

    return policy


# Factories: random-valid keeps per-episode dedupe state, so every policy is
# constructed fresh per episode.
_ANCHORS: dict[str, Callable[[], Callable[[str, int, random.Random], dict[str, Any]]]] = {
    "anchor:relief": lambda: (lambda obs, i, rng: choose_action(obs, i)),
    "anchor:random-valid": _make_random_valid,
    "anchor:noop": lambda: _noop,
    "anchor:catastrophic": lambda: _catastrophic,
}
# The known quality ordering for the V3 persistent zero-sum replay dynamics
# (best to worst).  Guardrail-valid random control is intentionally not
# treated as beneficial control.
_ANCHOR_ORDER = ("anchor:relief", "anchor:noop", "anchor:random-valid", "anchor:catastrophic")


@dataclass
class ModelSpec:
    label: str
    model: str
    base_url: str
    api_key_env: str = ""
    temperature: float = 0.2
    max_tokens: int = 512


@dataclass
class PolicyStats:
    returns: list[float] = field(default_factory=list)
    steps: int = 0
    rejected: int = 0
    noop_steps: int = 0
    invalid_calls: int = 0
    parse_failures: int = 0
    infra_errors: int = 0

    def row(self, label: str) -> dict[str, Any]:
        std = statistics.pstdev(self.returns) if len(self.returns) > 1 else 0.0
        return {
            "policy": label,
            "episodes": len(self.returns),
            "mean_return": round(statistics.fmean(self.returns), 4) if self.returns else None,
            "std_return": round(std, 4),
            "rejection_rate": round(self.rejected / self.steps, 4) if self.steps else 0.0,
            "noop_rate": round(self.noop_steps / self.steps, 4) if self.steps else 0.0,
            "invalid_calls": self.invalid_calls,
            "parse_failures": self.parse_failures,
            "infra_errors": self.infra_errors,
        }


def _chat_tools(row: dict) -> list[dict]:
    # Responses-API tool schemas -> chat-completions tool schemas.
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {}),
            },
        }
        for t in row["responses_create_params"].get("tools", [])
        if t.get("type") == "function"
    ]


def _parse_tool_call(message: dict) -> dict[str, Any] | None:
    calls = message.get("tool_calls") or []
    if calls:
        fn = calls[0].get("function", {})
        try:
            arguments = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            return None
        if not isinstance(fn.get("name"), str) or not isinstance(arguments, dict):
            return None
        return {"name": fn["name"], "arguments": arguments}
    # Fallback: a bare JSON object {"name": ..., "arguments": {...}} in content.
    content = message.get("content") or ""
    start, end = content.find("{"), content.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and isinstance(obj.get("arguments", {}), dict):
            return {"name": obj["name"], "arguments": obj.get("arguments") or {}}
    return None


async def _llm_action(
    session: aiohttp.ClientSession, spec: ModelSpec, row: dict, observation: str, step_idx: int
) -> dict[str, Any] | None:
    # The row's own input messages (system prompt + task prompt), then the
    # current rendered observation. Single-turn on purpose: each step stands
    # alone, so models are compared on state-reading, not context management.
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in row["responses_create_params"]["input"]
        if m.get("role") in ("system", "user")
    ]
    messages.append({"role": "user", "content": observation})
    headers = {}
    if spec.api_key_env:
        headers["Authorization"] = f"Bearer {os.environ[spec.api_key_env]}"
    payload = {
        "model": spec.model,
        "messages": messages,
        "tools": _chat_tools(row),
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
    }
    async with session.post(
        f"{spec.base_url.rstrip('/')}/chat/completions",
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=90),
    ) as response:
        response.raise_for_status()
        try:
            body = await response.json()
        except (json.JSONDecodeError, aiohttp.ContentTypeError):
            return None  # 200 with a non-JSON body: a parse failure, not a crash
    # Tolerate non-conforming 200 bodies ({"error": ...}, empty choices, null
    # message) the same way as unparseable content: infra_errors stays
    # reserved for transport-level failures.
    choices = body.get("choices") or [] if isinstance(body, dict) else []
    message = choices[0].get("message") if choices and isinstance(choices[0], dict) else None
    return _parse_tool_call(message) if isinstance(message, dict) else None


# --- Episode driver ----------------------------------------------------------


async def _run_episode(
    base_url: str,
    row: dict,
    action_fn: Callable[[str, int], Awaitable[dict[str, Any] | None]],
    stats: PolicyStats,
) -> None:
    async with aiohttp.ClientSession(base_url=base_url, cookie_jar=aiohttp.CookieJar(unsafe=True)) as session:

        async def post(url_path: str, payload: dict) -> dict:
            async with session.post(url_path, json=payload) as response:
                response.raise_for_status()
                return await response.json()

        async def step_env(action: dict[str, Any], step_idx: int) -> dict:
            return await post(
                "/step",
                {
                    "responses_create_params": row["responses_create_params"],
                    "response": _tool_response(action["name"], action["arguments"], step_idx),
                },
            )

        reset = await post("/reset", row)
        observation = reset["observation"]
        episode_return, step_idx = 0.0, 0
        terminated = truncated = False
        # The env terminates at the row's max_steps; the 4x margin only guards
        # against a served instance that never sets terminated/truncated.
        step_cap = 4 * int(row.get("max_steps", 16))
        try:
            while not (terminated or truncated) and step_idx < step_cap:
                action = await action_fn(observation, step_idx)
                if action is None:  # unparseable model reply: step as noop, count it
                    stats.parse_failures += 1
                    action = {"name": "noop", "arguments": {}}
                step = await step_env(action, step_idx)
                episode_return += float(step["reward"])
                stats.steps += 1
                if not step["info"].get("guardrail_accepted", True):
                    stats.rejected += 1
                if step["info"].get("error") == "invalid_tool_call":
                    # Hallucinated tool names score 0.0 server-side; without
                    # this counter a never-valid model would look spotless.
                    stats.invalid_calls += 1
                if action["name"] == "noop":
                    stats.noop_steps += 1
                observation = step["observation"]
                terminated, truncated = bool(step["terminated"]), bool(step["truncated"])
                step_idx += 1
        except (aiohttp.ClientError, asyncio.TimeoutError):
            # The episode is already reset server-side; abandoning it here
            # would leak its replay-pool slot for the rest of the sweep (the
            # reaper only reclaims sessions the server no longer reports as
            # live). Drain to termination with noops, then let the caller
            # count the drop.
            while not (terminated or truncated) and step_idx < step_cap:
                step = await step_env({"name": "noop", "arguments": {}}, step_idx)
                terminated, truncated = bool(step["terminated"]), bool(step["truncated"])
                step_idx += 1
            raise
        stats.returns.append(episode_return)


def _task_rows(base_row: dict, tasks: list[tuple[int, float]], repeats: int) -> list[dict]:
    # Every policy runs this exact row list, so cross-policy comparisons stay
    # paired; the +1000 seed offset just keeps repeat episodes distinct.
    rows = []
    for repeat in range(repeats):
        for seed, difficulty in tasks:
            row = json.loads(json.dumps(base_row))
            row.update(seed=seed + repeat * 1000, difficulty=difficulty)
            rows.append(row)
    return rows


async def _scripted_action(
    policy: Callable[[str, int, random.Random], dict[str, Any]],
    rng: random.Random,
    observation: str,
    step_idx: int,
) -> dict[str, Any]:
    return policy(observation, step_idx, rng)


async def _sweep(tasks: list[tuple[int, float]], repeats: int, specs: list[ModelSpec]) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    labels = [spec.label for spec in specs]
    if len(labels) != len(set(labels)):
        raise ValueError(
            f"duplicate model labels in --models: {sorted(set(x for x in labels if labels.count(x) > 1))}"
        )
    for spec in specs:
        if spec.api_key_env and not os.environ.get(spec.api_key_env):
            raise ValueError(f"model '{spec.label}': environment variable {spec.api_key_env} is not set")

    base_url = _start_local_server()
    base_row = _load_task_row()
    rows = _task_rows(base_row, tasks, repeats)
    results: dict[str, PolicyStats] = {}

    for label, factory in _ANCHORS.items():
        stats = results[label] = PolicyStats()
        for row in rows:
            action_fn = functools.partial(_scripted_action, factory(), random.Random(row["seed"]))
            await _run_episode(base_url, row, action_fn, stats)

    for spec in specs:
        stats = results[f"model:{spec.label}"] = PolicyStats()
        async with aiohttp.ClientSession() as llm_session:
            for row in rows:
                action_fn = functools.partial(_llm_action, llm_session, spec, row)
                try:
                    await _run_episode(base_url, row, action_fn, stats)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    stats.infra_errors += 1
                    print(f"model:{spec.label} seed={row['seed']}: episode dropped ({type(e).__name__}: {e})")

    relief_mean = results["anchor:relief"].row("anchor:relief")["mean_return"]
    table = []
    for label, stats in results.items():
        entry = stats.row(label)
        entry["vs_relief"] = round(entry["mean_return"] - relief_mean, 4) if entry["mean_return"] is not None else None
        table.append(entry)
    table.sort(key=lambda r: r["mean_return"] if r["mean_return"] is not None else -math.inf, reverse=True)

    anchor_means = {label: results[label].row(label)["mean_return"] for label in _ANCHOR_ORDER}
    ordered = all(anchor_means[a] > anchor_means[b] for a, b in zip(_ANCHOR_ORDER, _ANCHOR_ORDER[1:]))
    return {
        "tasks": [{"seed": s, "difficulty": d} for s, d in tasks],
        "repeats": repeats,
        "episodes_per_policy": len(rows),
        "profile": table,
        "anchor_ordering_ok": ordered,
        "anchor_order_expected": list(_ANCHOR_ORDER),
    }


def _print_report(report: dict[str, Any]) -> None:
    print(
        f"\ncapability sweep: {report['episodes_per_policy']} episodes/policy over "
        f"{len(report['tasks'])} fixed tasks x {report['repeats']} repeats\n"
    )
    header = (
        f"{'policy':<24} {'mean':>9} {'std':>7} {'vs relief':>10} {'reject%':>8} "
        f"{'noop%':>7} {'invalid':>8} {'parse-fail':>11} {'infra':>6} {'eps':>4}"
    )
    print(header)
    print("-" * len(header))
    for r in report["profile"]:
        mean = f"{r['mean_return']:>9.3f}" if r["mean_return"] is not None else f"{'--':>9}"
        vs = f"{r['vs_relief']:>+10.3f}" if r["vs_relief"] is not None else f"{'--':>10}"
        print(
            f"{r['policy']:<24} {mean} {r['std_return']:>7.3f} {vs} "
            f"{r['rejection_rate'] * 100:>7.1f}% {r['noop_rate'] * 100:>6.1f}% "
            f"{r['invalid_calls']:>8} {r['parse_failures']:>11} {r['infra_errors']:>6} {r['episodes']:>4}"
        )
    verdict = "PASS" if report["anchor_ordering_ok"] else "FAIL"
    print(f"\nanchor ordering (relief > noop > random-valid > catastrophic): {verdict}")
    if not report["anchor_ordering_ok"]:
        print("a broken anchor ordering means reward attribution is suspect -- investigate before training.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", help="JSON file with a list of model specs (see sweep_models.example.json)")
    parser.add_argument("--repeats", type=int, default=2, help="episodes per fixed task (default 2)")
    parser.add_argument("--out", help="write the full report as JSON to this path")
    args = parser.parse_args()

    specs = []
    if args.models:
        with open(args.models) as f:
            specs = [ModelSpec(**spec) for spec in json.load(f)]

    try:
        report = asyncio.run(_sweep(list(_DEFAULT_TASKS), args.repeats, specs))
    except ValueError as e:
        raise SystemExit(str(e)) from e
    _print_report(report)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"report written to {args.out}")
    # Nonzero exit on a broken anchor ladder, so the sweep works as a CI gate.
    if not report["anchor_ordering_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
