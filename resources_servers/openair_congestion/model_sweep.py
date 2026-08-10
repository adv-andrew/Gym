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
the real HTTP surface for scripted anchor policies with a known partial
order (relief beats noop and random-valid; both beat catastrophic play) and,
optionally, ranked OpenAI-compatible chat-completions models. The scripted
anchor check is a deterministic correctness gate. The model ladder is an empirical
gate: if a declared frontier model does not beat a smaller model on complete
paired profiles, investigate the prompt, task coverage, and reward before
training rather than assuming either the environment or model is correct.

Anchors need no model server or API key; the sweep runs fully offline on the
replay backend. LLM policies are described in a JSON file (see --models):

    [{"label": "frontier", "model": "<frontier-model>", "base_url": "https://api.example/v1",
      "api_key_env": "FRONTIER_API_KEY", "temperature": 0.2, "top_p": 0.95,
      "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": false},
      "capability_rank": 2}]

Each model receives every task row's own messages (system prompt + task
prompt), the current rendered observation as the latest user message, and the
row's tool schemas converted to chat-completions format. Each decision is
single-turn on
purpose: each step stands alone, so models are compared on state-reading, not
context management. A reply without a parseable tool call is counted as a
parse failure and sent to the environment as a terminal protocol violation; a
parseable call the environment rejects as an unknown tool is counted as
invalid; and a dead endpoint drops the episode as an infrastructure error.
Dropped episodes are drained server-side, so one model's failures never starve
the pool for the next. Unparseable replies are never upgraded to valid noops.
Generic exploratory sweeps may omit ``chat_template_kwargs``. Every endpoint in
a named Run 1B smoke or compliance profile must explicitly disable thinking so
the tool-only 512-token budget cannot be consumed by hidden reasoning before
the native tool call.

Usage:
    python resources_servers/openair_congestion/model_sweep.py
    python resources_servers/openair_congestion/model_sweep.py --models sweep_models.json --out sweep.json
    python resources_servers/openair_congestion/model_sweep.py --compliance-profile --models sweep_models.json
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import json
import math
import os
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping
from urllib.parse import urlsplit

import aiohttp
import numpy as np
from openair_congestion.schemas import SUPPORTED_REGIMES

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from resources_servers.openair_congestion.app import _strict_json_object
from resources_servers.openair_congestion.client import (
    _start_local_server,
    _tool_response,
    choose_action,
)


_EXAMPLE_JSONL = Path(__file__).parent / "data" / "example.jsonl"
_SCHEDULERS = ("PF", "RR", "MaxCI")
_BOOTSTRAP_METHOD = "regime_stratified_prompt_cluster_percentile"
_DEFAULT_BOOTSTRAP_DRAWS = 10_000
_COMPLIANCE_BOOTSTRAP_DRAWS = 50_000
_RUN1B_TEMPERATURE = 0.2
_RUN1B_TOP_P = 0.95
_RUN1B_MAX_TOKENS = 512
_MAX_MODEL_OUTPUT_TOKENS = 512
_PAIR_KEY = re.compile(r"^(0|[1-9][0-9]*):(0|[1-9][0-9]*)$")
_REQUEST_SEED_VERSION = "run1b-request-v1"
_REQUEST_SEED_DERIVATION = "sha256(run1b-request-v1:{prompt_index}:{response_index}:{step_index}) mod 2**31"
_REQUEST_CAPTURE_SCHEMA = "openair.run1b.request-capture.v1"
_RUN1B_CAPTURE_STEPS = 16
_RUN1B_LOCAL_BASE_URL = re.compile(r"http://127\.0\.0\.1:([1-9][0-9]{0,4})/v1")


def _load_example_rows(path: Path = _EXAMPLE_JSONL) -> list[dict[str, Any]]:
    """Load every checked-in task row; never silently profile only row one."""

    rows = [_strict_json_object(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 5:
        raise ValueError(f"reward profiling requires at least five example rows, got {len(rows)} from {path}")
    for row in rows:
        row.pop("agent_ref", None)
    return rows


def _validate_compliance_rows(rows: list[dict[str, Any]]) -> None:
    """Require one correctly labeled, one-hot task for every supported regime."""

    expected = set(SUPPORTED_REGIMES)
    counts = Counter(row.get("scenario_id") for row in rows)
    if set(counts) != expected or any(counts[regime] != 1 for regime in expected):
        raise ValueError(
            "compliance-profile example rows must cover every supported scenario_id exactly once; "
            f"expected {sorted(expected)}, got {dict(counts)}"
        )
    for index, row in enumerate(rows):
        scenario_id = row["scenario_id"]
        regime_mix = row.get("regime_mix")
        weight = regime_mix.get(scenario_id) if isinstance(regime_mix, dict) else None
        if (
            not isinstance(regime_mix, dict)
            or set(regime_mix) != {scenario_id}
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) != 1.0
        ):
            raise ValueError(
                f"compliance-profile row {index} ({scenario_id!r}) must have a one-hot regime_mix "
                f"of {{{scenario_id!r}: 1.0}}, got {regime_mix!r}"
            )


def _profile_task_rows(base_rows: list[dict[str, Any]], task_count: int) -> list[dict[str, Any]]:
    """Expand representative rows into deterministic prompts across all regimes."""

    if not base_rows:
        raise ValueError("at least one base task row is required")
    if task_count < 1:
        raise ValueError(f"task_count must be >= 1, got {task_count}")

    rows: list[dict[str, Any]] = []
    for task_index in range(task_count):
        base_index = task_index % len(base_rows)
        cycle = task_index // len(base_rows)
        row = json.loads(json.dumps(base_rows[base_index]))
        row["seed"] = int(row.get("seed", base_index)) + cycle * 1_000_000
        row["_profile_prompt_index"] = task_index
        rows.append(row)
    return rows


def _repeat_task_rows(prompt_rows: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
    """Create repeated responses without changing the prompt/environment seed."""

    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    rows: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompt_rows):
        for response_index in range(repeats):
            row = json.loads(json.dumps(prompt))
            row["_profile_prompt_index"] = prompt_index
            row["_profile_response_index"] = response_index
            rows.append(row)
    return rows


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
    # the persistent synthetic setpoint dynamics this deliberately unguided policy
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
# Display order for the four scripted anchors.
_ANCHOR_ORDER = ("anchor:relief", "anchor:noop", "anchor:random-valid", "anchor:catastrophic")
# This is intentionally a partial order. In a genuinely congested
# environment, random guardrail-valid control can occasionally help by
# accident, so requiring noop > random-valid would reward an inert simulator.
# Intentional relief must beat both; catastrophic rejected play must lose to
# both.
_ANCHOR_CONSTRAINTS = (
    ("anchor:relief", "anchor:noop"),
    ("anchor:relief", "anchor:random-valid"),
    ("anchor:noop", "anchor:catastrophic"),
    ("anchor:random-valid", "anchor:catastrophic"),
)


@dataclass
class ModelSpec:
    label: str
    model: str
    base_url: str
    api_key_env: str = ""
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 512
    # OpenAI-compatible servers may expose model-specific chat-template
    # controls. Keep this deliberately narrow: Run 1B only needs Qwen3's
    # documented thinking switch, and forwarding arbitrary template kwargs
    # would create an unreviewed prompt-contract surface.
    chat_template_kwargs: dict[str, bool] | None = None
    # Increasing integers encode the expected capability order. When two or
    # more models are configured, every model must have a unique rank.
    capability_rank: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or not 0.0 <= float(self.temperature) <= 2.0
        ):
            raise ValueError(f"temperature must be a finite number between 0 and 2, got {self.temperature!r}")
        if (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or not math.isfinite(float(self.top_p))
            or not 0.0 < float(self.top_p) <= 1.0
        ):
            raise ValueError(f"top_p must be a finite number greater than 0 and at most 1, got {self.top_p!r}")
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or not 1 <= self.max_tokens <= _MAX_MODEL_OUTPUT_TOKENS
        ):
            raise ValueError(
                f"max_tokens must be an integer between 1 and {_MAX_MODEL_OUTPUT_TOKENS}, got {self.max_tokens!r}"
            )
        if self.capability_rank is not None and (
            isinstance(self.capability_rank, bool)
            or not isinstance(self.capability_rank, int)
            or self.capability_rank < 1
        ):
            raise ValueError(f"capability_rank must be a positive integer when provided, got {self.capability_rank!r}")
        if self.chat_template_kwargs is not None:
            if (
                not isinstance(self.chat_template_kwargs, dict)
                or set(self.chat_template_kwargs) != {"enable_thinking"}
                or not isinstance(self.chat_template_kwargs.get("enable_thinking"), bool)
            ):
                raise ValueError("chat_template_kwargs must be omitted or exactly {'enable_thinking': <boolean>}")
            # Do not retain a caller-owned mutable mapping that could change
            # after validation and silently alter later model requests.
            self.chat_template_kwargs = {"enable_thinking": self.chat_template_kwargs["enable_thinking"]}


class _RequestCapture:
    """Capture exact prelaunch request payloads while excluding HTTP headers."""

    def __init__(
        self,
        path: Path,
        specs: list[ModelSpec],
        *,
        prompt_count: int,
        responses_per_prompt: int,
        steps_per_episode: int,
    ) -> None:
        dimensions = {
            "prompt_count": prompt_count,
            "responses_per_prompt": responses_per_prompt,
            "steps_per_episode": steps_per_episode,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"request capture {name} must be a positive integer")
        identities = [(spec.label, spec.model, spec.capability_rank) for spec in specs]
        if len(identities) != len(set(identities)):
            raise ValueError("request capture model identities must be unique")

        self.path = path.expanduser()
        if not self.path.parent.is_dir():
            raise FileNotFoundError(f"request capture receipt directory does not exist: {self.path.parent}")
        if os.path.lexists(self.path):
            raise FileExistsError(f"request capture destination already exists: {self.path}")
        # Keep incomplete work visibly separate from the final receipt. The
        # partial is exclusive-created in the destination directory so a stale
        # or concurrent attempt cannot be overwritten.
        self.partial_path = self.path.with_name(f"{self.path.name}.partial")
        self._output = self.partial_path.open("x", encoding="utf-8", newline="\n")
        self._specs = list(specs)
        self._model_indexes = {identity: index for index, identity in enumerate(identities)}
        self._prompt_count = prompt_count
        self._responses_per_prompt = responses_per_prompt
        self._steps_per_episode = steps_per_episode
        self._records: dict[tuple[int, int, int, int], dict[str, Any]] = {}
        self._finalized = False

    @staticmethod
    def _canonical_bytes(value: Any) -> bytes:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def record(
        self,
        spec: ModelSpec,
        row: dict[str, Any],
        *,
        step_idx: int,
        url: str,
        payload: dict[str, Any],
    ) -> None:
        if self._finalized:
            raise ValueError("request capture is already finalized")
        identity = (spec.label, spec.model, spec.capability_rank)
        model_index = self._model_indexes.get(identity)
        if model_index is None:
            raise ValueError(f"request capture saw an undeclared model: {identity!r}")
        prompt_index = row.get("_profile_prompt_index")
        response_index = row.get("_profile_response_index")
        coordinates = (prompt_index, response_index, step_idx)
        limits = (
            self._prompt_count,
            self._responses_per_prompt,
            self._steps_per_episode,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= limit
            for value, limit in zip(coordinates, limits, strict=True)
        ):
            raise ValueError(f"request capture saw an unexpected prompt/response/step coordinate: {coordinates!r}")
        key = (model_index, prompt_index, response_index, step_idx)
        if key in self._records:
            raise ValueError(f"duplicate request capture coordinate: {key!r}")
        parsed_url = urlsplit(url)
        if (
            parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ValueError("request capture URL must not contain credentials, a query, or a fragment")

        canonical_payload = self._canonical_bytes(payload)
        payload_copy = json.loads(canonical_payload)
        self._records[key] = {
            "schema": _REQUEST_CAPTURE_SCHEMA,
            "model": {
                "label": spec.label,
                "served_model_id": spec.model,
                "capability_rank": spec.capability_rank,
            },
            "coordinates": {
                "prompt_index": prompt_index,
                "response_index": response_index,
                "step_index": step_idx,
            },
            "url": url,
            "payload": payload_copy,
            "canonical_payload_sha256": hashlib.sha256(canonical_payload).hexdigest(),
        }

    def finalize(self) -> None:
        if self._finalized:
            raise ValueError("request capture is already finalized")
        expected = [
            (model_index, prompt_index, response_index, step_index)
            for model_index in range(len(self._specs))
            for prompt_index in range(self._prompt_count)
            for response_index in range(self._responses_per_prompt)
            for step_index in range(self._steps_per_episode)
        ]
        missing = [key for key in expected if key not in self._records]
        unexpected = sorted(set(self._records) - set(expected))
        if missing or unexpected:
            raise ValueError(f"request capture is incomplete: missing={len(missing)}, unexpected={len(unexpected)}")
        if self._output.closed:
            raise ValueError("request capture destination closed before finalization")
        if self._output.tell() != 0 or self.partial_path.stat().st_size != 0:
            raise ValueError("request capture partial changed after exclusive create")
        serialized = "".join(self._canonical_bytes(self._records[key]).decode("utf-8") + "\n" for key in expected)
        self._output.write(serialized)
        self._output.flush()
        os.fsync(self._output.fileno())
        self._output.close()
        if len(self.partial_path.read_text(encoding="utf-8").splitlines()) != len(expected):
            raise ValueError("request capture JSONL row count changed during finalization")
        # Hard-link publication is atomic and refuses to replace a final path
        # created while the smoke was running. Both names are in one directory,
        # so a successful link exposes only the already-fsynced complete inode.
        os.link(self.partial_path, self.path)
        self.partial_path.unlink()
        self._finalized = True

    def abort(self) -> None:
        """Close an unfinished capture while retaining its .partial evidence."""

        if self._output.closed:
            return
        try:
            self._output.close()
        except OSError:
            # Preserve the original sweep exception. The clearly named partial
            # remains fail-closed even if the filesystem also rejects close.
            pass


def _require_ranked_models(specs: list[ModelSpec], profile_name: str) -> None:
    """Prevent a named real-model profile from passing on anchors alone."""

    if len(specs) < 2:
        raise ValueError(f"{profile_name} requires at least two real models")
    if any(not spec.label.strip() or not spec.model.strip() or not spec.base_url.strip() for spec in specs):
        raise ValueError(f"{profile_name} requires non-empty label, model, and base_url values")
    identities = [spec.model.strip() for spec in specs]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{profile_name} requires distinct model identities")
    if any(spec.capability_rank is None for spec in specs):
        raise ValueError(f"{profile_name} requires capability_rank for every model")
    ranks = [int(spec.capability_rank) for spec in specs if spec.capability_rank is not None]
    if len(ranks) != len(set(ranks)):
        raise ValueError(f"{profile_name} requires unique capability_rank values")


def _require_compliance_models(specs: list[ModelSpec]) -> None:
    _require_ranked_models(specs, "--compliance-profile")


def _require_run1b_sampling(specs: list[ModelSpec]) -> None:
    expected = (_RUN1B_TEMPERATURE, _RUN1B_TOP_P, _RUN1B_MAX_TOKENS)
    for spec in specs:
        if spec.api_key_env:
            raise ValueError(
                "Run 1B named profiles require unauthenticated local loopback model endpoints; "
                f"model {spec.label!r} declared api_key_env={spec.api_key_env!r}"
            )
        endpoint = _RUN1B_LOCAL_BASE_URL.fullmatch(spec.base_url)
        if endpoint is None or int(endpoint.group(1)) > 65_535:
            raise ValueError(
                "Run 1B named profiles require base_url exactly "
                f"http://127.0.0.1:<port>/v1 with port 1-65535; model {spec.label!r} declared {spec.base_url!r}"
            )
        observed = (float(spec.temperature), float(spec.top_p), spec.max_tokens)
        if observed != expected:
            raise ValueError(
                "Run 1B requires the frozen sampling contract "
                f"temperature={expected[0]}, top_p={expected[1]}, max_tokens={expected[2]}; "
                f"model {spec.label!r} declared {observed}"
            )
        if spec.chat_template_kwargs != {"enable_thinking": False}:
            raise ValueError(
                "Run 1B requires chat_template_kwargs exactly "
                "{'enable_thinking': false}; "
                f"model {spec.label!r} declared {spec.chat_template_kwargs!r}"
            )


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    """Population Pearson correlation, or None when either side is constant."""

    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    dx = [value - mean_x for value in xs]
    dy = [value - mean_y for value in ys]
    denominator = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denominator == 0.0:
        return None
    return sum(x * y for x, y in zip(dx, dy, strict=True)) / denominator


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _parse_pair_key(pair_key: Any, path: str) -> tuple[int, int, str]:
    if not isinstance(pair_key, str) or not _PAIR_KEY.fullmatch(pair_key):
        raise ValueError(f"{path}.pair_key must be '<prompt_index>:<response_index>'")
    prompt_text, response_text = pair_key.split(":", 1)
    return int(prompt_text), int(response_text), pair_key


def _record_pair_coordinates(record: Mapping[str, Any], path: str = "episode record") -> tuple[int, int, str]:
    """Return validated prompt, response, and regime identities for one episode."""

    prompt_index, response_index, pair_key = _parse_pair_key(record.get("pair_key"), path)
    explicit_prompt_index = _nonnegative_int(record.get("prompt_index"), f"{path}.prompt_index")
    explicit_response_index = _nonnegative_int(record.get("response_index"), f"{path}.response_index")
    if explicit_prompt_index != prompt_index or explicit_response_index != response_index:
        raise ValueError(f"{path} prompt/response indexes disagree with pair_key {pair_key!r}")
    scenario_id = record.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError(f"{path}.scenario_id must be a non-empty string")
    return prompt_index, response_index, scenario_id


def _validate_pair_manifest(expected_pair_manifest: Mapping[str, str], *, expected_episodes: int) -> None:
    if not isinstance(expected_pair_manifest, Mapping) or len(expected_pair_manifest) != expected_episodes:
        raise ValueError("expected_pair_manifest size must equal expected_episodes")
    coordinates: set[tuple[int, int]] = set()
    scenarios_by_prompt: dict[int, set[str]] = {}
    for pair_key, scenario_id in expected_pair_manifest.items():
        prompt_index, response_index, _ = _parse_pair_key(pair_key, "expected_pair_manifest")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError(f"expected_pair_manifest[{pair_key!r}] must name a non-empty scenario")
        coordinates.add((prompt_index, response_index))
        scenarios_by_prompt.setdefault(prompt_index, set()).add(scenario_id)
    if any(len(scenarios) != 1 for scenarios in scenarios_by_prompt.values()):
        raise ValueError("each prompt in expected_pair_manifest must map to exactly one scenario")
    prompt_count = max(scenarios_by_prompt, default=-1) + 1
    response_count = max((response for _prompt, response in coordinates), default=-1) + 1
    expected_coordinates = {
        (prompt_index, response_index)
        for prompt_index in range(prompt_count)
        for response_index in range(response_count)
    }
    if coordinates != expected_coordinates:
        raise ValueError("expected_pair_manifest must describe contiguous Cartesian prompt/repeat support")


def _validated_episode_records(
    row: Mapping[str, Any],
    *,
    label: str,
    expected_pair_manifest: Mapping[str, str],
    require_complete_support: bool,
) -> dict[str, dict[str, Any]]:
    """Validate one policy row before any engineering or quality gate uses it."""

    episodes = _nonnegative_int(row.get("episodes"), f"{label}.episodes")
    infra_errors = _nonnegative_int(row.get("infra_errors"), f"{label}.infra_errors")
    records = row.get("episode_records")
    if not isinstance(records, list) or len(records) != episodes:
        raise ValueError(f"{label}.episode_records must contain exactly {episodes} records")

    indexed: dict[str, dict[str, Any]] = {}
    coordinates: set[tuple[int, int]] = set()
    parse_failures = invalid_calls = usable_episodes = 0
    for record_index, record in enumerate(records):
        path = f"{label}.episode_records[{record_index}]"
        if not isinstance(record, dict):
            raise ValueError(f"{path} must be an object")
        prompt_index, response_index, scenario_id = _record_pair_coordinates(record, path)
        pair_key = str(record["pair_key"])
        coordinate = (prompt_index, response_index)
        if pair_key in indexed or coordinate in coordinates:
            raise ValueError(f"{label} has duplicate prompt/response coordinates at {pair_key!r}")
        if pair_key not in expected_pair_manifest:
            raise ValueError(f"{path}.pair_key is outside planned prompt/repeat support")
        if scenario_id != expected_pair_manifest[pair_key]:
            raise ValueError(f"{path}.scenario_id disagrees with the task manifest")

        episode_return = record.get("return")
        if (
            isinstance(episode_return, bool)
            or not isinstance(episode_return, (int, float))
            or not math.isfinite(float(episode_return))
        ):
            raise ValueError(f"{path}.return must be a finite number")
        record_parse = _nonnegative_int(record.get("parse_failures"), f"{path}.parse_failures")
        record_invalid = _nonnegative_int(record.get("invalid_calls"), f"{path}.invalid_calls")
        usable = record.get("usable")
        if not isinstance(usable, bool) or usable != (record_parse == 0 and record_invalid == 0):
            raise ValueError(f"{path}.usable disagrees with parse/invalid failures")

        parse_failures += record_parse
        invalid_calls += record_invalid
        usable_episodes += int(usable)
        indexed[pair_key] = record
        coordinates.add(coordinate)

    expected_keys = set(expected_pair_manifest)
    observed_keys = set(indexed)
    if not observed_keys <= expected_keys:
        raise ValueError(f"{label}.episode_records contain pair keys outside planned support")
    if require_complete_support and observed_keys != expected_keys:
        raise ValueError(f"{label}.episode_records do not cover the exact planned prompt/repeat support")
    if len(expected_keys - observed_keys) != infra_errors:
        raise ValueError(f"{label}.infra_errors disagree with missing planned prompt/repeat records")
    if parse_failures != _nonnegative_int(row.get("parse_failures"), f"{label}.parse_failures"):
        raise ValueError(f"{label}.parse_failures disagree with episode records")
    if invalid_calls != _nonnegative_int(row.get("invalid_calls"), f"{label}.invalid_calls"):
        raise ValueError(f"{label}.invalid_calls disagree with episode records")
    if usable_episodes != _nonnegative_int(row.get("usable_episodes"), f"{label}.usable_episodes"):
        raise ValueError(f"{label}.usable_episodes disagree with episode records")
    mean_return = row.get("mean_return")
    if records:
        if (
            isinstance(mean_return, bool)
            or not isinstance(mean_return, (int, float))
            or not math.isfinite(float(mean_return))
        ):
            raise ValueError(f"{label}.mean_return must be a finite number when episodes are present")
        record_mean = statistics.fmean(float(record["return"]) for record in records)
        # PolicyStats.row serializes this aggregate to four decimal places;
        # accept either the raw mean or that representation, but not a row
        # that contradicts its episode records.
        if not math.isclose(float(mean_return), record_mean, rel_tol=0.0, abs_tol=0.000_050_000_001):
            raise ValueError(f"{label}.mean_return disagrees with episode records")
    elif mean_return is not None:
        raise ValueError(f"{label}.mean_return must be null when no episodes are present")
    return indexed


def _clustered_paired_summary(
    paired_deltas: list[tuple[int, str, float]],
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    """Summarize paired deltas using regime-stratified prompt clusters."""

    if not paired_deltas:
        raise ValueError("at least one paired delta is required")
    if draws < 1:
        raise ValueError("draws must be positive")

    clusters: dict[tuple[str, int], list[float]] = {}
    for prompt_index, scenario_id, delta in paired_deltas:
        value = float(delta)
        if not math.isfinite(value):
            raise ValueError("paired deltas must be finite")
        clusters.setdefault((scenario_id, prompt_index), []).append(value)

    cluster_means = {key: statistics.fmean(values) for key, values in clusters.items()}
    strata: dict[str, list[float]] = {}
    for (scenario_id, _prompt_index), value in sorted(cluster_means.items()):
        strata.setdefault(scenario_id, []).append(value)

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(draws, dtype=float)
    prompt_count = len(cluster_means)
    chunk_size = max(1, min(256, 2_000_000 // prompt_count))
    for start in range(0, draws, chunk_size):
        count = min(chunk_size, draws - start)
        sampled_sum = np.zeros(count, dtype=float)
        for scenario_id in sorted(strata):
            values = np.asarray(strata[scenario_id], dtype=float)
            indices = rng.integers(0, len(values), size=(count, len(values)))
            sampled_sum += values[indices].sum(axis=1)
        bootstrap_means[start : start + count] = sampled_sum / prompt_count

    ordered_cluster_means = list(cluster_means.values())
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "bootstrap_method": _BOOTSTRAP_METHOD,
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
        "prompt_clusters": prompt_count,
        "response_pairs": len(paired_deltas),
        "mean_delta": statistics.fmean(ordered_cluster_means),
        "median_delta": statistics.median(ordered_cluster_means),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "prompt_wins": sum(value > 0.0 for value in ordered_cluster_means),
        "prompt_ties": sum(value == 0.0 for value in ordered_cluster_means),
        "prompt_losses": sum(value < 0.0 for value in ordered_cluster_means),
    }


@dataclass
class PolicyStats:
    returns: list[float] = field(default_factory=list)
    steps: int = 0
    rejected: int = 0
    noop_steps: int = 0
    invalid_calls: int = 0
    parse_failures: int = 0
    infra_errors: int = 0
    tool_rewards: dict[str, list[float]] = field(default_factory=dict)
    tool_rejections: Counter[str] = field(default_factory=Counter)
    episode_records: list[dict[str, Any]] = field(default_factory=list)

    def record_step(self, tool_name: str, reward: float, *, rejected: bool) -> None:
        self.steps += 1
        self.tool_rewards.setdefault(tool_name, []).append(reward)
        if rejected:
            self.rejected += 1
            self.tool_rejections[tool_name] += 1
        if tool_name == "noop":
            self.noop_steps += 1

    def finish_episode(
        self,
        episode_return: float,
        *,
        scenario_id: str,
        tool_counts: Counter[str],
        rejected_steps: int,
        episode_steps: int,
        pair_key: str,
        parse_failures: int,
        invalid_calls: int,
    ) -> None:
        usable = parse_failures == 0 and invalid_calls == 0
        prompt_index, response_index, _ = _parse_pair_key(pair_key, "episode")
        self.returns.append(episode_return)
        self.episode_records.append(
            {
                "return": episode_return,
                "pair_key": pair_key,
                "prompt_index": prompt_index,
                "response_index": response_index,
                "usable": usable,
                "parse_failures": parse_failures,
                "invalid_calls": invalid_calls,
                "scenario_id": scenario_id,
                "tool_counts": dict(tool_counts),
                "rejection_rate": rejected_steps / episode_steps if episode_steps else 0.0,
                "noop_rate": tool_counts.get("noop", 0) / episode_steps if episode_steps else 0.0,
                "steps": episode_steps,
            }
        )

    def row(self, label: str) -> dict[str, Any]:
        std = statistics.pstdev(self.returns) if len(self.returns) > 1 else 0.0
        distribution = {
            "min": min(self.returns) if self.returns else None,
            "p05": _quantile(self.returns, 0.05),
            "p25": _quantile(self.returns, 0.25),
            "median": _quantile(self.returns, 0.50),
            "p75": _quantile(self.returns, 0.75),
            "p95": _quantile(self.returns, 0.95),
            "max": max(self.returns) if self.returns else None,
        }
        tool_metrics = {}
        for tool_name, rewards in sorted(self.tool_rewards.items()):
            calls = len(rewards)
            tool_metrics[tool_name] = {
                "calls": calls,
                "call_rate": round(calls / self.steps, 6) if self.steps else 0.0,
                "mean_step_reward": round(statistics.fmean(rewards), 6),
                "rejection_rate": round(self.tool_rejections[tool_name] / calls, 6),
            }

        episode_returns = [record["return"] for record in self.episode_records]
        tools = sorted({tool for record in self.episode_records for tool in record["tool_counts"]})
        correlations: dict[str, float | None] = {
            "rejection_rate": _correlation(
                [record["rejection_rate"] for record in self.episode_records], episode_returns
            ),
            "noop_rate": _correlation([record["noop_rate"] for record in self.episode_records], episode_returns),
        }
        for tool_name in tools:
            correlations[f"tool_rate:{tool_name}"] = _correlation(
                [
                    record["tool_counts"].get(tool_name, 0) / record["steps"] if record["steps"] else 0.0
                    for record in self.episode_records
                ],
                episode_returns,
            )

        returns_by_scenario = {}
        scenarios = sorted({record["scenario_id"] for record in self.episode_records})
        for scenario_id in scenarios:
            values = [record["return"] for record in self.episode_records if record["scenario_id"] == scenario_id]
            returns_by_scenario[scenario_id] = {
                "episodes": len(values),
                "mean_return": round(statistics.fmean(values), 6),
                "std_return": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
            }

        return {
            "policy": label,
            "episodes": len(self.returns),
            "mean_return": round(statistics.fmean(self.returns), 4) if self.returns else None,
            "std_return": round(std, 4),
            "return_distribution": {
                key: round(value, 6) if value is not None else None for key, value in distribution.items()
            },
            "rejection_rate": round(self.rejected / self.steps, 4) if self.steps else 0.0,
            "noop_rate": round(self.noop_steps / self.steps, 4) if self.steps else 0.0,
            "invalid_calls": self.invalid_calls,
            "parse_failures": self.parse_failures,
            "infra_errors": self.infra_errors,
            "usable_episodes": sum(bool(record["usable"]) for record in self.episode_records),
            "episode_records": list(self.episode_records),
            "tool_metrics": tool_metrics,
            "episode_return_correlations": {
                key: round(value, 6) if value is not None else None for key, value in correlations.items()
            },
            "returns_by_scenario": returns_by_scenario,
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
        # Preserve the environment's exactly-one-call contract. Selecting the
        # first item would silently turn a multi-call protocol violation into
        # a valid action and inflate that model's compliance profile.
        if len(calls) != 1:
            return None
        fn = calls[0].get("function", {})
        try:
            arguments = _strict_json_object(fn.get("arguments") or "{}")
        except (TypeError, ValueError):
            return None
        if not isinstance(fn.get("name"), str) or not isinstance(arguments, dict):
            return None
        return {"name": fn["name"], "arguments": arguments}
    # Gated profiles qualify the native endpoint/parser path, not a model that
    # merely prints JSON-looking prose.  Content-only output is therefore a
    # parse failure even when it contains an action-shaped object.
    return None


def _request_seed(row: dict[str, Any], step_idx: int) -> int:
    """Derive a provider-safe seed from the paired request coordinates."""

    prompt_index = int(row["_profile_prompt_index"])
    response_index = int(row["_profile_response_index"])
    if prompt_index < 0 or response_index < 0 or step_idx < 0:
        raise ValueError("prompt, response, and step indices must be non-negative")
    material = f"{_REQUEST_SEED_VERSION}:{prompt_index}:{response_index}:{step_idx}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31)


async def _llm_action(
    session: aiohttp.ClientSession,
    spec: ModelSpec,
    row: dict,
    observation: str,
    step_idx: int,
    *,
    request_capture: _RequestCapture | None = None,
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
        # The environment contract requires exactly one tool call every turn.
        # Enforce that protocol at the model endpoint as well as validating the
        # returned call; otherwise an API may legitimately emit prose under
        # its default "auto" policy and turn endpoint defaults into apparent
        # model-capability failures.
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "max_tokens": spec.max_tokens,
        "seed": _request_seed(row, step_idx),
    }
    if spec.chat_template_kwargs is not None:
        payload["chat_template_kwargs"] = dict(spec.chat_template_kwargs)
    url = f"{spec.base_url.rstrip('/')}/chat/completions"
    if request_capture is not None:
        request_capture.record(
            spec,
            row,
            step_idx=step_idx,
            url=url,
            payload=payload,
        )
    async with session.post(
        url,
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


def _unparseable_response(step_idx: int) -> dict[str, Any]:
    """Build a model response with no tool call so the server applies its protocol penalty."""

    return NeMoGymResponse(
        output=[
            NeMoGymResponseOutputMessage(
                id=f"msg_{step_idx}",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[],
                        text="Model response did not contain a parseable tool call.",
                        type="output_text",
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        id="r",
        created_at=0.0,
        model="capability-sweep",
        object="response",
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    ).model_dump()


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

        async def step_env(action: dict[str, Any] | None, step_idx: int) -> dict:
            response = (
                _unparseable_response(step_idx)
                if action is None
                else _tool_response(action["name"], action["arguments"], step_idx)
            )
            return await post(
                "/step",
                {
                    "responses_create_params": row["responses_create_params"],
                    "response": response,
                },
            )

        reset = await post("/reset", row)
        observation = reset["observation"]
        episode_return, step_idx = 0.0, 0
        terminated = truncated = False
        tool_counts: Counter[str] = Counter()
        episode_rejected = 0
        episode_parse_failures = 0
        episode_invalid_calls = 0
        # The env terminates at the row's max_steps; the 4x margin only guards
        # against a served instance that never sets terminated/truncated.
        step_cap = 4 * int(row.get("max_steps", 16))
        try:
            while not (terminated or truncated) and step_idx < step_cap:
                action = await action_fn(observation, step_idx)
                if action is None:
                    stats.parse_failures += 1
                    episode_parse_failures += 1
                step = await step_env(action, step_idx)
                reward = float(step["reward"])
                episode_return += reward
                tool_name = action["name"] if action is not None else "<parse_failure>"
                rejected = not step["info"].get("guardrail_accepted", True)
                stats.record_step(tool_name, reward, rejected=rejected)
                tool_counts[tool_name] += 1
                episode_rejected += int(rejected)
                if step["info"].get("error") == "invalid_tool_call":
                    stats.invalid_calls += 1
                    episode_invalid_calls += 1
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
        stats.finish_episode(
            episode_return,
            scenario_id=str(row.get("scenario_id") or "unspecified"),
            tool_counts=tool_counts,
            rejected_steps=episode_rejected,
            episode_steps=sum(tool_counts.values()),
            pair_key=(f"{row.get('_profile_prompt_index', row.get('seed'))}:{row.get('_profile_response_index', 0)}"),
            parse_failures=episode_parse_failures,
            invalid_calls=episode_invalid_calls,
        )


async def _scripted_action(
    policy: Callable[[str, int, random.Random], dict[str, Any]],
    rng: random.Random,
    observation: str,
    step_idx: int,
) -> dict[str, Any]:
    return policy(observation, step_idx, rng)


def _evaluate_model_ordering(
    profile: list[dict[str, Any]],
    specs: list[ModelSpec],
    *,
    expected_episodes: int,
    expected_pair_manifest: Mapping[str, str],
    compliance: bool = False,
    failure_rate_ceiling: float = 0.0,
) -> dict[str, Any]:
    """Evaluate usable, paired small-to-frontier return improvements."""

    if not math.isfinite(failure_rate_ceiling) or not 0.0 <= failure_rate_ceiling <= 1.0:
        raise ValueError(f"failure_rate_ceiling must be a finite number between 0 and 1, got {failure_rate_ceiling!r}")
    effective_failure_ceiling = 0.0 if compliance else failure_rate_ceiling
    _validate_pair_manifest(expected_pair_manifest, expected_episodes=expected_episodes)

    if len(specs) < 2:
        return {"status": "NOT_CONFIGURED", "expected": [], "observed": {}, "reason": "fewer than two models"}
    if any(spec.capability_rank is None for spec in specs):
        return {
            "status": "NOT_EVALUABLE",
            "expected": [],
            "observed": {},
            "reason": "every model needs capability_rank",
        }

    ordered_specs = sorted(specs, key=lambda spec: int(spec.capability_rank or 0))
    expected = [f"model:{spec.label}" for spec in ordered_specs]
    by_policy = {row["policy"]: row for row in profile}
    if any(label not in by_policy for label in expected):
        return {"status": "NOT_EVALUABLE", "expected": expected, "observed": {}, "reason": "missing model row"}

    observed = {label: by_policy[label]["mean_return"] for label in expected}
    all_records_by_label: dict[str, dict[str, dict[str, Any]]] = {}
    for label in expected:
        try:
            all_records_by_label[label] = _validated_episode_records(
                by_policy[label],
                label=label,
                expected_pair_manifest=expected_pair_manifest,
                require_complete_support=compliance,
            )
        except ValueError as error:
            return {
                "status": "NOT_EVALUABLE",
                "expected": expected,
                "observed": observed,
                "reason": str(error),
            }

    complete = all(
        by_policy[label]["episodes"] + by_policy[label]["infra_errors"] == expected_episodes
        and by_policy[label]["mean_return"] is not None
        for label in expected
    )
    if not complete:
        return {
            "status": "NOT_EVALUABLE",
            "expected": expected,
            "observed": observed,
            "reason": "one or more model profiles are incomplete",
        }

    failure_counts = {
        label: {
            "parse_failures": by_policy[label]["parse_failures"],
            "invalid_calls": by_policy[label]["invalid_calls"],
            "infra_errors": by_policy[label]["infra_errors"],
        }
        for label in expected
    }
    failure_rates = {
        label: min(1.0, sum(counts.values()) / expected_episodes) for label, counts in failure_counts.items()
    }
    if any(rate > effective_failure_ceiling for rate in failure_rates.values()):
        return {
            "status": "NOT_EVALUABLE",
            "expected": expected,
            "observed": observed,
            "failure_counts": failure_counts,
            "failure_rates": failure_rates,
            "failure_rate_ceiling": effective_failure_ceiling,
            "reason": (
                "one or more model profiles have parse, invalid-call, or "
                "infrastructure failures above the configured "
                f"failure-rate ceiling ({effective_failure_ceiling:.3f})"
            ),
        }

    paired_records: dict[str, dict[str, dict[str, Any]]] = {}
    paired_returns: dict[str, dict[str, float]] = {}
    for label in expected:
        paired_records[label] = {
            pair_key: record for pair_key, record in all_records_by_label[label].items() if record["usable"] is True
        }
        paired_returns[label] = {
            pair_key: float(record["return"]) for pair_key, record in paired_records[label].items()
        }

    reference_keys = set.intersection(*(set(paired_returns[label]) for label in expected))
    if not reference_keys:
        return {
            "status": "NOT_EVALUABLE",
            "expected": expected,
            "observed": observed,
            "failure_counts": failure_counts,
            "failure_rates": failure_rates,
            "failure_rate_ceiling": effective_failure_ceiling,
            "reason": "model profiles have no common usable prompt/repeat pairs",
        }
    if compliance and len(reference_keys) != expected_episodes:
        return {
            "status": "NOT_EVALUABLE",
            "expected": expected,
            "observed": observed,
            "failure_counts": failure_counts,
            "failure_rates": failure_rates,
            "failure_rate_ceiling": effective_failure_ceiling,
            "reason": "compliance mode requires every prompt/repeat pair to be usable",
        }
    comparisons: list[dict[str, Any]] = []
    passed = True
    sorted_keys = sorted(reference_keys)
    pair_coordinates: dict[str, tuple[int, int, str]] = {}
    for pair_key in sorted_keys:
        coordinates = {_record_pair_coordinates(paired_records[label][pair_key]) for label in expected}
        if len(coordinates) != 1:
            return {
                "status": "NOT_EVALUABLE",
                "expected": expected,
                "observed": observed,
                "failure_counts": failure_counts,
                "failure_rates": failure_rates,
                "failure_rate_ceiling": effective_failure_ceiling,
                "reason": f"model profiles disagree on prompt, response, or regime identity for pair {pair_key!r}",
            }
        pair_coordinates[pair_key] = coordinates.pop()

    prompt_clusters_by_scenario: dict[str, set[int]] = {
        scenario_id: set() for scenario_id in set(expected_pair_manifest.values())
    }
    for prompt_index, _response_index, scenario_id in pair_coordinates.values():
        prompt_clusters_by_scenario.setdefault(scenario_id, set()).add(prompt_index)
    underpowered_scenarios = sorted(
        scenario_id for scenario_id, prompt_indices in prompt_clusters_by_scenario.items() if len(prompt_indices) < 2
    )
    if underpowered_scenarios:
        return {
            "status": "NOT_EVALUABLE",
            "engineering_status": "PASS",
            "quality_status": "NOT_EVALUABLE",
            "expected": expected,
            "observed": observed,
            "failure_counts": failure_counts,
            "failure_rates": failure_rates,
            "failure_rate_ceiling": effective_failure_ceiling,
            "valid_paired_episodes": len(reference_keys),
            "comparisons": [],
            "reason": (
                "model-quality inference requires at least two prompt clusters per scenario; "
                f"underpowered scenarios: {underpowered_scenarios}"
            ),
        }

    bootstrap_draws = _COMPLIANCE_BOOTSTRAP_DRAWS if compliance else _DEFAULT_BOOTSTRAP_DRAWS
    for comparison_index, (weaker, stronger) in enumerate(zip(expected, expected[1:], strict=False)):
        paired_deltas = [
            (
                pair_coordinates[key][0],
                pair_coordinates[key][2],
                paired_returns[stronger][key] - paired_returns[weaker][key],
            )
            for key in sorted_keys
        ]
        summary = _clustered_paired_summary(
            paired_deltas,
            seed=comparison_index,
            draws=bootstrap_draws,
        )
        comparison_passed = summary["mean_delta"] > 0.0 and summary["ci95_low"] > 0.0
        passed = passed and comparison_passed
        comparisons.append(
            {
                "weaker": weaker,
                "stronger": stronger,
                "pairs": summary["response_pairs"],
                "paired_episodes": summary["response_pairs"],
                **summary,
                "status": "PASS" if comparison_passed else "FAIL",
            }
        )

    return {
        "status": "PASS" if passed else "FAIL",
        "engineering_status": "PASS",
        "quality_status": "PASS" if passed else "FAIL",
        "expected": expected,
        "observed": observed,
        "failure_counts": failure_counts,
        "failure_rates": failure_rates,
        "failure_rate_ceiling": effective_failure_ceiling,
        "valid_paired_episodes": len(reference_keys),
        "comparisons": comparisons,
        "reason": (
            None if passed else "paired 95% bootstrap interval did not show a positive adjacent-model improvement"
        ),
    }


async def _sweep(
    task_rows: list[dict[str, Any]],
    repeats: int,
    specs: list[ModelSpec],
    concurrency: int = 8,
    *,
    compliance_profile: bool = False,
    run1b_profile: bool = False,
    failure_rate_ceiling: float = 0.0,
    request_capture_out: Path | None = None,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if not task_rows:
        raise ValueError("at least one task row is required")
    if not 1 <= concurrency <= 32:
        raise ValueError(f"concurrency must be between 1 and 32, got {concurrency}")
    if not math.isfinite(failure_rate_ceiling) or not 0.0 <= failure_rate_ceiling <= 1.0:
        raise ValueError(f"failure_rate_ceiling must be a finite number between 0 and 1, got {failure_rate_ceiling!r}")
    if compliance_profile and failure_rate_ceiling != 0.0:
        raise ValueError("compliance profiles require failure_rate_ceiling=0")
    if compliance_profile and request_capture_out is not None:
        raise ValueError(
            "request capture is forbidden for compliance profiles; "
            "bind the passed prelaunch capture receipts externally"
        )
    if compliance_profile:
        run1b_profile = True
    labels = [spec.label for spec in specs]
    if len(labels) != len(set(labels)):
        raise ValueError(
            f"duplicate model labels in --models: {sorted(set(x for x in labels if labels.count(x) > 1))}"
        )
    for spec in specs:
        if spec.api_key_env and not os.environ.get(spec.api_key_env):
            raise ValueError(f"model '{spec.label}': environment variable {spec.api_key_env} is not set")
    ranks = [spec.capability_rank for spec in specs if spec.capability_rank is not None]
    if ranks and len(ranks) != len(specs):
        raise ValueError("every model must declare capability_rank when sweeping two or more models")
    if len(ranks) != len(set(ranks)):
        raise ValueError("model capability_rank values must be unique")
    sampling_contracts = {
        (
            float(spec.temperature),
            float(spec.top_p),
            spec.max_tokens,
            json.dumps(spec.chat_template_kwargs, allow_nan=False, sort_keys=True),
        )
        for spec in specs
    }
    if len(sampling_contracts) > 1:
        raise ValueError(
            "all models must use identical sampling and chat-template settings: "
            "temperature, top_p, max_tokens, and chat_template_kwargs"
        )
    if run1b_profile:
        _require_run1b_sampling(specs)

    expected_pair_manifest = {
        f"{prompt_index}:{response_index}": str(task_row.get("scenario_id") or "")
        for prompt_index, task_row in enumerate(task_rows)
        for response_index in range(repeats)
    }
    _validate_pair_manifest(expected_pair_manifest, expected_episodes=len(task_rows) * repeats)
    request_capture = None
    if request_capture_out is not None:
        if not specs:
            raise ValueError("request capture requires at least one model")
        bad_step_rows = [
            index
            for index, row in enumerate(task_rows)
            if isinstance(row.get("max_steps", 16), bool)
            or not isinstance(row.get("max_steps", 16), int)
            or row.get("max_steps", 16) != _RUN1B_CAPTURE_STEPS
        ]
        if bad_step_rows:
            raise ValueError(
                f"request capture requires max_steps=16 for every task row; invalid row indexes: {bad_step_rows}"
            )
        request_capture = _RequestCapture(
            request_capture_out,
            specs,
            prompt_count=len(task_rows),
            responses_per_prompt=repeats,
            steps_per_episode=_RUN1B_CAPTURE_STEPS,
        )

    try:
        base_url = _start_local_server()
        rows = _repeat_task_rows(task_rows, repeats)
        results: dict[str, PolicyStats] = {}
        semaphore = asyncio.Semaphore(concurrency)

        for label, factory in _ANCHORS.items():
            stats = results[label] = PolicyStats()

            async def run_anchor(row: dict[str, Any]) -> None:
                rng_seed = int(row["seed"]) * 1_000_003 + int(row["_profile_response_index"])
                action_fn = functools.partial(_scripted_action, factory(), random.Random(rng_seed))
                async with semaphore:
                    await _run_episode(base_url, row, action_fn, stats)

            await asyncio.gather(*(run_anchor(row) for row in rows))

        for spec in specs:
            stats = results[f"model:{spec.label}"] = PolicyStats()

            async def run_model(row: dict[str, Any], llm_session: aiohttp.ClientSession) -> None:
                async with semaphore:
                    action_fn = functools.partial(
                        _llm_action,
                        llm_session,
                        spec,
                        row,
                        request_capture=request_capture,
                    )
                    try:
                        await _run_episode(base_url, row, action_fn, stats)
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        stats.infra_errors += 1
                        print(f"model:{spec.label} seed={row['seed']}: episode dropped ({type(e).__name__}: {e})")

            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=concurrency)) as llm_session:
                await asyncio.gather(*(run_model(row, llm_session) for row in rows))

        if request_capture is not None:
            request_capture.finalize()
    except BaseException:
        if request_capture is not None:
            request_capture.abort()
        raise

    relief_mean = results["anchor:relief"].row("anchor:relief")["mean_return"]
    if relief_mean is None:
        raise RuntimeError("relief anchor produced no completed episodes")
    table = []
    for label, stats in results.items():
        entry = stats.row(label)
        entry["vs_relief"] = round(entry["mean_return"] - relief_mean, 4) if entry["mean_return"] is not None else None
        table.append(entry)
    table.sort(key=lambda r: r["mean_return"] if r["mean_return"] is not None else -math.inf, reverse=True)

    anchor_rows = {label: results[label].row(label) for label in _ANCHOR_ORDER}
    anchor_records = {}
    for label, row in anchor_rows.items():
        anchor_records[label] = _validated_episode_records(
            row,
            label=label,
            expected_pair_manifest=expected_pair_manifest,
            require_complete_support=True,
        )
    anchor_engineering_ok = all(
        row["infra_errors"] == 0
        and row["parse_failures"] == 0
        and row["invalid_calls"] == 0
        and row["usable_episodes"] == len(rows)
        for row in anchor_rows.values()
    )
    anchor_comparisons: list[dict[str, Any]] = []
    bootstrap_draws = _COMPLIANCE_BOOTSTRAP_DRAWS if compliance_profile else _DEFAULT_BOOTSTRAP_DRAWS
    for comparison_index, (better, worse) in enumerate(_ANCHOR_CONSTRAINTS):
        pair_keys = sorted(set(anchor_records[better]) & set(anchor_records[worse]))
        paired_deltas = [
            (
                _record_pair_coordinates(anchor_records[better][key])[0],
                _record_pair_coordinates(anchor_records[better][key])[2],
                float(anchor_records[better][key]["return"]) - float(anchor_records[worse][key]["return"]),
            )
            for key in pair_keys
        ]
        summary = _clustered_paired_summary(
            paired_deltas,
            seed=10_000 + comparison_index,
            draws=bootstrap_draws,
        )
        passed = summary["mean_delta"] > 0.0 and summary["ci95_low"] > 0.0
        anchor_comparisons.append(
            {
                "better": better,
                "worse": worse,
                "pairs": summary["response_pairs"],
                "paired_episodes": summary["response_pairs"],
                **summary,
                "status": "PASS" if passed else "FAIL",
            }
        )
    ordered = anchor_engineering_ok and all(comparison["status"] == "PASS" for comparison in anchor_comparisons)
    model_ordering = _evaluate_model_ordering(
        table,
        specs,
        expected_episodes=len(rows),
        expected_pair_manifest=expected_pair_manifest,
        compliance=compliance_profile,
        failure_rate_ceiling=failure_rate_ceiling,
    )
    sampling_contract = None
    if specs:
        sampling_contract = {
            "temperature": float(specs[0].temperature),
            "top_p": float(specs[0].top_p),
            "max_tokens": specs[0].max_tokens,
            "seed_derivation": _REQUEST_SEED_DERIVATION,
            "seed_version": _REQUEST_SEED_VERSION,
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "chat_template_kwargs": (
                dict(specs[0].chat_template_kwargs) if specs[0].chat_template_kwargs is not None else None
            ),
        }
    return {
        "backend": "replay",
        "model_specs": [
            {
                "label": spec.label,
                "model": spec.model,
                "capability_rank": int(spec.capability_rank) if spec.capability_rank is not None else None,
            }
            for spec in specs
        ],
        "sampling_contract": sampling_contract,
        "tasks": [
            {
                "prompt_index": row.get("_profile_prompt_index"),
                "seed": row.get("seed"),
                "difficulty": row.get("difficulty"),
                "scenario_id": row.get("scenario_id"),
                "regime_mix": json.loads(json.dumps(row.get("regime_mix"))),
            }
            for row in task_rows
        ],
        "prompts": len(task_rows),
        "responses_per_prompt": repeats,
        "episodes_per_policy": len(rows),
        "concurrency": concurrency,
        "compliance_profile": compliance_profile,
        "failure_rate_ceiling": 0.0 if compliance_profile else failure_rate_ceiling,
        "profile": table,
        "anchor_ordering_ok": ordered,
        "anchor_order_expected": list(_ANCHOR_ORDER),
        "anchor_order_constraints": [list(pair) for pair in _ANCHOR_CONSTRAINTS],
        "anchor_ordering_comparisons": anchor_comparisons,
        "model_ordering": model_ordering,
        "model_ordering_ok": (
            model_ordering["status"] == "PASS" if model_ordering["status"] in {"PASS", "FAIL"} else None
        ),
    }


def _print_report(report: dict[str, Any]) -> None:
    print(
        f"\ncapability sweep: {report['episodes_per_policy']} episodes/policy over "
        f"{report['prompts']} prompts x {report['responses_per_prompt']} responses\n"
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
    print(
        f"\nanchor partial order (relief > noop, relief > random-valid, noop/random-valid > catastrophic): {verdict}"
    )
    if not report["anchor_ordering_ok"]:
        print("a broken anchor ordering means reward attribution is suspect -- investigate before training.")
    model_ordering = report["model_ordering"]
    if model_ordering["status"] != "NOT_CONFIGURED":
        print(
            f"declared model capability ordering: {model_ordering['status']} "
            f"({model_ordering.get('reason') or 'strictly increasing mean return'})"
        )


def _model_gate_satisfied(model_ordering: Mapping[str, Any], *, configured_models: int) -> bool:
    if configured_models < 2:
        return True
    if model_ordering.get("status") == "PASS":
        return True
    return (
        model_ordering.get("status") == "NOT_EVALUABLE"
        and model_ordering.get("engineering_status") == "PASS"
        and model_ordering.get("quality_status") == "NOT_EVALUABLE"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", help="JSON file with a list of model specs (see sweep_models.example.json)")
    parser.add_argument("--task-count", type=int, default=5, help="number of deterministic prompts (default 5)")
    parser.add_argument("--repeats", type=int, default=2, help="responses per prompt (default 2)")
    parser.add_argument("--concurrency", type=int, default=8, help="concurrent episodes, 1-32 (default 8)")
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument(
        "--compliance-profile",
        action="store_true",
        help="run the contribution-guide minimum: 500 prompts x 16 responses",
    )
    profile.add_argument(
        "--engineering-smoke",
        action="store_true",
        help="run the Run 1B model/tool engineering gate: 1 prompt x 2 responses",
    )
    profile.add_argument(
        "--benchmark-smoke",
        action="store_true",
        help="run the Run 1B pre-expansion benchmark gate: 5 prompts x 2 responses",
    )
    parser.add_argument(
        "--max-failure-rate",
        type=float,
        default=0.0,
        help=("maximum parse/invalid/infrastructure failure fraction per model outside compliance mode (default 0)"),
    )
    parser.add_argument(
        "--request-capture-out",
        type=Path,
        help=(
            "JSONL path for exact model request payloads (HTTP headers excluded); "
            "named Run 1B profiles require unauthenticated local loopback endpoints; "
            "required by --engineering-smoke and --benchmark-smoke"
        ),
    )
    parser.add_argument("--out", help="write the full report as JSON to this path")
    args = parser.parse_args(argv)
    if args.request_capture_out is not None and args.out is not None:
        capture_path = args.request_capture_out.expanduser().resolve(strict=False)
        report_path = Path(args.out).expanduser().resolve(strict=False)
        if capture_path == report_path:
            parser.error("--request-capture-out and --out must resolve to different files")
    if (args.engineering_smoke or args.benchmark_smoke) and args.request_capture_out is None:
        parser.error("--request-capture-out is required for --engineering-smoke and --benchmark-smoke")
    if args.compliance_profile and args.request_capture_out is not None:
        parser.error(
            "--request-capture-out is forbidden with --compliance-profile; "
            "bind the passed prelaunch capture receipts externally"
        )

    specs = []
    if args.models:
        with open(args.models) as f:
            specs = [ModelSpec(**spec) for spec in json.load(f)]

    try:
        base_rows = _load_example_rows()
        if args.compliance_profile:
            _require_compliance_models(specs)
            _validate_compliance_rows(base_rows)
        elif args.engineering_smoke:
            _require_ranked_models(specs, "--engineering-smoke")
            if args.max_failure_rate != 0.0:
                raise ValueError("--engineering-smoke requires --max-failure-rate 0")
        elif args.benchmark_smoke:
            _require_ranked_models(specs, "--benchmark-smoke")
            if args.max_failure_rate != 0.0:
                raise ValueError("--benchmark-smoke requires --max-failure-rate 0")
        task_count = (
            500
            if args.compliance_profile
            else 1
            if args.engineering_smoke
            else 5
            if args.benchmark_smoke
            else args.task_count
        )
        repeats = (
            16 if args.compliance_profile else 2 if (args.engineering_smoke or args.benchmark_smoke) else args.repeats
        )
        task_rows = _profile_task_rows(base_rows, task_count)
        report = asyncio.run(
            _sweep(
                task_rows,
                repeats,
                specs,
                concurrency=args.concurrency,
                compliance_profile=args.compliance_profile,
                run1b_profile=args.engineering_smoke or args.benchmark_smoke,
                failure_rate_ceiling=args.max_failure_rate,
                request_capture_out=args.request_capture_out,
            )
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e
    _print_report(report)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"report written to {args.out}")
    # Nonzero exit on a broken anchor ladder or a failed/incomplete declared
    # multi-model hierarchy, so the command can be used as a pre-training gate.
    model_gate_ok = _model_gate_satisfied(report["model_ordering"], configured_models=len(specs))
    if (args.engineering_smoke and not model_gate_ok) or (
        not args.engineering_smoke and (not report["anchor_ordering_ok"] or not model_gate_ok)
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
