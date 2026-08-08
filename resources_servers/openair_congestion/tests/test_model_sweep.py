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

from resources_servers.openair_congestion import model_sweep as model_sweep_module
from resources_servers.openair_congestion.model_sweep import (
    _ANCHOR_CONSTRAINTS,
    _ANCHOR_ORDER,
    ModelSpec,
    PolicyStats,
    _evaluate_model_ordering,
    _llm_action,
    _load_example_rows,
    _make_random_valid,
    _parse_tool_call,
    _parse_topology,
    _profile_task_rows,
    _repeat_task_rows,
    _require_compliance_models,
    _sweep,
    _validate_compliance_rows,
)


def _free_port():
    # Import lazily so the source-isolated model_sweep module is fixed in
    # sys.modules before Ray mutates the namespace-package search path.
    from resources_servers.openair_congestion.client import _free_port as find_free_port

    return find_free_port()


def _episode_record(
    prompt_index: int,
    response_index: int,
    *,
    scenario_id: str = "bursty",
    episode_return: float = 0.0,
    parse_failures: int = 0,
    invalid_calls: int = 0,
    usable: bool | None = None,
) -> dict:
    if usable is None:
        usable = parse_failures == 0 and invalid_calls == 0
    return {
        "pair_key": f"{prompt_index}:{response_index}",
        "prompt_index": prompt_index,
        "response_index": response_index,
        "scenario_id": scenario_id,
        "return": episode_return,
        "usable": usable,
        "parse_failures": parse_failures,
        "invalid_calls": invalid_calls,
    }


def _profile_row(label: str, records: list[dict], *, infra_errors: int = 0) -> dict:
    returns = [float(record["return"]) for record in records]
    return {
        "policy": f"model:{label}",
        "mean_return": sum(returns) / len(returns) if returns else None,
        "episodes": len(records),
        "infra_errors": infra_errors,
        "parse_failures": sum(int(record["parse_failures"]) for record in records),
        "invalid_calls": sum(int(record["invalid_calls"]) for record in records),
        "usable_episodes": sum(record["usable"] is True for record in records),
        "episode_records": records,
    }


def _pair_manifest(*scenario_ids: str, responses: int = 1) -> dict[str, str]:
    return {
        f"{prompt_index}:{response_index}": scenario_id
        for prompt_index, scenario_id in enumerate(scenario_ids)
        for response_index in range(responses)
    }


def _ordering_result(
    weak_records: list[dict],
    strong_records: list[dict],
    expected_pair_manifest: dict[str, str],
    *,
    compliance: bool = True,
) -> dict:
    specs = [
        ModelSpec(label="small", model="small", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]
    return _evaluate_model_ordering(
        [_profile_row("small", weak_records), _profile_row("frontier", strong_records)],
        specs,
        expected_episodes=len(expected_pair_manifest),
        expected_pair_manifest=expected_pair_manifest,
        compliance=compliance,
    )


def test_model_ordering_rejects_pair_key_coordinate_disagreement():
    specs = [
        ModelSpec(label="small", model="small", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]
    malformed = _episode_record(0, 0)
    malformed["prompt_index"] = 99
    profile = [
        _profile_row("small", [malformed]),
        _profile_row("frontier", [{**malformed, "return": 1.0}]),
    ]

    result = _evaluate_model_ordering(
        profile,
        specs,
        expected_episodes=1,
        expected_pair_manifest=_pair_manifest("bursty"),
        compliance=True,
    )

    assert result["status"] == "NOT_EVALUABLE"
    assert "pair_key" in result["reason"]


@pytest.mark.parametrize("coordinate", ["prompt_index", "response_index"])
def test_model_ordering_rejects_boolean_explicit_coordinates(coordinate):
    weak = _episode_record(0, 0)
    weak[coordinate] = False

    result = _ordering_result(
        [weak],
        [_episode_record(0, 0, episode_return=1.0)],
        _pair_manifest("bursty"),
    )

    assert result["status"] == "NOT_EVALUABLE"
    assert f".{coordinate} must be a non-negative integer" in result["reason"]


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda record: record.update(pair_key="0"), "pair_key"),
        (lambda record: record.update(pair_key="1:0", prompt_index=1), "planned prompt/repeat support"),
        (lambda record: record.update(scenario_id="interference"), "task manifest"),
        (lambda record: record.update(return_=float("nan")), "finite number"),
    ],
)
def test_model_ordering_rejects_malformed_or_out_of_manifest_records(mutate, reason):
    weak = _episode_record(0, 0)
    if reason == "finite number":
        weak["return"] = float("nan")
    else:
        mutate(weak)
    strong = {**weak, "return": 1.0} if reason != "finite number" else _episode_record(0, 0, episode_return=1.0)

    result = _ordering_result([weak], [strong], _pair_manifest("bursty"))

    assert result["status"] == "NOT_EVALUABLE"
    assert reason in result["reason"]


@pytest.mark.parametrize(
    "record_updates, row_updates, reason",
    [
        ({"usable": "false"}, {}, "usable"),
        ({"usable": False}, {}, "usable"),
        ({"parse_failures": True, "usable": False}, {}, "parse_failures"),
        ({"parse_failures": 1, "usable": False}, {"parse_failures": 0}, "parse_failures disagree"),
        ({}, {"usable_episodes": 0}, "usable_episodes disagree"),
        ({}, {"mean_return": float("nan")}, "mean_return must be a finite number"),
        ({}, {"mean_return": 99.0}, "mean_return disagrees"),
    ],
)
def test_model_ordering_rejects_malformed_failure_accounting(record_updates, row_updates, reason):
    weak = _episode_record(0, 0)
    weak.update(record_updates)
    weak_row = _profile_row("small", [weak])
    weak_row.update(row_updates)
    strong = _episode_record(0, 0, episode_return=1.0)
    specs = [
        ModelSpec(label="small", model="small", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]

    result = _evaluate_model_ordering(
        [weak_row, _profile_row("frontier", [strong])],
        specs,
        expected_episodes=1,
        expected_pair_manifest=_pair_manifest("bursty"),
        compliance=True,
    )

    assert result["status"] == "NOT_EVALUABLE"
    assert reason in result["reason"]


def test_model_ordering_rejects_duplicate_or_incomplete_cartesian_coordinates():
    manifest = _pair_manifest("bursty", responses=2)
    duplicate = _episode_record(0, 0)
    weak = [duplicate, dict(duplicate)]
    strong = [_episode_record(0, 0, episode_return=1.0), _episode_record(0, 1, episode_return=1.0)]

    result = _ordering_result(weak, strong, manifest)

    assert result["status"] == "NOT_EVALUABLE"
    assert "duplicate" in result["reason"]


@pytest.mark.parametrize(
    "manifest, reason",
    [
        ({"0:0": "bursty", "0:1": "interference"}, "exactly one scenario"),
        ({"0:0": "bursty", "1:1": "interference"}, "Cartesian"),
    ],
)
def test_model_ordering_rejects_noncartesian_or_multiregime_prompt_manifest(manifest, reason):
    weak = [
        _episode_record(*map(int, pair_key.split(":")), scenario_id=scenario_id)
        for pair_key, scenario_id in manifest.items()
    ]
    strong = [{**record, "return": 1.0} for record in weak]

    with pytest.raises(ValueError, match=reason):
        _ordering_result(weak, strong, manifest)


def test_anchor_comparisons_validate_both_policies_against_the_task_manifest(monkeypatch):
    seen_stats = []

    async def fake_run_episode(base_url, row, action_fn, stats):
        stats_index = next((index for index, candidate in enumerate(seen_stats) if candidate is stats), None)
        if stats_index is None:
            seen_stats.append(stats)
            stats_index = len(seen_stats) - 1
        stats.finish_episode(
            float(4 - stats_index),
            scenario_id=row["scenario_id"],
            tool_counts={"noop": 1},
            rejected_steps=0,
            episode_steps=1,
            pair_key=f"{row['_profile_prompt_index']}:{row['_profile_response_index']}",
            parse_failures=0,
            invalid_calls=0,
        )
        # _ANCHORS insertion order is relief, random-valid, noop, catastrophic.
        # Corrupt only the worse side of relief > noop; checking the better side
        # alone would miss this and could still issue an anchor PASS.
        if stats_index == 2:
            stats.episode_records[-1]["scenario_id"] = "wrong-regime"

    monkeypatch.setattr(model_sweep_module, "_start_local_server", lambda: "http://unused")
    monkeypatch.setattr(model_sweep_module, "_run_episode", fake_run_episode)
    rows = _profile_task_rows(_load_example_rows(), task_count=1)

    with pytest.raises(ValueError, match=r"anchor:noop.*task manifest"):
        asyncio.run(_sweep(rows, repeats=1, specs=[], concurrency=1))


def test_anchor_environment_gate_fails_on_structurally_invalid_anchor_record(monkeypatch):
    seen_stats = []

    async def fake_run_episode(base_url, row, action_fn, stats):
        stats_index = next((index for index, candidate in enumerate(seen_stats) if candidate is stats), None)
        if stats_index is None:
            seen_stats.append(stats)
            stats_index = len(seen_stats) - 1
        # In insertion order, these returns make every declared anchor
        # comparison positive.  The relief record is nevertheless unusable,
        # so reward ordering alone must not qualify the environment gate.
        returns = (4.0, 3.0, 2.0, 1.0)
        stats.invalid_calls += int(stats_index == 0)
        stats.finish_episode(
            returns[stats_index],
            scenario_id=row["scenario_id"],
            tool_counts={"noop": 1},
            rejected_steps=0,
            episode_steps=1,
            pair_key=f"{row['_profile_prompt_index']}:{row['_profile_response_index']}",
            parse_failures=0,
            invalid_calls=int(stats_index == 0),
        )

    monkeypatch.setattr(model_sweep_module, "_start_local_server", lambda: "http://unused")
    monkeypatch.setattr(model_sweep_module, "_run_episode", fake_run_episode)
    rows = _profile_task_rows(_load_example_rows(), task_count=1)

    report = asyncio.run(_sweep(rows, repeats=1, specs=[], concurrency=1))

    assert all(comparison["status"] == "PASS" for comparison in report["anchor_ordering_comparisons"])
    assert report["anchor_ordering_ok"] is False


def test_episode_records_expose_prompt_and_response_coordinates():
    stats = PolicyStats()
    stats.finish_episode(
        1.25,
        scenario_id="bursty",
        tool_counts={"noop": 1},
        rejected_steps=0,
        episode_steps=1,
        pair_key="7:3",
        parse_failures=0,
        invalid_calls=0,
    )

    assert stats.episode_records[0]["prompt_index"] == 7
    assert stats.episode_records[0]["response_index"] == 3


def test_anchor_sweep_orders_policies_and_reports_profile():
    # Ordering is a cross-regime gate; evaluating only the first one or two
    # rows can correctly tie relief with noop when neither state needs action.
    rows = _profile_task_rows(_load_example_rows(), task_count=5)
    report = asyncio.run(_sweep(rows, repeats=1, specs=[], concurrency=2))

    assert report["episodes_per_policy"] == 5
    by_policy = {row["policy"]: row for row in report["profile"]}
    assert set(by_policy) == set(_ANCHOR_ORDER)
    assert all(row["episodes"] == 5 for row in report["profile"])

    # The known quality ladder must hold on the deterministic replay tasks,
    # and the report must say so itself.
    assert report["anchor_ordering_ok"] is True
    assert all(
        by_policy[better]["mean_return"] > by_policy[worse]["mean_return"] for better, worse in _ANCHOR_CONSTRAINTS
    )
    assert len(report["anchor_ordering_comparisons"]) == len(_ANCHOR_CONSTRAINTS)
    assert all(
        comparison["status"] == "PASS" and comparison["ci95_low"] > 0.0
        for comparison in report["anchor_ordering_comparisons"]
    )
    assert all(
        comparison["bootstrap_method"] == "regime_stratified_prompt_cluster_percentile"
        and comparison["bootstrap_draws"] == 10_000
        and comparison["prompt_clusters"] == 5
        and comparison["response_pairs"] == 5
        and comparison["prompt_wins"] + comparison["prompt_ties"] + comparison["prompt_losses"] == 5
        for comparison in report["anchor_ordering_comparisons"]
    )

    # Catastrophic play is rejected every step; valid play never is.
    assert by_policy["anchor:catastrophic"]["rejection_rate"] == 1.0
    assert by_policy["anchor:random-valid"]["rejection_rate"] == 0.0
    assert by_policy["anchor:noop"]["noop_rate"] == 1.0
    assert by_policy["anchor:relief"]["return_distribution"]["median"] is not None
    assert by_policy["anchor:relief"]["tool_metrics"]
    assert set(by_policy["anchor:relief"]["returns_by_scenario"]) == {row["scenario_id"] for row in rows}


def test_profile_rows_cover_every_example_regime_and_repeat_exact_prompts():
    base_rows = _load_example_rows()
    assert len(base_rows) == 5

    prompts = _profile_task_rows(base_rows, task_count=10)
    assert [row["scenario_id"] for row in prompts] == [
        "prb_exhaustion",
        "bursty",
        "interference",
        "prach_storm",
        "qos_competition",
    ] * 2
    assert len({row["seed"] for row in prompts}) == 10

    repeated = _repeat_task_rows(prompts, repeats=3)
    assert len(repeated) == 30
    for prompt_idx in range(10):
        group = repeated[prompt_idx * 3 : (prompt_idx + 1) * 3]
        assert {row["seed"] for row in group} == {prompts[prompt_idx]["seed"]}
        assert [row["_profile_response_index"] for row in group] == [0, 1, 2]


def test_model_ordering_requires_ranked_models_to_improve_monotonically():
    manifest = _pair_manifest(*(tuple("bursty" for _ in range(8))))
    weak = [_episode_record(index, 0, episode_return=-4.0) for index in range(8)]
    stronger = [_episode_record(index, 0, episode_return=-3.0) for index in range(8)]

    assert _ordering_result(weak, stronger, manifest, compliance=False)["status"] == "PASS"

    weaker_frontier = [_episode_record(index, 0, episode_return=-5.0) for index in range(8)]
    result = _ordering_result(weak, weaker_frontier, manifest, compliance=False)
    assert result["status"] == "FAIL"
    assert result["expected"] == ["model:small", "model:frontier"]


def test_model_ordering_bootstraps_regime_stratified_prompt_clusters():
    weak_records = []
    strong_records = []
    # Each regime has two prompt clusters with the same within-regime delta.
    # The interval is therefore deterministic at the prompt level even though
    # 64 response deltas exist; response-level resampling would count the wrong
    # experimental unit.
    prompt_contract = (("bursty", 10.0), ("bursty", 10.0), ("interference", -2.0), ("interference", -2.0))
    for prompt_index, (scenario_id, delta) in enumerate(prompt_contract):
        for response_index in range(16):
            weak_records.append(_episode_record(prompt_index, response_index, scenario_id=scenario_id))
            strong_records.append(
                _episode_record(prompt_index, response_index, scenario_id=scenario_id, episode_return=delta)
            )

    manifest = _pair_manifest(*(scenario for scenario, _delta in prompt_contract), responses=16)
    comparison = _ordering_result(weak_records, strong_records, manifest, compliance=False)["comparisons"][0]

    assert comparison["bootstrap_method"] == "regime_stratified_prompt_cluster_percentile"
    assert comparison["bootstrap_seed"] == 0
    assert comparison["bootstrap_draws"] == 10_000
    assert comparison["prompt_clusters"] == 4
    assert comparison["response_pairs"] == 64
    assert comparison["mean_delta"] == 4.0
    assert comparison["median_delta"] == 4.0
    assert comparison["ci95_low"] == 4.0
    assert comparison["ci95_high"] == 4.0
    assert comparison["prompt_wins"] == 2
    assert comparison["prompt_ties"] == 0
    assert comparison["prompt_losses"] == 2


def test_model_ordering_preserves_gate_precision_in_raw_comparison_fields():
    manifest = _pair_manifest("bursty", "bursty")
    weak = [_episode_record(0, 0), _episode_record(1, 0)]
    strong = [
        _episode_record(0, 0, episode_return=0.0000004),
        _episode_record(1, 0, episode_return=0.0000004),
    ]

    result = _ordering_result(weak, strong, manifest, compliance=False)
    comparison = result["comparisons"][0]

    assert comparison["status"] == "PASS"
    assert comparison["mean_delta"] == pytest.approx(0.0000004)
    assert comparison["ci95_low"] == pytest.approx(0.0000004)
    assert comparison["mean_delta"] > 0.0 and comparison["ci95_low"] > 0.0


def test_singleton_regime_smoke_is_engineering_only_not_a_quality_pass():
    scenarios = ("prb_exhaustion", "bursty", "interference", "prach_storm", "qos_competition")
    manifest = _pair_manifest(*scenarios)
    weak = [
        _episode_record(prompt_index, 0, scenario_id=scenario_id)
        for prompt_index, scenario_id in enumerate(scenarios)
    ]
    strong = [{**record, "return": 1.0} for record in weak]

    result = _ordering_result(weak, strong, manifest, compliance=False)

    assert result["status"] == "NOT_EVALUABLE"
    assert result["engineering_status"] == "PASS"
    assert result["quality_status"] == "NOT_EVALUABLE"
    assert result["comparisons"] == []
    assert "two prompt clusters per scenario" in result["reason"]


def test_quality_inference_counts_planned_scenarios_with_zero_common_usable_prompts():
    manifest = _pair_manifest("bursty", "bursty", "interference", "interference")
    weak = [
        _episode_record(index, 0, scenario_id=scenario_id)
        for index, scenario_id in enumerate(("bursty", "bursty", "interference", "interference"))
    ]
    strong = [
        _episode_record(
            index,
            0,
            scenario_id=scenario_id,
            episode_return=1.0,
            parse_failures=int(scenario_id == "interference"),
        )
        for index, scenario_id in enumerate(("bursty", "bursty", "interference", "interference"))
    ]
    specs = [
        ModelSpec(label="small", model="small", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]

    result = _evaluate_model_ordering(
        [_profile_row("small", weak), _profile_row("frontier", strong)],
        specs,
        expected_episodes=4,
        expected_pair_manifest=manifest,
        compliance=False,
        failure_rate_ceiling=0.5,
    )

    assert result["status"] == "NOT_EVALUABLE"
    assert "interference" in result["reason"]


def test_cli_gate_allows_only_explicit_engineering_only_smoke_to_progress():
    engineering_only = {
        "status": "NOT_EVALUABLE",
        "engineering_status": "PASS",
        "quality_status": "NOT_EVALUABLE",
    }
    malformed = {"status": "NOT_EVALUABLE", "reason": "incomplete records"}

    assert model_sweep_module._model_gate_satisfied(engineering_only, configured_models=2) is True
    assert model_sweep_module._model_gate_satisfied(malformed, configured_models=2) is False
    assert model_sweep_module._model_gate_satisfied({"status": "FAIL"}, configured_models=2) is False
    assert model_sweep_module._model_gate_satisfied({"status": "PASS"}, configured_models=2) is True


def test_compliance_model_ordering_uses_fifty_thousand_cluster_draws():
    manifest = _pair_manifest("bursty", "bursty")
    weak = [_episode_record(0, 0), _episode_record(1, 0)]
    strong = [_episode_record(0, 0, episode_return=1.0), _episode_record(1, 0, episode_return=1.0)]

    comparison = _ordering_result(weak, strong, manifest, compliance=True)["comparisons"][0]

    assert comparison["bootstrap_draws"] == 50_000


def test_all_parse_failures_are_not_evaluable():
    specs = [
        ModelSpec(label="small", model="small", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]
    manifest = _pair_manifest(*(tuple("bursty" for _ in range(8))))
    profile = []
    for rank, label in enumerate(("small", "frontier")):
        records = [
            _episode_record(
                index,
                0,
                episode_return=-1.0 + rank,
                parse_failures=1,
                usable=False,
            )
            for index in range(8)
        ]
        profile.append(_profile_row(label, records))

    result = _evaluate_model_ordering(
        profile,
        specs,
        expected_episodes=8,
        expected_pair_manifest=manifest,
    )

    assert result["status"] == "NOT_EVALUABLE"
    assert "parse" in result["reason"]


def test_partial_parse_failures_are_not_evaluable():
    specs = [
        ModelSpec(label="small", model="small", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]
    profile = []
    manifest = _pair_manifest(*(tuple("bursty" for _ in range(8))))
    for rank, label in enumerate(("small", "frontier")):
        records = [
            _episode_record(
                index,
                0,
                episode_return=float(rank),
                parse_failures=int(label == "frontier" and index == 0),
            )
            for index in range(8)
        ]
        profile.append(_profile_row(label, records))

    result = _evaluate_model_ordering(
        profile,
        specs,
        expected_episodes=8,
        expected_pair_manifest=manifest,
    )

    assert result["status"] == "NOT_EVALUABLE"


def test_noncompliance_failure_ceiling_uses_only_common_usable_pairs():
    specs = [
        ModelSpec(label="small", model="small", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]
    profile = []
    manifest = _pair_manifest(*(tuple("bursty" for _ in range(8))))
    for rank, label in enumerate(("small", "frontier")):
        records = [
            _episode_record(
                index,
                0,
                episode_return=float(rank),
                parse_failures=int(label == "frontier" and index == 0),
            )
            for index in range(8)
        ]
        profile.append(_profile_row(label, records))

    result = _evaluate_model_ordering(
        profile,
        specs,
        expected_episodes=8,
        expected_pair_manifest=manifest,
        compliance=False,
        failure_rate_ceiling=0.125,
    )

    assert result["status"] == "PASS"
    assert result["valid_paired_episodes"] == 7
    assert result["failure_rates"]["model:frontier"] == pytest.approx(0.125)
    assert result["comparisons"][0]["paired_episodes"] == 7


def test_noncompliance_accepts_multiple_step_failures_in_one_unusable_episode():
    manifest = _pair_manifest("bursty", "bursty", "bursty")
    weak = [_episode_record(index, 0) for index in range(3)]
    strong = [
        _episode_record(0, 0, episode_return=1.0, parse_failures=2),
        _episode_record(1, 0, episode_return=1.0),
        _episode_record(2, 0, episode_return=1.0),
    ]
    specs = [
        ModelSpec(label="small", model="small", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]

    result = _evaluate_model_ordering(
        [_profile_row("small", weak), _profile_row("frontier", strong)],
        specs,
        expected_episodes=3,
        expected_pair_manifest=manifest,
        compliance=False,
        failure_rate_ceiling=2 / 3,
    )

    assert result["status"] == "PASS"
    assert result["valid_paired_episodes"] == 2
    assert result["failure_counts"]["model:frontier"]["parse_failures"] == 2


def test_noncompliance_failure_ceiling_rejects_profiles_above_limit():
    specs = [
        ModelSpec(label="small", model="small", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]
    manifest = _pair_manifest(*(tuple("bursty" for _ in range(4))))
    profile = []
    for rank, label in enumerate(("small", "frontier")):
        records = [
            _episode_record(
                index,
                0,
                episode_return=float(rank),
                parse_failures=int(label == "frontier" and index == 0),
            )
            for index in range(4)
        ]
        profile.append(_profile_row(label, records))

    result = _evaluate_model_ordering(
        profile,
        specs,
        expected_episodes=4,
        expected_pair_manifest=manifest,
        compliance=False,
        failure_rate_ceiling=0.20,
    )

    assert result["status"] == "NOT_EVALUABLE"
    assert "failure-rate ceiling" in result["reason"]


def test_positive_mean_with_interval_crossing_zero_fails():
    specs = [
        ModelSpec(label="small", model="small", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]
    deltas = [-0.1, 0.1, 0.1, -0.05]
    weak_records = [_episode_record(index, 0) for index in range(len(deltas))]
    strong_records = [
        _episode_record(index, 0, episode_return=delta)
        for index, delta in enumerate(deltas)
    ]
    profile = [_profile_row("small", weak_records), _profile_row("frontier", strong_records)]
    manifest = _pair_manifest(*(tuple("bursty" for _ in deltas)))

    result = _evaluate_model_ordering(
        profile,
        specs,
        expected_episodes=len(deltas),
        expected_pair_manifest=manifest,
    )

    assert result["status"] == "FAIL"
    assert result["comparisons"][0]["ci95_low"] <= 0.0


def test_compliance_profile_requires_two_preranked_models():
    with pytest.raises(ValueError, match="at least two real models"):
        _require_compliance_models([])

    unranked = [
        ModelSpec(label="small", model="small", base_url="http://x"),
        ModelSpec(label="frontier", model="frontier", base_url="http://x"),
    ]
    with pytest.raises(ValueError, match="capability_rank"):
        _require_compliance_models(unranked)

    ranked = [
        ModelSpec(label="small", model="small", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]
    _require_compliance_models(ranked)

    duplicate_identity = [
        ModelSpec(label="small", model="same-model", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="same-model", base_url="http://x", capability_rank=2),
    ]
    with pytest.raises(ValueError, match="distinct model identities"):
        _require_compliance_models(duplicate_identity)

    missing_identity = [
        ModelSpec(label="small", model="", base_url="http://x", capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", capability_rank=2),
    ]
    with pytest.raises(ValueError, match="non-empty"):
        _require_compliance_models(missing_identity)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"temperature": float("nan")}, "temperature"),
        ({"temperature": -0.01}, "temperature"),
        ({"temperature": 2.01}, "temperature"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": True}, "max_tokens"),
        ({"max_tokens": 513}, "max_tokens"),
    ],
)
def test_model_spec_rejects_unsafe_temperature_and_token_limits(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ModelSpec(label="model", model="model", base_url="http://x", **kwargs)


@pytest.mark.parametrize("capability_rank", [True, 0, -1, 1.5, "1"])
def test_model_spec_rejects_nonpositive_or_noninteger_capability_rank(capability_rank):
    with pytest.raises(ValueError, match="capability_rank"):
        ModelSpec(
            label="model",
            model="model",
            base_url="http://x",
            capability_rank=capability_rank,
        )


def test_model_spec_pins_top_p_instead_of_inheriting_endpoint_defaults():
    spec = ModelSpec(label="model", model="model", base_url="http://x")

    assert getattr(spec, "top_p", None) == 0.95
    assert spec.max_tokens == 512
    assert spec.chat_template_kwargs is None


@pytest.mark.parametrize(
    "chat_template_kwargs",
    [
        True,
        [],
        {},
        {"enable_thinking": "false"},
        {"enable_thinking": False, "unreviewed_option": True},
    ],
)
def test_model_spec_rejects_unsafe_chat_template_kwargs(chat_template_kwargs):
    with pytest.raises(ValueError, match="chat_template_kwargs"):
        ModelSpec(
            label="model",
            model="model",
            base_url="http://x",
            chat_template_kwargs=chat_template_kwargs,
        )


def test_model_spec_copies_valid_chat_template_kwargs_before_use():
    caller_owned = {"enable_thinking": False}
    spec = ModelSpec(
        label="model",
        model="model",
        base_url="http://x",
        chat_template_kwargs=caller_owned,
    )

    caller_owned["enable_thinking"] = True

    assert spec.chat_template_kwargs == {"enable_thinking": False}


@pytest.mark.parametrize("top_p", [float("nan"), 0.0, -0.01, 1.01, True])
def test_model_spec_rejects_unsafe_top_p(top_p):
    with pytest.raises(ValueError, match="top_p"):
        ModelSpec(label="model", model="model", base_url="http://x", top_p=top_p)


def test_llm_request_pins_identical_sampling_template_kwargs_and_pair_derived_seed_across_models():
    class Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def raise_for_status(self):
            return None

        async def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [{"function": {"name": "noop", "arguments": "{}"}}],
                            "content": None,
                        }
                    }
                ]
            }

    class Session:
        def __init__(self):
            self.payloads = []

        def post(self, url, *, json, headers, timeout):
            self.payloads.append(json)
            return Response()

    row = _profile_task_rows(_load_example_rows(), task_count=8)[7]
    row["_profile_response_index"] = 3
    session = Session()
    decoding = {
        "temperature": 0.3,
        "top_p": 0.85,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    specs = [
        ModelSpec(label="small", model="small", base_url="http://x", **decoding),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", **decoding),
    ]

    for spec in specs:
        action = asyncio.run(_llm_action(session, spec, row, "- Cell 0: state", step_idx=4))
        assert action == {"name": "noop", "arguments": {}}

    sampling_contracts = [
        {
            key: payload[key]
            for key in ("temperature", "top_p", "max_tokens", "seed", "chat_template_kwargs")
        }
        for payload in session.payloads
    ]
    assert sampling_contracts == [
        {
            "temperature": 0.3,
            "top_p": 0.85,
            "max_tokens": 256,
            "seed": 756_251_775,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        {
            "temperature": 0.3,
            "top_p": 0.85,
            "max_tokens": 256,
            "seed": 756_251_775,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ]


def test_llm_request_omits_chat_template_kwargs_for_generic_sweep():
    class Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def raise_for_status(self):
            return None

        async def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [{"function": {"name": "noop", "arguments": "{}"}}],
                            "content": None,
                        }
                    }
                ]
            }

    class Session:
        def __init__(self):
            self.payload = None

        def post(self, url, *, json, headers, timeout):
            self.payload = json
            return Response()

    row = _profile_task_rows(_load_example_rows(), task_count=1)[0]
    row["_profile_response_index"] = 0
    session = Session()
    spec = ModelSpec(label="generic", model="generic", base_url="http://x")

    action = asyncio.run(_llm_action(session, spec, row, "- Cell 0: state", step_idx=0))

    assert action == {"name": "noop", "arguments": {}}
    assert "chat_template_kwargs" not in session.payload


def test_sweep_report_binds_credential_free_model_sampling_backend_and_task_contract(monkeypatch):
    seen_stats = []

    async def fake_run_episode(base_url, row, action_fn, stats):
        stats_index = next((index for index, candidate in enumerate(seen_stats) if candidate is stats), None)
        if stats_index is None:
            seen_stats.append(stats)
            stats_index = len(seen_stats) - 1
        stats.finish_episode(
            float(stats_index),
            scenario_id=row["scenario_id"],
            tool_counts={"noop": 1},
            rejected_steps=0,
            episode_steps=1,
            pair_key=f"{row['_profile_prompt_index']}:{row['_profile_response_index']}",
            parse_failures=0,
            invalid_calls=0,
        )

    monkeypatch.setattr(model_sweep_module, "_start_local_server", lambda: "http://unused")
    monkeypatch.setattr(model_sweep_module, "_run_episode", fake_run_episode)
    rows = _profile_task_rows(_load_example_rows(), task_count=2)
    specs = [
        ModelSpec(
            label="small",
            model="Qwen/Qwen3-1.7B",
            base_url="https://user:never-serialize@example.invalid/v1",
            temperature=0.3,
            top_p=0.85,
            max_tokens=256,
            chat_template_kwargs={"enable_thinking": False},
            capability_rank=1,
        ),
        ModelSpec(
            label="large",
            model="Qwen/Qwen3-8B",
            base_url="https://user:never-serialize@example.invalid/v1",
            temperature=0.3,
            top_p=0.85,
            max_tokens=256,
            chat_template_kwargs={"enable_thinking": False},
            capability_rank=2,
        ),
    ]

    report = asyncio.run(_sweep(rows, repeats=1, specs=specs, concurrency=1))

    assert report["backend"] == "replay"
    assert report["model_specs"] == [
        {"label": "small", "model": "Qwen/Qwen3-1.7B", "capability_rank": 1},
        {"label": "large", "model": "Qwen/Qwen3-8B", "capability_rank": 2},
    ]
    assert report["sampling_contract"] == {
        "temperature": 0.3,
        "top_p": 0.85,
        "max_tokens": 256,
        "seed_derivation": "sha256(run1b-request-v1:{prompt_index}:{response_index}:{step_index}) mod 2**31",
        "seed_version": "run1b-request-v1",
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert report["tasks"][0]["regime_mix"] == {"prb_exhaustion": 1.0}
    assert "never-serialize" not in json.dumps(report)


def test_compliance_profile_requires_exact_one_hot_regime_coverage():
    rows = _load_example_rows()
    _validate_compliance_rows(rows)

    duplicate = json.loads(json.dumps(rows))
    duplicate[-1] = json.loads(json.dumps(duplicate[0]))
    with pytest.raises(ValueError, match="exactly once"):
        _validate_compliance_rows(duplicate)

    mismatched = json.loads(json.dumps(rows))
    mismatched[0]["regime_mix"] = {"bursty": 1.0}
    with pytest.raises(ValueError, match="one-hot regime_mix"):
        _validate_compliance_rows(mismatched)

    extra = json.loads(json.dumps(rows))
    extra.append(json.loads(json.dumps(rows[0])))
    extra[-1]["scenario_id"] = "unknown"
    extra[-1]["regime_mix"] = {"unknown": 1.0}
    with pytest.raises(ValueError, match="exactly once"):
        _validate_compliance_rows(extra)


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
    # "cooperative" calls noop natively; "rambler" replies prose only (a
    # terminal protocol failure); "hallucinator" calls a tool that does not
    # exist (a terminal invalid call); "dead" 500s every request (episodes dropped as
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
            rows = _profile_task_rows(_load_example_rows(), task_count=1)
            return await _sweep(rows, repeats=1, specs=specs, concurrency=1)
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
    assert rambler["parse_failures"] == 1
    assert rambler["noop_rate"] == 0.0
    assert rambler["tool_metrics"]["<parse_failure>"]["calls"] == 1
    assert rambler["usable_episodes"] == 0
    # A terminal protocol penalty and a multi-step valid return are not
    # comparable on magnitude; usability, not score ordering, excludes the
    # malformed response from capability comparisons.
    assert rambler["episode_records"][0]["usable"] is False

    # LLM rows carry the reward-ceiling reference; noop-playing models must
    # match the noop anchor's return on the same paired task.
    assert cooperative["mean_return"] == by_policy["anchor:noop"]["mean_return"]
    assert cooperative["vs_relief"] is not None

    # The request carried the row's own messages plus the observation.
    payload = seen_payloads[0]
    assert payload["messages"][0]["role"] == "system" and payload["messages"][0]["content"]
    assert payload["messages"][-1]["role"] == "user" and "Cell 0" in payload["messages"][-1]["content"]
    assert any(t["function"]["name"] == "set_prb_cap" for t in payload["tools"])
    assert payload["tool_choice"] == "required"
    assert payload["parallel_tool_calls"] is False
    first_payload_by_model = {}
    for candidate in seen_payloads:
        first_payload_by_model.setdefault(candidate["model"], candidate)
    assert set(first_payload_by_model) == {"dead", "cooperative", "hallucinator", "rambler"}
    assert {
        (
            candidate["temperature"],
            candidate["top_p"],
            candidate["max_tokens"],
            candidate["seed"],
        )
        for candidate in first_payload_by_model.values()
    } == {(0.2, 0.95, 512, 355_554_858)}


def test_sweep_rejects_bad_config():
    with pytest.raises(ValueError, match="repeats"):
        asyncio.run(_sweep(_profile_task_rows(_load_example_rows(), 1), repeats=0, specs=[]))
    dup = [
        ModelSpec(label="same", model="a", base_url="http://127.0.0.1:1/v1"),
        ModelSpec(label="same", model="b", base_url="http://127.0.0.1:1/v1"),
    ]
    with pytest.raises(ValueError, match="duplicate model labels"):
        asyncio.run(_sweep(_profile_task_rows(_load_example_rows(), 1), repeats=1, specs=dup))

    mismatched_sampling = [
        ModelSpec(label="small", model="small", base_url="http://x", temperature=0.2, capability_rank=1),
        ModelSpec(label="frontier", model="frontier", base_url="http://x", temperature=0.3, capability_rank=2),
    ]
    with pytest.raises(ValueError, match="identical sampling"):
        asyncio.run(
            _sweep(
                _profile_task_rows(_load_example_rows(), 1),
                repeats=1,
                specs=mismatched_sampling,
            )
        )

    mismatched_template_kwargs = [
        ModelSpec(
            label="small",
            model="small",
            base_url="http://x",
            chat_template_kwargs={"enable_thinking": False},
            capability_rank=1,
        ),
        ModelSpec(
            label="frontier",
            model="frontier",
            base_url="http://x",
            chat_template_kwargs={"enable_thinking": True},
            capability_rank=2,
        ),
    ]
    with pytest.raises(ValueError, match="identical sampling and chat-template settings"):
        asyncio.run(
            _sweep(
                _profile_task_rows(_load_example_rows(), 1),
                repeats=1,
                specs=mismatched_template_kwargs,
            )
        )


def test_compliance_profile_rejects_identical_but_nonfrozen_sampling():
    specs = [
        ModelSpec(
            label="small",
            model="small",
            base_url="http://127.0.0.1:1/v1",
            temperature=0.3,
            top_p=0.95,
            max_tokens=512,
            capability_rank=1,
        ),
        ModelSpec(
            label="large",
            model="large",
            base_url="http://127.0.0.1:1/v1",
            temperature=0.3,
            top_p=0.95,
            max_tokens=512,
            capability_rank=2,
        ),
    ]

    with pytest.raises(ValueError, match="frozen sampling contract"):
        asyncio.run(
            _sweep(
                _profile_task_rows(_load_example_rows(), 1),
                repeats=1,
                specs=specs,
                compliance_profile=True,
            )
        )


@pytest.mark.parametrize(
    "chat_template_kwargs",
    [None, {"enable_thinking": True}],
)
@pytest.mark.parametrize("profile_kwargs", [{"run1b_profile": True}, {"compliance_profile": True}])
def test_run1b_profiles_require_thinking_disabled(chat_template_kwargs, profile_kwargs):
    specs = [
        ModelSpec(
            label="small",
            model="small",
            base_url="http://127.0.0.1:1/v1",
            chat_template_kwargs=chat_template_kwargs,
            capability_rank=1,
        ),
        ModelSpec(
            label="large",
            model="large",
            base_url="http://127.0.0.1:1/v1",
            chat_template_kwargs=chat_template_kwargs,
            capability_rank=2,
        ),
    ]

    with pytest.raises(ValueError, match="enable_thinking"):
        asyncio.run(
            _sweep(
                _profile_task_rows(_load_example_rows(), 1),
                repeats=1,
                specs=specs,
                **profile_kwargs,
            )
        )


def test_sweep_refuses_named_but_unset_api_key_env(monkeypatch):
    monkeypatch.delenv("SWEEP_TEST_MISSING_KEY", raising=False)
    spec = ModelSpec(label="x", model="x", base_url="http://127.0.0.1:1/v1", api_key_env="SWEEP_TEST_MISSING_KEY")
    with pytest.raises(ValueError, match="SWEEP_TEST_MISSING_KEY"):
        asyncio.run(_sweep(_profile_task_rows(_load_example_rows(), 1), repeats=1, specs=[spec]))


def test_parse_tool_call_accepts_one_native_call_and_rejects_content_json_or_garbage():
    native = {
        "tool_calls": [{"function": {"name": "noop", "arguments": "{}"}}],
        "content": None,
    }
    assert _parse_tool_call(native) == {"name": "noop", "arguments": {}}

    content = {"content": 'Sure: {"name": "set_scheduler_policy", "arguments": {"cell_id": 0, "policy": "PF"}}'}
    assert _parse_tool_call(content) is None

    assert _parse_tool_call({"content": "I would consider the network first."}) is None
    assert _parse_tool_call({"tool_calls": [{"function": {"name": "noop", "arguments": "{not json"}}]}) is None


def test_parse_tool_call_rejects_multiple_native_tool_calls():
    message = {
        "tool_calls": [
            {"function": {"name": "noop", "arguments": "{}"}},
            {
                "function": {
                    "name": "set_scheduler_policy",
                    "arguments": '{"cell_id": 0, "policy": "PF"}',
                }
            },
        ]
    }

    assert _parse_tool_call(message) is None


@pytest.mark.parametrize(
    "arguments",
    [
        '{"cell_id": 0, "weights": {"1": NaN}}',
        '{"cell_id": 0, "cell_id": 1, "policy": "PF"}',
    ],
)
def test_parse_tool_call_rejects_nonstandard_or_ambiguous_json(arguments):
    message = {
        "tool_calls": [
            {
                "function": {
                    "name": "set_qos_weights",
                    "arguments": arguments,
                }
            }
        ]
    }

    assert _parse_tool_call(message) is None
