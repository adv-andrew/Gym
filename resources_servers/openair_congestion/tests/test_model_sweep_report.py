# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest


_MODULE = "resources_servers.openair_congestion.model_sweep_report"


def _report_module():
    return importlib.import_module(_MODULE)


def _records(returns: list[float], scenarios: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for prompt_index, scenario_id in enumerate(scenarios):
        for response_index in range(2):
            value = returns[prompt_index * 2 + response_index]
            records.append(
                {
                    "return": value,
                    "pair_key": f"{prompt_index}:{response_index}",
                    "prompt_index": prompt_index,
                    "response_index": response_index,
                    "usable": True,
                    "parse_failures": 0,
                    "invalid_calls": 0,
                    "scenario_id": scenario_id,
                    "tool_counts": {"noop": 2},
                    "rejection_rate": 0.0,
                    "noop_rate": 1.0,
                    "steps": 2,
                }
            )
    return records


def _profile_row(policy: str, returns: list[float], scenarios: list[str]) -> dict[str, Any]:
    records = _records(returns, scenarios)
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    ordered = sorted(returns)

    def quantile(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    returns_by_scenario = {}
    for scenario in sorted(set(scenarios)):
        values = [
            value
            for prompt_index, label in enumerate(scenarios)
            if label == scenario
            for value in returns[prompt_index * 2 : prompt_index * 2 + 2]
        ]
        returns_by_scenario[scenario] = {
            "episodes": len(values),
            "mean_return": round(statistics.fmean(values), 6),
            "std_return": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
        }

    return {
        "policy": policy,
        "episodes": len(records),
        "mean_return": round(mean, 4),
        "std_return": round(math.sqrt(variance), 4),
        "return_distribution": {
            "min": min(returns),
            "p05": round(quantile(0.05), 6),
            "p25": round(quantile(0.25), 6),
            "median": round(quantile(0.50), 6),
            "p75": round(quantile(0.75), 6),
            "p95": round(quantile(0.95), 6),
            "max": max(returns),
        },
        "rejection_rate": 0.0,
        "noop_rate": 1.0,
        "invalid_calls": 0,
        "parse_failures": 0,
        "infra_errors": 0,
        "usable_episodes": len(records),
        "episode_records": records,
        "tool_metrics": {
            "noop": {
                "calls": len(records) * 2,
                "call_rate": 1.0,
                "mean_step_reward": round(sum(returns) / (len(records) * 2), 6),
                "rejection_rate": 0.0,
            }
        },
        "episode_return_correlations": {
            "rejection_rate": None,
            "noop_rate": None,
            "tool_rate:noop": None,
        },
        "returns_by_scenario": returns_by_scenario,
        "vs_relief": 0.0,
    }


def _comparison(
    *,
    left_key: str,
    left: str,
    right_key: str,
    right: str,
    left_returns: list[float],
    right_returns: list[float],
    scenarios: list[str],
    seed: int,
) -> dict[str, Any]:
    response_deltas = [a - b for a, b in zip(left_returns, right_returns, strict=True)]
    prompt_deltas = [sum(response_deltas[index : index + 2]) / 2 for index in range(0, len(response_deltas), 2)]
    mean_delta = sum(prompt_deltas) / len(prompt_deltas)
    ordered = sorted(prompt_deltas)
    strata: dict[str, list[float]] = {}
    for prompt_index, scenario in sorted(enumerate(scenarios), key=lambda item: (item[1], item[0])):
        strata.setdefault(scenario, []).append(prompt_deltas[prompt_index])
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(10_000, dtype=float)
    chunk_size = 256
    for start in range(0, 10_000, chunk_size):
        count = min(chunk_size, 10_000 - start)
        sampled_sum = np.zeros(count, dtype=float)
        for scenario in sorted(strata):
            values = np.asarray(strata[scenario], dtype=float)
            indices = rng.integers(0, len(values), size=(count, len(values)))
            sampled_sum += values[indices].sum(axis=1)
        bootstrap_means[start : start + count] = sampled_sum / len(prompt_deltas)
    ci95_low, ci95_high = (float(value) for value in np.quantile(bootstrap_means, [0.025, 0.975]))
    return {
        left_key: left,
        right_key: right,
        "pairs": 4,
        "paired_episodes": 4,
        "response_pairs": 4,
        "prompt_clusters": 2,
        "mean_delta": mean_delta,
        "median_delta": statistics.median(ordered),
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "prompt_wins": sum(value > 0 for value in prompt_deltas),
        "prompt_ties": sum(value == 0 for value in prompt_deltas),
        "prompt_losses": sum(value < 0 for value in prompt_deltas),
        "bootstrap_method": "regime_stratified_prompt_cluster_percentile",
        "bootstrap_seed": seed,
        "bootstrap_draws": 10_000,
        "status": "PASS" if mean_delta > 0 and ci95_low > 0 else "FAIL",
    }


def _raw_report(scenarios: list[str] | None = None) -> dict[str, Any]:
    # The default positive-path fixture has two prompt clusters in one regime,
    # which is the minimum support the sweep accepts for quality inference.
    scenarios = scenarios or ["bursty", "bursty"]
    if len(scenarios) != 2:
        raise ValueError("report fixture requires exactly two scenarios")
    returns = {
        "anchor:relief": [3.0, 1.0, 5.0, 3.0],
        "anchor:noop": [0.0, 0.0, 1.0, 1.0],
        "anchor:random-valid": [-1.0, -1.0, 0.0, 0.0],
        "anchor:catastrophic": [-5.0, -5.0, -5.0, -5.0],
        "model:small": [0.0, 2.0, 1.0, 3.0],
        "model:large": [2.0, 4.0, 2.0, 6.0],
    }
    profile = [_profile_row(policy, values, scenarios) for policy, values in returns.items()]
    by_policy = {row["policy"]: row for row in profile}
    relief_mean = by_policy["anchor:relief"]["mean_return"]
    for row in profile:
        row["vs_relief"] = round(row["mean_return"] - relief_mean, 4)

    constraints = [
        ("anchor:relief", "anchor:noop"),
        ("anchor:relief", "anchor:random-valid"),
        ("anchor:noop", "anchor:catastrophic"),
        ("anchor:random-valid", "anchor:catastrophic"),
    ]
    anchor_comparisons = [
        _comparison(
            left_key="better",
            left=better,
            right_key="worse",
            right=worse,
            left_returns=returns[better],
            right_returns=returns[worse],
            scenarios=scenarios,
            seed=10_000 + index,
        )
        for index, (better, worse) in enumerate(constraints)
    ]
    model_comparison = _comparison(
        left_key="stronger",
        left="model:large",
        right_key="weaker",
        right="model:small",
        left_returns=returns["model:large"],
        right_returns=returns["model:small"],
        scenarios=scenarios,
        seed=0,
    )
    return {
        "backend": "replay",
        "model_specs": [
            {"label": "small", "model": "qwen3-1.7b", "capability_rank": 1},
            {"label": "large", "model": "qwen3-8b", "capability_rank": 2},
        ],
        "sampling_contract": {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 512,
            "seed_derivation": "sha256(run1b-request-v1:{prompt_index}:{response_index}:{step_index}) mod 2**31",
            "seed_version": "run1b-request-v1",
            "tool_choice": "required",
            "parallel_tool_calls": False,
        },
        "tasks": [
            {
                "prompt_index": index,
                "seed": seed,
                "difficulty": "hard",
                "scenario_id": scenario,
                "regime_mix": {scenario: 1.0},
            }
            for index, (seed, scenario) in enumerate(zip((10, 20), scenarios, strict=True))
        ],
        "prompts": 2,
        "responses_per_prompt": 2,
        "episodes_per_policy": 4,
        "concurrency": 2,
        "compliance_profile": False,
        "failure_rate_ceiling": 0.0,
        "profile": profile,
        "anchor_ordering_ok": True,
        "anchor_order_expected": [
            "anchor:relief",
            "anchor:noop",
            "anchor:random-valid",
            "anchor:catastrophic",
        ],
        "anchor_order_constraints": [list(pair) for pair in constraints],
        "anchor_ordering_comparisons": anchor_comparisons,
        "model_ordering": {
            "status": "PASS",
            "engineering_status": "PASS",
            "quality_status": "PASS",
            "expected": ["model:small", "model:large"],
            "observed": {
                "model:small": by_policy["model:small"]["mean_return"],
                "model:large": by_policy["model:large"]["mean_return"],
            },
            "failure_counts": {
                "model:small": {"parse_failures": 0, "invalid_calls": 0, "infra_errors": 0},
                "model:large": {"parse_failures": 0, "invalid_calls": 0, "infra_errors": 0},
            },
            "failure_rates": {"model:small": 0.0, "model:large": 0.0},
            "failure_rate_ceiling": 0.0,
            "valid_paired_episodes": 4,
            "valid_paired_prompt_clusters": 2,
            "comparisons": [model_comparison],
            "reason": None,
        },
        "model_ordering_ok": True,
    }


def _metadata() -> dict[str, Any]:
    return {
        "run_id": "run1b_fixture_a3897fc",
        "backend": "replay",
        "source": {"commit": "a" * 40, "dirty": False},
        "sampling": {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 512,
            "seed_derivation": "sha256(run1b-request-v1:{prompt_index}:{response_index}:{step_index}) mod 2**31",
            "seed_version": "run1b-request-v1",
            "tool_choice": "required",
            "parallel_tool_calls": False,
        },
        "models": [
            {
                "policy": "model:small",
                "label": "small",
                "model_id": "Qwen/Qwen3-1.7B",
                "served_model_id": "qwen3-1.7b",
                "revision": "small-revision",
                "tokenizer_revision": "small-tokenizer-revision",
                "capability_rank": 1,
            },
            {
                "policy": "model:large",
                "label": "large",
                "model_id": "Qwen/Qwen3-8B",
                "served_model_id": "qwen3-8b",
                "revision": "large-revision",
                "tokenizer_revision": "large-tokenizer-revision",
                "capability_rank": 2,
            },
        ],
        "environment": {
            "hostname": "fixture-host",
            "gpu": ["NVIDIA H100 NVL"],
            "commands": [{"argv": ["python", "model_sweep.py"], "exit_code": 0}],
            "runtime": {"python": "3.12.3", "platform": "linux-x86_64"},
            "start_utc": "2026-08-08T12:00:00Z",
            "end_utc": "2026-08-08T13:00:00Z",
            "image": {
                "digest": "sha256:" + "b" * 64,
                "signature_status": "VERIFIED",
            },
        },
        "provenance": {
            "task_manifest_sha256": "1" * 64,
            "tool_schema_sha256": "2" * 64,
            "reward_sha256": "3" * 64,
            "dynamics_sha256": "4" * 64,
            "renderer_sha256": "5" * 64,
        },
        "run_log": "2026-08-08T12:00:00Z benchmark start\n2026-08-08T13:00:00Z benchmark end\n",
    }


def _write_inputs(tmp_path: Path, report: dict[str, Any], metadata: dict[str, Any]) -> tuple[Path, Path]:
    raw_path = tmp_path / "source-report.json"
    metadata_path = tmp_path / "metadata.json"
    raw_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=1) + "\n", encoding="utf-8")
    return raw_path, metadata_path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_build_package_is_complete_deterministic_and_hash_verifiable(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    module = _report_module()
    raw_path, metadata_path = _write_inputs(tmp_path, _raw_report(), _metadata())
    first = tmp_path / "first"
    second = tmp_path / "second"

    module.build_report_package(raw_path, metadata_path, first)
    module.build_report_package(raw_path, metadata_path, second)

    expected = {
        "benchmark_contract.json",
        "models.json",
        "raw_report.json",
        "episodes.jsonl",
        "summary.json",
        "summary.csv",
        "paired_deltas.csv",
        "benchmark.png",
        "benchmark.svg",
        "environment.json",
        "run.log",
        "SHA256SUMS",
    }
    assert {path.name for path in first.iterdir()} == expected
    assert {path.name for path in second.iterdir()} == expected
    assert (first / "raw_report.json").read_bytes() == raw_path.read_bytes()
    assert all((first / name).read_bytes() == (second / name).read_bytes() for name in expected)

    models = json.loads((first / "models.json").read_text(encoding="utf-8"))
    assert [model["policy"] for model in models] == ["model:small", "model:large"]
    assert all("base_url" not in model and "api_key_env" not in model for model in models)

    episodes = [json.loads(line) for line in (first / "episodes.jsonl").read_text().splitlines()]
    assert len(episodes) == 24
    assert episodes[0]["pair_key"] == "0:0"
    assert episodes[-1]["policy"] == "model:large"
    assert all(math.isfinite(row["return"]) for row in episodes)

    paired = _read_csv(first / "paired_deltas.csv")
    assert len(paired) == 10
    model_rows = [row for row in paired if row["comparison_kind"] == "model"]
    assert [float(row["prompt_mean_delta"]) for row in model_rows] == [2.0, 2.0]
    assert [int(row["response_pairs"]) for row in model_rows] == [2, 2]
    assert {row["run_id"] for row in paired} == {_metadata()["run_id"]}
    assert {row["source_commit"] for row in paired} == {_metadata()["source"]["commit"]}
    summary_rows = _read_csv(first / "summary.csv")
    assert {row["run_id"] for row in summary_rows} == {_metadata()["run_id"]}
    assert {row["source_commit"] for row in summary_rows} == {_metadata()["source"]["commit"]}
    assert (first / "run.log").read_text(encoding="utf-8") == _metadata()["run_log"]

    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == _metadata()["run_id"]
    assert summary["model_ordering_status"] == "PASS"
    assert {row["evaluation_status"] for row in summary["policies"] if row["kind"] == "model"} == {"EVALUABLE"}

    svg = (first / "benchmark.svg").read_text(encoding="utf-8")
    assert _metadata()["run_id"] in svg
    assert "2026-" not in svg
    assert (first / "benchmark.png").stat().st_size > 1_000

    checksum_lines = (first / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert checksum_lines == sorted(checksum_lines, key=lambda line: line.split("  ", 1)[1])
    assert len(checksum_lines) == len(expected) - 1
    for line in checksum_lines:
        digest, name = line.split("  ", 1)
        assert digest == hashlib.sha256((first / name).read_bytes()).hexdigest()


def test_failed_model_is_not_evaluable_and_keeps_its_diagnostic_mean(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    module = _report_module()
    report = _raw_report()
    large = next(row for row in report["profile"] if row["policy"] == "model:large")
    large["episode_records"][0]["parse_failures"] = 1
    large["episode_records"][0]["usable"] = False
    large["parse_failures"] = 1
    large["usable_episodes"] = 3
    report["model_ordering"] = {
        "status": "NOT_EVALUABLE",
        "expected": ["model:small", "model:large"],
        "observed": {"model:small": 1.5, "model:large": 3.5},
        "failure_counts": {
            "model:small": {"parse_failures": 0, "invalid_calls": 0, "infra_errors": 0},
            "model:large": {"parse_failures": 1, "invalid_calls": 0, "infra_errors": 0},
        },
        "failure_rates": {"model:small": 0.0, "model:large": 0.25},
        "failure_rate_ceiling": 0.0,
        "reason": "model:large has a parse failure",
    }
    report["model_ordering_ok"] = None
    raw_path, metadata_path = _write_inputs(tmp_path, report, _metadata())
    output = tmp_path / "package"

    module.build_report_package(raw_path, metadata_path, output)

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    rows = {row["policy"]: row for row in summary["policies"]}
    assert summary["model_ordering_reason"] == "model:large has a parse failure"
    assert rows["model:large"]["evaluation_status"] == "NOT_EVALUABLE"
    assert rows["model:large"]["mean_return"] == 3.5
    assert rows["model:large"]["mean_return"] != 0.0
    assert rows["model:small"]["evaluation_status"] == "EVALUABLE"
    svg = (output / "benchmark.svg").read_text(encoding="utf-8")
    assert "large - NOT EVALUABLE" in svg
    assert not [row for row in _read_csv(output / "paired_deltas.csv") if row["comparison_kind"] == "model"]


def test_infrastructure_drop_is_packaged_as_not_evaluable_diagnostic(tmp_path: Path, monkeypatch):
    """Removing the infra-diagnostic branch would make a real dropped episode unpackageable."""

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    module = _report_module()
    report = _raw_report()
    large = next(row for row in report["profile"] if row["policy"] == "model:large")
    large["episode_records"] = large["episode_records"][:3]
    large.update(
        {
            "episodes": 3,
            "mean_return": 2.6667,
            "std_return": 0.9428,
            "return_distribution": {
                "min": 2.0,
                "p05": 2.0,
                "p25": 2.0,
                "median": 2.0,
                "p75": 3.0,
                "p95": 3.8,
                "max": 4.0,
            },
            "infra_errors": 1,
            "usable_episodes": 3,
            "tool_metrics": {
                "noop": {
                    "calls": 6,
                    "call_rate": 1.0,
                    "mean_step_reward": 1.333333,
                    "rejection_rate": 0.0,
                }
            },
            "returns_by_scenario": {
                "bursty": {"episodes": 3, "mean_return": 2.666667, "std_return": 0.942809},
            },
            "vs_relief": -0.3333,
        }
    )
    report["model_ordering"] = {
        "status": "NOT_EVALUABLE",
        "expected": ["model:small", "model:large"],
        "observed": {"model:small": 1.5, "model:large": 2.6667},
        "failure_counts": {
            "model:small": {"parse_failures": 0, "invalid_calls": 0, "infra_errors": 0},
            "model:large": {"parse_failures": 0, "invalid_calls": 0, "infra_errors": 1},
        },
        "failure_rates": {"model:small": 0.0, "model:large": 0.25},
        "failure_rate_ceiling": 0.0,
        "reason": "model:large has one infrastructure drop",
    }
    report["model_ordering_ok"] = None
    raw_path, metadata_path = _write_inputs(tmp_path, report, _metadata())
    output = tmp_path / "package"

    module.build_report_package(raw_path, metadata_path, output)

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    large_summary = next(row for row in summary["policies"] if row["policy"] == "model:large")
    assert summary["model_ordering_status"] == "NOT_EVALUABLE"
    assert large_summary["evaluation_status"] == "NOT_EVALUABLE"
    assert large_summary["episodes"] == 3
    assert large_summary["expected_episodes"] == 4
    assert large_summary["infra_errors"] == 1
    assert large_summary["mean_return"] == 2.6667


def test_underpowered_smoke_preserves_engineering_pass_and_quality_not_evaluable(tmp_path: Path, monkeypatch):
    """A clean one-cluster/regime smoke is launch evidence, not a quality claim."""

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    report = _raw_report(["prb_exhaustion", "bursty"])
    report["model_ordering"] = {
        "status": "NOT_EVALUABLE",
        "engineering_status": "PASS",
        "quality_status": "NOT_EVALUABLE",
        "expected": ["model:small", "model:large"],
        "observed": {"model:small": 1.5, "model:large": 3.5},
        "failure_counts": {
            "model:small": {"parse_failures": 0, "invalid_calls": 0, "infra_errors": 0},
            "model:large": {"parse_failures": 0, "invalid_calls": 0, "infra_errors": 0},
        },
        "failure_rates": {"model:small": 0.0, "model:large": 0.0},
        "failure_rate_ceiling": 0.0,
        "valid_paired_episodes": 4,
        "comparisons": [],
        "reason": (
            "model-quality inference requires at least two prompt clusters per scenario; "
            "underpowered scenarios: ['bursty', 'prb_exhaustion']"
        ),
    }
    report["model_ordering_ok"] = None
    raw_path, metadata_path = _write_inputs(tmp_path, report, _metadata())
    output = tmp_path / "package"

    _report_module().build_report_package(raw_path, metadata_path, output)

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["model_ordering_status"] == "NOT_EVALUABLE"
    assert summary["model_engineering_status"] == "PASS"
    assert summary["model_quality_status"] == "NOT_EVALUABLE"
    assert not [row for row in _read_csv(output / "paired_deltas.csv") if row["comparison_kind"] == "model"]
    assert "QUALITY NOT EVALUABLE" in (output / "benchmark.svg").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("mutate_report", "mutate_metadata", "message"),
    [
        (
            lambda report: report["sampling_contract"].__setitem__("temperature", 0.3),
            lambda metadata: None,
            "sampling",
        ),
        (
            lambda report: None,
            lambda metadata: metadata["models"][0].__setitem__("served_model_id", "wrong-model"),
            "served_model_id",
        ),
        (
            lambda report: report.__setitem__("backend", "dataset_replay"),
            lambda metadata: None,
            "backend",
        ),
    ],
)
def test_metadata_must_match_raw_execution_contract(
    tmp_path: Path,
    monkeypatch,
    mutate_report: Callable[[dict[str, Any]], None],
    mutate_metadata: Callable[[dict[str, Any]], None],
    message: str,
):
    """Deleting raw/metadata binding would let callers relabel an unrelated run."""

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    report, metadata = _raw_report(), _metadata()
    mutate_report(report)
    mutate_metadata(metadata)
    raw_path, metadata_path = _write_inputs(tmp_path, report, metadata)

    with pytest.raises(ValueError, match=message):
        _report_module().build_report_package(raw_path, metadata_path, tmp_path / "package")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 0.3),
        ("top_p", 0.9),
        ("max_tokens", 256),
        ("seed_version", "other-seed-contract"),
        ("seed_derivation", "arbitrary caller supplied seed"),
    ],
)
def test_run1b_report_rejects_matching_but_nonfrozen_sampling(
    tmp_path: Path, monkeypatch, field: str, value: Any
):
    """Matching metadata cannot relabel a noncontract run as Run 1B."""

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    report, metadata = _raw_report(), _metadata()
    report["sampling_contract"][field] = value
    metadata["sampling"][field] = value
    raw_path, metadata_path = _write_inputs(tmp_path, report, metadata)

    with pytest.raises(ValueError, match="frozen Run 1B sampling contract"):
        _report_module().build_report_package(raw_path, metadata_path, tmp_path / "package")


def test_run1b_report_rejects_dirty_source_custody(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    metadata = _metadata()
    metadata["source"]["dirty"] = True
    raw_path, metadata_path = _write_inputs(tmp_path, _raw_report(), metadata)

    with pytest.raises(ValueError, match="clean source"):
        _report_module().build_report_package(raw_path, metadata_path, tmp_path / "package")


def test_compliance_label_requires_exact_full_support(tmp_path: Path, monkeypatch):
    """Weakening the support check would let a smoke masquerade as the full profile."""

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    report = _raw_report()
    report["compliance_profile"] = True
    raw_path, metadata_path = _write_inputs(tmp_path, report, _metadata())

    with pytest.raises(ValueError, match="500 prompts.*16 responses"):
        _report_module().build_report_package(raw_path, metadata_path, tmp_path / "package")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("bootstrap_seed", 7, "bootstrap_seed"), ("bootstrap_draws", 1, "bootstrap_draws")],
)
def test_bootstrap_seed_and_draw_count_are_frozen(tmp_path: Path, monkeypatch, field: str, value: int, message: str):
    """Trusting raw bootstrap settings would permit seed shopping or undersampling."""

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    report = _raw_report()
    report["anchor_ordering_comparisons"][0][field] = value
    raw_path, metadata_path = _write_inputs(tmp_path, report, _metadata())

    with pytest.raises(ValueError, match=message):
        _report_module().build_report_package(raw_path, metadata_path, tmp_path / "package")


def test_anchor_gate_requires_exact_frozen_order_and_constraints(tmp_path: Path, monkeypatch):
    """Default-true aggregation must not turn an omitted anchor gate into PASS."""

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    report = _raw_report()
    report["anchor_order_constraints"] = []
    report["anchor_ordering_comparisons"] = []
    report["anchor_ordering_ok"] = True
    raw_path, metadata_path = _write_inputs(tmp_path, report, _metadata())

    with pytest.raises(ValueError, match="anchor_order_constraints"):
        _report_module().build_report_package(raw_path, metadata_path, tmp_path / "package")


def test_anchor_gate_cannot_pass_with_an_unusable_anchor_record(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    report = _raw_report()
    relief = next(row for row in report["profile"] if row["policy"] == "anchor:relief")
    relief["episode_records"][0]["invalid_calls"] = 1
    relief["episode_records"][0]["usable"] = False
    relief["invalid_calls"] = 1
    relief["usable_episodes"] -= 1
    # Keep the positive reward comparisons and the claimed PASS unchanged;
    # the reporter must independently reject that fail-open combination.
    raw_path, metadata_path = _write_inputs(tmp_path, report, _metadata())

    with pytest.raises(ValueError, match="anchor_ordering_ok"):
        _report_module().build_report_package(raw_path, metadata_path, tmp_path / "package")


def test_task_support_requires_one_hot_supported_regime(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    report = _raw_report()
    report["tasks"][0]["regime_mix"] = {"bursty": 0.5, "interference": 0.5}
    raw_path, metadata_path = _write_inputs(tmp_path, report, _metadata())

    with pytest.raises(ValueError, match="one-hot regime_mix"):
        _report_module().build_report_package(raw_path, metadata_path, tmp_path / "package")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report["profile"][0]["episode_records"].__setitem__(
                1, copy.deepcopy(report["profile"][0]["episode_records"][0])
            ),
            "duplicate pair_key",
        ),
        (
            lambda report: report["profile"][0]["episode_records"].pop(),
            "episode_records",
        ),
        (
            lambda report: report["profile"][0]["episode_records"][0].__setitem__("return", float("nan")),
            "finite",
        ),
        (
            lambda report: report["model_ordering"]["comparisons"][0].__setitem__("mean_delta", 999.0),
            "mean_delta",
        ),
        (
            lambda report: report["model_ordering"]["comparisons"][0].__setitem__("ci95_high", 999.0),
            "ci95_high",
        ),
    ],
)
def test_validation_rejects_bad_episode_or_comparison_data(
    tmp_path: Path,
    monkeypatch,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    module = _report_module()
    report = _raw_report()
    mutate(report)
    raw_path, metadata_path = _write_inputs(tmp_path, report, _metadata())

    with pytest.raises(ValueError, match=message):
        module.build_report_package(raw_path, metadata_path, tmp_path / "package")


def test_metadata_with_a_credential_is_rejected(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    module = _report_module()
    metadata = _metadata()
    metadata["environment"]["api_key"] = "do-not-write-me"
    raw_path, metadata_path = _write_inputs(tmp_path, _raw_report(), metadata)

    with pytest.raises(ValueError, match="credential-like key"):
        module.build_report_package(raw_path, metadata_path, tmp_path / "package")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda metadata: metadata["environment"]["commands"][0]["argv"].extend(["--api-key", "supersecret"]),
        lambda metadata: metadata["environment"]["commands"][0]["argv"].extend(["--token=supersecret"]),
        lambda metadata: metadata.__setitem__("run_log", "HF_TOKEN=hf_do_not_package\n"),
        lambda metadata: metadata.__setitem__("run_log", "2026-08-08T12:00:00Z vllm --api-key supersecret\n"),
        lambda metadata: metadata.__setitem__("run_log", "2026-08-08T12:00:00Z export OPENAI_API_KEY=supersecret\n"),
        lambda metadata: metadata["environment"].__setitem__("clientSecret", "do-not-package"),
        lambda metadata: metadata.__setitem__("run_log", "-----BEGIN PRIVATE KEY-----\ndo-not-package\n"),
    ],
)
def test_cli_and_common_secret_forms_are_rejected_before_any_file_is_written(
    tmp_path: Path, monkeypatch, mutate: Callable[[dict[str, Any]], None]
):
    """Narrow key-name filtering must not leak credentials into handoff artifacts."""

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    metadata = _metadata()
    mutate(metadata)
    raw_path, metadata_path = _write_inputs(tmp_path, _raw_report(), metadata)
    output = tmp_path / "package"

    with pytest.raises(ValueError, match="credential|secret|private key"):
        _report_module().build_report_package(raw_path, metadata_path, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda metadata: metadata.pop("provenance"), "metadata.provenance"),
        (lambda metadata: metadata["provenance"].pop("reward_sha256"), "reward_sha256"),
        (lambda metadata: metadata["environment"].pop("commands"), "commands"),
        (lambda metadata: metadata["environment"].pop("runtime"), "runtime"),
        (lambda metadata: metadata["environment"].pop("gpu"), "gpu"),
        (lambda metadata: metadata["environment"].pop("start_utc"), "start_utc"),
        (lambda metadata: metadata["environment"]["image"].pop("signature_status"), "signature_status"),
        (lambda metadata: metadata.__setitem__("run_log", ""), "run_log"),
    ],
)
def test_strict_custody_metadata_is_required(
    tmp_path: Path,
    monkeypatch,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    metadata = _metadata()
    mutate(metadata)
    raw_path, metadata_path = _write_inputs(tmp_path, _raw_report(), metadata)

    with pytest.raises(ValueError, match=message):
        _report_module().build_report_package(raw_path, metadata_path, tmp_path / "package")


def test_raw_report_with_a_credential_is_rejected_before_copy(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    module = _report_module()
    report = _raw_report()
    report["authorization"] = "Bearer do-not-write-me"
    raw_path, metadata_path = _write_inputs(tmp_path, report, _metadata())

    with pytest.raises(ValueError, match="credential-like key"):
        module.build_report_package(raw_path, metadata_path, tmp_path / "package")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report["model_ordering"].__setitem__("failure_rate_ceiling", 0.1),
            "failure_rate_ceiling",
        ),
        (
            lambda report: report["model_ordering"].__setitem__("valid_paired_episodes", 3),
            "valid_paired_episodes",
        ),
        (
            lambda report: report["profile"][0]["return_distribution"].__setitem__("p95", 999.0),
            "return_distribution.p95",
        ),
        (
            lambda report: report["profile"][0]["returns_by_scenario"]["bursty"].__setitem__("episodes", 999),
            "returns_by_scenario",
        ),
        (
            lambda report: report["profile"][0]["tool_metrics"]["noop"].__setitem__("calls", 999),
            "tool_metrics.noop.calls",
        ),
        (
            lambda report: report["profile"][0]["episode_return_correlations"].__setitem__("noop_rate", 0.5),
            "episode_return_correlations.noop_rate",
        ),
    ],
)
def test_auxiliary_statistics_and_failure_support_are_recomputed(
    tmp_path: Path,
    monkeypatch,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    report = _raw_report()
    mutate(report)
    raw_path, metadata_path = _write_inputs(tmp_path, report, _metadata())

    with pytest.raises(ValueError, match=message):
        _report_module().build_report_package(raw_path, metadata_path, tmp_path / "package")


def test_comparison_receipts_accept_full_precision_values():
    """Real sweep comparisons must not be forced back to six-decimal serialization."""

    module = _report_module()
    delta = 0.123456789123
    prompt_rows = [{"prompt_index": index, "scenario_id": "bursty", "prompt_mean_delta": delta} for index in range(2)]
    low, high = module._clustered_bootstrap_ci(prompt_rows, seed=0, draws=10_000)
    better = {
        f"{prompt}:{response}": {
            "scenario_id": "bursty",
            "return": delta,
        }
        for prompt in range(2)
        for response in range(2)
    }
    worse = {
        f"{prompt}:{response}": {
            "scenario_id": "bursty",
            "return": 0.0,
        }
        for prompt in range(2)
        for response in range(2)
    }
    comparison = {
        "pairs": 4,
        "paired_episodes": 4,
        "response_pairs": 4,
        "prompt_clusters": 2,
        "mean_delta": delta,
        "median_delta": delta,
        "ci95_low": low,
        "ci95_high": high,
        "prompt_wins": 2,
        "prompt_ties": 0,
        "prompt_losses": 0,
        "bootstrap_method": "regime_stratified_prompt_cluster_percentile",
        "bootstrap_seed": 0,
        "bootstrap_draws": 10_000,
        "status": "PASS",
    }

    rows = module._validate_comparison(
        comparison,
        path="comparison",
        better_policy="model:large",
        worse_policy="model:small",
        better_records=better,
        worse_records=worse,
        prompts=2,
        responses=2,
        expected_seed=0,
        expected_draws=10_000,
    )

    assert rows[0]["mean_delta"] == delta


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda metadata: metadata["sampling"].__setitem__("top_p", 0.0), "top_p"),
        (lambda metadata: metadata["sampling"].__setitem__("temperature", 2.01), "temperature"),
        (
            lambda metadata: metadata["environment"]["commands"][0].__setitem__("exit_code", True),
            "exit_code",
        ),
        (
            lambda metadata: metadata["environment"].__setitem__("end_utc", "2026-08-08T11:00:00Z"),
            "end_utc",
        ),
    ],
)
def test_metadata_ranges_and_temporal_order_are_strict(
    tmp_path: Path,
    monkeypatch,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    metadata = _metadata()
    mutate(metadata)
    raw_path, metadata_path = _write_inputs(tmp_path, _raw_report(), metadata)

    with pytest.raises(ValueError, match=message):
        _report_module().build_report_package(raw_path, metadata_path, tmp_path / "package")


def test_bootstrap_recomputation_matches_sweep_integer_prompt_key_order():
    module = _report_module()
    prompt_rows = [
        {
            "prompt_index": prompt_index,
            "scenario_id": "same-regime",
            "prompt_mean_delta": float((prompt_index * prompt_index) % 17),
        }
        for prompt_index in range(12)
    ]
    # model_sweep sorts typed (scenario_id, integer prompt_index) cluster keys
    # before sampling; lexical order diverges once prompt 10 exists.
    ordered = np.asarray(
        [
            row["prompt_mean_delta"]
            for row in sorted(prompt_rows, key=lambda row: (row["scenario_id"], row["prompt_index"]))
        ]
    )
    rng = np.random.default_rng(19)
    indices = rng.integers(0, len(ordered), size=(257, len(ordered)))
    expected = tuple(float(value) for value in np.quantile(ordered[indices].mean(axis=1), [0.025, 0.975]))

    assert module._clustered_bootstrap_ci(prompt_rows, seed=19, draws=257) == expected


def test_cli_builds_the_same_package_contract(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    module = _report_module()
    raw_path, metadata_path = _write_inputs(tmp_path, _raw_report(), _metadata())
    output = tmp_path / "package"

    assert module.main(["--raw-report", str(raw_path), "--metadata", str(metadata_path), "--out", str(output)]) == 0
    assert json.loads((output / "benchmark_contract.json").read_text())["run_id"] == _metadata()["run_id"]
