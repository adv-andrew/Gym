# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and package one Run 1B model-sweep report.

The reporter is intentionally downstream of ``model_sweep.py``.  It does not
contact model servers or combine reports.  It independently recomputes the
report's statistics, validates them against explicit credential-free run
metadata, and writes reviewable tables, figures, and a custody manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PAIR_KEY = re.compile(r"^(0|[1-9][0-9]*):(0|[1-9][0-9]*)$")
_BOOTSTRAP_METHOD = "regime_stratified_prompt_cluster_percentile"
_DEFAULT_BOOTSTRAP_DRAWS = 10_000
_COMPLIANCE_BOOTSTRAP_DRAWS = 50_000
_RUN1B_SAMPLING = {
    "temperature": 0.2,
    "top_p": 0.95,
    "max_tokens": 512,
    "seed_derivation": "sha256(run1b-request-v1:{prompt_index}:{response_index}:{step_index}) mod 2**31",
    "seed_version": "run1b-request-v1",
    "tool_choice": "required",
    "parallel_tool_calls": False,
    "chat_template_kwargs": {"enable_thinking": False},
}
_ANCHOR_ORDER = (
    "anchor:relief",
    "anchor:noop",
    "anchor:random-valid",
    "anchor:catastrophic",
)
_ANCHOR_CONSTRAINTS = (
    ("anchor:relief", "anchor:noop"),
    ("anchor:relief", "anchor:random-valid"),
    ("anchor:noop", "anchor:catastrophic"),
    ("anchor:random-valid", "anchor:catastrophic"),
)
_SUPPORTED_REGIMES = {
    "bursty",
    "interference",
    "prach_storm",
    "prb_exhaustion",
    "qos_competition",
}
_SAMPLING_FIELDS = {
    "temperature",
    "top_p",
    "max_tokens",
    "seed_derivation",
    "seed_version",
    "tool_choice",
    "parallel_tool_calls",
    "chat_template_kwargs",
}
_HASH_FIELDS = {
    "task_manifest_sha256",
    "tool_schema_sha256",
    "reward_sha256",
    "dynamics_sha256",
    "renderer_sha256",
}
_CREDENTIAL_KEYS = {
    "api_key",
    "authorization",
    "bearer_token",
    "credential",
    "credentials",
    "password",
    "refresh_token",
    "secret",
    "access_token",
    "api_token",
    "auth_token",
    "client_secret",
    "private_key",
    "signing_key",
    "token",
    "x_api_key",
    "aws_access_key_id",
    "aws_secret_access_key",
}
_SECRET_FLAG = re.compile(
    r"(?i)(?:^|\s)--(?:api[-_]?key|access[-_]?token|auth[-_]?token|bearer[-_]?token|client[-_]?secret|"
    r"hf[-_]?token|openai[-_]?api[-_]?key|password|token)(?:=|\s|$)"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?:^|\s)(?:HF_TOKEN|OPENAI_API_KEY|HUGGINGFACE_TOKEN|API_KEY|X_API_KEY|ACCESS_TOKEN|AUTH_TOKEN|"
    r"BEARER_TOKEN|CLIENT_SECRET|PASSWORD|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY)\s*="
)
_SECRET_VALUE = re.compile(
    r"(?i)(?:\bsk-[A-Za-z0-9_-]{8,}|\bhf_[A-Za-z0-9]{8,}|\bghp_[A-Za-z0-9]{8,}|"
    r"\bgithub_pat_[A-Za-z0-9_]{8,}|\bglpat-[A-Za-z0-9_-]{8,}|\bAKIA[0-9A-Z]{16}\b)"
)
_ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_MODEL_PUBLIC_FIELDS = (
    "policy",
    "label",
    "model_id",
    "served_model_id",
    "revision",
    "model_revision",
    "tokenizer_revision",
    "capability_rank",
    "weights_sha256",
    "tokenizer_sha256",
)
_SUMMARY_FIELDS = (
    "run_id",
    "source_commit",
    "policy",
    "kind",
    "evaluation_status",
    "capability_rank",
    "model_id",
    "served_model_id",
    "model_revision",
    "episodes",
    "expected_episodes",
    "mean_return",
    "std_return",
    "rejection_rate",
    "noop_rate",
    "parse_failures",
    "invalid_calls",
    "infra_errors",
    "usable_episodes",
)
_PAIRED_FIELDS = (
    "run_id",
    "source_commit",
    "comparison_kind",
    "comparison_index",
    "better_policy",
    "worse_policy",
    "prompt_index",
    "scenario_id",
    "response_pairs",
    "prompt_mean_delta",
    "prompt_median_delta",
    "prompt_outcome",
    "comparison_status",
    "bootstrap_method",
    "bootstrap_seed",
    "bootstrap_draws",
    "mean_delta",
    "ci95_low",
    "ci95_high",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _load_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"could not read {label} {path}: {error}") from error
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be one UTF-8 JSON object: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    _require_finite_tree(value, label)
    return value, raw


def _require_finite_tree(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must contain only finite numbers")
    if isinstance(value, dict):
        for key, nested in value.items():
            _require_finite_tree(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _require_finite_tree(nested, f"{path}[{index}]")


def _normalized_key(value: Any) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value).strip())
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def _reject_credentials(value: Any, path: str = "metadata") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = _normalized_key(key)
            if normalized in _CREDENTIAL_KEYS:
                raise ValueError(f"credential-like key is forbidden in package metadata: {path}.{key}")
            _reject_credentials(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            if isinstance(nested, str) and _SECRET_FLAG.search(nested):
                raise ValueError(f"credential-like CLI flag is forbidden in package metadata: {path}[{index}]")
            _reject_credentials(nested, f"{path}[{index}]")
    elif isinstance(value, str):
        if re.search(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+=*", value):
            raise ValueError(f"credential-like value is forbidden in package metadata: {path}")
        if re.search(r"^[a-z][a-z0-9+.-]*://[^/@\s]+:[^/@\s]+@", value, flags=re.IGNORECASE):
            raise ValueError(f"credential-bearing URL is forbidden in package metadata: {path}")
        if _SECRET_FLAG.search(value) or _SECRET_ASSIGNMENT.search(value) or _SECRET_VALUE.search(value):
            raise ValueError(f"credential-like secret is forbidden in package metadata: {path}")
        if re.search(r"(?i)-----BEGIN (?:ENCRYPTED )?(?:RSA |EC |OPENSSH )?PRIVATE KEY-----", value):
            raise ValueError(f"private key material is forbidden in package metadata: {path}")


def _require_keys(value: Any, *, required: set[str], allowed: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        qualified = [f"{path}.{field}" for field in missing]
        raise ValueError(f"{path} is missing required fields: {qualified}")
    if unknown:
        raise ValueError(f"{path} contains unsupported fields: {unknown}")
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _sha256(value: Any, path: str) -> str:
    text = _nonempty_string(value, path)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return text


def _utc_timestamp(value: Any, path: str) -> datetime:
    text = _nonempty_string(value, path)
    if not text.endswith("Z"):
        raise ValueError(f"{path} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{path} must be an ISO-8601 UTC timestamp") from error
    if parsed.tzinfo != timezone.utc:
        raise ValueError(f"{path} must be UTC")
    return parsed


def _validate_sampling(value: Any, path: str) -> dict[str, Any]:
    sampling = _require_keys(value, required=_SAMPLING_FIELDS, allowed=_SAMPLING_FIELDS, path=path)
    top_p = _finite_number(sampling["top_p"], f"{path}.top_p")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"{path}.top_p must be greater than zero and at most one")
    temperature = _finite_number(sampling["temperature"], f"{path}.temperature")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError(f"{path}.temperature must be between zero and two")
    _positive_int(sampling["max_tokens"], f"{path}.max_tokens")
    _nonempty_string(sampling["seed_derivation"], f"{path}.seed_derivation")
    _nonempty_string(sampling["seed_version"], f"{path}.seed_version")
    if sampling["tool_choice"] != "required":
        raise ValueError(f"{path}.tool_choice must be 'required'")
    if sampling["parallel_tool_calls"] is not False:
        raise ValueError(f"{path}.parallel_tool_calls must be false")
    template_kwargs = _require_keys(
        sampling["chat_template_kwargs"],
        required={"enable_thinking"},
        allowed={"enable_thinking"},
        path=f"{path}.chat_template_kwargs",
    )
    if not isinstance(template_kwargs["enable_thinking"], bool):
        raise ValueError(f"{path}.chat_template_kwargs.enable_thinking must be boolean")
    observed = {field: sampling[field] for field in _RUN1B_SAMPLING}
    if observed != _RUN1B_SAMPLING:
        raise ValueError(
            f"{path} must equal the frozen Run 1B sampling contract {_RUN1B_SAMPLING}, got {observed}"
        )
    normalized = dict(sampling)
    normalized["chat_template_kwargs"] = {
        "enable_thinking": template_kwargs["enable_thinking"]
    }
    return normalized


def _sanitize_run_log(value: Any) -> str:
    log = _nonempty_string(value, "metadata.run_log")
    _reject_credentials(log, "metadata.run_log")
    normalized = _ANSI_ESCAPE.sub("", log.replace("\r\n", "\n").replace("\r", "\n"))
    if any(ord(character) < 32 and character not in "\n\t" for character in normalized):
        raise ValueError("metadata.run_log contains unsupported control characters")
    if len(normalized.encode("utf-8")) > 10_000_000:
        raise ValueError("metadata.run_log exceeds the 10 MB custody limit")
    return normalized.rstrip("\n") + "\n"


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _positive_int(value: Any, path: str) -> int:
    result = _nonnegative_int(value, path)
    if result == 0:
        raise ValueError(f"{path} must be positive")
    return result


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{path} must be a finite number")
    return float(value)


def _rate(value: Any, path: str) -> float:
    result = _finite_number(value, path)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{path} must be between zero and one")
    return result


def _same_reported(actual: float, reported: Any, digits: int, path: str) -> None:
    expected = round(actual, digits)
    value = _finite_number(reported, path)
    if value != expected:
        raise ValueError(f"{path} is inconsistent: expected {expected}, got {value}")


def _same_precise(actual: float, reported: Any, path: str) -> float:
    """Validate a full-precision sweep statistic without imposing rounding."""

    value = _finite_number(reported, path)
    if not math.isclose(value, actual, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{path} is inconsistent: expected {actual!r}, got {value!r}")
    return value


def _same_optional_reported(actual: float | None, reported: Any, digits: int, path: str) -> None:
    if actual is None:
        if reported is not None:
            raise ValueError(f"{path} is inconsistent: expected null, got {reported!r}")
        return
    _same_reported(actual, reported, digits, path)


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
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    dx = [value - mean_x for value in xs]
    dy = [value - mean_y for value in ys]
    denominator = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denominator == 0.0:
        return None
    return sum(left * right for left, right in zip(dx, dy, strict=True)) / denominator


def _validate_auxiliary_statistics(
    row: Mapping[str, Any], records: list[dict[str, Any]], *, infra_errors: int, path: str
) -> None:
    returns = [float(record["return"]) for record in records]
    distribution = _require_keys(
        row.get("return_distribution"),
        required={"min", "p05", "p25", "median", "p75", "p95", "max"},
        allowed={"min", "p05", "p25", "median", "p75", "p95", "max"},
        path=f"{path}.return_distribution",
    )
    expected_distribution = {
        "min": min(returns) if returns else None,
        "p05": _quantile(returns, 0.05),
        "p25": _quantile(returns, 0.25),
        "median": _quantile(returns, 0.50),
        "p75": _quantile(returns, 0.75),
        "p95": _quantile(returns, 0.95),
        "max": max(returns) if returns else None,
    }
    for field, expected in expected_distribution.items():
        _same_optional_reported(expected, distribution[field], 6, f"{path}.return_distribution.{field}")

    expected_scenarios: dict[str, dict[str, Any]] = {}
    for scenario_id in sorted({str(record["scenario_id"]) for record in records}):
        values = [float(record["return"]) for record in records if record["scenario_id"] == scenario_id]
        expected_scenarios[scenario_id] = {
            "episodes": len(values),
            "mean_return": statistics.fmean(values),
            "std_return": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }
    reported_scenarios = row.get("returns_by_scenario")
    if not isinstance(reported_scenarios, dict) or set(reported_scenarios) != set(expected_scenarios):
        raise ValueError(f"{path}.returns_by_scenario disagrees with episode records")
    for scenario_id, expected in expected_scenarios.items():
        reported = _require_keys(
            reported_scenarios[scenario_id],
            required={"episodes", "mean_return", "std_return"},
            allowed={"episodes", "mean_return", "std_return"},
            path=f"{path}.returns_by_scenario.{scenario_id}",
        )
        if (
            _nonnegative_int(reported["episodes"], f"{path}.returns_by_scenario.{scenario_id}.episodes")
            != expected["episodes"]
        ):
            raise ValueError(f"{path}.returns_by_scenario disagrees with episode records")
        _same_reported(
            float(expected["mean_return"]),
            reported["mean_return"],
            6,
            f"{path}.returns_by_scenario.{scenario_id}.mean_return",
        )
        _same_reported(
            float(expected["std_return"]),
            reported["std_return"],
            6,
            f"{path}.returns_by_scenario.{scenario_id}.std_return",
        )

    tools = sorted({tool for record in records for tool in record["tool_counts"]})
    expected_correlations: dict[str, float | None] = {
        "rejection_rate": _correlation([float(record["rejection_rate"]) for record in records], returns),
        "noop_rate": _correlation([float(record["noop_rate"]) for record in records], returns),
    }
    for tool in tools:
        expected_correlations[f"tool_rate:{tool}"] = _correlation(
            [record["tool_counts"].get(tool, 0) / record["steps"] if record["steps"] else 0.0 for record in records],
            returns,
        )
    reported_correlations = row.get("episode_return_correlations")
    if not isinstance(reported_correlations, dict) or set(reported_correlations) != set(expected_correlations):
        raise ValueError(f"{path}.episode_return_correlations disagrees with episode records")
    for field, expected in expected_correlations.items():
        _same_optional_reported(
            expected, reported_correlations[field], 6, f"{path}.episode_return_correlations.{field}"
        )

    metrics = row.get("tool_metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path}.tool_metrics must be an object")
    expected_calls = {tool: sum(int(record["tool_counts"].get(tool, 0)) for record in records) for tool in tools}
    if infra_errors == 0 and set(metrics) != set(expected_calls):
        raise ValueError(f"{path}.tool_metrics tools disagree with episode records")
    total_metric_calls = 0
    weighted_rewards = 0.0
    weighted_rejections = 0.0
    for tool, value in metrics.items():
        metric = _require_keys(
            value,
            required={"calls", "call_rate", "mean_step_reward", "rejection_rate"},
            allowed={"calls", "call_rate", "mean_step_reward", "rejection_rate"},
            path=f"{path}.tool_metrics.{tool}",
        )
        calls = _nonnegative_int(metric["calls"], f"{path}.tool_metrics.{tool}.calls")
        call_rate = _rate(metric["call_rate"], f"{path}.tool_metrics.{tool}.call_rate")
        mean_step_reward = _finite_number(metric["mean_step_reward"], f"{path}.tool_metrics.{tool}.mean_step_reward")
        rejection_rate = _rate(metric["rejection_rate"], f"{path}.tool_metrics.{tool}.rejection_rate")
        if infra_errors == 0 and calls != expected_calls.get(tool, 0):
            raise ValueError(f"{path}.tool_metrics.{tool}.calls disagrees with episode records")
        total_metric_calls += calls
        weighted_rewards += calls * mean_step_reward
        weighted_rejections += calls * rejection_rate
        if infra_errors == 0:
            total_steps = sum(int(record["steps"]) for record in records)
            expected_call_rate = calls / total_steps if total_steps else 0.0
            if not math.isclose(call_rate, round(expected_call_rate, 6), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{path}.tool_metrics.{tool}.call_rate disagrees with episode records")
    if infra_errors == 0:
        tolerance = max(1e-9, 0.500001e-6 * max(1, total_metric_calls))
        if not math.isclose(weighted_rewards, sum(returns), rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(f"{path}.tool_metrics mean_step_reward values disagree with episode returns")
        expected_rejections = sum(float(record["rejection_rate"]) * int(record["steps"]) for record in records)
        if not math.isclose(weighted_rejections, expected_rejections, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(f"{path}.tool_metrics rejection rates disagree with episode records")


def _validate_raw_execution_contract(report: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if report.get("backend") != "replay":
        raise ValueError("raw_report.backend must be the causal replay backend")
    raw_specs = report.get("model_specs")
    if not isinstance(raw_specs, list) or len(raw_specs) < 2:
        raise ValueError("raw_report.model_specs must contain at least two learned models")
    specs: list[dict[str, Any]] = []
    labels: set[str] = set()
    served_models: set[str] = set()
    ranks: set[int] = set()
    for index, value in enumerate(raw_specs):
        spec = _require_keys(
            value,
            required={"label", "model", "capability_rank"},
            allowed={"label", "model", "capability_rank"},
            path=f"raw_report.model_specs[{index}]",
        )
        label = _nonempty_string(spec["label"], f"raw_report.model_specs[{index}].label")
        served_model = _nonempty_string(spec["model"], f"raw_report.model_specs[{index}].model")
        rank = _positive_int(spec["capability_rank"], f"raw_report.model_specs[{index}].capability_rank")
        if label in labels or served_model in served_models or rank in ranks:
            raise ValueError("raw_report.model_specs requires unique labels, served models, and capability ranks")
        labels.add(label)
        served_models.add(served_model)
        ranks.add(rank)
        specs.append({"label": label, "model": served_model, "capability_rank": rank})
    return specs, _validate_sampling(report.get("sampling_contract"), "raw_report.sampling_contract")


def _validate_metadata(
    metadata: dict[str, Any],
    *,
    raw_specs: list[dict[str, Any]],
    raw_sampling: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    _reject_credentials(metadata)
    _require_keys(
        metadata,
        required={"run_id", "backend", "source", "sampling", "models", "provenance", "environment", "run_log"},
        allowed={"run_id", "backend", "source", "sampling", "models", "provenance", "environment", "run_log"},
        path="metadata",
    )
    run_id = metadata["run_id"]
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("metadata.run_id must be a caller-supplied, filesystem-safe identifier")
    if metadata["backend"] != "replay":
        raise ValueError("metadata.backend must be the causal replay backend")

    source = _require_keys(
        metadata["source"], required={"commit", "dirty"}, allowed={"commit", "dirty"}, path="metadata.source"
    )
    commit = source["commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ValueError("metadata.source.commit must be a full hexadecimal commit identity")
    if not isinstance(source["dirty"], bool):
        raise ValueError("metadata.source.dirty must be boolean")
    if source["dirty"]:
        raise ValueError("Run 1B custody requires a clean source checkout (metadata.source.dirty=false)")

    sampling = _validate_sampling(metadata["sampling"], "metadata.sampling")
    if sampling != raw_sampling:
        raise ValueError("metadata.sampling must exactly match raw_report.sampling_contract")

    raw_models = metadata["models"]
    if not isinstance(raw_models, list) or len(raw_models) != len(raw_specs):
        raise ValueError("metadata.models must exactly cover raw_report.model_specs")
    raw_by_label = {spec["label"]: spec for spec in raw_specs}
    models: list[dict[str, Any]] = []
    seen_policies: set[str] = set()
    seen_ids: set[str] = set()
    allowed_model_fields = {
        "policy",
        "label",
        "model_id",
        "served_model_id",
        "revision",
        "tokenizer_revision",
        "capability_rank",
        "weights_sha256",
        "tokenizer_sha256",
    }
    for index, value in enumerate(raw_models):
        raw_model = _require_keys(
            value,
            required={
                "policy",
                "label",
                "model_id",
                "served_model_id",
                "revision",
                "tokenizer_revision",
                "capability_rank",
            },
            allowed=allowed_model_fields,
            path=f"metadata.models[{index}]",
        )
        label = _nonempty_string(raw_model["label"], f"metadata.models[{index}].label")
        expected = raw_by_label.get(label)
        if expected is None:
            raise ValueError(f"metadata.models[{index}].label is absent from raw_report.model_specs")
        policy = _nonempty_string(raw_model["policy"], f"metadata.models[{index}].policy")
        if policy != f"model:{label}":
            raise ValueError(f"metadata.models[{index}].policy must equal 'model:{label}'")
        served_model = _nonempty_string(raw_model["served_model_id"], f"metadata.models[{index}].served_model_id")
        if served_model != expected["model"]:
            raise ValueError(f"metadata.models[{index}].served_model_id disagrees with raw_report.model_specs")
        rank = _positive_int(raw_model["capability_rank"], f"metadata.models[{index}].capability_rank")
        if rank != expected["capability_rank"]:
            raise ValueError(f"metadata.models[{index}].capability_rank disagrees with raw_report.model_specs")
        model_id = _nonempty_string(raw_model["model_id"], f"metadata.models[{index}].model_id")
        revision = _nonempty_string(raw_model["revision"], f"metadata.models[{index}].revision")
        _nonempty_string(raw_model["tokenizer_revision"], f"metadata.models[{index}].tokenizer_revision")
        if "weights_sha256" in raw_model:
            _sha256(raw_model["weights_sha256"], f"metadata.models[{index}].weights_sha256")
        if "tokenizer_sha256" in raw_model:
            _sha256(raw_model["tokenizer_sha256"], f"metadata.models[{index}].tokenizer_sha256")
        if policy in seen_policies or model_id in seen_ids:
            raise ValueError("metadata.models requires unique policies and model identities")
        seen_policies.add(policy)
        seen_ids.add(model_id)
        public = {field: raw_model[field] for field in _MODEL_PUBLIC_FIELDS if field in raw_model}
        public["revision"] = revision
        models.append(public)
    if set(raw_by_label) != {model["label"] for model in models}:
        raise ValueError("metadata.models must exactly cover raw_report.model_specs labels")
    models.sort(key=lambda model: int(model["capability_rank"]))

    provenance = _require_keys(
        metadata["provenance"], required=_HASH_FIELDS, allowed=_HASH_FIELDS, path="metadata.provenance"
    )
    for field in sorted(_HASH_FIELDS):
        _sha256(provenance[field], f"metadata.provenance.{field}")

    environment = _require_keys(
        metadata["environment"],
        required={"hostname", "gpu", "runtime", "commands", "start_utc", "end_utc", "image"},
        allowed={"hostname", "gpu", "runtime", "commands", "start_utc", "end_utc", "image"},
        path="metadata.environment",
    )
    _nonempty_string(environment["hostname"], "metadata.environment.hostname")
    gpu = environment["gpu"]
    if not isinstance(gpu, list) or not gpu:
        raise ValueError("metadata.environment.gpu must be a non-empty list")
    for index, identity in enumerate(gpu):
        _nonempty_string(identity, f"metadata.environment.gpu[{index}]")
    runtime = environment["runtime"]
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError("metadata.environment.runtime must be a non-empty object")
    _nonempty_string(runtime.get("python"), "metadata.environment.runtime.python")
    _nonempty_string(runtime.get("platform"), "metadata.environment.runtime.platform")
    commands = environment["commands"]
    if not isinstance(commands, list) or not commands:
        raise ValueError("metadata.environment.commands must be a non-empty list")
    for index, value in enumerate(commands):
        command = _require_keys(
            value,
            required={"argv", "exit_code"},
            allowed={"argv", "exit_code", "cwd"},
            path=f"metadata.environment.commands[{index}]",
        )
        argv = command["argv"]
        if not isinstance(argv, list) or not argv:
            raise ValueError(f"metadata.environment.commands[{index}].argv must be a non-empty list")
        for argument_index, argument in enumerate(argv):
            _nonempty_string(argument, f"metadata.environment.commands[{index}].argv[{argument_index}]")
        if isinstance(command["exit_code"], bool) or not isinstance(command["exit_code"], int):
            raise ValueError(f"metadata.environment.commands[{index}].exit_code must be an integer")
        if "cwd" in command:
            _nonempty_string(command["cwd"], f"metadata.environment.commands[{index}].cwd")
    start = _utc_timestamp(environment["start_utc"], "metadata.environment.start_utc")
    end = _utc_timestamp(environment["end_utc"], "metadata.environment.end_utc")
    if end < start:
        raise ValueError("metadata.environment.end_utc must not precede start_utc")
    image = _require_keys(
        environment["image"],
        required={"digest", "signature_status"},
        allowed={"digest", "signature_status"},
        path="metadata.environment.image",
    )
    digest = _nonempty_string(image["digest"], "metadata.environment.image.digest")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("metadata.environment.image.digest must be a sha256 container digest")
    if image["signature_status"] not in {"VERIFIED", "UNVERIFIED", "NOT_APPLICABLE", "NOT_CHECKED"}:
        raise ValueError("metadata.environment.image.signature_status is unsupported")

    environment_output = {
        "run_id": run_id,
        "source": source,
        "provenance": provenance,
        "environment": environment,
    }
    return models, environment_output, _sanitize_run_log(metadata["run_log"])


def _parse_pair(record: Mapping[str, Any], path: str) -> tuple[int, int, str]:
    pair_key = record.get("pair_key")
    if not isinstance(pair_key, str) or not _PAIR_KEY.fullmatch(pair_key):
        raise ValueError(f"{path}.pair_key must be '<prompt_index>:<response_index>'")
    prompt_text, response_text = pair_key.split(":", 1)
    prompt_index, response_index = int(prompt_text), int(response_text)
    if record.get("prompt_index") != prompt_index or record.get("response_index") != response_index:
        raise ValueError(f"{path} prompt/response indexes disagree with pair_key {pair_key!r}")
    return prompt_index, response_index, pair_key


def _validate_profile_row(
    row: dict[str, Any],
    *,
    row_index: int,
    expected_keys: set[str],
    task_scenarios: Mapping[int, str],
    expected_episodes: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    policy = row.get("policy")
    path = f"raw_report.profile[{row_index}]"
    if not isinstance(policy, str) or not policy:
        raise ValueError(f"{path}.policy must be non-empty")
    is_model = policy.startswith("model:")
    episodes = _nonnegative_int(row.get("episodes"), f"{path}.episodes")
    records = row.get("episode_records")
    if not isinstance(records, list) or len(records) != episodes:
        raise ValueError(f"{path}.episode_records must contain exactly {episodes} records")
    infra_errors = _nonnegative_int(row.get("infra_errors"), f"{path}.infra_errors")
    if is_model:
        if episodes + infra_errors != expected_episodes:
            raise ValueError(f"{path}.episodes plus infra_errors must equal planned support {expected_episodes}")
    elif episodes != expected_episodes or infra_errors:
        raise ValueError(f"{path} scripted anchor must contain complete planned support with no infra_errors")

    normalized: list[dict[str, Any]] = []
    records_by_key: dict[str, dict[str, Any]] = {}
    parse_failures = invalid_calls = usable_episodes = total_steps = noop_steps = 0
    rejected_steps = 0.0
    for record_index, record in enumerate(records):
        record_path = f"{path}.episode_records[{record_index}]"
        if not isinstance(record, dict):
            raise ValueError(f"{record_path} must be an object")
        prompt_index, response_index, pair_key = _parse_pair(record, record_path)
        if pair_key in records_by_key:
            raise ValueError(f"{path} has duplicate pair_key {pair_key!r}")
        if pair_key not in expected_keys:
            raise ValueError(f"{record_path}.pair_key is outside planned support")
        scenario_id = record.get("scenario_id")
        if scenario_id != task_scenarios[prompt_index]:
            raise ValueError(f"{record_path}.scenario_id disagrees with the task manifest")
        episode_return = _finite_number(record.get("return"), f"{record_path}.return")
        record_parse = _nonnegative_int(record.get("parse_failures"), f"{record_path}.parse_failures")
        record_invalid = _nonnegative_int(record.get("invalid_calls"), f"{record_path}.invalid_calls")
        usable = record.get("usable")
        if not isinstance(usable, bool) or usable != (record_parse == 0 and record_invalid == 0):
            raise ValueError(f"{record_path}.usable disagrees with parse/invalid failures")
        steps = _nonnegative_int(record.get("steps"), f"{record_path}.steps")
        tool_counts = record.get("tool_counts")
        if not isinstance(tool_counts, dict) or any(not isinstance(tool, str) or not tool for tool in tool_counts):
            raise ValueError(f"{record_path}.tool_counts must map tool names to counts")
        counts = {
            tool: _nonnegative_int(count, f"{record_path}.tool_counts.{tool}") for tool, count in tool_counts.items()
        }
        if sum(counts.values()) != steps:
            raise ValueError(f"{record_path}.tool_counts must sum to steps")
        rejection_rate = _rate(record.get("rejection_rate"), f"{record_path}.rejection_rate")
        noop_rate = _rate(record.get("noop_rate"), f"{record_path}.noop_rate")
        expected_noop_rate = counts.get("noop", 0) / steps if steps else 0.0
        if not math.isclose(noop_rate, expected_noop_rate, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{record_path}.noop_rate disagrees with tool_counts")

        parse_failures += record_parse
        invalid_calls += record_invalid
        usable_episodes += int(usable)
        total_steps += steps
        noop_steps += counts.get("noop", 0)
        rejected_steps += rejection_rate * steps
        normalized_record = {
            "policy": policy,
            "pair_key": pair_key,
            "prompt_index": prompt_index,
            "response_index": response_index,
            "scenario_id": scenario_id,
            "return": episode_return,
            "usable": usable,
            "parse_failures": record_parse,
            "invalid_calls": record_invalid,
            "rejection_rate": rejection_rate,
            "noop_rate": noop_rate,
            "steps": steps,
            "tool_counts": counts,
        }
        normalized.append(normalized_record)
        records_by_key[pair_key] = normalized_record

    missing = sorted(expected_keys - set(records_by_key))
    if len(missing) != infra_errors:
        raise ValueError(f"{path}.episode_records missing support does not equal infra_errors: {missing[:5]}")
    if parse_failures != _nonnegative_int(row.get("parse_failures"), f"{path}.parse_failures"):
        raise ValueError(f"{path}.parse_failures disagrees with episode records")
    if invalid_calls != _nonnegative_int(row.get("invalid_calls"), f"{path}.invalid_calls"):
        raise ValueError(f"{path}.invalid_calls disagrees with episode records")
    if usable_episodes != _nonnegative_int(row.get("usable_episodes"), f"{path}.usable_episodes"):
        raise ValueError(f"{path}.usable_episodes disagrees with episode records")

    returns = [record["return"] for record in normalized]
    expected_mean = statistics.fmean(returns) if returns else None
    _same_optional_reported(expected_mean, row.get("mean_return"), 4, f"{path}.mean_return")
    expected_std = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    _same_reported(expected_std, row.get("std_return"), 4, f"{path}.std_return")
    aggregate_noop_rate = noop_steps / total_steps if total_steps else 0.0
    aggregate_rejection_rate = rejected_steps / total_steps if total_steps else 0.0
    if infra_errors:
        _rate(row.get("noop_rate"), f"{path}.noop_rate")
        _rate(row.get("rejection_rate"), f"{path}.rejection_rate")
    else:
        _same_reported(aggregate_noop_rate, row.get("noop_rate"), 4, f"{path}.noop_rate")
        _same_reported(aggregate_rejection_rate, row.get("rejection_rate"), 4, f"{path}.rejection_rate")
    _validate_auxiliary_statistics(row, normalized, infra_errors=infra_errors, path=path)
    return normalized, records_by_key


def _prompt_deltas(
    better: Mapping[str, Mapping[str, Any]],
    worse: Mapping[str, Mapping[str, Any]],
    *,
    prompts: int,
    responses: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for prompt_index in range(prompts):
        deltas: list[float] = []
        scenario_id: str | None = None
        for response_index in range(responses):
            key = f"{prompt_index}:{response_index}"
            left = better[key]
            right = worse[key]
            if left["scenario_id"] != right["scenario_id"]:
                raise ValueError(f"comparison pair {key} has inconsistent scenario labels")
            scenario_id = str(left["scenario_id"])
            deltas.append(float(left["return"]) - float(right["return"]))
        mean_delta = statistics.fmean(deltas)
        result.append(
            {
                "prompt_index": prompt_index,
                "scenario_id": scenario_id,
                "response_pairs": len(deltas),
                "prompt_mean_delta": mean_delta,
                "prompt_median_delta": statistics.median(deltas),
                "prompt_outcome": "WIN" if mean_delta > 0 else "LOSS" if mean_delta < 0 else "TIE",
            }
        )
    return result


def _clustered_bootstrap_ci(prompt_rows: list[dict[str, Any]], *, seed: int, draws: int) -> tuple[float, float]:
    """Independently reproduce the sweep's regime-stratified interval."""

    strata: dict[str, list[float]] = {}
    for row in sorted(prompt_rows, key=lambda item: (str(item["scenario_id"]), int(item["prompt_index"]))):
        strata.setdefault(str(row["scenario_id"]), []).append(float(row["prompt_mean_delta"]))
    rng = np.random.default_rng(seed)
    prompt_count = len(prompt_rows)
    bootstrap_means = np.empty(draws, dtype=float)
    chunk_size = max(1, min(256, 2_000_000 // prompt_count))
    for start in range(0, draws, chunk_size):
        count = min(chunk_size, draws - start)
        sampled_sum = np.zeros(count, dtype=float)
        for scenario_id in sorted(strata):
            values = np.asarray(strata[scenario_id], dtype=float)
            indices = rng.integers(0, len(values), size=(count, len(values)))
            sampled_sum += values[indices].sum(axis=1)
        bootstrap_means[start : start + count] = sampled_sum / prompt_count
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def _validate_comparison(
    comparison: dict[str, Any],
    *,
    path: str,
    better_policy: str,
    worse_policy: str,
    better_records: Mapping[str, Mapping[str, Any]],
    worse_records: Mapping[str, Mapping[str, Any]],
    prompts: int,
    responses: int,
    expected_seed: int,
    expected_draws: int,
) -> list[dict[str, Any]]:
    prompt_rows = _prompt_deltas(better_records, worse_records, prompts=prompts, responses=responses)
    response_pairs = prompts * responses
    for field in ("pairs", "paired_episodes", "response_pairs"):
        if _nonnegative_int(comparison.get(field), f"{path}.{field}") != response_pairs:
            raise ValueError(f"{path}.{field} is inconsistent with paired response support")
    if _nonnegative_int(comparison.get("prompt_clusters"), f"{path}.prompt_clusters") != prompts:
        raise ValueError(f"{path}.prompt_clusters is inconsistent with prompt support")

    means = [row["prompt_mean_delta"] for row in prompt_rows]
    mean = _same_precise(statistics.fmean(means), comparison.get("mean_delta"), f"{path}.mean_delta")
    _same_precise(statistics.median(means), comparison.get("median_delta"), f"{path}.median_delta")
    outcomes = {name: sum(row["prompt_outcome"] == name for row in prompt_rows) for name in ("WIN", "TIE", "LOSS")}
    for field, outcome in (("prompt_wins", "WIN"), ("prompt_ties", "TIE"), ("prompt_losses", "LOSS")):
        if _nonnegative_int(comparison.get(field), f"{path}.{field}") != outcomes[outcome]:
            raise ValueError(f"{path}.{field} is inconsistent with prompt-level deltas")

    if comparison.get("bootstrap_method") != _BOOTSTRAP_METHOD:
        raise ValueError(f"{path}.bootstrap_method must be {_BOOTSTRAP_METHOD!r}")
    seed = _nonnegative_int(comparison.get("bootstrap_seed"), f"{path}.bootstrap_seed")
    draws = _positive_int(comparison.get("bootstrap_draws"), f"{path}.bootstrap_draws")
    if seed != expected_seed:
        raise ValueError(f"{path}.bootstrap_seed must equal the frozen seed {expected_seed}")
    if draws != expected_draws:
        raise ValueError(f"{path}.bootstrap_draws must equal the frozen draw count {expected_draws}")
    expected_low, expected_high = _clustered_bootstrap_ci(prompt_rows, seed=seed, draws=draws)
    low = _same_precise(expected_low, comparison.get("ci95_low"), f"{path}.ci95_low")
    high = _same_precise(expected_high, comparison.get("ci95_high"), f"{path}.ci95_high")
    expected_status = "PASS" if mean > 0.0 and low > 0.0 else "FAIL"
    if comparison.get("status") != expected_status:
        raise ValueError(f"{path}.status is inconsistent with the mean and confidence interval")
    return [
        {
            **row,
            "better_policy": better_policy,
            "worse_policy": worse_policy,
            "comparison_status": expected_status,
            "bootstrap_method": _BOOTSTRAP_METHOD,
            "bootstrap_seed": seed,
            "bootstrap_draws": draws,
            "mean_delta": mean,
            "ci95_low": low,
            "ci95_high": high,
        }
        for row in prompt_rows
    ]


def _validate_report(report: dict[str, Any], models: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = _positive_int(report.get("prompts"), "raw_report.prompts")
    responses = _positive_int(report.get("responses_per_prompt"), "raw_report.responses_per_prompt")
    expected_episodes = prompts * responses
    if _positive_int(report.get("episodes_per_policy"), "raw_report.episodes_per_policy") != expected_episodes:
        raise ValueError("raw_report.episodes_per_policy must equal prompts times responses_per_prompt")
    if not isinstance(report.get("compliance_profile"), bool):
        raise ValueError("raw_report.compliance_profile must be boolean")
    compliance_profile = bool(report["compliance_profile"])
    if compliance_profile and (prompts != 500 or responses != 16):
        raise ValueError("Run 1B compliance requires exactly 500 prompts and 16 responses per prompt")
    bootstrap_draws = _COMPLIANCE_BOOTSTRAP_DRAWS if compliance_profile else _DEFAULT_BOOTSTRAP_DRAWS
    failure_ceiling = _rate(report.get("failure_rate_ceiling"), "raw_report.failure_rate_ceiling")
    if failure_ceiling != 0.0:
        raise ValueError("Run 1B custody packages require a zero failure-rate ceiling")

    tasks = report.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != prompts:
        raise ValueError(f"raw_report.tasks must contain exactly {prompts} prompt rows")
    task_scenarios: dict[int, str] = {}
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or task.get("prompt_index") != index:
            raise ValueError("raw_report.tasks must have unique contiguous prompt_index values")
        scenario_id = task.get("scenario_id")
        if scenario_id not in _SUPPORTED_REGIMES:
            raise ValueError(f"raw_report.tasks[{index}].scenario_id must be a supported regime")
        seed = task.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"raw_report.tasks[{index}].seed must be an integer")
        regime_mix = task.get("regime_mix")
        if regime_mix != {scenario_id: 1.0}:
            raise ValueError(f"raw_report.tasks[{index}] must have a one-hot regime_mix for {scenario_id!r}")
        task_scenarios[index] = scenario_id
    if compliance_profile:
        scenario_counts = {scenario: list(task_scenarios.values()).count(scenario) for scenario in _SUPPORTED_REGIMES}
        if set(task_scenarios.values()) != _SUPPORTED_REGIMES or set(scenario_counts.values()) != {100}:
            raise ValueError("Run 1B compliance tasks must balance all five supported regimes at 100 prompts each")
    expected_keys = {f"{prompt}:{response}" for prompt in range(prompts) for response in range(responses)}

    anchor_order = report.get("anchor_order_expected")
    if anchor_order != list(_ANCHOR_ORDER):
        raise ValueError("raw_report.anchor_order_expected must equal the frozen four-anchor order")
    model_policies = [str(model["policy"]) for model in models]
    canonical_policies = list(anchor_order) + model_policies

    profile = report.get("profile")
    if not isinstance(profile, list):
        raise ValueError("raw_report.profile must be a list")
    by_policy: dict[str, dict[str, Any]] = {}
    records_by_policy: dict[str, dict[str, dict[str, Any]]] = {}
    flattened_by_policy: dict[str, list[dict[str, Any]]] = {}
    for row_index, row in enumerate(profile):
        if not isinstance(row, dict):
            raise ValueError(f"raw_report.profile[{row_index}] must be an object")
        policy = row.get("policy")
        if policy in by_policy:
            raise ValueError(f"raw_report.profile has duplicate policy {policy!r}")
        normalized, indexed = _validate_profile_row(
            row,
            row_index=row_index,
            expected_keys=expected_keys,
            task_scenarios=task_scenarios,
            expected_episodes=expected_episodes,
        )
        by_policy[str(policy)] = row
        flattened_by_policy[str(policy)] = normalized
        records_by_policy[str(policy)] = indexed
    if set(by_policy) != set(canonical_policies):
        raise ValueError("raw_report.profile policies do not exactly match the current anchors and metadata models")

    relief_mean = _finite_number(by_policy["anchor:relief"].get("mean_return"), "anchor:relief.mean_return")
    for policy, row in by_policy.items():
        row_mean = row.get("mean_return")
        expected_vs_relief = (
            None if row_mean is None else _finite_number(row_mean, f"{policy}.mean_return") - relief_mean
        )
        _same_optional_reported(expected_vs_relief, row.get("vs_relief"), 4, f"{policy}.vs_relief")

    constraints = report.get("anchor_order_constraints")
    comparisons = report.get("anchor_ordering_comparisons")
    if constraints != [list(pair) for pair in _ANCHOR_CONSTRAINTS]:
        raise ValueError("raw_report.anchor_order_constraints must equal the frozen four comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != len(_ANCHOR_CONSTRAINTS):
        raise ValueError("raw_report.anchor_ordering_comparisons must cover the frozen four comparisons")
    paired_rows: list[dict[str, Any]] = []
    anchor_pass = all(
        int(by_policy[policy]["parse_failures"]) == 0
        and int(by_policy[policy]["invalid_calls"]) == 0
        and int(by_policy[policy]["infra_errors"]) == 0
        and all(record["usable"] for record in flattened_by_policy[policy])
        for policy in anchor_order
    )
    for index, (constraint, comparison) in enumerate(zip(constraints, comparisons, strict=True)):
        path = f"raw_report.anchor_ordering_comparisons[{index}]"
        if (
            not isinstance(constraint, list)
            or len(constraint) != 2
            or any(policy not in anchor_order for policy in constraint)
            or not isinstance(comparison, dict)
        ):
            raise ValueError(f"{path} does not identify two declared anchors")
        better, worse = constraint
        if comparison.get("better") != better or comparison.get("worse") != worse:
            raise ValueError(f"{path} disagrees with anchor_order_constraints")
        rows = _validate_comparison(
            comparison,
            path=path,
            better_policy=better,
            worse_policy=worse,
            better_records=records_by_policy[better],
            worse_records=records_by_policy[worse],
            prompts=prompts,
            responses=responses,
            expected_seed=10_000 + index,
            expected_draws=bootstrap_draws,
        )
        paired_rows.extend({"comparison_kind": "anchor", "comparison_index": index, **row} for row in rows)
        anchor_pass = anchor_pass and comparison["status"] == "PASS"
    if report.get("anchor_ordering_ok") is not anchor_pass:
        raise ValueError("raw_report.anchor_ordering_ok disagrees with anchor comparisons")

    ordering = report.get("model_ordering")
    if not isinstance(ordering, dict) or ordering.get("expected") != model_policies:
        raise ValueError("raw_report.model_ordering.expected disagrees with metadata capability ranks")
    observed = ordering.get("observed")
    if not isinstance(observed, dict) or set(observed) != set(model_policies):
        raise ValueError("raw_report.model_ordering.observed must cover every metadata model")
    for policy in model_policies:
        row_mean = by_policy[policy].get("mean_return")
        actual_mean = None if row_mean is None else _finite_number(row_mean, f"{policy}.mean_return")
        _same_optional_reported(actual_mean, observed[policy], 4, f"raw_report.model_ordering.observed.{policy}")

    engineering_status: dict[str, str] = {}
    expected_failure_counts: dict[str, dict[str, int]] = {}
    for policy in model_policies:
        row = by_policy[policy]
        counts = {
            "parse_failures": int(row["parse_failures"]),
            "invalid_calls": int(row["invalid_calls"]),
            "infra_errors": int(row["infra_errors"]),
        }
        expected_failure_counts[policy] = counts
        all_usable = all(record["usable"] for record in flattened_by_policy[policy])
        engineering_status[policy] = "EVALUABLE" if sum(counts.values()) == 0 and all_usable else "NOT_EVALUABLE"
    if ordering.get("failure_counts") != expected_failure_counts:
        raise ValueError("raw_report.model_ordering.failure_counts disagrees with profile counts")
    failure_rates = ordering.get("failure_rates")
    if not isinstance(failure_rates, dict) or set(failure_rates) != set(model_policies):
        raise ValueError("raw_report.model_ordering.failure_rates must cover every model")
    for policy in model_policies:
        expected_rate = min(1.0, sum(expected_failure_counts[policy].values()) / expected_episodes)
        if not math.isclose(_rate(failure_rates[policy], f"model_ordering.failure_rates.{policy}"), expected_rate):
            raise ValueError("raw_report.model_ordering.failure_rates disagrees with profile counts")
    if (
        _rate(ordering.get("failure_rate_ceiling"), "raw_report.model_ordering.failure_rate_ceiling")
        != failure_ceiling
    ):
        raise ValueError("raw_report.model_ordering.failure_rate_ceiling disagrees with the report ceiling")

    all_evaluable = all(value == "EVALUABLE" for value in engineering_status.values())
    prompt_clusters_by_scenario: dict[str, set[int]] = {}
    for prompt_index, scenario_id in task_scenarios.items():
        prompt_clusters_by_scenario.setdefault(scenario_id, set()).add(prompt_index)
    underpowered_scenarios = sorted(
        scenario_id for scenario_id, prompt_indices in prompt_clusters_by_scenario.items() if len(prompt_indices) < 2
    )
    status = ordering.get("status")
    ordering_reason = ordering.get("reason")
    if ordering_reason is not None and (not isinstance(ordering_reason, str) or not ordering_reason.strip()):
        raise ValueError("raw_report.model_ordering.reason must be null or a non-empty string")
    model_comparisons = ordering.get("comparisons", [])
    aggregate_engineering_status = "PASS" if all_evaluable else "NOT_EVALUABLE"
    aggregate_quality_status = "NOT_EVALUABLE"
    if not all_evaluable:
        if status != "NOT_EVALUABLE" or model_comparisons:
            raise ValueError("failed learned models require NOT_EVALUABLE ordering with no quality comparisons")
        if ordering_reason is None:
            raise ValueError("failed learned models require model_ordering.reason")
        if report.get("model_ordering_ok") is not None:
            raise ValueError("raw_report.model_ordering_ok must be null when models are not evaluable")
        if "valid_paired_episodes" in ordering or "valid_paired_prompt_clusters" in ordering:
            raise ValueError("NOT_EVALUABLE model ordering must not claim valid paired support")
    elif underpowered_scenarios:
        expected_reason = (
            "model-quality inference requires at least two prompt clusters per scenario; "
            f"underpowered scenarios: {underpowered_scenarios}"
        )
        if compliance_profile:
            raise ValueError("Run 1B compliance cannot be an underpowered quality profile")
        if (
            status != "NOT_EVALUABLE"
            or ordering.get("engineering_status") != "PASS"
            or ordering.get("quality_status") != "NOT_EVALUABLE"
            or model_comparisons != []
            or ordering_reason != expected_reason
        ):
            raise ValueError("underpowered smoke must use the exact engineering-only NOT_EVALUABLE contract")
        if report.get("model_ordering_ok") is not None:
            raise ValueError("underpowered smoke requires null raw_report.model_ordering_ok")
        if (
            _nonnegative_int(ordering.get("valid_paired_episodes"), "raw_report.model_ordering.valid_paired_episodes")
            != expected_episodes
        ):
            raise ValueError("underpowered smoke valid_paired_episodes disagrees with complete support")
        if (
            "valid_paired_prompt_clusters" in ordering
            and _nonnegative_int(
                ordering["valid_paired_prompt_clusters"],
                "raw_report.model_ordering.valid_paired_prompt_clusters",
            )
            != prompts
        ):
            raise ValueError("underpowered smoke valid_paired_prompt_clusters disagrees with prompt support")
    else:
        if (
            status not in {"PASS", "FAIL"}
            or ordering.get("engineering_status") != "PASS"
            or ordering.get("quality_status") != status
            or not isinstance(model_comparisons, list)
        ):
            raise ValueError("complete learned models require PASS/FAIL model comparisons")
        if len(model_comparisons) != len(model_policies) - 1:
            raise ValueError("raw_report.model_ordering.comparisons must cover every adjacent rank")
        model_pass = True
        for index, (weaker, stronger, comparison) in enumerate(
            zip(model_policies[:-1], model_policies[1:], model_comparisons, strict=True)
        ):
            path = f"raw_report.model_ordering.comparisons[{index}]"
            if (
                not isinstance(comparison, dict)
                or comparison.get("weaker") != weaker
                or comparison.get("stronger") != stronger
            ):
                raise ValueError(f"{path} disagrees with metadata capability ranks")
            rows = _validate_comparison(
                comparison,
                path=path,
                better_policy=stronger,
                worse_policy=weaker,
                better_records=records_by_policy[stronger],
                worse_records=records_by_policy[weaker],
                prompts=prompts,
                responses=responses,
                expected_seed=index,
                expected_draws=bootstrap_draws,
            )
            paired_rows.extend({"comparison_kind": "model", "comparison_index": index, **row} for row in rows)
            model_pass = model_pass and comparison["status"] == "PASS"
        expected_status = "PASS" if model_pass else "FAIL"
        if status != expected_status or report.get("model_ordering_ok") is not model_pass:
            raise ValueError("raw_report model ordering verdict disagrees with its comparisons")
        aggregate_quality_status = str(status)
        if (
            _nonnegative_int(ordering.get("valid_paired_episodes"), "raw_report.model_ordering.valid_paired_episodes")
            != expected_episodes
        ):
            raise ValueError("raw_report.model_ordering.valid_paired_episodes disagrees with complete support")
        if (
            "valid_paired_prompt_clusters" in ordering
            and _nonnegative_int(
                ordering["valid_paired_prompt_clusters"], "raw_report.model_ordering.valid_paired_prompt_clusters"
            )
            != prompts
        ):
            raise ValueError("raw_report.model_ordering.valid_paired_prompt_clusters disagrees with prompt support")

    model_by_policy = {str(model["policy"]): model for model in models}
    policies: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for policy in canonical_policies:
        row = by_policy[policy]
        kind = "anchor" if policy.startswith("anchor:") else "model"
        model = model_by_policy.get(policy, {})
        evaluation_status = "EVALUABLE" if kind == "anchor" else engineering_status[policy]
        policies.append(
            {
                "policy": policy,
                "kind": kind,
                "evaluation_status": evaluation_status,
                "capability_rank": model.get("capability_rank"),
                "model_id": model.get("model_id"),
                "served_model_id": model.get("served_model_id"),
                "model_revision": model.get("revision"),
                "episodes": int(row["episodes"]),
                "expected_episodes": expected_episodes,
                "mean_return": None if row["mean_return"] is None else float(row["mean_return"]),
                "std_return": float(row["std_return"]),
                "rejection_rate": float(row["rejection_rate"]),
                "noop_rate": float(row["noop_rate"]),
                "parse_failures": int(row["parse_failures"]),
                "invalid_calls": int(row["invalid_calls"]),
                "infra_errors": int(row["infra_errors"]),
                "usable_episodes": int(row["usable_episodes"]),
            }
        )
        for record in sorted(
            flattened_by_policy[policy], key=lambda item: (item["prompt_index"], item["response_index"])
        ):
            episodes.append({"kind": kind, "evaluation_status": evaluation_status, **record})

    comparisons_summary = [{"kind": "anchor", **comparison} for comparison in comparisons] + (
        [{"kind": "model", **comparison} for comparison in model_comparisons] if all_evaluable else []
    )
    return {
        "prompts": prompts,
        "responses": responses,
        "expected_episodes": expected_episodes,
        "tasks": tasks,
        "anchor_order": anchor_order,
        "constraints": constraints,
        "policies": policies,
        "episodes": episodes,
        "paired_rows": paired_rows,
        "anchor_status": "PASS" if anchor_pass else "FAIL",
        "model_status": str(status),
        "model_engineering_status": aggregate_engineering_status,
        "model_quality_status": aggregate_quality_status,
        "model_reason": ordering_reason,
        "comparisons": comparisons_summary,
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render_graph(
    path_png: Path,
    path_svg: Path,
    *,
    run_id: str,
    policies: list[dict[str, Any]],
    model_quality_status: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    labels: list[str] = []
    for row in policies:
        label = row["policy"].removeprefix("anchor:").removeprefix("model:")
        if row["kind"] == "model" and row["evaluation_status"] == "NOT_EVALUABLE":
            label += " - NOT EVALUABLE"
        elif row["kind"] == "model" and model_quality_status == "NOT_EVALUABLE":
            label += " - QUALITY NOT EVALUABLE"
        labels.append(label)
    positions = list(range(len(policies)))
    means = [row["mean_return"] for row in policies]
    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "svg.fonttype": "none",
            "svg.hashsalt": "openair-run1b-report-v1",
        }
    ):
        figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
        return_axis, behavior_axis = axes
        for position, row, mean in zip(positions, policies, means, strict=True):
            if mean is None:
                return_axis.annotate(
                    "NO COMPLETED EPISODES\nNOT EVALUABLE",
                    (position, 0.03),
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    va="bottom",
                    color="#9b2525",
                    fontsize=7,
                )
                continue
            if row["kind"] == "anchor":
                return_axis.scatter(position, mean, marker="o", color="#3569a8", s=56, zorder=3)
            elif row["evaluation_status"] == "EVALUABLE":
                if model_quality_status == "NOT_EVALUABLE":
                    return_axis.scatter(position, mean, marker="^", color="#c27a00", s=64, zorder=3)
                    return_axis.annotate(
                        "QUALITY NOT EVALUABLE",
                        (position, mean),
                        xytext=(0, 9),
                        textcoords="offset points",
                        ha="center",
                        color="#8a5700",
                        fontsize=8,
                    )
                else:
                    return_axis.scatter(position, mean, marker="D", color="#159d76", s=56, zorder=3)
            else:
                return_axis.scatter(position, mean, marker="x", color="#c43b3b", s=75, linewidths=2.0, zorder=3)
                return_axis.annotate(
                    "NOT EVALUABLE",
                    (position, mean),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    color="#9b2525",
                    fontsize=8,
                )
        return_axis.axhline(0.0, color="#777777", linewidth=0.8, zorder=1)
        return_axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        return_axis.set_ylabel("Mean episode return\n(raw operational value)")
        return_axis.set_title("Returns; failed models are diagnostic only and are never assigned zero")

        width = 0.36
        behavior_axis.bar(
            [position - width / 2 for position in positions],
            [row["rejection_rate"] for row in policies],
            width=width,
            color="#e07a5f",
            label="Rejection rate",
        )
        behavior_axis.bar(
            [position + width / 2 for position in positions],
            [row["noop_rate"] for row in policies],
            width=width,
            color="#81b29a",
            label="Noop rate",
        )
        behavior_axis.set_ylim(0.0, 1.05)
        behavior_axis.set_ylabel("Fraction of steps")
        behavior_axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        behavior_axis.legend(loc="upper right", frameon=False, ncol=2)
        behavior_axis.set_xticks(positions, labels, rotation=22, ha="right")
        behavior_axis.set_xlabel("Policy")
        figure.suptitle(
            f"Run 1B capability sweep — {run_id}\nModel quality: {model_quality_status}",
            fontsize=13,
        )
        figure.tight_layout()
        figure.savefig(
            path_png,
            format="png",
            metadata={"Software": "OpenAir Run 1B deterministic reporter"},
        )
        figure.savefig(
            path_svg,
            format="svg",
            metadata={"Creator": "OpenAir Run 1B deterministic reporter", "Date": None},
        )
        plt.close(figure)


def _write_checksums(output: Path) -> None:
    entries = []
    for path in sorted(
        (candidate for candidate in output.iterdir() if candidate.name != "SHA256SUMS"), key=lambda p: p.name
    ):
        if not path.is_file():
            raise ValueError(f"unexpected non-file artifact in package: {path.name}")
        entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (output / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")


def build_report_package(raw_report_path: str | Path, metadata_path: str | Path, output_dir: str | Path) -> Path:
    """Validate one report and write its deterministic receipt package."""

    raw_path = Path(raw_report_path)
    metadata_input = Path(metadata_path)
    output = Path(output_dir)
    report, raw_bytes = _load_object(raw_path, "raw report")
    metadata, metadata_bytes = _load_object(metadata_input, "metadata")
    _reject_credentials(report, "raw_report")
    raw_specs, raw_sampling = _validate_raw_execution_contract(report)
    models, environment, run_log = _validate_metadata(metadata, raw_specs=raw_specs, raw_sampling=raw_sampling)
    validated = _validate_report(report, models)
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    output.mkdir(parents=True)

    raw_report_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    environment["input_sha256"] = {
        "raw_report": raw_report_sha256,
        "metadata": metadata_sha256,
    }
    run_id = str(metadata["run_id"])
    source_commit = str(metadata["source"]["commit"])
    for policy in validated["policies"]:
        policy["run_id"] = run_id
        policy["source_commit"] = source_commit
    for row in validated["paired_rows"]:
        row["run_id"] = run_id
        row["source_commit"] = source_commit

    contract = {
        "schema_version": 1,
        "benchmark": "run1b_real_model_capability_sweep",
        "run_id": metadata["run_id"],
        "backend": metadata["backend"],
        "source": metadata["source"],
        "sampling": raw_sampling,
        "model_specs": raw_specs,
        "provenance": metadata["provenance"],
        "raw_report_sha256": raw_report_sha256,
        "metadata_sha256": metadata_sha256,
        "compliance_profile": report["compliance_profile"],
        "failure_rate_ceiling": report["failure_rate_ceiling"],
        "prompts": validated["prompts"],
        "responses_per_prompt": validated["responses"],
        "episodes_per_policy": validated["expected_episodes"],
        "task_support": validated["tasks"],
        "anchor_order_expected": validated["anchor_order"],
        "anchor_order_constraints": validated["constraints"],
        "model_order_expected": [model["policy"] for model in models],
        "bootstrap_method": _BOOTSTRAP_METHOD,
    }
    summary = {
        "schema_version": 1,
        "run_id": metadata["run_id"],
        "anchor_ordering_status": validated["anchor_status"],
        "model_ordering_status": validated["model_status"],
        "model_engineering_status": validated["model_engineering_status"],
        "model_quality_status": validated["model_quality_status"],
        "model_ordering_reason": validated["model_reason"],
        "policies": validated["policies"],
        "comparisons": validated["comparisons"],
    }
    (output / "benchmark_contract.json").write_text(_canonical_json(contract), encoding="utf-8")
    (output / "models.json").write_text(_canonical_json(models), encoding="utf-8")
    (output / "raw_report.json").write_bytes(raw_bytes)
    with (output / "episodes.jsonl").open("w", encoding="utf-8", newline="") as handle:
        for episode in validated["episodes"]:
            handle.write(
                json.dumps(episode, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")
    (output / "summary.json").write_text(_canonical_json(summary), encoding="utf-8")
    _write_csv(output / "summary.csv", _SUMMARY_FIELDS, validated["policies"])
    _write_csv(output / "paired_deltas.csv", _PAIRED_FIELDS, validated["paired_rows"])
    (output / "environment.json").write_text(_canonical_json(environment), encoding="utf-8")
    (output / "run.log").write_text(run_log, encoding="utf-8", newline="")
    _render_graph(
        output / "benchmark.png",
        output / "benchmark.svg",
        run_id=str(metadata["run_id"]),
        policies=validated["policies"],
        model_quality_status=validated["model_quality_status"],
    )
    _write_checksums(output)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-report", required=True, help="raw JSON written by model_sweep.py")
    parser.add_argument("--metadata", required=True, help="explicit credential-free run metadata JSON")
    parser.add_argument("--out", required=True, help="new output directory for the custody package")
    args = parser.parse_args(argv)
    build_report_package(args.raw_report, args.metadata, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
