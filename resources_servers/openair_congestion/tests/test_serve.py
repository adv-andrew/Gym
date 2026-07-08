# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest


pytest.importorskip(
    "openair_congestion",
    reason="telco env package 'openair_congestion' not installed; see README Setup",
)

from resources_servers.openair_congestion import serve  # noqa: E402


def test_named_measured_profile_builds_sealed_validated_config():
    args = serve.build_parser().parse_args(
        [
            "--backend",
            "dataset_replay",
            "--dataset-path",
            "/tmp/provided.jsonl",
            "--reward-profile",
            "openair_v2_measured",
            "--max-steps",
            "12",
        ]
    )
    config = serve.config_from_args(args)
    assert config.num_workers == 1
    assert config.agent_max_steps == config.max_steps_default == 12
    assert config.reward_profile == "openair_v2_measured"
    assert config.reward_weights.w_sla == 0.0
    assert config.reward_weights.w_sla_level == 0.0
    assert config.reward_weights.w_buffer == 0.0
    assert config.reward_weights.w_action == 0.0
    assert config.observation_render == "resource_compact_pipe_v1"


def test_custom_profile_requires_explicit_weights():
    args = serve.build_parser().parse_args(["--reward-profile", "custom"])
    with pytest.raises(ValueError, match="requires"):
        serve.config_from_args(args)


def test_named_profile_rejects_unsealed_weight_override():
    args = serve.build_parser().parse_args(
        ["--reward-profile", "openair_v1", "--reward-weights-json", '{"w_reject": 0.1}']
    )
    with pytest.raises(ValueError, match="custom"):
        serve.config_from_args(args)


def test_replay_rejects_dataset_only_reward_profile():
    args = serve.build_parser().parse_args(["--backend", "replay", "--reward-profile", "openair_v2_measured"])
    with pytest.raises(ValueError, match="dataset_replay"):
        serve.config_from_args(args)


def test_run_server_hardcodes_one_uvicorn_worker(monkeypatch):
    config = serve.config_from_args(serve.build_parser().parse_args([]))
    app = object()
    captured = {}

    class FakeEnv:
        def __init__(self, *, config, server_client):
            captured["config"] = config
            captured["server_client"] = server_client

        def setup_webserver(self):
            return app

    monkeypatch.setattr(serve, "OpenAirCongestionEnv", FakeEnv)
    monkeypatch.setattr(serve, "ServerClient", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(serve.uvicorn, "run", lambda target, **kwargs: captured.update(target=target, **kwargs))

    serve.run_server(config)

    assert captured["target"] is app
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9110
    assert captured["workers"] == 1
