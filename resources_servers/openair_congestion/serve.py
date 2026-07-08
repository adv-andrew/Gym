# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone one-worker launcher for the OpenAir congestion resource server.

This bypasses the Gym head/model-server topology for local GRPO trainers that
talk directly to ``/reset``, ``/step``, and ``/close``. Dataset replay remains
recorded-observation pass-through: accepted actions do not change later KPIs.
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

import uvicorn
from omegaconf import OmegaConf

from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient
from resources_servers.openair_congestion.app import (
    OpenAirCongestionEnv,
    OpenAirCongestionResourcesServerConfig,
)
from resources_servers.openair_congestion.backends import NAMED_REWARD_PROFILE_OVERRIDES


def _reward_weights_json(value: str) -> dict[str, float]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid reward-weight JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("reward weights must decode to a JSON object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serve OpenAir congestion over the Gymnasium HTTP contract. "
            "dataset_replay is observational pass-through, not a causal simulator."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("replay", "dataset_replay"),
        default="replay",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9110)
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--replay-root", default="data/replay")
    parser.add_argument("--dataset-path", default="data/dataset/provided.jsonl")
    parser.add_argument("--cell-capacity-mbps", type=float, default=60.0)
    parser.add_argument(
        "--observation-render",
        choices=(
            "verbose_v1",
            "resource_compact_pipe_v1",
            "resource_candidate_pipe_v1",
            "t2_compact_pipe_v2",
        ),
        default="resource_compact_pipe_v1",
        help=(
            "Policy observation representation. Dataset GRPO defaults to the "
            "T/C/U/L/A resource form; strict T2 requires truthful P/D support."
        ),
    )
    parser.add_argument(
        "--reward-profile",
        choices=tuple(NAMED_REWARD_PROFILE_OVERRIDES) + ("custom",),
        default="openair_v1",
    )
    parser.add_argument(
        "--reward-weights-json",
        type=_reward_weights_json,
        default=None,
        metavar="JSON",
        help="Validated RewardWeights overrides; use reward-profile=custom for a custom map.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> OpenAirCongestionResourcesServerConfig:
    if args.reward_profile == "custom" and args.reward_weights_json is None:
        raise ValueError("reward-profile=custom requires --reward-weights-json")
    if args.reward_profile != "custom" and args.reward_weights_json is not None:
        raise ValueError("--reward-weights-json requires --reward-profile=custom; named profiles are sealed")
    reward_weights = (
        args.reward_weights_json
        if args.reward_profile == "custom"
        else NAMED_REWARD_PROFILE_OVERRIDES[args.reward_profile]
    )
    return OpenAirCongestionResourcesServerConfig(
        host=args.host,
        port=args.port,
        num_workers=1,
        entrypoint="resources_servers/openair_congestion/app.py",
        domain="agent",
        name="openair_congestion",
        backend=args.backend,
        replay_root=args.replay_root,
        pool_size=args.pool_size,
        max_steps_default=args.max_steps,
        agent_max_steps=args.max_steps,
        dataset_path=args.dataset_path,
        cell_capacity_mbps=args.cell_capacity_mbps,
        reward_profile=args.reward_profile,
        reward_weights=reward_weights or None,
        observation_render=args.observation_render,
    )


def run_server(config: OpenAirCongestionResourcesServerConfig) -> None:
    # The environment and session map are process-local. Deliberately expose
    # no workers flag: more than one worker would route a cookie to unrelated
    # state unless an external session store and sticky routing were added.
    server_client = ServerClient(
        head_server_config=BaseServerConfig(host="", port=0),
        global_config_dict=OmegaConf.create({}),
    )
    env = OpenAirCongestionEnv(config=config, server_client=server_client)
    uvicorn.run(
        env.setup_webserver(),
        host=config.host,
        port=config.port,
        workers=1,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run_server(config_from_args(args))


if __name__ == "__main__":
    main()
