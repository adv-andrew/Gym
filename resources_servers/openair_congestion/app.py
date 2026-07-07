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

"""5G RAN congestion control, gymnasium style.

Multi-turn: the model observes rolling 5s cell/UE KPIs each turn and issues
exactly one tool call from an 8-tool action space (7 actuators + noop; tool
schemas ride in each task row's responses_create_params.tools). /step applies
the action through the selected Backend and returns the next KPIs plus the
per-step reward computed inside the env (rewards.compute_breakdown), passed
through unchanged; the shared gymnasium_agent sums step rewards into the
episode return, like blackjack.

Backends (backends.py): 'replay' (default, offline deterministic),
'dataset_replay' (recorded dataset), 'oai_collector' (live OAI 5G lab, stub).
Selected via the config's ``backend`` field.

The telco env package 'openair_congestion' lives in the openair-rl-gym repo
and must be importable in this venv; see the README Setup section.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseFunctionToolCall
from resources_servers.gymnasium import GymnasiumServer

# backends guards the cross-repo 'openair_congestion' import; keep it ahead of
# the telco imports so a missing install fails with the pip hint.
from resources_servers.openair_congestion.backends import Backend, select_backend


# isort: split
from openair_congestion.render import to_user_text
from openair_congestion.schemas import ToolCall


class OpenAirCongestionResourcesServerConfig(BaseResourcesServerConfig):
    # Which Backend drives episodes: 'replay' (default, offline/CI-safe),
    # 'dataset_replay', or 'oai_collector' (live lab; stub today). The
    # OPENAIR_CONGESTION_BACKEND env var overrides. Extra YAML keys bind here
    # because the config node type uses ConfigDict(extra='allow').
    backend: str = "replay"
    # Replay-backend knobs; defaults match openair_congestion.replay_env.ReplayEnv.
    replay_root: str = "data/replay"
    pool_size: int = 32
    max_steps_default: int = 60
    # dataset_replay knobs: replay a recorded dataset (KPI snapshots or GRPO
    # rollout traces; see dataset_backend.py) instead of synthesizing
    # trajectories. cell_capacity_mbps feeds the reward's throughput
    # normalizer; trace episodes recording cell_capacity_mbps_total override it.
    dataset_path: str = "data/dataset/provided.jsonl"
    cell_capacity_mbps: float = 60.0
    # Truncation-budget fallback for task rows that omit max_steps. Must not
    # exceed the gymnasium_agent's max_steps in the yaml: the agent truncates
    # client-side without notifying the env, so a larger server budget would
    # strand the backend episode slot.
    agent_max_steps: int = 16


# Returned (with 0.0 reward, env not advanced) when the model's turn contains
# no tool call.
_NO_TOOL_CALL_MSG = (
    "No tool call detected. Issue exactly one tool call per turn from the "
    "configured action space (use `noop` to stand pat). Telemetry unchanged."
)


class OpenAirCongestionEnv(GymnasiumServer):
    """GymnasiumServer subclass: /reset + /step, driven by gymnasium_agent."""

    config: OpenAirCongestionResourcesServerConfig

    # Backend built once at startup so a bad replay_root / unknown backend
    # name fails at boot, not on the first rollout. Pydantic private attr.
    _backend: Optional[Backend] = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._backend = select_backend(self.config)

    @property
    def backend(self) -> Backend:
        assert self._backend is not None, "Backend not initialized (model_post_init)"
        return self._backend

    def _live_episode_ids(self) -> set[str]:
        """Episode ids currently owned by live sessions (for the leak reaper)."""
        return {state["episode_id"] for state in self.session_state.values()}

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        # A client retry can POST /reset twice with the same session cookie.
        # Close the previous episode first or its backend slot leaks forever.
        stale = self.session_state.pop(session_id, None)
        if stale is not None:
            try:
                self.backend.close(stale["episode_id"])
            except KeyError:
                pass  # already closed inside the env

        # `metadata` = extra task-row fields forwarded by gymnasium_agent.
        task_params = {
            key: metadata[key]
            for key in ("seed", "difficulty", "regime_mix", "scenario_id", "tier", "max_steps")
            if metadata.get(key) is not None
        }
        first_obs, meta = self.backend.reset(task_params, live_episode_ids=self._live_episode_ids())
        self.session_state[session_id] = {
            "episode_id": meta.episode_id,
            "cumulative_reward": 0.0,
            "n_steps": 0,
            # agent_steps counts model turns, n_steps env steps; a turn with
            # no tool call consumes a turn without advancing the env.
            "agent_steps": 0,
            # Cap at the agent's turn budget so the server truncates no later
            # than the agent and the episode slot is freed via close_session().
            "max_agent_steps": int(
                task_params.get("max_steps") or min(self.config.max_steps_default, self.config.agent_max_steps)
            ),
        }
        # Observation appended as a user message after the dataset prompt.
        return to_user_text(first_obs), {
            "episode_id": meta.episode_id,
            "seed": meta.seed,
            "scenario_id": meta.scenario_id,
            "tier": meta.tier,
        }

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        state = self.session_state.get(session_id)
        if state is None:
            # /step without /reset (defensive; gymnasium_agent always resets).
            return None, 0.0, False, True, {"error": "no_active_episode"}

        state["agent_steps"] += 1
        out_of_budget = state["agent_steps"] >= state["max_agent_steps"]

        calls = [item for item in action.output if getattr(item, "type", None) == "function_call"]

        # No tool call this turn: 0.0 reward, env not stepped, nudge the model.
        if not calls:
            return (
                None if out_of_budget else _NO_TOOL_CALL_MSG,
                0.0,
                False,
                out_of_budget,
                {"error": "no_tool_call", "tool_outputs": []},
            )

        # Exactly one tool call per turn: apply the first, answer extras with
        # an error output so every function_call still gets a matching
        # function_call_output in the conversation.
        call: NeMoGymResponseFunctionToolCall = calls[0]
        tool_outputs = [
            self.tool_output(extra, {"error": "one tool call per turn; only the first was applied"})
            for extra in calls[1:]
        ]

        # Normalise to the env's ToolCall. Unknown tool name / malformed JSON
        # arguments are rejected gracefully (0.0 reward, env not stepped);
        # pydantic ValidationError subclasses ValueError, as does JSONDecodeError.
        try:
            raw_args = json.loads(call.arguments) if (call.arguments or "").strip() else {}
            if not isinstance(raw_args, dict):
                raise ValueError(f"arguments must be a JSON object, got {type(raw_args).__name__}")
            tool_call = ToolCall(name=call.name, arguments=raw_args)
        except ValueError as exc:
            tool_outputs.insert(0, self.tool_output(call, {"accepted": False, "error": str(exc)}))
            return (
                None if out_of_budget else "Invalid tool call rejected; telemetry unchanged.",
                0.0,
                False,
                out_of_budget,
                {"error": "invalid_tool_call", "tool_outputs": tool_outputs},
            )

        # One env step. In-range-but-rejected actions (guardrail) come back as
        # accepted=False with the env's own penalty reward, never an exception.
        next_obs, reward, done, step_info = self.backend.step(state["episode_id"], tool_call)

        # The server returns the per-step reward; gymnasium_agent sums the
        # episode return.
        state["cumulative_reward"] += float(reward)
        state["n_steps"] += 1

        accepted = bool(step_info.get("guardrail_accepted", True))
        rejection_reason = step_info.get("rejection_reason")
        step_idx = step_info.get("step_idx", state["n_steps"])
        tool_outputs.insert(
            0,
            self.tool_output(
                call,
                {"accepted": accepted, "rejection_reason": rejection_reason, "step_idx": step_idx},
            ),
        )

        terminated = bool(done)
        truncated = (not terminated) and out_of_budget
        observation = None if (terminated or truncated) else to_user_text(next_obs)

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            {
                "tool_outputs": tool_outputs,
                "guardrail_accepted": accepted,
                "rejection_reason": rejection_reason,
                "step_idx": step_idx,
                "episode_id": state["episode_id"],
                "n_steps": state["n_steps"],
                "cumulative_reward": state["cumulative_reward"],
            },
        )

    async def close_session(self, session_id: Optional[str]) -> None:
        # Framework calls this when a step returns terminated or truncated:
        # free the backend episode slot, then drop the session entry.
        state = self.session_state.pop(session_id, None)
        if state is not None:
            try:
                self.backend.close(state["episode_id"])
            except KeyError:
                pass  # already closed inside the env


if __name__ == "__main__":
    OpenAirCongestionEnv.run_webserver()
