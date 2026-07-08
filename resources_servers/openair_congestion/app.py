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

import copy
import json
import math
from functools import lru_cache
from typing import Any, Callable, Literal, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseFunctionToolCall
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.gymnasium import GymnasiumServer

# backends guards the cross-repo 'openair_congestion' import; keep it ahead of
# the telco imports so a missing install fails with the pip hint.
from resources_servers.openair_congestion.backends import (
    Backend,
    select_backend,
    validate_reward_profile,
)
from resources_servers.openair_congestion.candidate_contract import (
    RESOURCE_CANDIDATE_CONTRACT,
    ResourceCandidateSupport,
    derive_resource_candidate_support,
    resource_action_key,
    validate_resource_candidate_guardrail_contract,
)


# isort: split
from openair_congestion.render import to_user_text
from openair_congestion.schemas import AgentAux, LastActionEcho, ToolCall


class RewardWeightOverrides(BaseModel):
    """Validated overrides for ``openair_congestion.rewards.RewardWeights``."""

    model_config = ConfigDict(extra="forbid")

    w_sla: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    w_tput: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    w_fair: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    w_buffer: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    w_sla_level: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    w_prb_level: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    w_access_level: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    w_fair_level: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    w_action: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    w_reject: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)


def _to_resource_compact_pipe_v1(
    observation: Any,
    *,
    include_action_instruction: bool = True,
) -> str:
    """Render the resource server's stable T/C/U/L/A compact contract.

    The published resource server supports older packaged telco environments
    that predate the T2 policy-feature module. Aggregate dataset traces also do
    not contain the capacity/candidate state needed to truthfully synthesize P
    or D rows, so those rows are deliberately omitted rather than fabricated.
    """
    global_obs = observation.global_
    lines = [
        f"T|{observation.t_s:.1f}|{observation.agent_aux.step_idx}|{global_obs.tier}|{observation.kpi_source_mode}"
    ]
    for cell in observation.cells:
        lines.append(
            f"C|{cell.cell_id}|{cell.prb_util_dl_p50:.3f}|"
            f"{cell.prb_util_dl_p99:.3f}|{cell.prb_util_ul_p50:.3f}|"
            f"{cell.sched_latency_ms_p99:.1f}|{cell.fairness_jain:.3f}|"
            f"{cell.prach_collision_rate:.3f}|{cell.rrc_connected_ues}|"
            f"{cell.sla_violations_last_window}"
        )
        for ue in cell.ues:
            requested = getattr(ue, "requested_mbps", None)
            admitted = getattr(ue, "admitted_mbps", None)
            max_prb = getattr(ue, "prb_cap_max_prb", None)
            requested = ue.offered_mbps if requested is None else requested
            admitted = ue.offered_mbps if admitted is None else admitted
            max_prb = 273 if max_prb is None else max_prb
            lines.append(
                f"U|{cell.cell_id}/{ue.ue_id}|{ue.qos_5qi}|"
                f"{requested:.3f}|{admitted:.3f}|{ue.delivered_mbps:.3f}|"
                f"{'on' if max_prb < 273 else 'off'}|{max_prb}|"
                f"{ue.sinr_db:.2f}|{ue.bler:.3f}|{ue.mcs_mean:.1f}|"
                f"{ue.buffer_occupancy_kb:.1f}|{ue.pdb_violations}"
            )
    aux = observation.agent_aux
    if aux.last_action is not None:
        arguments = json.dumps(aux.last_action.arguments, sort_keys=True, separators=(",", ":"))
        lines.append(f"L|{aux.last_action.name}|{arguments}|{aux.last_rejection or 'none'}")
    if include_action_instruction:
        lines.append("A|one_tool_call_or_noop")
    return "\n".join(lines)


def _to_resource_candidate_pipe_v1(
    observation: Any,
    support: ResourceCandidateSupport,
) -> str:
    """Render compact KPIs plus independently authenticated finite support."""

    base = _to_resource_compact_pipe_v1(
        observation,
        include_action_instruction=False,
    )
    return "\n".join([base, *support.render_rows()])


@lru_cache(maxsize=1)
def _load_t2_compact_renderer() -> Callable[[Any], str]:
    """Load the qualified T2 renderer or fail before the server accepts work."""
    try:
        from openair_congestion.render import to_compact_user_text
    except ImportError as exc:
        raise RuntimeError(
            "observation_render='t2_compact_pipe_v2' requires a telco env package "
            "that exports openair_congestion.render.to_compact_user_text with "
            "truthful T2 P/D policy rows; use 'resource_compact_pipe_v1' for "
            "aggregate dataset replay"
        ) from exc
    return to_compact_user_text


class OpenAirCongestionResourcesServerConfig(BaseResourcesServerConfig):
    model_config = ConfigDict(extra="forbid")

    # Resource-catalog metadata present in the normal Gym YAML.
    verified: bool = False
    description: Optional[str] = None
    value: Optional[str] = None

    # Which Backend drives episodes: 'replay' (default, offline/CI-safe),
    # 'dataset_replay', or 'oai_collector' (live lab; stub today). The
    # OPENAIR_CONGESTION_BACKEND env var overrides. Unknown YAML keys fail at
    # startup so misspelled reward or live-backend settings cannot be ignored.
    backend: Literal["replay", "dataset_replay", "oai_collector"] = "replay"
    # Replay-backend knobs; defaults match openair_congestion.replay_env.ReplayEnv.
    replay_root: str = Field(default="data/replay", min_length=1)
    pool_size: int = Field(default=32, gt=0)
    max_steps_default: int = Field(default=60, gt=0)
    # dataset_replay knobs: replay a recorded dataset (KPI snapshots or GRPO
    # rollout traces; see dataset_backend.py) instead of synthesizing
    # trajectories. cell_capacity_mbps feeds the reward's throughput
    # normalizer; trace episodes recording cell_capacity_mbps_total override it.
    dataset_path: str = Field(default="data/dataset/provided.jsonl", min_length=1)
    cell_capacity_mbps: float = Field(default=60.0, gt=0.0, allow_inf_nan=False)
    reward_profile: Literal[
        "openair_v1",
        "openair_v2_measured",
        "dataset_validity_v1",
        "custom",
    ] = "openair_v1"
    reward_weights: Optional[RewardWeightOverrides] = None
    observation_render: Literal[
        "verbose_v1",
        "resource_compact_pipe_v1",
        "resource_candidate_pipe_v1",
        "t2_compact_pipe_v2",
    ] = "verbose_v1"
    # Truncation-budget fallback for task rows that omit max_steps. Must not
    # exceed the gymnasium_agent's max_steps in the yaml: the agent truncates
    # client-side without notifying the env, so a larger server budget would
    # strand the backend episode slot.
    agent_max_steps: int = Field(default=16, gt=0)

    # Explicitly declared live-backend settings. The backend is still a stub,
    # but validating these now prevents a future deployment from silently
    # dropping a misspelled control-plane setting.
    kpi_url: Optional[str] = None
    oai_pool_size: Optional[int] = Field(default=None, gt=0)
    step_dt_s: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    steady_state_s: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    scenario_mode: Optional[str] = None

    @model_validator(mode="after")
    def bind_reward_profile_to_weights(self) -> "OpenAirCongestionResourcesServerConfig":
        overrides = self.reward_weights.model_dump(exclude_none=True) if self.reward_weights else None
        validate_reward_profile(self.reward_profile, overrides)
        if self.backend != "dataset_replay" and (self.reward_profile != "openair_v1" or overrides):
            raise ValueError(
                "reward_profile/reward_weights apply only to backend='dataset_replay'; "
                f"backend={self.backend!r} uses its environment-owned default reward"
            )
        if self.backend == "dataset_replay" and self.observation_render == "t2_compact_pipe_v2":
            raise ValueError(
                "backend='dataset_replay' cannot truthfully emit t2_compact_pipe_v2 P/D rows; "
                "use observation_render='resource_compact_pipe_v1'"
            )
        return self


class OpenAirCongestionEnv(GymnasiumServer):
    """GymnasiumServer subclass: /reset + /step, driven by gymnasium_agent."""

    config: OpenAirCongestionResourcesServerConfig

    # Backend built once at startup so a bad replay_root / unknown backend
    # name fails at boot, not on the first rollout. Pydantic private attr.
    _backend: Optional[Backend] = None
    _candidate_guardrail_contract: Optional[dict[str, Any]] = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.config.observation_render == "t2_compact_pipe_v2":
            # Resolve at boot, not on the first rollout, so a stale telco env
            # cannot advertise the qualified T2 contract and fail mid-run.
            _load_t2_compact_renderer()
        if self.config.observation_render == RESOURCE_CANDIDATE_CONTRACT:
            self._candidate_guardrail_contract = (
                validate_resource_candidate_guardrail_contract()
            )
        self._backend = select_backend(self.config)

    @property
    def backend(self) -> Backend:
        assert self._backend is not None, "Backend not initialized (model_post_init)"
        return self._backend

    def _live_episode_ids(self) -> set[str]:
        """Episode ids currently owned by live sessions (for the leak reaper)."""
        return {state["episode_id"] for state in self.session_state.values()}

    def _backend_receipt_info(self) -> dict[str, Any]:
        receipt = {
            **self.backend.receipt_info(),
            "observation_render": self.config.observation_render,
        }
        if self.config.observation_render == RESOURCE_CANDIDATE_CONTRACT:
            receipt["candidate_contract"] = RESOURCE_CANDIDATE_CONTRACT
            receipt["candidate_guardrail_contract"] = copy.deepcopy(
                self._candidate_guardrail_contract
            )
        return receipt

    @staticmethod
    def _merge_receipt_info(target: dict[str, Any], receipt: dict[str, Any], *, source: str) -> None:
        """Merge immutable receipt fields without hiding runtime conflicts."""
        conflicts = {
            key: (target[key], value) for key, value in receipt.items() if key in target and target[key] != value
        }
        if conflicts:
            details = ", ".join(
                f"{key}: runtime={runtime!r}, declared={declared!r}"
                for key, (runtime, declared) in sorted(conflicts.items())
            )
            raise RuntimeError(f"{source} receipt conflict ({details})")
        for key, value in receipt.items():
            target.setdefault(key, value)

    def _merge_backend_receipt_info(self, target: dict[str, Any]) -> None:
        # In particular, never overwrite ReplayEnv.step()'s dynamics_mode with
        # a server-side declaration. Matching values are retained verbatim;
        # stale or contradictory declarations fail the rollout.
        self._merge_receipt_info(
            target,
            self._backend_receipt_info(),
            source="backend runtime",
        )

    @staticmethod
    def _recent_excluded_actions(state: dict[str, Any]) -> list[ToolCall]:
        """Actions still inside the shared guardrail's two-step window."""

        current_step = int(state["agent_steps"])
        recent = [
            entry for entry in state.get("recent_accepted_actions", []) if current_step - int(entry["agent_step"]) < 2
        ]
        state["recent_accepted_actions"] = recent
        return [entry["action"] for entry in recent]

    @staticmethod
    def _seed_recent_actions(observation: Any) -> list[dict[str, Any]]:
        """Honor accepted action history surfaced by recorded trace row zero."""

        aux = observation.agent_aux
        last = aux.last_action
        if last is None or aux.last_rejection is not None or last.name == "noop":
            return []
        try:
            action = ToolCall(name=last.name, arguments=dict(last.arguments))
        except ValueError:
            # A malformed historical echo is dataset provenance, not a safe
            # action.  It cannot be repeated through the finite contract.
            return []
        return [{"action": action, "agent_step": 0}]

    def _candidate_support(
        self,
        observation: Any,
        state: dict[str, Any],
    ) -> ResourceCandidateSupport:
        capacity = self.backend.candidate_capacity_mbps_by_cell(
            state["episode_id"],
            observation,
        )
        return derive_resource_candidate_support(
            observation,
            capacity_mbps_by_cell=capacity,
            excluded_actions=self._recent_excluded_actions(state),
        )

    def _render_observation(
        self,
        observation: Any,
        *,
        candidate_support: Optional[ResourceCandidateSupport] = None,
    ) -> str:
        if self.config.observation_render == "resource_compact_pipe_v1":
            return _to_resource_compact_pipe_v1(observation)
        if self.config.observation_render == RESOURCE_CANDIDATE_CONTRACT:
            if candidate_support is None:
                raise RuntimeError("candidate render requires authenticated support")
            return _to_resource_candidate_pipe_v1(observation, candidate_support)
        if self.config.observation_render == "t2_compact_pipe_v2":
            return _load_t2_compact_renderer()(observation)
        return to_user_text(observation)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        # Gymnasium's terminal /step cleanup remains the normal path. This
        # explicit endpoint lets a trainer release a cookie-owned episode on
        # context truncation, cancellation, or any other early exit.
        app.post("/close")(self._close_endpoint)
        return app

    async def _close_endpoint(self, request: Request) -> dict[str, Any]:
        """Idempotently close only the episode owned by this session cookie."""
        session_id = request.session.get(SESSION_ID_KEY)
        closed = session_id in self.session_state
        await self.close_session(session_id)
        return {"ok": True, "closed": closed}

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
            for key in (
                "seed",
                "difficulty",
                "regime_mix",
                "scenario_id",
                "tier",
                "max_steps",
                "dataset_index",
            )
            if metadata.get(key) is not None
        }

        requested_steps_raw = task_params.get("max_steps", self.config.max_steps_default)
        if isinstance(requested_steps_raw, bool):
            raise ValueError("max_steps must be a positive integer, not bool")
        try:
            requested_steps_numeric = float(requested_steps_raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"max_steps must be a positive integer, got {requested_steps_raw!r}") from exc
        if (
            not math.isfinite(requested_steps_numeric)
            or requested_steps_numeric <= 0
            or not requested_steps_numeric.is_integer()
        ):
            raise ValueError(f"max_steps must be a positive integer, got {requested_steps_raw!r}")
        requested_steps = int(requested_steps_numeric)
        effective_steps = min(requested_steps, self.config.agent_max_steps)
        # Give the backend the same cap as the HTTP lifecycle. This keeps its
        # EpisodeMeta, terminal step, and the trainer's turn budget aligned.
        task_params["max_steps"] = effective_steps

        first_obs, meta = self.backend.reset(task_params, live_episode_ids=self._live_episode_ids())
        max_agent_steps = min(effective_steps, int(meta.max_steps))
        self.session_state[session_id] = {
            "episode_id": meta.episode_id,
            "cumulative_reward": 0.0,
            "n_steps": 0,
            # agent_steps counts model turns, n_steps env steps; a turn with
            # no tool call consumes a turn without advancing the env.
            "agent_steps": 0,
            # Cap at the agent's turn budget so the server truncates no later
            # than the agent and the episode slot is freed via close_session().
            "max_agent_steps": max_agent_steps,
            "recent_accepted_actions": self._seed_recent_actions(first_obs),
        }
        state = self.session_state[session_id]
        candidate_support: Optional[ResourceCandidateSupport] = None
        if self.config.observation_render == RESOURCE_CANDIDATE_CONTRACT:
            candidate_support = self._candidate_support(first_obs, state)
            state["candidate_support"] = candidate_support
        # Observation appended as a user message after the dataset prompt.
        reset_info = {
            "episode_id": meta.episode_id,
            "seed": meta.seed,
            "scenario_id": meta.scenario_id,
            "tier": meta.tier,
            "max_steps": meta.max_steps,
        }
        self._merge_backend_receipt_info(reset_info)
        self._merge_receipt_info(
            reset_info,
            self.backend.episode_receipt_info(meta.episode_id),
            source="backend episode",
        )
        if candidate_support is not None:
            self._merge_receipt_info(
                reset_info,
                candidate_support.receipt_fields(),
                source="candidate support",
            )
        return (
            self._render_observation(
                first_obs,
                candidate_support=candidate_support,
            ),
            reset_info,
        )

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        state = self.session_state.get(session_id)
        if state is None:
            # /step without /reset (defensive; gymnasium_agent always resets).
            info = {"error": "no_active_episode"}
            self._merge_backend_receipt_info(info)
            return None, 0.0, False, True, info

        state["agent_steps"] += 1
        out_of_budget = state["agent_steps"] >= state["max_agent_steps"]

        calls = [item for item in action.output if getattr(item, "type", None) == "function_call"]
        call: Optional[NeMoGymResponseFunctionToolCall] = None
        tool_outputs: list[dict[str, Any]] = []
        protocol_error: Optional[str] = None
        protocol_error_detail: Optional[str] = None

        # A malformed model turn still consumes one recorded/synthetic
        # transition and earns the backend's guardrail rejection penalty. This
        # prevents the policy from skipping predominantly-negative KPI steps by
        # emitting text, invalid JSON, or multiple calls. The surrogate action
        # is guaranteed to fail the shared guardrail before any actuator effect.
        if len(calls) != 1:
            protocol_error = "no_tool_call" if not calls else "multiple_tool_calls"
            protocol_error_detail = (
                "exactly one tool call is required" if calls else "no tool call detected; exactly one is required"
            )
            tool_outputs = [
                self.tool_output(
                    extra,
                    {"accepted": False, "error": protocol_error_detail},
                )
                for extra in calls
            ]
            tool_call = ToolCall(name="set_scheduler_policy", arguments={})
        else:
            call = calls[0]
            try:
                raw_args = json.loads(call.arguments) if (call.arguments or "").strip() else {}
                if not isinstance(raw_args, dict):
                    raise ValueError(f"arguments must be a JSON object, got {type(raw_args).__name__}")
                tool_call = ToolCall(name=call.name, arguments=raw_args)
            except ValueError as exc:
                protocol_error = "invalid_tool_call"
                protocol_error_detail = str(exc)
                tool_outputs = [
                    self.tool_output(
                        call,
                        {"accepted": False, "error": protocol_error_detail},
                    )
                ]
                tool_call = ToolCall(name="set_scheduler_policy", arguments={})

        submitted_support: Optional[ResourceCandidateSupport] = state.get("candidate_support")
        submitted_candidate_supported: Optional[bool] = None
        if submitted_support is not None:
            submitted_key = resource_action_key(tool_call)
            submitted_candidate_supported = any(
                resource_action_key(candidate) == submitted_key
                for candidate in submitted_support.actions
            )
            if protocol_error is None and not submitted_candidate_supported:
                assert call is not None
                protocol_error = "candidate_support_violation"
                protocol_error_detail = "tool call is not a member of the authenticated resource candidate support"
                tool_outputs = [
                    self.tool_output(
                        call,
                        {"accepted": False, "error": protocol_error_detail},
                    )
                ]
                # Consume the transition and charge the configured rejection
                # exactly as every other malformed policy turn does.
                tool_call = ToolCall(name="set_scheduler_policy", arguments={})

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
        if protocol_error is not None:
            if accepted:
                raise RuntimeError("protocol-rejection surrogate unexpectedly passed the guardrail")
            accepted = False
            rejection_reason = protocol_error_detail
            next_obs = next_obs.model_copy(
                update={
                    "agent_aux": AgentAux(
                        last_action=LastActionEcho(
                            name=protocol_error,
                            arguments={},
                        ),
                        last_reward=float(reward),
                        last_rejection=rejection_reason,
                        step_idx=int(step_idx),
                    )
                }
            )
        else:
            assert call is not None
            tool_outputs.append(
                self.tool_output(
                    call,
                    {
                        "accepted": accepted,
                        "rejection_reason": rejection_reason,
                        "step_idx": step_idx,
                    },
                )
            )

        if (
            submitted_support is not None
            and submitted_candidate_supported is True
            and protocol_error is None
            and not accepted
        ):
            # Emitted support is a server promise.  Continuing after a
            # candidate is unexpectedly rejected would contaminate a RunB2
            # validity/decision receipt, so end the episode and fail closed.
            await self.close_session(session_id)
            raise RuntimeError(
                f"authenticated resource candidate was rejected by backend: {rejection_reason or 'reason unavailable'}"
            )

        if accepted and protocol_error is None and tool_call.name != "noop":
            state["recent_accepted_actions"].append(
                {
                    "action": tool_call,
                    "agent_step": state["agent_steps"],
                }
            )

        terminated = bool(done)
        truncated = (not terminated) and out_of_budget
        next_candidate_support: Optional[ResourceCandidateSupport] = None
        if terminated or truncated:
            observation = None
        else:
            if submitted_support is not None:
                next_candidate_support = self._candidate_support(next_obs, state)
                state["candidate_support"] = next_candidate_support
            observation = self._render_observation(
                next_obs,
                candidate_support=next_candidate_support,
            )

        response_info = dict(step_info)
        response_info.update(
            {
                "tool_outputs": tool_outputs,
                "guardrail_accepted": accepted,
                "rejection_reason": rejection_reason,
                "step_idx": step_idx,
                "episode_id": state["episode_id"],
                "n_steps": state["n_steps"],
                "cumulative_reward": state["cumulative_reward"],
            }
        )
        self._merge_backend_receipt_info(response_info)
        if submitted_support is not None:
            response_info.update(
                {
                    "submitted_candidate_contract": submitted_support.contract,
                    "submitted_candidate_support_sha256": submitted_support.support_sha256,
                    "submitted_candidate_supported": submitted_candidate_supported,
                }
            )
        if next_candidate_support is not None:
            self._merge_receipt_info(
                response_info,
                next_candidate_support.receipt_fields(),
                source="next candidate support",
            )
        if protocol_error is not None:
            response_info.update(
                {
                    "error": protocol_error,
                    "protocol_rejection": True,
                }
            )

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            response_info,
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
