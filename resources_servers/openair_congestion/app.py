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

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from starlette.middleware.sessions import SessionMiddleware

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseFunctionToolCall
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.gymnasium import GymnasiumServer
from resources_servers.gymnasium import base as gymnasium_base_source

# backends guards the cross-repo 'openair_congestion' import; keep it ahead of
# the telco imports so a missing install fails with the pip hint.
from resources_servers.openair_congestion import backends as backends_source
from resources_servers.openair_congestion import candidate_contract
from resources_servers.openair_congestion.backends import (
    Backend,
    V10FixedReplayBackend,
    select_backend,
)


# isort: split
from openair_congestion import env as env_source
from openair_congestion import guardrail as guardrail_source
from openair_congestion import kpi_client as kpi_client_source
from openair_congestion import render as render_source
from openair_congestion import replay_env as replay_env_source
from openair_congestion import rewards as reward_source
from openair_congestion import schemas as schemas_source
from openair_congestion import t2_action_mask as t2_action_mask_source
from openair_congestion import t2_candidate_sampler as t2_candidate_sampler_source
from openair_congestion import t2_policy_features as t2_policy_features_source
from openair_congestion import tools as tools_source
from openair_congestion import v10_fixed_replay as v10_fixed_replay_source
from openair_congestion.render import to_user_text
from openair_congestion.replay_env import action_effect_version
from openair_congestion.rewards import DEFAULT_WEIGHTS, compute_breakdown
from openair_congestion.schemas import ToolCall
from openair_congestion.v10_fixed_replay import V10_FIXED_REPLAY_SCENARIO_SOURCE


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
    # ``auto`` preserves the standard server's historical optional-generator
    # behavior. V10 rejects it and requires the exact trainer-worktree sampler.
    replay_scenario_source: str = "auto"
    # dataset_replay knobs: replay a recorded dataset (KPI snapshots or GRPO
    # rollout traces; see dataset_backend.py) instead of synthesizing
    # trajectories. cell_capacity_mbps feeds the reward's throughput
    # normalizer; trace episodes recording cell_capacity_mbps_total override it.
    dataset_path: str = "data/dataset/provided.jsonl"
    cell_capacity_mbps: float = 60.0
    # ``standard`` preserves the generic multi-tool resource-server behavior.
    # The explicitly named V10 mode is intentionally narrower: its rendered
    # finite support is T2-only and contains noop + observation-derived UE PRB
    # caps, the only action scope the constrained RunB2 decoder can prove.
    protocol_mode: str = "standard"
    # V10 state is process-local and its task rows must be bounded before an
    # episode is opened.  The hard cap lives in candidate_contract; this
    # configurable lower/equal limit lets a launch deliberately use shorter
    # traces but can never silently open an unbounded episode.
    v10_max_steps: int = candidate_contract.RUNB2_V10_MAX_STEPS_HARD_CAP
    # Compatibility tripwire only: the old V10 prototype exposed this as an
    # independent value.  It is never read for behavior; V10 rejects any
    # supplied value rather than silently accepting a capacity mismatch.
    candidate_cell_capacity_mbps: float | None = None
    # Required only in V10 mode.  Inject a fresh value at launch (for example
    # `${oc.env:OPENAIR_V10_SESSION_SECRET}`); never place a secret in YAML or
    # derive it from the server class/name.  It is used only as the signing
    # key, never echoed in observations, receipts, or config digests.
    v10_session_secret: str | None = None
    # Required, secret-free launch bindings.  The V10 server does not accept a
    # caller-supplied prompt or reset corpus as implicit authority: the runner
    # must pin both SHA-256 values before this stateful process is started.
    v10_system_prompt_sha256: str | None = None
    v10_task_manifest_sha256: str | None = None
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

_STANDARD_PROTOCOL_MODE = "standard"
_SUPPORTED_PROTOCOL_MODES = frozenset({_STANDARD_PROTOCOL_MODE, candidate_contract.RUNB2_V10_PROTOCOL_MODE})
_V10_SESSION_COOKIE_NAME = "openair_v10_session"
_V10_SESSION_MAX_AGE_SECONDS = 15 * 60
_V10_SESSION_SECRET_MIN_CHARS = 43  # len(secrets.token_urlsafe(32))
_V10_TASK_BUDGET_RECEIPT_SCHEMA = "openair_runb2_v10_task_budget_receipt_v1"
_V10_RESET_RECEIPT_SCHEMA = "openair_runb2_v10_reset_receipt_v1"
_V10_TRANSITION_BINDING_SCHEMA = "openair_runb2_v10_transition_binding_v2"
_V10_RUNTIME_MANIFEST_SOURCE_SCHEMA = "openair_runb2_v10_runtime_manifest_v2"
_V10_REPLAY_SCENARIO_SOURCE = V10_FIXED_REPLAY_SCENARIO_SOURCE
_V10_CONGESTION_GEN_SOURCE_RELATIVE_PATHS = {
    "congestion_gen.package": "services/congestion-gen/congestion_gen/__init__.py",
    "congestion_gen.materializer": ("services/congestion-gen/congestion_gen/materializer.py"),
    "congestion_gen.sampler": "services/congestion-gen/congestion_gen/sampler.py",
    "congestion_gen.schemas": "services/congestion-gen/congestion_gen/schemas.py",
    "congestion_gen.validate": "services/congestion-gen/congestion_gen/validate.py",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_V10_REWARD_TERM_KEYS = frozenset(
    {
        "delta_sla",
        "delta_tput",
        "delta_fair",
        "level_sla",
        "level_prb",
        "level_access",
        "level_fair",
        "level_buffer",
        "action",
        "reject",
        "service_denial",
        "forced_termination",
        "total",
    }
)
_V10_REWARD_MEASUREMENT_KEYS = frozenset(
    {
        "delta_sla_violations",
        "delta_delivered_mbps",
        "delta_jain_fairness",
        "sla_violations",
        "aggregate_delivered_mbps",
        "mean_jain_fairness",
        "mean_elastic_jain_fairness",
        "prb_pressure",
        "access_pressure",
        "fairness_deficit",
        "buffer_pressure",
        "action_l1_norm",
        "cell_capacity_mbps_total",
        "n_ues",
        "requested_service_mbps",
        "admitted_service_mbps",
        "delivered_service_mbps",
        "forced_terminated_service_mbps",
        "cumulative_forced_terminated_service_mbps",
        "forced_termination_events",
        "step_forced_terminated_service_mbps",
        "step_forced_termination_events",
        "unadmitted_service_mbps",
        "undelivered_admitted_service_mbps",
    }
)
_V10_SERVER_STEP_RECEIPT_KEYS = frozenset(
    {
        "action",
        "submitted_action",
        "scalar_reward",
        "reward_measurements",
        "reward_measurements_sha256",
        "reward_terms",
        "reward_terms_sha256",
        "guardrail_accepted",
        "protocol_rejection",
        "submitted_candidate_contract",
        "submitted_candidate_support_sha256",
        "submitted_candidate_supported",
        "rejection_reason",
        "error",
        "kpi_source",
        "dynamics_mode",
        "reward_profile",
        "reward_weights",
        "terminated",
        "truncated",
        "training_usable",
        "runtime_manifest_sha256",
        "launch_contract",
        "task_budget_receipt_sha256",
        "reset_receipt_sha256",
        "transition_binding_sha256",
    }
)
_V10_TASK_BUDGET_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "protocol_mode",
        "tier",
        "task_params",
        "task_params_sha256",
        "requested_max_steps",
        "configured_v10_max_steps",
        "configured_agent_max_steps",
        "runtime_manifest_sha256",
        "launch_contract",
    }
)
_V10_RESET_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "task_budget_receipt_sha256",
        "runtime_manifest_sha256",
        "launch_contract",
        "effective_seed",
        "effective_scenario_id",
        "effective_tier",
        "initial_observation_sha256",
        "initial_candidate_support_sha256",
        "initial_binding_payload_sha256",
    }
)


class V10ProtocolError(RuntimeError):
    """The server cannot safely provide the constrained V10 replay contract."""


class OpenAirCongestionEnv(GymnasiumServer):
    """GymnasiumServer subclass: /reset + /step, driven by gymnasium_agent."""

    config: OpenAirCongestionResourcesServerConfig

    # Backend built once at startup so a bad replay_root / unknown backend
    # name fails at boot, not on the first rollout. Pydantic private attr.
    _backend: Optional[Backend] = None
    # Frozen once at process startup.  V10B must identify the code actually
    # imported by this process, not files re-read opportunistically during a
    # later observation after they may have changed on disk.
    _v10_runtime_manifest: dict[str, Any] | None = None
    _v10_runtime_manifest_sha256: str | None = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.config.protocol_mode not in _SUPPORTED_PROTOCOL_MODES:
            raise V10ProtocolError(f"protocol_mode must be one of {sorted(_SUPPORTED_PROTOCOL_MODES)!r}")
        if self._v10_protocol_enabled:
            # Validate the static V10 boundary before even constructing a
            # backend.  This avoids opening a dataset/live implementation on
            # an invalid constrained-decoding launch.
            self._validate_v10_configuration()
        self._backend = select_backend(self.config)
        if self._v10_protocol_enabled:
            if not isinstance(self._backend, V10FixedReplayBackend):
                raise V10ProtocolError(
                    "V10 constrained protocol requires backend='replay' with the "
                    "same-worktree congestion_gen source, never standard replay, dataset_replay, "
                    "or a live collector"
                )
            self._freeze_v10_runtime_manifest()

    @property
    def backend(self) -> Backend:
        assert self._backend is not None, "Backend not initialized (model_post_init)"
        return self._backend

    @property
    def _v10_protocol_enabled(self) -> bool:
        return self.config.protocol_mode == candidate_contract.RUNB2_V10_PROTOCOL_MODE

    @staticmethod
    def _is_strict_positive_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    def _validate_v10_configuration(self) -> None:
        """Fail before serving if V10 would be state-unsafe or unbounded."""

        if self.config.backend.strip().lower() != "replay":
            raise V10ProtocolError("V10 constrained protocol requires config backend='replay'")
        environment_backend = os.environ.get("OPENAIR_CONGESTION_BACKEND")
        if environment_backend is not None and environment_backend.strip().lower() != "replay":
            raise V10ProtocolError(
                "V10 constrained protocol requires OPENAIR_CONGESTION_BACKEND to be unset or 'replay'"
            )
        if self.config.num_workers not in (None, 1):
            raise V10ProtocolError(
                "V10 constrained protocol requires exactly one FastAPI worker; "
                "session state is intentionally process-local"
            )
        # V10 uses a plaintext loopback cookie because the rollout client and
        # stateful server live in one local process namespace.  Do not turn
        # this into a routable unauthenticated service by accident.  ``''`` is
        # accepted only for ASGI in-process tests; a real launch must use the
        # explicit loopback address recorded by its runner receipt.
        if self.config.host not in {"", "127.0.0.1", "::1"}:
            raise V10ProtocolError(
                "V10 constrained protocol requires an in-process test host or a loopback host (127.0.0.1/::1)"
            )
        if self.config.entrypoint not in {"", "app.py"}:
            raise V10ProtocolError("V10 constrained protocol requires the bound app.py entrypoint")
        if self.config.replay_scenario_source != _V10_REPLAY_SCENARIO_SOURCE:
            raise V10ProtocolError(
                f"V10 requires replay_scenario_source={_V10_REPLAY_SCENARIO_SOURCE!r}; 'auto' is forbidden"
            )
        if not self._is_strict_positive_int(self.config.agent_max_steps):
            raise V10ProtocolError("V10 agent_max_steps must be a positive integer")
        if (
            not self._is_strict_positive_int(self.config.v10_max_steps)
            or self.config.v10_max_steps > candidate_contract.RUNB2_V10_MAX_STEPS_HARD_CAP
            or self.config.v10_max_steps > self.config.agent_max_steps
        ):
            raise V10ProtocolError(
                "V10 v10_max_steps must be a positive integer no larger than both "
                f"agent_max_steps and {candidate_contract.RUNB2_V10_MAX_STEPS_HARD_CAP}",
            )
        # `cell_capacity_mbps` is a dataset-replay option that ReplayBackend
        # does not consume.  Refuse a conflicting value in V10 rather than
        # leave an apparent capacity knob that disagrees with the generator's
        # 250-Mbps replay reward/action-effect source.
        capacity = self.config.cell_capacity_mbps
        if (
            not isinstance(capacity, (int, float))
            or isinstance(capacity, bool)
            or not math.isfinite(float(capacity))
            or float(capacity) != candidate_contract.RUNB2_V10_CELL_CAPACITY_MBPS
        ):
            raise V10ProtocolError(
                "V10 fixes cell_capacity_mbps at 250.0 to match congestion_gen reward and candidate-capacity provenance"
            )
        if self.config.candidate_cell_capacity_mbps is not None:
            raise V10ProtocolError(
                "candidate_cell_capacity_mbps was removed for V10; replay capacity is pinned to 250.0 Mbps per cell"
            )
        secret = self.config.v10_session_secret
        # A cryptographic property cannot be inferred from a string, but an
        # explicit token-length launch injection is a meaningful fail-closed
        # boundary.  `secrets.token_urlsafe(32)` (or a 64-char hex token) is
        # suitable; a class/name-derived development default is not.
        if (
            not isinstance(secret, str)
            or not secret.isascii()
            or len(secret) < _V10_SESSION_SECRET_MIN_CHARS
            or not secret.strip()
        ):
            raise V10ProtocolError(
                "V10 requires an injected high-entropy v10_session_secret "
                "(at least 43 ASCII characters; use secrets.token_urlsafe(32))"
            )
        for field_name in ("v10_system_prompt_sha256", "v10_task_manifest_sha256"):
            value = getattr(self.config, field_name)
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise V10ProtocolError(f"V10 requires a pinned lowercase SHA-256 {field_name}")

        reward_capacity_override = os.environ.get("ENV_REWARD_CELL_CAPACITY_MBPS")
        if reward_capacity_override is not None and reward_capacity_override != "":
            if reward_capacity_override != "250.0":
                raise V10ProtocolError("V10 ENV_REWARD_CELL_CAPACITY_MBPS must be unset or exactly 250.0")

    def setup_session_middleware(self, app: FastAPI) -> None:
        """Use a launch-injected signing secret for V10 without leaking it as a cookie name."""

        if not self._v10_protocol_enabled:
            super().setup_session_middleware(app)
            return

        # model_post_init has already validated this value.  Preserve a
        # defensive assertion here because a subclass could call this method
        # during unusual test setup after mutating config.
        secret = self.config.v10_session_secret
        if not isinstance(secret, str) or not secret.isascii() or len(secret) < _V10_SESSION_SECRET_MIN_CHARS:
            raise V10ProtocolError("V10 session middleware secret was not validated")

        # Register this function before SessionMiddleware so its request-side
        # execution sees request.session, matching the base-server ordering.
        @app.middleware("http")
        async def add_session_id(request: Request, call_next):
            request.session[SESSION_ID_KEY] = request.session.get(SESSION_ID_KEY, str(uuid4()))
            response: Response = await call_next(request)
            return response

        # The training launch keeps this stateful server on its private/local
        # path, so an HTTPS-only cookie would break the loopback HTTP client.
        # We still use HttpOnly (Starlette default), Strict SameSite, a short
        # lifetime, and a non-secret cookie name; the high-entropy signing key
        # remains entirely server-side.
        app.add_middleware(
            SessionMiddleware,
            secret_key=secret,
            session_cookie=_V10_SESSION_COOKIE_NAME,
            max_age=_V10_SESSION_MAX_AGE_SECONDS,
            same_site="strict",
            https_only=False,
        )

    @staticmethod
    def _sha256_source_file(path: str | Path, *, label: str) -> str:
        """Hash an import-resolved source file; never substitute an ambient label."""

        try:
            source = Path(path)
            if source.is_symlink():
                raise OSError(f"symlinked source is not an immutable V10 closure member: {source}")
            resolved = source.resolve(strict=True)
            if not resolved.is_file():
                raise OSError(f"not a regular file: {resolved}")
            raw = resolved.read_bytes()
        except OSError as exc:
            raise V10ProtocolError(f"cannot hash V10 {label} source") from exc
        return hashlib.sha256(raw).hexdigest()

    def _v10_public_config(self) -> dict[str, Any]:
        """Return the complete secret-free configuration that affects V10.

        Host, port, and the cookie signing key are deliberately absent: the
        first two are transport custody checked by the V10 runner and the
        latter must never be written to a receipt.  Every option that changes
        replay transitions, reward arithmetic, or state lifetime is present.
        """

        if not isinstance(self._backend, V10FixedReplayBackend):
            raise V10ProtocolError("V10 scenario-source evidence requires V10FixedReplayBackend")
        source_evidence = self._backend.scenario_source_evidence()
        if not isinstance(source_evidence, dict) or (
            source_evidence.get("scenario_source") != _V10_REPLAY_SCENARIO_SOURCE
            or source_evidence.get("cell_capacity_mbps") != candidate_contract.RUNB2_V10_CELL_CAPACITY_MBPS
            or source_evidence.get("dynamic_congestion_gen_importable") is not True
            or source_evidence.get("dynamic_congestion_gen_configured") is not True
            or source_evidence.get("dynamic_congestion_gen_used") is not True
        ):
            raise V10ProtocolError("V10 replay source did not attest exact trainer-worktree congestion_gen use")
        evidence_files = source_evidence.get("congestion_gen_source_files")
        if not isinstance(evidence_files, dict) or set(evidence_files) != set(
            _V10_CONGESTION_GEN_SOURCE_RELATIVE_PATHS
        ):
            raise V10ProtocolError("V10 congestion_gen source evidence is incomplete")
        normalized_source_files: dict[str, dict[str, str]] = {}
        for source_id, relative_path in sorted(_V10_CONGESTION_GEN_SOURCE_RELATIVE_PATHS.items()):
            record = evidence_files.get(source_id)
            if (
                not isinstance(record, dict)
                or set(record) != {"relative_path", "sha256"}
                or record.get("relative_path") != relative_path
                or not isinstance(record.get("sha256"), str)
                or _SHA256_RE.fullmatch(record["sha256"]) is None
            ):
                raise V10ProtocolError(f"V10 congestion_gen source evidence drifted for {source_id}")
            normalized_source_files[source_id] = dict(record)
        return {
            "backend": self.config.backend,
            "replay_root": self.config.replay_root,
            "pool_size": self.config.pool_size,
            "max_steps_default": self.config.max_steps_default,
            "replay_scenario_source": self.config.replay_scenario_source,
            "cell_capacity_mbps": self.config.cell_capacity_mbps,
            "protocol_mode": self.config.protocol_mode,
            "v10_max_steps": self.config.v10_max_steps,
            "agent_max_steps": self.config.agent_max_steps,
            "num_workers": 1 if self.config.num_workers is None else self.config.num_workers,
            "reward_profile": candidate_contract.RUNB2_V10_REWARD_PROFILE,
            "reward_weights": asdict(DEFAULT_WEIGHTS),
            "scenario_source": _V10_REPLAY_SCENARIO_SOURCE,
            "dynamic_congestion_gen_importable": True,
            "dynamic_congestion_gen_configured": True,
            "dynamic_congestion_gen_used": True,
            "congestion_gen_source_files": normalized_source_files,
            "env_reward_cell_capacity_mbps": (os.environ.get("ENV_REWARD_CELL_CAPACITY_MBPS") or "unset"),
        }

    @staticmethod
    def _dependency_version(distribution: str) -> str:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    def _v10_runtime_source_modules(self) -> dict[str, Any]:
        """Return the reviewed Python closure for V10 replay behavior.

        This is deliberately an explicit, reviewable closure rather than an
        accidental snapshot of ``sys.modules``.  Every local module listed
        here participates directly in request/session handling, finite support
        construction, replay transitions, reward calculation, or KPI/render
        semantics. V10 executes ``congestion_gen`` from the same trainer
        worktree, so its complete package-import closure is mandatory here.
        """

        return {
            "resource_server.app": importlib.import_module(__name__),
            "resource_server.backends": backends_source,
            "resource_server.candidate_contract": candidate_contract,
            "gymnasium.package": importlib.import_module("resources_servers.gymnasium"),
            "gymnasium.base": gymnasium_base_source,
            "nemo_gym.base_resources_server": importlib.import_module("nemo_gym.base_resources_server"),
            "nemo_gym.server_utils": importlib.import_module("nemo_gym.server_utils"),
            "nemo_gym.openai_utils": importlib.import_module("nemo_gym.openai_utils"),
            "nemo_gym.config_types": importlib.import_module("nemo_gym.config_types"),
            "nemo_gym.global_config": importlib.import_module("nemo_gym.global_config"),
            "nemo_gym.reward_profile": importlib.import_module("nemo_gym.reward_profile"),
            "nemo_gym.profiling": importlib.import_module("nemo_gym.profiling"),
            "openair_congestion.package": importlib.import_module("openair_congestion"),
            "openair_congestion.env": env_source,
            "openair_congestion.guardrail": guardrail_source,
            "openair_congestion.kpi_client": kpi_client_source,
            "openair_congestion.render": render_source,
            "openair_congestion.replay_env": replay_env_source,
            "openair_congestion.rewards": reward_source,
            "openair_congestion.schemas": schemas_source,
            "openair_congestion.t2_action_mask": t2_action_mask_source,
            "openair_congestion.t2_candidate_sampler": t2_candidate_sampler_source,
            "openair_congestion.t2_policy_features": t2_policy_features_source,
            "openair_congestion.tools": tools_source,
            "openair_congestion.v10_fixed_replay": v10_fixed_replay_source,
            "congestion_gen.package": importlib.import_module("congestion_gen"),
            "congestion_gen.materializer": importlib.import_module("congestion_gen.materializer"),
            "congestion_gen.sampler": importlib.import_module("congestion_gen.sampler"),
            "congestion_gen.schemas": importlib.import_module("congestion_gen.schemas"),
            "congestion_gen.validate": importlib.import_module("congestion_gen.validate"),
        }

    def _v10_runtime_manifest_payload(self) -> dict[str, Any]:
        """Build the frozen, secret-free V10 runtime identity payload."""

        public_config = self._v10_public_config()
        source_files: dict[str, str] = {}
        for logical_name, module in sorted(self._v10_runtime_source_modules().items()):
            source_path = getattr(module, "__file__", None)
            if not isinstance(source_path, str) or not source_path:
                raise V10ProtocolError(f"V10 runtime source path is unavailable for {logical_name}")
            source_files[logical_name] = self._sha256_source_file(source_path, label=logical_name)
        for source_id, record in public_config["congestion_gen_source_files"].items():
            if source_files.get(source_id) != record["sha256"]:
                raise V10ProtocolError(f"V10 congestion_gen evidence/source closure mismatch at {source_id}")
        app_sha = source_files["resource_server.app"]
        return {
            "schema_version": _V10_RUNTIME_MANIFEST_SOURCE_SCHEMA,
            "entrypoint": {
                "logical_path": "resources_servers/openair_congestion/app.py",
                "sha256": app_sha,
            },
            "effective_public_config": public_config,
            "source_files": source_files,
            "dependency_versions": {
                "python": platform.python_version(),
                "fastapi": self._dependency_version("fastapi"),
                "starlette": self._dependency_version("starlette"),
                "pydantic": self._dependency_version("pydantic"),
                "pydantic_core": self._dependency_version("pydantic_core"),
                "numpy": self._dependency_version("numpy"),
                "itsdangerous": self._dependency_version("itsdangerous"),
                "openai": self._dependency_version("openai"),
                "congestion_gen": self._dependency_version("congestion_gen"),
                "uvicorn": self._dependency_version("uvicorn"),
                "omegaconf": self._dependency_version("omegaconf"),
                "ray": self._dependency_version("ray"),
                "aiohttp": self._dependency_version("aiohttp"),
                "orjson": self._dependency_version("orjson"),
            },
            "launch_invariants": {
                "backend": "replay",
                "fixed_cell_capacity_mbps": candidate_contract.RUNB2_V10_CELL_CAPACITY_MBPS,
                "single_worker": True,
                "loopback_or_inprocess_only": True,
                "scenario_source": _V10_REPLAY_SCENARIO_SOURCE,
                "dynamic_congestion_gen_importable": True,
                "dynamic_congestion_gen_configured": True,
                "dynamic_congestion_gen_used": True,
            },
        }

    def _freeze_v10_runtime_manifest(self) -> None:
        payload = self._v10_runtime_manifest_payload()
        # Canonical JSON round-trip prevents later mutation of a nested object
        # from changing the receipt this running process claims to expose.
        self._v10_runtime_manifest = json.loads(candidate_contract.canonical_json(payload))
        self._v10_runtime_manifest_sha256 = candidate_contract.canonical_json_sha256(self._v10_runtime_manifest)

    def _assert_v10_runtime_manifest_is_current(self) -> None:
        """Fail closed if code/config/environment drift after startup."""

        if self._v10_runtime_manifest is None or self._v10_runtime_manifest_sha256 is None:
            raise V10ProtocolError("V10 runtime manifest was not frozen at startup")
        self._validate_v10_configuration()
        current = self._v10_runtime_manifest_payload()
        current_sha = candidate_contract.canonical_json_sha256(current)
        if current_sha != self._v10_runtime_manifest_sha256:
            raise V10ProtocolError("V10 runtime source/config drifted after startup; restart and recollect receipts")

    def _v10_runtime_manifest_ref(self) -> dict[str, str]:
        self._assert_v10_runtime_manifest_is_current()
        assert self._v10_runtime_manifest_sha256 is not None
        return {
            "schema_version": candidate_contract.RUNB2_V10_RUNTIME_MANIFEST_SCHEMA,
            "sha256": self._v10_runtime_manifest_sha256,
        }

    def _v10_runtime_manifest_record(self) -> dict[str, Any]:
        self._assert_v10_runtime_manifest_is_current()
        assert self._v10_runtime_manifest is not None
        return json.loads(candidate_contract.canonical_json(self._v10_runtime_manifest))

    def _v10_launch_contract(self) -> dict[str, str]:
        self._assert_v10_runtime_manifest_is_current()
        system_prompt_sha = self.config.v10_system_prompt_sha256
        task_manifest_sha = self.config.v10_task_manifest_sha256
        assert isinstance(system_prompt_sha, str)
        assert isinstance(task_manifest_sha, str)
        return {
            "schema_version": candidate_contract.RUNB2_V10_LAUNCH_CONTRACT_SCHEMA,
            "system_prompt_sha256": system_prompt_sha,
            "task_manifest_sha256": task_manifest_sha,
        }

    def _v10_environment_contract(self) -> dict[str, Any]:
        """Return the server-owned causal source identity for every V10 turn."""

        if not self._v10_protocol_enabled:
            raise V10ProtocolError("V10 environment contract requested in standard mode")
        return {
            "schema_version": candidate_contract.RUNB2_V10_ACTION_EFFECT_CONTRACT_SCHEMA,
            "backend": "replay",
            "dynamics_mode": action_effect_version(),
            "action_affects_observation": True,
            "candidate_contract": candidate_contract.RESOURCE_CANDIDATE_CONTRACT,
        }

    def _v10_static_info(self, support: candidate_contract.CandidateSupport) -> dict[str, Any]:
        """Return fields that must agree at reset and every accepted step."""

        if (
            support.visible_binding_schema != candidate_contract.RUNB2_V10_VISIBLE_BINDING_SCHEMA
            or support.visible_binding_payload is None
        ):
            raise V10ProtocolError("V10 response attempted to expose support without its visible binding")
        return {
            "protocol_mode": candidate_contract.RUNB2_V10_PROTOCOL_MODE,
            "action_scope": candidate_contract.RUNB2_V10_ACTION_SCOPE,
            "environment_contract": self._v10_environment_contract(),
            "backend": "replay",
            "dynamics_mode": action_effect_version(),
            "action_affects_observation": True,
            "reward_profile": candidate_contract.RUNB2_V10_REWARD_PROFILE,
            "reward_weights": asdict(DEFAULT_WEIGHTS),
            # The V10B row carries the compact reference; expose the complete
            # server-authored object alongside the HTTP provenance so a source
            # builder can seal and independently rehash it without trusting a
            # mutable path on the training host.
            "server_runtime_manifest": self._v10_runtime_manifest_record(),
            "server_runtime_manifest_sha256": self._v10_runtime_manifest_ref()["sha256"],
            "v10_launch_contract": self._v10_launch_contract(),
            **support.metadata(),
        }

    def _v10_support_for_observation(self, observation: Any) -> candidate_contract.CandidateSupport:
        try:
            support = candidate_contract.build_t2_prb_support(observation)
            binding_payload = candidate_contract.build_visible_binding_payload(
                environment_contract=self._v10_environment_contract(),
                reward_weights=asdict(DEFAULT_WEIGHTS),
                runtime_manifest=self._v10_runtime_manifest_ref(),
                launch_contract=self._v10_launch_contract(),
                support=support,
            )
            return candidate_contract.attach_visible_binding(
                support,
                binding_payload=binding_payload,
            )
        except candidate_contract.CandidateContractError as exc:
            raise V10ProtocolError("cannot render the authoritative V10 T2 finite support") from exc

    def _live_episode_ids(self) -> set[str]:
        """Episode ids currently owned by live sessions (for the leak reaper)."""
        return {state["episode_id"] for state in self.session_state.values()}

    def _v10_task_max_steps(self, task_params: dict[str, Any]) -> int:
        """Require a finite, launch-bounded task budget before opening V10 state."""

        value = task_params.get("max_steps")
        if not self._is_strict_positive_int(value):
            raise V10ProtocolError("V10 reset requires an explicit positive integer max_steps")
        if value > self.config.v10_max_steps:
            raise V10ProtocolError("V10 task max_steps exceeds the configured V10 launch bound")
        return value

    def _v10_normalize_task_params(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Validate the complete deterministic reset identity before opening it.

        ReplayBackend supplies defaults for omitted fields, which is useful for
        the generic server but unacceptable for a sealed V10 branch corpus.
        V10 therefore treats its task row as an exact six-field contract.
        """

        expected = {
            "seed",
            "difficulty",
            "regime_mix",
            "scenario_id",
            "tier",
            "max_steps",
        }
        actual = set(metadata)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise V10ProtocolError(
                f"V10 reset task row must have the exact deterministic schema (missing={missing}, extra={extra})"
            )
        seed = metadata["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise V10ProtocolError("V10 task seed must be an integer")
        difficulty = metadata["difficulty"]
        if (
            not isinstance(difficulty, (int, float))
            or isinstance(difficulty, bool)
            or not math.isfinite(float(difficulty))
            or not 0.0 <= float(difficulty) <= 1.0
        ):
            raise V10ProtocolError("V10 task difficulty must be finite in [0, 1]")
        regime_mix = metadata["regime_mix"]
        if not isinstance(regime_mix, dict) or not regime_mix:
            raise V10ProtocolError("V10 task regime_mix must be a non-empty object")
        normalized_mix: dict[str, float] = {}
        supported_regimes = frozenset(getattr(schemas_source, "SUPPORTED_REGIMES", ()))
        for name, weight in regime_mix.items():
            if (
                not isinstance(name, str)
                or not name
                or name not in supported_regimes
                or not isinstance(weight, (int, float))
                or isinstance(weight, bool)
                or not math.isfinite(float(weight))
                or not 0.0 <= float(weight) <= 1.0
            ):
                raise V10ProtocolError("V10 task regime_mix is invalid")
            normalized_mix[name] = float(weight)
        if not math.isclose(sum(normalized_mix.values()), 1.0, rel_tol=0.0, abs_tol=1e-3):
            raise V10ProtocolError("V10 task regime_mix weights must sum to 1.0")
        scenario_id = metadata["scenario_id"]
        if not isinstance(scenario_id, str) or not scenario_id:
            raise V10ProtocolError("V10 task scenario_id must be a non-empty string")
        if metadata["tier"] != "T2":
            raise V10ProtocolError("V10 constrained protocol requires tier='T2'")
        max_steps = self._v10_task_max_steps(metadata)
        return {
            "seed": seed,
            "difficulty": float(difficulty),
            "regime_mix": normalized_mix,
            "scenario_id": scenario_id,
            "tier": "T2",
            "max_steps": max_steps,
        }

    def _v10_task_budget_receipt(self, *, task_params: dict[str, Any], requested_max_steps: int) -> dict[str, Any]:
        """Seal the exact reset task and the server budget it accepted.

        A runner's request alone is not evidence that the stateful server
        applied that bound.  This receipt is computed before the episode state
        is exposed and is then linked to every usable transition.
        """

        normalized_task_params = json.loads(candidate_contract.canonical_json(task_params))
        receipt = {
            "schema_version": _V10_TASK_BUDGET_RECEIPT_SCHEMA,
            "protocol_mode": candidate_contract.RUNB2_V10_PROTOCOL_MODE,
            "tier": "T2",
            "task_params": normalized_task_params,
            "task_params_sha256": candidate_contract.canonical_json_sha256(normalized_task_params),
            "requested_max_steps": requested_max_steps,
            "configured_v10_max_steps": self.config.v10_max_steps,
            "configured_agent_max_steps": self.config.agent_max_steps,
            "runtime_manifest_sha256": self._v10_runtime_manifest_ref()["sha256"],
            "launch_contract": self._v10_launch_contract(),
        }
        if set(receipt) != _V10_TASK_BUDGET_RECEIPT_KEYS:
            raise V10ProtocolError("V10 task budget receipt schema drifted")
        return receipt

    def _v10_reset_receipt(
        self,
        *,
        task_budget_receipt_sha256: str,
        meta: Any,
        support: candidate_contract.CandidateSupport,
    ) -> dict[str, Any]:
        """Seal reset provenance without leaking an ephemeral episode id."""

        if support.visible_binding_payload is None:
            raise V10ProtocolError("V10 reset support has no visible binding payload")
        receipt = {
            "schema_version": _V10_RESET_RECEIPT_SCHEMA,
            "task_budget_receipt_sha256": task_budget_receipt_sha256,
            "runtime_manifest_sha256": self._v10_runtime_manifest_ref()["sha256"],
            "launch_contract": self._v10_launch_contract(),
            "effective_seed": meta.seed,
            "effective_scenario_id": meta.scenario_id,
            "effective_tier": meta.tier,
            "initial_observation_sha256": candidate_contract.text_sha256(support.observation_text),
            "initial_candidate_support_sha256": support.support_sha256,
            "initial_binding_payload_sha256": candidate_contract.canonical_json_sha256(
                support.visible_binding_payload
            ),
        }
        if set(receipt) != _V10_RESET_RECEIPT_KEYS:
            raise V10ProtocolError("V10 reset receipt schema drifted")
        return receipt

    @staticmethod
    def _v10_receipt_info(state: dict[str, Any]) -> dict[str, Any]:
        """Return deep copies of immutable per-episode provenance receipts."""

        required = (
            "v10_task_budget_receipt",
            "v10_task_budget_receipt_sha256",
            "v10_reset_receipt",
            "v10_reset_receipt_sha256",
        )
        if any(key not in state for key in required):
            raise V10ProtocolError("V10 session lost reset provenance receipts")
        task_receipt = state["v10_task_budget_receipt"]
        reset_receipt = state["v10_reset_receipt"]
        task_sha = state["v10_task_budget_receipt_sha256"]
        reset_sha = state["v10_reset_receipt_sha256"]
        if (
            candidate_contract.canonical_json_sha256(task_receipt) != task_sha
            or candidate_contract.canonical_json_sha256(reset_receipt) != reset_sha
        ):
            raise V10ProtocolError("V10 session reset provenance receipt was mutated")
        return {
            "task_budget_receipt": json.loads(candidate_contract.canonical_json(task_receipt)),
            "task_budget_receipt_sha256": task_sha,
            "reset_receipt": json.loads(candidate_contract.canonical_json(reset_receipt)),
            "reset_receipt_sha256": reset_sha,
        }

    def _close_episode_after_v10_failure(self, episode_id: str) -> None:
        """Best-effort cleanup used only after an already-opened V10 episode fails contract checks."""

        try:
            self.backend.close(episode_id)
        except KeyError:
            pass

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
        task_params = (
            self._v10_normalize_task_params(metadata)
            if self._v10_protocol_enabled
            else {
                key: metadata[key]
                for key in (
                    "seed",
                    "difficulty",
                    "regime_mix",
                    "scenario_id",
                    "tier",
                    "max_steps",
                )
                if metadata.get(key) is not None
            }
        )
        v10_max_steps = task_params["max_steps"] if self._v10_protocol_enabled else None
        first_obs, meta = self.backend.reset(task_params, live_episode_ids=self._live_episode_ids())
        try:
            support = self._v10_support_for_observation(first_obs) if self._v10_protocol_enabled else None
            if support is not None:
                assert v10_max_steps is not None
                task_budget_receipt = self._v10_task_budget_receipt(
                    task_params=task_params,
                    requested_max_steps=v10_max_steps,
                )
                task_budget_receipt_sha256 = candidate_contract.canonical_json_sha256(task_budget_receipt)
                reset_receipt = self._v10_reset_receipt(
                    task_budget_receipt_sha256=task_budget_receipt_sha256,
                    meta=meta,
                    support=support,
                )
                reset_receipt_sha256 = candidate_contract.canonical_json_sha256(reset_receipt)
        except Exception:
            # A failed render/binding after reset is still an opened replay
            # episode.  Do not wait for an HTTP close that will never arrive.
            self._close_episode_after_v10_failure(meta.episode_id)
            raise
        self.session_state[session_id] = {
            "episode_id": meta.episode_id,
            "cumulative_reward": 0.0,
            "n_steps": 0,
            # agent_steps counts model turns, n_steps env steps; a turn with
            # no tool call consumes a turn without advancing the env.
            "agent_steps": 0,
            # Cap at the agent's turn budget so the server truncates no later
            # than the agent and the episode slot is freed via close_session().
            "max_agent_steps": (
                v10_max_steps
                if v10_max_steps is not None
                else int(
                    task_params.get("max_steps") or min(self.config.max_steps_default, self.config.agent_max_steps)
                )
            ),
        }
        if support is not None:
            self.session_state[session_id]["candidate_support"] = support
            self.session_state[session_id]["last_observation"] = first_obs
            self.session_state[session_id]["v10_task_budget_receipt"] = task_budget_receipt
            self.session_state[session_id]["v10_task_budget_receipt_sha256"] = task_budget_receipt_sha256
            self.session_state[session_id]["v10_reset_receipt"] = reset_receipt
            self.session_state[session_id]["v10_reset_receipt_sha256"] = reset_receipt_sha256
        # Observation appended as a user message after the dataset prompt.
        info = {
            "episode_id": meta.episode_id,
            "seed": meta.seed,
            "scenario_id": meta.scenario_id,
            "tier": meta.tier,
        }
        if support is not None:
            info.update(self._v10_static_info(support))
            info.update(self._v10_receipt_info(self.session_state[session_id]))
        return (
            support.observation_text if support is not None else to_user_text(first_obs),
            info,
        )

    @staticmethod
    def _finite_mapping(
        value: Any,
        *,
        label: str,
        expected_keys: frozenset[str] | None = None,
    ) -> dict[str, float]:
        if not isinstance(value, dict) or not value:
            raise V10ProtocolError(f"{label} must be a non-empty mapping")
        normalized: dict[str, float] = {}
        for key, raw in value.items():
            if not isinstance(key, str) or not isinstance(raw, (int, float)) or isinstance(raw, bool):
                raise V10ProtocolError(f"{label} must contain only finite numeric values")
            number = float(raw)
            if not math.isfinite(number):
                raise V10ProtocolError(f"{label} must contain only finite numeric values")
            normalized[key] = number
        if expected_keys is not None and set(normalized) != expected_keys:
            missing = sorted(expected_keys - set(normalized))
            extra = sorted(set(normalized) - expected_keys)
            raise V10ProtocolError(f"{label} has the wrong exact schema (missing={missing}, extra={extra})")
        return normalized

    @staticmethod
    def _assert_close_mapping(actual: dict[str, float], expected: dict[str, float], *, label: str) -> None:
        if set(actual) != set(expected):
            raise V10ProtocolError(f"{label} keys differ from the recomputed reward")
        for key, expected_value in expected.items():
            if not math.isclose(actual[key], expected_value, rel_tol=0.0, abs_tol=1.0e-12):
                raise V10ProtocolError(f"{label}.{key} differs from the recomputed V10 reward")

    def _validated_v10_reward_info(
        self,
        *,
        reward: float,
        step_info: dict[str, Any],
        prev_observation: Any,
        next_observation: Any,
        action: ToolCall,
        pre_support: candidate_contract.CandidateSupport,
        post_support: candidate_contract.CandidateSupport,
    ) -> tuple[dict[str, float], dict[str, float], bool]:
        """Prove replay reported the exact, complete reward we independently recompute."""

        if step_info.get("kpi_source") != "replay":
            raise V10ProtocolError("V10 replay step returned a non-replay KPI source")
        if step_info.get("dynamics_mode") != action_effect_version():
            raise V10ProtocolError("V10 replay step returned the wrong dynamics identity")
        accepted = step_info.get("guardrail_accepted")
        if type(accepted) is not bool:
            raise V10ProtocolError("V10 replay step must report boolean guardrail_accepted")
        measurements = self._finite_mapping(
            step_info.get("reward_measurements"),
            label="reward_measurements",
            expected_keys=_V10_REWARD_MEASUREMENT_KEYS,
        )
        pre_capacity_rows = [dict(row) for row in pre_support.capacity_milli_mbps_by_cell]
        post_capacity_rows = [dict(row) for row in post_support.capacity_milli_mbps_by_cell]
        if pre_capacity_rows != post_capacity_rows:
            raise V10ProtocolError("V10 action scope must not change the replay cell-capacity topology")
        expected_capacity_total = candidate_contract.RUNB2_V10_CELL_CAPACITY_MBPS * len(post_capacity_rows)
        if not math.isclose(
            measurements["cell_capacity_mbps_total"],
            expected_capacity_total,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise V10ProtocolError("V10 reward capacity total does not match the bound candidate cells")
        terms = self._finite_mapping(
            step_info.get("reward_terms"),
            label="reward_terms",
            expected_keys=_V10_REWARD_TERM_KEYS,
        )
        if "total" not in terms:
            raise V10ProtocolError("reward_terms must include the scalar total")
        term_sum = sum(value for key, value in terms.items() if key != "total")
        if not math.isclose(term_sum, terms["total"], rel_tol=0.0, abs_tol=1.0e-9):
            raise V10ProtocolError("reward_terms do not sum to their total")
        if not math.isfinite(float(reward)) or not math.isclose(
            float(reward), terms["total"], rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise V10ProtocolError("step scalar reward differs from reward_terms.total")
        recomputed = compute_breakdown(
            prev_obs=prev_observation,
            curr_obs=next_observation,
            action=action,
            rejected=not accepted,
            weights=DEFAULT_WEIGHTS,
            cell_capacity_mbps=candidate_contract.RUNB2_V10_CELL_CAPACITY_MBPS,
            reward_version=candidate_contract.RUNB2_V10_REWARD_PROFILE,
        )
        expected_measurements = self._finite_mapping(
            recomputed.get("measurements"),
            label="recomputed_reward_measurements",
            expected_keys=_V10_REWARD_MEASUREMENT_KEYS,
        )
        expected_terms = self._finite_mapping(
            recomputed.get("terms"),
            label="recomputed_reward_terms",
            expected_keys=_V10_REWARD_TERM_KEYS,
        )
        expected_total = recomputed.get("total")
        if (
            not isinstance(expected_total, (int, float))
            or isinstance(expected_total, bool)
            or not math.isfinite(float(expected_total))
            or not math.isclose(float(expected_total), expected_terms["total"], rel_tol=0.0, abs_tol=1.0e-12)
        ):
            raise V10ProtocolError("recomputed V10 reward did not produce a finite total")
        self._assert_close_mapping(
            measurements,
            expected_measurements,
            label="reward_measurements",
        )
        self._assert_close_mapping(terms, expected_terms, label="reward_terms")
        if not math.isclose(float(reward), float(expected_total), rel_tol=0.0, abs_tol=1.0e-12):
            raise V10ProtocolError("step scalar reward differs from recomputed V10 reward")
        return measurements, terms, accepted

    @staticmethod
    def _v10_server_step_receipt(
        *,
        action: dict[str, Any],
        scalar_reward: float,
        reward_measurements: dict[str, float],
        reward_terms: dict[str, float],
        guardrail_accepted: bool,
        submitted_candidate_support_sha256: str,
        submitted_candidate_supported: bool,
        rejection_reason: Any,
        terminated: bool,
        truncated: bool,
        training_usable: bool,
        runtime_manifest_sha256: str,
        launch_contract: dict[str, Any],
        task_budget_receipt_sha256: str,
        reset_receipt_sha256: str,
        transition_binding_sha256: str,
    ) -> dict[str, Any]:
        """Build the exact server-authored receipt required for one V10 step.

        Both map hashes use V10B canonical JSON plus trailing LF.  A source
        builder must consume this object as-is rather than reconstructing a
        receipt from loosely related response fields.
        """

        receipt = {
            "action": json.loads(candidate_contract.canonical_json(action)),
            "submitted_action": json.loads(candidate_contract.canonical_json(action)),
            "scalar_reward": float(scalar_reward),
            "reward_measurements": json.loads(candidate_contract.canonical_json(reward_measurements)),
            "reward_measurements_sha256": candidate_contract.canonical_json_sha256(reward_measurements),
            "reward_terms": json.loads(candidate_contract.canonical_json(reward_terms)),
            "reward_terms_sha256": candidate_contract.canonical_json_sha256(reward_terms),
            "guardrail_accepted": guardrail_accepted,
            "protocol_rejection": False,
            "submitted_candidate_contract": candidate_contract.RESOURCE_CANDIDATE_CONTRACT,
            "submitted_candidate_support_sha256": submitted_candidate_support_sha256,
            "submitted_candidate_supported": submitted_candidate_supported,
            "rejection_reason": rejection_reason,
            "error": None,
            "kpi_source": "replay",
            "dynamics_mode": action_effect_version(),
            "reward_profile": candidate_contract.RUNB2_V10_REWARD_PROFILE,
            "reward_weights": asdict(DEFAULT_WEIGHTS),
            "terminated": terminated,
            "truncated": truncated,
            "training_usable": training_usable,
            "runtime_manifest_sha256": runtime_manifest_sha256,
            "launch_contract": json.loads(candidate_contract.canonical_json(launch_contract)),
            "task_budget_receipt_sha256": task_budget_receipt_sha256,
            "reset_receipt_sha256": reset_receipt_sha256,
            "transition_binding_sha256": transition_binding_sha256,
        }
        if set(receipt) != _V10_SERVER_STEP_RECEIPT_KEYS:
            raise V10ProtocolError("V10 server step receipt schema drifted")
        return receipt

    def _v10_protocol_rejection(
        self,
        *,
        state: dict[str, Any],
        reason: str,
        tool_outputs: list[dict[str, Any]],
        submitted_action: dict[str, Any] | None = None,
        environment_transition_discarded: bool = False,
    ) -> tuple[Optional[str], float, bool, bool, dict[str, Any]]:
        """Return a terminal, non-trainable receipt for a V10 protocol failure."""

        support = state.get("candidate_support")
        if not isinstance(support, candidate_contract.CandidateSupport):
            raise V10ProtocolError("V10 session lost its pre-step candidate support")
        static_info = self._v10_static_info(support)
        # No observation is returned on a quarantined turn.  Do not attach a
        # `server_observation_binding` whose hash describes the *previous*
        # prompt and could be misread as a returned transition observation.
        static_info.pop("server_observation_binding", None)
        info: dict[str, Any] = {
            "tool_outputs": tool_outputs,
            "episode_id": state["episode_id"],
            "n_steps": state["n_steps"],
            "cumulative_reward": state["cumulative_reward"],
            **static_info,
            **self._v10_receipt_info(state),
            "submitted_candidate_contract": candidate_contract.RESOURCE_CANDIDATE_CONTRACT,
            "submitted_candidate_support_sha256": support.support_sha256,
            "submitted_candidate_supported": False,
            "submitted_action": submitted_action,
            "action": submitted_action,
            "scalar_reward": 0.0,
            "guardrail_accepted": False,
            "protocol_rejection": True,
            "rejection_reason": reason,
            "error": "v10_protocol_rejection",
            "kpi_source": None,
            "terminated": False,
            "truncated": True,
            "training_eligible": False,
            "rollout_usable": False,
            "training_usable": False,
            "terminal_quarantine": True,
            "environment_transition_discarded": environment_transition_discarded,
            # No usable environment transition is exposed after a protocol
            # failure.  Fabricating an env reward decomposition would be worse
            # than an explicit absence, even if a later contract check failed
            # after the backend had already advanced internally.
            "reward_measurements": None,
            "reward_terms": None,
        }
        return None, 0.0, False, True, info

    async def _quarantine_v10_session(
        self,
        *,
        session_id: Optional[str],
        state: dict[str, Any],
        reason: str,
        tool_outputs: list[dict[str, Any]],
        submitted_action: dict[str, Any] | None = None,
        environment_transition_discarded: bool = False,
    ) -> tuple[Optional[str], float, bool, bool, dict[str, Any]]:
        """Build a terminal receipt, then free the process-local backend slot now."""

        try:
            return self._v10_protocol_rejection(
                state=state,
                reason=reason,
                tool_outputs=tool_outputs,
                submitted_action=submitted_action,
                environment_transition_discarded=environment_transition_discarded,
            )
        finally:
            # A runtime-manifest drift can make receipt construction itself
            # fail closed.  Release in ``finally`` so that failure cannot
            # strand a replay-pool slot or leave stale session state behind.
            # The HTTP endpoint also calls close_session on `truncated=True`;
            # this eager release keeps direct method callers equally safe.
            await self._release_session(session_id)

    def _v10_transition_binding(
        self,
        pre_support: candidate_contract.CandidateSupport,
        post_support: candidate_contract.CandidateSupport,
        *,
        action: dict[str, Any],
        task_budget_receipt_sha256: str,
        reset_receipt_sha256: str,
    ) -> dict[str, Any]:
        """Return the exact pre/post server receipt consumed by a branch builder."""

        if pre_support.visible_binding_payload is None or post_support.visible_binding_payload is None:
            raise V10ProtocolError("V10 transition support lost its visible binding")
        pre_capacity_rows = [dict(row) for row in pre_support.capacity_milli_mbps_by_cell]
        post_capacity_rows = [dict(row) for row in post_support.capacity_milli_mbps_by_cell]
        if pre_capacity_rows != post_capacity_rows:
            raise V10ProtocolError("V10 transition changed its fixed replay cell-capacity topology")
        return {
            "schema_version": _V10_TRANSITION_BINDING_SCHEMA,
            "action": json.loads(candidate_contract.canonical_json(action)),
            "runtime_manifest_sha256": self._v10_runtime_manifest_ref()["sha256"],
            "launch_contract": self._v10_launch_contract(),
            "task_budget_receipt_sha256": task_budget_receipt_sha256,
            "reset_receipt_sha256": reset_receipt_sha256,
            "pre_observation_sha256": candidate_contract.text_sha256(pre_support.observation_text),
            "pre_candidate_support_sha256": pre_support.support_sha256,
            "pre_binding_payload_sha256": candidate_contract.canonical_json_sha256(
                pre_support.visible_binding_payload
            ),
            "pre_cell_count": len(pre_capacity_rows),
            "pre_capacity_milli_mbps_total": sum(row["capacity_milli_mbps"] for row in pre_capacity_rows),
            "post_observation": post_support.observation_text,
            "post_observation_sha256": candidate_contract.text_sha256(post_support.observation_text),
            "post_candidate_support_sha256": post_support.support_sha256,
            "post_cell_count": len(post_capacity_rows),
            "post_capacity_milli_mbps_total": sum(row["capacity_milli_mbps"] for row in post_capacity_rows),
            "post_binding_payload": json.loads(
                candidate_contract.canonical_json(post_support.visible_binding_payload)
            ),
        }

    async def _step_v10(
        self,
        *,
        action: NeMoGymResponse,
        state: dict[str, Any],
        session_id: Optional[str],
        out_of_budget: bool,
    ) -> tuple[Optional[str], float, bool, bool, dict[str, Any]]:
        """Execute exactly one pre-support-bound V10 candidate or reject it."""

        support = state.get("candidate_support")
        if not isinstance(support, candidate_contract.CandidateSupport):
            await self._release_session(session_id)
            raise V10ProtocolError("V10 session has no authoritative candidate support")
        prev_observation = state.get("last_observation")
        if prev_observation is None:
            await self._release_session(session_id)
            raise V10ProtocolError("V10 session has no pre-step observation")
        receipt_info = self._v10_receipt_info(state)
        calls = [item for item in action.output if getattr(item, "type", None) == "function_call"]
        if len(calls) != 1:
            if not calls:
                reason = "exactly_one_function_call_required"
                outputs: list[dict[str, Any]] = []
            else:
                reason = "multiple_function_calls_forbidden"
                outputs = [
                    self.tool_output(
                        call,
                        {"accepted": False, "error": reason},
                    )
                    for call in calls
                ]
            return await self._quarantine_v10_session(
                session_id=session_id,
                state=state,
                reason=reason,
                tool_outputs=outputs,
            )

        call: NeMoGymResponseFunctionToolCall = calls[0]
        canonical: dict[str, Any] | None = None
        try:
            raw_args = (
                candidate_contract.parse_strict_json(
                    call.arguments,
                    label="V10 function-call arguments",
                )
                if (call.arguments or "").strip()
                else {}
            )
            if not isinstance(raw_args, dict):
                raise ValueError("arguments must be a JSON object")
            parsed = ToolCall(name=call.name, arguments=raw_args)
            canonical = candidate_contract.canonical_action(parsed)
        except (ValueError, candidate_contract.CandidateContractError) as exc:
            reason = f"invalid_candidate:{exc}"
            return await self._quarantine_v10_session(
                session_id=session_id,
                state=state,
                reason=reason,
                tool_outputs=[self.tool_output(call, {"accepted": False, "error": reason})],
            )
        if not candidate_contract.is_supported_action(canonical, support):
            reason = "candidate_not_in_pre_step_support"
            return await self._quarantine_v10_session(
                session_id=session_id,
                state=state,
                reason=reason,
                tool_outputs=[self.tool_output(call, {"accepted": False, "error": reason})],
                submitted_action=canonical,
            )

        tool_call = ToolCall.model_validate(canonical)
        try:
            next_obs, reward, done, step_info = self.backend.step(state["episode_id"], tool_call)
        except Exception:
            return await self._quarantine_v10_session(
                session_id=session_id,
                state=state,
                reason="backend_step_failure",
                tool_outputs=[
                    self.tool_output(
                        call,
                        {"accepted": False, "error": "backend_step_failure"},
                    )
                ],
                submitted_action=canonical,
                environment_transition_discarded=True,
            )
        try:
            next_support = self._v10_support_for_observation(next_obs)
            measurements, terms, accepted = self._validated_v10_reward_info(
                reward=float(reward),
                step_info=step_info,
                prev_observation=prev_observation,
                next_observation=next_obs,
                action=tool_call,
                pre_support=support,
                post_support=next_support,
            )
            next_static_info = self._v10_static_info(next_support)
            transition_binding = self._v10_transition_binding(
                support,
                next_support,
                action=canonical,
                task_budget_receipt_sha256=receipt_info["task_budget_receipt_sha256"],
                reset_receipt_sha256=receipt_info["reset_receipt_sha256"],
            )
        except Exception as exc:
            # Any inconsistency after the backend has advanced is a discarded
            # transition, never a half-valid training example.  Quarantine it
            # and eagerly release the episode instead of relying on a client
            # to notice an exception and call /close.
            return await self._quarantine_v10_session(
                session_id=session_id,
                state=state,
                reason=f"server_contract_failure:{type(exc).__name__}",
                tool_outputs=[
                    self.tool_output(
                        call,
                        {"accepted": False, "error": "server_contract_failure"},
                    )
                ],
                submitted_action=canonical,
                environment_transition_discarded=True,
            )
        state["candidate_support"] = next_support
        state["last_observation"] = next_obs
        state["cumulative_reward"] += float(reward)
        state["n_steps"] += 1
        rejection_reason = step_info.get("rejection_reason")
        terminated = bool(done)
        truncated = (not terminated) and out_of_budget
        training_usable = accepted and not terminated and not truncated
        # `server_observation_binding` must describe an observation actually
        # returned in this response.  Terminal/truncated steps still retain a
        # post transition receipt below, but do not claim a fresh model prompt.
        response_static_info = dict(next_static_info)
        if terminated or truncated:
            response_static_info.pop("server_observation_binding", None)
        transition_binding_sha256 = candidate_contract.canonical_json_sha256(transition_binding)
        runtime_manifest_sha256 = self._v10_runtime_manifest_ref()["sha256"]
        launch_contract = self._v10_launch_contract()
        server_step_receipt = self._v10_server_step_receipt(
            action=canonical,
            scalar_reward=float(reward),
            reward_measurements=measurements,
            reward_terms=terms,
            guardrail_accepted=accepted,
            submitted_candidate_support_sha256=support.support_sha256,
            submitted_candidate_supported=True,
            rejection_reason=rejection_reason,
            terminated=terminated,
            truncated=truncated,
            training_usable=training_usable,
            runtime_manifest_sha256=runtime_manifest_sha256,
            launch_contract=launch_contract,
            task_budget_receipt_sha256=receipt_info["task_budget_receipt_sha256"],
            reset_receipt_sha256=receipt_info["reset_receipt_sha256"],
            transition_binding_sha256=transition_binding_sha256,
        )
        server_step_receipt_sha256 = candidate_contract.canonical_json_sha256(server_step_receipt)
        info: dict[str, Any] = {
            "tool_outputs": [
                self.tool_output(
                    call,
                    {
                        "accepted": accepted,
                        "rejection_reason": rejection_reason,
                        "step_idx": step_info.get("step_idx", state["n_steps"]),
                    },
                )
            ],
            "episode_id": state["episode_id"],
            "n_steps": state["n_steps"],
            "cumulative_reward": state["cumulative_reward"],
            **response_static_info,
            **receipt_info,
            "submitted_candidate_contract": candidate_contract.RESOURCE_CANDIDATE_CONTRACT,
            "submitted_candidate_support_sha256": support.support_sha256,
            "submitted_candidate_supported": True,
            "submitted_action": canonical,
            "action": canonical,
            "scalar_reward": float(reward),
            "guardrail_accepted": accepted,
            "protocol_rejection": False,
            "rejection_reason": rejection_reason,
            "error": None,
            "step_idx": step_info.get("step_idx", state["n_steps"]),
            "reward_measurements": measurements,
            "reward_terms": terms,
            "kpi_source": step_info["kpi_source"],
            "terminated": terminated,
            "truncated": truncated,
            "training_eligible": training_usable,
            "rollout_usable": training_usable,
            "training_usable": training_usable,
            "terminal_quarantine": False,
            "server_step_receipt": server_step_receipt,
            "server_step_receipt_sha256": server_step_receipt_sha256,
            "server_transition_binding": transition_binding,
            "server_transition_binding_sha256": transition_binding_sha256,
            "prb_cap_dynamics": step_info.get("prb_cap_dynamics", {}),
        }
        if terminated or truncated:
            # Direct method callers do not go through GymnasiumServer's HTTP
            # endpoint (which also closes terminal sessions).  Release here as
            # well so the bounded V10 episode cannot retain a replay slot.
            await self._release_session(session_id)
        return (
            None if (terminated or truncated) else next_support.observation_text,
            float(reward),
            terminated,
            truncated,
            info,
        )

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        state = self.session_state.get(session_id)
        if state is None:
            # /step without /reset (defensive; gymnasium_agent always resets).
            if self._v10_protocol_enabled:
                return (
                    None,
                    0.0,
                    False,
                    True,
                    {
                        "error": "no_active_episode",
                        "scalar_reward": 0.0,
                        "protocol_rejection": True,
                        "guardrail_accepted": False,
                        "terminated": False,
                        "truncated": True,
                        "training_eligible": False,
                        "rollout_usable": False,
                        "training_usable": False,
                        "terminal_quarantine": True,
                    },
                )
            return None, 0.0, False, True, {"error": "no_active_episode"}

        state["agent_steps"] += 1
        out_of_budget = state["agent_steps"] >= state["max_agent_steps"]

        if self._v10_protocol_enabled:
            return await self._step_v10(
                action=action,
                state=state,
                session_id=session_id,
                out_of_budget=out_of_budget,
            )

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

    async def _release_session(self, session_id: Optional[str]) -> dict[str, Any]:
        """Free one backend slot exactly once and retain no stale session state."""

        state = self.session_state.pop(session_id, None)
        if state is None:
            return {"ok": True, "already_closed": True, "summary": {}}
        try:
            summary = self.backend.close(state["episode_id"])
        except KeyError:
            # The underlying env can close an episode on a terminal step.  It
            # is still safe to consume our session state exactly once.
            summary = {"ok": True, "already_closed_by_backend": True}
        return {"ok": True, "already_closed": False, "summary": summary}

    async def close_session(self, session_id: Optional[str]) -> None:
        # Framework calls this when a step returns terminated or truncated.
        await self._release_session(session_id)

    async def explicit_close(self, session_id: Optional[str]) -> dict[str, Any]:
        """Cookie-scoped, idempotent HTTP cleanup for a branch client finally block."""

        return await self._release_session(session_id)


if __name__ == "__main__":
    OpenAirCongestionEnv.run_webserver()
