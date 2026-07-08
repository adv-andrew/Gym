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

"""Backend abstraction for the openair_congestion resources server.

The gymnasium-style server in ``app.py`` talks to a :class:`Backend`, which
owns episode lifecycles:

    reset(task_params, live_episode_ids=...) -> (Observation, EpisodeMeta)
    step(episode_id, tool_call)              -> (Observation, reward, terminated, info)
    close(episode_id)                        -> summary dict

Three drivers implement the contract:

- :class:`ReplayBackend` ('replay', the default): offline and deterministic.
  Wraps the cross-repo ``openair_congestion.replay_env.ReplayEnv``; no 5G
  lab, no GPU, no KPI exporter, no wall-clock sleeps.
- ``DatasetReplayBackend`` ('dataset_replay', in ``dataset_backend.py``):
  offline replay of a recorded dataset (KPI snapshots or GRPO rollout
  traces) instead of seed-synthesized trajectories.
- :class:`OAICollectorBackend` ('oai_collector'): online collection from a
  live OpenAirInterface 5G docker stack; a stub until lab wiring lands.

Selection is via :func:`select_backend`. The YAML ``backend:`` field is the
canonical switch; the ``OPENAIR_CONGESTION_BACKEND`` env var overrides it for
local development.

Episode slots are finite (``pool_size``) and normally free via close(). If a
rollout dies between /reset and its terminal /step the slot would leak, so
``reset()`` accepts ``live_episode_ids`` — the episode ids still owned by
live sessions — and backends reap orphaned episodes when the pool is
exhausted.

Rewards are not touched here: ``ReplayEnv.step()`` computes the per-step
reward internally via ``rewards.compute_breakdown()`` and this layer passes
its total through unchanged.
"""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Optional, Protocol


# Import guard for the cross-repo env package: fail with the install hint
# rather than a bare ModuleNotFoundError. app.py and dataset_backend.py import
# this module first, so the guard covers them too.
try:
    import openair_congestion  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only when unpackaged
    raise ImportError(
        "Could not import the 'openair_congestion' telco env package. It lives "
        "in the openair-rl-gym repo and must be installed in this venv, e.g.:\n"
        "  pip install -e <openair-rl-gym>/env/nemo_gym/envs/openair_congestion\n"
        "See resources_servers/openair_congestion/README.md (Setup)."
    ) from exc

from openair_congestion import replay_env as _replay_env  # noqa: E402
from openair_congestion import rewards as _rewards  # noqa: E402
from openair_congestion.schemas import EpisodeMeta, Observation, ToolCall  # noqa: E402


ReplayEnv = _replay_env.ReplayEnv


NAMED_REWARD_PROFILE_OVERRIDES: dict[str, dict[str, float]] = {
    "openair_v1": {},
    "openair_v2_measured": {
        "w_sla": 0.0,
        "w_sla_level": 0.0,
        "w_buffer": 0.0,
        "w_action": 0.0,
    },
    # Recorded KPIs are pass-through, so this profile intentionally makes
    # validity the entire objective: accepted calls score exactly zero and a
    # rejected call scores exactly -0.5.  Keeping every coefficient explicit
    # prevents a future RewardWeights default from silently changing that
    # claim while retaining the same receipt label.
    "dataset_validity_v1": {
        "w_sla": 0.0,
        "w_tput": 0.0,
        "w_fair": 0.0,
        "w_buffer": 0.0,
        "w_sla_level": 0.0,
        "w_prb_level": 0.0,
        "w_access_level": 0.0,
        "w_fair_level": 0.0,
        "w_action": 0.0,
        "w_reject": 0.5,
    },
}


def _replay_action_effect_version() -> str:
    """Resolve and cross-check the action-effect model executed by ReplayEnv.

    The telco environment owns this identifier.  Requiring its constant and
    callable receipt to agree prevents this server from silently advertising a
    stale, locally hard-coded dynamics version after that environment changes.
    """
    exported = getattr(_replay_env, "ACTION_EFFECT_VERSION", None)
    reporter = getattr(_replay_env, "action_effect_version", None)
    if not isinstance(exported, str) or not exported or exported.strip() != exported:
        raise RuntimeError("ReplayEnv must export a non-empty, whitespace-free ACTION_EFFECT_VERSION receipt")
    if not callable(reporter):
        raise RuntimeError("ReplayEnv must export callable action_effect_version()")
    try:
        reported = reporter()
    except Exception as exc:
        raise RuntimeError("ReplayEnv action_effect_version() receipt failed") from exc
    if not isinstance(reported, str) or not reported or reported.strip() != reported:
        raise RuntimeError("ReplayEnv action_effect_version() must return a non-empty, whitespace-free string")
    if reported != exported:
        raise RuntimeError(
            "ReplayEnv action-effect receipt mismatch: "
            f"ACTION_EFFECT_VERSION={exported!r}, action_effect_version()={reported!r}"
        )
    return reported


def validate_reward_profile(profile: str, overrides: Optional[dict[str, float]]) -> None:
    """Keep the receipt label bound to the effective named-profile overrides."""
    supplied = overrides or {}
    if profile == "custom":
        if not supplied:
            raise ValueError("reward_profile='custom' requires non-empty reward_weights")
        return
    expected = NAMED_REWARD_PROFILE_OVERRIDES.get(profile)
    if expected is None:
        raise ValueError(
            f"unknown reward_profile {profile!r}; valid: {sorted(NAMED_REWARD_PROFILE_OVERRIDES)} + ['custom']"
        )
    if supplied != expected:
        raise ValueError(f"reward_profile={profile!r} requires exact reward_weights={expected!r}; got {supplied!r}")


class TelemetryDriver(Protocol):
    """KPI read-path seam for online backends.

    In the live stack, telemetry is scraped from a Prometheus-style
    kpi-exporter endpoint (``openair_congestion.kpi_client.fetch(url)``).
    :class:`OAICollectorBackend` can later accept an injectable driver (real
    exporter vs. recorded fixture) without changing the Backend contract.
    ReplayBackend needs no driver: its KPIs come from the trajectory built at
    reset().
    """

    def fetch(self, url: str) -> dict[str, Any]:  # pragma: no cover - protocol
        """Scrape one KPI snapshot (raw metric name -> value) from `url`."""
        ...


class Backend(ABC):
    """Episode-oriented environment driver behind the gymnasium server.

    ``task_params`` is the plain dict of scenario controls taken from the
    task row (seed / difficulty / regime_mix / scenario_id / tier /
    max_steps); keys map 1:1 onto ``ReplayEnv.reset()`` keyword arguments.
    """

    backend_name = "unknown"
    dynamics_mode = "unknown"
    action_affects_observation = False
    reward_profile = "unknown"

    def receipt_info(self) -> dict[str, Any]:
        """Return immutable backend semantics included in every HTTP receipt."""
        return {
            "backend": self.backend_name,
            "dynamics_mode": self.dynamics_mode,
            "action_affects_observation": self.action_affects_observation,
            "reward_profile": self.reward_profile,
        }

    def episode_receipt_info(self, episode_id: str) -> dict[str, Any]:
        """Return episode-specific provenance for reset receipts, if any."""
        return {}

    def candidate_capacity_mbps_by_cell(
        self,
        episode_id: str,
        observation: Observation,
    ) -> dict[int, float]:
        """Return the effective capacity used to derive finite candidates.

        Candidate rendering is opt-in.  Backends must expose the same
        per-cell capacity that their dynamics/reward path uses rather than
        allowing the server to guess from tier names or a stale constant.
        """

        raise RuntimeError(f"backend {self.backend_name!r} does not expose candidate capacity")

    @abstractmethod
    def reset(
        self, task_params: dict[str, Any], *, live_episode_ids: Optional[set[str]] = None
    ) -> tuple[Observation, EpisodeMeta]:
        """Start a new episode; returns (first Observation, EpisodeMeta).

        ``meta.episode_id`` is the handle for subsequent step()/close() calls.
        ``live_episode_ids`` is the set of episode ids still owned by live
        sessions; backends may use it to reap orphaned episodes when their
        pool is exhausted.
        """

    @abstractmethod
    def step(self, episode_id: str, tool_call: ToolCall) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Apply one action; returns (next_obs, reward, terminated, info).

        ``reward`` is the per-step total already computed inside the env
        (rewards.compute_breakdown), passed through unchanged. ``info``
        carries guardrail_accepted / rejection_reason / step_idx /
        reward_terms / reward_measurements / kpi_source / dynamics_mode.
        """

    @abstractmethod
    def close(self, episode_id: str) -> dict[str, Any]:
        """Release the episode slot; returns a summary like {ok, n_steps}."""


class ReplayBackend(Backend):
    """Offline deterministic driver wrapping ``ReplayEnv`` (the default).

    Standalone: no lab, no GPU, no exporter. One shared ReplayEnv instance
    manages all episodes by episode_id; it is internally locked (per-episode
    threading.RLock), so a single instance is safe across concurrent sessions.

    Leak safety: this backend tracks every episode id it creates. If
    ``ReplayEnv.reset()`` raises its pool-exhausted RuntimeError, episodes not
    referenced by any live session (``live_episode_ids``) are closed as leaked
    and the reset is retried exactly once.
    """

    backend_name = "replay"
    action_affects_observation = True
    reward_profile = "env_default"

    def receipt_info(self) -> dict[str, Any]:
        return {
            **super().receipt_info(),
            "reward_weights": asdict(_rewards.DEFAULT_WEIGHTS),
        }

    def __init__(
        self,
        *,
        replay_root: str = "data/replay",
        pool_size: int = 32,
        max_steps_default: int = 60,
    ) -> None:
        # The installed telco environment is the sole authority for synthetic
        # dynamics provenance. Resolve it at server boot and fail closed if its
        # two public receipts are absent or inconsistent.
        self.dynamics_mode = _replay_action_effect_version()
        self._env = ReplayEnv(
            replay_root=replay_root,
            pool_size=pool_size,
            max_steps_default=max_steps_default,
        )
        # Episode ids created here and not yet closed, for the leak reaper.
        self._open_episode_ids: set[str] = set()
        self._track_lock = threading.Lock()

    def reset(
        self, task_params: dict[str, Any], *, live_episode_ids: Optional[set[str]] = None
    ) -> tuple[Observation, EpisodeMeta]:
        try:
            first_obs, meta = self._reset_env(task_params)
        except RuntimeError as exc:
            if "pool exhausted" not in str(exc):
                raise
            # Reap episodes no session owns anymore (crashed rollouts), retry once.
            self._reap_leaked(live_episode_ids or set())
            first_obs, meta = self._reset_env(task_params)
        with self._track_lock:
            self._open_episode_ids.add(meta.episode_id)
        return first_obs, meta

    def _reset_env(self, task_params: dict[str, Any]) -> tuple[Observation, EpisodeMeta]:
        # Keys map 1:1 to ReplayEnv.reset() kwargs; defaults mirror the env's.
        return self._env.reset(
            seed=int(task_params.get("seed", 0)),
            difficulty=float(task_params.get("difficulty", 0.5)),
            regime_mix=task_params.get("regime_mix"),
            scenario_id=task_params.get("scenario_id"),
            tier=str(task_params.get("tier", "replay")),
            max_steps=task_params.get("max_steps"),
        )

    def candidate_capacity_mbps_by_cell(
        self,
        episode_id: str,
        observation: Observation,
    ) -> dict[int, float]:
        # ReplayEnv does not yet publish capacity in EpisodeMeta.  Read the
        # exact fingerprint owned by the installed runtime under its lock;
        # fail closed if that versioned internal contract disappears instead
        # of falling back to a guessed 60 Mbps label.
        lock = getattr(self._env, "_lock", None)
        episodes = getattr(self._env, "_episodes", None)
        if lock is None or not isinstance(episodes, dict):
            raise RuntimeError("installed ReplayEnv does not expose episode capacity")
        with lock:
            episode = episodes.get(episode_id)
            fingerprint = getattr(episode, "fingerprint", None)
            capacity = getattr(fingerprint, "cell_capacity_mbps", None)
        try:
            capacity_value = float(capacity)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("ReplayEnv episode has invalid cell capacity") from exc
        return {cell.cell_id: capacity_value for cell in observation.cells}

    def _reap_leaked(self, live_episode_ids: set[str]) -> None:
        with self._track_lock:
            leaked = [eid for eid in self._open_episode_ids if eid not in live_episode_ids]
        for episode_id in leaked:
            try:
                self._env.close(episode_id)
            except KeyError:
                pass  # already gone inside the env
            with self._track_lock:
                self._open_episode_ids.discard(episode_id)

    def step(self, episode_id: str, tool_call: ToolCall) -> tuple[Observation, float, bool, dict[str, Any]]:
        observation, reward, terminated, info = self._env.step(episode_id, tool_call)
        runtime_mode = info.get("dynamics_mode")
        if runtime_mode != self.dynamics_mode:
            state = "missing" if runtime_mode is None else f"inconsistent ({runtime_mode!r})"
            raise RuntimeError(
                f"ReplayEnv step dynamics_mode receipt is {state}; expected installed runtime {self.dynamics_mode!r}"
            )
        return observation, reward, terminated, info

    def close(self, episode_id: str) -> dict[str, Any]:
        with self._track_lock:
            self._open_episode_ids.discard(episode_id)
        return self._env.close(episode_id)


class OAICollectorBackend(Backend):
    """Live 5G RAN KPI collection from the OAI docker stack (stub; wiring
    deferred until lab access).

    Read path (per ``openair_congestion.env.LiveEnv``, the wiring target):
    - TelemetryDriver = kpi-exporter scrape at ``KPI_EXPORTER_URL`` (default
      ``http://localhost:9090/metrics``) via ``kpi_client.fetch(url)``.
    - reset(): optionally start a scenario controller ('t1_runner' traffic
      simulator), wait ``steady_state_s`` for metrics to stabilise, scrape the
      first snapshot, build the first Observation.
    - step(): guardrail-check the action, apply the actuator (currently only
      set_admission_policy is wired, to traffic-side stream suppression),
      sleep ``step_dt_s`` of wall-clock, scrape again, compute the reward via
      rewards.compute_breakdown(), return the same 4-tuple as ReplayBackend.

    Constructor knobs (all forwarded by select_backend today; the env-var
    fallbacks live in the future LiveEnv wiring):
    - kpi_url          <- KPI_EXPORTER_URL (Prometheus /metrics endpoint)
    - pool_size        <- ENV_POOL_SIZE      (episode slots, default 4)
    - step_dt_s        <- ENV_STEP_DT_S      (seconds between steps, default 1.0)
    - steady_state_s   <- ENV_STEADY_STATE_S (settle time at reset, default 1.0)
    - scenario_mode    <- ENV_SCENARIO_MODE  ('t1_runner' wires the traffic sim)

    Not yet connected: FlexRIC RC / OAI telnet RAN control (only
    set_admission_policy actuation exists today), multi-cell setups beyond T1
    (2 cells x 4 UEs), and the lab 5G docker stack itself.
    """

    backend_name = "oai_collector"
    dynamics_mode = "live_oai_collector"
    action_affects_observation = True
    reward_profile = "env_default"

    def __init__(
        self,
        *,
        kpi_url: Optional[str] = None,
        pool_size: Optional[int] = None,
        step_dt_s: Optional[float] = None,
        steady_state_s: Optional[float] = None,
        max_steps_default: int = 60,
        scenario_mode: Optional[str] = None,
    ) -> None:
        raise NotImplementedError(
            "oai_collector backend requires lab access to the OAI 5G docker "
            "stack (KPI exporter + traffic runner). Use backend='replay' for "
            "standalone SFT/GRPO."
        )

    # Signatures mirror ReplayBackend exactly so the swap is config-only.
    def reset(
        self, task_params: dict[str, Any], *, live_episode_ids: Optional[set[str]] = None
    ) -> tuple[Observation, EpisodeMeta]:  # pragma: no cover
        raise NotImplementedError

    def step(
        self, episode_id: str, tool_call: ToolCall
    ) -> tuple[Observation, float, bool, dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    def close(self, episode_id: str) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


def select_backend(config: Any) -> Backend:
    """Build the configured Backend (dependency injection point for app.py).

    Precedence: OPENAIR_CONGESTION_BACKEND env var > config.backend > 'replay'.
    ``config`` is duck-typed (the app's config object) so this module never
    imports app.py.
    """
    name = os.environ.get("OPENAIR_CONGESTION_BACKEND") or getattr(config, "backend", None) or "replay"
    name = name.strip().lower()
    if name == "replay":
        return ReplayBackend(
            replay_root=getattr(config, "replay_root", "data/replay"),
            pool_size=getattr(config, "pool_size", 32),
            max_steps_default=getattr(config, "max_steps_default", 60),
        )
    if name == "dataset_replay":
        # Local import so the default replay path never pays for (or fails
        # on) ingestion code.
        from resources_servers.openair_congestion.dataset_backend import (
            DatasetReplayBackend,
        )

        reward_weights = getattr(config, "reward_weights", None)
        if hasattr(reward_weights, "model_dump"):
            reward_weights = reward_weights.model_dump(exclude_none=True)

        return DatasetReplayBackend(
            dataset_path=getattr(config, "dataset_path", "data/dataset/provided.jsonl"),
            pool_size=getattr(config, "pool_size", 32),
            max_steps_default=getattr(config, "max_steps_default", 60),
            cell_capacity_mbps=getattr(config, "cell_capacity_mbps", 60.0),
            reward_profile=getattr(config, "reward_profile", "openair_v1"),
            reward_weights=reward_weights,
        )
    if name == "oai_collector":
        # Forward all documented knobs now (the stub ignores them) so the
        # future lab wiring is config-only. 'oai_pool_size' is a distinct
        # yaml key because 'pool_size' already configures the replay backend
        # (32) while the live env's own default is 4.
        return OAICollectorBackend(
            kpi_url=getattr(config, "kpi_url", None),
            pool_size=getattr(config, "oai_pool_size", None),
            step_dt_s=getattr(config, "step_dt_s", None),
            steady_state_s=getattr(config, "steady_state_s", None),
            scenario_mode=getattr(config, "scenario_mode", None),
            max_steps_default=getattr(config, "max_steps_default", 60),
        )
    raise ValueError(f"unknown backend {name!r}; valid: 'replay', 'dataset_replay', 'oai_collector'")
