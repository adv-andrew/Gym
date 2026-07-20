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

from openair_congestion.replay_env import ReplayEnv  # noqa: E402
from openair_congestion.schemas import EpisodeMeta, Observation, ToolCall  # noqa: E402
from openair_congestion.v10_fixed_replay import (  # noqa: E402
    V10_FIXED_REPLAY_SCENARIO_SOURCE,
    V10CongestionGenReplayEnv,
)


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

    def __init__(
        self,
        *,
        replay_root: str = "data/replay",
        pool_size: int = 32,
        max_steps_default: int = 60,
    ) -> None:
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
        return self._env.step(episode_id, tool_call)

    def close(self, episode_id: str) -> dict[str, Any]:
        with self._track_lock:
            self._open_episode_ids.discard(episode_id)
        return self._env.close(episode_id)


class V10FixedReplayBackend(ReplayBackend):
    """Replay driver requiring the exact same-worktree ``congestion_gen``."""

    def __init__(
        self,
        *,
        replay_root: str = "data/replay",
        pool_size: int = 32,
        max_steps_default: int = 60,
    ) -> None:
        self._env = V10CongestionGenReplayEnv(
            replay_root=replay_root,
            pool_size=pool_size,
            max_steps_default=max_steps_default,
        )
        self._open_episode_ids: set[str] = set()
        self._track_lock = threading.Lock()

    def scenario_source_evidence(self) -> dict[str, Any]:
        """Expose the fixed env's immutable source/runtime-use evidence."""

        return self._env.scenario_source_evidence()


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
        common = {
            "replay_root": getattr(config, "replay_root", "data/replay"),
            "pool_size": getattr(config, "pool_size", 32),
            "max_steps_default": getattr(config, "max_steps_default", 60),
        }
        scenario_source = getattr(config, "replay_scenario_source", "auto")
        if scenario_source == "auto":
            return ReplayBackend(**common)
        if scenario_source == V10_FIXED_REPLAY_SCENARIO_SOURCE:
            return V10FixedReplayBackend(**common)
        raise ValueError(
            "unknown replay_scenario_source "
            f"{scenario_source!r}; expected 'auto' or "
            f"{V10_FIXED_REPLAY_SCENARIO_SOURCE!r}"
        )
    if name == "dataset_replay":
        # Local import so the default replay path never pays for (or fails
        # on) ingestion code.
        from resources_servers.openair_congestion.dataset_backend import (
            DatasetReplayBackend,
        )

        return DatasetReplayBackend(
            dataset_path=getattr(config, "dataset_path", "data/dataset/provided.jsonl"),
            pool_size=getattr(config, "pool_size", 32),
            max_steps_default=getattr(config, "max_steps_default", 60),
            cell_capacity_mbps=getattr(config, "cell_capacity_mbps", 60.0),
            reward_weights=getattr(config, "reward_weights", None),
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
