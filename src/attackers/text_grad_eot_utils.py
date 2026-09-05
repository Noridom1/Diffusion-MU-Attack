"""Dependency-light helpers for the timestep-EOT attack."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def split_timestep_strata(
    sampled_t: Iterable[int], n_strata: int = 3
) -> tuple[tuple[int, ...], ...]:
    """Split sampled timesteps into balanced, non-empty low/mid/high strata."""

    values = tuple(sorted(int(t) for t in sampled_t))
    if n_strata <= 0:
        raise ValueError("n_strata must be positive")
    if len(values) < n_strata:
        raise ValueError(
            f"Need at least {n_strata} sampled timesteps, got {len(values)}"
        )
    return tuple(tuple(int(t) for t in part) for part in np.array_split(values, n_strata))


def make_timestep_schedule(
    sampled_t: Iterable[int],
    num_steps: int,
    *,
    timesteps_per_update: int = 3,
    seed: int = 0,
) -> tuple[tuple[int, ...], ...]:
    """Sample one timestep from each stratum for every optimizer update."""

    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    if timesteps_per_update != 3:
        raise ValueError(
            "The EOT implementation currently requires timesteps_per_update=3 "
            "(one low, one middle, and one high timestep)"
        )
    strata = split_timestep_strata(sampled_t, n_strata=timesteps_per_update)
    rng = np.random.default_rng(seed)
    return tuple(
        tuple(int(rng.choice(stratum)) for stratum in strata)
        for _ in range(num_steps)
    )


def evaluation_steps(total_steps: int, interval: int) -> tuple[int, ...]:
    """Return optimizer steps at which full generation/classifier eval runs."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if interval <= 0:
        raise ValueError("eval_interval must be positive")
    steps = list(range(interval, total_steps + 1, interval))
    if not steps or steps[-1] != total_steps:
        steps.append(total_steps)
    return tuple(steps)
