#!/usr/bin/env python3
"""Simulate single-cell pharmacokinetic variability using Reactors SSA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import reactors


@dataclass(frozen=True)
class PKModel:
    stoich: np.ndarray
    initial_state: np.ndarray
    rate_constants: np.ndarray
    reaction_type_codes: np.ndarray
    t_points: np.ndarray
    complex_index: int = 2


def build_model() -> PKModel:
    # Species: Drug (D), Target (T), Complex (C)
    stoich = np.array(
        [
            [-1, -1, 1],   # D + T -> C (binding)
            [1, 1, -1],    # C -> D + T (unbinding)
            [0, 0, -1],    # C -> ∅ (complex removal / response consumption)
        ],
        dtype=np.int32,
    )

    initial_state = np.array([100, 50, 0], dtype=np.int32)
    rate_constants = np.array([0.0008, 0.05, 0.01], dtype=np.float64)
    reaction_type_codes = np.full(stoich.shape[0], reactors.ReactionType.MASS_ACTION, dtype=np.int32)
    t_points = np.linspace(0.0, 120.0, 241)

    return PKModel(
        stoich=stoich,
        initial_state=initial_state,
        rate_constants=rate_constants,
        reaction_type_codes=reaction_type_codes,
        t_points=t_points,
        complex_index=2,
    )


MODEL = build_model()
N_TRAJ = 20000
SEED = 321
RESPONSE_THRESHOLD = 25  # half of initial target pool
THERAPEUTIC_WINDOW = (10.0, 60.0)
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def simulate_population() -> np.ndarray:
    ensemble = reactors.simulate_ensemble(
        stoich=MODEL.stoich,
        initial_state=MODEL.initial_state,
        rate_constants=MODEL.rate_constants,
        reaction_type_codes=MODEL.reaction_type_codes,
        t_end=MODEL.t_points[-1],
        n_trajectories=N_TRAJ,
        t_points=MODEL.t_points,
        seed=SEED,
    )
    return np.asarray(ensemble)


def first_passage_time(traj: np.ndarray, threshold: int) -> float:
    values = traj[:, MODEL.complex_index]
    indices = np.flatnonzero(values >= threshold)
    if indices.size == 0:
        return np.inf
    return MODEL.t_points[indices[0]]


def summarize_response_times(times: np.ndarray) -> None:
    finite = times[np.isfinite(times)]
    window_mask = (times > THERAPEUTIC_WINDOW[0]) & (times < THERAPEUTIC_WINDOW[1])
    efficacy = np.mean(window_mask)

    print(f"Simulated trajectories: {N_TRAJ}")
    print(f"Reached threshold: {finite.size} ({finite.size / N_TRAJ:0.2%})")
    if finite.size:
        print(f"Median response time: {np.median(finite):0.2f}")
        print(f"Therapeutic window ({THERAPEUTIC_WINDOW[0]}, {THERAPEUTIC_WINDOW[1]}) efficacy: {efficacy:0.2%}")
    else:
        print("No cells reached the response threshold.")


def plot_response_distribution(times: np.ndarray) -> None:
    finite = times[np.isfinite(times)]
    fig, ax = plt.subplots(figsize=(7, 4))
    if finite.size:
        ax.hist(finite, bins=50, color="#ff7f00", alpha=0.8, density=True)
    ax.axvspan(*THERAPEUTIC_WINDOW, color="#33a02c", alpha=0.2, label="therapeutic window")
    ax.set_xlabel("time to 50% target occupancy")
    ax.set_ylabel("density")
    ax.set_title("Response-time distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pk_response_times.png", dpi=200)
    plt.close(fig)


def plot_sample_trajectories(trajectories: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    idxs = rng.choice(trajectories.shape[0], size=5, replace=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    for idx in idxs:
        ax.plot(MODEL.t_points, trajectories[idx, :, MODEL.complex_index], alpha=0.6)
    ax.axhline(RESPONSE_THRESHOLD, color="#6a3d9a", linestyle="--", label="threshold")
    ax.set_xlabel("time")
    ax.set_ylabel("complex copies")
    ax.set_title("Sample complex-occupancy traces")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pk_sample_traces.png", dpi=200)
    plt.close(fig)


def run_example() -> None:
    trajectories = simulate_population()
    response_times = np.array([first_passage_time(traj, RESPONSE_THRESHOLD) for traj in trajectories])
    summarize_response_times(response_times)
    plot_response_distribution(response_times)
    plot_sample_trajectories(trajectories)
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_example()
