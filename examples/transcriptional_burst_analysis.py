#!/usr/bin/env python3
"""Analyze transcriptional burst statistics from Reactors SSA trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

import reactors


@dataclass(frozen=True)
class BurstSetup:
    stoich: np.ndarray
    initial_state: np.ndarray
    rate_constants: np.ndarray
    reaction_type_codes: np.ndarray
    reaction_type_params: np.ndarray
    t_points: np.ndarray


def build_setup() -> BurstSetup:
    # Species ordering: promoter_off, promoter_on, mRNA
    stoich = np.array(
        [
            [-1, 1, 0],   # promoter_off -> promoter_on
            [1, -1, 0],   # promoter_on -> promoter_off
            [0, 0, 1],    # promoter_on-driven transcription (Hill activation)
            [0, 0, -1],   # mRNA degradation
        ],
        dtype=np.int32,
    )

    initial_state = np.array([1, 0, 0], dtype=np.int32)
    rate_constants = np.array([0.2, 0.1, 25.0, 0.5], dtype=np.float64)
    reaction_type_codes = np.array(
        [
            reactors.ReactionType.MASS_ACTION,
            reactors.ReactionType.MASS_ACTION,
            reactors.ReactionType.HILL,
            reactors.ReactionType.MASS_ACTION,
        ],
        dtype=np.int32,
    )

    # Hill rows: [activator_index, hill_n, K]
    reaction_type_params = np.array(
        [
            [0.0, 0.0, 0.0],  # unused mass-action
            [0.0, 0.0, 0.0],  # unused mass-action
            [1.0, 1.0, 0.5],  # promoter_on activates transcription (n=1)
            [0.0, 0.0, 0.0],  # unused mass-action
        ],
        dtype=np.float64,
    )

    t_points = np.arange(0.0, 500.1, 0.5)

    return BurstSetup(
        stoich=stoich,
        initial_state=initial_state,
        rate_constants=rate_constants,
        reaction_type_codes=reaction_type_codes,
        reaction_type_params=reaction_type_params,
        t_points=t_points,
    )


SETUP = build_setup()
N_TRAJECTORIES = 2048
SEED = 1234
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def simulate_timeseries() -> np.ndarray:
    trajectories = reactors.simulate_ensemble(
        stoich=SETUP.stoich,
        initial_state=SETUP.initial_state,
        rate_constants=SETUP.rate_constants,
        reaction_type_codes=SETUP.reaction_type_codes,
        reaction_type_params=SETUP.reaction_type_params,
        t_end=SETUP.t_points[-1],
        n_trajectories=N_TRAJECTORIES,
        t_points=SETUP.t_points,
        seed=SEED,
    )
    return np.asarray(trajectories)


def detect_bursts(
    promoter_on: Sequence[int],
    mrna_counts: Sequence[int],
    times: Sequence[float],
) -> List[Tuple[float, float]]:
    bursts: List[Tuple[float, float]] = []
    in_burst = False
    start_idx = 0
    start_mrna = 0

    for idx, state in enumerate(promoter_on):
        active = state > 0
        if active and not in_burst:
            in_burst = True
            start_idx = idx
            start_mrna = mrna_counts[idx]
        elif not active and in_burst:
            delta = mrna_counts[idx] - start_mrna
            duration = times[idx] - times[start_idx]
            bursts.append((max(delta, 0), max(duration, 0.0)))
            in_burst = False

    return bursts


def analyze_bursts(trajectories: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sizes: List[float] = []
    durations: List[float] = []
    freqs = np.zeros(trajectories.shape[0], dtype=np.float64)

    for idx in range(trajectories.shape[0]):
        promoter = trajectories[idx, :, 1]
        mrna = trajectories[idx, :, 2]
        bursts = detect_bursts(promoter, mrna, SETUP.t_points)
        if bursts:
            burst_sizes, burst_durations = zip(*bursts)
            sizes.extend(burst_sizes)
            durations.extend(burst_durations)
        freqs[idx] = len(bursts) / SETUP.t_points[-1]

    return np.array(sizes), np.array(durations), freqs


def plot_histogram(data: np.ndarray, xlabel: str, title: str, filename: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(data, bins=50, color="#1f78b4", alpha=0.75, density=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def plot_frequency_distribution(freqs: np.ndarray, filename: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(freqs, bins=40, color="#33a02c", alpha=0.8)
    ax.set_xlabel("bursts per unit time")
    ax.set_ylabel("trajectory count")
    ax.set_title("Burst frequency across trajectories")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def plot_sample_traces(trajectories: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    sel = np.random.default_rng(0).choice(trajectories.shape[0], size=5, replace=False)
    for idx in sel:
        axes[0].plot(SETUP.t_points, trajectories[idx, :, 1], alpha=0.7)
        axes[1].plot(SETUP.t_points, trajectories[idx, :, 2], alpha=0.7)
    axes[0].set_ylabel("promoter_on copies")
    axes[1].set_ylabel("mRNA copies")
    axes[1].set_xlabel("time")
    axes[0].set_title("Sample trajectories (promoter state)")
    axes[1].set_title("Sample trajectories (mRNA counts)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "burst_sample_traces.png", dpi=200)
    plt.close(fig)


def run_example() -> None:
    trajectories = simulate_timeseries()
    plot_sample_traces(trajectories)
    burst_sizes, burst_durations, burst_freqs = analyze_bursts(trajectories)

    plot_histogram(
        burst_sizes,
        xlabel="mRNAs produced",
        title="Burst size distribution",
        filename=OUTPUT_DIR / "burst_size_distribution.png",
    )
    plot_histogram(
        burst_durations,
        xlabel="duration (time units)",
        title="Burst duration distribution",
        filename=OUTPUT_DIR / "burst_duration_distribution.png",
    )
    plot_frequency_distribution(burst_freqs, OUTPUT_DIR / "burst_frequency_histogram.png")

    mean_size = burst_sizes.mean() if burst_sizes.size else 0.0
    mean_duration = burst_durations.mean() if burst_durations.size else 0.0
    mean_freq = burst_freqs.mean()
    print("Collected trajectories:", trajectories.shape[0])
    print(f"Mean burst size: {mean_size:0.2f} mRNAs")
    print(f"Mean burst duration: {mean_duration:0.2f} time units")
    print(f"Mean burst frequency: {mean_freq:0.3f} bursts/unit time")
    print(f"Saved plots under {OUTPUT_DIR}")


if __name__ == "__main__":
    run_example()
