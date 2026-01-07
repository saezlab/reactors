#!/usr/bin/env python3
"""Birth–death SSA example that saves summary plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import reactors

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_example() -> None:
    stoich = np.array([[1], [-1]], dtype=np.int32)  # Ø→X, X→Ø
    initial_state = np.array([0], dtype=np.int32)
    rate_constants = np.array([5.0, 0.5])
    reaction_types = np.array(
        [reactors.ReactionType.MASS_ACTION, reactors.ReactionType.MASS_ACTION],
        dtype=np.int32,
    )
    t_points = np.linspace(0.0, 25.0, 200)

    trajectories = reactors.simulate_ensemble(
        stoich=stoich,
        initial_state=initial_state,
        rate_constants=rate_constants,
        reaction_type_codes=reaction_types,
        t_end=25.0,
        n_trajectories=512,
        t_points=t_points,
        seed=42,
    ).squeeze(-1)

    mean_trace = trajectories.mean(axis=0)
    p10, p90 = np.percentile(trajectories, [10, 90], axis=0)
    theoretical = (rate_constants[0] / rate_constants[1]) * (1.0 - np.exp(-rate_constants[1] * t_points))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(t_points, p10, p90, color="#a6cee3", alpha=0.6, label="10–90% percentile")
    ax.plot(t_points, mean_trace, label="SSA mean", color="#1f78b4", linewidth=2)
    ax.plot(t_points, theoretical, label="Mean-field", color="#33a02c", linestyle="--")
    for idx in range(5):
        ax.plot(t_points, trajectories[idx], color="#ff7f00", alpha=0.4, linewidth=1.0)
    ax.set_xlabel("time")
    ax.set_ylabel("copy number")
    ax.set_title("Birth–death process ensemble")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "birth_death_timeseries.png", dpi=200)
    plt.close(fig)

    final_states = reactors.simulate_ensemble(
        stoich=stoich,
        initial_state=initial_state,
        rate_constants=rate_constants,
        reaction_type_codes=reaction_types,
        t_end=25.0,
        n_trajectories=2000,
        mode="final",
        seed=123,
    ).squeeze()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(final_states, bins=range(0, final_states.max() + 2), density=True, color="#6a3d9a")
    ax.set_xlabel("final copy number")
    ax.set_ylabel("probability density")
    ax.set_title("Birth–death steady-state distribution")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "birth_death_histogram.png", dpi=200)
    plt.close(fig)

    print("Saved plots to", OUTPUT_DIR)


if __name__ == "__main__":
    run_example()
