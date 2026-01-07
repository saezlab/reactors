#!/usr/bin/env python3
"""Two-gene toggle-style SSA example with Hill propensities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import reactors

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_example() -> None:
    # Species: A, B
    stoich = np.array(
        [
            [1, 0],   # regulated production of A (Hill)
            [0, 1],   # regulated production of B (Hill)
            [-1, 0],  # degradation of A
            [0, -1],  # degradation of B
            [1, 0],   # basal production of A (mass-action)
            [0, 1],   # basal production of B (mass-action)
        ],
        dtype=np.int32,
    )
    initial_state = np.array([0, 0], dtype=np.int32)
    rate_constants = np.array([40.0, 40.0, 1.2, 1.2, 2.0, 2.0])
    reaction_types = np.array(
        [
            reactors.ReactionType.HILL,
            reactors.ReactionType.HILL,
            reactors.ReactionType.MASS_ACTION,
            reactors.ReactionType.MASS_ACTION,
            reactors.ReactionType.MASS_ACTION,
            reactors.ReactionType.MASS_ACTION,
        ],
        dtype=np.int32,
    )

    # Hill parameters: [activator_index, n, K]
    reaction_type_params = np.array(
        [
            [1, 3.0, 20.0],  # B represses A (Hill)
            [0, 3.0, 20.0],  # A represses B (Hill)
            [0, 0.0, 0.0],   # unused mass-action rows
            [0, 0.0, 0.0],
            [0, 0.0, 0.0],
            [0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    t_points = np.linspace(0.0, 60.0, 400)

    trajectories = reactors.simulate_ensemble(
        stoich=stoich,
        initial_state=initial_state,
        rate_constants=rate_constants,
        reaction_type_codes=reaction_types,
        reaction_type_params=reaction_type_params,
        t_end=60.0,
        n_trajectories=256,
        t_points=t_points,
        seed=7,
    )

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for idx in range(6):
        axes[0].plot(t_points, trajectories[idx, :, 0], alpha=0.6)
        axes[1].plot(t_points, trajectories[idx, :, 1], alpha=0.6)
    axes[0].set_ylabel("A copies")
    axes[1].set_ylabel("B copies")
    axes[1].set_xlabel("time")
    axes[0].set_title("Sample trajectories (toggle-style positive feedback)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "toggle_switch_timeseries.png", dpi=200)
    plt.close(fig)

    final_states = reactors.simulate_ensemble(
        stoich=stoich,
        initial_state=initial_state,
        rate_constants=rate_constants,
        reaction_type_codes=reaction_types,
        reaction_type_params=reaction_type_params,
        t_end=60.0,
        n_trajectories=1000,
        mode="final",
        seed=99,
    )

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(final_states[:, 0], final_states[:, 1], s=10, alpha=0.4)
    ax.set_xlabel("A copies")
    ax.set_ylabel("B copies")
    ax.set_title("Final-state scatter (A vs B)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "toggle_switch_scatter.png", dpi=200)
    plt.close(fig)

    print("Saved plots to", OUTPUT_DIR)


if __name__ == "__main__":
    run_example()
