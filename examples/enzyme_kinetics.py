#!/usr/bin/env python3
"""Enzyme kinetics SSA example inspired by Catalyst.jl documentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import reactors

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_example() -> None:
    # Species ordering: S (substrate), E (enzyme), SE (bound complex), P (product)
    stoich = np.array(
        [
            [-1, -1, 1, 0],  # S + E -> SE (binding)
            [1, 1, -1, 0],   # SE -> S + E (unbinding)
            [0, 1, -1, 1],   # SE -> P + E (catalysis)
        ],
        dtype=np.int32,
    )
    initial_state = np.array([50, 10, 0, 0], dtype=np.int32)
    rate_constants = np.array([0.01, 0.05, 0.1])  # k_on, k_off, k_cat
    reaction_type_codes = np.full(3, reactors.ReactionType.MASS_ACTION, dtype=np.int32)
    t_points = np.linspace(0.0, 20.0, 400)

    ensemble = reactors.simulate_ensemble(
        stoich=stoich,
        initial_state=initial_state,
        rate_constants=rate_constants,
        reaction_type_codes=reaction_type_codes,
        t_end=20.0,
        n_trajectories=256,
        t_points=t_points,
        seed=2025,
    )

    mean_trace = ensemble.mean(axis=0)
    sample_indices = np.linspace(0, ensemble.shape[0] - 1, 6, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    species_labels = ["Substrate (S)", "Enzyme (E)", "Complex (SE)", "Product (P)"]
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    for idx, ax in enumerate(axes.flat):
        ax.plot(t_points, mean_trace[:, idx], color=colors[idx], linewidth=2.0, label="Mean")
        for traj_idx in sample_indices:
            ax.plot(
                t_points,
                ensemble[traj_idx, :, idx],
                color=colors[idx],
                alpha=0.25,
                linewidth=0.8,
            )
        ax.set_ylabel("copies")
        ax.set_title(species_labels[idx])
        ax.legend(loc="upper right")
    axes[1, 0].set_xlabel("time")
    axes[1, 1].set_xlabel("time")
    fig.tight_layout()
    out_path = OUTPUT_DIR / "enzyme_kinetics_timeseries.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved enzyme kinetics SSA plot to", out_path)


if __name__ == "__main__":
    run_example()
