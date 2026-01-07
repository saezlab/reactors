#!/usr/bin/env python3
"""Approximate Catalyst's Michaelis–Menten ODE example via Reactors ensemble means."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import reactors

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_example() -> None:
    # Species: S (substrate), E (enzyme), SE (complex), P (product)
    stoich = np.array(
        [
            [-1, -1, 1, 0],  # S + E -> SE (binding)
            [1, 1, -1, 0],   # SE -> S + E (dissociation)
            [0, 1, -1, 1],   # SE -> P + E (product formation)
        ],
        dtype=np.int32,
    )
    initial_state = np.array([50, 10, 0, 0], dtype=np.int32)
    rate_constants = np.array([0.01, 0.1, 0.1], dtype=np.float64)  # kB, kD, kP
    reaction_type_codes = np.full(3, reactors.ReactionType.MASS_ACTION, dtype=np.int32)

    t_end = 200.0
    t_points = np.linspace(0.0, t_end, 801)
    n_trajectories = 4096

    ensemble = reactors.simulate_ensemble(
        stoich=stoich,
        initial_state=initial_state,
        rate_constants=rate_constants,
        reaction_type_codes=reaction_type_codes,
        t_end=t_end,
        n_trajectories=n_trajectories,
        t_points=t_points,
        seed=2025,
    )

    mean_traces = ensemble.mean(axis=0)
    lower, upper = np.percentile(ensemble, [5, 95], axis=0)
    species_labels = ["Substrate (S)", "Enzyme (E)", "Complex (SE)", "Product (P)"]
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, label in enumerate(species_labels):
        ax.fill_between(
            t_points,
            lower[:, idx],
            upper[:, idx],
            color=colors[idx],
            alpha=0.18,
            linewidth=0.0,
        )
        ax.plot(
            t_points,
            mean_traces[:, idx],
            color=colors[idx],
            linewidth=2.5,
            label=label,
        )
    ax.set_xlabel("time")
    ax.set_ylabel("copy number")
    ax.set_title("Michaelis–Menten ensemble mean ± 90% bands (Reactors)")
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    output_path = OUTPUT_DIR / "michaelis_menten_deterministic.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved Michaelis–Menten deterministic approximation to {output_path}")


if __name__ == "__main__":
    run_example()
