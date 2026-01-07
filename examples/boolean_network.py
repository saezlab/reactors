#!/usr/bin/env python3
"""Boolean network example encoded with expression propensities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import reactors

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Each node wants to satisfy a Boolean formula built from the other nodes.
# Here we create a cyclic logic loop reminiscent of a repressilator:
# A tries to be the opposite of C, B tries to be the opposite of A, and so on.
SPECIES = ("A", "B", "C")
FORMULAS = (
    "1.0 - s2",  # A turns on when C is off
    "1.0 - s0",  # B turns on when A is off
    "1.0 - s1",  # C turns on when B is off
)
SWITCH_RATE = 35.0  # fast rate so species quickly satisfy their formulas


def build_boolean_model() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return stoichiometry, rate constants, reaction codes, and expressions."""
    n_species = len(SPECIES)
    stoich_rows: list[np.ndarray] = []
    reaction_types: list[int] = []
    expressions: list[str] = []

    for idx, formula in enumerate(FORMULAS):
        # Activation reaction when the target formula evaluates to 1
        row_on = np.zeros(n_species, dtype=np.int32)
        row_on[idx] = 1
        stoich_rows.append(row_on)
        reaction_types.append(reactors.ReactionType.EXPRESSION)
        expressions.append(f"{SWITCH_RATE} * (1.0 - s{idx}) * ({formula})")

        # Deactivation reaction when the formula evaluates to 0
        row_off = np.zeros(n_species, dtype=np.int32)
        row_off[idx] = -1
        stoich_rows.append(row_off)
        reaction_types.append(reactors.ReactionType.EXPRESSION)
        expressions.append(f"{SWITCH_RATE} * s{idx} * (1.0 - ({formula}))")

    stoich = np.stack(stoich_rows)
    # Expression reactions ignore rate_constants, but Reactors still expects the array.
    rate_constants = np.zeros(len(stoich_rows), dtype=np.float64)
    reaction_type_codes = np.array(reaction_types, dtype=np.int32)
    return stoich, rate_constants, reaction_type_codes, expressions


def plot_results(
    ensemble: np.ndarray, final_states: np.ndarray, t_points: np.ndarray, out_path: Path
) -> None:
    """Plot Boolean traces plus the final-state distribution in a single figure."""
    fig = plt.figure(figsize=(8, 7), layout="constrained")
    gs = fig.add_gridspec(len(SPECIES) + 1, 1, height_ratios=[1, 1, 1, 0.9], hspace=0.35)
    axes_ts: list[plt.Axes] = []
    for idx in range(len(SPECIES)):
        share = axes_ts[0] if axes_ts else None
        axes_ts.append(fig.add_subplot(gs[idx, 0], sharex=share))
    ax_hist = fig.add_subplot(gs[-1, 0])

    colors = ("#4c72b0", "#dd8452", "#55a868")
    mean_states = ensemble.mean(axis=0)
    sample_ids = np.arange(min(6, ensemble.shape[0]))
    for species_idx, ax in enumerate(axes_ts):
        for traj_idx in sample_ids:
            ax.step(
                t_points,
                ensemble[traj_idx, :, species_idx],
                where="post",
                color=colors[species_idx],
                alpha=0.45,
                linewidth=1.1,
            )
        ax.plot(
            t_points,
            mean_states[:, species_idx],
            color="black",
            linewidth=2.0,
            label="Mean active probability" if species_idx == 0 else None,
        )
        ax.set_ylabel(SPECIES[species_idx])
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0, 1])
    axes_ts[-1].set_xlabel("time")
    axes_ts[0].legend(loc="upper right")

    counts = np.zeros(2 ** len(SPECIES), dtype=np.int32)
    for state in final_states:
        idx = (state[0] << 2) | (state[1] << 1) | state[2]
        counts[idx] += 1
    probs = counts / counts.sum()
    labels = [format(idx, "03b") for idx in range(len(probs))]
    ax_hist.bar(np.arange(len(probs)), probs, color="#4c72b0")
    ax_hist.set_xticks(np.arange(len(probs)), labels)
    ax_hist.set_xlabel("Final state (ABC)")
    ax_hist.set_ylabel("Probability")
    ax_hist.set_ylim(0.0, probs.max() * 1.25 if probs.max() > 0 else 1.0)
    ax_hist.set_title("Distribution at t_end")

    fig.suptitle("Boolean network encoded as SSA reactions", fontsize=14)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_example() -> None:
    stoich, rate_constants, reaction_types, expressions = build_boolean_model()
    initial_state = np.array([1, 0, 0], dtype=np.int32)
    t_points = np.linspace(0.0, 20.0, 201)

    ensemble = reactors.simulate_ensemble(
        stoich=stoich,
        initial_state=initial_state,
        rate_constants=rate_constants,
        reaction_type_codes=reaction_types,
        reaction_expressions=expressions,
        t_end=t_points[-1],
        n_trajectories=256,
        t_points=t_points,
        seed=2024,
    )

    final_states = reactors.simulate_ensemble(
        stoich=stoich,
        initial_state=initial_state,
        rate_constants=rate_constants,
        reaction_type_codes=reaction_types,
        reaction_expressions=expressions,
        t_end=t_points[-1],
        n_trajectories=4096,
        mode="final",
        seed=99,
    )
    plot_results(
        ensemble,
        final_states,
        t_points,
        OUTPUT_DIR / "boolean_network.png",
    )
    print("Saved Boolean network plot to", OUTPUT_DIR / "boolean_network.png")


if __name__ == "__main__":
    run_example()
