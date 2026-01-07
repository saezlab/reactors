#!/usr/bin/env python3
"""Minimal RNA splicing dynamics: transcription -> pre-mRNA -> mature RNA."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import reactors

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_example() -> None:
    # Species: [pre-mRNA, mature mRNA]
    stoich = np.array(
        [
            [1, 0],   # transcription: Ø -> pre-mRNA
            [-1, 1],  # splicing: pre-mRNA -> mature
            [0, -1],  # degradation of mature mRNA
        ],
        dtype=np.int32,
    )
    initial_state = np.array([0, 0], dtype=np.int32)
    rate_constants = np.array([25.0, 2.0, 0.7])
    reaction_types = np.full(3, reactors.ReactionType.MASS_ACTION, dtype=np.int32)
    t_points = np.linspace(0.0, 40.0, 300)

    trajectories = reactors.simulate_ensemble(
        stoich=stoich,
        initial_state=initial_state,
        rate_constants=rate_constants,
        reaction_type_codes=reaction_types,
        t_end=40.0,
        n_trajectories=400,
        t_points=t_points,
        seed=314,
    )

    mean_traj = trajectories.mean(axis=0)
    p10, p90 = np.percentile(trajectories, [10, 90], axis=0)

    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.3, 1.0], hspace=0.45, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    ax.fill_between(t_points, p10[:, 0], p90[:, 0], alpha=0.3, color="#fb9a99")
    ax.fill_between(t_points, p10[:, 1], p90[:, 1], alpha=0.3, color="#a6cee3")
    ax.plot(t_points, mean_traj[:, 0], color="#e31a1c", label="pre-mRNA (mean)")
    ax.plot(t_points, mean_traj[:, 1], color="#1f78b4", label="mature mRNA (mean)")
    for idx in range(5):
        ax.plot(t_points, trajectories[idx, :, 0], color="#e31a1c", alpha=0.25, linewidth=1.0)
        ax.plot(t_points, trajectories[idx, :, 1], color="#1f78b4", alpha=0.25, linewidth=1.0)
    ax.set_xlabel("time")
    ax.set_ylabel("copy number")
    ax.set_title("RNA splicing dynamics (SSA ensemble)")
    ax.legend()
    final_states = reactors.simulate_ensemble(
        stoich=stoich,
        initial_state=initial_state,
        rate_constants=rate_constants,
        reaction_type_codes=reaction_types,
        t_end=40.0,
        n_trajectories=2000,
        mode="final",
        seed=99,
    )

    ax_hist_pre = fig.add_subplot(gs[1, 0])
    ax_hist_mat = fig.add_subplot(gs[1, 1], sharey=ax_hist_pre)
    ax_hist_pre.hist(final_states[:, 0], bins=40, color="#e31a1c", alpha=0.85)
    ax_hist_pre.set_title("Pre-mRNA final counts")
    ax_hist_pre.set_xlabel("copies")
    ax_hist_pre.set_ylabel("probability")

    ax_hist_mat.hist(final_states[:, 1], bins=40, color="#1f78b4", alpha=0.85)
    ax_hist_mat.set_title("Mature mRNA final counts")
    ax_hist_mat.set_xlabel("copies")
    plt.setp(ax_hist_mat.get_yticklabels(), visible=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rna_splicing_overview.png", dpi=200)
    plt.close(fig)

    print("Saved RNA splicing plots to", OUTPUT_DIR)


if __name__ == "__main__":
    run_example()
