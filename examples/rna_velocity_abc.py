#!/usr/bin/env python3
"""Approximate Bayesian computation demo for an RNA-velocity-style network."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import qmc

import reactors


@dataclass(frozen=True)
class VelocityNetwork:
    stoich: np.ndarray
    initial_state: np.ndarray
    reaction_type_codes: np.ndarray
    reaction_type_params: np.ndarray


def build_velocity_network() -> VelocityNetwork:
    # Species: signal, U1, S1, U2, S2
    stoich = np.array(
        [
            [-1, 0, 0, 0, 0],   # signal decay
            [0, 1, 0, 0, 0],    # transcription gene 1 (Hill on signal)
            [0, -1, 1, 0, 0],   # splicing gene 1
            [0, 0, -1, 0, 0],   # mature RNA 1 decay
            [0, -1, 0, 0, 0],   # unspliced 1 decay (slow leak)
            [0, 0, 0, 1, 0],    # transcription gene 2 (Hill on S1)
            [0, 0, 0, -1, 1],   # splicing gene 2
            [0, 0, 0, 0, -1],   # mature RNA 2 decay
            [0, 0, 0, -1, 0],   # unspliced 2 decay (slow leak)
        ],
        dtype=np.int32,
    )

    reaction_type_codes = np.array(
        [
            reactors.ReactionType.MASS_ACTION,
            reactors.ReactionType.HILL,
            reactors.ReactionType.MASS_ACTION,
            reactors.ReactionType.MASS_ACTION,
            reactors.ReactionType.MASS_ACTION,
            reactors.ReactionType.HILL,
            reactors.ReactionType.MASS_ACTION,
            reactors.ReactionType.MASS_ACTION,
            reactors.ReactionType.MASS_ACTION,
        ],
        dtype=np.int32,
    )

    reaction_type_params = np.array(
        [
            [0.0, 0.0, 0.0],  # signal decay
            [0.0, 2.0, 4.0],  # signal activates gene 1 transcription
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 1.8, 5.0],  # mature gene 1 activates gene 2 transcription
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    initial_state = np.array([12, 0, 0, 0, 0], dtype=np.int32)
    return VelocityNetwork(stoich, initial_state, reaction_type_codes, reaction_type_params)


MODEL = build_velocity_network()
PARAM_NAMES = ["k_tx1", "k_splice1", "k_deg1", "k_tx2", "k_splice2", "k_deg2"]
TRUE_PARAMS = np.array([1.4, 0.65, 0.25, 0.85, 0.5, 0.18])
PRIOR_RANGES = np.array(
    [
        [0.4, 2.5],   # k_tx1
        [0.25, 1.2],  # k_splice1
        [0.1, 0.5],   # k_deg1
        [0.3, 1.4],   # k_tx2
        [0.2, 0.9],   # k_splice2
        [0.08, 0.35], # k_deg2
    ],
    dtype=np.float64,
)
SIGNAL_DECAY = 0.12
UNSPLICED_DECAY = np.array([0.05, 0.04], dtype=np.float64)
T_END = 40.0
T_POINTS = np.linspace(0.0, T_END, 161)
OBS_CELLS = 1800
SIM_CELLS_PER_PROPOSAL = 800
N_BINS = 6
ABC_PROPOSALS = 900
ABC_ACCEPT = 150
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def assemble_rates(params: np.ndarray) -> np.ndarray:
    rates = np.array(
        [
            SIGNAL_DECAY,
            params[0],
            params[1],
            params[2],
            UNSPLICED_DECAY[0],
            params[3],
            params[4],
            params[5],
            UNSPLICED_DECAY[1],
        ],
        dtype=np.float64,
    )
    return rates


def simulate_population(
    params: np.ndarray,
    *,
    n_cells: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate n_cells trajectories and draw a random capture time per cell."""
    rng = np.random.default_rng(seed)
    ensemble = reactors.simulate_ensemble(
        stoich=MODEL.stoich,
        initial_state=MODEL.initial_state,
        rate_constants=assemble_rates(params),
        reaction_type_codes=MODEL.reaction_type_codes,
        reaction_type_params=MODEL.reaction_type_params,
        t_end=T_END,
        n_trajectories=n_cells,
        t_points=T_POINTS,
        seed=seed,
    )
    ensemble = np.asarray(ensemble, dtype=np.int32)
    time_indices = rng.integers(0, T_POINTS.size, size=n_cells)
    cells = ensemble[np.arange(n_cells), time_indices][:, 1:]
    capture_times = T_POINTS[time_indices]
    return cells.astype(np.float64), capture_times


def compute_summary_stats(counts: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Time-binned means plus covariance summaries for RNA velocity comparisons."""
    bins = np.linspace(0.0, T_END, N_BINS + 1)
    summary = []
    for idx in range(N_BINS):
        mask = (times >= bins[idx]) & (times < bins[idx + 1])
        if mask.sum() < 5:
            subset = counts
        else:
            subset = counts[mask]
        summary.extend(
            [
                subset[:, 0].mean(),
                subset[:, 1].mean(),
                subset[:, 2].mean(),
                subset[:, 3].mean(),
            ]
        )

    cov_u1_s1 = np.cov(counts[:, 0], counts[:, 1], ddof=0)[0, 1]
    cov_u2_s2 = np.cov(counts[:, 2], counts[:, 3], ddof=0)[0, 1]
    cov_s1_u2 = np.cov(counts[:, 1], counts[:, 2], ddof=0)[0, 1]
    summary.extend([cov_u1_s1, cov_u2_s2, cov_s1_u2])
    return np.asarray(summary, dtype=np.float64)


def compute_spliced_velocities(counts: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Per-cell dS/dt from splicing and decay kinetics."""
    v1 = params[1] * counts[:, 0] - params[2] * counts[:, 1]
    v2 = params[4] * counts[:, 2] - params[5] * counts[:, 3]
    return np.stack([v1, v2], axis=1)


def standardized_distance(summary: np.ndarray, reference: np.ndarray, scale: np.ndarray) -> float:
    diff = (summary - reference) / scale
    return float(np.sqrt(np.dot(diff, diff)))


def sample_prior(rng: np.random.Generator) -> np.ndarray:
    unit = rng.random(len(PARAM_NAMES))
    low = np.log(PRIOR_RANGES[:, 0])
    high = np.log(PRIOR_RANGES[:, 1])
    return np.exp(low + unit * (high - low))


def run_abc(observed_summary: np.ndarray, scale: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=rng.integers(1_000_000))
    unit_samples = sampler.random(ABC_PROPOSALS)
    low = np.log(PRIOR_RANGES[:, 0])
    high = np.log(PRIOR_RANGES[:, 1])
    log_params = low + unit_samples * (high - low)
    proposals = np.exp(log_params)

    distances = []
    summaries = []
    accepted_params = []

    for idx, params in enumerate(proposals):
        cells, times = simulate_population(
            params,
            n_cells=SIM_CELLS_PER_PROPOSAL,
            seed=seed + idx * 13 + 5,
        )
        sim_summary = compute_summary_stats(cells, times)
        dist = standardized_distance(sim_summary, observed_summary, scale)
        distances.append(dist)
        summaries.append(sim_summary)
        accepted_params.append(params)

    distances = np.asarray(distances)
    summaries = np.asarray(summaries)
    accepted_params = np.asarray(accepted_params)
    order = np.argsort(distances)
    keep = order[:ABC_ACCEPT]
    return accepted_params[keep], summaries[keep], distances[keep]


def make_figure(
    obs_counts: np.ndarray,
    obs_times: np.ndarray,
    obs_summary: np.ndarray,
    best_counts: np.ndarray,
    best_times: np.ndarray,
    best_summary: np.ndarray,
    accepted_params: np.ndarray,
    true_velocities: np.ndarray,
    pred_velocities: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1], wspace=0.35, hspace=0.4)

    ax_g1 = fig.add_subplot(gs[0, 0])
    sample_rng = np.random.default_rng(42)
    sample_size = min(350, len(obs_counts))
    sample_idx = sample_rng.choice(len(obs_counts), size=sample_size, replace=False)
    ax_g1.scatter(obs_times[sample_idx], obs_counts[sample_idx, 0], s=10, alpha=0.4, label="U1 observed", color="#fb9a99")
    ax_g1.scatter(obs_times[sample_idx], obs_counts[sample_idx, 1], s=10, alpha=0.4, label="S1 observed", color="#1f78b4")
    ax_g1.scatter(best_times[sample_idx], best_counts[sample_idx, 0], s=10, alpha=0.4, label="U1 simulated", color="#e31a1c")
    ax_g1.scatter(best_times[sample_idx], best_counts[sample_idx, 1], s=10, alpha=0.4, label="S1 simulated", color="#03396c")
    ax_g1.set_xlabel("capture time")
    ax_g1.set_ylabel("gene 1 copies")
    ax_g1.set_title("Gene 1 snapshot counts")
    ax_g1.legend(fontsize=8, loc="upper right")

    ax_g2 = fig.add_subplot(gs[0, 1])
    ax_g2.scatter(obs_times[sample_idx], obs_counts[sample_idx, 2], s=10, alpha=0.4, label="U2 observed", color="#fdbf6f")
    ax_g2.scatter(obs_times[sample_idx], obs_counts[sample_idx, 3], s=10, alpha=0.4, label="S2 observed", color="#b2df8a")
    ax_g2.scatter(best_times[sample_idx], best_counts[sample_idx, 2], s=10, alpha=0.4, label="U2 simulated", color="#ff7f00")
    ax_g2.scatter(best_times[sample_idx], best_counts[sample_idx, 3], s=10, alpha=0.4, label="S2 simulated", color="#33a02c")
    ax_g2.set_xlabel("capture time")
    ax_g2.set_ylabel("gene 2 copies")
    ax_g2.set_title("Gene 2 snapshot counts")
    ax_g2.legend(fontsize=8, loc="upper right")

    ax_field = fig.add_subplot(gs[0, 2])
    ax_field.scatter(obs_counts[:, 1], obs_counts[:, 3], s=8, color="#dcdcdc", alpha=0.35, linewidths=0)
    arrow_subset = sample_rng.choice(len(obs_counts), size=min(160, len(obs_counts)), replace=False)
    s1_coords = obs_counts[arrow_subset, 1]
    s2_coords = obs_counts[arrow_subset, 3]
    true_vecs = true_velocities[arrow_subset]
    pred_vecs = pred_velocities[arrow_subset]
    vec_stack = np.vstack([true_vecs, pred_vecs])
    norm_scale = np.percentile(np.linalg.norm(vec_stack, axis=1), 80)
    norm_scale = max(norm_scale, 0.4)
    display_scale = 1.05
    scaled_true = display_scale * true_vecs / norm_scale
    scaled_pred = display_scale * pred_vecs / norm_scale
    ax_field.quiver(
        s1_coords,
        s2_coords,
        scaled_true[:, 0],
        scaled_true[:, 1],
        color="#5f6368",
        alpha=0.6,
        scale_units="xy",
        angles="xy",
        scale=1.0,
        width=0.0028,
        headwidth=3.5,
        headlength=4.5,
        zorder=2,
    )
    ax_field.quiver(
        s1_coords,
        s2_coords,
        scaled_pred[:, 0],
        scaled_pred[:, 1],
        color="#ff7f00",
        alpha=0.9,
        scale_units="xy",
        angles="xy",
        scale=1.0,
        width=0.0028,
        headwidth=3.5,
        headlength=4.5,
        zorder=3,
    )
    ax_field.set_xlabel("S1 copies")
    ax_field.set_ylabel("S2 copies")
    ax_field.set_title("RNA velocity field (obs cells)")
    legend_handles = [
        Line2D([0], [0], color="#5f6368", lw=2, label="true dS/dt"),
        Line2D([0], [0], color="#ff7f00", lw=2, label="ABC inferred dS/dt"),
    ]
    ax_field.legend(handles=legend_handles, fontsize=8, loc="upper left")

    ax_summary = fig.add_subplot(gs[1, 0])
    bin_edges = np.linspace(0.0, T_END, N_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    obs_means = obs_summary.reshape(N_BINS, 4)
    best_means = best_summary.reshape(N_BINS, 4)
    ax_summary.plot(bin_centers, obs_means[:, 0], color="#fb9a99", label="obs U1")
    ax_summary.plot(bin_centers, obs_means[:, 1], color="#1f78b4", label="obs S1")
    ax_summary.plot(bin_centers, obs_means[:, 2], color="#fdbf6f", label="obs U2")
    ax_summary.plot(bin_centers, obs_means[:, 3], color="#b2df8a", label="obs S2")
    ax_summary.plot(bin_centers, best_means[:, 0], "--", color="#e31a1c", label="sim U1")
    ax_summary.plot(bin_centers, best_means[:, 1], "--", color="#03396c", label="sim S1")
    ax_summary.plot(bin_centers, best_means[:, 2], "--", color="#ff7f00", label="sim U2")
    ax_summary.plot(bin_centers, best_means[:, 3], "--", color="#33a02c", label="sim S2")
    ax_summary.set_xlabel("time bin center")
    ax_summary.set_ylabel("mean copies")
    ax_summary.set_title("Time-binned means: observed vs ABC best-fit")
    ax_summary.legend(fontsize=8, ncols=2)

    ax_params = fig.add_subplot(gs[1, 1])
    posterior_mean = accepted_params.mean(axis=0)
    posterior_std = accepted_params.std(axis=0)
    positions = np.arange(len(PARAM_NAMES))
    ax_params.errorbar(
        positions - 0.05,
        posterior_mean,
        yerr=posterior_std,
        fmt="o",
        color="#1f78b4",
        label="posterior mean ± std",
    )
    ax_params.scatter(positions + 0.05, TRUE_PARAMS, color="#33a02c", label="ground truth", zorder=3)
    ax_params.set_xticks(positions)
    ax_params.set_xticklabels(PARAM_NAMES, rotation=45, ha="right")
    ax_params.set_ylabel("rate constant")
    ax_params.set_title("Accepted ABC samples ({} kept)".format(len(accepted_params)))
    ax_params.legend(fontsize=9)

    ax_vel_compare = fig.add_subplot(gs[1, 2])
    ax_vel_compare.scatter(
        true_velocities[:, 0],
        pred_velocities[:, 0],
        color="#1f78b4",
        alpha=0.4,
        s=18,
        label="gene 1",
    )
    ax_vel_compare.scatter(
        true_velocities[:, 1],
        pred_velocities[:, 1],
        color="#33a02c",
        alpha=0.4,
        s=18,
        label="gene 2",
    )
    vel_min = min(true_velocities.min(), pred_velocities.min())
    vel_max = max(true_velocities.max(), pred_velocities.max())
    ax_vel_compare.plot([vel_min, vel_max], [vel_min, vel_max], color="#555555", linestyle="--", linewidth=1.2)
    ax_vel_compare.set_xlabel("true dS/dt")
    ax_vel_compare.set_ylabel("ABC dS/dt")
    ax_vel_compare.set_title("Per-cell velocity agreement")
    ax_vel_compare.legend(fontsize=8, loc="upper left")

    fig.suptitle("Reactors-driven ABC for RNA velocity kinetics", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rna_velocity_abc.png", dpi=220)
    plt.close(fig)


def run_example() -> None:
    obs_counts, obs_times = simulate_population(
        TRUE_PARAMS,
        n_cells=OBS_CELLS,
        seed=2025,
    )
    obs_summary = compute_summary_stats(obs_counts, obs_times)
    scale = np.maximum(np.abs(obs_summary), 1.0)

    accepted_params, accepted_summaries, distances = run_abc(obs_summary, scale, seed=99)
    best_idx = np.argmin(distances)
    best_params = accepted_params[best_idx]
    best_counts, best_times = simulate_population(
        best_params,
        n_cells=OBS_CELLS,
        seed=551,
    )
    best_summary = compute_summary_stats(best_counts, best_times)
    obs_true_vel = compute_spliced_velocities(obs_counts, TRUE_PARAMS)
    obs_pred_vel = compute_spliced_velocities(obs_counts, best_params)

    make_figure(
        obs_counts,
        obs_times,
        obs_summary[: N_BINS * 4],
        best_counts,
        best_times,
        best_summary[: N_BINS * 4],
        accepted_params,
        obs_true_vel,
        obs_pred_vel,
    )

    print("Ground truth parameters:")
    for name, value in zip(PARAM_NAMES, TRUE_PARAMS):
        print(f"  {name:>10s}: {value:0.3f}")
    print("Posterior means (accepted samples):")
    for name, value in zip(PARAM_NAMES, accepted_params.mean(axis=0)):
        print(f"  {name:>10s}: {value:0.3f}")
    print(f"Median ABC distance: {np.median(distances):0.3f}")
    corr_g1 = np.corrcoef(obs_true_vel[:, 0], obs_pred_vel[:, 0])[0, 1]
    corr_g2 = np.corrcoef(obs_true_vel[:, 1], obs_pred_vel[:, 1])[0, 1]
    print(f"Velocity correlation (gene 1): {corr_g1:0.2f}")
    print(f"Velocity correlation (gene 2): {corr_g2:0.2f}")
    print("Saved figure to", OUTPUT_DIR / "rna_velocity_abc.png")


if __name__ == "__main__":
    run_example()
