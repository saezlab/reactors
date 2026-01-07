#!/usr/bin/env python3
"""Parameter inference demo using Reactors within a simple Metropolis–Hastings loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

import reactors


@dataclass(frozen=True)
class TwoStatePromoterModel:
    stoich: np.ndarray
    initial_state: np.ndarray
    reaction_type_codes: np.ndarray
    reaction_type_params: np.ndarray
    mrna_index: int = 2


def build_model() -> TwoStatePromoterModel:
    # Species: promoter_off, promoter_on, mRNA
    stoich = np.array(
        [
            [-1, 1, 0],   # promoter_off -> promoter_on
            [1, -1, 0],   # promoter_on -> promoter_off
            [0, 0, 1],    # regulated transcription (Hill on promoter_on)
            [0, 0, -1],   # mRNA degradation
        ],
        dtype=np.int32,
    )

    initial_state = np.array([1, 0, 0], dtype=np.int32)  # start OFF
    reaction_type_codes = np.array(
        [
            reactors.ReactionType.MASS_ACTION,
            reactors.ReactionType.MASS_ACTION,
            reactors.ReactionType.HILL,
            reactors.ReactionType.MASS_ACTION,
        ],
        dtype=np.int32,
    )

    # Hill params rows: [activator_index, hill_n, K]
    reaction_type_params = np.array(
        [
            [0.0, 0.0, 0.0],  # unused mass-action
            [0.0, 0.0, 0.0],  # unused mass-action
            [1.0, 1.0, 1e-3],  # promoter_on activates transcription
            [0.0, 0.0, 0.0],  # unused mass-action
        ],
        dtype=np.float64,
    )

    return TwoStatePromoterModel(
        stoich=stoich,
        initial_state=initial_state,
        reaction_type_codes=reaction_type_codes,
        reaction_type_params=reaction_type_params,
        mrna_index=2,
    )


MODEL = build_model()
T_END = 50.0
TRUE_PARAMS = np.array([0.4, 0.25, 18.0, 0.45], dtype=np.float64)  # k_on, k_off, k_tx, k_deg
EXPERIMENT_TRAJ = 4000
EXPERIMENT_SEED = 2024
LIKELIHOOD_TRAJ = 512
LIKELIHOOD_SEED = 11
LIKELIHOOD_SCALE = 10.0
STEP_SCALES = np.array([0.25, 0.25, 0.15, 0.15])  # log-space std dev per parameter
N_SAMPLES = 450
BURN_IN = 150
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def simulate_mrna_counts(
    params: np.ndarray,
    *,
    n_trajectories: int,
    seed: int,
) -> np.ndarray:
    rates = np.asarray(params, dtype=np.float64)
    result = reactors.simulate_ensemble(
        stoich=MODEL.stoich,
        initial_state=MODEL.initial_state,
        rate_constants=rates,
        reaction_type_codes=MODEL.reaction_type_codes,
        reaction_type_params=MODEL.reaction_type_params,
        t_end=T_END,
        n_trajectories=n_trajectories,
        mode="final",
        seed=seed,
    )
    return np.asarray(result)[:, MODEL.mrna_index]


EXPERIMENTAL_COUNTS = simulate_mrna_counts(
    TRUE_PARAMS,
    n_trajectories=EXPERIMENT_TRAJ,
    seed=EXPERIMENT_SEED,
)

LOG_PRIOR_MEAN = np.log(np.array([0.3, 0.3, 15.0, 0.4]))
LOG_PRIOR_STD = np.array([0.6, 0.6, 0.4, 0.4])


def log_prior(params: np.ndarray) -> float:
    if np.any(params <= 0.0):
        return -np.inf
    log_params = np.log(params)
    diff = (log_params - LOG_PRIOR_MEAN) / LOG_PRIOR_STD
    return -0.5 * np.dot(diff, diff)


def log_likelihood(params: np.ndarray) -> float:
    simulated = simulate_mrna_counts(
        params,
        n_trajectories=LIKELIHOOD_TRAJ,
        seed=LIKELIHOOD_SEED,
    )
    distance = wasserstein_distance(simulated, EXPERIMENTAL_COUNTS)
    return -distance / LIKELIHOOD_SCALE


def log_posterior(params: np.ndarray) -> float:
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params)


def metropolis_hastings(
    initial_params: np.ndarray,
    n_samples: int,
    step_scales: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_samples, initial_params.size), dtype=np.float64)
    log_posts = np.zeros(n_samples, dtype=np.float64)

    current = initial_params.copy()
    current_logp = log_posterior(current)
    accepted = 0

    for idx in range(n_samples):
        proposal = np.exp(np.log(current) + rng.normal(scale=step_scales))
        proposal_logp = log_posterior(proposal)
        log_alpha = proposal_logp - current_logp
        if np.log(rng.random()) < log_alpha:
            current = proposal
            current_logp = proposal_logp
            accepted += 1
        samples[idx] = current
        log_posts[idx] = current_logp

    acceptance_ratio = accepted / n_samples
    return samples, log_posts, acceptance_ratio


def save_trace_plot(samples: np.ndarray) -> None:
    labels = ["k_on", "k_off", "k_tx", "k_deg"]
    fig, axes = plt.subplots(len(labels), 1, figsize=(7, 8), sharex=True)
    for idx, ax in enumerate(axes):
        ax.plot(samples[:, idx], color="#1f78b4", linewidth=1.0)
        ax.axhline(TRUE_PARAMS[idx], color="#33a02c", linestyle="--", linewidth=1.2)
        ax.set_ylabel(labels[idx])
    axes[-1].set_xlabel("iteration")
    fig.suptitle("Metropolis–Hastings traces")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "parameter_traces.png", dpi=200)
    plt.close(fig)


def save_distribution_plot(best_params: np.ndarray) -> None:
    best_counts = simulate_mrna_counts(
        best_params,
        n_trajectories=EXPERIMENT_TRAJ,
        seed=123,
    )
    max_count = int(max(EXPERIMENTAL_COUNTS.max(), best_counts.max()) + 1)
    bins = np.arange(0, max_count + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        EXPERIMENTAL_COUNTS,
        bins=bins,
        density=True,
        alpha=0.5,
        label="experimental",
        color="#6a3d9a",
    )
    ax.hist(
        best_counts,
        bins=bins,
        density=True,
        alpha=0.5,
        label="simulated (MAP)",
        color="#ff7f00",
    )
    ax.set_xlabel("mRNA copies")
    ax.set_ylabel("probability density")
    ax.set_title("Distribution match at inferred parameters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "distribution_fit.png", dpi=200)
    plt.close(fig)


def run_example() -> None:
    initial_guess = np.array([0.3, 0.3, 12.0, 0.3])
    samples, log_posts, acc = metropolis_hastings(
        initial_params=initial_guess,
        n_samples=N_SAMPLES,
        step_scales=STEP_SCALES,
        seed=7,
    )

    posterior_samples = samples[BURN_IN:]
    posterior_mean = posterior_samples.mean(axis=0)
    best_idx = np.argmax(log_posts[BURN_IN:]) + BURN_IN
    best_params = samples[best_idx]

    save_trace_plot(samples)
    save_distribution_plot(best_params)

    labels = ["k_on", "k_off", "k_tx", "k_deg"]
    print("True parameters:")
    for label, value in zip(labels, TRUE_PARAMS):
        print(f"  {label:>6}: {value:0.3f}")
    print("Posterior mean (after burn-in):")
    for label, value in zip(labels, posterior_mean):
        print(f"  {label:>6}: {value:0.3f}")
    print("MAP estimate:")
    for label, value in zip(labels, best_params):
        print(f"  {label:>6}: {value:0.3f}")
    print(f"Acceptance ratio: {acc:0.2f}")
    print(f"Saved plots under {OUTPUT_DIR}")


if __name__ == "__main__":
    run_example()
