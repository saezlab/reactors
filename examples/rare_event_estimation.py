#!/usr/bin/env python3
"""Estimate rare-event probabilities (extinction) via large Reactors ensembles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import reactors


OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_model():
    # Single species with positive feedback production + degradation.
    stoich = np.array(
        [
            [1],   # basal production (mass-action)
            [1],   # self-activation via Hill, drives bistability
            [-1],  # degradation
        ],
        dtype=np.int32,
    )
    initial_state = np.array([60], dtype=np.int32)
    rate_constants = np.array([2.0, 55.0, 0.65], dtype=np.float64)
    reaction_type_codes = np.array(
        [
            reactors.ReactionType.MASS_ACTION,
            reactors.ReactionType.HILL,
            reactors.ReactionType.MASS_ACTION,
        ],
        dtype=np.int32,
    )
    reaction_type_params = np.array(
        [
            [0.0, 0.0, 0.0],  # unused for mass-action
            [0.0, 2.5, 40.0],  # species 0 activates its own production
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return stoich, initial_state, rate_constants, reaction_type_codes, reaction_type_params


def simulate_chunk(n_trajectories: int, seed: int, t_end: float, model_inputs):
    stoich, initial_state, rate_constants, reaction_type_codes, reaction_type_params = model_inputs
    finals = reactors.simulate_ensemble(
        stoich=stoich,
        initial_state=initial_state,
        rate_constants=rate_constants,
        reaction_type_codes=reaction_type_codes,
        reaction_type_params=reaction_type_params,
        t_end=t_end,
        n_trajectories=n_trajectories,
        mode="final",
        seed=seed,
    )
    return np.asarray(finals).squeeze(axis=-1)


def estimate_probability(total_trajectories: int, chunk_size: int, seed: int, t_end: float, threshold: int):
    model_inputs = build_model()
    successes = 0
    recorded = []
    total = 0
    for chunk_idx, start in enumerate(range(0, total_trajectories, chunk_size)):
        batch = min(chunk_size, total_trajectories - start)
        finals = simulate_chunk(batch, seed + chunk_idx, t_end, model_inputs)
        events = finals < threshold
        successes += events.sum()
        recorded.append(finals)
        total += batch
    samples = np.concatenate(recorded)
    p_hat = successes / total
    std_err = np.sqrt(p_hat * (1 - p_hat) / total)
    return p_hat, std_err, samples


def plot_histogram(samples: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(samples, bins=range(0, max(1, samples.max()) + 2), density=True, color="#6a3d9a", alpha=0.8)
    ax.set_xlabel("final copies")
    ax.set_ylabel("probability density")
    ax.set_title("Final-state distribution")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rare_event_histogram.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate extinction probability via Reactors ensembles")
    parser.add_argument("--n-trajectories", type=int, default=50000, help="total trajectories to sample")
    parser.add_argument("--chunk-size", type=int, default=10000, help="trajectories per simulation chunk")
    parser.add_argument("--threshold", type=int, default=5, help="copy-number threshold for extinction")
    parser.add_argument("--t-end", type=float, default=500.0, help="simulation horizon")
    parser.add_argument("--seed", type=int, default=101, help="base RNG seed")
    return parser.parse_args()


def run_example() -> None:
    args = parse_args()
    p_hat, std_err, samples = estimate_probability(
        total_trajectories=args.n_trajectories,
        chunk_size=args.chunk_size,
        seed=args.seed,
        t_end=args.t_end,
        threshold=args.threshold,
    )
    ci_low = max(0.0, p_hat - 1.96 * std_err)
    ci_high = min(1.0, p_hat + 1.96 * std_err)
    plot_histogram(samples)
    print(f"Simulated trajectories: {args.n_trajectories}")
    print(f"Extinction threshold: < {args.threshold} copies")
    print(f"Estimated P(extinction): {p_hat:0.5f}")
    print(f"95% CI: ({ci_low:0.5f}, {ci_high:0.5f})")
    print(f"Standard error: {std_err:0.5f}")
    print(f"Saved histogram to {OUTPUT_DIR / 'rare_event_histogram.png'}")


if __name__ == "__main__":
    run_example()
