#!/usr/bin/env python3
"""Optimize single-cell PK parameters with SciPy black-box search."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import reactors


@dataclass(frozen=True)
class PKModel:
    stoich: np.ndarray
    initial_state: np.ndarray
    reaction_type_codes: np.ndarray
    t_points: np.ndarray
    complex_index: int = 2


def build_model() -> PKModel:
    stoich = np.array(
        [
            [-1, -1, 1],   # D + T -> C (binding)
            [1, 1, -1],    # C -> D + T (unbinding)
            [0, 0, -1],    # C -> ∅ (clearance / response)
        ],
        dtype=np.int32,
    )
    initial_state = np.array([100, 50, 0], dtype=np.int32)
    reaction_type_codes = np.full(stoich.shape[0], reactors.ReactionType.MASS_ACTION, dtype=np.int32)
    t_points = np.linspace(0.0, 120.0, 241)
    return PKModel(
        stoich=stoich,
        initial_state=initial_state,
        reaction_type_codes=reaction_type_codes,
        t_points=t_points,
        complex_index=2,
    )


MODEL = build_model()
RESPONSE_THRESHOLD = 25
THERAPEUTIC_WINDOW = (10.0, 60.0)
LOW_FIDELITY_TRAJ = 2000
HIGH_FIDELITY_TRAJ = 20000
SEED = 123
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def simulate_population(rate_constants: np.ndarray, n_trajectories: int, seed: int) -> np.ndarray:
    ensemble = reactors.simulate_ensemble(
        stoich=MODEL.stoich,
        initial_state=MODEL.initial_state,
        rate_constants=rate_constants,
        reaction_type_codes=MODEL.reaction_type_codes,
        t_end=MODEL.t_points[-1],
        n_trajectories=n_trajectories,
        t_points=MODEL.t_points,
        seed=seed,
    )
    return np.asarray(ensemble)


def first_passage_time(traj: np.ndarray, threshold: int) -> float:
    counts = traj[:, MODEL.complex_index]
    indices = np.flatnonzero(counts >= threshold)
    if indices.size == 0:
        return np.inf
    return MODEL.t_points[indices[0]]


def efficacy_from_rates(rate_constants: np.ndarray, n_trajectories: int, seed: int) -> float:
    trajectories = simulate_population(rate_constants, n_trajectories, seed)
    times = np.array([first_passage_time(traj, RESPONSE_THRESHOLD) for traj in trajectories])
    window = (times > THERAPEUTIC_WINDOW[0]) & (times < THERAPEUTIC_WINDOW[1])
    return window.mean()


@lru_cache(maxsize=64)
def objective_cached(log_rates: Tuple[float, float, float]) -> float:
    rate_constants = np.exp(np.array(log_rates))
    efficacy = efficacy_from_rates(rate_constants, LOW_FIDELITY_TRAJ, SEED)
    return -efficacy  # minimize negative efficacy


def run_optimizer() -> Tuple[np.ndarray, float]:
    initial_rates = np.array([8e-4, 0.05, 0.01])
    initial_log = np.log(initial_rates)
    result = minimize(
        lambda logs: objective_cached(tuple(logs)),
        x0=initial_log,
        method="Powell",
        options={"maxiter": 60, "xtol": 1e-3, "ftol": 1e-3},
    )
    best_logs = result.x
    best_rates = np.exp(best_logs)
    best_efficacy = -objective_cached(tuple(best_logs))
    return best_rates, best_efficacy


def plot_contrast(baseline: np.ndarray, optimized: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["k_on", "k_off", "k_clear"]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, baseline, width, label="baseline")
    ax.bar(x + width / 2, optimized, width, label="optimized")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("rate constant")
    ax.set_title("PK parameter optimization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pk_optimizer_rates.png", dpi=200)
    plt.close(fig)


def plot_response_hist(rate_constants: np.ndarray, filename: Path) -> None:
    trajectories = simulate_population(rate_constants, HIGH_FIDELITY_TRAJ, seed=SEED + 1)
    times = np.array([first_passage_time(traj, RESPONSE_THRESHOLD) for traj in trajectories])
    finite = times[np.isfinite(times)]
    fig, ax = plt.subplots(figsize=(7, 4))
    if finite.size:
        ax.hist(finite, bins=60, density=True, color="#1f78b4", alpha=0.8)
    ax.axvspan(*THERAPEUTIC_WINDOW, color="#33a02c", alpha=0.2, label="therapeutic window")
    ax.set_xlabel("time to hit threshold")
    ax.set_ylabel("density")
    ax.set_title("Response times at optimized rates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def run_example() -> None:
    baseline = np.array([8e-4, 0.05, 0.01])
    baseline_efficacy = efficacy_from_rates(baseline, LOW_FIDELITY_TRAJ, SEED)
    best_rates, best_efficacy = run_optimizer()
    high_fidelity_efficacy = efficacy_from_rates(best_rates, HIGH_FIDELITY_TRAJ, SEED + 2)

    plot_contrast(baseline, best_rates)
    plot_response_hist(best_rates, OUTPUT_DIR / "pk_optimizer_response_times.png")

    labels = ["k_on", "k_off", "k_clear"]
    print("Baseline parameters:")
    for name, value in zip(labels, baseline):
        print(f"  {name:>7}: {value:.4f}")
    print(f"Baseline efficacy (low-fidelity): {baseline_efficacy:0.3f}")

    print("\nOptimized parameters:")
    for name, value in zip(labels, best_rates):
        print(f"  {name:>7}: {value:.4f}")
    print(f"Optimizer efficacy (low-fidelity): {best_efficacy:0.3f}")
    print(f"Optimizer efficacy (20k trajectories): {high_fidelity_efficacy:0.3f}")
    print(f"Saved plots under {OUTPUT_DIR}")


if __name__ == "__main__":
    run_example()
