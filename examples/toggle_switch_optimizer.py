#!/usr/bin/env python3
"""Optimize toggle-switch repression strengths for robust bistability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

import reactors


@dataclass(frozen=True)
class ToggleModel:
    stoich: np.ndarray
    initial_state: np.ndarray
    reaction_type_codes: np.ndarray
    base_hill_params: np.ndarray


def build_model() -> ToggleModel:
    stoich = np.array(
        [
            [1, 0],   # regulated production of A (Hill)
            [0, 1],   # regulated production of B (Hill)
            [-1, 0],  # degradation of A
            [0, -1],  # degradation of B
            [1, 0],   # basal production of A
            [0, 1],   # basal production of B
        ],
        dtype=np.int32,
    )
    initial_state = np.array([5, 5], dtype=np.int32)
    reaction_type_codes = np.array(
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
    base_hill_params = np.array(
        [
            [1.0, 3.0, 1.0],  # placeholder (activator, n, K)
            [0.0, 3.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return ToggleModel(
        stoich=stoich,
        initial_state=initial_state,
        reaction_type_codes=reaction_type_codes,
        base_hill_params=base_hill_params,
    )


MODEL = build_model()
T_END = 80.0
FINAL_TRAJ = 200
SEED = 5
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_RATE_CONSTANTS = np.array([40.0, 40.0, 1.2, 1.2, 2.0, 2.0], dtype=np.float64)


def build_rate_constants(k_prod_a: float, k_prod_b: float, k_deg_a: float, k_deg_b: float) -> np.ndarray:
    rates = BASE_RATE_CONSTANTS.copy()
    rates[0] = k_prod_a
    rates[1] = k_prod_b
    rates[2] = k_deg_a
    rates[3] = k_deg_b
    return rates


def simulate_final_states(rate_constants: np.ndarray, hill_params: np.ndarray) -> np.ndarray:
    finals = reactors.simulate_ensemble(
        stoich=MODEL.stoich,
        initial_state=MODEL.initial_state,
        rate_constants=rate_constants,
        reaction_type_codes=MODEL.reaction_type_codes,
        reaction_type_params=hill_params,
        t_end=T_END,
        n_trajectories=FINAL_TRAJ,
        mode="final",
        seed=SEED,
    )
    return np.asarray(finals)


def bimodality_score(states: np.ndarray) -> float:
    # proxy: average absolute difference between A and B populations
    diff = np.abs(states[:, 0] - states[:, 1])
    return diff.mean()


def objective(params: np.ndarray) -> float:
    k_prod_a, k_prod_b, k_deg_a, k_deg_b, hill_a, hill_b = params
    rate_constants = build_rate_constants(k_prod_a, k_prod_b, k_deg_a, k_deg_b)
    hill_params = MODEL.base_hill_params.copy()
    hill_params[0, 2] = hill_a
    hill_params[1, 2] = hill_b
    finals = simulate_final_states(rate_constants, hill_params)
    score = bimodality_score(finals)
    return -score


BOUNDS = [
    (5.0, 60.0),  # k_prod_a
    (5.0, 60.0),  # k_prod_b
    (0.1, 2.0),   # k_deg_a
    (0.1, 2.0),   # k_deg_b
    (5.0, 40.0),  # Hill K for A repression by B
    (5.0, 40.0),  # Hill K for B repression by A
]


class ProgressCallback:
    def __init__(self):
        self.best_value = float("inf")

    def __call__(self, xk, convergence):
        value = objective(xk)
        if value < self.best_value:
            self.best_value = value
            params = xk
            score = -value
            print(
                "Improved candidate: k_prod_A={:.2f}, k_prod_B={:.2f}, k_deg_A={:.2f}, k_deg_B={:.2f}, K_A={:.2f}, K_B={:.2f}, score={:.3f}".format(
                    *params, score
                )
            )


def run_optimizer() -> Tuple[np.ndarray, float]:
    callback = ProgressCallback()
    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        maxiter=40,
        popsize=8,
        polish=False,
        callback=callback,
    )
    best_params = result.x
    best_score = -result.fun
    return best_params, best_score


def plot_final_states(states: np.ndarray, filename: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(states[:, 0], states[:, 1], s=10, alpha=0.4)
    ax.set_xlabel("A copies")
    ax.set_ylabel("B copies")
    ax.set_title("Final-state scatter")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def run_example() -> None:
    baseline_params = np.array([40.0, 40.0, 1.2, 1.2, 20.0, 20.0])
    baseline_hill = MODEL.base_hill_params.copy()
    baseline_hill[0, 2] = baseline_params[4]
    baseline_hill[1, 2] = baseline_params[5]
    baseline_states = simulate_final_states(build_rate_constants(*baseline_params[:4]), baseline_hill)
    baseline_score = bimodality_score(baseline_states)

    best_params, best_score = run_optimizer()
    optimized_hill = MODEL.base_hill_params.copy()
    optimized_hill[0, 2] = best_params[4]
    optimized_hill[1, 2] = best_params[5]
    optimized_states = simulate_final_states(build_rate_constants(*best_params[:4]), optimized_hill)

    plot_final_states(baseline_states, OUTPUT_DIR / "toggle_optimizer_baseline.png")
    plot_final_states(optimized_states, OUTPUT_DIR / "toggle_optimizer_best.png")

    labels = ["k_prod_A", "k_prod_B", "k_deg_A", "k_deg_B", "K_A", "K_B"]
    print("Baseline parameters:")
    for label, value in zip(labels, baseline_params):
        print(f"  {label:>9}: {value:0.3f}")
    print(f"Baseline bimodality score: {baseline_score:0.3f}")

    print("\nOptimized parameters:")
    for label, value in zip(labels, best_params):
        print(f"  {label:>9}: {value:0.3f}")
    print(f"Optimized bimodality score: {best_score:0.3f}")
    print(f"Saved plots under {OUTPUT_DIR}")


if __name__ == "__main__":
    run_example()
