#!/usr/bin/env python3
"""COVID-19-style SIR ensemble simulation that saves a summary plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import reactors

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_example() -> None:
    population = 100_000
    initial_infected = 250
    initial_state = np.array(
        [population - initial_infected, initial_infected, 0], dtype=np.int32
    )

    stoich = np.array(
        [
            [-1, 1, 0],  # Infection: S + I -> 2I
            [0, -1, 1],  # Recovery: I -> R
        ],
        dtype=np.int32,
    )

    infectious_period = 7.0  # days
    gamma = 1.0 / infectious_period
    r0 = 2.5  # basic reproduction number
    beta = r0 * gamma
    infection_prefactor = beta / population

    reaction_types = np.array(
        [reactors.ReactionType.EXPRESSION, reactors.ReactionType.MASS_ACTION],
        dtype=np.int32,
    )
    reaction_expressions = [
        f"{infection_prefactor:.12e} * s0 * s1",  # normalized by population
        None,
    ]
    rate_constants = np.array([0.0, gamma], dtype=np.float64)  # first slot ignored for expression

    t_end = 200.0
    t_points = np.linspace(0.0, t_end, 401)
    trajectories = reactors.simulate_ensemble(
        stoich=stoich,
        initial_state=initial_state,
        rate_constants=rate_constants,
        reaction_type_codes=reaction_types,
        reaction_expressions=reaction_expressions,
        t_end=t_end,
        n_trajectories=2048,
        t_points=t_points,
        seed=2020,
    )

    means = trajectories.mean(axis=0)
    percentiles = np.percentile(trajectories, [10, 90], axis=0)

    colors = {
        "S": "#1f78b4",
        "I": "#e31a1c",
        "R": "#33a02c",
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, label in enumerate(("S", "I", "R")):
        ax.plot(t_points, means[:, idx], label=f"{label} mean", color=colors[label], linewidth=2)
        ax.fill_between(
            t_points,
            percentiles[0, :, idx],
            percentiles[1, :, idx],
            color=colors[label],
            alpha=0.15,
            label=f"{label} 10–90%",
        )

    ax.set_xlabel("days")
    ax.set_ylabel("individuals")
    ax.set_title("COVID-19-style SIR ensemble (exact SSA)")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "covid_sir_timeseries.png", dpi=200)
    plt.close(fig)
    print("Saved plot to", OUTPUT_DIR / "covid_sir_timeseries.png")


if __name__ == "__main__":
    run_example()
