#!/usr/bin/env python3
"""Cell volume growth with external sunlight drive + division events.

This mirrors a Catalyst.jl example, but here we emulate it with SSA segments
using Reactors. Time-varying rates and deterministic division are handled at
the Python layer by running a sequence of short SSA simulations.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import reactors

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def simulate_cell_growth(
    *,
    use_michaelis_menten: bool = False,
    t_end: float = 20.0,
    dt_chunk: float = 0.1,
    v_threshold: int = 50,
    g_growth: float = 0.3,
    k_p: float = 100.0,
    k_i: float = 60.0,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    stoich = np.array(
        [
            [0, -1, 1],   # G -> Gp
            [0, 1, -1],   # Gp -> G
            [1, 1, -1],   # growth: Gp -> G + V
        ],
        dtype=np.int32,
    )
    if use_michaelis_menten:
        reaction_type_codes = np.array(
            [
                reactors.ReactionType.MASS_ACTION,
                reactors.ReactionType.MASS_ACTION,
                reactors.ReactionType.MICHAELIS_MENTEN,
            ],
            dtype=np.int32,
        )
        # Michaelis–Menten parameters: [substrate_index, K_m]
        reaction_type_params = np.array(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [2.0, 35.0],
            ],
            dtype=np.float64,
        )
    else:
        reaction_type_codes = np.full(3, reactors.ReactionType.MASS_ACTION, dtype=np.int32)
        reaction_type_params = None
    state = np.array([25, 50, 0], dtype=np.int32)  # V, G, Gp
    times = [0.0]
    history = [state.copy()]
    t = 0.0
    step = 0

    while t < t_end - 1e-9:
        dt = min(dt_chunk, t_end - t)
        volume = max(state[0], 1)
        sunlight = k_p * (math.sin(t + 0.5 * dt) + 1.0) / volume
        dephos = k_i / volume
        growth_rate = g_growth
        rates = np.array([sunlight, dephos, growth_rate])

        result = reactors.simulate_ensemble(
            stoich=stoich,
            initial_state=state,
            rate_constants=rates,
            reaction_type_codes=reaction_type_codes,
            reaction_type_params=reaction_type_params,
            t_end=dt,
            n_trajectories=1,
            mode="final",
            seed=seed + step,
        )
        state = result[0].astype(np.int32)
        t += dt

        if state[0] >= v_threshold:
            state[0] //= 2

        times.append(t)
        history.append(state.copy())
        step += 1

    return np.array(times), np.vstack(history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cell growth with optional Michaelis–Menten kinetics")
    parser.add_argument(
        "--use-michaelis-menten",
        action="store_true",
        help="Enable saturating growth (Michaelis–Menten) for the third reaction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    times, states = simulate_cell_growth(use_michaelis_menten=args.use_michaelis_menten)
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    labels = ["Cell volume (V)", "G (unphosphorylated)", "Gᴾ (phosphorylated)"]
    colors = ["#1b9e77", "#d95f02", "#7570b3"]
    for idx, ax in enumerate(axes):
        ax.step(times, states[:, idx], where="post", color=colors[idx])
        ax.set_ylabel(labels[idx])
    axes[-1].set_xlabel("time")
    fig.suptitle("Cell growth with sunlight-driven phosphorylation (SSA segments)")
    fig.tight_layout()
    out_path = OUTPUT_DIR / "cell_growth_timeseries.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved cell growth plot to", out_path)


if __name__ == "__main__":
    main()
