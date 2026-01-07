#!/usr/bin/env python3
"""Large-scale random GRN benchmark to stress-test the SSA engine."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

import reactors


@dataclass
class BenchmarkModel:
    stoich: np.ndarray
    rate_constants: np.ndarray
    reaction_type_codes: np.ndarray
    initial_state: np.ndarray


def build_random_model(
    n_species: int,
    n_reactions: int,
    birth_fraction: float = 0.2,
    degradation_fraction: float = 0.4,
    seed: int = 123,
) -> BenchmarkModel:
    rng = np.random.default_rng(seed)
    stoich = np.zeros((n_reactions, n_species), dtype=np.int32)
    rate_constants = rng.uniform(0.05, 1.5, size=n_reactions)
    reaction_type_codes = np.full(n_reactions, reactors.ReactionType.MASS_ACTION, dtype=np.int32)

    n_birth = int(birth_fraction * n_reactions)
    n_degrade = int(degradation_fraction * n_reactions)
    indices = np.arange(n_reactions)
    rng.shuffle(indices)
    birth_idx = set(indices[:n_birth])
    degrade_idx = set(indices[n_birth : n_birth + n_degrade])

    for r in range(n_reactions):
        if r in birth_idx:
            target = rng.integers(0, n_species)
            stoich[r, target] = 1
        elif r in degrade_idx:
            target = rng.integers(0, n_species)
            stoich[r, target] = -1
        else:
            source = rng.integers(0, n_species)
            target = (source + rng.integers(1, n_species)) % n_species
            stoich[r, source] = -1
            stoich[r, target] += 1

    initial_state = rng.poisson(lam=50, size=n_species).astype(np.int32)
    return BenchmarkModel(stoich, rate_constants, reaction_type_codes, initial_state)


def run_benchmark(
    n_species: int,
    n_reactions: int,
    n_trajectories: int,
    t_end: float,
    n_threads: int | None,
    seed: int,
) -> None:
    model = build_random_model(n_species, n_reactions, seed=seed)
    print(
        f"Benchmark: {n_species} species, {n_reactions} reactions, "
        f"{n_trajectories} trajectories, t_end={t_end}, threads={n_threads or 'auto'}"
    )

    start = time.perf_counter()
    result = reactors.simulate_ensemble(
        stoich=model.stoich,
        initial_state=model.initial_state,
        rate_constants=model.rate_constants,
        reaction_type_codes=model.reaction_type_codes,
        t_end=t_end,
        n_trajectories=n_trajectories,
        mode="final",
        n_threads=None if n_threads == 0 else n_threads,
        seed=seed,
    )
    elapsed = time.perf_counter() - start
    mean_state = result.mean(axis=0)
    print(
        f"Completed in {elapsed:.2f}s "
        f"({n_trajectories / elapsed if elapsed > 0 else float('inf'):.1f} traj/s). "
        f"Mean terminal copy range: [{mean_state.min():.1f}, {mean_state.max():.1f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Large random GRN benchmark")
    parser.add_argument("--species", type=int, default=64, help="Number of species")
    parser.add_argument("--reactions", type=int, default=256, help="Number of reactions")
    parser.add_argument("--trajectories", type=int, default=2048, help="Number of trajectories")
    parser.add_argument("--t-end", type=float, default=50.0, help="SSA end time")
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Number of worker threads (0 for Rayon default)",
    )
    parser.add_argument("--seed", type=int, default=2024, help="Base RNG seed")
    args = parser.parse_args()

    run_benchmark(
        n_species=args.species,
        n_reactions=args.reactions,
        n_trajectories=args.trajectories,
        t_end=args.t_end,
        n_threads=args.threads,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
