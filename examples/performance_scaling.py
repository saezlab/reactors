#!/usr/bin/env python3
"""Generate multi-run performance charts to compare Reactors versions."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
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


def measure_run(
    model: BenchmarkModel,
    n_trajectories: int,
    t_end: float,
    n_threads: int | None,
    seed: int,
) -> tuple[float, float]:
    start = time.perf_counter()
    reactors.simulate_ensemble(
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
    throughput = n_trajectories / elapsed if elapsed > 0 else float("inf")
    return elapsed, throughput


def sweep_threads(
    model: BenchmarkModel,
    thread_values: list[int],
    n_trajectories: int,
    t_end: float,
    repetitions: int,
    seed: int,
) -> list[dict]:
    results = []
    base_seed = seed
    for threads in thread_values:
        for rep in range(repetitions):
            elapsed, throughput = measure_run(
                model,
                n_trajectories=n_trajectories,
                t_end=t_end,
                n_threads=threads,
                seed=base_seed + rep,
            )
            results.append(
                {
                    "kind": "threads",
                    "threads": threads,
                    "trajectories": n_trajectories,
                    "elapsed": elapsed,
                    "throughput": throughput,
                    "rep": rep,
                }
            )
    return results


def sweep_trajectories(
    model: BenchmarkModel,
    trajectories_values: list[int],
    n_threads: int | None,
    t_end: float,
    repetitions: int,
    seed: int,
) -> list[dict]:
    results = []
    base_seed = seed
    for n_traj in trajectories_values:
        for rep in range(repetitions):
            elapsed, throughput = measure_run(
                model,
                n_trajectories=n_traj,
                t_end=t_end,
                n_threads=n_threads,
                seed=base_seed + rep,
            )
            results.append(
                {
                    "kind": "trajectories",
                    "threads": n_threads if n_threads is not None else 0,
                    "trajectories": n_traj,
                    "elapsed": elapsed,
                    "throughput": throughput,
                    "rep": rep,
                }
            )
    return results


def summarize_thread_results(results: list[dict]) -> tuple[list[str], list[float]]:
    labels: list[str] = []
    throughputs: list[float] = []
    grouped: dict[int, list[float]] = {}
    for row in results:
        grouped.setdefault(row["threads"], []).append(row["throughput"])
    for threads, values in sorted(grouped.items(), key=lambda kv: kv[0]):
        labels.append("auto" if threads == 0 else str(threads))
        throughputs.append(float(np.mean(values)))
    return labels, throughputs


def summarize_trajectory_results(results: list[dict]) -> tuple[list[int], list[float]]:
    traj_counts: list[int] = []
    per_traj: list[float] = []
    grouped: dict[int, list[float]] = {}
    for row in results:
        grouped.setdefault(row["trajectories"], []).append(row["elapsed"] / row["trajectories"])
    for n_traj, values in sorted(grouped.items()):
        traj_counts.append(n_traj)
        per_traj.append(float(np.mean(values)))
    return traj_counts, per_traj


def _draw_thread_plot(ax: plt.Axes, labels: list[str], throughputs: list[float]) -> None:
    ax.plot(labels, throughputs, marker="o")
    ax.set_xlabel("Threads (Rayon pool)")
    ax.set_ylabel("Trajectories per second")
    ax.set_title("Throughput vs. Threads")
    ax.grid(True, linestyle="--", alpha=0.4)


def _draw_trajectory_plot(ax: plt.Axes, traj_counts: list[int], per_traj: list[float]) -> None:
    ax.plot(traj_counts, per_traj, marker="s", color="tab:orange")
    ax.set_xlabel("Trajectories per ensemble run")
    ax.set_ylabel("Seconds per trajectory")
    ax.set_title("Cost vs. Ensemble Size")
    ax.set_xscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_thread_scaling(results: list[dict], output_path: Path) -> tuple[list[str], list[float]]:
    labels, values = summarize_thread_results(results)
    fig, ax = plt.subplots(figsize=(6, 4))
    _draw_thread_plot(ax, labels, values)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return labels, values


def plot_trajectory_scaling(results: list[dict], output_path: Path) -> tuple[list[int], list[float]]:
    traj_counts, per_traj = summarize_trajectory_results(results)
    fig, ax = plt.subplots(figsize=(6, 4))
    _draw_trajectory_plot(ax, traj_counts, per_traj)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return traj_counts, per_traj


def plot_summary_figure(
    thread_labels: list[str],
    thread_throughput: list[float],
    traj_counts: list[int],
    per_traj: list[float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _draw_thread_plot(axes[0], thread_labels, thread_throughput)
    _draw_trajectory_plot(axes[1], traj_counts, per_traj)
    fig.suptitle("Reactors Performance Overview", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_metrics_csv(results: list[dict], output_path: Path) -> None:
    fieldnames = ["kind", "threads", "trajectories", "rep", "elapsed", "throughput"]
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def parse_int_list(text: str) -> list[int]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate performance plots for Reactors.")
    parser.add_argument("--species", type=int, default=256, help="Number of species in the random model")
    parser.add_argument("--reactions", type=int, default=1024, help="Number of reactions in the random model")
    parser.add_argument("--t-end", type=float, default=50.0, help="SSA end time")
    parser.add_argument(
        "--thread-grid",
        type=parse_int_list,
        default=[1, 2, 4, 8, 0],
        help="Comma-separated thread counts (0 uses Rayon default)",
    )
    parser.add_argument(
        "--trajectories-grid",
        type=parse_int_list,
        default=[128, 256, 512, 1024, 2048],
        help="Comma-separated trajectory counts for scaling plot",
    )
    parser.add_argument(
        "--threads-for-trajectories",
        type=int,
        default=0,
        help="Thread setting used during trajectory sweep (0 for Rayon default)",
    )
    parser.add_argument("--trajectories", type=int, default=512, help="Baseline trajectory count for thread sweep")
    parser.add_argument("--repetitions", type=int, default=2, help="Number of repeated runs per grid point")
    parser.add_argument("--seed", type=int, default=2025, help="Base RNG seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/output"),
        help="Directory where plots and CSV metrics are written",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Building random model with {args.species} species / {args.reactions} reactions "
        f"(seed={args.seed})"
    )
    model = build_random_model(args.species, args.reactions, seed=args.seed)

    print("Running thread sweep...")
    thread_results = sweep_threads(
        model=model,
        thread_values=args.thread_grid,
        n_trajectories=args.trajectories,
        t_end=args.t_end,
        repetitions=args.repetitions,
        seed=args.seed,
    )

    print("Running trajectory sweep...")
    traj_results = sweep_trajectories(
        model=model,
        trajectories_values=args.trajectories_grid,
        n_threads=args.threads_for_trajectories if args.threads_for_trajectories != 0 else None,
        t_end=args.t_end,
        repetitions=args.repetitions,
        seed=args.seed + 10_000,
    )

    all_results = thread_results + traj_results
    csv_path = args.output_dir / "performance_metrics.csv"
    write_metrics_csv(all_results, csv_path)
    print(f"Wrote metrics table to {csv_path}")

    threads_png = args.output_dir / "performance_threads.png"
    thread_labels, thread_throughput = plot_thread_scaling(thread_results, threads_png)
    print(f"Saved thread scaling plot to {threads_png}")

    traj_png = args.output_dir / "performance_trajectories.png"
    traj_counts, per_traj = plot_trajectory_scaling(traj_results, traj_png)
    print(f"Saved trajectory scaling plot to {traj_png}")

    summary_png = args.output_dir / "performance_summary.png"
    plot_summary_figure(thread_labels, thread_throughput, traj_counts, per_traj, summary_png)
    print(f"Saved summary plot to {summary_png}")


if __name__ == "__main__":
    main()
