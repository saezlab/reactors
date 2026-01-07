#!/usr/bin/env python3
"""Sparse GRN ensemble with correlations and driver motifs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import reactors

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class NetworkConfig:
    n_genes: int = 50
    min_fanin: int = 3
    max_fanin: int = 8
    hill_n: int = 2
    k_act: float = 20.0
    k_rep: float = 18.0
    basal: float = 0.6
    drive: float = 19.0
    toggle_drive: float = 32.0
    toggle_k: float = 15.0
    toggle_n: int = 3
    degradation_rate: float = 0.90
    seed: int = 7
    driver_mode: str = "tristable"  # or "bistable"


def power_expr(var: str, n: int) -> str:
    return "*".join([var] * n) if n > 1 else var


def hill_term(idx: int, k_half: float, n: int) -> str:
    species = f"s{idx}"
    num = power_expr(species, n)
    denom_const = k_half**n
    return f"({num})/({denom_const} + {num})"


def repress_term(idx: int, k_half: float, n: int) -> str:
    species = f"(s{idx}/{k_half})"
    num = power_expr(species, n)
    return f"1.0/(1.0 + {num})"


def build_sparse_grn(
    config: NetworkConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], tuple[int, ...]]:
    """Construct stoichiometry, rate constants, reaction codes, and expressions."""
    rng = np.random.default_rng(config.seed)
    n = config.n_genes
    stoich_rows: list[np.ndarray] = []
    reaction_types: list[int] = []
    expressions: list[str] = []
    rate_constants: list[float] = []

    if config.driver_mode == "bistable":
        driver_genes = (0, 1)
        edges = [(0, 1), (1, 0)]
    elif config.driver_mode == "tristable":
        driver_genes = (0, 1, 2)
        # 3-cycle of mutual repression to encourage three attractors.
        edges = [(0, 2), (1, 0), (2, 1)]
    else:
        raise ValueError("driver_mode must be 'bistable' or 'tristable'")

    for target, regulator in edges:
        row = np.zeros(n, dtype=np.int32)
        row[target] = 1
        stoich_rows.append(row)
        reaction_types.append(reactors.ReactionType.EXPRESSION)
        expr = (
            f"{config.basal} + {config.toggle_drive} * "
            f"{repress_term(regulator, config.toggle_k, config.toggle_n)}"
        )
        expressions.append(expr)
        rate_constants.append(0.0)  # ignored for expression reactions

        row_deg = np.zeros(n, dtype=np.int32)
        row_deg[target] = -1
        stoich_rows.append(row_deg)
        reaction_types.append(reactors.ReactionType.MASS_ACTION)
        expressions.append(None)
        rate_constants.append(config.degradation_rate)

    start_idx = len(driver_genes)

    # Remaining genes get sparse random regulators (mix of activation/repression).
    for target in range(start_idx, n):
        fanin = rng.integers(config.min_fanin, config.max_fanin + 1)
        regulators: list[int] = rng.choice(
            [idx for idx in range(n) if idx != target], size=fanin, replace=False
        ).tolist()
        signs = rng.choice([-1, 1], size=fanin)

        activators = [reg for reg, sign in zip(regulators, signs) if sign > 0]
        repressors = [reg for reg, sign in zip(regulators, signs) if sign < 0]

        if activators:
            act_terms = " + ".join(
                hill_term(idx, config.k_act, config.hill_n) for idx in activators
            )
            act_block = f"({act_terms})/{len(activators)}"
        else:
            act_block = "1.0"  # fall back to basal-only production

        if repressors:
            rep_block = " * ".join(
                repress_term(idx, config.k_rep, config.hill_n) for idx in repressors
            )
        else:
            rep_block = "1.0"

        row = np.zeros(n, dtype=np.int32)
        row[target] = 1
        stoich_rows.append(row)
        reaction_types.append(reactors.ReactionType.EXPRESSION)
        expressions.append(f"{config.basal} + {config.drive} * {act_block} * {rep_block}")
        rate_constants.append(0.0)

        row_deg = np.zeros(n, dtype=np.int32)
        row_deg[target] = -1
        stoich_rows.append(row_deg)
        reaction_types.append(reactors.ReactionType.MASS_ACTION)
        expressions.append(None)
        rate_constants.append(config.degradation_rate)

    stoich = np.stack(stoich_rows)
    reaction_type_codes = np.array(reaction_types, dtype=np.int32)
    rate_constants_arr = np.array(rate_constants, dtype=np.float64)
    return stoich, rate_constants_arr, reaction_type_codes, expressions, tuple(driver_genes)


def plot_results(
    final_states: np.ndarray,
    t_points: Sequence[float],
    ensemble: np.ndarray,
    out_path: Path,
    driver_genes: Sequence[int],
    intervention_times: Sequence[float] | None = None,
) -> None:
    """Render scatter of the toggle pair plus a correlation heatmap for all genes."""
    log_counts = np.log1p(final_states)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(log_counts, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    fig = plt.figure(figsize=(15, 5.5), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0])

    ax_time = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_heatmap = fig.add_subplot(gs[0, 2])

    # Driver motif trajectories: show median only for clarity.
    driver_list = list(driver_genes)
    driver_trajs = ensemble[:, :, driver_list]  # (n_traj, n_t, n_drivers)
    n_traj = driver_trajs.shape[0]
    cmap = plt.get_cmap("Dark2")
    time_label = "Tristable ring drivers" if len(driver_list) == 3 else "Bistable toggle drivers"
    for idx, gene_idx in enumerate(driver_list):
        trajectories = driver_trajs[:, :, idx]
        median = np.median(trajectories, axis=0)
        color = cmap(idx)
        ax_time.plot(t_points, median, color=color, linewidth=2.0, label=f"Gene {gene_idx}")

    ax_time.set_xlabel("Time")
    ax_time.set_ylabel("Copy count")
    ax_time.set_title(f"{time_label} median over {n_traj} trajectories")

    # Mark intervention times (if any) to show when nudges occurred.
    if intervention_times:
        shown_label = False
        for t in intervention_times:
            label = "Intervention" if not shown_label else None
            ax_time.axvline(
                t, color="#555555", linestyle="--", linewidth=1.1, alpha=0.7, label=label
            )
            shown_label = True

    ax_time.legend(loc="upper right", frameon=False)

    gene_x, gene_y = driver_genes[0], driver_genes[1]
    ax_scatter.scatter(
        final_states[:, gene_x],
        final_states[:, gene_y],
        s=10,
        alpha=0.25,
        color="#4c72b0",
        edgecolors="none",
    )
    ax_scatter.set_xlabel(f"Gene {gene_x} copies (driver A)")
    ax_scatter.set_ylabel(f"Gene {gene_y} copies (driver B)")
    motif_label = "Bistable toggle" if len(driver_genes) == 2 else "Tristable ring (pairwise view)"
    ax_scatter.set_title(f"{motif_label} embedded in 50-gene network")

    im = ax_heatmap.imshow(
        corr,
        cmap=mcolors.LinearSegmentedColormap.from_list(
            "blue-white-red", ["#2c7bb6", "#f7f7f7", "#b2182b"]
        ),
        norm=mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
        origin="lower",
        interpolation="nearest",
    )
    ax_heatmap.set_title("Log1p expression correlations (50 genes)")
    ax_heatmap.set_xlabel("Gene index")
    ax_heatmap.set_ylabel("Gene index")
    fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04, label="Pearson r")

    # Annotate time span for context.
    fig.suptitle(
        f"50-gene sparse GRN ensemble: {ensemble.shape[0]} trajectories, "
        f"{len(t_points)} timepoints, t_end={t_points[-1]:.0f}",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_example() -> None:
    parser = argparse.ArgumentParser(description="Sparse GRN ensemble with driver motifs.")
    parser.add_argument(
        "--driver",
        choices=("bistable", "tristable"),
        default="tristable",
        help="Choose driver motif embedded in the GRN.",
    )
    parser.add_argument(
        "--no-intervention",
        action="store_true",
        help="Disable the mid-run species pulses applied to the driver genes.",
    )
    parser.add_argument(
        "--pulse",
        nargs=3,
        action="append",
        metavar=("GENE", "TIME", "DELTA"),
        help="Add a pulse: e.g., '--pulse g0 25 60' adds 60 copies to gene 0 at t=25. "
        "Repeatable.",
    )
    filtered_args = [arg for arg in sys.argv[1:] if arg != "--"]
    args = parser.parse_args(filtered_args)

    def parse_pulse(arg_list: list[str]) -> tuple[int, float, float]:
        gene_label, t_str, delta_str = arg_list
        if not gene_label.startswith("g") or not gene_label[1:].isdigit():
            raise argparse.ArgumentTypeError("GENE must look like g0, g1, g2, ...")
        gene_idx = int(gene_label[1:])
        try:
            time = float(t_str)
            delta = float(delta_str)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("TIME and DELTA must be numeric") from exc
        return gene_idx, time, delta

    config = NetworkConfig(driver_mode=args.driver)
    stoich, rate_constants, reaction_types, expressions, driver_genes = build_sparse_grn(
        config
    )

    initial_state = np.zeros(config.n_genes, dtype=np.int32)
    t_points = np.linspace(0.0, 60.0, 241)

    interventions = []
    if args.pulse:
        parsed = [parse_pulse(p) for p in args.pulse]
        for gene_idx, time, delta in parsed:
            if gene_idx < 0 or gene_idx >= config.n_genes:
                raise ValueError(f"Pulse gene index {gene_idx} outside 0..{config.n_genes-1}")
            interventions.append({"time": time, "species_delta": [(gene_idx, delta)]})
        interventions.sort(key=lambda entry: entry["time"])
    elif not args.no_intervention:
        interventions = [
            # A gentle push toward the first driver midway through the run.
            {"time": 25.0, "species_delta": [(driver_genes[0], 30)]},
            # Later, nudge toward the second driver (or another driver gene).
            {"time": 40.0, "species_delta": [(driver_genes[1], 30)]},
        ]

    ensemble = reactors.simulate_ensemble(
        stoich=stoich,
        initial_state=initial_state,
        rate_constants=rate_constants,
        reaction_type_codes=reaction_types,
        reaction_expressions=expressions,
        t_end=t_points[-1],
        n_trajectories=512,
        t_points=t_points,
        interventions=interventions,
        seed=5,
    )

    final_states = reactors.simulate_ensemble(
        stoich=stoich,
        initial_state=initial_state,
        rate_constants=rate_constants,
        reaction_type_codes=reaction_types,
        reaction_expressions=expressions,
        t_end=t_points[-1],
        n_trajectories=4000,
        interventions=interventions,
        mode="final",
        seed=6,
    )

    out_path = OUTPUT_DIR / "grn_sparse_network.png"
    intervention_times = [plan["time"] for plan in interventions] if interventions else []
    plot_results(final_states, t_points, ensemble, out_path, driver_genes, intervention_times)
    print("Saved GRN correlations figure to", out_path)


if __name__ == "__main__":
    run_example()
