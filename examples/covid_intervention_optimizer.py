#!/usr/bin/env python3
"""Optimize COVID-19 intervention timing/strength to reduce hospital load"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import dual_annealing

import reactors

# Output configuration
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Epidemiological / SIR model parameters
POPULATION = 100_000
INITIAL_INFECTED = 250
INITIAL_STATE = np.array([POPULATION - INITIAL_INFECTED, INITIAL_INFECTED, 0], dtype=np.int32)
INFECTIOUS_PERIOD = 7.0  # days
GAMMA = 1.0 / INFECTIOUS_PERIOD
R0_BASE = 2.5
BASE_BETA = R0_BASE * GAMMA
INFECTION_RATE = BASE_BETA / POPULATION

REACTION_TYPES = np.array(
    [reactors.ReactionType.EXPRESSION, reactors.ReactionType.MASS_ACTION], dtype=np.int32
)

STOICH_SIR = np.array(
    [
        [-1, 1, 0],  # Infection: S + I -> 2I
        [0, -1, 1],  # Recovery: I -> R
    ],
    dtype=np.int32,
)

# Simulation parameters
HORIZON = 180.0  # days
DT = 1.0  # reporting cadence
N_TRAJ = 64
SEED_BASE = 2024

# Control parameters and bounds
DURATION_BOUNDS = (5.0, 60.0)
INTERVENTION_DAY_BOUNDS = (5.0, 40.0)
REDUCTION_BOUNDS = (0.1, 0.9)  # fraction of beta removed when intervention starts
IMPROVEMENT_TOL = 1e-6


# Cached baseline ensemble (shared pre-intervention history)
_baseline_traj: np.ndarray | None = None  # shape (N_TRAJ, T, 3)
_baseline_times: np.ndarray | None = None  # shape (T,)


def build_time_grid(duration: float) -> np.ndarray:
    steps = max(int(np.round(duration / DT)), 1)
    return np.linspace(0.0, duration, steps + 1)


def infection_expression(beta: float) -> tuple[np.ndarray, list[str | None]]:
    prefactor = beta / POPULATION
    expressions = [f"{prefactor:.12e} * s0 * s1", None]
    return np.array([0.0, GAMMA], dtype=np.float64), expressions


# --- Baseline ensemble (shared pre-intervention history) --------------------


def _get_baseline_ensemble() -> tuple[np.ndarray, np.ndarray]:
    """Compute (once) and return the baseline ensemble over [0, HORIZON].

    Returns
    -------
    traj : (N_TRAJ, T, 3)
        Full SIR trajectories under constant BASE_BETA.
    times : (T,)
        Time grid over [0, HORIZON].
    """

    global _baseline_traj, _baseline_times

    if _baseline_traj is None or _baseline_times is None:
        t_points = build_time_grid(HORIZON)
        rate_constants, expressions = infection_expression(BASE_BETA)
        traj = reactors.simulate_ensemble(
            stoich=STOICH_SIR,
            initial_state=INITIAL_STATE,
            rate_constants=rate_constants,
            reaction_type_codes=REACTION_TYPES,
            reaction_expressions=expressions,
            t_end=HORIZON,
            n_trajectories=N_TRAJ,
            t_points=t_points,
            seed=SEED_BASE,
        )
        _baseline_traj = traj
        _baseline_times = t_points

    return _baseline_traj, _baseline_times


# --- Strategy evaluation ----------------------------------------------------


@dataclass
class StrategyResult:
    peak_load: float
    times: np.ndarray
    percentile90: np.ndarray
    mean_trace: np.ndarray
    start_day: float
    duration: float
    reduction: float


def _baseline_result(capture_series: bool) -> StrategyResult:
    traj, times = _get_baseline_ensemble()
    infected = traj[:, :, 1]
    perc90 = np.percentile(infected, 90, axis=0)
    peak = float(perc90.max())
    mean_trace = infected.mean(axis=0)

    if not capture_series:
        times = np.array([])
        perc90 = np.array([])
        mean_trace = np.array([])

    return StrategyResult(peak, times, perc90, mean_trace, HORIZON, 0.0, 0.0)


def run_strategy(
    start_day: float,
    duration: float,
    reduction: float,
    capture_series: bool = False,
) -> StrategyResult:
    """Evaluate a single intervention strategy.

    The baseline ensemble is reused so that pre-intervention history
    is identical to the baseline up to the intervention start.
    """

    # Raw parameters (continuous); used only for validation.
    start_day = float(start_day)
    duration = float(duration)
    reduction = float(reduction)

    # Quick validity checks
    if (
        reduction <= 0.0
        or duration <= 0.0
        or start_day >= HORIZON
        or start_day + duration <= 0.0
    ):
        return _baseline_result(capture_series)

    traj_base, times_base = _get_baseline_ensemble()
    n_times = times_base.size

    start_idx = int(round(start_day / DT))
    start_idx = max(0, min(start_idx, n_times - 1))
    start_time = times_base[start_idx]

    if start_time >= HORIZON:
        return _baseline_result(capture_series)

    dur_idx = max(int(round(duration / DT)), 1)
    end_idx = min(start_idx + dur_idx, n_times - 1)
    end_time = times_base[end_idx]

    if end_time <= start_time:
        return _baseline_result(capture_series)

    segments_times: list[np.ndarray] = []
    segments_infected: list[np.ndarray] = []

    def append_segment(times_seg: np.ndarray, infected_seg: np.ndarray) -> None:
        if times_seg.size == 0 or infected_seg.size == 0:
            return
        if segments_times:
            times_seg = times_seg[1:]
            infected_seg = infected_seg[:, 1:]
        if times_seg.size == 0:
            return
        segments_times.append(times_seg)
        segments_infected.append(infected_seg)

    if start_idx > 0:
        pre_times = times_base[: start_idx + 1]
        pre_infected = traj_base[:, : start_idx + 1, 1]
        append_segment(pre_times, pre_infected)

    states_current = np.ascontiguousarray(traj_base[:, start_idx, :], dtype=np.int32)
    current_time = start_time

    rate_constants, expressions = infection_expression(BASE_BETA * (1.0 - reduction))

    if end_time > start_time:
        mid_times = times_base[start_idx:end_idx + 1]
        initial_times = np.full(N_TRAJ, current_time, dtype=np.float64)
        mid = reactors.simulate_ensemble(
            stoich=STOICH_SIR,
            initial_state=INITIAL_STATE,
            initial_states=states_current,
            initial_times=initial_times,
            rate_constants=rate_constants,
            reaction_type_codes=REACTION_TYPES,
            reaction_expressions=expressions,
            t_end=end_time,
            n_trajectories=N_TRAJ,
            t_points=mid_times,
            seed=SEED_BASE + 1_000,
        )
        append_segment(mid_times, mid[:, :, 1])
        states_current = np.ascontiguousarray(mid[:, -1, :], dtype=np.int32)
        current_time = end_time

    if HORIZON > current_time:
        tail_times = times_base[end_idx:]
        initial_times = np.full(N_TRAJ, current_time, dtype=np.float64)
        base_rates, base_expr = infection_expression(BASE_BETA)
        tail = reactors.simulate_ensemble(
            stoich=STOICH_SIR,
            initial_state=INITIAL_STATE,
            initial_states=states_current,
            initial_times=initial_times,
            rate_constants=base_rates,
            reaction_type_codes=REACTION_TYPES,
            reaction_expressions=base_expr,
            t_end=HORIZON,
            n_trajectories=N_TRAJ,
            t_points=tail_times,
            seed=SEED_BASE + 5_000,
        )
        append_segment(tail_times, tail[:, :, 1])

    if segments_times:
        combined_times = np.concatenate(segments_times)
        combined_infected = np.concatenate(segments_infected, axis=1)
    else:
        combined_times = times_base[:1]
        combined_infected = traj_base[:, :1, 1]

    perc90 = np.percentile(combined_infected, 90, axis=0)
    peak = float(perc90.max())
    mean_trace = combined_infected.mean(axis=0)

    if not capture_series:
        combined_times = np.array([])
        perc90 = np.array([])
        mean_trace = np.array([])

    return StrategyResult(
        peak_load=peak,
        times=combined_times,
        percentile90=perc90,
        mean_trace=mean_trace,
        start_day=start_time,
        duration=end_time - start_time,
        reduction=reduction,
    )


# Global tracker for best value during optimization
_best_peak = float("inf")


def objective(params: np.ndarray, max_runtime: float, start_time: float) -> float:
    """Objective function for the optimizer.

    params = [start_day, duration, reduction]
    Returns 90th-percentile peak infected count.
    """
    global _best_peak

    start_day, duration, reduction = params

    # Simple bound check (dual_annealing also enforces bounds, this is a fast guard).
    if not (
        INTERVENTION_DAY_BOUNDS[0] <= start_day <= INTERVENTION_DAY_BOUNDS[1]
        and DURATION_BOUNDS[0] <= duration <= DURATION_BOUNDS[1]
        and REDUCTION_BOUNDS[0] <= reduction <= REDUCTION_BOUNDS[1]
    ):
        return float("inf")

    # Hard wall-clock budget
    if time.time() - start_time > max_runtime:
        return float("inf")

    peak = run_strategy(start_day, duration, reduction).peak_load

    if peak + IMPROVEMENT_TOL < _best_peak:
        _best_peak = peak
        print(
            f"New best candidate: start={start_day:.2f}, duration={duration:.2f}, "
            f"reduce beta by {reduction:.2%}, 90th-percentile peak={peak:.0f}"
        )

    return peak


def optimize_strategy(max_runtime: float) -> StrategyResult:
    """Optimize start day, duration, and beta reduction via dual_annealing.

    Stopping is controlled effectively by the wall-clock time check
    inside the objective; we do not tune maxiter here.
    """
    global _best_peak
    _best_peak = float("inf")

    start_time = time.time()

    bounds = [
        INTERVENTION_DAY_BOUNDS,
        DURATION_BOUNDS,
        REDUCTION_BOUNDS,
    ]

    def wrapped_objective(x: np.ndarray) -> float:
        return objective(x, max_runtime=max_runtime, start_time=start_time)

    result = dual_annealing(
        wrapped_objective,
        bounds=bounds,
        no_local_search=True,
        # rely on time check in objective; use default maxiter
    )

    best_start, best_duration, best_reduction = result.x
    return run_strategy(best_start, best_duration, best_reduction, capture_series=True)


def baseline_strategy() -> StrategyResult:
    return _baseline_result(capture_series=True)


def plot_result(best: StrategyResult, baseline: StrategyResult) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.fill_between(
        best.times,
        0,
        best.percentile90,
        color="#fb9a99",
        alpha=0.2,
        label="Intervention 90th %",
    )
    ax.plot(best.times, best.mean_trace, color="#e31a1c", linewidth=2, label="Intervention mean")

    ax.fill_between(
        baseline.times,
        0,
        baseline.percentile90,
        color="#a6cee3",
        alpha=0.15,
        label="Baseline 90th %",
    )
    ax.plot(baseline.times, baseline.mean_trace, color="#1f78b4", linewidth=2, label="Baseline mean")

    start = best.start_day
    end = min(start + best.duration, HORIZON)
    ax.axvspan(start, end, color="#6a3d9a", alpha=0.12, label="Intervention window")
    ax.axvline(start, color="#6a3d9a", linestyle="--", linewidth=1.5)

    ax.set_xlabel("days")
    ax.set_ylabel("infected individuals")
    ax.set_title("COVID-19 SIR intervention optimization (SSA)")
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "covid_intervention_optimization.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COVID intervention optimizer")
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=5.0,
        help="Maximum wall-clock time for optimization (minutes)",
    )
    return parser.parse_args()


def run_example() -> None:
    args = parse_args()
    max_runtime = max(1.0, args.max_minutes * 60.0)
    print(
        "Optimizing intervention timing/strength "
        f"(budget: {args.max_minutes:.1f} min, method: dual_annealing)..."
    )

    best = optimize_strategy(max_runtime)
    baseline = baseline_strategy()

    print(
        "Best strategy: start day "
        f"{best.start_day:.1f}, duration {best.duration:.1f}, beta reduction {best.reduction:.2%} "
        f"-> peak 90th-percentile infected = {best.peak_load:.0f}"
    )

    plot_result(best, baseline)


if __name__ == "__main__":
    run_example()
