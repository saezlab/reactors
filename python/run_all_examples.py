#!/usr/bin/env python3
"""Run every Python example script in sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    name: str
    relative_path: str


# Python-only demos that do not require external toolchains.
CORE_EXAMPLES = [
    Example("birth-death", "examples/birth_death.py"),
    Example("boolean-network", "examples/boolean_network.py"),
    Example("toggle-switch", "examples/toggle_switch.py"),
    Example("rna-splicing", "examples/rna_splicing.py"),
    Example("enzyme-kinetics", "examples/enzyme_kinetics.py"),
    Example("cell-growth", "examples/cell_growth.py"),
    Example("parameter-inference-mcmc", "examples/parameter_inference_mcmc.py"),
    Example(
        "transcriptional-burst-analysis", "examples/transcriptional_burst_analysis.py"
    ),
    Example("single-cell-pharmacokinetics", "examples/single_cell_pharmacokinetics.py"),
    Example("pk-optimizer", "examples/pk_optimizer.py"),
    Example("toggle-switch-optimizer", "examples/toggle_switch_optimizer.py"),
    Example("covid-sir", "examples/covid_sir.py"),
    Example("rna-velocity-abc", "examples/rna_velocity_abc.py"),
    Example(
        "michaelis-menten-deterministic", "examples/michaelis_menten_deterministic.py"
    ),
    Example("grn-sparse", "examples/grn_sparse_network.py"),
]

# Heavy, timing-focused demos that should be opt-in.
PERFORMANCE_EXAMPLES = [
    Example("large-benchmark", "examples/large_benchmark.py"),
    Example("performance-scaling", "examples/performance_scaling.py"),
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run every example script sequentially."
    )
    parser.add_argument(
        "--include-performance",
        action="store_true",
        help="Also run the heavier performance/benchmark examples.",
    )
    return parser.parse_args()


def run_example(example: Example, project_root: Path) -> None:
    script_path = project_root / example.relative_path
    if not script_path.exists():
        raise FileNotFoundError(f"{example.name} script missing: {script_path}")
    print(f"\n=== Running example: {example.name} ({example.relative_path}) ===")
    subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        cwd=project_root,
    )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    examples = list(CORE_EXAMPLES)
    if args.include_performance:
        examples.extend(PERFORMANCE_EXAMPLES)
    for example in examples:
        run_example(example, project_root)
    print("\nAll requested examples completed successfully.")


if __name__ == "__main__":
    main()
