"""Python package for the GRN SSA extension module."""

from importlib import metadata
from enum import IntEnum

from .reactors import simulate_ensemble


class ReactionType(IntEnum):
    """IntEnum aliases for reaction_type_codes."""

    MASS_ACTION = 0
    HILL = 1
    MICHAELIS_MENTEN = 2
    EXPRESSION = 3


__all__ = ["ReactionType", "simulate_ensemble", "__version__"]


def __getattr__(name: str):
    if name == "__version__":
        try:
            return metadata.version("reactors")
        except metadata.PackageNotFoundError:  # pragma: no cover
            return "0.0.0"
    raise AttributeError(name)
