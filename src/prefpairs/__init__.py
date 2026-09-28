"""PrefPairs: an RLHF preference-data toolkit.

Collect pairwise judgments, audit annotation quality, aggregate preferences with
Bradley-Terry and Elo, and export DPO, KTO and reward-model datasets.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("prefpairs")
except PackageNotFoundError:  # pragma: no cover - only when running from a raw checkout
    __version__ = "0.0.0"

__all__ = ["__version__"]
