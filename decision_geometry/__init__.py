"""Tools for analyzing decision-related population geometry."""

from .analysis import AnalysisResult, analyze_population
from .data import PopulationDataset, load_population

__all__ = [
    "AnalysisResult",
    "PopulationDataset",
    "analyze_population",
    "load_population",
]
