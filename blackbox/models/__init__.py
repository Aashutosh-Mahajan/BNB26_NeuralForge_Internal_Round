"""Optional model foundations; the live default remains the labelled heuristic."""
from .linear import LinearStepRanker, train_ranker
__all__ = ["LinearStepRanker", "train_ranker"]
