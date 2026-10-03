"""Diagnosis models: M1 Transformer, M2 LightGBM, M3 autoencoder, M4 Ochiai spectrum and their ensemble.

The logistic ranker in ``linear`` is a legacy baseline kept for comparison.
"""
from .linear import LinearStepRanker, train_ranker

__all__ = ["LinearStepRanker", "train_ranker"]
