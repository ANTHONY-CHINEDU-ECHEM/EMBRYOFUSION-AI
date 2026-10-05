"""Metrics with bootstrap uncertainty, including paired comparisons between models."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def summary_metrics(y: np.ndarray, p: np.ndarray, rounds: int = 500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(rounds):
        index = rng.integers(0, len(y), len(y))
        if y[index].min() != y[index].max():
            aucs.append(roc_auc_score(y[index], p[index]))
    return {"roc_auc": float(roc_auc_score(y, p)), "roc_auc_ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
            "pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, np.clip(p, 0, 1)))}


def paired_auc_difference(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, rounds: int = 500, seed: int = 0) -> dict:
    """Bootstrap the AUC difference (a minus b) on the same resampled cases.

    Pairing matters: both models are scored on identical patients, so most of
    the sampling noise cancels and a small but real gain can be detected.
    """
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(rounds):
        index = rng.integers(0, len(y), len(y))
        if y[index].min() != y[index].max():
            differences.append(roc_auc_score(y[index], p_a[index]) - roc_auc_score(y[index], p_b[index]))
    differences = np.array(differences)
    return {"difference": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
            "ci": [float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5))],
            "share_of_resamples_positive": float((differences > 0).mean())}
