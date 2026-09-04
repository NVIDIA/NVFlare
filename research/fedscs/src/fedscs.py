"""
FedSCS: Federated Learning with Stable Cosine Similarity.

This module contains the algorithmic implementation of FedSCS.
It is framework-independent and can be used by the NVFlare
research example.
"""

from typing import Dict, List, Tuple

import numpy as np


def flatten_update(update: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten a model update into a single vector."""
    parts = []

    for name in sorted(update.keys()):
        value = np.asarray(update[name], dtype=np.float64)
        parts.append(value.reshape(-1))

    if not parts:
        return np.array([], dtype=np.float64)

    return np.concatenate(parts)


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Compute numerically stable cosine similarity."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm < eps or b_norm < eps:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def compute_fedscs_weights(
    updates: Dict[str, Dict[str, np.ndarray]],
    previous_scores: Dict[str, float] | None = None,
    eps: float = 1e-12,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute FedSCS aggregation weights.

    Parameters
    ----------
    updates:
        Mapping from client name to model update.

    previous_scores:
        Previous stability/trust score for each client.

    eps:
        Numerical stability constant.

    Returns
    -------
    weights:
        Normalized FedSCS aggregation weights.

    scores:
        Updated stability scores.
    """
    client_names = list(updates.keys())

    if not client_names:
        return {}, {}

    previous_scores = previous_scores or {}

    vectors = {
        client: flatten_update(updates[client])
        for client in client_names
    }

    raw_scores = {}

    for client in client_names:
        own_update = vectors[client]

        peer_updates = [
            vectors[other]
            for other in client_names
            if other != client
        ]

        if not peer_updates:
            raw_scores[client] = 1.0
            continue

        peer_sum = np.sum(peer_updates, axis=0)

        similarity = cosine_similarity(
            own_update,
            peer_sum,
            eps=eps,
        )

        # FedSCS uses non-negative similarity.
        raw_scores[client] = max(similarity, 0.0)

    scores = {}

    for client in client_names:
        previous = previous_scores.get(client, 0.0)

        # First round uses the current score directly.
        if client not in previous_scores:
            current = raw_scores[client]
        else:
            # Running stability/trust score.
            current = (
                previous
                + (raw_scores[client] - previous)
                / max(len(previous_scores) + 1, 1)
            )

        scores[client] = max(float(current), 0.0)

    # If all scores are zero, fall back to uniform weighting.
    total = sum(scores.values())

    if total <= eps:
        uniform = 1.0 / len(client_names)

        weights = {
            client: uniform
            for client in client_names
        }

        return weights, scores

    weights = {
        client: scores[client] / total
        for client in client_names
    }

    return weights, scores


def aggregate_models(
    models: Dict[str, Dict[str, np.ndarray]],
    weights: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """
    Aggregate client models using FedSCS weights.
    """
    if not models:
        return {}

    client_names = list(models.keys())
    parameter_names = models[client_names[0]].keys()

    aggregated = {}

    for name in parameter_names:
        result = None

        for client in client_names:
            value = np.asarray(models[client][name])

            weighted = weights[client] * value

            if result is None:
                result = weighted.astype(np.float64)
            else:
                result += weighted

        aggregated[name] = result.astype(
            np.asarray(models[client_names[0]][name]).dtype
        )

    return aggregated
