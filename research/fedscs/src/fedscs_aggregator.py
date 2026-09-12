# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FedSCS model aggregator.

FedSCS (Federated Learning with Stable Cosine Similarity) computes
client trust scores from the cosine similarity between each client's
model update and the aggregate update of its peers.

For client i:

    delta_i = w_i - w_global

    peer_i = sum_{j != i} delta_j

    s_i = max(cos(delta_i, peer_i), 0)

The score is combined with temporal stability information from the
previous round and normalized to obtain aggregation weights.

The final aggregation uses the bounded client updates:

    w_global_next = w_global + sum_i(alpha_i * clipped_delta_i)

where clipped_delta_i is the client update after applying the
configured L2 update-norm bound.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nvflare.apis.fl_constant import FLContextKey
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.apis.shareable import Shareable
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.app_constant import AppConstants


class FedSCSAggregator(ModelAggregator):
    """FedSCS model aggregator with bounded client updates."""

    def __init__(
        self,
        eps: float = 1e-12,
        max_update_norm: float = 10.0,
    ):
        super().__init__()

        self.eps = eps

        if max_update_norm <= 0:
            raise ValueError(
                "max_update_norm must be positive"
            )

        self.max_update_norm = max_update_norm

        # Scores from the previous completed round.
        self.previous_scores: Dict[str, float] = {}

        # Client models received in the current round.
        self.client_models: Dict[str, Dict[str, np.ndarray]] = {}

        # Bounded client updates for the current round.
        self.client_updates: Dict[str, Dict[str, np.ndarray]] = {}

        # Client metrics received in the current round.
        self.client_metrics: Dict[str, Dict[str, Any]] = {}

        # Global model at the beginning of the current round.
        self.global_model: Optional[Dict[str, np.ndarray]] = None

        # Current-round FedSCS statistics.
        self.raw_scores: Dict[str, float] = {}
        self.current_scores: Dict[str, float] = {}
        self.current_weights: Dict[str, float] = {}

        self.current_round: Optional[int] = None

    def _flatten_model(
        self,
        model: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Flatten all model parameters into one vector."""
        if not model:
            return np.array([], dtype=np.float64)

        parts = []

        for name in sorted(model.keys()):
            value = np.asarray(
                model[name],
                dtype=np.float64,
            )

            parts.append(
                value.reshape(-1)
            )

        if not parts:
            return np.array([], dtype=np.float64)

        return np.concatenate(parts)

    def _validate_model_schema(
        self,
        model: Dict[str, np.ndarray],
        reference: Dict[str, np.ndarray],
        client_name: str,
    ) -> None:
        """Validate parameter names and shapes."""
        model_keys = set(model.keys())
        reference_keys = set(reference.keys())

        missing = reference_keys - model_keys
        unexpected = model_keys - reference_keys

        if missing or unexpected:
            raise ValueError(
                f"Parameter schema mismatch for client "
                f"{client_name}: "
                f"missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

        for name in sorted(reference_keys):
            model_value = np.asarray(
                model[name]
            )
            reference_value = np.asarray(
                reference[name]
            )

            if model_value.shape != reference_value.shape:
                raise ValueError(
                    f"Parameter shape mismatch for client "
                    f"{client_name}, parameter '{name}': "
                    f"client={model_value.shape}, "
                    f"global={reference_value.shape}"
                )

    def _clip_update(
        self,
        update: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Clip a client update to the configured global L2 norm."""
        if not update:
            raise ValueError(
                "Cannot clip an empty client update."
            )

        flat = self._flatten_model(update)

        if flat.size == 0:
            raise ValueError(
                "Cannot clip an empty client update."
            )

        if not np.isfinite(flat).all():
            raise ValueError(
                "Client update contains non-finite values."
            )

        norm = float(
            np.linalg.norm(flat)
        )

        if not np.isfinite(norm):
            raise ValueError(
                "Client update norm is non-finite."
            )

        if norm <= self.max_update_norm:
            clipped = {
                name: np.asarray(
                    value,
                    dtype=np.float64,
                ).copy()
                for name, value in update.items()
            }
        else:
            scale = self.max_update_norm / norm

            clipped = {
                name: (
                    np.asarray(
                        value,
                        dtype=np.float64,
                    ) * scale
                )
                for name, value in update.items()
            }

        clipped_flat = self._flatten_model(
            clipped
        )

        if not np.isfinite(clipped_flat).all():
            raise ValueError(
                "Clipped client update contains "
                "non-finite values."
            )

        clipped_norm = float(
            np.linalg.norm(clipped_flat)
        )

        if not np.isfinite(clipped_norm):
            raise ValueError(
                "Clipped client update norm is non-finite."
            )

        if clipped_norm > self.max_update_norm * (
            1.0 + 1e-10
        ):
            raise ValueError(
                "Clipped client update exceeds "
                "max_update_norm."
            )

        return clipped

    def _compute_updates(self) -> None:
        """Compute and bound client updates."""
        if self.global_model is None:
            raise RuntimeError(
                "Global model is missing."
            )

        self.client_updates.clear()

        for client, client_model in self.client_models.items():
            self._validate_model_schema(
                client_model,
                self.global_model,
                client,
            )

            update = {}

            for name in sorted(
                self.global_model.keys()
            ):
                client_value = np.asarray(
                    client_model[name],
                    dtype=np.float64,
                )

                global_value = np.asarray(
                    self.global_model[name],
                    dtype=np.float64,
                )

                delta = (
                    client_value - global_value
                )

                if not np.isfinite(delta).all():
                    raise ValueError(
                        f"Non-finite update for client "
                        f"{client}, parameter '{name}'"
                    )

                update[name] = delta

            self.client_updates[client] = (
                self._clip_update(update)
            )

    def _cosine_similarity(
        self,
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        """Compute numerically stable cosine similarity."""
        if first.size == 0 or second.size == 0:
            return 0.0

        if first.shape != second.shape:
            raise ValueError(
                "Cosine similarity vectors must have "
                "the same shape."
            )

        if not np.isfinite(first).all():
            raise ValueError(
                "First vector contains non-finite values."
            )

        if not np.isfinite(second).all():
            raise ValueError(
                "Second vector contains non-finite values."
            )

        first_norm = float(
            np.linalg.norm(first)
        )
        second_norm = float(
            np.linalg.norm(second)
        )

        if (
            not np.isfinite(first_norm)
            or not np.isfinite(second_norm)
        ):
            raise ValueError(
                "Cosine similarity norm is non-finite."
            )

        if (
            first_norm <= self.eps
            or second_norm <= self.eps
        ):
            return 0.0

        similarity = float(
            np.dot(first, second)
            / (
                first_norm
                * second_norm
            )
        )

        if not np.isfinite(similarity):
            return 0.0

        return float(
            np.clip(
                similarity,
                -1.0,
                1.0,
            )
        )

    def _compute_fedscs_scores(self) -> None:
        """Compute FedSCS scores and aggregation weights."""
        clients = list(
            self.client_updates.keys()
        )

        if not clients:
            raise ValueError(
                "No client updates available."
            )

        flattened_updates = {
            client: self._flatten_model(
                self.client_updates[client]
            )
            for client in clients
        }

        self.raw_scores.clear()
        self.current_scores.clear()
        self.current_weights.clear()

        for client in clients:
            update = flattened_updates[client]

            peer = None

            for other_client in clients:
                if other_client == client:
                    continue

                other_update = (
                    flattened_updates[other_client]
                )

                if peer is None:
                    peer = other_update.copy()
                else:
                    peer += other_update

            # With only one client there is no peer update.
            if peer is None:
                similarity = 0.0
            else:
                similarity = (
                    self._cosine_similarity(
                        update,
                        peer,
                    )
                )

            # FedSCS uses only non-negative similarity.
            raw_score = max(
                similarity,
                0.0,
            )

            if not np.isfinite(raw_score):
                raw_score = 0.0

            self.raw_scores[client] = (
                raw_score
            )

        # Temporal stability.
        for client in clients:
            current_score = self.raw_scores[
                client
            ]

            previous_score = self.previous_scores.get(
                client,
                current_score,
            )

            if not np.isfinite(previous_score):
                previous_score = current_score

            # Stability is based on the agreement between
            # current and previous scores.
            if (
                abs(current_score) <= self.eps
                and abs(previous_score) <= self.eps
            ):
                stability = 1.0
            else:
                denominator = max(
                    abs(current_score),
                    abs(previous_score),
                    self.eps,
                )

                stability = 1.0 - (
                    abs(
                        current_score
                        - previous_score
                    )
                    / denominator
                )

                stability = float(
                    np.clip(
                        stability,
                        0.0,
                        1.0,
                    )
                )

            score = (
                current_score
                * stability
            )

            if not np.isfinite(score):
                score = 0.0

            self.current_scores[client] = (
                score
            )

        score_sum = float(
            sum(
                self.current_scores.values()
            )
        )

        if (
            not np.isfinite(score_sum)
            or score_sum <= self.eps
        ):
            # Safe fallback when all FedSCS scores are zero.
            uniform_weight = (
                1.0 / len(clients)
            )

            for client in clients:
                self.current_weights[client] = (
                    uniform_weight
                )
        else:
            for client in clients:
                weight = (
                    self.current_scores[client]
                    / score_sum
                )

                if not np.isfinite(weight):
                    weight = 0.0

                self.current_weights[client] = (
                    weight
                )

        # Normalize once more to protect against
        # floating-point accumulation error.
        weight_sum = float(
            sum(
                self.current_weights.values()
            )
        )

        if (
            not np.isfinite(weight_sum)
            or weight_sum <= self.eps
        ):
            uniform_weight = (
                1.0 / len(clients)
            )

            for client in clients:
                self.current_weights[client] = (
                    uniform_weight
                )
        else:
            for client in clients:
                self.current_weights[client] /= (
                    weight_sum
                )

    def _aggregate_models(
        self,
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate bounded client updates using FedSCS weights.

        Client updates are clipped before FedSCS scoring and the same
        bounded updates are used for the final aggregation. This
        prevents an oversized finite client model from bypassing the
        configured update-norm bound.

        The aggregation is:

            w_global_next =
                w_global
                + sum_i(alpha_i * clipped_delta_i)
        """
        clients = list(
            self.client_updates.keys()
        )

        if not clients:
            return {}

        if self.global_model is None:
            raise RuntimeError(
                "Global model is missing."
            )

        # The global model defines the required
        # parameter schema.
        reference_keys = set(
            self.global_model.keys()
        )

        # Validate every clipped update against
        # the global-model schema.
        for client in clients:
            client_keys = set(
                self.client_updates[client].keys()
            )

            if client_keys != reference_keys:
                raise ValueError(
                    "Parameter schema mismatch for "
                    f"client {client}: "
                    f"missing="
                    f"{sorted(reference_keys - client_keys)}, "
                    f"unexpected="
                    f"{sorted(client_keys - reference_keys)}"
                )

        aggregated = {}

        for name in sorted(
            reference_keys
        ):
            global_value = np.asarray(
                self.global_model[name],
                dtype=np.float64,
            )

            if not np.isfinite(
                global_value
            ).all():
                raise ValueError(
                    "Global model parameter "
                    f"'{name}' contains "
                    "non-finite values."
                )

            update_sum = None

            for client in clients:
                update = np.asarray(
                    self.client_updates[client][
                        name
                    ],
                    dtype=np.float64,
                )

                # Validate parameter shape.
                if (
                    update.shape
                    != global_value.shape
                ):
                    raise ValueError(
                        f"Parameter shape mismatch "
                        f"for '{name}' from client "
                        f"{client}: "
                        f"update={update.shape}, "
                        f"global={global_value.shape}"
                    )

                if not np.isfinite(
                    update
                ).all():
                    raise ValueError(
                        f"Non-finite clipped update "
                        f"for client {client}, "
                        f"parameter '{name}'"
                    )

                weight = self.current_weights.get(
                    client
                )

                if weight is None:
                    raise RuntimeError(
                        f"Missing aggregation weight "
                        f"for client {client}"
                    )

                if not np.isfinite(weight):
                    raise ValueError(
                        f"Non-finite aggregation "
                        f"weight for client {client}"
                    )

                weighted_update = (
                    weight * update
                )

                if not np.isfinite(
                    weighted_update
                ).all():
                    raise ValueError(
                        "Non-finite weighted update "
                        f"for client {client}, "
                        f"parameter '{name}'"
                    )

                if update_sum is None:
                    update_sum = (
                        weighted_update.copy()
                    )
                else:
                    update_sum += (
                        weighted_update
                    )

            if update_sum is None:
                raise RuntimeError(
                    f"Failed to aggregate "
                    f"parameter '{name}'"
                )

            # Apply the weighted bounded update to
            # the round-start global model.
            value = (
                global_value
                + update_sum
            )

            if not np.isfinite(
                value
            ).all():
                raise ValueError(
                    "Non-finite aggregated parameter "
                    f"'{name}'"
                )

            aggregated[name] = value

        return aggregated

    def accept_model(
        self,
        model: FLModel,
    ) -> bool:
        """Accept a client model for the current round."""
        if model is None:
            raise ValueError(
                "Received an empty FLModel."
            )

        params = model.params

        if not params:
            raise ValueError(
                "Received an FLModel with empty parameters."
            )

        client_name = (
            model.meta.get(
                "client_name"
            )
            or model.meta.get(
                "site_name"
            )
            or model.meta.get(
                FLContextKey.IDENTITY_NAME
            )
        )

        if not client_name:
            client_name = (
                model.meta.get(
                    "client"
                )
                or model.meta.get(
                    "site"
                )
            )

        if not client_name:
            raise ValueError(
                "Unable to determine client name "
                "from received FLModel."
            )

        client_name = str(
            client_name
        )

        # Load the global model only once per round.
        if self.global_model is None:
            raise RuntimeError(
                "Global model has not been initialized."
            )

        client_model = {}

        for name, value in params.items():
            array_value = np.asarray(
                value,
                dtype=np.float64,
            )

            if not np.isfinite(
                array_value
            ).all():
                raise ValueError(
                    f"Client {client_name} parameter "
                    f"'{name}' contains "
                    "non-finite values."
                )

            client_model[name] = (
                array_value.copy()
            )

        self._validate_model_schema(
            client_model,
            self.global_model,
            client_name,
        )

        self.client_models[
            client_name
        ] = client_model

        metrics = {}

        if model.metrics:
            for key, value in model.metrics.items():
                if isinstance(
                    value,
                    (int, float, np.integer, np.floating),
                ):
                    numeric_value = float(
                        value
                    )

                    if np.isfinite(
                        numeric_value
                    ):
                        metrics[key] = (
                            numeric_value
                        )

        self.client_metrics[
            client_name
        ] = metrics

        return True

    def load_model(
        self,
        model: FLModel,
    ) -> None:
        """Set the global model at the beginning of a round."""
        if model is None:
            raise ValueError(
                "Global model cannot be None."
            )

        params = model.params

        if not params:
            raise ValueError(
                "Global model has empty parameters."
            )

        global_model = {}

        for name, value in params.items():
            array_value = np.asarray(
                value,
                dtype=np.float64,
            )

            if not np.isfinite(
                array_value
            ).all():
                raise ValueError(
                    f"Global parameter '{name}' "
                    "contains non-finite values."
                )

            global_model[name] = (
                array_value.copy()
            )

        self.global_model = global_model

        self.client_models.clear()
        self.client_updates.clear()
        self.client_metrics.clear()

        self.raw_scores.clear()
        self.current_scores.clear()
        self.current_weights.clear()

        self.current_round = model.meta.get(
            AppConstants.CURRENT_ROUND
        )

    def aggregate_model(
        self,
        fl_ctx: Any = None,
    ) -> Optional[FLModel]:
        """Compute FedSCS scores and aggregate client models."""
        if self.global_model is None:
            raise RuntimeError(
                "Global model is missing."
            )

        if not self.client_models:
            return None

        # Compute bounded client updates.
        self._compute_updates()

        # Compute FedSCS scores and normalized weights
        # from the bounded updates.
        self._compute_fedscs_scores()

        # IMPORTANT:
        # Final aggregation also uses the bounded updates.
        aggregated = self._aggregate_models()

        if not aggregated:
            return None

        # Final safety validation.
        for name, value in aggregated.items():
            if not np.isfinite(
                value
            ).all():
                raise ValueError(
                    "Aggregated model contains "
                    f"non-finite values in '{name}'."
                )

        # Save current scores for temporal stability
        # in the next round.
        self.previous_scores = (
            self.current_scores.copy()
        )

        return FLModel(
            params=aggregated,
            metrics={
                "fedscs_num_clients": len(
                    self.client_models
                ),
                "fedscs_score_mean": float(
                    np.mean(
                        list(
                            self.current_scores.values()
                        )
                    )
                ),
            },
        )

    def reset_stats(self) -> None:
        """Reset current-round state while preserving history."""
        self.client_models.clear()
        self.client_updates.clear()
        self.client_metrics.clear()

        self.global_model = None

        self.raw_scores.clear()
        self.current_scores.clear()
        self.current_weights.clear()

        self.current_round = None
