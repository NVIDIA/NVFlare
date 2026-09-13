# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""FedSCS model aggregator for NVIDIA FLARE."""

from typing import Dict, Optional

import numpy as np

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator


class FedSCSAggregator(ModelAggregator):
    """Aggregate client DIFF updates using Stable Cosine Similarity."""

    def __init__(self, eps: float = 1e-12):
        super().__init__()

        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.eps = eps

        self.client_updates: Dict[str, Dict[str, np.ndarray]] = {}

        # s_i^(t-1)
        self.previous_scores: Dict[str, float] = {}

        # Current s_i^(t), c_i^(t), and a_i^(t)
        self.current_scores: Dict[str, float] = {}
        self.current_scs: Dict[str, float] = {}
        self.current_weights: Dict[str, float] = {}

        self.round_number = 0
        self._params_type: Optional[ParamsType] = None

    @staticmethod
    def _flatten_update(update: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten parameters into a deterministic vector."""
        if not update:
            return np.empty(0, dtype=np.float64)

        return np.concatenate([np.asarray(update[name], dtype=np.float64).reshape(-1) for name in sorted(update)])

    def _validate_update(
        self,
        update: Dict[str, np.ndarray],
        client_name: str,
    ) -> None:
        """Validate a client update."""
        if not update:
            raise ValueError(f"Client '{client_name}' provided an empty update.")

        for name, value in update.items():
            array = np.asarray(value)

            if not np.isfinite(array).all():
                raise ValueError(f"Client '{client_name}' parameter '{name}' " "contains non-finite values.")

    def _validate_schema(
        self,
        update: Dict[str, np.ndarray],
        reference: Dict[str, np.ndarray],
        client_name: str,
    ) -> None:
        """Ensure all client updates have the same parameter schema."""
        if set(update) != set(reference):
            missing = sorted(set(reference) - set(update))
            extra = sorted(set(update) - set(reference))

            raise ValueError(
                f"Parameter schema mismatch for client '{client_name}'. " f"Missing: {missing}; Extra: {extra}."
            )

        for name in reference:
            shape = np.asarray(update[name]).shape
            reference_shape = np.asarray(reference[name]).shape

            if shape != reference_shape:
                raise ValueError(
                    f"Parameter '{name}' for client '{client_name}' " f"has shape {shape}; expected {reference_shape}."
                )

    def _compute_cosine_similarity(
        self,
        update: Dict[str, np.ndarray],
        peer_sum: Dict[str, np.ndarray],
    ) -> float:
        """Compute rho_i^(t) from the FedSCS formulation."""
        update_vector = self._flatten_update(update)
        peer_vector = self._flatten_update(peer_sum)

        update_norm = np.linalg.norm(update_vector)
        peer_norm = np.linalg.norm(peer_vector)

        if update_norm <= self.eps or peer_norm <= self.eps:
            return 0.0

        cosine = np.dot(update_vector, peer_vector) / (update_norm * peer_norm)

        # rho_i^(t) = max(cosine, 0)
        return float(max(cosine, 0.0))

    def _compute_scs(self) -> None:
        """Compute s_i^(t), nu_i^(t), and c_i^(t)."""
        client_names = list(self.client_updates)

        if not client_names:
            raise RuntimeError("No client updates received.")

        self.round_number += 1
        t = self.round_number

        reference = self.client_updates[client_names[0]]

        # Compute sum_j Delta w_j^(t).
        total_update = {
            name: np.zeros_like(
                np.asarray(value),
                dtype=np.float64,
            )
            for name, value in reference.items()
        }

        for client_name in client_names:
            update = self.client_updates[client_name]

            self._validate_schema(
                update,
                reference,
                client_name,
            )

            for name, value in update.items():
                total_update[name] += np.asarray(
                    value,
                    dtype=np.float64,
                )

        current_scores = {}
        current_scs = {}

        for client_name in client_names:
            update = self.client_updates[client_name]

            # g_i^(t) = sum_{j != i} Delta w_j^(t)
            peer_sum = {
                name: total_update[name] - np.asarray(value, dtype=np.float64) for name, value in update.items()
            }

            rho = self._compute_cosine_similarity(
                update,
                peer_sum,
            )

            # The paper defines s_i^(0) = 1.
            previous = self.previous_scores.get(
                client_name,
                1.0,
            )

            # s_i^(t) =
            # ((t-1)/t) s_i^(t-1) + (1/t) rho_i^(t)
            score = ((t - 1.0) / t) * previous + (1.0 / t) * rho

            # nu_i^(t) =
            # |(s_i^(t)-s_i^(t-1))/(s_i^(t-1)+epsilon)|
            volatility = abs((score - previous) / (previous + self.eps))

            # c_i^(t) = s_i^(t)/(1 + nu_i^(t))
            scs = score / (1.0 + volatility)

            current_scores[client_name] = float(score)
            current_scs[client_name] = float(scs)

        total_scs = sum(current_scs.values())

        if total_scs <= self.eps:
            raise RuntimeError("FedSCS produced zero total trust score; " "unable to compute aggregation weights.")

        self.current_scores = current_scores
        self.current_scs = current_scs
        self.current_weights = {client_name: scs / total_scs for client_name, scs in current_scs.items()}

        # ---------------------------------------------------------------
        # Round-wise FedSCS server logging.
        # ---------------------------------------------------------------

        print("")
        print("=" * 70)
        print(f"FedSCS Round {self.round_number}")
        print("=" * 70)

        print("Client Scores:")
        for client_name in sorted(self.current_scores):
            print(f"  {client_name}: " f"{self.current_scores[client_name]:.6f}")

        print("Stable Cosine Similarity:")
        for client_name in sorted(self.current_scs):
            print(f"  {client_name}: " f"{self.current_scs[client_name]:.6f}")

        print("Client Weights:")
        for client_name in sorted(self.current_weights):
            print(f"  {client_name}: " f"{self.current_weights[client_name]:.6f}")

        print(f"Weight sum: " f"{sum(self.current_weights.values()):.6f}")

        print("=" * 70)

    def accept_model(self, model: FLModel) -> bool:
        """Accept one client DIFF update."""
        if model is None:
            raise ValueError("model must not be None.")

        if model.params is None:
            raise ValueError("Received client model has no parameters.")

        if model.params_type != ParamsType.DIFF:
            raise ValueError("FedSCSAggregator expects ParamsType.DIFF updates, " f"got {model.params_type}.")

        if not isinstance(model.params, dict):
            raise TypeError("Client parameters must be provided as a dictionary.")

        client_name = None

        if model.meta:
            client_name = model.meta.get("client_name")

            if client_name is None:
                client_name = model.meta.get("client_id")

        if not client_name:
            raise ValueError("Client model metadata must contain " "'client_name' or 'client_id'.")

        client_name = str(client_name)

        self._validate_update(
            model.params,
            client_name,
        )

        if self._params_type is None:
            self._params_type = model.params_type
        elif model.params_type != self._params_type:
            raise ValueError("All client updates must use ParamsType.DIFF.")

        if self.client_updates:
            reference = next(iter(self.client_updates.values()))

            self._validate_schema(
                model.params,
                reference,
                client_name,
            )

        self.client_updates[client_name] = {name: np.asarray(value).copy() for name, value in model.params.items()}

        return True

    def aggregate_model(self) -> FLModel:
        """Return the FedSCS weighted DIFF."""
        if not self.client_updates:
            raise RuntimeError("No client updates received.")

        self._compute_scs()

        client_names = list(self.client_updates)
        reference = self.client_updates[client_names[0]]

        aggregated_delta = {
            name: np.zeros_like(
                np.asarray(value),
                dtype=np.float64,
            )
            for name, value in reference.items()
        }

        for client_name in client_names:
            weight = self.current_weights[client_name]
            update = self.client_updates[client_name]

            for name, value in update.items():
                aggregated_delta[name] += weight * np.asarray(value, dtype=np.float64)

        return FLModel(
            params=aggregated_delta,
            params_type=ParamsType.DIFF,
            metrics={
                "fedscs_scores": dict(self.current_scores),
                "fedscs_scs": dict(self.current_scs),
                "fedscs_weights": dict(self.current_weights),
                "fedscs_round": self.round_number,
                "fedscs_num_clients": len(client_names),
            },
        )

    def reset_stats(self) -> None:
        """Reset per-round state while retaining historical trust scores."""
        if self.current_scores:
            self.previous_scores = dict(self.current_scores)

        self.client_updates.clear()
        self.current_scores.clear()
        self.current_scs.clear()
        self.current_weights.clear()
        self._params_type = None
