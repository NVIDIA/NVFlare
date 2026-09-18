# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from typing import Dict, Optional, Tuple

import numpy as np

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.aggregators.model_aggregator import ModelAggregator
from nvflare.app_common.app_constant import AppConstants


class FedSCSAggregator(ModelAggregator):
    """Aggregate client DIFF updates using Stable Cosine Similarity."""

    def __init__(
        self,
        expected_schema: Dict[str, Tuple[int, ...]],
        max_update_norm: Optional[float] = 10.0,
        eps: float = 1e-12,
    ):
        super().__init__()

        if not expected_schema:
            raise ValueError("expected_schema must not be empty.")

        if max_update_norm is not None and (not np.isfinite(max_update_norm) or max_update_norm <= 0.0):
            raise ValueError("max_update_norm must be positive and finite.")

        if not np.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be positive and finite.")

        self.expected_schema = {name: tuple(shape) for name, shape in expected_schema.items()}

        self.max_update_norm = float(max_update_norm) if max_update_norm is not None else None
        self.eps = float(eps)

        self.client_updates: Dict[str, Dict[str, np.ndarray]] = {}

        # Historical score s_i^(t-1).
        self.previous_scores: Dict[str, float] = {}

        # Current s_i^(t), c_i^(t), and a_i^(t).
        self.current_scores: Dict[str, float] = {}
        self.current_scs: Dict[str, float] = {}
        self.current_weights: Dict[str, float] = {}

        # FedSCS uses one-based mathematical round numbering.
        self.round_number = 0

        self._params_type: Optional[ParamsType] = None

        # Preserve the original parameter dtype while using float64
        # internally for numerically safer aggregation.
        self._input_dtypes: Dict[str, np.dtype] = {}

    @staticmethod
    def _flatten_update(
        update: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Flatten parameters into a deterministic float64 vector."""
        if not update:
            return np.empty(0, dtype=np.float64)

        try:
            with np.errstate(over="raise", invalid="raise"):
                flattened = np.concatenate(
                    [
                        np.asarray(
                            update[name],
                            dtype=np.float64,
                        ).reshape(-1)
                        for name in sorted(update)
                    ]
                )
        except (FloatingPointError, TypeError, ValueError) as e:
            raise ValueError(
                "Non-finite or overflowing value encountered while " "flattening the client update."
            ) from e

        if not np.all(np.isfinite(flattened)):
            raise ValueError("Client update contains non-finite values after flattening.")

        return flattened

    def _validate_update(
        self,
        update: Dict[str, np.ndarray],
        client_name: str,
    ) -> None:
        """Validate parameter values received from a client."""
        if not update:
            raise ValueError(f"Client '{client_name}' provided an empty update.")

        for name, value in update.items():
            try:
                array = np.asarray(value)
            except Exception as e:
                raise ValueError(
                    f"Unable to convert client '{client_name}' parameter " f"'{name}' to a NumPy array."
                ) from e

            if not np.issubdtype(array.dtype, np.number):
                raise ValueError(f"Client '{client_name}' parameter '{name}' " "must have a numeric dtype.")

            try:
                finite = np.isfinite(array).all()
            except TypeError as e:
                raise ValueError(
                    f"Unable to validate client '{client_name}' parameter " f"'{name}' for finite values."
                ) from e

            if not finite:
                raise ValueError(f"Client '{client_name}' parameter '{name}' " "contains non-finite values.")

    def _validate_schema(
        self,
        update: Dict[str, np.ndarray],
        client_name: str,
    ) -> None:
        """Validate a client update against the authoritative schema."""
        expected_keys = set(self.expected_schema)
        received_keys = set(update)

        if received_keys != expected_keys:
            missing = sorted(expected_keys - received_keys)
            extra = sorted(received_keys - expected_keys)

            raise ValueError(
                f"Parameter schema mismatch for client '{client_name}'. " f"Missing: {missing}; Extra: {extra}."
            )

        for name, expected_shape in self.expected_schema.items():
            actual_shape = np.asarray(update[name]).shape

            if actual_shape != expected_shape:
                raise ValueError(
                    f"Parameter '{name}' for client '{client_name}' "
                    f"has shape {actual_shape}; expected {expected_shape}."
                )

    def _bound_update_norm(
        self,
        update: Dict[str, np.ndarray],
        client_name: str,
    ) -> Dict[str, np.ndarray]:
        """Bound the L2 norm of a received DIFF update.

        This is an implementation-level safety bound. It does not alter
        the FedSCS cosine similarity because positive scalar rescaling
        preserves cosine similarity.
        """
        if self.max_update_norm is None:
            return {name: np.asarray(value).copy() for name, value in update.items()}

        vector = self._flatten_update(update)

        try:
            with np.errstate(over="raise", invalid="raise"):
                update_norm = np.linalg.norm(vector)
        except FloatingPointError as e:
            raise ValueError(f"Overflow while computing update norm for client " f"'{client_name}'.") from e

        if not np.isfinite(update_norm):
            raise ValueError(f"Client '{client_name}' update norm is non-finite.")

        if update_norm <= self.max_update_norm:
            return {name: np.asarray(value).copy() for name, value in update.items()}

        scale = self.max_update_norm / update_norm

        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid update scaling factor for client " f"'{client_name}'.")

        bounded_update = {}

        for name, value in update.items():
            array = np.asarray(value)

            try:
                with np.errstate(
                    over="raise",
                    invalid="raise",
                    under="ignore",
                ):
                    bounded_value = array * scale
            except FloatingPointError as e:
                raise ValueError(f"Overflow while bounding parameter '{name}' " f"for client '{client_name}'.") from e

            if not np.all(np.isfinite(bounded_value)):
                raise ValueError(
                    f"Bounded parameter '{name}' for client " f"'{client_name}' contains non-finite values."
                )

            bounded_update[name] = bounded_value

        return bounded_update

    def _compute_cosine_similarity(
        self,
        update: Dict[str, np.ndarray],
        peer_sum: Dict[str, np.ndarray],
    ) -> float:
        """Compute rho_i^(t) from the FedSCS formulation."""
        update_vector = self._flatten_update(update)
        peer_vector = self._flatten_update(peer_sum)

        try:
            with np.errstate(over="raise", invalid="raise"):
                update_norm = np.linalg.norm(update_vector)
                peer_norm = np.linalg.norm(peer_vector)
        except FloatingPointError as e:
            raise ValueError("Overflow or invalid arithmetic while computing " "cosine-similarity norms.") from e

        if not np.isfinite(update_norm) or not np.isfinite(peer_norm):
            raise ValueError("Non-finite norm encountered while computing " "cosine similarity.")

        if update_norm <= self.eps or peer_norm <= self.eps:
            return 0.0

        try:
            with np.errstate(
                over="raise",
                invalid="raise",
                divide="raise",
            ):
                dot_product = np.dot(update_vector, peer_vector)
                denominator = update_norm * peer_norm
                cosine = dot_product / denominator
        except FloatingPointError as e:
            raise ValueError("Overflow or non-finite value encountered while " "computing cosine similarity.") from e

        if not np.isfinite(dot_product):
            raise ValueError("Cosine similarity dot product is non-finite.")

        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("Cosine similarity denominator is non-finite " "or non-positive.")

        if not np.isfinite(cosine):
            raise ValueError("Cosine similarity computation produced " "a non-finite value.")

        # rho_i^(t) = max(cosine, 0).
        return float(max(cosine, 0.0))

    def _compute_scs(self) -> None:
        """Compute s_i^(t), nu_i^(t), and c_i^(t)."""
        client_names = list(self.client_updates)

        if not client_names:
            raise RuntimeError("No client updates received.")

        if len(client_names) < 2:
            raise RuntimeError("FedSCS requires at least two participating clients.")

        # NVFLARE uses zero-based round numbering internally.
        # FedSCS uses one-based mathematical round numbering.
        current_round = self.fl_ctx.get_prop(AppConstants.CURRENT_ROUND)

        if current_round is None:
            raise RuntimeError("Current FL round is not available in the " "aggregator context.")

        self.round_number = int(current_round) + 1
        t = self.round_number

        # The expected schema comes from the server-side model
        # architecture, not from any client submission.
        total_update = {name: np.zeros(shape, dtype=np.float64) for name, shape in self.expected_schema.items()}

        # Compute sum_j Delta w_j^(t).
        for client_name in client_names:
            update = self.client_updates[client_name]

            self._validate_schema(update, client_name)

            for name in self.expected_schema:
                try:
                    with np.errstate(
                        over="raise",
                        invalid="raise",
                    ):
                        total_update[name] += np.asarray(
                            update[name],
                            dtype=np.float64,
                        )
                except FloatingPointError as e:
                    raise ValueError(f"Overflow or invalid arithmetic while " f"aggregating parameter '{name}'.") from e

                if not np.all(np.isfinite(total_update[name])):
                    raise ValueError(f"Aggregated parameter '{name}' became " "non-finite.")

        current_scores = {}
        current_scs = {}

        for client_name in client_names:
            update = self.client_updates[client_name]

            # g_i^(t) = sum_{j != i} Delta w_j^(t).
            peer_sum = {}

            for name in self.expected_schema:
                try:
                    with np.errstate(
                        over="raise",
                        invalid="raise",
                    ):
                        peer_sum[name] = total_update[name] - np.asarray(
                            update[name],
                            dtype=np.float64,
                        )
                except FloatingPointError as e:
                    raise ValueError(
                        f"Overflow or invalid arithmetic while " f"computing peer consensus for parameter " f"'{name}'."
                    ) from e

                if not np.all(np.isfinite(peer_sum[name])):
                    raise ValueError(f"Peer consensus for parameter '{name}' " "became non-finite.")

            rho = self._compute_cosine_similarity(
                update,
                peer_sum,
            )

            # The paper defines s_i^(0) = 1.
            previous = self.previous_scores.get(
                client_name,
                1.0,
            )

            if not np.isfinite(previous):
                raise ValueError(f"Previous FedSCS score for client " f"'{client_name}' is non-finite.")

            # s_i^(t) =
            # ((t-1)/t) s_i^(t-1) + (1/t) rho_i^(t).
            try:
                with np.errstate(
                    over="raise",
                    invalid="raise",
                ):
                    score = ((t - 1.0) / t) * previous + (1.0 / t) * rho
            except FloatingPointError as e:
                raise ValueError(f"Overflow while computing FedSCS score for " f"client '{client_name}'.") from e

            if not np.isfinite(score):
                raise ValueError(f"FedSCS score for client '{client_name}' " "is non-finite.")

            # nu_i^(t) =
            # |(s_i^(t)-s_i^(t-1))/(s_i^(t-1)+epsilon)|.
            try:
                with np.errstate(
                    over="raise",
                    invalid="raise",
                    divide="raise",
                ):
                    volatility = abs((score - previous) / (previous + self.eps))
            except FloatingPointError as e:
                raise ValueError(
                    f"Overflow or invalid arithmetic while "
                    f"computing FedSCS volatility for client "
                    f"'{client_name}'."
                ) from e

            if not np.isfinite(volatility):
                raise ValueError(f"FedSCS volatility for client " f"'{client_name}' is non-finite.")

            # c_i^(t) = s_i^(t)/(1 + nu_i^(t)).
            try:
                with np.errstate(
                    over="raise",
                    invalid="raise",
                    divide="raise",
                ):
                    scs = score / (1.0 + volatility)
            except FloatingPointError as e:
                raise ValueError(
                    f"Overflow or invalid arithmetic while " f"computing FedSCS score for client " f"'{client_name}'."
                ) from e

            if not np.isfinite(scs):
                raise ValueError(f"Stable Cosine Similarity for client " f"'{client_name}' is non-finite.")

            current_scores[client_name] = float(score)
            current_scs[client_name] = float(scs)

        total_scs = float(sum(current_scs.values()))

        if not np.isfinite(total_scs):
            raise RuntimeError("FedSCS produced a non-finite total trust score.")

        self.current_scores = current_scores
        self.current_scs = current_scs

        # If every client has zero trust, use uniform weights rather
        # than aborting the entire FL job.
        if total_scs <= self.eps:
            self.warning("FedSCS produced zero total trust score; " "falling back to uniform aggregation weights.")

            uniform_weight = 1.0 / len(client_names)

            self.current_weights = {client_name: uniform_weight for client_name in client_names}
        else:
            self.current_weights = {client_name: scs / total_scs for client_name, scs in current_scs.items()}

        if not all(np.isfinite(weight) for weight in self.current_weights.values()):
            raise RuntimeError("FedSCS produced non-finite aggregation weights.")

        self.info(
            f"FedSCS Round {self.round_number}: "
            f"{len(client_names)} clients, "
            f"weight_sum="
            f"{sum(self.current_weights.values()):.6f}"
        )

        self.info("FedSCS client scores:")

        for client_name in sorted(self.current_scores):
            self.info(
                f"  {client_name}: "
                f"score={self.current_scores[client_name]:.6f}, "
                f"scs={self.current_scs[client_name]:.6f}, "
                f"weight={self.current_weights[client_name]:.6f}"
            )

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

        # Validate against the authoritative server-side schema.
        # The first client is never trusted to define the schema.
        self._validate_schema(
            model.params,
            client_name,
        )

        # Record the original dtype for each parameter so that internal
        # float64 accumulation can be converted back before returning.
        for name, value in model.params.items():
            array = np.asarray(value)

            if name not in self._input_dtypes:
                self._input_dtypes[name] = array.dtype
            elif array.dtype != self._input_dtypes[name]:
                raise ValueError(
                    f"Parameter '{name}' for client '{client_name}' "
                    f"has dtype {array.dtype}; expected "
                    f"{self._input_dtypes[name]}."
                )

        if self._params_type is None:
            self._params_type = model.params_type
        elif model.params_type != self._params_type:
            raise ValueError("All client updates must use ParamsType.DIFF.")

        # Apply a defense-in-depth magnitude bound before storing
        # the update. Positive scalar rescaling preserves cosine
        # similarity.
        bounded_update = self._bound_update_norm(
            model.params,
            client_name,
        )

        self.client_updates[client_name] = bounded_update

        return True

    def aggregate_model(self) -> FLModel:
        """Return the FedSCS weighted DIFF."""
        if not self.client_updates:
            raise RuntimeError("No client updates received.")

        self._compute_scs()

        client_names = list(self.client_updates)

        # Construct the output strictly from the authoritative schema.
        # Accumulate in float64 for numerical safety.
        aggregated_delta = {
            name: np.zeros(
                shape,
                dtype=np.float64,
            )
            for name, shape in self.expected_schema.items()
        }

        for client_name in client_names:
            weight = self.current_weights[client_name]
            update = self.client_updates[client_name]

            if not np.isfinite(weight):
                raise RuntimeError(f"Non-finite aggregation weight for client " f"'{client_name}'.")

            self._validate_schema(
                update,
                client_name,
            )

            for name in self.expected_schema:
                try:
                    with np.errstate(
                        over="raise",
                        invalid="raise",
                    ):
                        aggregated_delta[name] += weight * np.asarray(
                            update[name],
                            dtype=np.float64,
                        )
                except FloatingPointError as e:
                    raise ValueError(
                        f"Overflow or invalid arithmetic while " f"computing aggregated parameter '{name}'."
                    ) from e

                if not np.all(np.isfinite(aggregated_delta[name])):
                    raise ValueError(f"Final aggregated parameter '{name}' " "is non-finite.")

        # Final defense-in-depth validation before returning the DIFF.
        for name, value in aggregated_delta.items():
            if not np.all(np.isfinite(value)):
                raise RuntimeError(f"Final aggregated DIFF parameter '{name}' " "contains non-finite values.")

        # Restore the original parameter dtype. This keeps the
        # server-side model in its original dtype (e.g., float32)
        # while preserving float64 accumulation internally.
        for name in aggregated_delta:
            if name not in self._input_dtypes:
                raise RuntimeError(f"Missing input dtype for parameter '{name}'.")

            aggregated_delta[name] = aggregated_delta[name].astype(
                self._input_dtypes[name],
                copy=False,
            )

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
        """Reset per-round state while retaining historical scores."""
        if self.current_scores:
            self.previous_scores = dict(self.current_scores)

        self.client_updates.clear()
        self.current_scores.clear()
        self.current_scs.clear()
        self.current_weights.clear()
        self._params_type = None
