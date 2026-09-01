"""
FedSCS: Federated Learning with Stable Cosine Similarity.

NVFlare ModelAggregator implementation.

delta_i = w_i - w_global

peer_i = sum_{j != i} delta_j

s_i = max(cos(delta_i, peer_i), 0)

S_i^t = S_i^{t-1} +
        (s_i^t - S_i^{t-1}) / (N_prev + 1)

alpha_i = S_i / sum_j S_j

w_global^{t+1} = sum_i alpha_i w_i
"""

from typing import Any, Dict, Optional

import numpy as np

from nvflare.app_common.abstract.fl_model import (
    FLModel,
    ParamsType,
)
from nvflare.app_common.aggregators.model_aggregator import (
    ModelAggregator,
)
from nvflare.app_common.app_constant import AppConstants


class FedSCSAggregator(ModelAggregator):
    """
    FedSCS aggregation for full model weights.

    Uses NVFlare ModelAggregator interface.

    The official global model update remains controlled by NVFlare.
    This aggregator only computes the aggregated model.
    """

    def __init__(
        self,
        eps: float = 1e-12,
    ):
        super().__init__()

        self.eps = eps

        # --------------------------------------------------------------
        # Persistent across rounds
        # --------------------------------------------------------------

        self.previous_scores: Dict[str, float] = {}

        # --------------------------------------------------------------
        # Current round state
        # --------------------------------------------------------------

        self.client_models: Dict[
            str,
            Dict[str, np.ndarray],
        ] = {}

        self.client_updates: Dict[
            str,
            Dict[str, np.ndarray],
        ] = {}

        self.client_metrics: Dict[
            str,
            Dict[str, float],
        ] = {}

        # --------------------------------------------------------------
        # Round-start global model
        # --------------------------------------------------------------

        self.global_model: Optional[
            Dict[str, np.ndarray]
        ] = None

        # --------------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------------

        self.raw_scores: Dict[str, float] = {}

        self.current_scores: Dict[str, float] = {}

        self.current_weights: Dict[str, float] = {}

        self.current_round: Optional[int] = None


    # ==============================================================
    # Utility functions
    # ==============================================================

    def _to_numpy(
        self,
        value: Any,
    ) -> np.ndarray:
        """
        Convert tensor-like object to numpy.
        """

        if hasattr(value, "detach"):
            value = value.detach()

        if hasattr(value, "cpu"):
            value = value.cpu()

        if hasattr(value, "numpy"):
            value = value.numpy()

        return np.asarray(value)


    def _flatten_model(
        self,
        model: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Convert model parameters to numpy arrays.
        """

        return {
            name: self._to_numpy(value).copy()
            for name, value in model.items()
        }


    def _flatten_update(
        self,
        update: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Flatten update dictionary into one vector.
        """

        parts = []

        for name in sorted(update.keys()):

            value = np.asarray(
                update[name],
                dtype=np.float64,
            )

            parts.append(
                value.reshape(-1)
            )

        if not parts:
            return np.array(
                [],
                dtype=np.float64,
            )

        return np.concatenate(parts)


    def _cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Stable cosine similarity.
        """

        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)

        if (
            a_norm < self.eps
            or b_norm < self.eps
        ):
            return 0.0

        score = np.dot(a, b) / (
            a_norm * b_norm
        )

        if not np.isfinite(score):
            return 0.0

        return float(score)


    # ==============================================================
    # Global model handling
    # ==============================================================

    def _get_current_round(
        self,
    ) -> Optional[int]:
        """
        Get current FL round.
        """

        if self.fl_ctx is None:
            return None

        try:
            return self.fl_ctx.get_prop(
                AppConstants.CURRENT_ROUND
            )

        except Exception:
            return None


    def _load_global_model(
        self,
    ) -> bool:
        """
        Load round-start global model from NVFlare context.
        """

        if self.fl_ctx is None:
            self.warning(
                "FedSCS cannot access FLContext."
            )
            return False


        try:
            global_model = self.fl_ctx.get_prop(
                AppConstants.GLOBAL_MODEL
            )

        except Exception as e:
            self.warning(
                f"Failed reading GLOBAL_MODEL: {e}"
            )
            return False


        if global_model is None:
            self.warning(
                "GLOBAL_MODEL unavailable."
            )
            return False


        # NVFlare model object
        if hasattr(global_model, "weights"):

            weights = global_model.weights

            if weights is not None:

                self.global_model = (
                    self._flatten_model(weights)
                )

                self.info(
                    "FedSCS loaded global model."
                )

                return True


        # Dictionary model
        if isinstance(
            global_model,
            dict,
        ):

            if "weights" in global_model:

                weights = global_model["weights"]

                if isinstance(
                    weights,
                    dict,
                ):

                    self.global_model = (
                        self._flatten_model(weights)
                    )

                    return True


            self.global_model = (
                self._flatten_model(global_model)
            )

            return True


        self.warning(
            "Unable to extract GLOBAL_MODEL."
        )

        return False


    # ==============================================================
    # Client update computation
    # ==============================================================

    def _compute_updates(
        self,
    ) -> None:
        """
        Compute:

            delta_i = client_model - global_model
        """

        if self.global_model is None:
            raise RuntimeError(
                "Global model is missing."
            )


        self.client_updates = {}


        for client, model in self.client_models.items():

            update = {}


            for name, client_value in model.items():

                if name not in self.global_model:

                    raise KeyError(
                        f"{name} missing in global model"
                    )


                client_value = np.asarray(
                    client_value,
                    dtype=np.float64,
                )

                global_value = np.asarray(
                    self.global_model[name],
                    dtype=np.float64,
                )


                update[name] = (
                    client_value
                    -
                    global_value
                )


            self.client_updates[client] = update
    # ==============================================================
    # FedSCS scoring
    # ==============================================================

    def _compute_fedscs_scores(
        self,
    ) -> None:
        """
        Compute FedSCS stability scores.

        Steps:

            1. cosine(delta_i, sum(delta_j))
            2. temporal stability update
            3. normalize weights
        """

        clients = list(
            self.client_updates.keys()
        )

        if not clients:
            self.raw_scores = {}
            self.current_scores = {}
            self.current_weights = {}
            return


        vectors = {
            client: self._flatten_update(
                self.client_updates[client]
            )
            for client in clients
        }


        # ----------------------------------------------------------
        # Raw similarity score
        # ----------------------------------------------------------

        self.raw_scores = {}


        for client in clients:

            own_update = vectors[client]


            peer_updates = [
                vectors[c]
                for c in clients
                if c != client
            ]


            # Single client case

            if not peer_updates:

                self.raw_scores[client] = 1.0
                continue


            peer_sum = np.sum(
                peer_updates,
                axis=0,
            )


            score = self._cosine_similarity(
                own_update,
                peer_sum,
            )


            self.raw_scores[client] = max(
                float(score),
                0.0,
            )



        # ----------------------------------------------------------
        # Temporal stability
        # ----------------------------------------------------------

        self.current_scores = {}


        previous_count = len(
            self.previous_scores
        )


        denominator = max(
            previous_count + 1,
            1,
        )


        for client in clients:


            current_score = (
                self.raw_scores[client]
            )


            if client in self.previous_scores:

                previous = (
                    self.previous_scores[client]
                )

                current_score = (
                    previous
                    +
                    (
                        current_score
                        -
                        previous
                    )
                    /
                    denominator
                )


            self.current_scores[client] = max(
                float(current_score),
                0.0,
            )



        # ----------------------------------------------------------
        # Normalize
        # ----------------------------------------------------------

        total = sum(
            self.current_scores.values()
        )


        if total <= self.eps:

            uniform = (
                1.0 /
                len(clients)
            )

            self.current_weights = {
                c: uniform
                for c in clients
            }


        else:

            self.current_weights = {

                c:
                self.current_scores[c]
                /
                total

                for c in clients
            }



    # ==============================================================
    # Model aggregation
    # ==============================================================

    def _aggregate_models(
        self,
    ) -> Dict[str, np.ndarray]:
        """
        Weighted aggregation using FedSCS weights.
        """

        clients = list(
            self.client_models.keys()
        )

        aggregated = {}


        if not clients:
            return aggregated



        parameters = (
            self.client_models[
                clients[0]
            ].keys()
        )


        for name in parameters:


            value_sum = None


            for client in clients:


                if name not in self.client_models[client]:
                    continue


                value = np.asarray(
                    self.client_models[client][name],
                    dtype=np.float64,
                )


                weighted_value = (
                    self.current_weights[client]
                    *
                    value
                )


                if value_sum is None:

                    value_sum = (
                        weighted_value.copy()
                    )

                else:

                    value_sum += (
                        weighted_value
                    )


            if value_sum is not None:

                aggregated[name] = value_sum


        return aggregated



    # ==============================================================
    # Client identification
    # ==============================================================

    def _extract_client_name(
        self,
        model: FLModel,
    ) -> str:

        if model.meta:

            for key in (
                "client_name",
                "site_name",
                "client",
            ):

                value = model.meta.get(key)

                if value is not None:
                    return str(value)


        return "?"



    # ==============================================================
    # NVFlare ModelAggregator API
    # ==============================================================

    def accept_model(
        self,
        model: FLModel,
    ) -> None:
        """
        Receive one client model.
        """


        if model.params is None:

            self.warning(
                "Received empty client model."
            )

            return



        client_name = (
            self._extract_client_name(model)
        )



        if client_name in self.client_models:

            self.warning(
                f"Duplicate contribution from {client_name}"
            )

            return



        # Load global model once per round

        if not self.client_models:


            self.current_round = (
                self._get_current_round()
            )


            self.global_model = None


            if not self._load_global_model():

                raise RuntimeError(
                    "FedSCS could not load global model."
                )


            self.info(
                f"FedSCS round {self.current_round}: "
                "global model loaded."
            )



        self.client_models[client_name] = (
            self._flatten_model(
                model.params
            )
        )



        # Store client metrics

        if model.metrics:

            self.client_metrics[client_name] = {

                str(k):
                float(v)

                for k, v in model.metrics.items()

                if isinstance(
                    v,
                    (
                        int,
                        float,
                        np.integer,
                        np.floating,
                    )
                )

            }



        self.info(
            f"Accepted FedSCS contribution from {client_name}"
        )



    def aggregate_model(
        self,
    ) -> FLModel:
        """
        Generate global model.
        """


        if not self.client_models:

            raise RuntimeError(
                "No client models received."
            )


        self.info(
            f"FedSCS aggregation: "
            f"{len(self.client_models)} clients "
            f"round={self.current_round}"
        )


        self._compute_updates()


        self._compute_fedscs_scores()



        for client in self.client_models:


            self.info(
                f"FedSCS client={client} "
                f"raw={self.raw_scores[client]:.6f} "
                f"stability={self.current_scores[client]:.6f} "
                f"weight={self.current_weights[client]:.6f}"
            )



        aggregated_model = (
            self._aggregate_models()
        )


        # keep history

        self.previous_scores = dict(
            self.current_scores
        )



        # Aggregate metrics

        aggregated_metrics = {}


        if self.client_metrics:

            metric_names = set()

            for metrics in self.client_metrics.values():

                metric_names.update(
                    metrics.keys()
                )


            for metric in metric_names:

                values = []

                for client, metrics in self.client_metrics.items():

                    if metric in metrics:

                        values.append(
                            metrics[metric]
                        )


                if values:

                    aggregated_metrics[metric] = (
                        float(np.mean(values))
                    )



        return FLModel(

            params=aggregated_model,

            params_type=ParamsType.FULL,

            current_round=self.current_round,


            metrics=(
                aggregated_metrics
                if aggregated_metrics
                else None
            ),


            meta={

                "nr_aggregated":
                    len(self.client_models),


                "current_round":
                    self.current_round,


                "fedscs_weights":
                    dict(self.current_weights),


                "fedscs_raw_scores":
                    dict(self.raw_scores),


                "fedscs_stability_scores":
                    dict(self.current_scores),

            },

        )



    def reset_stats(
        self,
    ) -> None:
        """
        Reset only current round state.

        Keep previous_scores.
        """


        self.client_models = {}

        self.client_updates = {}

        self.client_metrics = {}


        self.raw_scores = {}

        self.current_scores = {}

        self.current_weights = {}


        self.global_model = None

        self.current_round = None



    # ==============================================================
    # Optional external API
    # ==============================================================

    def set_global_model(
        self,
        model: Dict[str, Any],
    ) -> None:

        self.global_model = {

            k:
            self._to_numpy(v).copy()

            for k, v in model.items()

        }



    def get_scores(
        self,
    ) -> Dict[str, float]:

        return dict(
            self.current_scores
        )



    def get_weights(
        self,
    ) -> Dict[str, float]:

        return dict(
            self.current_weights
        )
            
