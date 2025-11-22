# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


"""
Feature Election Controller for NVIDIA FLARE
Implements the Feature Election algorithm from the FLASH framework
"""

import logging
from typing import Dict, Optional

import numpy as np

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather

logger = logging.getLogger(__name__)


class FeatureElectionController(ScatterAndGather):
    """
    Feature Election Controller that aggregates feature selections from multiple clients
    and produces a global feature mask based on weighted voting.
    """

    def __init__(
        self,
        freedom_degree: float = 0.1,
        aggregation_mode: str = "weighted",
        min_clients: int = 2,
        num_rounds: int = 1,
        task_name: str = "feature_election",
        train_timeout: int = 0,
    ):
        """
        Initialize Feature Election Controller

        Args:
            freedom_degree: Parameter controlling feature selection (0=intersection, 1=union)
            aggregation_mode: 'weighted' or 'uniform' aggregation
            min_clients: Minimum number of clients required for election
            num_rounds: Number of election rounds
            task_name: Name of the feature election task
        """
        super().__init__(
            min_clients=min_clients,
            num_rounds=num_rounds,
            start_round=0,
            wait_time_after_min_received=10,
            train_task_name=task_name,
            train_timeout=train_timeout,
        )

        # Validate inputs
        if not 0 <= freedom_degree <= 1:
            raise ValueError("freedom_degree must be between 0 and 1")
        if aggregation_mode not in ["weighted", "uniform"]:
            raise ValueError("aggregation_mode must be 'weighted' or 'uniform'")

        self.freedom_degree = freedom_degree
        self.aggregation_mode = aggregation_mode
        self.custom_task_name = task_name

        # Results storage
        self.global_feature_mask = None
        self.client_scores = {}
        self.num_features = None

    def start_controller(self, fl_ctx: FLContext) -> None:
        """Start the controller"""
        logger.info(f"Starting Feature Election Controller with freedom_degree={self.freedom_degree}")
        super().start_controller(fl_ctx)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        """Main control flow - overrides parent to add custom logging"""
        logger.info("Starting Feature Election workflow")
        super().control_flow(abort_signal, fl_ctx)
        logger.info("Feature Election workflow completed")

    def aggregate(self, fl_ctx: FLContext) -> None:
        """
        Custom aggregation method for feature election
        This is called by the parent ScatterAndGather class
        """
        # Get the aggregator component
        aggregator = self._get_aggregator()
        if aggregator is None:
            self.panic("No aggregator configured!", fl_ctx)
            return

        # Reset for new aggregation round
        self.client_scores = {}

        try:
            # Get client submissions
            aggr_result = aggregator.aggregate(fl_ctx)

            if not aggr_result:
                logger.warning("No aggregation results received")
                return

            # Process the aggregated results
            self._process_aggregated_results(aggr_result, fl_ctx)

        except Exception as e:
            logger.error(f"Error during feature election aggregation: {e}")
            self.panic(f"Aggregation failed: {e}", fl_ctx)

    def _process_aggregated_results(self, aggr_result: Shareable, fl_ctx: FLContext) -> None:
        """Process aggregated results from clients"""
        try:
            # Extract client contributions
            client_data = self._extract_client_data(aggr_result)

            if not client_data:
                logger.warning("No valid client data extracted")
                return

            # Run feature election algorithm
            self.global_feature_mask = self._aggregate_selections(client_data)

            # Store results in FLContext for persistence
            fl_ctx.set_prop("global_feature_mask", self.global_feature_mask.tolist())
            fl_ctx.set_prop("feature_election_results", self.get_results())

            logger.info(f"Feature election completed: {np.sum(self.global_feature_mask)} features selected")

        except Exception as e:
            logger.error(f"Error processing aggregated results: {e}")
            raise

    def _extract_client_data(self, aggr_result: Shareable) -> Dict[str, Dict]:
        """Extract client data from aggregation result"""
        client_data = {}

        # The aggregator result should contain contributions from all clients
        # This is a simplified extraction - you may need to adjust based on your aggregator implementation

        # Look for client contributions in the shareable
        for key in aggr_result.keys():
            if key.startswith("client_"):
                client_name = key.replace("client_", "")
                client_contrib = aggr_result.get(key)

                if self._validate_selection(client_contrib):
                    client_data[client_name] = {
                        "selected_features": np.array(client_contrib.get("selected_features")),
                        "feature_scores": np.array(client_contrib.get("feature_scores")),
                        "num_samples": client_contrib.get("num_samples", 1),
                        "initial_score": client_contrib.get("initial_score", 0),
                        "fs_score": client_contrib.get("fs_score", 0),
                    }

        logger.info(f"Extracted data from {len(client_data)} clients")
        return client_data

    def _validate_selection(self, selection_data: Dict) -> bool:
        """Validate client selection data"""
        if not selection_data:
            return False

        required_keys = ["selected_features", "feature_scores"]

        # Check required keys
        for key in required_keys:
            if key not in selection_data or selection_data[key] is None:
                return False

        # Validate array dimensions
        try:
            selected = np.array(selection_data["selected_features"])
            scores = np.array(selection_data["feature_scores"])

            if len(selected) != len(scores):
                return False

            # Set num_features on first valid response
            if self.num_features is None:
                self.num_features = len(selected)
            elif len(selected) != self.num_features:
                return False

        except Exception as e:
            logger.warning(f"Error validating selection data: {e}")
            return False

        return True

    def _aggregate_selections(self, client_selections: Dict[str, Dict]) -> np.ndarray:
        """
        Core Feature Election algorithm implementation

        Args:
            client_selections: Dictionary of client selection data

        Returns:
            Global feature mask (binary array)
        """
        num_clients = len(client_selections)
        logger.info(f"Aggregating selections from {num_clients} clients")

        # Convert to numpy arrays
        masks = []
        scores = []
        weights = []
        total_samples = 0

        for client_name, selection in client_selections.items():
            masks.append(selection["selected_features"])
            scores.append(selection["feature_scores"])
            num_samples = selection["num_samples"]
            weights.append(num_samples)
            total_samples += num_samples

            # Store client scores
            self.client_scores[client_name] = {
                "initial_score": selection.get("initial_score", 0),
                "fs_score": selection.get("fs_score", 0),
                "num_features": int(np.sum(selection["selected_features"])),
                "num_samples": num_samples,
            }

            # Log client statistics
            logger.info(f"Client {client_name}: {np.sum(masks[-1])} features selected, " f"{num_samples} samples")

        masks = np.array(masks)
        scores = np.array(scores)
        weights = np.array(weights) / total_samples if total_samples > 0 else np.ones(len(weights)) / len(weights)

        # Calculate intersection and union
        intersection_mask = self._get_intersection(masks)
        union_mask = self._get_union(masks)

        logger.info(f"Intersection: {np.sum(intersection_mask)} features")
        logger.info(f"Union: {np.sum(union_mask)} features")

        # Handle edge cases
        if self.freedom_degree == 0:
            global_mask = intersection_mask
        elif self.freedom_degree == 1:
            global_mask = union_mask
        else:
            # Main algorithm: select from difference set based on weighted voting
            global_mask = self._weighted_election(masks, scores, weights, intersection_mask, union_mask)

        logger.info(f"Global mask: {np.sum(global_mask)} features selected")

        return global_mask

    def _weighted_election(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        weights: np.ndarray,
        intersection_mask: np.ndarray,
        union_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Perform weighted election for features in (union - intersection)
        """
        # Get difference set
        difference_mask = union_mask & ~intersection_mask

        if not np.any(difference_mask):
            # No features in difference, return intersection
            return intersection_mask

        # Scale scores and apply weights
        scaled_scores = np.zeros_like(scores)

        for i, (client_mask, client_scores) in enumerate(zip(masks, scores)):
            # Scale selected features to [0, 1]
            selected = client_mask.astype(bool)

            if np.any(selected):
                selected_scores = client_scores[selected]
                if len(selected_scores) > 0:
                    min_score = np.min(selected_scores)
                    max_score = np.max(selected_scores)
                    range_score = max_score - min_score

                    if range_score > 0:
                        scaled_scores[i][selected] = (client_scores[selected] - min_score) / range_score
                    else:
                        scaled_scores[i][selected] = 1.0

            # Zero out intersection features (they're already selected)
            scaled_scores[i][intersection_mask] = 0.0

            # Apply client weight if in weighted mode
            if self.aggregation_mode == "weighted":
                scaled_scores[i] *= weights[i]

        # Aggregate scores across clients
        aggregated_scores = np.sum(scaled_scores, axis=0)

        # Select top features from difference set based on freedom_degree
        n_additional = int(np.ceil(np.sum(difference_mask) * self.freedom_degree))

        if n_additional > 0:
            diff_indices = np.where(difference_mask)[0]
            diff_scores = aggregated_scores[difference_mask]

            if len(diff_scores) > 0:
                # Partition index is k, number of features to select is -k
                k = -min(n_additional, len(diff_scores))
                # Get indices of top scoring features
                top_indices = np.argpartition(diff_scores, k)
                top_indices = top_indices[k:]

                # Create selected difference mask
                selected_difference = np.zeros_like(difference_mask)
                selected_difference[diff_indices[top_indices]] = True

                # Combine with intersection
                global_mask = intersection_mask | selected_difference
            else:
                global_mask = intersection_mask
        else:
            global_mask = intersection_mask

        return global_mask

    def _get_aggregator(self) -> Optional[Aggregator]:
        """Get the aggregator component"""
        return self.aggregator

    @staticmethod
    def _get_intersection(masks: np.ndarray) -> np.ndarray:
        """Get intersection of all feature masks"""
        return np.all(masks, axis=0)

    @staticmethod
    def _get_union(masks: np.ndarray) -> np.ndarray:
        """Get union of all feature masks"""
        return np.any(masks, axis=0)

    def get_results(self) -> Dict:
        """Get feature election results"""
        return {
            "global_feature_mask": self.global_feature_mask.tolist() if self.global_feature_mask is not None else None,
            "num_features_selected": (
                int(np.sum(self.global_feature_mask)) if self.global_feature_mask is not None else 0
            ),
            "freedom_degree": self.freedom_degree,
            "aggregation_mode": self.aggregation_mode,
            "client_scores": self.client_scores,
            "total_clients": len(self.client_scores),
        }
