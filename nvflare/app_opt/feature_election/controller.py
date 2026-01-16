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

import logging
from typing import Dict

import numpy as np

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller, Task
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

logger = logging.getLogger(__name__)


class FeatureElectionController(Controller):
    """
    Advanced controller that performs Feature Election, Auto-tuning, and downstream Training.
    Inherits directly from base Controller for full workflow control.
    """

    def __init__(
        self,
        freedom_degree: float = 0.5,
        aggregation_mode: str = "weighted",
        min_clients: int = 2,
        num_rounds: int = 5,
        task_name: str = "feature_election",
        train_timeout: int = 300,
        auto_tune: bool = False,
        tuning_rounds: int = 0,
    ):
        super().__init__()

        # Configuration
        self.freedom_degree = freedom_degree
        self.aggregation_mode = aggregation_mode
        self.custom_task_name = task_name
        self.min_clients = min_clients
        self.fl_rounds = num_rounds
        self.train_timeout = train_timeout
        self.auto_tune = auto_tune
        self.tuning_rounds = tuning_rounds if auto_tune else 0

        # State
        self.global_feature_mask = None
        self.global_weights = None
        self.cached_client_selections = {}
        self.phase_results = {}

        # Hill Climbing for auto-tuning
        self.tuning_history = []
        self.search_step = 0.1
        self.current_direction = 1
        self.current_tuning_score = 0.0

        self.n_features = None

    def start_controller(self, fl_ctx: FLContext) -> None:
        logger.info("Initializing FeatureElectionController (Base Controller Mode)")

    def stop_controller(self, fl_ctx: FLContext) -> None:
        logger.info("Stopping Feature Election Controller")

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        """
        Called when a result is received for an unknown task.
        This is a fallback - normally results come through task_done_cb.
        """
        logger.warning(f"Received result for unknown task '{task_name}' from {client.name}")

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        """Main Orchestration Loop"""
        try:
            # --- PHASE 1: LOCAL FEATURE SELECTION (ELECTION) ---
            if not self._phase_one_election(abort_signal, fl_ctx):
                return

            # --- PHASE 2: TUNING & GLOBAL MASKING ---
            self._phase_two_tuning_and_masking(abort_signal, fl_ctx)

            # --- PHASE 3: AGGREGATION ROUNDS (FL TRAINING) ---
            self._phase_three_aggregation(abort_signal, fl_ctx)

            logger.info("Feature Election Workflow Completed Successfully.")

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            import traceback

            traceback.print_exc()

    # ==============================================================================
    # PHASE IMPLEMENTATIONS
    # ==============================================================================

    def _result_received_cb(self, client_task: ClientTask, fl_ctx: FLContext):
        """
        Callback called when a result is received from a client.
        This is the proper way to collect results in NVFLARE.
        """
        client_name = client_task.client.name
        result = client_task.result

        if result is None:
            logger.warning(f"No result from client {client_name}")
            return

        rc = result.get_return_code()
        if rc != ReturnCode.OK:
            logger.warning(f"Client {client_name} returned error: {rc}")
            return

        # Store the result
        self.phase_results[client_name] = result
        logger.debug(f"Received result from {client_name}")

    def _broadcast_and_gather(
        self, task_data: Shareable, abort_signal: Signal, fl_ctx: FLContext, timeout: int = 0
    ) -> Dict[str, Shareable]:
        """
        Helper to send tasks and collect results safely.
        Uses result_received_cb to properly collect results.
        """
        # Clear buffer
        self.phase_results = {}

        # Create Task with callback
        task = Task(
            name=self.custom_task_name,
            data=task_data,
            timeout=timeout,
            result_received_cb=self._result_received_cb,
        )

        # Broadcast and wait for results
        self.broadcast_and_wait(
            task=task,
            min_responses=self.min_clients,
            wait_time_after_min_received=5,
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

        # Also collect any results from client_tasks (backup method)
        for client_task in task.client_tasks:
            client_name = client_task.client.name
            if client_name not in self.phase_results and client_task.result is not None:
                rc = client_task.result.get_return_code()
                if rc == ReturnCode.OK:
                    self.phase_results[client_name] = client_task.result
                    logger.debug(f"Collected result from task.client_tasks: {client_name}")

        logger.info(f"Collected {len(self.phase_results)} results")
        return self.phase_results

    def _phase_one_election(self, abort_signal: Signal, fl_ctx: FLContext) -> bool:
        logger.info("=== PHASE 1: Local Feature Selection & Election ===")

        task_data = Shareable()
        task_data["request_type"] = "feature_selection"

        # Broadcast and collect results
        results = self._broadcast_and_gather(task_data, abort_signal, fl_ctx)

        if not results:
            logger.error("No feature votes received. Aborting.")
            return False

        # Extract client data
        self.cached_client_selections = self._extract_client_data(results)

        if not self.cached_client_selections:
            logger.error("Received responses, but failed to extract selection data. Aborting.")
            return False

        logger.info(f"Phase 1 Complete. Processed votes from {len(self.cached_client_selections)} clients.")
        return True

    def _phase_two_tuning_and_masking(self, abort_signal: Signal, fl_ctx: FLContext):
        logger.info("=== PHASE 2: Tuning & Global Mask Generation ===")

        # 1. Run Tuning Loop (if enabled)
        if self.auto_tune and self.tuning_rounds > 0:
            logger.info(f"Starting Auto-tuning ({self.tuning_rounds} rounds)...")

            for i in range(self.tuning_rounds):
                if abort_signal.triggered:
                    logger.warning("Abort signal received during tuning")
                    break

                # Evaluate current freedom_degree
                mask = self._aggregate_selections(self.cached_client_selections)

                task_data = Shareable()
                task_data["request_type"] = "tuning_eval"
                task_data["tuning_mask"] = mask.tolist()

                results = self._broadcast_and_gather(task_data, abort_signal, fl_ctx)

                # Aggregate Scores
                scores = []
                for v in results.values():
                    if "tuning_score" in v:
                        scores.append(v["tuning_score"])
                score = sum(scores) / len(scores) if scores else 0.0

                logger.info(
                    f"Tuning Round {i + 1}/{self.tuning_rounds}: FD={self.freedom_degree:.4f} -> Score={score:.4f}"
                )
                self.tuning_history.append((self.freedom_degree, score))

                # Calculate next FD for next iteration (if not last round)
                if i < self.tuning_rounds - 1:
                    self.freedom_degree = self._calculate_next_fd(first_step=(i == 0))

            # Select best FD from evaluated options
            if self.tuning_history:
                best_fd, best_score = max(self.tuning_history, key=lambda x: x[1])
                self.freedom_degree = best_fd
                logger.info(f"Tuning Complete. Optimal Freedom Degree: {best_fd:.4f} (Score: {best_score:.4f})")
            else:
                logger.warning("No tuning results, keeping initial freedom_degree")

        # 2. Generate Final Mask
        final_mask = self._aggregate_selections(self.cached_client_selections)
        self.global_feature_mask = final_mask
        n_sel = np.sum(final_mask)
        logger.info(
            f"Final Global Mask: {n_sel} features selected "
            f"(FD={self.freedom_degree:.4f}, aggregation_mode={self.aggregation_mode})"
        )

        # 3. Distribute mask to clients
        task_data = Shareable()
        task_data["request_type"] = "apply_mask"
        task_data["global_feature_mask"] = final_mask.tolist()

        self._broadcast_and_gather(task_data, abort_signal, fl_ctx)
        logger.info("Global mask distributed to all clients")

    def _phase_three_aggregation(self, abort_signal: Signal, fl_ctx: FLContext):
        logger.info(f"=== PHASE 3: Aggregation Rounds (FL Training - {self.fl_rounds} Rounds) ===")

        for i in range(1, self.fl_rounds + 1):
            if abort_signal.triggered:
                logger.warning("Abort signal received during FL training")
                break

            logger.info(f"--- FL Round {i}/{self.fl_rounds} ---")

            task_data = Shareable()
            task_data["request_type"] = "train"
            if self.global_weights:
                task_data["params"] = self.global_weights

            results = self._broadcast_and_gather(task_data, abort_signal, fl_ctx, timeout=self.train_timeout)

            # Aggregate Weights (FedAvg)
            self._aggregate_weights(results)

        logger.info("FL Training phase complete")

    # ==============================================================================
    # HELPER METHODS
    # ==============================================================================

    def _aggregate_weights(self, results: Dict[str, Shareable]):
        """FedAvg-style weight aggregation"""
        total_samples = 0
        weighted_weights = None

        for shareable in results.values():
            if "params" not in shareable:
                continue
            n = shareable.get("num_samples", 1)
            weights = shareable.get("params")
            if weights is not None:
                for k, v in weights.items():
                    # Ensure v is a numpy array before operations
                    v_array = np.array(v)
                    if k not in weighted_weights:
                        logger.warning(f"Unexpected weight key '{k}' from client, skipping")
                        continue
                    if weighted_weights[k].shape != v_array.shape:
                        logger.error(f"Weight shape mismatch for key '{k}': expected {weighted_weights[k].shape}, got {v_array.shape}")
                        continue
                    weighted_weights[k] += v_array * n
                total_samples += n

        if total_samples > 0 and weighted_weights is not None:
            self.global_weights = {k: v / total_samples for k, v in weighted_weights.items()}
            logger.info(f"Aggregated weights from {len(results)} clients ({total_samples} samples)")

    def _extract_client_data(self, results: Dict[str, Shareable]) -> Dict[str, Dict]:
        """Extract feature selection data from client results"""
        client_data = {}
        for key, contrib in results.items():
            if "selected_features" in contrib:
                selected = np.array(contrib["selected_features"])

                # Get n_features from first client response
                if self.n_features is None:
                    self.n_features = len(selected)
                    logger.debug(f"Inferred n_features={self.n_features} from {key}")

                client_data[key] = {
                    "selected_features": selected,
                    "feature_scores": np.array(contrib["feature_scores"]),
                    "num_samples": contrib.get("num_samples", 1),
                }
                logger.debug(f"Extracted {np.sum(contrib['selected_features'])} features from {key}")
        return client_data

    def _aggregate_selections(self, client_selections: Dict[str, Dict]) -> np.ndarray:
        """
        Aggregate feature selections from all clients.

        Freedom degree controls the blend between intersection and union:
        - FD=0: Intersection (only features selected by ALL clients)
        - FD=1: Union (features selected by ANY client)
        - 0<FD<1: Weighted voting based on scores
        """
        if not client_selections:
            logger.warning("No client selections to aggregate")
            n = self.n_features
            if n is None:
                logger.error("Cannot create empty mask: self.n_features is None")
                raise ValueError("Total number of features (n_features) must be known before aggregation")
            return np.zeros(n, dtype=bool)

        masks = [s["selected_features"] for s in client_selections.values()]
        scores = [s["feature_scores"] for s in client_selections.values()]
        weights = [s["num_samples"] for s in client_selections.values()]

        masks = np.array(masks)
        scores = np.array(scores)
        total = sum(weights)
        weights = np.array(weights) / total if total > 0 else np.ones(len(weights)) / len(weights)

        intersection = np.all(masks, axis=0)
        union = np.any(masks, axis=0)

        # Handle edge cases
        if self.freedom_degree <= 0.05:
            return intersection
        if self.freedom_degree >= 0.99:
            return union

        return self._weighted_election(masks, scores, weights, intersection, union)

    def _weighted_election(
        self, masks: np.ndarray, scores: np.ndarray, weights: np.ndarray, intersection: np.ndarray, union: np.ndarray
    ) -> np.ndarray:
        """
        Perform weighted voting for features in the difference set.
        Uses aggregation_mode to determine weighting strategy.
        """
        diff_mask = union & ~intersection
        if not np.any(diff_mask):
            return intersection

        # Compute aggregated scores based on aggregation_mode
        agg_scores = np.zeros(len(intersection))

        # Determine weights based on aggregation mode
        if self.aggregation_mode == "uniform":
            # Equal weight for all clients
            effective_weights = np.ones(len(weights)) / len(weights)
        else:  # "weighted" mode (default)
            # Use sample-size-based weights
            effective_weights = weights

        for i, (m, s) in enumerate(zip(masks, scores)):
            valid = m.astype(bool)
            if np.any(valid):
                min_s, max_s = np.min(s[valid]), np.max(s[valid])
                if max_s > min_s:
                    # Normal case: normalize to [0, 1]
                    norm_s = (s - min_s) / (max_s - min_s)
                else:
                    # All scores are equal: use uniform scores of 0.5 for consistency
                    norm_s = np.full_like(s, 0.5)
                agg_scores += norm_s * effective_weights[i]

        # Select top features based on freedom_degree
        n_add = int(np.ceil(np.sum(diff_mask) * self.freedom_degree))
        if n_add > 0:
            diff_scores = agg_scores[diff_mask]
            n_add = min(n_add, len(diff_scores))
            if n_add > 0:
                cutoff = np.partition(diff_scores, -n_add)[-n_add]
                selected_diff = (agg_scores >= cutoff) & diff_mask
                return intersection | selected_diff

        return intersection

    def _calculate_next_fd(self, first_step: bool) -> float:
        """Hill-climbing to find optimal freedom degree"""
        MIN_FD, MAX_FD = 0.05, 1.0

        if first_step:
            return np.clip(self.freedom_degree + self.search_step, MIN_FD, MAX_FD)

        if len(self.tuning_history) < 2:
            return self.freedom_degree

        curr_fd, curr_score = self.tuning_history[-1]
        prev_fd, prev_score = self.tuning_history[-2]

        if curr_score > prev_score:
            new_fd = curr_fd + (self.current_direction * self.search_step)
        else:
            self.current_direction *= -1
            self.search_step *= 0.5
            new_fd = prev_fd + (self.current_direction * self.search_step)

        return np.clip(new_fd, MIN_FD, MAX_FD)
