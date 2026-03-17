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

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal

try:
    from PyImpetus import PPIMBC

    PYIMPETUS_AVAILABLE = True
except ImportError:
    PYIMPETUS_AVAILABLE = False

logger = logging.getLogger(__name__)

LASSO_ELASTIC_NET_ZERO_THRESHOLD: float = 1e-6


class FeatureElectionExecutor(Executor):
    """
    Client-side executor for the Feature Election federated workflow.

    Handles four request types dispatched by ``FeatureElectionController``:

    * ``feature_selection`` — runs the configured FS method on local data and returns
      a boolean feature mask and per-feature scores.
    * ``tuning_eval`` — evaluates a candidate mask proposed by the controller during
      the hill-climbing phase and returns the local score.
    * ``apply_mask`` — permanently slices ``X_train`` / ``X_val`` to the selected
      features.  **Idempotent**: if the same mask is received a second time (e.g. due
      to task retransmission) the call returns ``OK`` immediately without modifying data.
    * ``train`` — performs one FedAvg round on the masked feature set and returns the
      updated model weights.

    Args:
        fs_method: Feature selection algorithm.  One of ``'lasso'``, ``'elastic_net'``,
            ``'mutual_info'``, ``'random_forest'``, ``'pyimpetus'``.
        fs_params: Extra keyword arguments forwarded to the FS algorithm.
        eval_metric: ``'f1'`` (weighted) or ``'accuracy'``, used for tuning eval and
            local scoring.
        task_name: Must match the ``task_name`` on ``FeatureElectionController``.

    Note:
        Call :meth:`set_data` before the executor is registered with the FL runtime.
        ``FeatureElectionExecutor`` has no ``client_id`` attribute; use
        ``fl_ctx.get_identity_name()`` inside ``_load_data_if_needed`` to retrieve the
        site name assigned by the FL platform.
    """

    def __init__(
        self,
        fs_method: str = "lasso",
        fs_params: Optional[Dict] = None,
        eval_metric: str = "f1",
        task_name: str = "feature_election",
    ):
        super().__init__()
        self.fs_method = fs_method.lower()
        self.fs_params = fs_params or {}
        self.eval_metric = eval_metric
        self.task_name = task_name

        # Data
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

        # Scaler fitted on X_train; stored so _handle_train and _handle_tuning_eval
        # use the same parameters rather than each reconstructing an identical instance.
        # Reset to None whenever X_train changes (set_data, apply_mask).
        self.scaler = None

        # Use LogisticRegression with LBFGS solver - much faster convergence than SGDClassifier
        # for small-to-medium datasets. warm_start=True allows incremental training across rounds.
        self.global_feature_mask = None
        self.model = LogisticRegression(max_iter=1000, solver="lbfgs", warm_start=True, random_state=42)
        self._model_initialized = False  # Track if model has been fit

        self._set_default_params()

    def _set_default_params(self):
        defaults = {
            "lasso": {"alpha": 0.01},
            "elastic_net": {"alpha": 0.01, "l1_ratio": 0.5},
            "mutual_info": {},
            "random_forest": {"n_estimators": 100, "random_state": 42},
            "pyimpetus": {"p_val_thresh": 0.05},
        }
        if self.fs_method in defaults:
            self.fs_params = {**defaults[self.fs_method], **self.fs_params}

    def set_data(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """
        Set data for the executor.
        X_val and y_val are optional; if not provided, training data is used for evaluation.
        """
        # Validate that feature_names matches X_train dimensions to prevent misalignment
        if feature_names is not None:
            if len(feature_names) != X_train.shape[1]:
                raise ValueError(
                    f"Length of feature_names ({len(feature_names)}) must match "
                    f"number of features in X_train ({X_train.shape[1]})."
                )

        # Coerce pandas inputs to numpy so positional indexing inside _handle_train
        # and elsewhere is always consistent regardless of the DataFrame/Series index.
        self.X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        self.y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
        self.scaler = None  # invalidate cached scaler whenever X_train changes

        # If X_val is provided, ensure it has the same feature count as X_train
        if X_val is not None:
            X_val = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val = y_val.values if isinstance(y_val, pd.Series) else y_val
            if X_val.shape[1] != self.X_train.shape[1]:
                raise ValueError(
                    f"X_val feature count ({X_val.shape[1]}) does not match "
                    f"X_train feature count ({self.X_train.shape[1]})."
                )
            self.X_val = X_val
            self.y_val = y_val
        else:
            self.X_val = self.X_train
            self.y_val = self.y_train

        self.feature_names = feature_names

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name != self.task_name:
            return make_reply(ReturnCode.TASK_UNKNOWN)

        request_type = shareable.get("request_type")

        if request_type == "feature_selection":
            return self._handle_feature_selection()
        elif request_type == "tuning_eval":
            return self._handle_tuning_eval(shareable)
        elif request_type == "apply_mask":
            return self._handle_apply_mask(shareable)
        elif request_type == "train":
            return self._handle_train(shareable)
        else:
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def evaluate_model(self, X_train, y_train, X_val, y_val, scaler=None) -> float:
        """
        Helper method to train and evaluate a model locally.
        Required for the 'simulate_election' functionality and tests.

        Args:
            scaler: Optional pre-fitted ``StandardScaler``.  When provided the data
                is transformed (not fit-transformed), ensuring the same normalisation
                parameters are used as those established on the same feature set by the
                caller.  When ``None`` a fresh scaler is fitted on ``X_train``.
        """
        if len(y_train) == 0 or len(y_val) == 0:
            return 0.0

        try:
            # Scale — reuse a caller-supplied scaler when available so that
            # tuning-eval and training normalise with identical parameters.
            if scaler is None:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
            else:
                X_train_scaled = scaler.transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Quick train
            model = LogisticRegression(max_iter=200, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)

            if self.eval_metric == "accuracy":
                return accuracy_score(y_val, y_pred)
            return f1_score(y_val, y_pred, average="weighted")
        except Exception as e:
            logger.warning(f"Local evaluation failed: {e}")
            return 0.0

    def _handle_feature_selection(self) -> Shareable:
        if self.X_train is None:
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        try:
            mask, scores = self.perform_feature_selection()
            if not np.any(mask):
                # Warn here so the server log shows a client-side explanation.
                # The controller's _extract_client_data will independently skip
                # this vote, but without this message the silence there looks like
                # a missing client rather than a regularisation issue.
                logger.warning(
                    f"Feature selection produced an all-False mask "
                    f"(fs_method={self.fs_method!r}, fs_params={self.fs_params}). "
                    "This client's vote will be excluded from global mask aggregation. "
                    "Consider lowering the regularisation strength "
                    "(e.g. reduce 'alpha' for Lasso/ElasticNet)."
                )
            resp = make_reply(ReturnCode.OK)
            resp["selected_features"] = mask.tolist()
            resp["feature_scores"] = scores.tolist()
            resp["num_samples"] = len(self.X_train)
            return resp
        except Exception as e:
            logger.error(f"FS failed: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _handle_tuning_eval(self, shareable: Shareable) -> Shareable:
        try:
            mask = np.array(shareable.get("tuning_mask"), dtype=bool)
            if self.X_train is None or np.sum(mask) == 0:
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            if len(mask) != self.X_train.shape[1]:
                logger.error(f"Tuning mask length ({len(mask)}) doesn't match feature count ({self.X_train.shape[1]})")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            X_tr = self.X_train[:, mask]
            X_v = self.X_val[:, mask]

            # Fit a dedicated scaler for this candidate mask so that the
            # normalisation used during tuning evaluation is identical to what
            # _handle_train would apply for the same feature set.  Passing it
            # into evaluate_model avoids a redundant fit_transform on the same
            # data and ensures consistent per-feature statistics.
            tuning_scaler = StandardScaler()
            tuning_scaler.fit(X_tr)
            score = self.evaluate_model(X_tr, self.y_train, X_v, self.y_val, scaler=tuning_scaler)

            resp = make_reply(ReturnCode.OK)
            resp["tuning_score"] = float(score)
            resp["num_samples"] = len(self.y_train)
            return resp
        except Exception as e:
            logger.error(f"Tuning eval failed: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _handle_apply_mask(self, shareable: Shareable) -> Shareable:
        try:
            if self.X_train is None:
                logger.error("apply_mask received before set_data was called")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            mask = np.array(shareable.get("global_feature_mask"), dtype=bool)

            # Idempotency guard: if this exact mask was already applied (e.g. task
            # retransmission), X_train is already sliced down to the selected features
            # so re-applying would raise an IndexError.  Return OK immediately.
            if self.global_feature_mask is not None:
                if np.array_equal(mask, self.global_feature_mask):
                    logger.info("Mask already applied (duplicate task delivery); returning OK")
                    return make_reply(ReturnCode.OK)
                # A different mask arrived after the first was already applied — the
                # executor's feature space has already been permanently reduced, so
                # this is an unrecoverable state.  Log clearly and fail fast.
                logger.error(
                    f"Received a different mask after mask was already applied. "
                    f"Expected mask length {len(self.global_feature_mask)} "
                    f"(checksum {self.global_feature_mask.sum()}), "
                    f"got length {len(mask)} (checksum {mask.sum()}). "
                    "This is unrecoverable — the executor feature space has already been reduced."
                )
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            # Validate mask length against the *current* (pre-mask) feature count
            if len(mask) != self.X_train.shape[1]:
                logger.error(f"Mask length ({len(mask)}) doesn't match number of features ({self.X_train.shape[1]})")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            logger.info(f"Permanently applying mask: {np.sum(mask)} features selected")

            self.global_feature_mask = mask
            self.X_train = self.X_train[:, mask]
            self.X_val = self.X_val[:, mask]
            if self.feature_names is not None:
                self.feature_names = [self.feature_names[i] for i in np.where(mask)[0]]
            self.scaler = None  # feature count changed; cached scaler is invalid
            return make_reply(ReturnCode.OK)
        except Exception as e:
            logger.error(f"Mask application failed: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _handle_train(self, shareable: Shareable) -> Shareable:
        try:
            # Fit the scaler once per feature set; reuse across rounds so training
            # and evaluation always use identical normalisation parameters.
            if self.scaler is None:
                self.scaler = StandardScaler()
                self.scaler.fit(self.X_train)
            X_tr = self.scaler.transform(self.X_train)

            # Parse global parameters from the server if present.  We extract them
            # here but assign them immediately before the warm-start fit below, so
            # the weight assignment is always the last write to coef_/intercept_
            # before model.fit() — regardless of whether the model needed an init fit.
            global_coef = None
            global_intercept = None
            if "params" in shareable:
                p = shareable["params"]
                if "weight_0" in p and "weight_1" in p:
                    coef = np.array(p["weight_0"])
                    if coef.ndim == 1:
                        coef = coef.reshape(1, -1)  # Binary: (n_features,) -> (1, n_features)
                    global_coef = coef
                    global_intercept = np.array(p["weight_1"])

            # Ensure model structure (coef_ shape) is established before any weight
            # assignment.  Guarantee at least one sample per class so LogisticRegression
            # does not raise "only one class in data" on sorted or tiny splits.
            if not self._model_initialized:
                unique_classes = np.unique(self.y_train)
                init_idx = [int(np.where(self.y_train == c)[0][0]) for c in unique_classes]
                n_extra = max(0, min(10, len(self.y_train)) - len(init_idx))
                # np.setdiff1d avoids building a full O(n) intermediate list before
                # slicing; assume_unique=True skips the dedup sort since init_idx
                # already contains one distinct index per class.
                remaining = np.setdiff1d(np.arange(len(self.y_train)), init_idx, assume_unique=True)
                init_idx += remaining[:n_extra].tolist()
                self.model.fit(X_tr[init_idx], self.y_train[init_idx])
                self._model_initialized = True

            # Assign aggregated weights immediately before the warm-start fit so that
            # model.fit() always starts from the global model — never from the init fit.
            # Handles both binary (1, n_features) and multi-class (n_classes, n_features).
            if global_coef is not None:
                self.model.coef_ = global_coef
                self.model.intercept_ = global_intercept

            # Train with warm_start=True continues from current weights
            self.model.fit(X_tr, self.y_train)
            self._model_initialized = True

            resp = make_reply(ReturnCode.OK)
            # Send full coef_ to support both binary and multi-class classification
            resp["params"] = {"weight_0": self.model.coef_.tolist(), "weight_1": self.model.intercept_.tolist()}
            resp["num_samples"] = len(self.X_train)
            return resp
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def perform_feature_selection(self) -> Tuple[np.ndarray, np.ndarray]:
        n_features = self.X_train.shape[1]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train)

        if self.fs_method == "lasso":
            # Intentional: Lasso regression is used on classification labels as a
            # continuous proxy target.  This follows the FLASH paper methodology
            # (Christofilogiannis et al., FLTA 2025): L1 regularisation drives
            # irrelevant feature coefficients to exactly zero, giving a sparse
            # boolean mask directly from the non-zero entries.  Using the regression
            # form rather than LogisticRegression avoids multi-class coefficient
            # expansion and keeps the output a single coefficient vector.
            s = Lasso(**self.fs_params).fit(X_scaled, self.y_train)
            scores = np.abs(s.coef_)
            return scores > LASSO_ELASTIC_NET_ZERO_THRESHOLD, scores

        elif self.fs_method == "elastic_net":
            # Intentional: ElasticNet regression on classification labels (same
            # rationale as Lasso above).  The L1+L2 mix handles correlated features
            # better than pure Lasso while still producing exact zeros for selection.
            s = ElasticNet(**self.fs_params).fit(X_scaled, self.y_train)
            scores = np.abs(s.coef_)
            return scores > LASSO_ELASTIC_NET_ZERO_THRESHOLD, scores

        elif self.fs_method == "mutual_info":
            # "k" controls how many top features to select; defaults to top 50%.
            # Pop it before forwarding the rest to mutual_info_classif so sklearn
            # does not receive an unexpected keyword argument.
            mi_params = dict(self.fs_params)
            k_raw = mi_params.pop("k", None)
            k = max(1, int(k_raw)) if k_raw is not None else max(1, n_features // 2)
            # Pop random_state so the caller can override it via fs_params without
            # triggering "got multiple values for keyword argument 'random_state'".
            random_state = mi_params.pop("random_state", 42)
            scores = mutual_info_classif(self.X_train, self.y_train, random_state=random_state, **mi_params)
            mask = np.zeros(n_features, dtype=bool)
            mask[np.argsort(scores)[-k:]] = True
            return mask, scores

        elif self.fs_method == "random_forest":
            # "k" controls how many top features to select; defaults to top 50%.
            # Pop it before forwarding the rest to RandomForestClassifier so sklearn
            # does not receive an unexpected keyword argument.
            rf_params = dict(self.fs_params)
            k_raw = rf_params.pop("k", None)
            k = max(1, int(k_raw)) if k_raw is not None else max(1, n_features // 2)
            rf = RandomForestClassifier(**rf_params)
            rf.fit(self.X_train, self.y_train)
            scores = rf.feature_importances_
            mask = np.zeros(n_features, dtype=bool)
            mask[np.argsort(scores)[-k:]] = True
            return mask, scores

        elif self.fs_method == "pyimpetus":
            if not PYIMPETUS_AVAILABLE:
                # REPRODUCIBILITY NOTE: This fallback uses StandardScaler-transformed
                # data (X_scaled), while the real PyImpetus path below operates on raw
                # self.X_train — PyImpetus performs its own internal conditional
                # independence tests and works best without pre-scaling.  As a result,
                # feature scores and the selected feature set will differ between
                # environments where PyImpetus is and is not installed, making
                # simulation results non-reproducible across them.
                # Install PyImpetus (`pip install PyImpetus`) to remove this
                # inconsistency and use the full PPIMBC algorithm.
                logger.warning(
                    "PyImpetus is not installed — falling back to mutual_info with "
                    "StandardScaler-transformed data.  Feature scores will differ from "
                    "a PyImpetus-enabled environment because PyImpetus operates on raw "
                    "(unscaled) data.  Install PyImpetus for consistent, reproducible results."
                )
                scores = mutual_info_classif(X_scaled, self.y_train, random_state=42)
                mask = np.zeros(n_features, dtype=bool)
                k = max(1, n_features // 2)
                mask[np.argsort(scores)[-k:]] = True
                return mask, scores

            # Extract base model separately, then forward remaining fs_params as kwargs
            base_model = self.fs_params.get("model", LogisticRegression(max_iter=1000, random_state=42))
            ppimbc_kwargs = {k: v for k, v in self.fs_params.items() if k != "model"}
            model = PPIMBC(base_model, **ppimbc_kwargs)
            selected_features = model.fit(self.X_train, self.y_train)
            mask = np.zeros(n_features, dtype=bool)
            mask[selected_features] = True
            scores = np.zeros(n_features)
            scores[selected_features] = 1.0
            return mask, scores

        else:
            raise ValueError(
                f"Unknown fs_method: {self.fs_method!r}. "
                "Supported methods: 'lasso', 'elastic_net', 'mutual_info', 'random_forest', 'pyimpetus'."
            )
