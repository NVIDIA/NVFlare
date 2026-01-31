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
from typing import Dict, Optional, Tuple

import numpy as np
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


class FeatureElectionExecutor(Executor):
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
            "random_forest": {"n_estimators": 100},
            "pyimpetus": {"p_val_thresh": 0.05},
        }
        if self.fs_method in defaults:
            self.fs_params = {**defaults[self.fs_method], **self.fs_params}

    def set_data(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """
        Set data for the executor.
        X_val and y_val are optional; if not provided, training data is used for evaluation.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val if X_val is not None else X_train
        self.y_val = y_val if y_val is not None else y_train
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

    def evaluate_model(self, X_train, y_train, X_val, y_val) -> float:
        """
        Helper method to train and evaluate a model locally.
        Required for the 'simulate_election' functionality and tests.
        """
        if len(y_train) == 0 or len(y_val) == 0:
            return 0.0

        try:
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
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

            X_tr = self.X_train[:, mask]
            X_v = self.X_val[:, mask]

            # Use helper
            score = self.evaluate_model(X_tr, self.y_train, X_v, self.y_val)

            resp = make_reply(ReturnCode.OK)
            resp["tuning_score"] = float(score)
            return resp
        except Exception as e:
            logger.error(f"Tuning eval failed: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _handle_apply_mask(self, shareable: Shareable) -> Shareable:
        try:
            mask = np.array(shareable.get("global_feature_mask"), dtype=bool)

            # Validate mask length
            if len(mask) != self.X_train.shape[1]:
                logger.error(f"Mask length ({len(mask)}) doesn't match number of features ({self.X_train.shape[1]})")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            logger.info(f"Permanently applying mask: {np.sum(mask)} features selected")

            self.X_train = self.X_train[:, mask]
            self.X_val = self.X_val[:, mask]
            return make_reply(ReturnCode.OK)
        except Exception as e:
            logger.error(f"Mask application failed: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _handle_train(self, shareable: Shareable) -> Shareable:
        try:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(self.X_train)

            # Load global parameters if available (from previous round's aggregation)
            if "params" in shareable:
                p = shareable["params"]
                if "weight_0" in p and "weight_1" in p:
                    # Initialize model structure if needed
                    if not self._model_initialized:
                        # Quick fit to establish coef_ shape, then overwrite
                        self.model.fit(X_tr[:min(10, len(self.y_train))], self.y_train[:min(10, len(self.y_train))])
                        self._model_initialized = True
                    # Set aggregated weights
                    self.model.coef_ = np.array([p["weight_0"]])
                    self.model.intercept_ = np.array(p["weight_1"])

            # Train with warm_start=True continues from current weights
            self.model.fit(X_tr, self.y_train)
            self._model_initialized = True

            resp = make_reply(ReturnCode.OK)
            resp["params"] = {"weight_0": self.model.coef_[0].tolist(), "weight_1": self.model.intercept_.tolist()}
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
            s = Lasso(**self.fs_params).fit(X_scaled, self.y_train)
            scores = np.abs(s.coef_)
            return scores > 1e-6, scores

        elif self.fs_method == "elastic_net":
            s = ElasticNet(**self.fs_params).fit(X_scaled, self.y_train)
            scores = np.abs(s.coef_)
            return scores > 1e-6, scores

        elif self.fs_method == "mutual_info":
            scores = mutual_info_classif(self.X_train, self.y_train, random_state=42)
            mask = np.zeros(n_features, dtype=bool)
            k = max(1, n_features // 2)
            mask[np.argsort(scores)[-k:]] = True
            return mask, scores

        elif self.fs_method == "random_forest":
            rf = RandomForestClassifier(**self.fs_params)
            rf.fit(self.X_train, self.y_train)
            scores = rf.feature_importances_
            mask = np.zeros(n_features, dtype=bool)
            k = max(1, n_features // 2)
            mask[np.argsort(scores)[-k:]] = True
            return mask, scores

        elif self.fs_method == "pyimpetus":
            if not PYIMPETUS_AVAILABLE:
                logger.warning("PyImpetus not available, falling back to mutual_info")
                scores = mutual_info_classif(X_scaled, self.y_train, random_state=42)
                mask = np.zeros(n_features, dtype=bool)
                k = max(1, n_features // 2)
                mask[np.argsort(scores)[-k:]] = True
                return mask, scores

            model = PPIMBC(self.fs_params.get("model", LogisticRegression(max_iter=1000, random_state=42)))
            selected_features = model.fit(self.X_train, self.y_train)
            mask = np.zeros(n_features, dtype=bool)
            mask[selected_features] = True
            scores = np.zeros(n_features)
            scores[selected_features] = 1.0
            return mask, scores

        else:
            return np.ones(n_features, dtype=bool), np.ones(n_features)
