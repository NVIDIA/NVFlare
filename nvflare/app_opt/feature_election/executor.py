"""
Feature Election Client Executor for NVIDIA FLARE
Handles local feature selection and responds to server requests
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
import logging
from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif, mutual_info_classif,
    RFE
)
from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import warnings
warnings.filterwarnings('ignore')

# Try to import PyImpetus

try:
    from PyImpetus import PPIMBC
    PYIMPETUS_AVAILABLE = True
except ImportError:
    PYIMPETUS_AVAILABLE = False


logger = logging.getLogger(__name__)


class FeatureElectionExecutor(Executor):
    """
    Client-side executor for Feature Election
    Performs local feature selection and communicates with the server
    """

    def __init__(
        self,
        fs_method: str = "lasso",
        fs_params: Optional[Dict] = None,
        eval_metric: str = "f1",
        quick_eval: bool = True,
        task_name: str = "feature_election"
    ):
        """
        Initialize Feature Election Executor

        Args:
            fs_method: Feature selection method
                      ('lasso', 'elastic_net', 'mutual_info', 'chi2', 'f_classif',
                       'rfe', 'random_forest', 'selectkbest', 'pyimpetus', 'ppimbc')
            fs_params: Parameters for the feature selection method
            eval_metric: Metric for evaluation ('f1', 'accuracy', 'auc')
            quick_eval: Whether to perform quick evaluation (5 epochs vs full training)
            task_name: Name of the feature election task
        """
        super().__init__()

        self.fs_method = fs_method.lower()
        self.fs_params = fs_params or {}
        self.eval_metric = eval_metric
        self.quick_eval = quick_eval
        self.task_name = task_name

        # Data placeholders
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.feature_names = None

        # Results storage
        self.selected_features = None
        self.feature_scores = None
        self.global_feature_mask = None

        # Set default parameters based on method
        self._set_default_params()

    def _set_default_params(self):
        """Set default parameters for each feature selection method"""
        defaults = {
            "lasso": {"alpha": 0.01, "max_iter": 1000},
            "elastic_net": {"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 1000},
            "mutual_info": {"n_neighbors": 3, "random_state": 42},
            "chi2": {"k": 10},
            "f_classif": {"k": 10},
            "rfe": {"n_features_to_select": 10, "step": 1},
            "random_forest": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
            "selectkbest": {"k": 10, "score_func": "f_classif"},
            "pyimpetus": {
                "model": "random_forest",
                "p_val_thresh": 0.05,
                "num_sim": 50,
                "random_state": 42,
                "verbose": 0
            },
            "ppimbc": {
                "model": "random_forest",
                "p_val_thresh": 0.05,
                "num_sim": 50,
                "random_state": 42,
                "verbose": 0
            }
        }

        if self.fs_method in defaults:
            # Merge with user-provided params (user params override defaults)
            self.fs_params = {**defaults[self.fs_method], **self.fs_params}

    def set_data(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                 feature_names: Optional[list] = None):
        """
        Set training and validation data

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Feature names (optional)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val if X_val is not None else X_train
        self.y_val = y_val if y_val is not None else y_train

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        logger.info(f"Data set: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal
    ) -> Shareable:
        """
        Execute feature election task

        Args:
            task_name: Name of the task
            shareable: Input shareable from server
            fl_ctx: FL context
            abort_signal: Abort signal

        Returns:
            Response shareable
        """
        if task_name != self.task_name:
            return make_reply(ReturnCode.TASK_UNKNOWN)

        request_type = shareable.get("request_type")

        if request_type == "feature_selection":
            # Perform local feature selection
            return self._handle_feature_selection(shareable, fl_ctx, abort_signal)
        elif request_type == "apply_mask":
            # Apply global mask from server
            return self._handle_apply_mask(shareable, fl_ctx)
        else:
            logger.error(f"Unknown request type: {request_type}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _handle_feature_selection(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal
    ) -> Shareable:
        """Handle feature selection request from server"""

        if self.X_train is None:
            logger.error("No training data available")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        try:
            # Perform feature selection
            selected_mask, feature_scores = self._perform_feature_selection()

            # Evaluate performance with selected features
            initial_score = self._evaluate_model(
                self.X_train, self.y_train, self.X_val, self.y_val
            )

            # Apply feature mask and evaluate
            X_train_selected = self.X_train[:, selected_mask]
            X_val_selected = self.X_val[:, selected_mask]
            fs_score = self._evaluate_model(
                X_train_selected, self.y_train, X_val_selected, self.y_val
            )

            # Log results
            n_selected = np.sum(selected_mask)
            n_total = len(selected_mask)
            logger.info(f"Selected {n_selected}/{n_total} features")
            logger.info(f"Initial score: {initial_score:.4f}, FS score: {fs_score:.4f}")

            # Store results
            self.selected_features = selected_mask
            self.feature_scores = feature_scores

            # Create response
            response = make_reply(ReturnCode.OK)
            response["selected_features"] = selected_mask.tolist()
            response["feature_scores"] = feature_scores.tolist()
            response["num_samples"] = len(self.X_train)
            response["initial_score"] = float(initial_score)
            response["fs_score"] = float(fs_score)

            return response

        except Exception as e:
            logger.error(f"Feature selection failed: {str(e)}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _perform_feature_selection(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform feature selection using specified method

        Returns:
            Tuple of (selected_mask, feature_scores)
        """
        n_features = self.X_train.shape[1]

        # Handle PyImpetus methods
        if self.fs_method in ["pyimpetus", "ppimbc"]:
            return self._perform_pyimpetus_selection()

        # Scale data for methods that need it
        if self.fs_method in ["lasso", "elastic_net"]:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.X_train)
        else:
            X_scaled = self.X_train

        if self.fs_method == "lasso":
            selector = Lasso(**self.fs_params)
            selector.fit(X_scaled, self.y_train)
            feature_scores = np.abs(selector.coef_)
            # For Lasso, use non-zero coefficients as selected
            selected_mask = feature_scores > 1e-6  # Small threshold for numerical stability

        elif self.fs_method == "elastic_net":
            selector = ElasticNet(**self.fs_params)
            selector.fit(X_scaled, self.y_train)
            feature_scores = np.abs(selector.coef_)
            selected_mask = feature_scores > 1e-6

        elif self.fs_method == "mutual_info":
            feature_scores = mutual_info_classif(
                X_scaled, self.y_train,
                n_neighbors=self.fs_params.get("n_neighbors", 3),
                random_state=self.fs_params.get("random_state", 42)
            )
            k = min(self.fs_params.get("k", 10), n_features)
            selected_indices = np.argsort(feature_scores)[-k:]
            selected_mask = np.zeros(n_features, dtype=bool)
            selected_mask[selected_indices] = True

        elif self.fs_method == "chi2":
            # Chi2 requires non-negative features
            X_positive = X_scaled - np.min(X_scaled, axis=0)
            feature_scores, _ = chi2(X_positive, self.y_train)
            k = min(self.fs_params.get("k", 10), n_features)
            selected_indices = np.argsort(feature_scores)[-k:]
            selected_mask = np.zeros(n_features, dtype=bool)
            selected_mask[selected_indices] = True

        elif self.fs_method == "f_classif":
            feature_scores, _ = f_classif(X_scaled, self.y_train)
            k = min(self.fs_params.get("k", 10), n_features)
            selected_indices = np.argsort(feature_scores)[-k:]
            selected_mask = np.zeros(n_features, dtype=bool)
            selected_mask[selected_indices] = True

        elif self.fs_method == "rfe":
            estimator = LogisticRegression(max_iter=1000, random_state=42)
            selector = RFE(
                estimator,
                n_features_to_select=min(self.fs_params.get("n_features_to_select", 10), n_features),
                step=self.fs_params.get("step", 1)
            )
            selector.fit(X_scaled, self.y_train)
            selected_mask = selector.support_
            feature_scores = selector.ranking_.astype(float)
            # Convert ranking to scores (lower ranking = better)
            feature_scores = 1.0 / feature_scores

        elif self.fs_method == "random_forest":
            rf = RandomForestClassifier(**self.fs_params)
            rf.fit(X_scaled, self.y_train)
            feature_scores = rf.feature_importances_
            k = min(self.fs_params.get("k", 10), n_features)
            selected_indices = np.argsort(feature_scores)[-k:]
            selected_mask = np.zeros(n_features, dtype=bool)
            selected_mask[selected_indices] = True

        elif self.fs_method == "selectkbest":
            score_func_name = self.fs_params.get("score_func", "f_classif")
            if score_func_name == "chi2":
                X_positive = X_scaled - np.min(X_scaled, axis=0)
                score_func = chi2
                X_to_use = X_positive
            elif score_func_name == "mutual_info":
                score_func = mutual_info_classif
                X_to_use = X_scaled
            else:
                score_func = f_classif
                X_to_use = X_scaled

            selector = SelectKBest(
                score_func=score_func,
                k=min(self.fs_params.get("k", 10), n_features)
            )
            selector.fit(X_to_use, self.y_train)
            selected_mask = selector.get_support()
            feature_scores = selector.scores_

        else:
            # Default: select all features
            logger.warning(f"Unknown method {self.fs_method}, selecting all features")
            selected_mask = np.ones(n_features, dtype=bool)
            feature_scores = np.ones(n_features)

        # Ensure we have at least one feature selected
        if np.sum(selected_mask) == 0:
            logger.warning("No features selected, selecting top feature")
            if len(feature_scores) > 0:
                top_feature = np.argmax(feature_scores)
                selected_mask = np.zeros(n_features, dtype=bool)
                selected_mask[top_feature] = True

        # Normalize scores to [0, 1]
        if np.max(feature_scores) > np.min(feature_scores):
            feature_scores = (feature_scores - np.min(feature_scores)) / \
                           (np.max(feature_scores) - np.min(feature_scores))
        else:
            # If all scores are same, use binary scores
            feature_scores = selected_mask.astype(float)

        return selected_mask, feature_scores

    def _perform_pyimpetus_selection(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform feature selection using PyImpetus methods
        PyImpetus returns selected feature indices, not coefficients
        """
        if not PYIMPETUS_AVAILABLE:
            logger.error("PyImpetus not available. Install with: pip install PyImpetus")
            n_features = self.X_train.shape[1]
            # Fallback to mutual info
            feature_scores = mutual_info_classif(self.X_train, self.y_train, random_state=42)
            k = min(10, n_features)
            selected_indices = np.argsort(feature_scores)[-k:]
            selected_mask = np.zeros(n_features, dtype=bool)
            selected_mask[selected_indices] = True
            return selected_mask, feature_scores

        try:
            # Get PyImpetus parameters
            model_type = self.fs_params.get("model", "random_forest")
            p_val_thresh = self.fs_params.get("p_val_thresh", 0.05)
            num_sim = self.fs_params.get("num_sim", 50)
            random_state = self.fs_params.get("random_state", 42)
            verbose = self.fs_params.get("verbose", 0)

            n_features = self.X_train.shape[1]

            logger.info(f"Running PyImpetus with {n_features} features")

            # Initialize base model
            if model_type == "random_forest":
                base_model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    max_depth=None
                )
            elif model_type == "logistic":
                base_model = LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                    solver='liblinear'
                )
            else:
                base_model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_state
                )

            # Use PPIMBC for feature selection
            if self.fs_method == "pyimpetus":
                selector = PPIMBC(
                    base_model,
                    p_val_thresh=p_val_thresh,
                    num_sim=num_sim,
                    random_state=random_state,
                    verbose=verbose
                )
            # Fit the selector
            selector.fit(self.X_train, self.y_train)

            # Get selected features - PyImpetus returns INDICES of selected features
            selected_indices = selector.selected_features_

            logger.info(f"PyImpetus selected {len(selected_indices)} features: {selected_indices}")

            # Create binary mask from selected indices
            selected_mask = np.zeros(n_features, dtype=bool)
            if len(selected_indices) > 0:
                selected_mask[selected_indices] = True
            else:
                logger.warning("PyImpetus selected 0 features, using fallback")
                # Fallback: select top 10% features using mutual info
                feature_scores_fallback = mutual_info_classif(self.X_train, self.y_train, random_state=42)
                k = max(1, n_features // 10)
                selected_indices = np.argsort(feature_scores_fallback)[-k:]
                selected_mask[selected_indices] = True
                selected_indices = np.where(selected_mask)[0]

            # Create feature scores
            if hasattr(selector, 'p_vals_') and len(selector.p_vals_) == n_features:
                # Use -log(p_value) as score (higher = more significant)
                epsilon = 1e-10
                feature_scores = -np.log10(selector.p_vals_ + epsilon)
                # Normalize to [0, 1]
                if np.max(feature_scores) > 0:
                    feature_scores = feature_scores / np.max(feature_scores)
                logger.info("Created scores from p-values")
            else:
                # Binary scores: 1 for selected, 0 for not selected
                feature_scores = np.zeros(n_features)
                feature_scores[selected_indices] = 1.0
                logger.info("Created binary scores")

            logger.info(f"Final PyImpetus selection: {np.sum(selected_mask)}/{n_features} features")
            return selected_mask, feature_scores

        except Exception as e:
            logger.error(f"PyImpetus feature selection failed: {str(e)}")
            # Fallback to mutual information
            logger.info("Falling back to mutual information feature selection")
            n_features = self.X_train.shape[1]
            feature_scores = mutual_info_classif(self.X_train, self.y_train, random_state=42)
            k = min(10, n_features)
            selected_indices = np.argsort(feature_scores)[-k:]
            selected_mask = np.zeros(n_features, dtype=bool)
            selected_mask[selected_indices] = True
            return selected_mask, feature_scores

    def _evaluate_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """
        Quick evaluation of model performance

        Returns:
            Performance score
        """

        # Skip evaluation if validation set is too small
        if len(y_val) < 5:
            return 0.5  # Return neutral score

        # Train simple model
        model = LogisticRegression(max_iter=100 if self.quick_eval else 1000, random_state=42)

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            if self.eval_metric == "f1":
                score = f1_score(y_val, y_pred, average='weighted')
            elif self.eval_metric == "accuracy":
                score = accuracy_score(y_val, y_pred)
            elif self.eval_metric == "auc":
                if len(np.unique(y_val)) == 2:
                    y_proba = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, y_proba)
                else:
                    # Fall back to f1 for multi-class
                    score = f1_score(y_val, y_pred, average='weighted')
            else:
                score = f1_score(y_val, y_pred, average='weighted')

            return max(score, 0.0)  # Ensure non-negative score
        except Exception as e:
            logger.warning(f"Model evaluation failed: {e}, returning default score")
            return 0.5

    def _handle_apply_mask(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Handle apply mask request from server"""

        global_mask = shareable.get("global_feature_mask")
        if global_mask is None:
            logger.error("No global mask received")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # Store global mask
        self.global_feature_mask = np.array(global_mask, dtype=bool)

        # Log results
        logger.info(f"Received global mask: {np.sum(self.global_feature_mask)} features selected")

        # Apply mask to training data if needed
        if self.X_train is not None:
            self.X_train = self.X_train[:, self.global_feature_mask]
            if self.X_val is not None:
                self.X_val = self.X_val[:, self.global_feature_mask]

            # Update feature names
            if self.feature_names is not None:
                self.feature_names = [
                    name for i, name in enumerate(self.feature_names)
                    if self.global_feature_mask[i]
                ]

        return make_reply(ReturnCode.OK)

    def get_selected_features(self) -> Optional[np.ndarray]:
        """Get the global feature mask after election"""
        return self.global_feature_mask

    def get_feature_names(self) -> Optional[list]:
        """Get names of selected features"""
        if self.global_feature_mask is not None and self.feature_names is not None:
            return [
                name for i, name in enumerate(self.feature_names)
                if self.global_feature_mask[i]
            ]
        return None

    def get_pyimpetus_info(self) -> Dict[str, Any]:
        """Get information about PyImpetus availability and methods"""
        info = {
            "pyimpetus_available": PYIMPETUS_AVAILABLE,
            "supported_methods": ["pyimpetus", "ppimbc"] if PYIMPETUS_AVAILABLE else [],
            "current_method": self.fs_method,
            "is_using_pyimpetus": self.fs_method in ["pyimpetus", "ppimbc"] and PYIMPETUS_AVAILABLE
        }
        return info