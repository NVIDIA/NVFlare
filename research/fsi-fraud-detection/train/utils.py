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

import argparse
import fcntl
import os
import time
import traceback
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import GradientShap, IntegratedGradients
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from nvflare.apis.analytix import AnalyticsData, LogWriterName
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.widgets.streaming import AnalyticsReceiver


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) with class weighting.

    Downweights easy examples (high-confidence correct predictions) so the
    model focuses training signal on hard, misclassified samples — particularly
    useful for hard-to-detect fraud types like TYPE3.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    gamma=2 is the canonical value from the original paper.

    NOTE: pt is computed from raw softmax probabilities so that class weights
    (alpha) do not distort the focal modulation term.
    """

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha  # per-class weights tensor, one entry per class
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Single log_softmax + nll_loss(reduction='none') uses optimized C++/CUDA path
        log_softmax = F.log_softmax(inputs, dim=1)
        nll = F.nll_loss(log_softmax, targets, reduction="none")  # -log_pt per sample
        pt = (-nll).exp()  # p_t = exp(log_pt)
        focal_weight = (1.0 - pt).pow(self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            return (alpha_t * focal_weight * nll).mean()
        return (focal_weight * nll).mean()


class CustomReceiver(AnalyticsReceiver):
    def initialize(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "#####  Initializing CustomReceiver #####")

    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin: str):
        self.log_info(fl_ctx, "#####  Saving CustomReceiver #####")

        dxo = from_shareable(shareable)
        data = AnalyticsData.from_dxo(dxo, receiver=LogWriterName.MLFLOW)
        if not data:
            self.log_info(fl_ctx, "#####  No data to save #####")
            return

        self.log_info(
            fl_ctx,
            f"#####  Data type: {data.data_type}, origin: {record_origin}, value: {data.value}, step: {data.step}, tag: {data.tag} #####",
        )

    def finalize(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Finalizing CustomReceiver")


def evaluate_on_test_datasets(model, test_datasets_dict, device):
    """
    Evaluate model on multiple test datasets.

    Args:
        model: The model to evaluate
        test_datasets_dict: Dictionary of {dataset_name: (features_tensor, labels_tensor)}
        device: The device to use for evaluation

    Returns:
        Dictionary of metrics for each test dataset
    """
    all_metrics = {}

    model.to(device)
    model.eval()
    with torch.no_grad():
        for dataset_name, (test_features_tensor, test_labels_tensor) in test_datasets_dict.items():
            test_outputs = model(test_features_tensor)
            test_pred = torch.argmax(test_outputs, dim=1)

            # Convert to numpy for sklearn metrics
            test_pred_np = test_pred.cpu().numpy()
            test_labels_np = test_labels_tensor.cpu().numpy()

            # Calculate all metrics using sklearn (binary: positive class = fraud, label 1)
            n_classes = len(np.unique(test_labels_np))
            assert n_classes == 2, "Only binary classification is supported"
            # Balanced accuracy = mean of per-class recall, so not dominated by majority class
            test_acc = balanced_accuracy_score(test_labels_np, test_pred_np)
            test_precision = precision_score(
                test_labels_np, test_pred_np, average="binary", pos_label=1, zero_division=0
            )
            test_recall = recall_score(test_labels_np, test_pred_np, average="binary", pos_label=1, zero_division=0)
            test_f1 = f1_score(test_labels_np, test_pred_np, average="binary", pos_label=1, zero_division=0)
            test_cm = confusion_matrix(test_labels_np, test_pred_np)

            all_metrics[dataset_name] = {
                "accuracy": test_acc,
                "precision": test_precision,
                "recall": test_recall,
                "f1_score": test_f1,
                "confusion_matrix": test_cm,
                "num_samples": len(test_features_tensor),
            }

    return all_metrics


def plot_attribution_summary(attribution_metrics, plot_prefix="", save_fig=False, save_pdf=False):
    """
    Plot attribution summary plot from pre-computed metrics using Captum.

    Args:
        attribution_metrics: Dictionary containing attribution metrics from compute_attributions
        plot_prefix: Prefix for saved plot files
    """
    try:
        attributions = attribution_metrics["attributions"]
        sample_features = attribution_metrics["sample_features"]
        feature_names = attribution_metrics["feature_names"]

        attributions_for_plot = attributions
        # Check if we need to handle the shape differently
        if len(attributions_for_plot.shape) == 3:
            # If 3D array (samples, features, classes), take mean across classes
            attributions_for_plot = np.mean(attributions_for_plot, axis=2)

        plt.figure(figsize=(5, 4))

        # Create violin plot similar to SHAP
        fig, ax = plt.subplots(figsize=(20, 16))
        ax.violinplot(
            [attributions_for_plot[:, i] for i in range(attributions_for_plot.shape[1])],
            positions=range(attributions_for_plot.shape[1]),
            vert=False,
        )
        ax.set_yticks(range(attributions_for_plot.shape[1]))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel("Attribution Value")
        ax.set_title("Feature Attribution Summary")

        if save_fig:
            save_name = f"{plot_prefix}_attribution_summary_plot.png"
            plt.tight_layout()
            out_dir = os.path.dirname(save_name)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            plt.savefig(save_name, dpi=300, bbox_inches="tight")
            if save_pdf:
                plt.savefig(save_name.replace(".png", ".pdf"), bbox_inches="tight")
            plt.close()
            print(f"Attribution summary plot saved successfully to {save_name}")
    except Exception as e:
        traceback.print_exc()
        print(f"Error plotting attribution summary: {e}")


def _feature_category_for_interpretability(name: str) -> str:
    """
    Map feature name to category for interpretability legend: Location, Account Type, Amount.
    """
    n = (name or "").upper().replace(" ", "_")
    if "PHY_AND_TOWER_DISTANCE" in n or ("TOWER" in n and "DISTANCE" in n):
        return "Location"
    if any(x in n for x in ("AMOUNT_SCALED", "AMOUNT_RATIO")):
        return "Amount"
    # Account type, account age, duration, activity
    if any(x in n for x in ("ACCOUNT_TYPE", "ACCOUNT_AGE", "DURATION_SINCE_LAST_ACTIVITY", "ACTIVITY")):
        return "Account Type"
    return "Account Type"  # default for remaining features


def plot_attribution_feature_importance(attribution_metrics, plot_prefix="", save_fig=False, save_pdf=False):
    """
    Plot attribution feature importance bar chart from pre-computed metrics using Captum.

    Args:
        attribution_metrics: Dictionary containing attribution metrics from compute_attributions
        plot_prefix: Prefix for saved plot files
    """
    try:
        attributions = attribution_metrics["attributions"]
        feature_names = attribution_metrics["feature_names"]

        # Handle case where attributions is a list (multiple outputs)
        attributions_for_importance = attributions

        # Check if we need to handle the shape differently
        if len(attributions_for_importance.shape) == 3:
            # If 3D array (samples, features, classes), take mean across classes
            attributions_for_importance = np.mean(attributions_for_importance, axis=2)

        feature_importance = np.mean(np.abs(attributions_for_importance), axis=0)

        # Sort by importance (descending); reverse for barh so most important is on top
        sort_idx = np.argsort(feature_importance)[::-1]
        feature_importance = feature_importance[sort_idx]
        feature_names = [feature_names[i] for i in sort_idx]

        # Category colors for interpretability: Location, Account Type, Amount
        category_colors = {
            "Location": "#2e7d32",
            "Account Type": "#1565c0",
            "Amount": "#9e9e9e",
        }
        colors = [category_colors[_feature_category_for_interpretability(n)] for n in feature_names]

        # barh draws first list item at bottom; reverse so most important is on top
        feature_names = feature_names[::-1]
        feature_importance = feature_importance[::-1]
        colors = colors[::-1]

        n_features = len(feature_names)
        fig_height = max(6, n_features * 0.4)
        fig, ax = plt.subplots(figsize=(7, fig_height))

        ax.barh(feature_names, feature_importance, color=colors, height=0.75)
        ax.set_xlabel("Mean |Attribution value|", fontsize=11)
        ax.set_ylabel("")
        ax.set_title("Feature Importance", fontsize=12)
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=9)
        # More x-axis ticks for easier reading
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=5))
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Legend: Location, Account Type, Amount
        legend_elements = [
            Patch(facecolor=category_colors["Location"], label="Location"),
            Patch(facecolor=category_colors["Account Type"], label="Account Type"),
            Patch(facecolor=category_colors["Amount"], label="Amount"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

        if save_fig:
            save_name = f"{plot_prefix}_attribution_feature_importance.png"
            fig.tight_layout()
            out_dir = os.path.dirname(save_name)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            fig.savefig(save_name, dpi=300, bbox_inches="tight")
            if save_pdf:
                fig.savefig(save_name.replace(".png", ".pdf"), bbox_inches="tight")
            plt.close(fig)
            print(f"Attribution feature importance plot saved successfully to {save_name}")
    except Exception as e:
        traceback.print_exc()
        print(f"Error plotting attribution feature importance: {e}")


def plot_all_attribution_plots(attribution_metrics, plot_prefix="", save_fig=False, save_pdf=False):
    """
    Generate all attribution plots from pre-computed metrics using Captum.

    Args:
        attribution_metrics: Dictionary containing attribution metrics from compute_attributions
        plot_prefix: Prefix for saved plot files
        save_fig: Whether to save the plots
        save_pdf: If True, also save PDF versions (e.g. for use from plot_shap_example.py).
    """
    plot_attribution_summary(attribution_metrics, plot_prefix, save_fig, save_pdf)
    plot_attribution_feature_importance(attribution_metrics, plot_prefix, save_fig, save_pdf)


def compute_attributions(model, test_features, test_labels, n_samples=100, plot_prefix="", feature_names=None):
    """
    Compute feature attributions using Captum library for PyTorch models.

    This function uses Captum's IntegratedGradients and GradientShap methods to compute
    feature importance.

    Args:
        model: Trained PyTorch model
        test_features: Test feature data
        test_labels: Test label data
        n_samples: Number of samples to use for attribution computation (for performance)
        plot_prefix: Prefix for saved plot files
        feature_names: List of feature names/column names to display in plots

    Returns:
        dict: Dictionary containing attribution metrics
    """
    try:
        # Get device from model
        device = next(model.parameters()).device

        # Sample a subset of test data for attribution computation (for performance)
        if len(test_features) > n_samples:
            indices = np.random.choice(len(test_features), n_samples, replace=False)
            sample_features = test_features[indices]
            sample_labels = test_labels[indices]
        else:
            sample_features = test_features
            sample_labels = test_labels

        # Convert to PyTorch tensors
        sample_features_tensor = torch.FloatTensor(sample_features).to(device)
        sample_labels_tensor = torch.LongTensor(sample_labels).to(device)

        # Create a background dataset for GradientShap (using a subset of the data)
        background_size = min(50, len(sample_features))
        background_indices = np.random.choice(len(sample_features), background_size, replace=False)
        background_data = torch.FloatTensor(sample_features[background_indices]).to(device)

        # Set model to evaluation mode
        model.eval()

        # Compute attributions using IntegratedGradients
        ig = IntegratedGradients(model)
        attributions_ig = ig.attribute(sample_features_tensor, target=sample_labels_tensor, n_steps=50)

        # Compute attributions using GradientShap for comparison
        gs = GradientShap(model)
        attributions_gs = gs.attribute(
            sample_features_tensor,
            baselines=background_data,
            target=sample_labels_tensor,
        )

        # Use IntegratedGradients as primary attribution method
        attributions = attributions_ig.cpu().detach().numpy()

        # Create feature names for all features
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(sample_features.shape[1])]
        elif len(feature_names) != sample_features.shape[1]:
            print(
                f"Warning: feature_names length ({len(feature_names)}) doesn't match number of features ({sample_features.shape[1]})"
            )
            feature_names = [f"Feature_{i}" for i in range(sample_features.shape[1])]

        print(f"Using feature names: {feature_names}")
        print(f"Attributions shape: {attributions.shape}")
        print(f"Sample features shape: {sample_features.shape}")
        print(f"Background data shape: {background_data.shape}")

        # Generate plots only when plot_prefix is set (avoids writing to cwd when prefix is "")
        if plot_prefix:
            plot_all_attribution_plots(
                {
                    "attributions": attributions,
                    "sample_features": sample_features,
                    "feature_names": feature_names,
                },
                plot_prefix,
                save_fig=True,
            )

        # Compute feature importance metrics for the return value
        attributions_for_importance = attributions
        if len(attributions_for_importance.shape) == 3:
            # If 3D array (samples, features, classes), take mean across classes
            attributions_for_importance = np.mean(attributions_for_importance, axis=2)

        feature_importance = np.mean(np.abs(attributions_for_importance), axis=0)
        total_importance = np.sum(feature_importance)

        # Create metrics dictionary (keeping similar structure to SHAP for compatibility)
        attribution_metrics = {
            "attributions": attributions,
            "sample_features": sample_features,
            "feature_names": feature_names,
            "feature_importance": feature_importance,
            "total_importance": float(total_importance),
            "samples_used": len(sample_features),
            # Keep SHAP-like keys for backward compatibility
            "shap_values": attributions,
            "shap_sample_features": sample_features,
            "shap_feature_names": feature_names,
            "shap_feature_importance": feature_importance,
            "shap_total_importance": float(total_importance),
            "shap_samples_used": len(sample_features),
        }

        # Save the attribution values to a file when a prefix is given
        if plot_prefix:
            np.save(f"{plot_prefix}_attribution_metrics.npy", attribution_metrics)

        return attribution_metrics

    except Exception as e:
        print(f"Error computing attributions: {e}")
        traceback.print_exc()
        # Return default metrics if attribution computation fails
        return {}


# Alias for backward compatibility
def compute_shapley_values(model, test_features, test_labels, n_samples=100, plot_prefix="", feature_names=None):
    """
    Backward compatibility alias for compute_attributions.
    """
    return compute_attributions(model, test_features, test_labels, n_samples, plot_prefix, feature_names)


class MLflowCallback:
    """
    Custom PyTorch callback for logging training metrics to MLflow.

    This callback logs training and validation metrics at the end of each epoch
    to the provided MLflow writer.
    """

    def __init__(self, mlflow_writer):
        self.mlflow_writer = mlflow_writer
        self.global_epoch = 0

    def log_metrics(self, train_loss, train_accuracy, val_accuracy=None, current_lr=None):
        """
        Log training metrics to MLflow.

        Args:
            train_loss: Training loss
            train_accuracy: Training accuracy
            val_accuracy: Validation accuracy (optional)
            current_lr: Current learning rate (optional)
        """
        # Log training metrics
        self.mlflow_writer.log_metric("train_loss", train_loss, self.global_epoch)
        self.mlflow_writer.log_metric("train_accuracy", train_accuracy, self.global_epoch)

        # Log validation metrics if available
        if val_accuracy is not None:
            self.mlflow_writer.log_metric("val_accuracy", val_accuracy, self.global_epoch)

        # Log current learning rate if available
        if current_lr is not None:
            self.mlflow_writer.log_metric("learning_rate", current_lr, self.global_epoch)

        self.global_epoch += 1


class MetricsCollectionFilter(DXOFilter):
    """
    A DXO filter that collects and stores metrics from federated learning clients.

    This filter processes DXO objects containing metrics from clients during federated
    learning rounds and stores them in a structured format for later analysis.

    The all_metrics dictionary has the following structure:

    all_metrics = {
        "round0": {
            "client_1": {
                # Metrics from client_1 in round 0
                # Structure depends on what metrics are sent in dxo.meta["initial_metrics"]
                # Common keys might include:
                # - "train_loss": float
                # - "train_accuracy": float
                # - "val_accuracy": float
                # - "attribution_metrics": dict (from compute_attributions)
                # - "shap_values": numpy array
                # - "feature_importance": numpy array
                # - Any other metrics sent by the client
            },
            "client_2": {
                # Metrics from client_2 in round 0
                # Same structure as client_1
            },
            # ... more clients
        },
        "round1": {
            "client_1": {
                # Metrics from client_1 in round 1
            },
            "client_2": {
                # Metrics from client_2 in round 1
            },
            # ... more clients
        },
        # ... more rounds
    }

    The exact structure of individual client metrics depends on what is sent
    in the DXO's meta["initial_metrics"] field. This typically includes:
    - Training metrics (loss, accuracy)
    - Validation metrics
    - Attribution/SHAP values and feature importance
    - Any other custom metrics computed by the client

    The metrics are saved to a .npy file with file locking to prevent concurrent
    access issues in multi-client environments.
    """

    def __init__(self):
        super().__init__(
            supported_data_kinds=[DataKind.WEIGHT_DIFF, DataKind.WEIGHTS],
            data_kinds_to_filter=None,
        )

        # Global dictionary to store the clients' metrics for each round
        self.all_metrics = {}
        self._save_path = None

    def _safe_save_with_lock(self, data, file_path, max_retries=5, retry_delay=0.1):
        """
        Safely save data to file with file locking to prevent concurrent access.

        Args:
            data: Data to save
            file_path: Path to save the file
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            bool: True if successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Open file for writing with exclusive lock
                with open(file_path, "wb") as f:
                    # Try to acquire exclusive lock (non-blocking)
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                    # Save the data
                    np.save(f, data)

                    # Lock is automatically released when file is closed

                return True

            except (OSError, IOError) as e:
                if attempt < max_retries - 1:
                    self.log_warning(
                        None,
                        f"Failed to acquire lock for {file_path}, attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay}s...",
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.log_error(
                        None,
                        f"Failed to save {file_path} after {max_retries} attempts: {e}",
                    )
                    return False
            except Exception as e:
                self.log_error(None, f"Unexpected error saving {file_path}: {e}")
                return False

        return False

    def process_dxo(self, dxo, shareable, fl_ctx) -> Union[None, "DXO"]:
        """
        Process DXO objects, extract FLModels, store them globally, and dump to JSON.

        Args:
            dxo: The DXO object received
            shareable: The shareable object
            fl_ctx: The FL context

        Returns:
            The processed DXO object
        """
        try:
            if self._save_path is None:
                workspace = fl_ctx.get_engine().get_workspace()
                app_root = workspace.get_app_dir(fl_ctx.get_job_id())
                self._save_path = os.path.join(app_root, "metrics.npy")

            # get shap metrics from dxo
            metrics = dxo.meta["initial_metrics"]
            self.log_info(fl_ctx, f"Collecting metrics {metrics.keys()}")

            # Sim/server DXO path often omits CURRENT_ROUND on fl_ctx; use client FLModel meta (train/client.py).
            current_round = fl_ctx.get_prop(FLMetaKey.CURRENT_ROUND)
            if current_round is None:
                current_round = dxo.meta.get("current_round")
            if current_round is None:
                self.log_warning(fl_ctx, "Could not resolve round (fl_ctx, dxo.meta['current_round']")

            peer_context = fl_ctx.get_peer_context()
            client_name = peer_context.get_identity_name()

            if f"round{current_round}" not in self.all_metrics:
                self.all_metrics[f"round{current_round}"] = {}
            self.all_metrics[f"round{current_round}"][client_name] = metrics

            # Dump global dictionary to file with file locking
            success = self._safe_save_with_lock(self.all_metrics, self._save_path)

            if success:
                self.log_info(
                    fl_ctx,
                    f"Saved SHAP metrics for round {current_round} and client {client_name} at {self._save_path}",
                )
            else:
                self.log_error(
                    fl_ctx,
                    f"Failed to save SHAP metrics for round {current_round} and client {client_name} at {self._save_path}",
                )
        except Exception as e:
            self.log_error(fl_ctx, f"Error processing DXO in MetricsCollectionFilter: {e}")

        # Return the DXO unchanged
        return dxo


def load_numpy(file_path):
    """
    Load SHAP metrics from a saved .npy file.

    Args:
        file_path: Path to the saved SHAP metrics file

    Returns:
        dict: Loaded SHAP metrics
    """
    try:
        return np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")


def str2bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
