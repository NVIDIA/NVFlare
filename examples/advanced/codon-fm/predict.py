#!/usr/bin/env python3
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
Prediction script for evaluating the trained federated model on test data
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

# Add train-nn directory to path to import model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "jobs/train-nn"))
from model import Network


def load_test_embeddings_and_targets(data_dir, test_csv_path):
    """
    Load test embeddings and targets.
    If embeddings don't exist, we need to generate them first.

    Args:
        data_dir: Directory containing federated data
        test_csv_path: Path to test_data.csv

    Returns:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        targets: numpy array of shape (n_samples,)
    """
    # Check if test embeddings exist
    test_embeddings_path = os.path.join(data_dir, "test_data.embeddings.npy")
    test_targets_path = os.path.join(data_dir, "test_data.targets.npy")

    if os.path.exists(test_embeddings_path) and os.path.exists(test_targets_path):
        print(f"Loading test embeddings from {test_embeddings_path}")
        embeddings = np.load(test_embeddings_path)
        targets = np.load(test_targets_path)
        print(f"Loaded {len(embeddings)} test samples")
        return embeddings, targets
    else:
        print(f"⚠️  Test embeddings not found at {test_embeddings_path}")
        print("Please generate test embeddings first using the embeddings workflow")
        print("Or provide the path to existing test embeddings")
        sys.exit(1)


def load_model(checkpoint_path, device="cpu"):
    """
    Load the trained model from checkpoint

    Args:
        checkpoint_path: Path to the model checkpoint (.pth file)
        device: Device to load model on

    Returns:
        model: Loaded PyTorch model
    """
    model = Network()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle NVFlare checkpoint format
    if "train_conf" in checkpoint:
        print("Training configuration:")
        print(checkpoint["train_conf"])
        print()

    # Extract model weights
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Assume the checkpoint is the state dict itself
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"✓ Model loaded from {checkpoint_path}")
    return model


def predict(model, embeddings, device="cpu", batch_size=64):
    """
    Run predictions on embeddings

    Args:
        model: Trained PyTorch model
        embeddings: numpy array of embeddings
        device: Device to run predictions on
        batch_size: Batch size for predictions

    Returns:
        predictions: numpy array of predictions
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i : i + batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
            preds = model(batch_tensor)
            predictions.append(preds.cpu().numpy())

    return np.concatenate(predictions)


def evaluate_predictions(targets, predictions):
    """
    Evaluate predictions using various metrics

    Args:
        targets: Ground truth values
        predictions: Model predictions

    Returns:
        dict with evaluation metrics
    """
    r2 = r2_score(targets, predictions)
    pearson_r, pearson_p = pearsonr(targets, predictions)
    mse = np.mean((targets - predictions) ** 2)
    mae = np.mean(np.abs(targets - predictions))

    results = {"r2": r2, "pearson_r": pearson_r, "pearson_p": pearson_p, "mse": mse, "mae": mae, "rmse": np.sqrt(mse)}

    return results


def plot_results(targets, predictions, metrics, save_path=None):
    """
    Plot prediction results

    Args:
        targets: Ground truth values
        predictions: Model predictions
        metrics: Dictionary of evaluation metrics
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    fig.suptitle("Translation Efficiency Prediction Results", fontsize=16)

    # Scatter plot: Predicted vs Actual
    axes[0].scatter(targets, predictions, alpha=0.5, s=20)

    # Add diagonal line (perfect prediction)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

    axes[0].set_xlabel("Actual Translation Efficiency", fontsize=12)
    axes[0].set_ylabel("Predicted Translation Efficiency", fontsize=12)
    axes[0].set_title("Predicted vs Actual Values", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Add metrics text
    metrics_text = f"R² = {metrics['r2']:.3f}\n"
    metrics_text += f"Pearson r = {metrics['pearson_r']:.3f}\n"
    metrics_text += f"RMSE = {metrics['rmse']:.3f}\n"
    metrics_text += f"MAE = {metrics['mae']:.3f}"

    axes[0].text(
        0.05,
        0.95,
        metrics_text,
        transform=axes[0].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=11,
    )

    # Target distribution
    axes[1].hist(targets, bins=50, alpha=0.7, edgecolor="black", label="Actual")
    axes[1].hist(predictions, bins=50, alpha=0.7, edgecolor="black", label="Predicted")
    axes[1].axvline(
        targets.mean(), color="blue", linestyle="--", label=f"Actual Mean = {targets.mean():.3f}", linewidth=2
    )
    axes[1].axvline(
        predictions.mean(),
        color="orange",
        linestyle="--",
        label=f"Predicted Mean = {predictions.mean():.3f}",
        linewidth=2,
    )
    axes[1].set_xlabel("Translation Efficiency", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Distribution Comparison", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run predictions on test data using trained federated model")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/data/nvflare/simulation/train-nn/server/simulate_job/app_server/FL_global_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_dir", type=str, default="/data/federated_data", help="Directory containing test embeddings"
    )
    parser.add_argument(
        "--test_embeddings", type=str, default=None, help="Path to test embeddings .npy file (overrides data_dir)"
    )
    parser.add_argument(
        "--test_targets", type=str, default=None, help="Path to test targets .npy file (overrides data_dir)"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for predictions")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run predictions on",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/data/predictions", help="Directory to save predictions and plots"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Translation Efficiency Prediction")
    print("=" * 80)
    print()

    # Load model
    print(f"Using model: {args.checkpoint_path}")
    model = load_model(args.checkpoint_path, device=args.device)

    # Load test data
    print()
    print("Loading test data...")

    if args.test_embeddings and args.test_targets:
        embeddings = np.load(args.test_embeddings)
        targets = np.load(args.test_targets)
        print(f"Loaded {len(embeddings)} test samples from provided paths")
    else:
        test_csv = os.path.join(args.data_dir, "test_data.csv")
        embeddings, targets = load_test_embeddings_and_targets(args.data_dir, test_csv)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Targets shape: {targets.shape}")
    print()

    # Run predictions
    print("Running predictions...")
    predictions = predict(model, embeddings, device=args.device, batch_size=args.batch_size)
    print(f"✓ Generated {len(predictions)} predictions")
    print()

    # Evaluate
    print("Evaluating predictions...")
    metrics = evaluate_predictions(targets, predictions)

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"R² Score:         {metrics['r2']:.4f}")
    print(f"Pearson r:        {metrics['pearson_r']:.4f} (p-value: {metrics['pearson_p']:.2e})")
    print(f"RMSE:             {metrics['rmse']:.4f}")
    print(f"MAE:              {metrics['mae']:.4f}")
    print(f"MSE:              {metrics['mse']:.4f}")
    print("=" * 80)
    print()

    # Save predictions
    predictions_file = os.path.join(args.output_dir, "predictions.npy")
    np.save(predictions_file, predictions)
    print(f"✓ Predictions saved to {predictions_file}")

    # Save metrics
    metrics_file = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("Translation Efficiency Prediction Metrics\n")
        f.write("=" * 80 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key:20s}: {value:.6f}\n")
    print(f"✓ Metrics saved to {metrics_file}")

    # Save detailed results
    results_df = pd.DataFrame(
        {
            "actual": targets,
            "predicted": predictions,
            "error": targets - predictions,
            "abs_error": np.abs(targets - predictions),
        }
    )
    results_csv = os.path.join(args.output_dir, "detailed_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"✓ Detailed results saved to {results_csv}")

    # Plot results
    print()
    print("Generating plots...")
    plot_file = os.path.join(args.output_dir, "prediction_results.png")
    plot_results(targets, predictions, metrics, save_path=plot_file)

    print()
    print("✓ Prediction completed successfully!")


if __name__ == "__main__":
    main()
