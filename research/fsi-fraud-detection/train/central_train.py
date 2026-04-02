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
import glob
import os

import numpy as np
import sklearn
import torch
import torch.optim as optim
from misc.data import all_model_parameters, flag, numerical_features, prepare_dataset
from misc.data_io import load_csv_data_from_path, print_directory_tree, validate_data_features
from misc.experiments import data_paths
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from train.model import SimpleNetwork
from train.utils import FocalLoss, compute_shapley_values, evaluate_on_test_datasets

PATH = "pt_model.weights.pth"


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="NVFlare Deep Learning Client for Financial Fraud Detection")
    parser.add_argument(
        "--data_selection",
        type=str,
        required=True,
        help="Data selection",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 0.0005)")
    parser.add_argument(
        "--data_features",
        type=list,
        default=all_model_parameters,
        help="List of data features (default: `all_model_parameters`)",
    )
    parser.add_argument("--flag", type=str, default=flag, help="Flag for the data (default: `flag`)")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory (default: `results`)")
    parser.add_argument(
        "--width_factor", type=float, default=1.0, help="Scale factor for hidden layer sizes (default: 1.0)"
    )
    parser.add_argument(
        "--dropout_p", type=float, default=0.0, help="Dropout probability after hidden layers (default: 0.0)"
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma (default: 2.0)")
    parser.add_argument(
        "--shap_every",
        type=int,
        default=-1,
        help="Compute SHAP/attributions every N epochs; -1 = last epoch only, 0 = never (default: -1)",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        help="Enable cuDNN benchmark for faster conv/matmul (non-deterministic)",
    )
    args = parser.parse_args()

    if args.data_selection not in data_paths:
        raise ValueError(
            f"Data selection {args.data_selection} not found in data_paths. Available selections: {data_paths.keys()}"
        )

    os.makedirs(args.results_dir, exist_ok=True)

    client_name = "central"

    # Display directory tree of /workspace/dataset
    print(f"\n=== Directory tree of {client_name} at /workspace/dataset ===")
    print("/workspace/dataset")
    print_directory_tree("/workspace/dataset", max_depth=3, endswith=".csv")
    print("=" * 45 + "\n")

    # Load CSV data using the utility function
    print(f"Loading data for client {client_name} with selection {args.data_selection}")

    data_selection_paths = data_paths[args.data_selection][client_name]
    data_root = data_selection_paths["data_root"]
    train_data_path = os.path.join(data_root, data_selection_paths["train_data_path"])
    test_data_path_pattern = os.path.join(data_root, data_selection_paths["test_data_path"])

    # Find training files
    glob_result = sorted(glob.glob(train_data_path, recursive=True))
    if len(glob_result) == 0:
        raise FileNotFoundError(f"No valid train files found at: {train_data_path}")
    train_data_path = glob_result
    print(f"Train data path: {train_data_path}")

    # Check if test_data_path contains wildcards
    if "*" in test_data_path_pattern or "?" in test_data_path_pattern:
        # Use glob to find matching files
        test_data_paths = sorted(glob.glob(test_data_path_pattern))
        if not test_data_paths:
            raise FileNotFoundError(f"No test files found matching pattern: {test_data_path_pattern}")
        print(f"Found {len(test_data_paths)} test files matching pattern: {test_data_path_pattern}")
        for path in test_data_paths:
            print(f"  - {path}")

        # assert len(test_data_paths) == 25, "Expected 25 test files, got " + str(len(test_data_paths))
        assert len(test_data_paths) == 20, "Expected 20 test files, got " + str(
            len(test_data_paths)
        )  # Datasets later than 3/2/2026
    else:
        # Single test file
        if not os.path.isfile(test_data_path_pattern):
            raise FileNotFoundError(f"No valid test filepath at: {test_data_path_pattern}")
        test_data_paths = [test_data_path_pattern]
        print(f"Test data path: {test_data_path_pattern}")

    print(f"Train data path: {train_data_path}")

    try:
        # Load CSV data using the utility function
        df_train = load_csv_data_from_path(
            data_path=train_data_path,
            data_features=None,  # all features are loaded
        )

        # Load all test files
        test_dataframes = {}
        for test_path in test_data_paths:
            # Extract a meaningful name from the file path
            test_name = os.path.basename(test_path).replace(".csv", "")
            df_test_tmp = load_csv_data_from_path(
                data_path=test_path,
                data_features=None,  # all features are loaded
            )
            test_dataframes[test_name] = df_test_tmp
            print(f"Loaded test dataset '{test_name}' with {len(df_test_tmp)} samples")

        # Scale data
        global_scaler = sklearn.preprocessing.StandardScaler()
        global_scaler = global_scaler.fit(prepare_dataset(df_train).loc[:, numerical_features])

        # Prepare dataset
        print(f"Preparing data with features: {all_model_parameters}")
        df_train = prepare_dataset(df_train, scaler=global_scaler)
        if df_train.empty:
            raise ValueError(f"Train dataset is empty after preprocessing for client {client_name}")

        # Prepare all test datasets
        for test_name, df_test in test_dataframes.items():
            prepared_test_df = prepare_dataset(df_test, scaler=global_scaler)
            if prepared_test_df.empty:
                raise ValueError(f"Test dataset '{test_name}' is empty after preprocessing for client {client_name}")
            test_dataframes[test_name] = prepared_test_df

        # Validate the loaded data
        validate_data_features(df_train, args.data_features)
        for test_name, df_test in test_dataframes.items():
            validate_data_features(df_test, args.data_features)
    except Exception as e:
        raise Exception(f"Load data for client {client_name} failed! {e}")

    train_features = df_train[args.data_features].values
    train_labels = df_train[args.flag].values

    # Extract features and labels for all test datasets
    test_datasets = {}
    for test_name, df_test in test_dataframes.items():
        test_features = df_test[args.data_features].values
        test_labels = df_test[flag].values
        test_datasets[test_name] = {"features": test_features, "labels": test_labels}

    # Get the number of features for model input shape
    n_features = train_features.shape[1]
    n_classes = len(np.unique(train_labels))

    print(f"\nLoaded data with {n_features} features:")
    print("train_features: ", train_features.shape)
    print("train_labels: ", train_labels.shape)
    for test_name, test_data in test_datasets.items():
        print(f"test_features ({test_name}): ", test_data["features"].shape)
        print(f"test_labels ({test_name}): ", test_data["labels"].shape)

    # Convert numpy arrays to PyTorch tensors and keep on same device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_features_tensor = torch.FloatTensor(train_features).to(device, non_blocking=True)
    train_labels_tensor = torch.LongTensor(train_labels).to(device, non_blocking=True)

    # Convert all test datasets to tensors
    test_datasets_tensors = {}
    for test_name, test_data in test_datasets.items():
        test_features_tensor = torch.FloatTensor(test_data["features"]).to(device)
        test_labels_tensor = torch.LongTensor(test_data["labels"]).to(device)
        test_datasets_tensors[test_name] = (test_features_tensor, test_labels_tensor)

    # Class weights for imbalanced fraud flag (reduce false negatives by upweighting fraud class)
    class_counts = np.bincount(train_labels, minlength=n_classes)
    class_weights = 1.0 / (class_counts.astype(np.float64) + 1e-8)
    class_weights = class_weights / class_weights.sum() * n_classes
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(
        f"Class counts (fraud flag): {dict(enumerate(class_counts))}, loss weights: {class_weights.round(4).tolist()}"
    )

    # Create model, optimizer, and loss function
    model = SimpleNetwork(
        input_size=n_features,
        num_classes=n_classes,
        width_factor=args.width_factor,
        dropout_p=args.dropout_p,
    ).to(device)
    if device.type == "cuda" and args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled (faster, non-deterministic)")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=args.focal_gamma)

    print(f"Model created with {n_features} input features and {n_classes} classes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Loss: FocalLoss(gamma={args.focal_gamma}, alpha=class_weights)")
    print(f"Using cosine annealing LR scheduler: initial_lr={args.lr}, T_max={args.epochs}, eta_min={args.lr * 0.1}")

    # Create cosine annealing learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.1,
    )

    model.to(device)

    all_metrics = {}
    best_f1 = -1.0
    best_epoch = 0
    best_model_path = os.path.join(args.results_dir, "best_model.pth")

    # Training loop: data already on GPU so pin_memory must be False (pinning is for CPU tensors only)
    train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
    )

    for epoch in range(args.epochs):
        # ── Training ──────────────────────────────────────────────────────────
        print(f"\n=== Training Epoch {epoch + 1}/{args.epochs} ===")

        model.train()
        # Accumulate on GPU to avoid per-batch .item() CPU sync (sync only when logging)
        running_loss = torch.tensor(0.0, device=device)
        running_correct = torch.tensor(0, dtype=torch.long, device=device)
        running_total = 0
        n_batches = 0
        for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                running_loss += loss.detach()
                _, predicted = torch.max(outputs, 1)
                running_correct += (predicted == batch_labels).sum()
            running_total += batch_labels.size(0)
            n_batches += 1

            if batch_idx > 0 and batch_idx % 1000 == 0:
                current_lr = scheduler.get_last_lr()[0]
                avg_loss = (running_loss / n_batches).item()
                acc = running_correct.item() / running_total
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | Batch {batch_idx + 1}/{len(train_loader)} | Loss: {avg_loss:.6f} | Acc: {acc:.4f} | LR: {current_lr}"
                )

        # Step the learning rate scheduler
        scheduler.step()

        # ── Evaluation (after training) ────────────────────────────────────
        print(f"\n=== Evaluating Global Model after Epoch {epoch + 1}/{args.epochs} ===")
        global_metrics_all = evaluate_on_test_datasets(model, test_datasets_tensors, device)

        first_test_name = list(global_metrics_all.keys())[0]

        for test_name, metrics in global_metrics_all.items():
            print(f"\nLocal Model Metrics on '{test_name}' ({metrics['num_samples']} samples):")
            print(
                f"  Accuracy: {metrics['accuracy']:.4f} | "
                f"Precision: {metrics['precision']:.4f} | "
                f"Recall: {metrics['recall']:.4f} | "
                f"F1-score: {metrics['f1_score']:.4f}"
            )

        # Build comprehensive metrics dictionary
        metrics = {}
        for test_name, test_metrics in global_metrics_all.items():
            metrics[f"global_{test_name}_accuracy"] = test_metrics["accuracy"]
            metrics[f"global_{test_name}_precision"] = test_metrics["precision"]
            metrics[f"global_{test_name}_recall"] = test_metrics["recall"]
            metrics[f"global_{test_name}_f1_score"] = test_metrics["f1_score"]
            metrics[f"global_{test_name}_confusion_matrix"] = test_metrics["confusion_matrix"]

        # Compute SHAP/attributions only when requested (expensive: many forward/backward passes)
        run_shap = (args.shap_every == -1 and epoch == args.epochs - 1) or (
            args.shap_every > 0 and (epoch + 1) % args.shap_every == 0
        )
        shap_metrics = None
        if run_shap:
            print("Computing Shapley values (attributions)...")
            first_test_features = test_datasets[first_test_name]["features"]
            first_test_labels = test_datasets[first_test_name]["labels"]
            plot_prefix = os.path.join(args.results_dir, f"epoch{epoch}")
            shap_metrics = compute_shapley_values(
                model,
                first_test_features,
                first_test_labels,
                n_samples=100,
                plot_prefix=plot_prefix,
                feature_names=args.data_features,
            )
            if shap_metrics:
                print(f"SHAP completed. Used {shap_metrics['shap_samples_used']} samples.")
            else:
                print("SHAP computation failed. Skipping SHAP metrics.")
        metrics["shap_metrics"] = shap_metrics

        # Save metrics to file (mimic MetricsCollectionFilter)
        all_metrics[f"round{epoch}"] = {"central": metrics}

        # Early-stopping: track best mean F1 across all test datasets
        mean_f1 = float(np.mean([global_metrics_all[tn]["f1_score"] for tn in global_metrics_all]))
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> New best model at epoch {epoch + 1}: mean F1={best_f1:.4f} (saved to {best_model_path})")

        # Persist metrics after every epoch so partial results are available
        np.save(os.path.join(args.results_dir, "metrics.npy"), all_metrics)

    # Load best model and run a final evaluation tagged as "best"
    print(f"\n=== Best model was at epoch {best_epoch + 1} with mean F1={best_f1:.4f} ===")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_global_metrics = evaluate_on_test_datasets(model, test_datasets_tensors, device)
    best_metrics: dict = {}
    for test_name, test_metrics in best_global_metrics.items():
        best_metrics[f"global_{test_name}_accuracy"] = test_metrics["accuracy"]
        best_metrics[f"global_{test_name}_precision"] = test_metrics["precision"]
        best_metrics[f"global_{test_name}_recall"] = test_metrics["recall"]
        best_metrics[f"global_{test_name}_f1_score"] = test_metrics["f1_score"]
        best_metrics[f"global_{test_name}_confusion_matrix"] = test_metrics["confusion_matrix"]
    all_metrics["best"] = {
        "central": best_metrics,
        "epoch": best_epoch,
        "mean_f1": best_f1,
    }

    # Save all metrics to file
    np.save(os.path.join(args.results_dir, "metrics.npy"), all_metrics)


if __name__ == "__main__":
    main()
