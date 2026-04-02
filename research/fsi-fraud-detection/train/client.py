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

import argparse
import copy
import glob
import os
import time

import numpy as np
import sklearn
import torch
import torch.optim as optim
from misc.data import all_model_parameters, flag, numerical_features, prepare_dataset
from misc.data_io import load_csv_data_from_path, print_directory_tree, validate_data_features
from misc.experiments import data_paths
from model import SimpleNetwork

# Opacus for differential privacy
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from train.utils import FocalLoss, str2bool
from utils import MLflowCallback, compute_shapley_values, evaluate_on_test_datasets

# (1) import nvflare client API
import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import ParamsType
from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss
from nvflare.client.tracking import MLflowWriter

# install dependencies
# os.system("python -m pip install opacus")
# os.system("python -m pip install captum")
# result = os.system("python -m pip install numpy==1.26.4")
# print(f"Pip Install Result: {result}")


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
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 0.0005)")
    parser.add_argument(
        "--data_features",
        type=list,
        default=all_model_parameters,
        help="List of data features (default: `all_model_parameters`)",
    )
    parser.add_argument("--flag", type=str, default=flag, help="Flag for the data (default: `flag`)")
    parser.add_argument("--fedproxloss_mu", type=float, default=0, help="FedProx loss mu (default: 0)")
    parser.add_argument(
        "--local_only", type=str2bool, default=False, help="Whether to only train locally (default: False)"
    )
    # Differential privacy arguments
    parser.add_argument(
        "--enable_dp", type=str2bool, default=False, help="Enable differential privacy (default: False)"
    )
    parser.add_argument(
        "--target_epsilon", type=float, default=10.0, help="Target privacy budget epsilon (default: 10.0)"
    )
    parser.add_argument("--target_delta", type=float, default=1e-5, help="Target privacy budget delta (default: 1e-5)")
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Max gradient norm for DP clipping (default: 1.0)"
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma (default: 2.0)")
    parser.add_argument(
        "--shap_every",
        type=int,
        default=-1,
        help="Compute SHAP every N rounds; -1 = last round only, 0 = never (default: -1)",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        help="Enable cuDNN benchmark for faster kernels (non-deterministic)",
    )
    parser.add_argument(
        "--balanced_sampling",
        type=str2bool,
        default=False,
        help="Use class-balanced sampling (WeightedRandomSampler) during training (default: False)",
    )
    args = parser.parse_args()

    if args.data_selection not in data_paths:
        raise ValueError(
            f"Data selection {args.data_selection} not found in data_paths. Available selections: {data_paths.keys()}"
        )

    print(f"args.local_only: {args.local_only}")
    print(f"args.fedproxloss_mu: {args.fedproxloss_mu}")
    print(f"args.enable_dp: {args.enable_dp}")
    if args.enable_dp:
        print(f"  - target_epsilon: {args.target_epsilon}")
        print(f"  - target_delta: {args.target_delta}")
        print(f"  - max_grad_norm: {args.max_grad_norm}")
    if args.local_only and args.fedproxloss_mu > 0:
        raise ValueError("FedProx loss is not supported when local only training is enabled")

    # (2) initializes NVFlare client API
    flare.init()
    client_name = flare.get_site_name()

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
    if data_selection_paths["scaling_data_path"] is not None:
        scaling_data_path = os.path.join(data_root, data_selection_paths["scaling_data_path"])
    else:
        scaling_data_path = None

    if not os.path.isfile(train_data_path):
        raise FileNotFoundError(f"No valid train filepath at: {train_data_path}")

    # Check if test_data_path contains wildcards
    if "*" in test_data_path_pattern or "?" in test_data_path_pattern:
        # Use glob to find matching files
        test_data_paths = sorted(glob.glob(test_data_path_pattern))
        if not test_data_paths:
            raise FileNotFoundError(f"No test files found matching pattern: {test_data_path_pattern}")
        print(f"Found {len(test_data_paths)} test files matching pattern: {test_data_path_pattern}")
        for path in test_data_paths:
            print(f"  - {path}")

        # assert len(test_data_paths) == 5, "Expected 5 test files, got " + str(len(test_data_paths))
        assert len(test_data_paths) == 4, "Expected 4 test files, got " + str(len(test_data_paths))  # new data 3/3/2026
    else:
        # Single test file
        if not os.path.isfile(test_data_path_pattern):
            raise FileNotFoundError(f"No valid test filepath at: {test_data_path_pattern}")
        test_data_paths = [test_data_path_pattern]
        print(f"Test data path: {test_data_path_pattern}")

    if scaling_data_path is not None and not os.path.isfile(scaling_data_path):
        raise FileNotFoundError(f"No valid scaling filepath at: {scaling_data_path}")

    print(f"Train data path: {train_data_path}")
    print(f"Scaling data path: {scaling_data_path}")

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

        # Load scaler data if available
        # Load and concatenate all scaler data files
        if scaling_data_path is not None and os.path.isfile(scaling_data_path):
            df_scaling = load_csv_data_from_path(
                data_path=scaling_data_path,
                data_features=None,  # all features are loaded
            )

            # Concatenate all scaler dataframes
            global_scaler = sklearn.preprocessing.StandardScaler()
            global_scaler = global_scaler.fit(prepare_dataset(df_scaling).loc[:, numerical_features])
        else:
            print("[WARNING] No valid scaler data files found")
            df_scaling = None
            global_scaler = None

        # Prepare dataset
        print(f"Preparing data with features: {all_model_parameters}")
        df_train = prepare_dataset(df_train, scaler=global_scaler)

        # Prepare all test datasets
        for test_name, df_test in test_dataframes.items():
            test_dataframes[test_name] = prepare_dataset(df_test, scaler=global_scaler)

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

    # Convert numpy arrays to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_features_tensor = torch.FloatTensor(train_features).to(device, non_blocking=True)
    train_labels_tensor = torch.LongTensor(train_labels).to(device, non_blocking=True)
    # Data already on GPU so pin_memory must be False (only CPU tensors can be pinned)
    train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
    if args.balanced_sampling:
        class_counts = np.bincount(train_labels, minlength=n_classes)
        class_weights = 1.0 / (class_counts.astype(np.float64) + 1e-8)
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            pin_memory=False,
            num_workers=0,
        )
        print(f"Balanced sampling enabled: class weights for sampler {class_weights.round(4).tolist()}")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=0,
        )
    # Convert all test datasets to tensors
    test_datasets_tensors = {}
    for test_name, test_data in test_datasets.items():
        test_features_tensor = torch.FloatTensor(test_data["features"]).to(device, non_blocking=True)
        test_labels_tensor = torch.LongTensor(test_data["labels"]).to(device, non_blocking=True)
        test_datasets_tensors[test_name] = (test_features_tensor, test_labels_tensor)

    # Class weights for imbalanced fraud flag (reduce false negatives by upweighting fraud class)
    class_counts = np.bincount(train_labels, minlength=n_classes)
    class_weights = 1.0 / (class_counts.astype(np.float64) + 1e-8)
    class_weights = class_weights / class_weights.sum() * n_classes
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(
        f"Class counts (fraud flag): {dict(enumerate(class_counts))}, loss weights: {class_weights.round(4).tolist()}"
    )

    # Create model, optimizer, and loss function (Focal loss for imbalanced fraud)
    model = SimpleNetwork(input_size=n_features, num_classes=n_classes).to(device)
    if device.type == "cuda" and args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled (faster, non-deterministic)")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=args.focal_gamma)
    if args.fedproxloss_mu > 0:
        print(f"using FedProx loss with mu {args.fedproxloss_mu}")
        criterion_prox = PTFedProxLoss(mu=args.fedproxloss_mu)

    print(f"Model created with {n_features} input features and {n_classes} classes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Loss: FocalLoss(gamma={args.focal_gamma}, alpha=class_weights)")
    print(f"Using cosine annealing LR scheduler: initial_lr={args.lr}, T_max={args.epochs}, eta_min={args.lr * 0.1}")

    mlflow = MLflowWriter()

    # Create the callback to log training metrics to MLflow
    mlflow_callback = MLflowCallback(mlflow)

    # (optional) print system info
    system_info = flare.system_info()
    print(f"NVFlare system info: {system_info}")

    # (3) gets FLModel from NVFlare
    while flare.is_running():
        input_model = flare.receive()
        print(f"current_round={input_model.current_round}")

        if args.enable_dp and input_model.current_round == 0:
            print("Validating model for Opacus compatibility...")
            errors = ModuleValidator.validate(model, strict=False)
            if errors:
                print(f"Model has compatibility issues: {errors}")
                print("Attempting to fix model for Opacus...")
                model = ModuleValidator.fix(model)
                print("Model fixed for Opacus compatibility")
                model.to(device)

            print("\n=== Initializing Differential Privacy (Global) ===")
            print(f"Target epsilon: {args.target_epsilon}")
            print(f"Target delta: {args.target_delta}")
            print(f"Max gradient norm: {args.max_grad_norm}")
            print("Note: Privacy budget will be tracked across ALL federated rounds")
            print("=" * 60)

            # Initialize Opacus PrivacyEngine ONLY in first round if DP is enabled
            # This ensures privacy accounting accumulates across all federated rounds
            print("\n=== Configuring Privacy Engine (Round 0) ===")

            # Calculate total epochs across ALL federated rounds
            total_epochs_all_rounds = args.epochs * input_model.total_rounds

            privacy_engine = PrivacyEngine()
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                epochs=total_epochs_all_rounds,  # Total across ALL rounds
                target_epsilon=args.target_epsilon,
                target_delta=args.target_delta,
                max_grad_norm=args.max_grad_norm,
            )
            # Access privacy parameters from optimizer (Opacus 1.5+)
            dp_noise_multiplier = optimizer.noise_multiplier
            sample_rate = optimizer.expected_batch_size / len(train_dataset)
            print(f"Privacy engine initialized for {input_model.total_rounds} federated rounds")
            print(f"Total epochs for privacy accounting: {total_epochs_all_rounds}")
            print(f"Noise multiplier: {dp_noise_multiplier:.4f}")
            print(f"Poisson sampling rate: {sample_rate:.6f}")
            print(f"Privacy budget will accumulate across all {input_model.total_rounds} rounds")
            print("=" * 60)

        if input_model.current_round == 0:
            # Create cosine annealing learning rate scheduler
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.epochs * input_model.total_rounds,
                eta_min=args.lr * 0.1,
            )

        # (4) loads model from NVFlare
        if not args.local_only:
            # Opacus adds "_module." prefix to state dict. Add it here to match local state dict
            if args.enable_dp and hasattr(model, "_module"):
                global_params = {}
                for k, v in input_model.params.items():
                    global_params[f"_module.{k}"] = v
            else:
                global_params = input_model.params
            model.load_state_dict(global_params)
        else:
            print("[WARNING] simulating local only training")
        model.to(device)

        # Copy global model for computing weight differences and for FedProx loss
        global_model = copy.deepcopy(model)
        for param in global_model.parameters():
            param.requires_grad = False

        # (5) evaluate aggregated/received model on all test datasets
        print(f"\n=== Evaluating Global Model (Round {input_model.current_round}) ===")
        global_metrics_all = evaluate_on_test_datasets(model, test_datasets_tensors, device)

        for test_name, metrics in global_metrics_all.items():
            print(f"\nGlobal Model Metrics on '{test_name}' ({metrics['num_samples']} samples):")
            print(
                f"  Accuracy: {metrics['accuracy']:.4f} | "
                f"Precision: {metrics['precision']:.4f} | "
                f"Recall: {metrics['recall']:.4f} | "
                f"F1-score: {metrics['f1_score']:.4f}"
            )

        # Use the first test dataset metrics for the main "accuracy" metric (for backward compatibility)
        first_test_name = list(global_metrics_all.keys())[0]
        test_global_acc = global_metrics_all[first_test_name]["accuracy"]
        test_global_precision = global_metrics_all[first_test_name]["precision"]
        test_global_recall = global_metrics_all[first_test_name]["recall"]
        test_global_f1 = global_metrics_all[first_test_name]["f1_score"]
        test_global_cm = global_metrics_all[first_test_name]["confusion_matrix"]

        # Training loop: accumulate on GPU to avoid per-batch .item() CPU sync
        model.train()
        steps = args.epochs * len(train_loader)
        for epoch in range(args.epochs):
            print(f"Round {input_model.current_round} | Epoch {epoch + 1}/{args.epochs}")

            running_loss = torch.tensor(0.0, device=device)
            running_correct = torch.tensor(0, dtype=torch.long, device=device)
            n_batches = 0
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)

                if args.fedproxloss_mu > 0:
                    fed_prox_loss = criterion_prox(model, global_model)
                    loss += fed_prox_loss

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    running_loss += loss.detach()
                    _, predicted = torch.max(outputs, 1)
                    running_correct += (predicted == batch_labels).sum()
                n_batches += 1

            # Log training metrics and current learning rate (single sync per epoch)
            total_samples = len(train_dataset)
            train_acc = running_correct.item() / total_samples
            avg_loss = (running_loss / max(n_batches, 1)).item()
            current_lr = scheduler.get_last_lr()[0]

            # Log standard metrics
            mlflow_callback.log_metrics(train_loss=avg_loss, train_accuracy=train_acc, current_lr=current_lr)

            # Log epsilon if DP is enabled
            if args.enable_dp and privacy_engine is not None:
                epsilon = privacy_engine.get_epsilon(args.target_delta)
                print(
                    f"  Training metrics: Loss={avg_loss:.4f}, Accuracy={train_acc:.4f}, LR={current_lr:.6f}, ε={epsilon:.2f} (cumulative)"
                )
                # Log privacy epsilon separately
                mlflow.log_metric("privacy_epsilon", epsilon)
            else:
                print(f"  Training metrics: Loss={avg_loss:.4f}, Accuracy={train_acc:.4f}, LR={current_lr:.6f}")

            # Step the learning rate scheduler
            scheduler.step()

        print("Finished Training")
        # get current job_id
        job_id = flare.system_info().get("job_id")
        # Save the unwrapped model if using DP
        if args.enable_dp and hasattr(model, "_module"):
            torch.save(model._module.state_dict(), os.path.join(job_id, PATH))
        else:
            torch.save(model.state_dict(), os.path.join(job_id, PATH))

        # Final local evaluation on all test datasets
        print(f"\n=== Evaluating Local Model (Round {input_model.current_round}) ===")
        local_metrics_all = evaluate_on_test_datasets(model, test_datasets_tensors, device)

        for test_name, metrics in local_metrics_all.items():
            print(f"\nLocal Model Metrics on '{test_name}' ({metrics['num_samples']} samples):")
            print(
                f"  Accuracy: {metrics['accuracy']:.4f} | "
                f"Precision: {metrics['precision']:.4f} | "
                f"Recall: {metrics['recall']:.4f} | "
                f"F1-score: {metrics['f1_score']:.4f}"
            )

        # Use the first test dataset metrics for the main "accuracy" metric (for backward compatibility)
        test_acc = local_metrics_all[first_test_name]["accuracy"]
        local_precision = local_metrics_all[first_test_name]["precision"]
        local_recall = local_metrics_all[first_test_name]["recall"]
        local_f1 = local_metrics_all[first_test_name]["f1_score"]
        local_cm = local_metrics_all[first_test_name]["confusion_matrix"]
        # Build comprehensive metrics dictionary
        metrics = {
            "accuracy": test_global_acc,  # by convention, 'accuracy' is the key for the global accuracy used by the server for model selection
            "precision": test_global_precision,
            "recall": test_global_recall,
            "f1_score": test_global_f1,
            "global_confusion_matrix": test_global_cm,
            "local_accuracy": test_acc,
            "local_precision": local_precision,
            "local_recall": local_recall,
            "local_f1_score": local_f1,
            "local_confusion_matrix": local_cm,
        }

        # Add differential privacy metrics if enabled
        if args.enable_dp and privacy_engine is not None:
            final_epsilon = privacy_engine.get_epsilon(args.target_delta)
            metrics["privacy_epsilon"] = final_epsilon
            metrics["privacy_delta"] = args.target_delta
            metrics["privacy_max_grad_norm"] = args.max_grad_norm
            metrics["privacy_noise_multiplier"] = optimizer.noise_multiplier
            metrics["privacy_current_round"] = input_model.current_round
            metrics["privacy_total_rounds"] = input_model.total_rounds
            print(f"\n=== Privacy Budget (Round {input_model.current_round}/{input_model.total_rounds - 1}) ===")
            print(f"Cumulative ε = {final_epsilon:.2f} (target: {args.target_epsilon})")
            print(f"δ = {args.target_delta}")
            if input_model.current_round < input_model.total_rounds - 1:
                print("⚠️  Privacy budget is accumulating! More rounds remaining.")
            else:
                print("✓ Final cumulative privacy budget after all rounds")
            print("=" * 60)

        # Add all test dataset metrics with prefixes
        for test_name, test_metrics in global_metrics_all.items():
            metrics[f"global_{test_name}_accuracy"] = test_metrics["accuracy"]
            metrics[f"global_{test_name}_precision"] = test_metrics["precision"]
            metrics[f"global_{test_name}_recall"] = test_metrics["recall"]
            metrics[f"global_{test_name}_f1_score"] = test_metrics["f1_score"]
            metrics[f"global_{test_name}_confusion_matrix"] = test_metrics["confusion_matrix"]

        for test_name, test_metrics in local_metrics_all.items():
            metrics[f"local_{test_name}_accuracy"] = test_metrics["accuracy"]
            metrics[f"local_{test_name}_precision"] = test_metrics["precision"]
            metrics[f"local_{test_name}_recall"] = test_metrics["recall"]
            metrics[f"local_{test_name}_f1_score"] = test_metrics["f1_score"]
            metrics[f"local_{test_name}_confusion_matrix"] = test_metrics["confusion_matrix"]

        # Compute SHAP only when requested (expensive). -1 = last round only, 0 = never, N>0 = every N rounds
        run_shap = not args.enable_dp and (
            (args.shap_every == -1 and input_model.current_round == input_model.total_rounds - 1)
            or (args.shap_every > 0 and input_model.current_round % args.shap_every == 0)
        )
        shap_metrics = None
        if run_shap:
            print("Computing Shapley values...")
            first_test_features = test_datasets[first_test_name]["features"]
            first_test_labels = test_datasets[first_test_name]["labels"]
            plot_prefix = os.path.join(job_id, f"round{input_model.current_round}")
            shap_metrics = compute_shapley_values(
                model,
                first_test_features,
                first_test_labels,
                n_samples=100,
                plot_prefix=plot_prefix,
                feature_names=args.data_features,
            )
        elif args.enable_dp:
            print("[WARNING] Skip SHAP with DP")
        if shap_metrics:
            print(f"SHAP computation completed. Used {shap_metrics['shap_samples_used']} samples.")
        else:
            print("SHAP computation failed. Skipping SHAP metrics.")
        metrics["shap_metrics"] = shap_metrics

        # (6) construct trained FL model (A dict of {parameter name: parameter weights} from the PyTorch model)
        # Combine accuracy and SHAP metrics

        # compute delta model, global model has the primary key set
        local_weights = model.cpu().state_dict()
        global_weights = global_model.cpu().state_dict()
        missing_params = []
        model_diff = {}
        diff_norm = 0.0
        for name in global_weights:
            if name not in local_weights:
                missing_params.append(name)
                continue
            model_diff[name] = np.subtract(local_weights[name], global_weights[name], dtype=np.float32)
            diff_norm += np.linalg.norm(model_diff[name])
        if len(model_diff) == 0 or len(missing_params) > 0:
            raise ValueError(
                f"No weight differences computed or missing parameters! Missing parameters: {missing_params}"
            )
        print(f"Computed weight differences on {len(model_diff)} layers. Diff norm: {diff_norm}")

        # If model is wrapped by Opacus, extract the original module
        if args.enable_dp:
            # Remove prefix added by Opacus again to match global state dict
            new_model_diff = {}
            for k, v in model_diff.items():
                new_model_diff[k.replace("_module.", "")] = v
            model_diff = new_model_diff

        output_model = flare.FLModel(
            params=model_diff,
            params_type=ParamsType.DIFF,
            metrics=metrics,
            meta={
                "NUM_STEPS_CURRENT_ROUND": steps,
                # MetricsCollectionFilter reads this from dxo.meta when fl_ctx has no CURRENT_ROUND (e.g. sim).
                "current_round": input_model.current_round,
            },
        )

        # Log metrics for each test dataset
        mlflow_sleep_time = 0.1  # some sleep is required to avoid skipping metrics on the server (probably a race condition due to limited bandwidth/server resources)
        for test_name, test_metrics in global_metrics_all.items():
            for metric_name in ["accuracy", "precision", "recall", "f1_score"]:
                metric_value = test_metrics[metric_name]
                if np.isnan(metric_value) or metric_value is None:
                    test_metrics[metric_name] = -1.0
                    print(f"Warning: {metric_name} is NaN")

            print(f"Logging global metrics for {test_name}")
            test_name = test_name.replace("[", "").replace("]", "")
            test_name = test_name.split("_", 1)[1] if "_" in test_name else test_name

            mlflow.log_metric(f"global_{test_name}_accuracy", test_metrics["accuracy"], input_model.current_round)
            time.sleep(mlflow_sleep_time)
            mlflow.log_metric(f"global_{test_name}_precision", test_metrics["precision"], input_model.current_round)
            time.sleep(mlflow_sleep_time)
            mlflow.log_metric(f"global_{test_name}_recall", test_metrics["recall"], input_model.current_round)
            time.sleep(mlflow_sleep_time)
            mlflow.log_metric(f"global_{test_name}_f1_score", test_metrics["f1_score"], input_model.current_round)
            time.sleep(mlflow_sleep_time)

        for test_name, test_metrics in local_metrics_all.items():
            print(f"Logging local metrics for {test_name}")
            test_name = test_name.replace("[", "").replace("]", "")
            test_name = test_name.split("_", 1)[1] if "_" in test_name else test_name
            mlflow.log_metric(f"local_{test_name}_accuracy", test_metrics["accuracy"], input_model.current_round)
            time.sleep(mlflow_sleep_time)
            mlflow.log_metric(f"local_{test_name}_precision", test_metrics["precision"], input_model.current_round)
            time.sleep(mlflow_sleep_time)
            mlflow.log_metric(f"local_{test_name}_recall", test_metrics["recall"], input_model.current_round)
            time.sleep(mlflow_sleep_time)
            mlflow.log_metric(f"local_{test_name}_f1_score", test_metrics["f1_score"], input_model.current_round)
            time.sleep(mlflow_sleep_time)

        # (7) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
