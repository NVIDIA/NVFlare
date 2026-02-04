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

import json
import os

import numpy as np
from filelock import FileLock
from src.net import Net

from nvflare import FedJob
from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.pt.file_model_locator import PTFileModelLocator
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_opt.tracking.tb.tb_receiver import TBAnalyticsReceiver
from nvflare.job_config.script_runner import ScriptRunner


def load_cifar10_labels():
    """Load CIFAR10 training labels for data splitting."""
    import torchvision
    import torchvision.transforms as transforms

    dataset_path = "/tmp/nvflare/data/cifar10"
    transform = transforms.Compose([transforms.ToTensor()])

    # Use file lock to prevent race condition when downloading dataset
    os.makedirs(dataset_path, exist_ok=True)
    lock_file = os.path.join(dataset_path, "download.lock")
    with FileLock(lock_file):
        trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)

    labels = np.array([label for _, label in trainset])
    return labels


def partition_data(num_sites, alpha, seed=0):
    """
    Partition CIFAR-10 data using Dirichlet sampling.

    Args:
        num_sites: Number of sites to partition data into
        alpha: Dirichlet distribution parameter (controls heterogeneity)
        seed: Random seed for reproducibility

    Returns:
        site_idx: Dictionary mapping site index to list of data indices
    """
    np.random.seed(seed)

    train_labels = load_cifar10_labels()

    min_size = 0
    K = 10  # Number of classes in CIFAR-10
    N = len(train_labels)
    site_idx = {}

    print(f"Partitioning CIFAR-10 dataset into {num_sites} sites with alpha={alpha}")

    # Split data using Dirichlet sampling
    while min_size < 10:
        idx_batch = [[] for _ in range(num_sites)]
        # For each class in the dataset
        for k in range(K):
            idx_k = np.where(train_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_sites))
            # Balance
            proportions = np.array([p * (len(idx_j) < N / num_sites) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    # Shuffle
    for j in range(num_sites):
        np.random.shuffle(idx_batch[j])
        site_idx[j] = idx_batch[j]

    return site_idx


def create_data_splits(split_dir, num_sites, alpha, seed=0):
    """
    Create and save data splits for federated learning.

    Args:
        split_dir: Directory to save split data
        num_sites: Number of sites
        alpha: Dirichlet distribution parameter
        seed: Random seed
    """
    # Check if splits already exist
    if os.path.exists(split_dir):
        # Check if all site files exist
        all_exist = all(os.path.exists(os.path.join(split_dir, f"site-{i + 1}.npy")) for i in range(num_sites))
        if all_exist:
            print(f"Data splits already exist at {split_dir}, skipping creation")
            return

    print(f"Creating data splits at {split_dir}")
    os.makedirs(split_dir, exist_ok=True)

    # Partition the data
    site_idx = partition_data(num_sites, alpha, seed)

    # Collect class distribution summary
    train_labels = load_cifar10_labels()
    class_sum = {}
    for site in range(num_sites):
        site_labels = train_labels[site_idx[site]]
        class_counts = {int(k): int(np.sum(site_labels == k)) for k in range(10)}
        class_sum[site] = class_counts

    # Write summary file
    sum_file_name = os.path.join(split_dir, "summary.txt")
    with open(sum_file_name, "w") as sum_file:
        sum_file.write(f"Number of clients: {num_sites}\n")
        sum_file.write(f"Dirichlet sampling parameter: {alpha}\n")
        sum_file.write("Class counts for each client:\n")
        sum_file.write(json.dumps(class_sum, indent=2))

    # Save site data files
    for site in range(num_sites):
        site_name = f"site-{site + 1}"
        site_file_name = os.path.join(split_dir, f"{site_name}.npy")
        np.save(site_file_name, np.array(site_idx[site]))
        print(f"Saved {site_name} data ({len(site_idx[site])} samples) to: {site_file_name}")

    # Print summary
    print("\nClass distribution summary:")
    for site, classes in class_sum.items():
        site_name = f"site-{site + 1}"
        total_samples = sum(classes.values())
        print(f"  {site_name}: {total_samples} samples - {classes}")

    print(f"\nData splits created at: {split_dir}")


if __name__ == "__main__":
    n_clients = 2
    num_rounds = 2
    train_script = "src/cifar10_fl_partitioned.py"

    # Data partitioning parameters
    alpha = 0.5  # Dirichlet distribution parameter for data heterogeneity
    data_split_root = f"/tmp/nvflare/data/cifar10_splits/clients{n_clients}_alpha{alpha}"

    # Create data splits for federated learning (heterogeneous data distribution)
    create_data_splits(data_split_root, n_clients, alpha)

    job = FedJob(name="cifar10_fedavg_xsite_val")

    # Add TensorBoard receiver on server (for external process mode)
    job.to_server(TBAnalyticsReceiver(events=["fed.analytix_log_stats"]))

    # Set up model persistor and shareable generator
    shareable_generator_id = job.to_server(FullModelShareableGenerator(), id="shareable_generator")
    persistor_id = job.to_server(PTFileModelPersistor(model=Net()), id="persistor")

    # Set up aggregator for federated averaging
    aggregator_id = job.to_server(
        InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS), id="aggregator"
    )

    # Define the ScatterAndGather controller workflow for training
    train_controller = ScatterAndGather(
        min_clients=n_clients,
        num_rounds=num_rounds,
        aggregator_id=aggregator_id,
        persistor_id=persistor_id,
        shareable_generator_id=shareable_generator_id,
        train_task_name="train",
    )
    job.to_server(train_controller)

    # Add model selector widget
    job.to_server(IntimeModelSelector(key_metric="accuracy"))

    # Set up model locator for cross-site evaluation
    model_locator_id = job.to_server(PTFileModelLocator(pt_persistor_id=persistor_id), id="model_locator")

    # Define the cross-site model evaluation workflow
    xsite_eval_controller = CrossSiteModelEval(
        model_locator_id=model_locator_id,
        submit_model_timeout=600,
        validation_timeout=6000,
        cleanup_models=False,
        validation_task_name="validate",  # Must match executor's evaluate_task_name default
        submit_model_task_name="submit_model",
    )
    job.to_server(xsite_eval_controller)

    # Add validation JSON generator for cross-site evaluation results
    job.to_server(ValidationJsonGenerator())

    # Add clients with ScriptRunner (use external process for cross-site validation)
    for i in range(n_clients):
        executor = ScriptRunner(
            script=train_script,
            script_args=f"--data_split_path {data_split_root}",  # Pass data split path to clients
            launch_external_process=True,  # Required for cross-site validation
        )
        target = f"site-{i + 1}"
        # Executor must handle train, validate, and submit_model tasks for cross-site validation
        job.to(executor, target, tasks=["train", "validate", "submit_model"])
        # Add local TB receiver for each client
        job.to(TBAnalyticsReceiver(events=["analytix_log_stats"]), target)

    # job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
