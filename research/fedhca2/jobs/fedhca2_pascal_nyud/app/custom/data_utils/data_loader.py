"""
NVFLARE Data Loader for FedHCA2
Integrates real datasets with NVFLARE client data partitioning
"""

import json
import os

import numpy as np
import torch

# Import from the original FedHCA2 core
from fedhca2_core.datasets.custom_dataset import get_dataset as fedhca2_get_dataset
from fedhca2_core.datasets.custom_transforms import get_transformations
from fedhca2_core.datasets.utils.configs import NUM_TRAIN_IMAGES, TEST_SCALE, TRAIN_SCALE
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


class MyPath(object):
    """
    Dataset path configuration
    """

    @staticmethod
    def db_root_dir(database=''):
        # Use absolute path to avoid issues with NVFLARE workspace copying
        db_root = '/home/suizhi/NVFlare/research/fedhca2_nvflare/data'
        
        db_names = {'PASCALContext': 'PASCALContext', 'NYUDv2': 'NYUDv2'}

        if database in db_names:
            return os.path.join(db_root, db_names[database])
        elif not database:
            return db_root
        else:
            raise NotImplementedError


def get_dataset(dataname, tasks, train=True, dataidxs=None, transform=None, local_rank=0):
    """
    Get dataset using the original FedHCA2 dataset loading logic
    """
    # Use the original FedHCA2 dataset loader
    dataset = fedhca2_get_dataset(dataname, train, tasks, transform, dataidxs, local_rank)
    return dataset





def get_dataloader(train, configs, dataset, sampler=None):
    """
    Create dataloader from dataset using original FedHCA2 logic
    """
    # Use original FedHCA2 dataloader directly
    from fedhca2_core.datasets.custom_dataset import get_dataloader as fedhca2_get_dataloader
    return fedhca2_get_dataloader(train, configs, dataset, sampler)


def get_client_data_partition(client_id, client_config, exp_config):
    """
    Get data partition indices for a specific client based on FedHCA2 logic
    """
    dataname = client_config['dataname']
    is_single_task = client_config.get('is_single_task', True)

    # Determine total number of training images for this dataset
    total_images = NUM_TRAIN_IMAGES.get(dataname, 1000)

    # Generate all indices and partition based on client type
    np.random.seed(42)  # Fixed seed for reproducible partitioning
    all_indices = np.random.permutation(total_images)

    if is_single_task:
        # ST (Single-Task) dataset partitioning
        st_datasets = exp_config.get("ST_Datasets", [])
        for dataset_config in st_datasets:
            if dataset_config["dataname"] == dataname:
                task_dict = dataset_config["task_dict"]
                n_clients = sum(task_dict.values())

                # Split data among single-task clients
                batch_indices = np.array_split(all_indices, n_clients)
                return batch_indices[client_id % n_clients]
    else:
        # MT (Multi-Task) dataset partitioning
        mt_datasets = exp_config.get("MT_Datasets", [])
        for dataset_config in mt_datasets:
            if dataset_config["dataname"] == dataname:
                n_clients = dataset_config["client_num"]

                # Split data among multi-task clients
                batch_indices = np.array_split(all_indices, n_clients)
                return batch_indices[client_id % n_clients]

    # Fallback: return full dataset
    return list(range(total_images))
