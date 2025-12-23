import os

import numpy as np
import torch
from torch.optim import Optimizer


def evaluate(model, data_loader, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader:
            # (optional) use GPU to speed things up
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total



def compute_model_diff(model, global_model):
    """Compute the difference between local and global model weights.

    Args:
        model: The local trained model
        global_model: The global model received from server

    Returns:
        tuple: (model_diff, diff_norm) where model_diff is a dict of weight differences
               and diff_norm is the total norm of differences

    Raises:
        ValueError: If no weight differences are computed or parameters are missing
    """
    local_weights = model.state_dict()
    global_weights = global_model.state_dict()
    missing_params = []
    model_diff = {}
    diff_norm = 0.0

    for name in global_weights:
        if name not in local_weights:
            missing_params.append(name)
            continue
        # Use PyTorch operations for subtraction and convert to numpy for serialization
        model_diff[name] = (local_weights[name] - global_weights[name]).cpu()
        diff_norm += torch.linalg.norm(model_diff[name])

    if len(model_diff) == 0 or len(missing_params) > 0:
        raise ValueError(f"No weight differences computed or missing parameters! Missing parameters: {missing_params}")

    if torch.isnan(diff_norm) or torch.isinf(diff_norm):
        raise ValueError(f"Diff norm is NaN or Inf! Diff norm: {diff_norm}")

    print(f"Computed weight differences on {len(model_diff)} layers. Diff norm: {diff_norm}")

    return model_diff, diff_norm

def get_lr_values(optimizer: Optimizer):
    """
    This function is used to get the learning rates of the optimizer.
    """
    return [group["lr"] for group in optimizer.state_dict()["param_groups"]]
