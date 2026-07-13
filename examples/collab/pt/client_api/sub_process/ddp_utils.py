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

"""DDP utilities for Collab Client API.

This module provides helper functions for using NVFlare Client API with
PyTorch DDP (DistributedDataParallel) in Collab subprocess mode.

In Collab subprocess mode, only rank 0 is connected to the server. These
utilities handle the synchronization needed to avoid deadlocks.
"""

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel

# =============================================================================
# Option 1: Checkpoint-based sync
# =============================================================================


def receive_with_checkpoint(
    rank: int,
    model: nn.Module,
    checkpoint_dir: str = "/tmp/nvflare/ddp_checkpoints",
) -> Tuple[Optional[FLModel], bool]:
    """Receive model from server with checkpoint-based DDP sync.

    This function handles all the synchronization needed for DDP:
    1. All ranks call flare.receive() (only rank 0 blocks)
    2. Barrier to sync after receive
    3. Broadcast continue/stop signal
    4. Rank 0 saves checkpoint, others load

    Args:
        rank: The DDP rank of this process.
        model: The PyTorch model to sync weights to.
        checkpoint_dir: Directory for checkpoint files.

    Returns:
        Tuple of (FLModel or None, should_continue: bool)
        - If should_continue is False, training should stop.
        - FLModel contains the model params and metadata.

    Example:
        while True:
            fl_model, running = receive_with_checkpoint(rank, net)
            if not running:
                break
            # ... train with DDP ...
            if rank == 0:
                flare.send(output_model)
    """
    # All ranks call receive (rank 0 blocks, others return None)
    input_model = flare.receive()

    # Sync after receive
    dist.barrier()

    # Rank 0 broadcasts whether to continue
    should_continue = torch.tensor([1 if (rank != 0 or input_model is not None) else 0])
    dist.broadcast(should_continue, src=0)

    if should_continue.item() == 0:
        return None, False

    # Sync model via checkpoint
    site_name = flare.get_site_name()
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_{site_name}.pth")

    if rank == 0:
        # Load params from server if available
        if input_model.params:
            model.load_state_dict(input_model.params)
        # Always save checkpoint (even if no params, use current model weights)
        torch.save(model.state_dict(), checkpoint_path)

    dist.barrier()
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    return input_model, True


# =============================================================================
# Option 2: dist.broadcast-based sync (recommended)
# =============================================================================


def broadcast_model(model: nn.Module, src: int = 0) -> None:
    """Broadcast model parameters from src rank to all ranks.

    This is more efficient than checkpoint-based sync as it uses
    the DDP communication backend directly (NCCL/Gloo).

    Args:
        model: The PyTorch model to broadcast.
        src: Source rank (default 0).
    """
    for param in model.parameters():
        dist.broadcast(param.data, src=src)


def receive_with_broadcast(
    rank: int,
    model: nn.Module,
) -> Tuple[Optional[FLModel], bool]:
    """Receive model from server with broadcast-based DDP sync.

    This function handles all the synchronization needed for DDP:
    1. All ranks call flare.receive() (only rank 0 blocks)
    2. Barrier to sync after receive
    3. Broadcast continue/stop signal
    4. Rank 0 loads weights, then broadcasts to all ranks

    Args:
        rank: The DDP rank of this process.
        model: The PyTorch model to sync weights to.

    Returns:
        Tuple of (FLModel or None, should_continue: bool)

    Example:
        while True:
            fl_model, running = receive_with_broadcast(rank, net)
            if not running:
                break
            ddp_model = DDP(net)
            # ... train ...
    """
    # All ranks call receive (rank 0 blocks, others return None)
    input_model = flare.receive()

    # Sync after receive
    dist.barrier()

    # Rank 0 broadcasts whether to continue
    should_continue = torch.tensor([1 if (rank != 0 or input_model is not None) else 0])
    dist.broadcast(should_continue, src=0)

    if should_continue.item() == 0:
        return None, False

    # Rank 0 loads weights if available, then broadcasts to all
    if rank == 0:
        if input_model.params:
            model.load_state_dict(input_model.params)

    # Broadcast current model weights from rank 0 to all ranks
    broadcast_model(model, src=0)

    return input_model, True
