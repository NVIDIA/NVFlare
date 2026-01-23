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

"""Collab tracking module for experiment metrics.

This module provides tracking writers that mimic the APIs of popular
experiment tracking libraries (TensorBoard, MLflow, W&B). The writers
work in BOTH in-process and subprocess execution modes:

- In-process mode: Fires events directly to event_manager
- Subprocess mode: Sends metrics via CellNet to parent CollabExecutor

Users just change the import statement - the execution mode is
automatically detected.

Example - TensorBoard:
    # Original:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    writer.add_scalar("loss", 0.5, global_step=100)

    # Collab (same API, works in both modes):
    from nvflare.collab.tracking import SummaryWriter
    writer = SummaryWriter()
    writer.add_scalar("loss", 0.5, global_step=100)

Example - MLflow:
    # Original:
    import mlflow
    mlflow.log_metric("loss", 0.5, step=100)

    # Collab (same API):
    from nvflare.collab.tracking import mlflow
    mlflow.log_metric("loss", 0.5, step=100)

Example - W&B:
    # Original:
    import wandb
    wandb.init(project="my-project")
    wandb.log({"loss": 0.5})

    # Collab (same API):
    from nvflare.collab.tracking import wandb
    wandb.init(project="my-project")
    wandb.log({"loss": 0.5})
"""

# W&B-compatible module
# MLflow-compatible module
from . import mlflow_writer as mlflow
from . import wandb_writer as wandb

# Auto-detecting writer (recommended for generic use)
from .auto_writer import AutoWriter
from .auto_writer import get_writer as get_auto_writer

# Base classes
from .base_writer import BaseWriter

# In-process writer (for direct event firing)
from .inprocess_writer import InProcessWriter, get_inprocess_writer, set_inprocess_writer

# Metrics handler for receiving metrics from subprocess
from .metrics_handler import MetricsRelay, setup_metrics_handler

# Also export specific writer classes
from .mlflow_writer import MLflowWriter

# Subprocess writer (for CellNet relay)
from .subprocess_writer import SubprocessWriter, get_writer, set_writer

# TensorBoard-compatible writer
from .tensorboard_writer import SummaryWriter, TensorBoardWriter
from .wandb_writer import WandbWriter
