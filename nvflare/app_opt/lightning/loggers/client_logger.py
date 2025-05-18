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
Client Logger
-------------
"""

from argparse import Namespace
from typing import Any, Optional, Union

from lightning.fabric.utilities.logger import _add_prefix
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from typing_extensions import override

import nvflare
from nvflare.client.tracking import MLflowWriter
from nvflare.fuel.utils.log_utils import get_obj_logger


class ClientLogger(Logger):
    """NVFLARE Client Logger for PyTorch Lightning.

    A logger that integrates with NVFLARE's tracking system to log metrics during federated training.
    This logger sends metrics to the NVFLARE server or Client for tracking and visualization.

    Args:
        prefix (str, optional): Prefix to add to all metric names. Defaults to "".

    Example:
        >>> from nvflare.app_opt.lightning.loggers import ClientLogger
        >>> logger = ClientLogger(prefix="client1")
        >>> trainer = Trainer(logger=logger)
        >>> # Metrics will be logged with prefix "client1-accuracy", "client1-loss", etc.

    Note:
        - Only supports metric logging, hyperparameter logging is not supported
        - Metrics are only logged from global rank 0
        - Although NVFLARE's MLflowWriter internally is used for metric tracking, it is used as a wrapper to send metrics to the NVFLARE server or Client,
         this can be replaced with any other writers such as TensorBoardWriter.
    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(self, prefix: str = ""):
        """Initialize the NVFLARE Client Logger.

        Args:
            prefix (str, optional): Prefix to add to all metric names. Defaults to "".
        """
        super().__init__()
        self._prefix = prefix
        self._metrics_writer = MLflowWriter()
        self._logger = get_obj_logger(self)

    @override
    @rank_zero_only
    def log_hyperparams(self, params: Union[dict[str, Any], Namespace]) -> None:
        """Log hyperparameters (Not supported).

        Args:
            params (Union[dict[str, Any], Namespace]): Hyperparameters to log.

        Note:
            This method is not supported and will log a warning.
        """
        self._logger.warning("log_hyperparams is not supported.")

    @override
    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to NVFLARE tracking system.

        Args:
            metrics (dict[str, float]): Dictionary of metric names and values.
            step (Optional[int], optional): Step number for the metrics. Defaults to None.

        Raises:
            AssertionError: If called from a non-zero global rank.
        """
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        metrics = dict(_add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR))
        self._metrics_writer.log_metrics(metrics=metrics, step=step)

    @property
    @override
    def name(self) -> Optional[str]:
        """Get name.

        Returns:
            Optional[str]: The name "nvflare".
        """
        return "nvflare"

    @property
    @override
    def version(self) -> Optional[str]:
        """Get version.

        Returns:
            Optional[str]: The current NVFLARE version.
        """
        return nvflare.__version__
