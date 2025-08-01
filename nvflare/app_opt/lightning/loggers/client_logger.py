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
MLflow Logger
-------------
"""

from argparse import Namespace
from collections.abc import Mapping
from typing import Any, Optional, Union

from lightning_fabric.utilities.logger import _add_prefix
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from typing_extensions import override

import nvflare
from nvflare.client.tracking import MLflowWriter
from nvflare.fuel.utils.log_utils import get_obj_logger


class ClientLogger(Logger):
    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        prefix: str = "",
    ):
        super().__init__()
        self._prefix = prefix
        self._metric_writer = MLflowWriter()
        self._logger = get_obj_logger(self)

    @override
    @rank_zero_only
    def log_hyperparams(self, params: Union[dict[str, Any], Namespace]) -> None:
        self._logger.warning("log_hyperparams is not supported.")

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        metrics = dict(_add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR))

        self._metric_writer.log_metrics(metrics=metrics, step=step)
        pass

    @override
    @rank_zero_only
    def finalize(self, status: str = "success") -> None:
        pass

    @property
    @override
    def save_dir(self) -> Optional[str]:

        return None

    @property
    @override
    def name(self) -> Optional[str]:
        """Get the experiment id.

        Returns:
            The experiment id.

        """
        return "nvflare"

    @property
    @override
    def version(self) -> Optional[str]:

        return getattr(nvflare, "__version__", "unknown")

    @override
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        pass
