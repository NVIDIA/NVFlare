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

from abc import ABC, abstractmethod
from typing import Any, Set

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext


class ModelManager(FLComponent, ABC):
    """Abstract base class for model managers in federated learning.

    This class defines the interface that all model managers must implement.
    Model managers are responsible for handling model updates, aggregation,
    and version control in federated learning workflows.
    """

    def __init__(self):
        FLComponent.__init__(self)
        """Initialize the ModelManager.
        ModelManager keeps track of three things:
        - current_model holding the current global model
        - current_model_version holding the current global model version
        - updates containing all received updates for updating the global model
        """
        self.current_model = None
        self.current_model_version = 0
        self.updates = {}

    @abstractmethod
    def initialize_model(self, model: Any, fl_ctx: FLContext) -> None:
        """Initialize the model manager with an initial model.

        Args:
            model: The initial model
            fl_ctx: FLContext object

            Returns: none
        """
        pass

    @abstractmethod
    def generate_new_model(self, fl_ctx: FLContext) -> None:
        """Generate a new model version based on accumulated updates.

        Args:
            fl_ctx: FLContext object

            Returns: none
        """
        pass

    @abstractmethod
    def prune_model_versions(self, versions_to_keep: Set[int], fl_ctx: FLContext) -> None:
        """Prune the model versions that are no longer active.

        Args:
            versions_to_keep: Set of model versions to keep
            fl_ctx: FLContext object

            Returns: none
        """
        pass

    @abstractmethod
    def process_updates(self, model_updates: Any, fl_ctx: FLContext) -> bool:
        """Process incoming model updates from clients.

        Args:
            model_updates: updates collected from clients
            fl_ctx: FLContext object

        Returns:
            bool: Whether the updates were successfully processed
        """
        pass

    def get_current_model(self, fl_ctx: FLContext) -> Any:
        """Get the current model.

        Args:
            fl_ctx: FLContext object

        Returns:
            The current model
        """
        return self.current_model
