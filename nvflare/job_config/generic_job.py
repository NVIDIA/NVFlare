# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any, List, Optional

from .api import ControllerApp, ExecutorApp, FedJob
from .defs import JobTargetType


class GenericJob(ABC, FedJob):
    def __init__(
        self,
        name: str = "fed_job",
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
    ):
        super().__init__(name, min_clients, mandatory_clients)
        self.set_up_server()
        app = self.get_app(JobTargetType.SERVER)
        if not isinstance(app, ControllerApp):
            raise RuntimeError(f"set_up_server() must set a valid ControllerApp but got {type(app)}")

    @abstractmethod
    def set_up_server(self):
        pass

    @abstractmethod
    def get_executor_app(self, client: str) -> ExecutorApp:
        pass

    def to(
        self,
        obj: Any,
        target: str,
        id=None,
        **kwargs,
    ):
        target_type = JobTargetType.get_target_type(target)
        if target_type == JobTargetType.CLIENT and not isinstance(obj, ExecutorApp):
            app = self.get_app(target)
            if not app:
                # client app is not assigned yet
                app = self.get_executor_app(target)
                if not isinstance(app, ExecutorApp):
                    raise RuntimeError(f"get_executor_app() must return a valid ExecutorApp but got {type(app)}")
                super().to(app, target)

        return super().to(obj, target, id, **kwargs)
