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


from abc import ABC
from typing import List, Optional, Union

from pydantic import BaseModel, PrivateAttr, field_validator


class ExecEnv(BaseModel, ABC):
    workspace_dir: str
    clients: Union[int, List[str], None] = None
    gpus: Union[int, List[int], None] = None
    _client_names: List[str] = PrivateAttr(default=None)  # Private attribute for client names

    def model_post_init(self, __context):  # v2 pydantic
        super().model_post_init(__context)
        if self.clients is None:
            self.clients = 2
        if isinstance(self.clients, int):
            self._client_names = [f"site-{i + 1}" for i in range(self.clients)]
        else:
            self._client_names = self.clients

    @property
    def client_names(self) -> List[str]:
        if self._client_names is None:
            raise ValueError("client_names not initialized – check model_post_init or inputs.")
        return self._client_names

    @field_validator("clients")
    @classmethod
    def check_clients(cls, v):
        if isinstance(v, int):
            if v <= 0:
                raise ValueError("Number of clients must be positive")
        elif isinstance(v, list):
            if len(v) <= 0:
                raise ValueError("Client list must not be empty")
            if any(not name.strip() for name in v):
                raise ValueError("Client names must not be empty strings")
        return v

    @field_validator("gpus")
    @classmethod
    def check_gpus(cls, v):
        if isinstance(v, int):
            if v < 0:
                raise ValueError("Number of GPUs must be non-negative")
        elif isinstance(v, list):
            if len(v) <= 0:
                raise ValueError("GPU list must not be empty")
            if any(gpu_id < 0 for gpu_id in v):
                raise ValueError("GPU IDs must be non-negative")
            if len(v) != len(set(v)):
                raise ValueError("GPU IDs must be unique")
        return v

    def setup(self):
        print(f"Setting up environment in {self.workspace_dir} with {len(self.client_names)} clients")
        print(f"Using GPUs: {self.gpus}")


class SimEnv(ExecEnv):
    clients: Union[int, List[str]] = None
    workspace_dir: Optional[str] = None

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.clients is None:
            self.clients = 2
        if self.workspace_dir is None:
            self.workspace_dir = "/tmp/nvflare/workspace"


class PoCEnv(ExecEnv):
    clients: Union[int, List[str]] = None


class ProdEnv(ExecEnv):
    admin_dir: str

    def setup(self):
        print(f"Setting up production environment with input directory {self.admin_dir}")
