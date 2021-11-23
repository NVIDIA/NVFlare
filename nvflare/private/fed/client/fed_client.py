# Copyright (c) 2021, NVIDIA CORPORATION.
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

"""The client of the federated training process"""
import pickle
from typing import List, Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.filter import Filter
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.private.event import fire_event
from .fed_client_base import FederatedClientBase
from ..utils.numproto import proto_to_bytes


class FederatedClient(FederatedClientBase):
    """
    Federated client-side implementation.
    """

    def __init__(
        self,
        client_name,
        client_args,
        secure_train,
        server_args=None,
        retry_timeout=30,
        client_state_processors: Optional[List[Filter]] = None,
        handlers: Optional[List[FLComponent]] = None,
        executors: Optional[List[Executor]] = None,
        compression=None,
        enable_byoc=False
    ):
        # We call the base implementation directly.
        super().__init__(
            client_name=client_name,
            client_args=client_args,
            secure_train=secure_train,
            server_args=server_args,
            retry_timeout=retry_timeout,
            client_state_processors=client_state_processors,
            handlers=handlers,
            compression=compression,
        )

        self.executors = executors
        self.enable_byoc = enable_byoc

    def fetch_task(self, fl_ctx: FLContext):
        fire_event(EventType.BEFORE_PULL_TASK, self.handlers, fl_ctx)

        pull_success, task_name, remote_tasks = self.pull_task(fl_ctx)
        fire_event(EventType.AFTER_PULL_TASK, self.handlers, fl_ctx)
        self.logger.info(f"pull_task completed. Task name:{task_name} Status:{pull_success} ")
        return pull_success, task_name, remote_tasks

    def extract_shareable(self, responses, fl_ctx: FLContext):
        shareable = Shareable()
        peer_context = FLContext()
        for item in responses:
            shareable = shareable.from_bytes(proto_to_bytes(item.data.params["data"]))
            peer_context = pickle.loads(proto_to_bytes(item.data.params["fl_context"]))

        fl_ctx.set_peer_context(peer_context)
        shareable.set_peer_props(peer_context.get_all_public_props())

        return shareable
