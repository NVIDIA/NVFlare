# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import grpc

from nvflare.apis.fl_context import FLContext
from nvflare.private.fed.client.admin import RequestProcessor
from nvflare.private.fed.client.client_req_processors import ClientRequestProcessors
from nvflare.private.fed.client.fed_client import FederatedClient


class BaseClientDeployer:
    def __init__(self):
        """To init the BaseClientDeployer."""
        self.multi_gpu = False
        self.outbound_filters = None
        self.inbound_filters = None
        self.federated_client = None
        self.model_validator = None
        self.cross_val_participating = False
        self.model_registry_path = None
        self.cross_val_timeout = None
        self.executors = None

        self.req_processors = ClientRequestProcessors.request_processors

    def build(self, build_ctx):
        self.server_config = build_ctx["server_config"]
        self.client_config = build_ctx["client_config"]
        self.secure_train = build_ctx["secure_train"]
        self.client_name = build_ctx["client_name"]
        self.host = build_ctx["server_host"]
        self.enable_byoc = build_ctx["enable_byoc"]
        self.overseer_agent = build_ctx["overseer_agent"]
        self.components = build_ctx["client_components"]
        self.handlers = build_ctx["client_handlers"]

    def set_model_manager(self, model_manager):
        self.model_manager = model_manager

    def create_fed_client(self, args, sp_target=None):
        if sp_target:
            for item in self.server_config:
                service = item["service"]
                service["target"] = sp_target
        servers = [{t["name"]: t["service"]} for t in self.server_config]
        retry_timeout = 30
        if "retry_timeout" in self.client_config:
            retry_timeout = self.client_config["retry_timeout"]

        compression = grpc.Compression.NoCompression
        if "Deflate" == self.client_config.get("compression"):
            compression = grpc.Compression.Deflate
        elif "Gzip" == self.client_config.get("compression"):
            compression = grpc.Compression.Gzip

        for _, processor in self.components.items():
            if isinstance(processor, RequestProcessor):
                self.req_processors.append(processor)

        self.federated_client = FederatedClient(
            client_name=str(self.client_name),
            # We only deploy the first server right now .....
            server_args=sorted(servers)[0],
            client_args=self.client_config,
            secure_train=self.secure_train,
            retry_timeout=retry_timeout,
            executors=self.executors,
            compression=compression,
            enable_byoc=self.enable_byoc,
            overseer_agent=self.overseer_agent,
            args=args,
            components=self.components,
            handlers=self.handlers,
        )
        return self.federated_client

    def finalize(self, fl_ctx: FLContext):
        self.close()

    def close(self):
        # if self.federated_client:
        #     self.federated_client.model_manager.close()
        pass
