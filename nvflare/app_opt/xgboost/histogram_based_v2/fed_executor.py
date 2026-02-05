# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import threading
import uuid

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.xgboost.histogram_based_v2.adaptors.grpc_client_adaptor import GrpcClientAdaptor
from nvflare.app_opt.xgboost.histogram_based_v2.runners.xgb_client_runner import XGBClientRunner

from .executor import XGBExecutor
from .sec.client_handler import ClientSecurityHandler


class FedXGBHistogramExecutor(XGBExecutor):
    # Class-level cache keyed by client name so all executor instances for a client
    # share the same adaptor (fixes "rank not set" when config and start use different instances).
    _adaptor_cache: dict = {}
    _adaptor_cache_lock = threading.Lock()

    def __init__(
        self,
        data_loader_id: str,
        per_msg_timeout=60.0,
        tx_timeout=600.0,
        model_file_name="model.json",
        metrics_writer_id: str = None,
        in_process=True,
    ):
        XGBExecutor.__init__(
            self,
            adaptor_component_id="",
            per_msg_timeout=per_msg_timeout,
            tx_timeout=tx_timeout,
        )
        self.data_loader_id = data_loader_id
        # do not let use specify int_server_grpc_options in this version - always use default
        self.int_server_grpc_options = None
        self.model_file_name = model_file_name
        self.metrics_writer_id = metrics_writer_id
        self.in_process = in_process

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Clear adaptor cache on END_RUN so executor can be reused in next run."""
        if event_type == EventType.END_RUN:
            client_name = fl_ctx.get_identity_name()
            key = client_name if (client_name and client_name != "") else "_default_"
            with FedXGBHistogramExecutor._adaptor_cache_lock:
                FedXGBHistogramExecutor._adaptor_cache.pop(key, None)
        super().handle_event(event_type, fl_ctx)

    def get_adaptor(self, fl_ctx: FLContext):
        client_name = fl_ctx.get_identity_name()
        with FedXGBHistogramExecutor._adaptor_cache_lock:
            cache = FedXGBHistogramExecutor._adaptor_cache
            if client_name in cache:
                return cache[client_name]
            # Simulator/recipe may call with None or different identity for same process.
            # If there is exactly one cached adaptor, reuse it so config and start share it.
            if (client_name is None or client_name == "") and len(cache) == 1:
                return next(iter(cache.values()))

        engine = fl_ctx.get_engine()
        handler = ClientSecurityHandler()
        engine.add_component(str(uuid.uuid4()), handler)

        runner = XGBClientRunner(
            data_loader_id=self.data_loader_id,
            model_file_name=self.model_file_name,
            metrics_writer_id=self.metrics_writer_id,
        )
        runner.initialize(fl_ctx)
        adaptor = GrpcClientAdaptor(
            int_server_grpc_options=self.int_server_grpc_options,
            in_process=self.in_process,
            per_msg_timeout=self.per_msg_timeout,
            tx_timeout=self.tx_timeout,
        )
        adaptor.set_runner(runner)
        with FedXGBHistogramExecutor._adaptor_cache_lock:
            cache = FedXGBHistogramExecutor._adaptor_cache
            key = client_name if (client_name and client_name != "") else "_default_"
            if key not in cache:
                cache[key] = adaptor
                return adaptor
            return cache[key]
