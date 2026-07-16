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


import os
from enum import Enum
from threading import Lock
from typing import Optional

from nvflare.client.cell.bootstrap import (
    BOOTSTRAP_FILE_ENV_VAR,
    CELL_API_TYPE,
    get_bootstrap_client_api_type,
    read_bootstrap_config,
)
from nvflare.client.constants import CLIENT_API_CONFIG
from nvflare.fuel.data_event.data_bus import DataBus

from .api_spec import CLIENT_API_KEY, CLIENT_API_TYPE_KEY, APISpec
from .ex_process.api import ExProcessClientAPI
from .in_process.api import InProcessClientAPI

DEFAULT_CONFIG = f"config/{CLIENT_API_CONFIG}"
data_bus = DataBus()


class ClientAPIType(Enum):
    IN_PROCESS_API = "IN_PROCESS_API"
    EX_PROCESS_API = "EX_PROCESS_API"
    # external_process execution mode's trainer-side Cell engine (CellClientAPI): selected
    # when NVFlare's ExternalProcessBackend launched this process. Reads its bootstrap config
    # from the NVFLARE_CLIENT_API_BOOTSTRAP env var (not the legacy config_file).
    CELL_API = CELL_API_TYPE


class APIContext:
    def __init__(self, rank: Optional[str] = None, config_file: Optional[str] = None):
        self.rank = rank
        self.config_file = config_file if config_file else DEFAULT_CONFIG
        self._explicit_config_file = bool(config_file)
        self._typed_bootstrap_file = None
        self._shutdown_lock = Lock()
        self._closed = False

        self.api_type = self._resolve_api_type()
        self.api = self._create_client_api(self.api_type)
        self.api.init(rank=self.rank)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup the client API when the context ends."""
        self.shutdown()

    @property
    def is_shutdown(self) -> bool:
        """Whether this context has released its Client API resources."""
        # InProcessBackend owns the DataBus API and closes it directly at job teardown.
        # Treat that owner-side close as a stopped context too, so a later job in the same
        # process does not reuse a cached APIContext that still points at the old backend.
        return self._closed or getattr(self.api, "closed", False) is True

    def shutdown(self):
        """Shuts down this context exactly once."""
        with self._shutdown_lock:
            if self._closed:
                return None
            self._closed = True
            try:
                return self.api.shutdown()
            finally:
                # APIContext is itself a context manager and may be shut down directly.
                # Notify the public cache/lifecycle owner so those paths cannot bypass the
                # Cell one-shot gate or leave a stopped non-Cell context cached.
                from .api import _on_context_shutdown

                _on_context_shutdown(self)

    def _resolve_api_type(self) -> ClientAPIType:
        """Resolve the API engine from a typed bootstrap or the legacy type environment."""
        typed_api_type = None
        typed_config_path = None
        if self._explicit_config_file:
            typed_config_path = self.config_file
            try:
                config = read_bootstrap_config(self.config_file)
            except (OSError, ValueError):
                # Missing/non-dict files retain the old behavior: the selected API engine
                # owns validation of its legacy config. A valid dict containing either
                # typed marker is validated below and never silently downgraded.
                config = None
            if config is not None:
                typed_api_type = get_bootstrap_client_api_type(config, self.config_file)
        if typed_api_type is None:
            # ExternalProcessBackend supplies one self-identifying bootstrap path. This is
            # sufficient to select CellClientAPI; no duplicate CLIENT_API_TYPE marker is needed.
            # It also supersedes an explicitly supplied legacy client_api_config.json: existing
            # trainer scripts may still pass that file, but launched mode must use this run's
            # Cell endpoint and token.
            bootstrap_path = os.environ.get(BOOTSTRAP_FILE_ENV_VAR)
            if bootstrap_path:
                config = read_bootstrap_config(bootstrap_path)
                typed_api_type = get_bootstrap_client_api_type(config, bootstrap_path)
                if typed_api_type is None:
                    raise ValueError(f"Client API bootstrap {bootstrap_path} is missing its typed envelope")
                typed_config_path = bootstrap_path

        env_api_type_name = os.environ.get(CLIENT_API_TYPE_KEY)
        if typed_api_type is not None:
            if env_api_type_name is not None and env_api_type_name != typed_api_type:
                raise ValueError(
                    f"Client API bootstrap {typed_config_path} declares {typed_api_type!r}, "
                    f"but {CLIENT_API_TYPE_KEY} is {env_api_type_name!r}"
                )
            self._typed_bootstrap_file = typed_config_path
            return ClientAPIType(typed_api_type)

        api_type_name = env_api_type_name or ClientAPIType.IN_PROCESS_API.value
        return ClientAPIType(api_type_name)

    def _create_client_api(self, api_type: ClientAPIType) -> APISpec:
        """Creates a new client_api based on the provided API type."""
        if api_type == ClientAPIType.IN_PROCESS_API:
            api = data_bus.get_data(CLIENT_API_KEY)
            if not isinstance(api, InProcessClientAPI):
                raise RuntimeError(f"api {api} is not a valid InProcessClientAPI")
            return api
        elif api_type == ClientAPIType.CELL_API:
            # Deferred import: the Cell engine pulls in cellnet/payload machinery the other
            # API types never need; keep flare.init import-light for in_process/ex_process.
            from nvflare.client.cell.api import CellClientAPI

            if self._typed_bootstrap_file:
                return CellClientAPI(bootstrap_file=self._typed_bootstrap_file)
            return CellClientAPI()
        else:
            return ExProcessClientAPI(config_file=self.config_file)
