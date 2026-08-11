# Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Direct Cell transport for analytics emitted by a launched process."""

import json
import os
import tempfile
import time
from typing import Any, Optional

from nvflare.apis.analytix import AnalyticsDataType, LogWriterName
from nvflare.apis.utils.analytix_utils import create_analytic_dxo
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.utils import new_cell_message
from nvflare.fuel.utils.log_utils import get_obj_logger

ANALYTICS_BOOTSTRAP_ENV = "NVFLARE_ANALYTICS_BOOTSTRAP"
ANALYTICS_BOOTSTRAP_FILE = "analytics_bootstrap.json"
CHANNEL = "metrics"
TOPIC_LOG = "log"
REQUEST_TIMEOUT = 10.0
CONNECT_TIMEOUT = 30.0

REQUIRED_BOOTSTRAP_KEYS = (
    "connect_url",
    "receiver_fqcn",
    "client_fqcn",
)


def resolve_bootstrap(config_file: str = None) -> str:
    env_file = os.environ.get(ANALYTICS_BOOTSTRAP_ENV)
    config_path = os.path.abspath(config_file) if config_file else None
    env_path = os.path.abspath(env_file) if env_file else None
    if config_path and env_path and config_path != env_path:
        raise ValueError(
            f"analytics bootstrap conflict: config_file={config_path!r} differs from "
            f"{ANALYTICS_BOOTSTRAP_ENV}={env_path!r}"
        )
    path = config_path or env_path
    if not path:
        raise RuntimeError(f"analytics bootstrap is not configured; set {ANALYTICS_BOOTSTRAP_ENV}")
    return path


def validate_bootstrap(config: dict, path: str = "<analytics bootstrap>") -> None:
    if not isinstance(config, dict):
        raise ValueError(f"invalid analytics bootstrap {path}: expected a JSON object")
    for key in REQUIRED_BOOTSTRAP_KEYS:
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"invalid analytics bootstrap {path}: {key} must be a non-empty string")
    receiver, client = config["receiver_fqcn"], config["client_fqcn"]
    if FQCN.validate(receiver) or FQCN.validate(client) or FQCN.get_parent(client) != receiver:
        raise ValueError(f"invalid analytics bootstrap {path}: client must be a direct child of receiver")


def read_bootstrap(config_file: str = None) -> tuple[str, dict]:
    path = resolve_bootstrap(config_file)
    with open(path, "r") as f:
        config = json.load(f)
    validate_bootstrap(config, path)
    return path, config


def write_bootstrap(path: str, config: dict) -> None:
    """Atomically write an owner-only analytics bootstrap."""
    validate_bootstrap(config, path)
    target = os.path.abspath(path)
    fd, temp = tempfile.mkstemp(dir=os.path.dirname(target), prefix=".analytics-", suffix=".tmp")
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as f:
            fd = -1
            json.dump(config, f)
        os.replace(temp, target)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.remove(temp)
        except FileNotFoundError:
            pass
        raise


class MetricsSender:
    """Send analytic DXOs over a credential-free child Cell to a local MetricRelay."""

    def __init__(self, config: Optional[dict] = None, config_file: Optional[str] = None):
        self.logger = get_obj_logger(self)
        if config is not None and config_file is not None:
            raise ValueError("specify either config or config_file, not both")
        if config is not None:
            validate_bootstrap(config)
            self.config_file = None
            self.config = dict(config)
        else:
            self.config_file, self.config = read_bootstrap(config_file)
        self.cell = None
        self.rank = None
        self.initialized = False
        self.closed = False

    def init(self, rank=None) -> None:
        if self.closed:
            raise RuntimeError("MetricsSender is closed")
        if self.initialized:
            return
        if rank is None:
            rank = os.environ.get("RANK", "0")
        elif isinstance(rank, int):
            rank = str(rank)
        elif not isinstance(rank, str):
            raise ValueError(f"rank must be a string or an integer but got {type(rank)}")
        self.rank = rank
        if self.rank == "0":
            self.cell = Cell(
                fqcn=self.config["client_fqcn"],
                root_url=None,
                secure=False,
                credentials={},
                parent_url=self.config["connect_url"],
            )
            try:
                self.cell.start()
                deadline = time.monotonic() + CONNECT_TIMEOUT
                while not self.cell.is_cell_connected(self.config["receiver_fqcn"]):
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"metrics Cell did not connect after {CONNECT_TIMEOUT}s")
                    time.sleep(0.1)
            except BaseException:
                self.cell.stop()
                self.cell = None
                raise
        self.initialized = True

    def add(self, tag: str, value: Any, data_type: AnalyticsDataType, **kwargs) -> bool:
        if self.rank != "0" or not self.initialized or self.closed:
            return False
        writer = kwargs.pop("writer", LogWriterName.TORCH_TB)
        dxo = create_analytic_dxo(tag=tag, value=value, data_type=data_type, writer=writer, **kwargs)
        try:
            reply = self.cell.send_request(
                CHANNEL,
                TOPIC_LOG,
                self.config["receiver_fqcn"],
                new_cell_message({}, dxo.to_dict()),
                REQUEST_TIMEOUT,
            )
        except Exception as e:
            self.logger.warning(f"failed to send metric {tag!r}: {e}")
            return False
        if reply is None or reply.get_header(MessageHeaderKey.RETURN_CODE) != CellReturnCode.OK:
            self.logger.warning(f"failed to send metric {tag!r}")
            return False
        return True

    def shutdown(self) -> None:
        if self.closed:
            return
        self.closed = True
        cell, self.cell = self.cell, None
        if cell:
            cell.stop()

    close = shutdown
