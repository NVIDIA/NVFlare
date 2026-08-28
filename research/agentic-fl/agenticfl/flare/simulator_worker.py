# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""NVFlare simulator worker entry point using Unix-domain parent IPC."""

from __future__ import annotations

import time
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

from agenticfl.flare.simulator_compat import simulator_worker_ipc_path

from nvflare.fuel.f3.mpm import MainProcessMonitor as mpm
from nvflare.private.fed.app.simulator import simulator_worker


def _listen_on_unix_socket(address: Path) -> Any:
    address.unlink(missing_ok=True)
    listener = Listener(str(address))
    try:
        return listener.accept()
    finally:
        listener.close()


def main() -> None:
    args = simulator_worker.parse_arguments()
    address = simulator_worker_ipc_path(args.parent_pid, args.port)
    original_create_connection = simulator_worker._create_connection
    simulator_worker._create_connection = lambda _port: _listen_on_unix_socket(address)
    try:
        mpm.run(
            main_func=simulator_worker.main,
            run_dir=args.workspace,
            args=args,
        )
    finally:
        simulator_worker._create_connection = original_create_connection
        address.unlink(missing_ok=True)
    time.sleep(2)


if __name__ == "__main__":
    main()
