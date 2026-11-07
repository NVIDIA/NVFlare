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

"""Prepare one site, run its FLARE process, and handle Compose stop signals."""
import json
import os
import signal
import subprocess
import sys
import time

from common import WORKSPACE, write_json


def configure_server():
    """Use container-local storage and enable the server's catalog manager."""
    from nvflare.lighter.utils import update_storage_locations

    local = WORKSPACE / "local"
    resources_file = local / "resources.json"
    update_storage_locations(
        local_dir=str(local),
        workspace=str(WORKSPACE),
        default_resource_name="resources.json" if resources_file.exists() else "resources.json.default",
    )
    resources = json.loads(resources_file.read_text())
    manager = "nvflare.app_common.fdn.catalog.server.manager.CatalogManager"
    if not any(c.get("id") == "catalog_manager" for c in resources["components"]):
        resources["components"].append({"id": "catalog_manager", "path": manager})
    if manager not in resources["class_allow_list"]:
        resources["class_allow_list"].append(manager)
    write_json(resources_file, resources)


def main():
    role = os.environ["FLARE_ROLE"]
    if role == "server":
        configure_server()
    else:
        from prepare_data import prepare

        prepare(role)
    for name in ("shutdown.fl", "restart.fl", "pid.fl", "daemon_pid.fl"):
        (WORKSPACE / name).unlink(missing_ok=True)
    log = WORKSPACE / "log.txt"
    offset = log.stat().st_size if log.exists() else 0
    module = "server.server_train" if role == "server" else "client.client_train"
    config = "fed_server.json" if role == "server" else "fed_client.json"
    child = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "nvflare.private.fed.app." + module,
            "-m",
            str(WORKSPACE),
            "-s",
            config,
            "--set",
            "secure_train=true",
            f"uid={role}",
            "org=nvidia",
            "config_folder=config",
        ],
        start_new_session=True,
    )

    stopping_at = None

    def stop(signum, frame):
        nonlocal stopping_at
        if stopping_at is None and child.poll() is None:
            stopping_at = time.monotonic()
            os.killpg(child.pid, signal.SIGINT)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    write_json(WORKSPACE / ".startup.json", {"pid": child.pid, "log_offset": offset})
    try:
        while child.poll() is None:
            if stopping_at is not None and time.monotonic() - stopping_at >= 10:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
            time.sleep(0.2)
        raise SystemExit(child.returncode)
    finally:
        (WORKSPACE / ".startup.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
