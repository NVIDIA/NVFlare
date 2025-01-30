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

import argparse
import json
import logging
import os
import sys
import threading

from nvflare.apis.fl_constant import ConnectionSecurity, ConnPropKey, WorkspaceConstants
from nvflare.apis.workspace import Workspace
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.f3.drivers.net_utils import SSL_ROOT_CERT
from nvflare.fuel.f3.mpm import MainProcessMonitor as mpm
from nvflare.fuel.utils.config_service import ConfigService, search_file
from nvflare.fuel.utils.log_utils import configure_logging
from nvflare.fuel.utils.url_utils import make_url


class CellnetMonitor:
    def __init__(self, stop_event: threading.Event, workspace: str):
        self.stop_event = stop_event
        self.workspace = workspace

    def cellnet_stopped(self):
        touch_file = os.path.join(self.workspace, WorkspaceConstants.SHUTDOWN_FILE)
        with open(touch_file, "a"):
            os.utime(touch_file, None)
        self.stop_event.set()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-m", type=str, help="WORKSPACE folder", required=True)
    parser.add_argument("--relay_config", "-s", type=str, help="relay config json file", required=True)
    parser.add_argument("--set", metavar="KEY=VALUE", nargs="*")
    args = parser.parse_args()
    return args


def main(args):
    workspace = Workspace(root_dir=args.workspace)
    for name in [WorkspaceConstants.RESTART_FILE, WorkspaceConstants.SHUTDOWN_FILE]:
        try:
            f = workspace.get_file_path_in_root(name)
            if os.path.exists(f):
                os.remove(f)
        except Exception as ex:
            print(f"Could not remove file '{name}': {ex}.  Please check your system before starting FL.")
            sys.exit(-1)

    relay_config_file = workspace.get_file_path_in_startup(args.relay_config)
    with open(relay_config_file, "rt") as f:
        relay_config = json.load(f)

    if not isinstance(relay_config, dict):
        raise RuntimeError(f"invalid relay config file {args.relay_config}")

    my_identity = relay_config.get(ConnPropKey.IDENTITY)
    if not my_identity:
        raise RuntimeError(f"invalid relay config file {args.relay_config}: missing identity")

    parent = relay_config.get(ConnPropKey.PARENT)
    if not parent:
        raise RuntimeError(f"invalid relay config file {args.relay_config}: missing parent")

    parent_address = parent.get(ConnPropKey.ADDRESS)
    if not parent_address:
        raise RuntimeError(f"invalid relay config file {args.relay_config}: missing parent.address")

    parent_scheme = parent.get(ConnPropKey.SCHEME)
    if not parent_scheme:
        raise RuntimeError(f"invalid relay config file {args.relay_config}: missing parent.scheme")

    parent_fqcn = parent.get(ConnPropKey.FQCN)
    if not parent_fqcn:
        raise RuntimeError(f"invalid relay config file {args.relay_config}: missing parent.fqcn")

    configure_logging(workspace, workspace.get_root_dir())

    logger = logging.getLogger()

    stop_event = threading.Event()
    monitor = CellnetMonitor(stop_event, args.workspace)

    ConfigService.initialize(
        section_files={},
        config_path=[args.workspace],
    )

    root_cert_path = search_file(SSL_ROOT_CERT, args.workspace)
    if not root_cert_path:
        raise ValueError(f"cannot find {SSL_ROOT_CERT} from config path {args.workspace}")

    credentials = {
        DriverParams.CA_CERT.value: root_cert_path,
    }

    conn_security = parent.get(ConnPropKey.CONNECTION_SECURITY)
    secure = True
    if conn_security:
        credentials[DriverParams.CONNECTION_SECURITY.value] = conn_security
        if conn_security == ConnectionSecurity.INSECURE:
            secure = False
    parent_url = make_url(parent_scheme, parent_address, secure)

    if parent_fqcn == FQCN.ROOT_SERVER:
        my_fqcn = my_identity
        root_url = parent_url
        parent_url = None
    else:
        my_fqcn = FQCN.join([parent_fqcn, my_identity])
        root_url = None

    cell = Cell(
        fqcn=my_fqcn,
        root_url=root_url,
        secure=secure,
        credentials=credentials,
        create_internal_listener=True,
        parent_url=parent_url,
    )
    NetAgent(cell, agent_closed_cb=monitor.cellnet_stopped)
    cell.start()

    # wait until stopped
    logger.info(f"Started relay {my_identity=} {my_fqcn=} {root_url=} {parent_url=} {parent_fqcn=}")
    stop_event.wait()
    cell.stop()
    logger.info(f"Relay stopped.")


if __name__ == "__main__":
    args = parse_arguments()
    rc = mpm.run(main_func=main, run_dir=args.workspace, args=args)
    sys.exit(rc)
