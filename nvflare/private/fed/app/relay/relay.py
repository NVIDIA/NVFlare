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
import os
import sys
import threading

from nvflare.apis.fl_constant import SecureTrainConst, WorkspaceConstants
from nvflare.apis.workspace import Workspace
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.fuel.f3.drivers.driver_params import ConnectionSecurity, DriverParams
from nvflare.fuel.f3.drivers.net_utils import SSL_ROOT_CERT
from nvflare.fuel.f3.mpm import MainProcessMonitor as mpm
from nvflare.fuel.utils.config_service import ConfigService, search_file
from nvflare.fuel.utils.log_utils import configure_logging
from nvflare.fuel.utils.url_utils import make_url


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

    my_identity = relay_config.get("identity")
    if not my_identity:
        raise RuntimeError(f"invalid relay config file {args.relay_config}: missing identity")

    parent = relay_config.get("parent")
    if not parent:
        raise RuntimeError(f"invalid relay config file {args.relay_config}: missing parent")

    parent_address = parent.get("address")
    if not parent_address:
        raise RuntimeError(f"invalid relay config file {args.relay_config}: missing parent.address")

    parent_scheme = parent.get("scheme")
    if not parent_scheme:
        raise RuntimeError(f"invalid relay config file {args.relay_config}: missing parent.scheme")

    parent_fqcn = parent.get("fqcn")
    if not parent_fqcn:
        raise RuntimeError(f"invalid relay config file {args.relay_config}: missing parent.fqcn")

    configure_logging(workspace, workspace.get_root_dir())

    stop_event = threading.Event()

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

    conn_security = parent.get(SecureTrainConst.CONNECTION_SECURITY)
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
    net_agent = NetAgent(cell)
    cell.start()

    # wait until stopped
    print(f"started relay {my_identity=} {my_fqcn=}")
    stop_event.wait()


if __name__ == "__main__":
    args = parse_arguments()
    rc = mpm.run(main_func=main, run_dir=args.workspace, args=args)
    sys.exit(rc)
