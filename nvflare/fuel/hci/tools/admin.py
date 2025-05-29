# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import os

from nvflare.apis.workspace import Workspace
from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.hci.client.api_spec import AdminConfigKey
from nvflare.fuel.hci.client.cli import AdminClient
from nvflare.fuel.hci.client.config import FLAdminClientStarterConfigurator
from nvflare.fuel.hci.client.file_transfer import FileTransferModule
from nvflare.security.logging import secure_format_exception


def main():
    """
    Script to launch the admin client to issue admin commands to the server.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-m", type=str, help="WORKSPACE folder", required=True)

    parser.add_argument(
        "--fed_admin", "-s", type=str, help="json file with configurations for launching admin client", required=True
    )
    parser.add_argument("--cli_history_size", type=int, default=1000)
    parser.add_argument("--with_debug", action="store_true")

    args = parser.parse_args()

    try:
        os.chdir(args.workspace)
        workspace = Workspace(root_dir=args.workspace)
        conf = FLAdminClientStarterConfigurator(workspace=workspace)
        conf.configure()
    except ConfigError as e:
        print(f"ConfigError: {secure_format_exception(e)}")
        return

    try:
        admin_config = conf.config_data["admin"]
    except KeyError:
        print("Missing admin section in fed_admin configuration.")
        return

    modules = []

    if admin_config.get(AdminConfigKey.WITH_FILE_TRANSFER):
        modules.append(
            FileTransferModule(
                upload_dir=admin_config.get(AdminConfigKey.UPLOAD_DIR),
                download_dir=admin_config.get(AdminConfigKey.DOWNLOAD_DIR),
            )
        )

    if args.with_debug:
        print("File Transfer: {}".format(admin_config.get(AdminConfigKey.WITH_FILE_TRANSFER)))
        if admin_config.get(AdminConfigKey.WITH_FILE_TRANSFER):
            print("  Upload Dir: {}".format(admin_config.get(AdminConfigKey.UPLOAD_DIR)))
            print("  Download Dir: {}".format(admin_config.get(AdminConfigKey.DOWNLOAD_DIR)))

    client = AdminClient(
        admin_config=admin_config,
        cmd_modules=modules,
        debug=args.with_debug,
        username=admin_config.get(AdminConfigKey.USERNAME, ""),
        handlers=conf.handlers,
        cli_history_dir=args.workspace,
        cli_history_size=args.cli_history_size,
    )

    client.run()


if __name__ == "__main__":
    main()
