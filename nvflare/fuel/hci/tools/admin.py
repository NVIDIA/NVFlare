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

import argparse
import os

from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.hci.client.cli import AdminClient, CredentialType
from nvflare.fuel.hci.client.file_transfer import FileTransferModule
from nvflare.private.fed.app.fl_conf import FLAdminClientStarterConfigurator


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
        workspace = os.path.join(args.workspace, "startup")
        conf = FLAdminClientStarterConfigurator(app_root=workspace, admin_config_file_name=args.fed_admin)
        conf.configure()
    except ConfigError as ex:
        print("ConfigError:", str(ex))

    try:
        admin_config = conf.config_data["admin"]
    except KeyError:
        print("Missing admin section in fed_admin configuration.")

    modules = []

    if admin_config.get("with_file_transfer"):
        modules.append(
            FileTransferModule(upload_dir=admin_config.get("upload_dir"), download_dir=admin_config.get("download_dir"))
        )

    ca_cert = admin_config.get("ca_cert", "")
    client_cert = admin_config.get("client_cert", "")
    client_key = admin_config.get("client_key", "")

    if admin_config.get("with_ssl"):
        if len(ca_cert) <= 0:
            print("missing CA Cert file name field ca_cert in fed_admin configuration")
            return

        if len(client_cert) <= 0:
            print("missing Client Cert file name field client_cert in fed_admin configuration")
            return

        if len(client_key) <= 0:
            print("missing Client Key file name field client_key in fed_admin configuration")
            return
    else:
        ca_cert = None
        client_key = None
        client_cert = None

    if args.with_debug:
        print("SSL: {}".format(admin_config.get("with_ssl")))
        print("File Transfer: {}".format(admin_config.get("with_file_transfer")))

        if admin_config.get("with_file_transfer"):
            print("  Upload Dir: {}".format(admin_config.get("upload_dir")))
            print("  Download Dir: {}".format(admin_config.get("download_dir")))

    client = AdminClient(
        prompt=admin_config.get("prompt", "> "),
        cmd_modules=modules,
        ca_cert=ca_cert,
        client_cert=client_cert,
        client_key=client_key,
        upload_dir=admin_config.get("upload_dir"),
        download_dir=admin_config.get("download_dir"),
        require_login=admin_config.get("with_login", True),
        credential_type=CredentialType.PASSWORD if admin_config.get("cred_type") == "password" else CredentialType.CERT,
        debug=args.with_debug,
        overseer_agent=conf.overseer_agent,
        # cli_history_size=args.cli_history_size,
    )

    client.run()


if __name__ == "__main__":
    main()
