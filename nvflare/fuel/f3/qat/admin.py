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

import argparse

from nvflare.fuel.common.excepts import ConfigError
from nvflare.fuel.f3.qat.net_config import NetConfig
from nvflare.fuel.hci.client.cli import AdminClient, CredentialType
from nvflare.fuel.hci.client.static_service_finder import StaticServiceFinder
from nvflare.fuel.utils.config_service import ConfigService


def main():
    """
    Script to launch the admin client to issue admin commands to the server.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", "-c", type=str, help="config folder", required=False, default=".")
    parser.add_argument(
        "--config_file", "-f", type=str, help="config file name", required=False, default="net_config.json"
    )
    args = parser.parse_args()

    ConfigService.initialize(section_files={}, config_path=[args.config_dir])
    net_config = NetConfig(args.config_file)
    admin_host, admin_port = net_config.get_admin()
    if not admin_host or not admin_port:
        raise ConfigError("missing admin host or port in net_config")

    service_finder = StaticServiceFinder(host=admin_host, port=int(admin_port))

    client = AdminClient(
        credential_type=CredentialType.PASSWORD,
        service_finder=service_finder,
    )
    client.run()


if __name__ == "__main__":
    main()
