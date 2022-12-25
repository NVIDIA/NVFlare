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
import logging
from nvflare.fuel.f3.qat.server import Server
from nvflare.fuel.utils.config_service import ConfigService


def main():
    """
    Script to launch the admin client to issue admin commands to the server.
    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", "-c", type=str, help="config folder", required=False, default=".")
    parser.add_argument("--config_file", "-f", type=str, help="config file name", required=False, default="net_config.json")
    args = parser.parse_args()

    ConfigService.initialize(section_files={}, config_path=[args.config_dir])
    server = Server(
        config_path=args.config_dir,
        config_file=args.config_file
    )
    server.start()
    server.run()


if __name__ == "__main__":
    main()
