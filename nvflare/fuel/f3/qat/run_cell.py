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
from nvflare.fuel.f3.qat.cell_runner import CellRunner
from nvflare.fuel.utils.config_service import ConfigService


def main():
    """
    Script to launch the admin client to issue admin commands to the server.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", "-c", type=str, help="config folder", required=True)
    parser.add_argument("--name", "-n", type=str, help="my cell name", required=True)
    parser.add_argument("--parent_fqcn", "-pn", type=str, help="parent cell name", required=False, default="")
    parser.add_argument("--parent_url", "-pu", type=str, help="parent cell url", required=True, default="")
    args = parser.parse_args()

    ConfigService.initialize(section_files={}, config_path=[args.config_dir])
    runner = CellRunner(
        config_path=args.config_dir,
        my_name=args.name,
        parent_url=args.parent_url,
        parent_fqcn=args.parent_fqcn
    )
    runner.run()


if __name__ == "__main__":
    main()
