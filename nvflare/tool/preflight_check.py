# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.tool.package_checker import (
    ClientPackageChecker,
    NVFlareConsolePackageChecker,
    OverseerPackageChecker,
    ServerPackageChecker,
)


def define_preflight_check_parser(parser):
    parser.add_argument("-p", "--package_path", required=True, type=str, help="path to specific package")


def check_packages(args):
    package_path = args.package_path
    if not os.path.isdir(package_path):
        print(f"package_path {package_path} is not a valid directory.")
        return

    if not os.path.isdir(os.path.join(package_path, "startup")):
        print(f"package in {package_path} is not in the correct format.")
        return

    package_checkers = [
        OverseerPackageChecker(),
        ServerPackageChecker(),
        ClientPackageChecker(),
        NVFlareConsolePackageChecker(),
    ]
    for p in package_checkers:
        p.init(package_path=package_path)
        ret_code = 0
        if p.should_be_checked():
            ret_code = p.check()
        p.print_report()

        if ret_code == 1:
            p.stop_dry_run(force=False)
        elif ret_code == 2:
            p.stop_dry_run(force=True)


def main():
    parser = argparse.ArgumentParser("nvflare preflight check")
    define_preflight_check_parser(parser)
    args = parser.parse_args()
    check_packages(args)


if __name__ == "__main__":
    main()
