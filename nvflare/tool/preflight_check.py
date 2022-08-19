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

from nvflare.tool.package_checker import (
    ClientPackageChecker,
    NVFlareConsolePackageChecker,
    OverseerPackageChecker,
    ServerPackageChecker,
)


def define_preflight_check_parser(parser):
    parser.add_argument("--package_root", required=True, type=str, help="root folder of all the packages")
    parser.add_argument("--packages", type=str, nargs="*")


def check_packages(args):
    package_root = args.package_root
    if not os.path.isdir(package_root):
        print(f"package_root {package_root} is not a valid directory.")
        return

    if not args.packages:
        print("Did not specify any package.")
        return

    package_names = list(os.listdir(package_root))
    package_to_check = args.packages
    for name in package_to_check:
        if name not in package_names:
            print(f"package name {name} is not in the specified root dir.")
            return

        if not os.path.isdir(os.path.join(package_root, name, "startup")):
            print(f"package {name} is not in the correct format.")
            return

    package_checkers = [
        OverseerPackageChecker(),
        ServerPackageChecker(),
        ClientPackageChecker(),
        NVFlareConsolePackageChecker(),
    ]
    for p in package_checkers:
        for name in package_to_check:
            package_path = os.path.abspath(os.path.join(package_root, name))
            p.init(package_path=package_path)
            if p.should_be_checked():
                p.check()
        p.print_report()


def main():
    parser = argparse.ArgumentParser("nvflare preflight check")
    define_preflight_check_parser(parser)
    args = parser.parse_args()
    check_packages(args)


if __name__ == "__main__":
    main()
