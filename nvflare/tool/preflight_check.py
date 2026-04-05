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
import sys

from nvflare.tool.package_checker import ClientPackageChecker, NVFlareConsolePackageChecker, ServerPackageChecker

_preflight_parser = None


def define_preflight_check_parser(parser):
    global _preflight_parser
    _preflight_parser = parser
    parser.add_argument("-p", "--package_path", required=True, type=str, help="path to specific package")
    parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")


def check_packages(args):
    from nvflare.tool.cli_output import output_error, output_ok
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _preflight_parser,
        "nvflare preflight_check",
        ["nvflare preflight_check -p /path/to/package"],
        sys.argv[1:],
    )
    package_path = args.package_path

    if not os.path.isdir(package_path):
        output_error("INVALID_ARGS", exit_code=4, detail=f"package_path {package_path} is not a valid directory")
        return

    if not os.path.isdir(os.path.join(package_path, "startup")):
        output_error("INVALID_ARGS", exit_code=4, detail=f"package in {package_path} is not in the correct format")
        return

    package_checkers = [
        ServerPackageChecker(),
        ClientPackageChecker(),
        NVFlareConsolePackageChecker(),
    ]

    checks = []
    overall_pass = True

    for p in package_checkers:
        p.init(package_path=package_path)
        ret_code = 0
        if p.should_be_checked():
            ret_code = p.check()
        p.print_report()

        component_name = p.__class__.__name__.replace("PackageChecker", "").lower()
        status = "pass" if ret_code == 0 else "fail"
        if status == "fail":
            overall_pass = False
        checks.append({"component": component_name, "status": status, "details": ""})

        if ret_code == 1:
            p.stop_dry_run(force=False)
        elif ret_code == 2:
            p.stop_dry_run(force=True)

    overall = "pass" if overall_pass else "fail"
    output_ok(
        {
            "package": os.path.abspath(package_path),
            "checks": checks,
            "overall": overall,
        }
    )

    if not overall_pass:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser("nvflare preflight check")
    define_preflight_check_parser(parser)
    args = parser.parse_args()
    check_packages(args)


if __name__ == "__main__":
    main()
