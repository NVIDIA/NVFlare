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

from nvflare.tool.package_checker import ClientPackageChecker, NVFlareConsolePackageChecker, ServerPackageChecker
from nvflare.tool.package_checker.package_checker import CheckStatus

_preflight_parser = None


def define_preflight_check_parser(parser):
    global _preflight_parser
    _preflight_parser = parser
    parser.add_argument(
        "-p",
        "--package-path",
        "--package_path",  # backward compat
        dest="package_path",
        required=True,
        type=str,
        help="path to specific package",
    )
    parser.add_argument("--schema", action="store_true", help="print command schema as JSON and exit")


def check_packages(args):
    from nvflare.tool.cli_output import output_error, output_ok, print_human
    from nvflare.tool.cli_schema import handle_schema_flag

    handle_schema_flag(
        _preflight_parser,
        "nvflare preflight-check",
        ["nvflare preflight-check -p /path/to/package"],
        getattr(args, "_argv", []),
    )

    if getattr(args, "_raw_sub_command", None) == "preflight_check":
        print_human("Note: 'preflight_check' is deprecated; use 'nvflare preflight-check' instead.")
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
        check_status = CheckStatus.PASS
        if p.should_be_checked():
            check_status = p.check()
        p.print_report()

        component_name = p.__class__.__name__.replace("PackageChecker", "").lower()
        status = "fail" if check_status in [CheckStatus.FAIL, CheckStatus.FAIL_WITH_CLEANUP] else "pass"
        if status == "fail":
            overall_pass = False
        check_result = {"component": component_name, "status": status}
        details = getattr(p, "last_error", None)
        if isinstance(details, str) and details:
            check_result["details"] = details
        checks.append(check_result)

        if check_status == CheckStatus.PASS_WITH_CLEANUP:
            p.stop_dry_run(force=False)
        elif check_status == CheckStatus.FAIL_WITH_CLEANUP:
            p.stop_dry_run(force=True)

    overall = "pass" if overall_pass else "fail"
    output_ok(
        {
            "package": os.path.abspath(package_path),
            "checks": checks,
            "overall": overall,
        },
        exit_code=0 if overall_pass else 1,
    )


def main():
    parser = argparse.ArgumentParser("nvflare preflight check")
    define_preflight_check_parser(parser)
    args = parser.parse_args()
    check_packages(args)


if __name__ == "__main__":
    main()
