# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Local admission-gate entry point for NVFLARE agent skills."""

import argparse
import json

try:
    from .lints import run_v1_lints
except ImportError as e:
    # Only fall back for the script-vs-package case (relative import with no
    # parent package: e.name is None). A missing third-party dep (e.g. PyYAML,
    # e.name == "yaml") must re-raise with its real message instead of being
    # masked as "No module named 'lints'".
    if e.name is not None:
        raise
    from lints import run_v1_lints


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python dev_tools/agent/skills/checks/cli.py",
        description="Run deterministic v1 lint checks for NVFLARE agent skills.",
    )
    parser.add_argument("--skills-root", default="skills", help="path to the skills source root")
    parser.add_argument(
        "--evals-root",
        default=None,
        help="path to the eval-suite root (default: dev_tools/agent/skill_evals beside the skills root)",
    )
    parser.add_argument("--format", choices=["text", "json"], default="text", help="output format")
    parser.add_argument("--check", action="append", help="run one lint ID; may be repeated")
    args = parser.parse_args(argv)

    try:
        result = run_v1_lints(
            args.skills_root,
            evals_root=args.evals_root or None,
            checks=args.check,
        )
    except ValueError as e:
        _print_invalid_args(str(e), args.format)
        return 4
    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_text_result(result)
    return 0 if result["status"] == "ok" else 1


def _print_invalid_args(message: str, output_format: str) -> None:
    hint = "Use --check with one of the supported lint IDs."
    if output_format == "json":
        print(
            json.dumps(
                {
                    "schema_version": "1",
                    "status": "error",
                    "passed": False,
                    "error_code": "INVALID_ARGS",
                    "message": message,
                    "hint": hint,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"error: {message}")
        print(f"hint: {hint}")


def _print_text_result(result: dict) -> None:
    summary = result["summary"]
    print(
        "agent skill checks: "
        f"{summary.get('error_count', 0)} error(s), "
        f"{summary.get('warning_count', 0)} warning(s), "
        f"{summary.get('info_count', 0)} info finding(s)"
    )
    for finding in result["findings"]:
        location = finding["file"]
        if finding.get("line"):
            location = f"{location}:{finding['line']}"
        skill = f" [{finding['skill']}]" if finding.get("skill") else ""
        print(f"{finding['severity'].upper()} {finding['id']}{skill} {location}: {finding['message']}")
        if finding.get("hint"):
            print(f"  hint: {finding['hint']}")


if __name__ == "__main__":
    raise SystemExit(main())
