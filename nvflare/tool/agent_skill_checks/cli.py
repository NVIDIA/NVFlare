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
from pathlib import Path

from nvflare.tool.agent_skill_checks.lints import run_v1_lints


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m nvflare.tool.agent_skill_checks",
        description="Run deterministic v1 lint checks for NVFLARE agent skills.",
    )
    parser.add_argument("--skills-root", default="skills", help="path to the skills source root")
    parser.add_argument("--docs-root", help="optional docs/design root for cross-document link checks")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="output format")
    parser.add_argument("--check", action="append", help="run one lint ID; may be repeated")
    args = parser.parse_args(argv)

    result = run_v1_lints(
        Path(args.skills_root),
        docs_root=Path(args.docs_root) if args.docs_root else None,
        checks=args.check,
    )
    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_text_result(result)
    return 0 if result["status"] == "ok" else 1


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
