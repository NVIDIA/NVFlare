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

"""Out-of-band audit coverage check; never a transport/release enforcement filter."""

import argparse
import json
from pathlib import Path


def coverage(reports, expected):
    """Require one complete measurement per expected (site, round), not just a report count."""
    groups = {}
    for report in reports:
        if report.get("stage") == "client_task_result":
            groups.setdefault((report.get("site"), report.get("round")), []).append(report)
    problems = []
    for key in sorted(set(expected) | set(groups)):
        values = groups.get(key, [])
        if key not in expected or len(values) != 1:
            problems.append({"site": key[0], "round": key[1], "reason": "unexpected, missing or duplicate report"})
        elif values[0].get("status") != "completed":
            problems.append({"site": key[0], "round": key[1], "reason": values[0].get("status", "unknown")})
        elif (
            not values[0].get("baseline_fingerprint")
            or not values[0].get("model_fingerprint")
            or any(
                values[0].get("sample_counts", {}).get(score) != {"members": 8, "nonmembers": 8}
                for score in ("likelihood", "tra")
            )
        ):
            problems.append({"site": key[0], "round": key[1], "reason": "incomplete evidence"})
    return {
        "complete": not problems,
        "expected": len(expected),
        "reports": sum(map(len, groups.values())),
        "problems": problems,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace")
    args = parser.parse_args()
    reports = [json.loads(p.read_text()) for p in Path(args.workspace).rglob("memorization_audits/*.json")]
    result = coverage(reports, {(site, round_) for site in ("site-1", "site-2") for round_ in (0, 1)})
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
