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

"""Aggregate manifest-bound site evidence into a production acceptance report."""

from pathlib import Path

from ..common.errors import require
from ..common.io import digest_file, read_json
from .bundle import load_public_keys, required_acceptance_checks, verify_bundle, verify_receipt_signature


def evidence_files(paths):
    """Return explicit result files, recursively expanding evidence directories."""
    files = []
    for value in paths:
        path = Path(value).resolve()
        if path.is_dir():
            files.extend(sorted(path.rglob("result.json")))
        else:
            require(path.is_file(), "Acceptance evidence path does not exist")
            files.append(path)
    require(files, "No acceptance result files were found")
    require(len(set(files)) == len(files), "Acceptance result file was supplied more than once")
    return files


def aggregate(directory, paths, trusted_keys):
    """Build the exact report consumed by ``cvmctl admin approve``.

    Each result is intentionally small and portable. It names the finalized
    manifest digest and platform plus the checks established by that evidence.
    The report records the result file's digest, so the signed approval remains
    traceable without copying private lab logs into the public bundle.
    """
    directory = Path(directory).resolve()
    manifest = verify_bundle(directory)
    manifest_sha256 = digest_file(directory / "cvm_manifest.json")
    required = required_acceptance_checks(manifest)
    trusted_keys = load_public_keys(trusted_keys)
    checks = {}
    for path in evidence_files(paths):
        result = read_json(path)
        verify_receipt_signature(result, trusted_keys)
        require(
            result.get("schema_version") == 1
            and result.get("manifest_sha256") == manifest_sha256
            and result.get("platform") == manifest["platform"],
            "Acceptance result covers another finalized manifest or platform",
        )
        outcomes = result.get("checks")
        require(
            isinstance(outcomes, dict)
            and set(outcomes) <= required
            and all(
                isinstance(name, str) and isinstance(item, dict) and set(item) == {"passed"} and item["passed"] is True
                for name, item in outcomes.items()
            ),
            "Acceptance result contains invalid or inapplicable checks",
        )
        if not outcomes:
            continue
        evidence_sha256 = digest_file(path)
        for name in outcomes:
            require(name not in checks, "Acceptance check is claimed by more than one result")
            checks[name] = {"passed": True, "evidence_sha256": evidence_sha256}
    missing = required - set(checks)
    require(not missing, "Missing acceptance checks: " + ", ".join(sorted(missing)))
    return {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "platform": manifest["platform"],
        "checks": checks,
    }
