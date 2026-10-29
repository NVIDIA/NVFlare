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

"""Explicit site-operator reconciliation of an otherwise unresolved task.

This is a manual attestation, not scheduler release evidence. It is deliberately
absent from the CJ Cell API. Use only after ruling out live or delayed work.
"""

import argparse
import hashlib
import json
import os
import time

from nvflare.app_opt.job_launcher.slurm.manager import _job_key, _write_exclusive
from nvflare.private.fed.task_worker.protocol import _open_regular

RELEASE_FILE = "operator-release.json"


def release_attempt(workspace, identity, execution_id, reason, *, confirm_no_active_allocation=False):
    if not confirm_no_active_allocation:
        raise ValueError("operator must confirm there is no live or delayed allocation")
    if len(identity) != 4 or any(not isinstance(value, str) or not value for value in identity):
        raise ValueError("full job/site/task/attempt identity is required")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("operator reconciliation requires an audit reason")
    key = hashlib.sha256(json.dumps(tuple(identity), separators=(",", ":")).encode()).hexdigest()
    if (
        not isinstance(execution_id, str)
        or not execution_id.startswith("slurm:")
        or execution_id.rsplit(":", 1)[-1] != key
    ):
        raise ValueError("execution identity does not match this attempt")
    directory = os.path.join(os.path.realpath(workspace), ".nvflare_slurm", "tasks", _job_key(key))
    with _open_regular(os.path.join(directory, "identity.json")) as stream:
        actual = json.load(stream)
    if actual != dict(zip(("job_id", "site_name", "task_id", "attempt_id"), identity)):
        raise ValueError("stored task identity differs from the requested release")
    path = os.path.join(directory, RELEASE_FILE)
    record = dict(
        identity=list(identity), execution_id=execution_id, reason=reason.strip(), uid=os.getuid(), time=time.time()
    )
    _write_exclusive(path, json.dumps(record).encode(), 0o600)
    return path


def operator_release_reason(handle):
    try:
        with _open_regular(os.path.join(handle.job_dir, RELEASE_FILE)) as stream:
            record = json.loads(stream.read(65537))
    except FileNotFoundError:
        return None
    if (
        not isinstance(record, dict)
        or record.get("identity") != list(handle.request.identity)
        or record.get("execution_id") != handle.execution_id
        or record.get("uid") != os.getuid()
        or not isinstance(record.get("reason"), str)
        or not record["reason"].strip()
    ):
        raise ValueError("operator release record is invalid or has a stale allocation identity")
    return record["reason"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--site-name", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--execution-id", required=True, help="Exact identity from scheduler.jsonl")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--confirm-no-active-allocation", action="store_true")
    args = parser.parse_args()
    try:
        path = release_attempt(
            args.workspace,
            (args.job_id, args.site_name, args.task_id, args.attempt_id),
            args.execution_id,
            args.reason,
            confirm_no_active_allocation=args.confirm_no_active_allocation,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Recorded operator release: {path}")


if __name__ == "__main__":
    main()
