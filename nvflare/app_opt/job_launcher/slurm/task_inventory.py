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

"""Read-only inventory of retained Slurm task attempts for operator reconciliation."""

import argparse
import hashlib
import json
import os
import re

_KEY = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = ("job_id", "site_name", "task_id", "attempt_id")


def _read_attempt(path, key):
    item = {"directory_key": key, "state": "unconfirmed"}
    identity_path = os.path.join(path, "identity.json")
    journal_path = os.path.join(path, "scheduler.jsonl")
    try:
        if os.path.islink(identity_path) or os.path.islink(journal_path):
            raise ValueError("task control file is a symlink")
        with open(identity_path, encoding="utf-8") as stream:
            identity = json.load(stream)
        if not isinstance(identity, dict) or any(not isinstance(identity.get(field), str) for field in _FIELDS):
            raise ValueError("invalid task identity")
        values = tuple(identity[field] for field in _FIELDS)
        digest = hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("utf-8")).hexdigest()
        if hashlib.sha256(digest.encode("utf-8")).hexdigest() != key:
            raise ValueError("task identity does not match directory key")
        item.update({field: identity[field] for field in _FIELDS})
        item["attempt_key"] = digest
        item["scheduler_job_name"] = f"nvfl-task-{digest[:48]}"
        item["scheduler_comment"] = f"nvfl-task:{digest}"
        item["execution_id"] = f"slurm:local:unresolved:{digest}"
        try:
            with open(journal_path, encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("invalid scheduler journal record")
                    execution_id = record.get("execution_id")
                    if not isinstance(execution_id, str) or not execution_id.endswith(f":{digest}"):
                        raise ValueError("scheduler journal identity does not match directory key")
                    item["execution_id"] = execution_id
                    if record.get("event") == "status":
                        if record.get("settled") is True and record.get("phase") != "terminal":
                            raise ValueError("nonterminal scheduler status cannot confirm settlement")
                        item["state"] = "confirmed_settled" if record.get("settled") is True else "unconfirmed"
                        item["phase"] = record.get("phase")
                        item["failure_reason"] = record.get("failure_reason")
                    if record.get("event") == "cancel_ack":
                        item["cancel_acknowledgement"] = record.get("status")
        except FileNotFoundError:
            # A crash before the first scheduler record leaves no journal.
            pass
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        item["state"] = "unconfirmed"
        item["inventory_error"] = f"{type(error).__name__}: {error}"
    return item


def inventory(workspace):
    """Return retained attempts; only a recorded native settlement is marked confirmed."""
    root = os.path.join(workspace, ".nvflare_slurm", "tasks")
    if os.path.islink(root):
        raise ValueError("task control directory is a symlink")
    with os.scandir(root) as entries:
        paths = sorted((entry.name, entry.path) for entry in entries if entry.is_dir(follow_symlinks=False))
    return [_read_attempt(path, key) for key, path in paths if _KEY.fullmatch(key)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, help="CP site workspace containing .nvflare_slurm/tasks")
    args = parser.parse_args(argv)
    try:
        attempts = inventory(args.workspace)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for attempt in attempts:
        print(json.dumps(attempt, sort_keys=True))


if __name__ == "__main__":
    main()
