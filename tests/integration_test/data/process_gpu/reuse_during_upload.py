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

"""Reuse the settled worker's exact UUID while its real CJ publication waits."""

import argparse
import json
import sqlite3
import time
from pathlib import Path

from nvflare.app_common.resource_managers.gpu_reservation_ledger import GpuReservationLedger
from nvflare.app_common.task_launcher.process_launcher import ProcessTaskLauncher
from tests.integration_test.slow.process_task_gpu_test import _launch, _ready


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allocator-config", required=True)
    parser.add_argument("--pause-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.pause_root).resolve()
    deadline = time.monotonic() + 120
    while not list(root.glob("pause-*.json")):
        if time.monotonic() >= deadline:
            raise TimeoutError("no real CJ publication pause was observed")
        time.sleep(0.05)
    markers = list(root.glob("pause-*.json"))
    if len(markers) != 1:
        raise RuntimeError("use a fresh pause directory for exactly one scheduled CJ case")
    paused = json.loads(markers[0].read_text())
    identity = json.dumps(paused["identity"], separators=(",", ":"))
    partition = json.loads(Path(args.allocator_config).read_text())
    ledger = Path(partition["reservation_root"]) / "process-gpu.sqlite3"
    with sqlite3.connect(ledger.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT devices,released,released_at,execution_id FROM attempts WHERE identity=?", (identity,)
        ).fetchone()
    if row is None or row[1] != 1 or len(json.loads(row[0])) != 1:
        raise RuntimeError("the paused CJ's exact GPU reservation is not confirmed released")
    partition["devices"] = json.loads(row[0])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    launcher = ProcessTaskLauncher(gpu_allocator=GpuReservationLedger(**partition))
    probe = _launch(launcher, output / "reuse.json")
    try:
        if not probe.wait_for_settlement(60).succeeded:
            raise RuntimeError("independent real-CUDA reuse worker failed")
        facts = _ready(output / "reuse.json")
        if facts["visible"] != partition["devices"][0]:
            raise RuntimeError("reuse worker did not receive the prior task's exact UUID")
    finally:
        probe.cancel()
    resume = Path(paused["resume"]).resolve()
    if resume.parent != root:
        raise ValueError("resume marker must belong to the scheduled pause directory")
    resume.touch(exist_ok=False)
    finished = None
    dropped = False
    deadline = time.monotonic() + 120
    while finished is None:
        for path in root.glob("*.jsonl"):
            for line in path.read_text().splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event["identity"] != paused["identity"]:
                    continue
                dropped |= event["event"] == "receipt_observation_dropped"
                if event["event"] == "cj_publication_finished":
                    finished = event
        if finished is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("real CJ publication did not finish")
            time.sleep(0.05)
    if not (finished["success"] and finished["receipt"] == "RECEIVED" and dropped):
        raise RuntimeError("the real CJ did not recover the deliberately lost receipt observation")
    if not row[2] < paused["time"] < facts["computed_at"] < finished["time"]:
        raise RuntimeError("GPU release/reuse/publication timeline did not establish required ordering")
    evidence = {
        "paused": paused,
        "original_execution_id": row[3],
        "original_released_at": row[2],
        "reuse_execution_id": probe.execution_id,
        "reuse": facts,
        "publication": finished,
        "receipt_retry_verified": dropped,
    }
    (output / "timeline.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
