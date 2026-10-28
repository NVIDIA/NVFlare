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
import time
from pathlib import Path

from nvflare.app_common.task_launcher.process_launcher import ProcessTaskLauncher
from tests.integration_test.tools.process_gpu.cuda_probe import launch_probe, wait_for_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices-config", required=True)
    parser.add_argument("--runtime-root", required=True, help="paused CJ run/.nvflare/task-execution directory")
    parser.add_argument("--compute-root", required=True, help="original job/process-gpu-models directory")
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
    attempt_id = paused["identity"][3]
    runtime = Path(args.runtime_root).resolve()
    completion = json.loads((runtime / "attempts" / attempt_id / "completion.json").read_text())
    identity = completion["identity"]
    if [identity[k] for k in ("job_id", "site_name", "task_id", "attempt_id")] != paused["identity"]:
        raise RuntimeError("completion does not match the paused CJ task")
    settled = [
        record
        for line in (runtime / "diagnostics.jsonl").read_text().splitlines()
        if (record := json.loads(line)).get("event") == "settled"
        and all(record.get(k) == identity[k] for k in ("job_id", "site_name", "task_id", "attempt_id"))
    ]
    if len(settled) != 1:
        raise RuntimeError("the paused CJ task has no unique confirmed settlement record")
    devices = json.loads(Path(args.devices_config).read_text())["gpu_devices"]
    if len(devices) != 1:
        raise ValueError("the scheduled reuse probe requires the original task's one assigned device")
    original_records = list(Path(args.compute_root).glob(f"{completion['worker_pid']}-*.json"))
    if len(original_records) != 1:
        raise RuntimeError("the settled worker must have exactly one compute record")
    original_compute = json.loads(original_records[0].read_text())
    if (
        original_compute["pid"] != completion["worker_pid"]
        or original_compute["cuda_visible_devices"] != devices[0]
        or original_compute["allocated_bytes"] <= 0
    ):
        raise RuntimeError("compute evidence does not match the original worker's assigned device")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    launcher = ProcessTaskLauncher(gpu_devices=devices)
    probe = launch_probe(launcher, output / "reuse.json")
    try:
        if not probe.wait_for_settlement(60).succeeded:
            raise RuntimeError("independent real-CUDA reuse worker failed")
        facts = wait_for_probe(output / "reuse.json")
        if facts["visible"] != devices[0]:
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
    if not settled[0]["timestamp"] < paused["time"] < facts["computed_at"] < finished["time"]:
        raise RuntimeError("worker settlement/reuse/publication timeline did not establish required ordering")
    evidence = {
        "paused": paused,
        "original_execution_id": settled[0]["execution_id"],
        "original_settled_at": settled[0]["timestamp"],
        "original_worker_pid": completion["worker_pid"],
        "original_compute": original_compute,
        "reuse_execution_id": probe.execution_id,
        "reuse": facts,
        "publication": finished,
        "receipt_retry_verified": dropped,
    }
    (output / "timeline.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
