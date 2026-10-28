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

"""Exported jobs run on the real NVFlare 2.8.1 simulator: portability, the
external-process client, failed and silent clients, late replies, and a
retried round.

Every job here is exported for deployment (fl.nvflare.env: export) and the
exported folder is what runs, as it would on a real federation.
"""

from __future__ import annotations

import json
import re
import shutil

import torch

from tests.nvflare_real.conftest import (
    NAME,
    export,
    mock_config,
    server_log,
    server_outputs,
    simulate,
    use_fault_client,
)


def _final_checkpoint(workspace):
    return server_outputs(workspace) / "checkpoints" / "server_round_final.pt"


def _updates_per_round(workspace) -> list[int]:
    return [int(n) for n in re.findall(r"round \d+  updates=(\d+)", server_log(workspace))]


_HELD_OUT = {"tasks": [{"suite": "mock", "dataset_task_index": 3, "eval_task_id": 3, "task_name": "task_3"}]}


def test_exported_job_runs_after_moving_it_and_hiding_its_configs(tmp_path):
    """The job carries its run config and the files it names, and no path of
    the submitting machine: moved elsewhere, with the source configs gone, it
    still runs."""
    cfg = mock_config()
    cfg["backend"]["extra"] = {"train_config_path": "pipeline.yaml"}  # packaged; the mock backend ignores it
    cfg["data"]["partitioning"] = {"scheme": "iid", "held_out": "heldout.json"}
    job = export(tmp_path, cfg, extra_files={"pipeline.yaml": "policy: {type: mock}\n",
                                             "heldout.json": json.dumps(_HELD_OUT)})

    # The script in custom/; the run config and the files it names in config/,
    # where `nvflare job submit` leaves them as they are.
    assert sorted(p.name for p in (job / "app" / "custom").iterdir()) == ["client_script.py"]
    config = job / "app" / "config"
    assert {"decent_vla_run.yaml", "pipeline.yaml", "heldout.json"} <= {p.name for p in config.iterdir()}
    for f in config.iterdir():
        assert str(tmp_path) not in f.read_text(), f.name

    moved = tmp_path / "elsewhere" / NAME
    shutil.copytree(job, moved)
    shutil.rmtree(tmp_path / "src")
    shutil.rmtree(tmp_path / "jobs")
    ws = simulate(moved, tmp_path / "ws")

    assert _final_checkpoint(ws).exists()
    assert _updates_per_round(ws) == [2, 2]
    manifest = json.loads((server_outputs(ws) / "partition_manifest.json").read_text())
    assert manifest["held_out_task_names"] == ["task_3"]


def test_external_process_client_runs_the_packaged_script(tmp_path):
    job = export(tmp_path, mock_config(nvflare={"launch_external_process": True}))

    client_cfg = json.loads((job / "app" / "config" / "config_fed_client.json").read_text())
    (launcher,) = [c for c in client_cfg["components"] if c["id"] == "launcher"]
    assert launcher["args"]["script"] == "python3 -u custom/client_script.py --config decent_vla_run.yaml"

    ws = simulate(job, tmp_path / "ws")
    assert _final_checkpoint(ws).exists()
    assert _updates_per_round(ws) == [2, 2]


def test_rounds_go_on_without_a_failing_client_when_min_clients_allows(tmp_path):
    """FLARE's default error policy aborted the whole job on one failed client;
    now the failure is reported and the round loop's min_clients decides."""
    job = export(tmp_path, mock_config(min_clients=1, round_timeout_s=60))
    use_fault_client(job)
    ws = simulate(job, tmp_path / "ws", fault="c1:raise")

    log = server_log(ws)
    assert "c1 failed round 0: RuntimeError: injected failure on c1" in log
    assert "FATAL_SYSTEM_ERROR" not in log
    assert _updates_per_round(ws) == [1, 1]
    assert _final_checkpoint(ws).exists()


def test_rounds_go_on_without_a_silent_client_when_min_clients_allows(tmp_path):
    """No round timeout and a client that does not answer: the quorum reaches
    FLARE as min_responses, so each round closes on the first update instead
    of waiting for the silent client forever."""
    job = export(tmp_path, mock_config(min_clients=1))
    use_fault_client(job)
    ws = simulate(job, tmp_path / "ws", fault="c1:sleep_once:30")

    assert _updates_per_round(ws) == [1, 1]
    assert _final_checkpoint(ws).exists()


def test_rounds_go_on_without_a_client_whose_script_died(tmp_path):
    """c1's script dies at start, so FLARE answers its tasks with an error code
    and drops them, yet counts them toward min_responses. The round still
    waits for c0, which is slower than the grace period."""
    job = export(tmp_path, mock_config(min_clients=1, round_timeout_s=60,
                                       nvflare={"wait_time_after_min_received": 1}))
    use_fault_client(job)
    ws = simulate(job, tmp_path / "ws", fault="c1:die,c0:sleep_once:5")

    assert _updates_per_round(ws) == [1, 1]
    assert _final_checkpoint(ws).exists()


def test_an_unreachable_quorum_fails_the_run_cleanly(tmp_path):
    job = export(tmp_path, mock_config(round_timeout_s=60))  # min_clients: all
    use_fault_client(job)
    ws = simulate(job, tmp_path / "ws", fault="c1:raise")

    log = server_log(ws)
    assert "RoundFailure: round 0: 1/2 updates after retry" in log
    assert not _final_checkpoint(ws).exists()


def test_a_late_reply_is_sent_again_not_trained_again(tmp_path):
    """c1 finishes round 0 after the round timeout, so FLARE drops its reply and
    the server sends the round again. c1 answers with its stored reply: its
    optimizer steps 2 per round, not 4 in round 0."""
    job = export(tmp_path, mock_config(round_timeout_s=10, persist_client_state=True, save_every_n_rounds=1))
    use_fault_client(job)
    ws = simulate(job, tmp_path / "ws", fault="c1:sleep_once:12")

    assert "attempt 1: got 1/2 updates" in server_log(ws)
    assert _final_checkpoint(ws).exists()
    states = ws / "c1" / "simulate_job" / "outputs" / NAME / "client_states"
    blob = torch.load(states / "c1_round_0001.pt", map_location="cpu", weights_only=False)
    assert blob["global_step"] == 4
    assert {int(s["step"]) for s in blob["optimizer"]["state"].values()} == {4}


def test_a_failed_round_is_retried_from_the_state_saved_before_it(tmp_path):
    """c1 trains round 1, then fails before replying, after its optimizer has
    stepped. It drops that state, and the server's retry trains round 1 from
    the state c1 saved after round 0: its optimizer steps 2 per round, not 6
    in all."""
    job = export(tmp_path, mock_config(round_timeout_s=60, persist_client_state=True, save_every_n_rounds=1))
    use_fault_client(job)
    ws = simulate(job, tmp_path / "ws", fault="c1:fail_after_round:1")

    assert "c1 failed round 1: RuntimeError: injected failure on c1 after training round 1" in server_log(ws)
    assert _updates_per_round(ws) == [2, 2]
    assert _final_checkpoint(ws).exists()
    states = ws / "c1" / "simulate_job" / "outputs" / NAME / "client_states"
    blob = torch.load(states / "c1_round_0001.pt", map_location="cpu", weights_only=False)
    assert blob["global_step"] == 4
    assert {int(s["step"]) for s in blob["optimizer"]["state"].values()} == {4}
