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

import os
import queue
import socket
from contextlib import ExitStack
from datetime import timedelta
from unittest.mock import patch

import pytest

from tests.timing_utils import stop_processes

torch = pytest.importorskip("torch")
pl = pytest.importorskip("pytorch_lightning")

gloo_available = torch.distributed.is_available() and torch.distributed.is_gloo_available()
nccl_available = torch.distributed.is_available() and torch.distributed.is_nccl_available()
two_gpus = torch.cuda.is_available() and torch.cuda.device_count() >= 2
running_under_xdist = bool(os.environ.get("PYTEST_XDIST_WORKER"))


class _FakeFlare:
    def __init__(self, tasks, rank):
        self._rank = rank
        self._tasks = list(tasks)
        self._current = None
        self.events = []
        self.sent_steps = []

    def receive(self, *args, **kwargs):
        if self._current is None:
            if not self._tasks:
                self.events.append("receive:none")
                return None
            self._current = self._tasks.pop(0)
        self.events.append("receive")
        return self._current[1]

    def is_running(self):
        return self.receive() is not None

    def _task(self):
        if self._rank != 0:
            raise RuntimeError("only rank 0 can call is_train/is_evaluate/is_submit_model!")
        return self._current[0] if self._current is not None else None

    def is_train(self):
        return self._task() == "train"

    def is_evaluate(self):
        return self._task() == "evaluate"

    def is_submit_model(self):
        return self._task() == "submit_model"

    def send(self, model, clear_cache=True):
        from nvflare.app_common.abstract.fl_model import MetaKey

        self.sent_steps.append(model.meta.get(MetaKey.NUM_STEPS_CURRENT_ROUND) if model.metrics is None else "metrics")
        self.events.append("send")
        if clear_cache:
            self.clear()

    def clear(self):
        self._current = None


def _worker(rank, port, result_queue, accelerator):
    import torch.nn.functional as F
    from pytorch_lightning.plugins.environments import LightningEnvironment
    from pytorch_lightning.strategies import DDPStrategy
    from torch.utils.data import DataLoader, TensorDataset

    from nvflare.app_common.abstract.fl_model import FLModel
    from nvflare.app_opt.lightning.api import patch as lightning_patch

    class Net(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2)
            self.seen = []
            self.validated = 0

        def training_step(self, batch, batch_idx):
            self.seen.append((self.fc.weight.device.type, float(self.fc.weight.detach()[0, 0])))
            inputs, labels = batch
            return F.cross_entropy(self.fc(inputs), labels)

        def validation_step(self, batch, batch_idx):
            inputs, labels = batch
            self.log("val_loss", F.cross_entropy(self.fc(inputs), labels), sync_dist=True)

        def on_validation_end(self):
            self.validated += 1

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE="2", NODE_RANK="0", LOCAL_RANK=str(rank)
        )
        weights = [{"fc.weight": torch.full((2, 2), float(r)), "fc.bias": torch.zeros(2)} for r in range(3)]
        tasks = [("train", FLModel(params=weights[r], current_round=r)) for r in range(2)]
        tasks += [("evaluate", FLModel(params=weights[2])), ("submit_model", FLModel(params={}))]
        fake = _FakeFlare(tasks if rank == 0 else [], rank)
        api = "nvflare.app_opt.lightning.api"
        with ExitStack() as stack:
            stack.enter_context(patch(f"{api}.init"))
            stack.enter_context(patch(f"{api}.get_config", return_value={}))
            for name in ("receive", "is_train", "is_evaluate", "is_submit_model", "send", "clear"):
                stack.enter_context(patch(f"{api}.{name}", new=getattr(fake, name)))

            backend = "nccl" if accelerator == "gpu" else "gloo"
            trainer = pl.Trainer(
                accelerator=accelerator,
                devices=2,
                strategy=DDPStrategy(process_group_backend=backend, timeout=timedelta(seconds=60)),
                plugins=[LightningEnvironment()],
                max_epochs=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                num_sanity_val_steps=0,
            )
            lightning_patch(trainer, sync_every_n_steps=2)
            inputs = torch.arange(8, dtype=torch.float32).unsqueeze(1).repeat(1, 2)
            loader = DataLoader(TensorDataset(inputs, torch.arange(8) % 2), batch_size=1, shuffle=False)
            net = Net()
            trainer.fit(net, train_dataloaders=loader)
            while True:
                evaluate = trainer.global_rank == 0 and fake.is_running() and fake.is_evaluate()
                if not trainer.strategy.broadcast(evaluate, src=0):
                    break
                trainer.validate(net, dataloaders=loader)
            module = trainer.lightning_module
        result_queue.put(
            {
                "rank": rank,
                "ok": True,
                "global_step": trainer.global_step,
                "sent_steps": fake.sent_steps,
                "events": fake.events,
                "seen": module.seen,
                "validated": net.validated,
                "pending_task": fake._current[0] if fake._current else None,
            }
        )
    except Exception as e:
        result_queue.put({"rank": rank, "ok": False, "error": repr(e)})


@pytest.mark.skipif(running_under_xdist, reason="nested torch.multiprocessing DDP tests are unstable under xdist")
@pytest.mark.parametrize(
    "accelerator",
    [
        pytest.param("cpu", marks=pytest.mark.skipif(not gloo_available, reason="torch.distributed gloo is required")),
        pytest.param(
            "gpu",
            marks=pytest.mark.skipif(not (nccl_available and two_gpus), reason="NCCL and two GPUs are required"),
        ),
    ],
)
def test_step_sync_under_ddp_pauses_all_ranks_together_and_only_rank_zero_exchanges(accelerator):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    ctx = torch.multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(rank, port, result_queue, accelerator)) for rank in range(2)]
    try:
        for proc in procs:
            proc.start()
    finally:
        stop_processes(*procs, timeout=60)
    assert all(proc.exitcode == 0 for proc in procs)

    results = []
    for _ in procs:
        try:
            results.append(result_queue.get(timeout=5))
        except queue.Empty:
            pytest.fail("timed out waiting for a DDP step-sync worker result")
    rank0, rank1 = sorted(results, key=lambda result: result["rank"])
    assert rank0["ok"] and rank1["ok"], results

    assert rank0["global_step"] == rank1["global_step"] == 4
    assert rank0["sent_steps"] == [2, 2, "metrics"]
    assert rank0["validated"] == rank1["validated"] == 1
    assert rank0["pending_task"] == "submit_model"
    assert rank1["events"] == []
    assert rank0["seen"] == rank1["seen"]
    assert {device for device, _ in rank0["seen"]} == {"cuda" if accelerator == "gpu" else "cpu"}
    assert [rank0["seen"][0][1], rank0["seen"][2][1]] == [0.0, 1.0]
