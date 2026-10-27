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

"""NVFlare server workflow: decent-vla's round loop inside a ModelController.

This module imports nvflare at import time (the class must exist at module
level so the job config can reference it by path); everything else in the
package keeps nvflare imports lazy. The controller's run() is exactly the
shared `runtime/server_session.run_fl_server` over an NVFlareCommunicator —
no nvflare workflow (FedAvg controller, aggregator, persistor) is involved,
so decent-vla strategies, checkpoints and resume behave identically to the
local plane.
"""

from __future__ import annotations

import logging

from nvflare.app_common.workflows.model_controller import ModelController

from decent_vla.core.communicator import Topology
from decent_vla.core.strategy import ClientUpdate
from decent_vla.runtime.nvflare import codec
from decent_vla.runtime.server_session import run_fl_server, write_final_artifacts

log = logging.getLogger(__name__)


def site_of(cid: str, site_prefix: str) -> str:
    """Inverse of client.cid_from_site: decent-vla cid -> nvflare site name."""
    return cid if not site_prefix else f"{site_prefix}{int(cid[1:]) + 1}"


class NVFlareCommunicator:
    """Server-side Communicator over ModelController.send_model_and_wait."""

    def __init__(self, controller: ModelController, num_clients: int, site_prefix: str = "",
                 nvf_cfg=None):
        self.controller = controller
        self.num_clients = num_clients
        self.site_prefix = site_prefix
        self._nvf_cfg = nvf_cfg      # fl.nvflare block, for memory_gc_rounds
        self._pending = None

    def broadcast_state(self, state, plan):
        # Stash only — the round loop broadcasts again before a retry pass, and
        # the actual send must go ONLY to the clients collect_updates targets.
        self._pending = (state, plan)

    def _cleanup_memory(self, plan) -> None:
        """Reclaim the finished round's payload — NVFlare will not do it for us.

        `wf_comm_server._do_check_tasks` retires a finished Task without nulling
        `task.data`; `_release_task_resources` (which does) is only reachable from
        `finalize_run`, i.e. once per JOB. Since `Task.client_tasks` and
        `ClientTask.task` reference each other, the retired task is a genuine
        cycle: refcounting can never free its payload, only a gen-2 collection.
        NVFlare's own answer is `cleanup_memory()` (gc.collect + malloc_trim),
        but it hangs that off the CONCRETE workflows
        (BaseFedAvg._maybe_cleanup_memory) — subclassing ModelController, as we
        do, opts out of it silently.

        MEASURED without this: the server's RSS ratcheted ~6.7 GiB per round
        (one fp16 payload), reaching 76 GiB by round ~8 and tripping the cgroup
        cap. Logs RSS either side so the effect is a
        measurement, not a belief.
        """
        n = getattr(self._nvf_cfg, "memory_gc_rounds", 1) if self._nvf_cfg else 1
        if not n or (int(getattr(plan, "round_num", 0)) + 1) % int(n):
            return
        try:
            import os

            import psutil

            from nvflare.fuel.utils.memory_utils import cleanup_memory

            proc = psutil.Process(os.getpid())
            before = proc.memory_info().rss / 2**30
            cleanup_memory()
            after = proc.memory_info().rss / 2**30
            log.info(
                "nvflare server memory cleanup after round %d: RSS %.2f -> %.2f GiB (freed %.2f)",
                int(getattr(plan, "round_num", 0)) + 1, before, after, before - after,
            )
        except Exception as exc:  # never let bookkeeping kill a 24h run
            log.warning("memory cleanup skipped (%s)", exc)

    def collect_updates(self, expected, timeout_s=None) -> list[ClientUpdate]:
        state, plan = self._pending
        # Send ONLY to the expected clients: the round loop's retry passes the
        # *missing* subset, and re-training clients that already responded
        # would double-step their persistent optimizer state within one round.
        import math

        targets = [site_of(cid, self.site_prefix) for cid in expected]
        data = codec.round_flmodel(state, plan)
        # Release our handles before blocking for replies. `_pending` in
        # particular must not survive the round: once round_loop rebinds
        # global_state to the aggregate's result, a live `_pending` also pins the
        # PREVIOUS master, so two full fp32 copies coexist through the next
        # round. round_loop re-broadcasts before every collect (retry included),
        # so clearing it here is safe.
        del state
        self._pending = None
        results = self.controller.send_model_and_wait(
            task_name="train",
            data=data,
            targets=targets,
            # nvflare Tasks take integer seconds and treat 0 as "never time
            # out", so ceil rather than truncate: int(0.5)==0 would silently
            # turn a sub-second fl.round_timeout_s into an infinite wait.
            timeout=math.ceil(timeout_s) if timeout_s else 0,  # 0 = wait forever
        )
        del data  # the broadcast envelope is dead once the round has returned
        # A client that errors mid-round yields no usable FLModel; skip
        # anything without our meta payload and let the round loop's
        # min_clients/retry handle the resulting partial result.
        # Decode destructively: the reply's numpy payload and the decoded
        # tensors would otherwise both be reachable (the controller keeps its
        # results list for the round). Nothing reads a reply's params again.
        updates = [
            codec.flmodel_to_update(m, consume=True)
            for m in results
            if m is not None and m.params and codec.META_KEY in (m.meta or {})
        ]
        self._cleanup_memory(plan)
        updates = [u for u in updates if u.client_id in set(expected)]
        seen: set[str] = set()
        for u in updates:
            if u.client_id in seen:
                raise RuntimeError(
                    f"nvflare: duplicate client_id {u.client_id!r} in round replies — "
                    f"two sites resolved to the same cid (check site names / "
                    f"fl.nvflare.site_prefix); a duplicate shard would be silently "
                    f"double-counted in aggregation"
                )
            seen.add(u.client_id)
        return updates

    def receive_round(self):  # pragma: no cover
        raise RuntimeError("server-side communicator")

    def send_update(self, update):  # pragma: no cover
        raise RuntimeError("server-side communicator")

    def topology(self) -> Topology:
        return Topology(kind="star")

    def close(self) -> None:
        pass


class DecentVLAController(ModelController):
    """The job's server workflow. Ctor args must stay JSON-serializable —
    the job config re-instantiates the controller from them on the server.

    `config_path`, `output_dir` and `workdir` are absolutized at job-build
    time because simulator/POC server processes run from a workspace cwd,
    not the repo.
    """

    def __init__(self, config_path: str, output_dir: str | None = None, workdir: str | None = None):
        # persistor_id="" skips BaseModelController's persistor lookup — model
        # init/persistence is decent-vla's job (bootstrap backend + our own
        # server_round_*.pt), not a LearnablePersistor's.
        super().__init__(persistor_id="")
        self.config_path = config_path
        self.output_dir = output_dir
        self.workdir = workdir

    def run(self):
        from decent_vla.configs import load_run_config
        from decent_vla.runtime.nvflare.client import restore_workdir

        restore_workdir(self.workdir, log=log)
        cfg = load_run_config(self.config_path)
        if self.output_dir:
            cfg.fl.output_dir = self.output_dir
        codec.set_wire_dtype(cfg.fl.wire_dtype)
        log.info(
            "nvflare server: %d clients x %d rounds (strategy %s) -> %s",
            cfg.data.num_clients, cfg.fl.num_rounds, cfg.fl.strategy.name, cfg.fl.output_dir,
        )
        outcome = run_fl_server(
            cfg,
            make_communicator=lambda: NVFlareCommunicator(
                self,
                num_clients=cfg.data.num_clients,
                site_prefix=cfg.fl.nvflare.site_prefix,
                nvf_cfg=cfg.fl.nvflare,
            ),
        )
        ckpt = write_final_artifacts(cfg, outcome)
        log.info("nvflare FL complete: %d rounds -> %s", cfg.fl.num_rounds, ckpt)
