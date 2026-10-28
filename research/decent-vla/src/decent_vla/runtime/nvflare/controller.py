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
    """Server-side Communicator over ModelController.broadcast_model (blocking)."""

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

    def _reply_site(self, m) -> str | None:
        """The site a reply came from (FLARE names it; our meta is the fallback)."""
        meta = m.meta or {}
        if meta.get("client_name"):
            return meta["client_name"]
        ours = meta.get(codec.META_KEY) or meta.get(codec.ERROR_KEY) or {}
        cid = ours.get("client_id")
        return site_of(cid, self.site_prefix) if cid else None

    def collect_updates(self, expected, timeout_s=None, min_responses=None) -> list[ClientUpdate]:
        state, plan = self._pending
        # Send ONLY to the expected clients: the round loop's retry passes the
        # *missing* subset, and re-training clients that already responded
        # would double-step their persistent optimizer state within one round.
        import math
        import time

        targets = [site_of(cid, self.site_prefix) for cid in expected]
        need = min(min_responses, len(targets)) if min_responses else len(targets)
        data = codec.round_flmodel(state, plan)
        # Release our handles before blocking for replies. `_pending` in
        # particular must not survive the round: once round_loop rebinds
        # global_state to the aggregate's result, a live `_pending` also pins the
        # PREVIOUS master, so two full fp32 copies coexist through the next
        # round. round_loop re-broadcasts before every collect (retry included),
        # so clearing it here is safe.
        del state
        self._pending = None

        def usable(m) -> bool:
            if m is None or not m.params or codec.META_KEY not in (m.meta or {}):
                return False
            # A reply stamped with another round (one FLARE delivered after its
            # task closed) must not be aggregated into this one.
            return codec.reply_round(m) in (None, plan.round_num)

        # The round loop's quorum goes to FLARE as min_responses, so FLARE stops
        # waiting once enough clients answer (plus wait_time_after_min_received
        # for the rest, as FLARE's ScatterAndGather does) instead of waiting for
        # every target. But FLARE counts every answer, a failed one too, so a
        # wait can end before `need` usable updates are in. Then wait again for
        # the clients that have not answered, asking only for the shortfall. A
        # client still training gets the round again and answers it with its
        # stored reply (client.client_main), not with a second training pass.
        grace = int(getattr(self._nvf_cfg, "wait_time_after_min_received", 10)) if self._nvf_cfg else 10
        deadline = time.monotonic() + timeout_s if timeout_s else None
        pending, results = list(targets), []
        rejected = getattr(self.controller, "rejected_replies", [])
        while pending:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                break
            rejected.clear()
            # broadcast_model(blocking=True) is send_model_and_wait plus the grace period.
            got = self.controller.broadcast_model(
                task_name="train",
                data=data,
                targets=pending,
                min_responses=min(need - sum(map(usable, results)), len(pending)),
                # nvflare Tasks take integer seconds and treat 0 as "never time
                # out", so ceil rather than truncate: int(0.5)==0 would silently
                # turn a sub-second fl.round_timeout_s into an infinite wait.
                timeout=math.ceil(remaining) if remaining else 0,  # 0 = wait forever
                wait_time_after_min_received=grace,
                blocking=True,
            )
            results += got
            answered = {self._reply_site(m) for m in got if m is not None}
            # Replies FLARE rejected (an error code, e.g. from a client script that
            # died) never reach `got` but did count toward min_responses.
            for site, rc in list(rejected):
                log.warning("nvflare: %s failed round %d (%s)", site, plan.round_num, rc)
                answered.add(site)
            if not answered & set(pending):
                break  # timed out or aborted: no pending client answered
            pending = [t for t in pending if t not in answered]
            if sum(map(usable, results)) >= need:
                break
        del data  # the broadcast envelope is dead once the round has returned
        # A client whose round failed replies with an error (codec.ERROR_KEY)
        # and no params; FLARE-level errors are dropped before they get here
        # (ignore_result_error=True). Both count as missing, and the round
        # loop's min_clients/retry decides what happens next.
        for m in results:
            err = codec.reply_error(m)
            if err:
                log.warning("nvflare: %s failed round %d: %s",
                            self._reply_site(m) or "a client", plan.round_num, err)
        # Decode destructively: the reply's numpy payload and the decoded
        # tensors would otherwise both be reachable (the controller keeps its
        # results list for the round). Nothing reads a reply's params again.
        seen: dict[str, str] = {}  # cid -> site of its first reply
        updates = []
        for m in results:
            if not usable(m):
                continue
            site, cid = self._reply_site(m), (m.meta[codec.META_KEY] or {}).get("client_id")
            if cid in seen:
                if seen[cid] == site:
                    # The same site twice in one round (a reply FLARE delivered
                    # into the next wait): keep the first.
                    log.warning("nvflare: dropped a second reply from %s for round %d", site, plan.round_num)
                    continue
                raise RuntimeError(
                    f"nvflare: duplicate client_id {cid!r} in round replies — "
                    f"two sites resolved to the same cid (check site names / "
                    f"fl.nvflare.site_prefix); a duplicate shard would be silently "
                    f"double-counted in aggregation"
                )
            seen[cid] = site
            updates.append(codec.flmodel_to_update(m, consume=True))
        self._cleanup_memory(plan)
        return [u for u in updates if u.client_id in set(expected)]

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

    `config_path` names the run config packaged in the job (see
    `job.build_job`) and resolves against this app's `config/` folder.
    `output_dir` and `workdir` are the submitting host's paths and are set
    only for simulator runs, which execute on that host; a deployed job
    leaves them unset and uses site paths (`client.apply_site_paths`).
    """

    def __init__(self, config_path: str, output_dir: str | None = None, workdir: str | None = None):
        # persistor_id="" skips BaseModelController's persistor lookup — model
        # init/persistence is decent-vla's job (bootstrap backend + our own
        # server_round_*.pt), not a LearnablePersistor's.
        # ignore_result_error=True: FLARE drops a failed client reply instead of
        # aborting the job (its default aborts once one error makes min_responses
        # unreachable). The round loop is then the only place that decides what
        # a missing reply means: min_clients, one retry, then RoundFailure.
        super().__init__(persistor_id="", ignore_result_error=True)
        self.config_path = config_path
        self.output_dir = output_dir
        self.workdir = workdir
        self.rejected_replies: list[tuple[str, str]] = []

    def _accept_train_result(self, client_name, result, fl_ctx, is_unknown_task=False):
        # BaseModelController hook (nvflare 2.8.1). A reply FLARE rejects never
        # reaches the results list, yet it counts toward min_responses, so the
        # communicator has to know that this client answered.
        accepted = super()._accept_train_result(
            client_name=client_name, result=result, fl_ctx=fl_ctx, is_unknown_task=is_unknown_task
        )
        if not accepted and not is_unknown_task:
            self.rejected_replies.append((client_name, str(result.get_return_code())))
        return accepted

    def run(self):
        from pathlib import Path

        from decent_vla.runtime.nvflare.client import apply_site_paths, load_job_config

        job_id = self.fl_ctx.get_job_id()
        workspace = self.engine.get_workspace()
        cfg = load_job_config(Path(workspace.get_app_config_dir(job_id)) / self.config_path)
        apply_site_paths(cfg, run_dir=workspace.get_run_dir(job_id), workdir=self.workdir,
                         output_dir=self.output_dir, log=log)
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
