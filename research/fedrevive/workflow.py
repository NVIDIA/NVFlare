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

"""Research phase scheduling over core async acceptance and Collab lifecycle."""

import queue
from pathlib import Path

from data import load_manifest
from schedule import PhaseSchedule
from storage import DiskAssignment, TensorStore

from nvflare.collab import collab
from nvflare.collab.fedbuff_recipe import FedBuffTrainingCoordinator, FedBuffWorkflow
from nvflare.collab.recipe import CollabRecipe


class PhaseCoordinator(FedBuffTrainingCoordinator):
    """Disk bases and speculative reservations needed by a simulated-time budget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_dir = Path(collab.workspace.get_run_dir(collab.fl_ctx.get_job_id()))
        self.bases = TensorStore(self.run_dir / "assignment_bases")

    @property
    def available_slots(self):
        # The reference reserves the configured concurrent assignments even near the version/time horizon.
        # Do not apply core's exact remaining-contribution budget to this policy.
        slots = self.concurrency - len(self.active)
        return 0 if self._closed or self.finished or slots < self.min_open_slots else slots

    def reserve_for_phase(self, logical_names, run_seed):
        if not self.bases.path(self.version).exists():
            self.bases.put(self.version, self.model)
        assignments = []
        for name in logical_names[: self.available_slots]:
            assignment = DiskAssignment(
                self._next_id,
                None,
                name,
                self.version,
                str(self.bases.path(self.version)),
                {"train_seed": run_seed * 1_000_000 + self._next_id + 1},
            )
            self._next_id += 1
            self.active[assignment.assignment_id] = assignment
            assignments.append(assignment)
        return assignments


class FedReviveWorkflow(FedBuffWorkflow):
    """Only phase/worker policy differs; core owns calls, cleanup and aggregation."""

    _coordinator_class = PhaseCoordinator

    def __init__(
        self,
        model_loader,
        model_ops,
        num_rounds,
        concurrency,
        prepared_data_root,
        run_seed,
        max_time,
        buffer_size=1,
        min_open_slots=1,
        server_lr=1.0,
        update_transform=None,
        call_timeout=3600.0,
        observer=None,
    ):
        super().__init__(
            model_loader,
            model_ops,
            num_rounds,
            concurrency,
            buffer_size,
            min_open_slots,
            server_lr,
            update_transform=update_transform,
            call_timeout=call_timeout,
            observer=observer,
        )
        if not max_time > 0:
            raise ValueError("max_time must be positive")
        self.prepared_data_root = prepared_data_root
        self.run_seed = run_seed
        self.max_time = max_time

    def _run(self, coordinator, names, submit, receive):
        profiles = load_manifest(self.prepared_data_root)["runtime_profiles"]
        scheduler = PhaseSchedule(profiles, self.run_seed, self.max_time)
        results = TensorStore(coordinator.run_dir / "assignment_results")
        scheduler.initialize(coordinator, names)
        inflight = {}
        completed_sites = {}

        def dispatch():
            for name, assignment in scheduler.dispatch():
                seed = assignment.metadata["train_seed"]
                submit(name, assignment, train_seeds={name: seed})
                inflight[name] = assignment.assignment_id
                self._notify("on_time", scheduler.time)
                self._emit("dispatch", assignment, coordinator.version, name)

        def collect():
            name, identity, result, error = receive()
            if inflight.pop(name) != identity:
                raise ValueError("physical result identity mismatch")
            if error is not None or result is None:
                raise RuntimeError(f"assignment {identity} failed: {error or 'no result'}")
            results.put(identity, result)
            completed_sites[identity] = name
            scheduler.on_result(name, identity)

        try:
            while not coordinator.finished and not scheduler.stopped:
                if collab.is_aborted:
                    raise RuntimeError("FedRevive run aborted")
                dispatch()
                identity = scheduler.next_result()
                self._notify("on_time", scheduler.time)
                if identity is not None:
                    assignment = coordinator.active[identity]
                    result = results.load(identity)
                    version = coordinator.version
                    published = coordinator.accept(assignment, self._normalize(result, assignment))
                    scheduler.on_accept(identity)
                    self._emit("accept", assignment, version, completed_sites.pop(identity))
                    if published:
                        self._notify("on_model", coordinator.model, coordinator.version)
                    # Reference evaluation precedes logical refill.
                    continue
                if scheduler.stopped:
                    break
                if not inflight:
                    raise RuntimeError("FedRevive workflow cannot make progress")
                try:
                    collect()
                except queue.Empty:
                    continue
            # Preserve all physically completed results on disk without accepting
            # beyond the horizon. Remote calls finish or hit the core timeout.
            while inflight:
                if collab.is_aborted:
                    raise RuntimeError("FedRevive run aborted during drain")
                try:
                    collect()
                except queue.Empty:
                    continue
        finally:
            scheduler.close()


class FedReviveRecipe(CollabRecipe):
    """Research composition; phase/time/storage controls stay outside FedBuffRecipe."""

    def __init__(
        self,
        *,
        name,
        client,
        model_loader,
        model_ops,
        num_rounds,
        concurrency,
        prepared_data_root,
        run_seed,
        max_time,
        buffer_size=1,
        min_open_slots=1,
        server_lr=1.0,
        update_transform=None,
        call_timeout=3600.0,
        observer=None,
        **collab_options,
    ):
        workflow = FedReviveWorkflow(
            model_loader,
            model_ops,
            num_rounds,
            concurrency,
            prepared_data_root,
            run_seed,
            max_time,
            buffer_size,
            min_open_slots,
            server_lr,
            update_transform=update_transform,
            call_timeout=call_timeout,
            observer=observer,
        )
        super().__init__(job_name=name, server=workflow, client=client, **collab_options)
