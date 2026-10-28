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

"""Tests for the nvflare controller/communicator/client envelopes, using the
fake nvflare from tests/nvflare/conftest.py. The nvflare modules are imported
INSIDE each test: the autouse fixture evicts them from sys.modules so they
bind to the fake nvflare namespace.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from decent_vla.core.strategy import RoundPlan
from decent_vla.runtime import client_session


@pytest.fixture
def run_cfg_path(tmp_path, monkeypatch):
    cfg = {
        "name": "t",
        "backend": {"name": "mock"},
        "data": {
            "partitioning": "task",
            "num_clients": 2,
            "batch_size": 4,
            "extra": {"n_tasks": 4, "eps_per_task": 8},
        },
        "fl": {
            "strategy": {"name": "fedavg"},
            "rounds": 2,
            "local_steps": 2,
            "execution": "nvflare",
            "output_dir": str(tmp_path / "out"),
            # fp32 BOTH ways: test_local_and_nvflare_paths_are_equivalent asserts
            # atol=1e-6 against LocalRunner, whose in-process communicator never
            # casts at all. With the default fp16 wire the broadcast alone would
            # perturb weights by ~5e-4, so the assertion would be measuring
            # fp16 rounding rather than what it exists to measure: that the two
            # planes' plumbing is numerically identical.
            "wire_dtype": "fp32",
        },
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("DECENT_VLA_CONFIG", str(p))
    # The warm stores live in the SHARED session module (not evicted by the
    # fake-nvflare fixture) — clear them so state never leaks between tests.
    client_session._STATES.clear()
    client_session._CACHE.clear()
    return p


class LoopbackRouter:
    """In-process double for the server->client leg: decodes the outgoing
    FLModel, runs the shared client session for each target site, and returns
    the encoded replies — exactly what nvflare's task machinery + our client
    loop do. Counts per-cid handling for retry assertions."""

    def __init__(self, cfg_path, site_prefix=""):
        self.cfg_path = str(cfg_path)
        self.site_prefix = site_prefix
        self.handled: dict[str, int] = {}
        self.seen_rounds: list[int] = []

    def _reply(self, cid, data):
        from decent_vla.runtime.nvflare import codec

        global_state, plan = codec.flmodel_to_round(data)
        self.seen_rounds.append(plan.round_num)
        update = client_session.handle_round(cid, self.cfg_path, global_state, plan)
        self.handled[cid] = self.handled.get(cid, 0) + 1
        return codec.update_to_flmodel(update)

    def __call__(self, task_name, data, targets, timeout):
        from decent_vla.runtime.nvflare.client import cid_from_site

        assert task_name == "train"
        return [self._reply(cid_from_site(t, self.site_prefix), data) for t in targets]


def _controller(run_cfg_path, router=None):
    from decent_vla.runtime.nvflare.controller import DecentVLAController

    ctrl = DecentVLAController(config_path=str(run_cfg_path))
    ctrl.loopback = router or LoopbackRouter(run_cfg_path)
    return ctrl


# ---------------------------------------------------------------- identity


def test_cid_site_mapping_roundtrip():
    from decent_vla.runtime.nvflare.client import cid_from_site
    from decent_vla.runtime.nvflare.controller import site_of

    assert cid_from_site("c0", "") == "c0"
    assert cid_from_site("c12", "") == "c12"
    assert cid_from_site("site-1", "site-") == "c0"
    assert cid_from_site("site-8", "site-") == "c7"
    for cid, prefix in [("c3", ""), ("c3", "site-")]:
        assert cid_from_site(site_of(cid, prefix), prefix) == cid


def test_cid_site_mapping_rejects_unmappable_names():
    from decent_vla.runtime.nvflare.client import cid_from_site

    with pytest.raises(ValueError, match="site name"):
        cid_from_site("hospital-a", "")
    with pytest.raises(ValueError, match="site_prefix"):
        cid_from_site("hospital-a", "site-")


# ---------------------------------------------------------------- client loop


def test_client_loop_trains_and_persists_state(run_cfg_path, fake_nvflare):
    """flare.init -> receive/train/send until the job ends; the SAME warm
    ClientState (optimizer object, cumulative global_step) across rounds."""
    from decent_vla.runtime.nvflare import codec
    from decent_vla.runtime.nvflare.client import client_main

    cfg_factory_state = _bootstrap_state(run_cfg_path)
    flare = fake_nvflare.client
    flare.site_name = "c0"
    for r in range(2):
        plan = RoundPlan(selected_clients=["c0", "c1"], local_steps=2, round_num=r)
        flare.incoming.append(codec.round_flmodel(cfg_factory_state, plan))

    client_main(str(run_cfg_path))

    assert flare.inited
    assert len(flare.sent) == 2
    ups = [codec.flmodel_to_update(m) for m in flare.sent]
    assert all(u.client_id == "c0" for u in ups)
    assert all(u.num_samples > 0 for u in ups)
    assert "c0" in client_session._STATES
    assert client_session._STATES["c0"].global_step == 4


def test_client_replays_a_repeated_round(run_cfg_path, fake_nvflare):
    """The server sends a round again when a reply was late or lost. The client
    already trained it, so it sends the stored reply back instead of training
    again (which would step its warm optimizer twice for one update)."""
    import numpy as np

    from decent_vla.runtime.nvflare import codec
    from decent_vla.runtime.nvflare.client import client_main

    state = _bootstrap_state(run_cfg_path)
    raw = yaml.safe_load(run_cfg_path.read_text())
    raw["fl"]["round_timeout_s"] = 60  # a resend can happen, so replies are kept
    run_cfg_path.write_text(yaml.safe_dump(raw))
    flare = fake_nvflare.client
    for r in (0, 0, 1):
        plan = RoundPlan(selected_clients=["c0", "c1"], local_steps=2, round_num=r)
        flare.incoming.append(codec.round_flmodel(state, plan))

    client_main(str(run_cfg_path))

    assert len(flare.sent) == 3
    first, again, _ = flare.sent
    assert first.params and set(first.params) == set(again.params)
    assert all(np.array_equal(first.params[k], again.params[k]) for k in first.params)
    assert client_session._STATES["c0"].global_step == 4  # 2 rounds x 2 steps, not 3 x 2


def test_client_reports_a_failed_round_and_keeps_serving(run_cfg_path, fake_nvflare, monkeypatch):
    from decent_vla.runtime.nvflare import codec
    from decent_vla.runtime.nvflare.client import client_main

    real = client_session.handle_round

    def flaky(cid, config_path, global_state, plan, configure=None):
        if plan.round_num == 0:
            raise ValueError("client c0 round 0 trained 0 of 2 planned steps")
        return real(cid, config_path, global_state, plan, configure=configure)

    monkeypatch.setattr(client_session, "handle_round", flaky)
    state = _bootstrap_state(run_cfg_path)
    flare = fake_nvflare.client
    for r in (0, 1):
        plan = RoundPlan(selected_clients=["c0", "c1"], local_steps=2, round_num=r)
        flare.incoming.append(codec.round_flmodel(state, plan))

    client_main(str(run_cfg_path))

    failed, trained = flare.sent
    assert "trained 0 of 2" in codec.reply_error(failed)
    assert not failed.params
    assert codec.reply_error(trained) is None
    assert codec.flmodel_to_update(trained).client_id == "c0"


@pytest.mark.parametrize("persist, adam_steps", [(False, 2), (True, 4)])
def test_a_failed_round_is_retried_from_a_rebuilt_state(run_cfg_path, monkeypatch, persist, adam_steps):
    """A round that fails after its optimizer stepped drops the client's state,
    so the retry does not train on top of it: it starts from the state saved
    before the round (fl.persist_client_state), otherwise from fresh optimizer
    state, at the round's step either way."""
    raw = yaml.safe_load(run_cfg_path.read_text())
    raw["fl"].update(persist_client_state=persist, save_every_n_rounds=1)
    run_cfg_path.write_text(yaml.safe_dump(raw))
    state, cfg = _bootstrap_state(run_cfg_path), str(run_cfg_path)

    def plan(r):
        return RoundPlan(selected_clients=["c0", "c1"], local_steps=2, round_num=r)

    client_session.handle_round("c0", cfg, state, plan(0))
    real = client_session.run_client_round

    def fails_after_training(**kwargs):
        real(**kwargs)
        raise RuntimeError("failed after training")

    monkeypatch.setattr(client_session, "run_client_round", fails_after_training)
    with pytest.raises(RuntimeError, match="failed after training"):
        client_session.handle_round("c0", cfg, state, plan(1))
    assert "c0" not in client_session._STATES

    monkeypatch.setattr(client_session, "run_client_round", real)
    client_session.handle_round("c0", cfg, state, plan(1))
    st = client_session._STATES["c0"]
    assert st.global_step == 4
    assert {int(s["step"]) for s in st.optimizer.state_dict()["state"].values()} == {adam_steps}


def test_client_keeps_no_reply_when_no_resend_can_happen(run_cfg_path, fake_nvflare):
    """Without fl.round_timeout_s or fl.min_clients the server waits for every
    client, so the last reply (a full delta) is not kept; a repeated round,
    which should not occur, gets an error instead of a second training pass."""
    from decent_vla.runtime.nvflare import codec
    from decent_vla.runtime.nvflare.client import client_main

    state = _bootstrap_state(run_cfg_path)
    flare = fake_nvflare.client
    for r in (0, 0):
        plan = RoundPlan(selected_clients=["c0", "c1"], local_steps=2, round_num=r)
        flare.incoming.append(codec.round_flmodel(state, plan))

    client_main(str(run_cfg_path))

    assert "no stored reply" in codec.reply_error(flare.sent[1])
    assert client_session._STATES["c0"].global_step == 2


def test_client_that_cannot_start_answers_with_its_error(run_cfg_path, fake_nvflare, monkeypatch):
    from decent_vla.runtime.nvflare import codec
    from decent_vla.runtime.nvflare.client import client_main

    def broken(*args, **kwargs):
        raise FileNotFoundError("DECENT_VLA_WORKDIR '/nope' does not exist on this site")

    monkeypatch.setattr(client_session, "load_session", broken)
    state = _bootstrap_state(run_cfg_path)
    flare = fake_nvflare.client
    for r in (0, 1):
        plan = RoundPlan(selected_clients=["c0", "c1"], local_steps=2, round_num=r)
        flare.incoming.append(codec.round_flmodel(state, plan))

    client_main(str(run_cfg_path))

    assert len(flare.sent) == 2
    assert all("cannot start: FileNotFoundError" in codec.reply_error(m) for m in flare.sent)


def test_client_rejects_a_round_older_than_its_last(run_cfg_path, fake_nvflare):
    from decent_vla.runtime.nvflare import codec
    from decent_vla.runtime.nvflare.client import client_main

    state = _bootstrap_state(run_cfg_path)
    flare = fake_nvflare.client
    for r in (1, 0):
        plan = RoundPlan(selected_clients=["c0", "c1"], local_steps=2, round_num=r)
        flare.incoming.append(codec.round_flmodel(state, plan))

    client_main(str(run_cfg_path))

    assert "older than this client's last round 1" in codec.reply_error(flare.sent[1])


def _bootstrap_state(run_cfg_path):
    from decent_vla.configs import load_run_config
    from decent_vla.runtime.builders import build_backend_factory, resolve_device

    cfg = load_run_config(run_cfg_path)
    factory, _ = build_backend_factory(cfg, resolve_device(cfg.device))
    return factory().get_state("shared")


# ---------------------------------------------------------------- communicator


def test_collect_updates_sends_only_to_expected(run_cfg_path):
    from decent_vla.runtime.nvflare.controller import NVFlareCommunicator

    router = LoopbackRouter(run_cfg_path)

    class Ctrl:
        sent: list[list[str]] = []

        def broadcast_model(self, task_name="train", data=None, targets=None, min_responses=None,
                            timeout=0, wait_time_after_min_received=0, blocking=True):
            Ctrl.sent.append(list(targets))
            return router(task_name, data, targets, timeout)

    comm = NVFlareCommunicator(Ctrl(), num_clients=2)
    plan = RoundPlan(selected_clients=["c0", "c1"], local_steps=2, round_num=0)
    comm.broadcast_state(_bootstrap_state(run_cfg_path), plan)
    updates = comm.collect_updates(expected=["c1"])

    assert Ctrl.sent == [["c1"]]  # only c1's site got the task
    assert [u.client_id for u in updates] == ["c1"]


def test_collect_updates_rounds_timeout_up_to_whole_seconds(run_cfg_path):
    """nvflare Tasks take int seconds and treat 0 as wait-forever: int(0.5)
    would silently make a sub-second fl.round_timeout_s infinite. ceil waits
    at least as long as requested."""
    from decent_vla.runtime.nvflare.controller import NVFlareCommunicator

    router = LoopbackRouter(run_cfg_path)
    timeouts = []

    class Ctrl:
        def broadcast_model(self, task_name="train", data=None, targets=None, min_responses=None,
                            timeout=0, wait_time_after_min_received=0, blocking=True):
            timeouts.append(timeout)
            return router(task_name, data, targets, timeout)

    comm = NVFlareCommunicator(Ctrl(), num_clients=2)
    state = _bootstrap_state(run_cfg_path)
    plan = RoundPlan(selected_clients=["c0", "c1"], local_steps=2, round_num=0)
    for t in (0.5, 90.9, None):
        comm.broadcast_state(state, plan)
        comm.collect_updates(expected=["c0"], timeout_s=t)
    assert timeouts == [1, 91, 0]
    assert all(isinstance(t, int) for t in timeouts)


def test_collect_updates_rejects_duplicate_client_ids(run_cfg_path):
    from decent_vla.runtime.nvflare import codec
    from decent_vla.runtime.nvflare.controller import NVFlareCommunicator

    state = _bootstrap_state(run_cfg_path)

    class DuplicateCtrl:
        """Both sites resolve to cid c0 (a naming bug): every reply stamps c0."""

        def broadcast_model(self, task_name="train", data=None, targets=None, min_responses=None,
                            timeout=0, wait_time_after_min_received=0, blocking=True):
            gs, plan = codec.flmodel_to_round(data)
            replies = []
            for site in targets:
                up = client_session.handle_round("c0", str(run_cfg_path), gs, plan)
                reply = codec.update_to_flmodel(up)
                reply.meta["client_name"] = site  # FLARE names the sender
                replies.append(reply)
            return replies

    comm = NVFlareCommunicator(DuplicateCtrl(), num_clients=2)
    plan = RoundPlan(selected_clients=["c0", "c1"], local_steps=2, round_num=0)
    comm.broadcast_state(state, plan)
    with pytest.raises(RuntimeError, match="duplicate client_id"):
        comm.collect_updates(expected=["c0", "c1"])


def test_collect_updates_skips_malformed_replies(run_cfg_path, fake_nvflare):
    """A crashed client yields an FLModel with no usable payload; the
    communicator must skip it (min_clients/retry handles the shortfall)."""
    from nvflare.app_common.abstract.fl_model import FLModel

    from decent_vla.runtime.nvflare.controller import NVFlareCommunicator

    router = LoopbackRouter(run_cfg_path)

    class HalfBrokenCtrl:
        def broadcast_model(self, task_name="train", data=None, targets=None, min_responses=None,
                            timeout=0, wait_time_after_min_received=0, blocking=True):
            good = router(task_name, data, [t for t in targets if t == "c0"], timeout)
            return good + ([FLModel(params=None), None] if "c1" in targets else [])

    comm = NVFlareCommunicator(HalfBrokenCtrl(), num_clients=2)
    plan = RoundPlan(selected_clients=["c0", "c1"], local_steps=2, round_num=0)
    comm.broadcast_state(_bootstrap_state(run_cfg_path), plan)
    updates = comm.collect_updates(expected=["c0", "c1"])
    assert [u.client_id for u in updates] == ["c0"]


def test_collect_updates_waits_past_failed_replies(run_cfg_path, fake_nvflare):
    """FLARE counts a failed reply toward min_responses, so a wait can end before
    the quorum of usable updates is in. The communicator waits again for the
    clients that have not answered, asking only for the shortfall."""
    from decent_vla.runtime.nvflare import codec
    from decent_vla.runtime.nvflare.controller import NVFlareCommunicator

    router = LoopbackRouter(run_cfg_path)
    calls = []

    class Ctrl:
        def broadcast_model(self, task_name="train", data=None, targets=None, min_responses=None,
                            timeout=0, wait_time_after_min_received=0, blocking=True):
            calls.append((list(targets), min_responses, wait_time_after_min_received))
            if "c1" in targets:  # c1 fails fast; FLARE returns at the first answer
                return [codec.error_flmodel("c1", 0, "RuntimeError: boom")]
            return router(task_name, data, targets, timeout)

    from decent_vla.configs.schemas import NVFlareConfig

    comm = NVFlareCommunicator(Ctrl(), num_clients=2, nvf_cfg=NVFlareConfig(wait_time_after_min_received=7))
    plan = RoundPlan(selected_clients=["c0", "c1"], local_steps=2, round_num=0)
    comm.broadcast_state(_bootstrap_state(run_cfg_path), plan)
    updates = comm.collect_updates(expected=["c0", "c1"], min_responses=1)

    # Each wait asks for the shortfall, with the configured grace period.
    assert calls == [(["c0", "c1"], 1, 7), (["c0"], 1, 7)]
    assert [u.client_id for u in updates] == ["c0"]


def test_collect_updates_counts_replies_flare_rejected(run_cfg_path):
    """A reply FLARE rejects (an error code from a dead client script) never
    reaches the results but counts toward min_responses: the controller
    records it, so the communicator waits on for the clients still training."""
    from decent_vla.runtime.nvflare.controller import NVFlareCommunicator

    router = LoopbackRouter(run_cfg_path)
    calls = []

    class Ctrl:
        rejected_replies: list = []

        def broadcast_model(self, task_name="train", data=None, targets=None, min_responses=None,
                            timeout=0, wait_time_after_min_received=0, blocking=True):
            calls.append(list(targets))
            if "c1" in targets:  # c1's script died: FLARE got TASK_ABORTED and dropped it
                self.rejected_replies.append(("c1", "TASK_ABORTED"))
                return []
            return router(task_name, data, targets, timeout)

    comm = NVFlareCommunicator(Ctrl(), num_clients=2)
    plan = RoundPlan(selected_clients=["c0", "c1"], local_steps=2, round_num=0)
    comm.broadcast_state(_bootstrap_state(run_cfg_path), plan)
    updates = comm.collect_updates(expected=["c0", "c1"], min_responses=1)

    assert calls == [["c0", "c1"], ["c0"]]
    assert [u.client_id for u in updates] == ["c0"]


def test_collect_updates_drops_replies_from_another_round_and_repeats(run_cfg_path, fake_nvflare):
    from decent_vla.runtime.nvflare import codec
    from decent_vla.runtime.nvflare.controller import NVFlareCommunicator

    state = _bootstrap_state(run_cfg_path)

    class Ctrl:
        def broadcast_model(self, task_name="train", data=None, targets=None, min_responses=None,
                            timeout=0, wait_time_after_min_received=0, blocking=True):
            gs, plan = codec.flmodel_to_round(data)
            replies = []
            for cid, rnd in (("c0", plan.round_num - 1), ("c1", plan.round_num), ("c1", plan.round_num)):
                up = client_session.handle_round(cid, str(run_cfg_path), gs, plan)
                reply = codec.update_to_flmodel(up, round_num=rnd)
                reply.meta["client_name"] = cid
                replies.append(reply)
            return replies

    comm = NVFlareCommunicator(Ctrl(), num_clients=2)
    plan = RoundPlan(selected_clients=["c0", "c1"], local_steps=2, round_num=1)
    comm.broadcast_state(state, plan)
    updates = comm.collect_updates(expected=["c0", "c1"], min_responses=1)

    # c0's reply answers round 0, and c1's second reply repeats its first.
    assert [u.client_id for u in updates] == ["c1"]
    assert "round" not in updates[0].metrics


def test_load_job_config_resolves_packaged_files(run_cfg_path, tmp_path):
    from decent_vla.runtime.nvflare.client import load_job_config

    raw = yaml.safe_load(run_cfg_path.read_text())
    raw["backend"]["extra"] = {"train_config_path": "pipeline.yaml"}
    raw["data"]["held_out"] = "heldout.json"
    (tmp_path / "pipeline.yaml").write_text("x: 1\n")
    cfg_path = tmp_path / "decent_vla_run.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    cfg = load_job_config(cfg_path)
    assert cfg.backend.extra["train_config_path"] == str(tmp_path / "pipeline.yaml")
    assert cfg.data.held_out == str(tmp_path / "heldout.json")


def test_site_paths_of_a_simulator_job_restore_the_submitting_host(run_cfg_path, tmp_path, monkeypatch):
    import os

    from decent_vla.configs import load_run_config
    from decent_vla.runtime.nvflare.client import apply_site_paths

    cfg = load_run_config(run_cfg_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "submit").mkdir()
    apply_site_paths(cfg, run_dir=tmp_path / "run", workdir=str(tmp_path / "submit"), output_dir="/abs/out")
    assert os.getcwd() == str(tmp_path / "submit")
    assert cfg.fl.output_dir == "/abs/out"


def test_site_paths_of_a_deployed_job_stay_on_the_site(run_cfg_path, tmp_path, monkeypatch):
    """A deployed job carries no host paths: a relative output folder goes under
    the job's run folder; DECENT_VLA_WORKDIR is the site's root for relative
    data paths, and a wrong one is an error, not a silent fallback."""
    import os

    from decent_vla.configs import load_run_config
    from decent_vla.runtime.nvflare.client import apply_site_paths

    cfg = load_run_config(run_cfg_path)
    cfg.fl.output_dir = "outputs/run"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DECENT_VLA_WORKDIR", raising=False)
    apply_site_paths(cfg, run_dir=tmp_path / "job")
    assert cfg.fl.output_dir == str(tmp_path / "job" / "outputs" / "run")
    assert os.getcwd() == str(tmp_path)

    (tmp_path / "site").mkdir()
    monkeypatch.setenv("DECENT_VLA_WORKDIR", str(tmp_path / "site"))
    cfg.fl.output_dir = "/shared/out"  # an absolute output folder is kept
    apply_site_paths(cfg, run_dir=tmp_path / "job")
    assert os.getcwd() == str(tmp_path / "site")
    assert cfg.fl.output_dir == "/shared/out"

    monkeypatch.setenv("DECENT_VLA_WORKDIR", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="DECENT_VLA_WORKDIR"):
        apply_site_paths(cfg, run_dir=tmp_path / "job")


def test_controller_leaves_client_errors_to_the_round_loop(run_cfg_path):
    """FLARE's default aborts the job once one error makes min_responses
    unreachable; the controller turns that off so the round loop decides."""
    ctrl = _controller(run_cfg_path)
    assert ctrl.init_kwargs.get("ignore_result_error") is True


# ---------------------------------------------------------------- controller runs


def test_controller_run_writes_all_artifacts(run_cfg_path):
    from decent_vla.cli.main import _latest_checkpoint
    from decent_vla.configs import load_run_config

    ctrl = _controller(run_cfg_path)
    ctrl.run()

    cfg = load_run_config(run_cfg_path)
    out_dir = Path(cfg.fl.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    assert (out_dir / "partition_manifest.json").exists()
    # Raw blob keeps the server_round_ prefix so eval's _latest_checkpoint
    # skips it; the backend-format round_0002.pt is what eval picks up.
    assert (ckpt_dir / "server_round_final.pt").exists()
    ckpt = ckpt_dir / "round_0002.pt"
    assert ckpt.exists()
    assert _latest_checkpoint(out_dir) == ckpt
    blob = torch.load(ckpt_dir / "server_round_final.pt", map_location="cpu", weights_only=True)
    assert blob["round"] == 1
    assert all(v.dtype == torch.float32 for v in blob["state"].values())


def test_local_and_nvflare_paths_are_equivalent(run_cfg_path):
    from decent_vla.configs import load_run_config
    from decent_vla.runtime.builders import (
        build_backend_factory,
        build_strategy,
        partition_for,
        resolve_device,
    )
    from decent_vla.runtime.local_runner import LocalRunner

    ctrl = _controller(run_cfg_path)
    ctrl.run()
    cfg = load_run_config(run_cfg_path)
    nv_state = torch.load(
        Path(cfg.fl.output_dir) / "checkpoints" / "server_round_final.pt",
        map_location="cpu",
        weights_only=True,
    )["state"]

    device = resolve_device(cfg.device)
    factory, cls = build_backend_factory(cfg, device)
    chunks, _ = partition_for(cfg, cls)
    local = LocalRunner(
        backend_factory=factory,
        strategy=build_strategy(cfg),
        client_ids=["c0", "c1"],
        partition_indices=chunks,
        data_cfg=cfg.data,
        num_rounds=cfg.fl.num_rounds,
        local_steps=cfg.fl.local_steps,
        optim_cfg=cfg.backend.optim,
        device=device,
    ).run()

    assert set(local.final_state) == set(nv_state)
    for k in local.final_state:
        assert torch.allclose(local.final_state[k], nv_state[k], atol=1e-6), k


def test_retry_resends_to_missing_client_only(run_cfg_path):
    from decent_vla.configs import load_run_config

    class DropC1FirstRouter(LoopbackRouter):
        """First send never yields c1's reply; later calls behave normally —
        the resend must go to c1 only, without re-training c0 (which would
        double-step its warm optimizer)."""

        def __init__(self, cfg_path):
            super().__init__(cfg_path)
            self.n_calls = 0

        def __call__(self, task_name, data, targets, timeout):
            self.n_calls += 1
            if self.n_calls == 1:
                targets = [t for t in targets if t != "c1"]
            return super().__call__(task_name, data, targets, timeout)

    cfg = load_run_config(run_cfg_path)
    cfg.fl.num_rounds = 1  # unused: controller reloads from file; num_rounds=2 is fine too
    router = DropC1FirstRouter(run_cfg_path)
    ctrl = _controller(run_cfg_path, router)
    ctrl.run()

    # Round 1: attempt targets both (c1 dropped), retry targets c1 only.
    assert ctrl.sent[0] == ["c0", "c1"]
    assert ctrl.sent[1] == ["c1"]
    # Across 2 rounds each client trained exactly twice — c0 never re-trained
    # by the retry pass.
    assert router.handled == {"c0": 2, "c1": 2}


# ---------------------------------------------------------------- resume


def test_nvflare_auto_resumes_from_latest_server_checkpoint(run_cfg_path, monkeypatch):
    from decent_vla.configs import load_run_config
    from decent_vla.runtime.round_loop import save_server_checkpoint

    monkeypatch.delenv("DECENT_VLA_NO_RESUME", raising=False)
    cfg = load_run_config(run_cfg_path)
    state = _bootstrap_state(run_cfg_path)
    ckpt_dir = Path(cfg.fl.output_dir) / "checkpoints"
    save_server_checkpoint(ckpt_dir, 0, state)  # server_round_0001.pt, round=0
    # Decoy: the final blob must be excluded from the resume scan.
    torch.save({"state": state, "round": 99}, ckpt_dir / "server_round_final.pt")

    router = LoopbackRouter(run_cfg_path)
    ctrl = _controller(run_cfg_path, router)
    ctrl.run()

    # num_rounds=2 is the TOTAL; round 0 is done, so exactly 1 round runs.
    assert ctrl.sent == [["c0", "c1"]]
    assert router.seen_rounds == [1, 1]


def test_nvflare_resume_opt_out_env_var(run_cfg_path, monkeypatch):
    from decent_vla.configs import load_run_config
    from decent_vla.runtime.round_loop import save_server_checkpoint

    monkeypatch.setenv("DECENT_VLA_NO_RESUME", "1")
    cfg = load_run_config(run_cfg_path)
    save_server_checkpoint(
        Path(cfg.fl.output_dir) / "checkpoints", 0, _bootstrap_state(run_cfg_path)
    )

    router = LoopbackRouter(run_cfg_path)
    ctrl = _controller(run_cfg_path, router)
    ctrl.run()

    # Fresh run: checkpoint ignored, all 2 rounds run starting at 0.
    assert len(ctrl.sent) == 2
    assert router.seen_rounds[:2] == [0, 0]


def test_nvflare_resume_nothing_to_do(run_cfg_path, monkeypatch):
    from decent_vla.configs import load_run_config
    from decent_vla.runtime.round_loop import save_server_checkpoint

    monkeypatch.delenv("DECENT_VLA_NO_RESUME", raising=False)
    cfg = load_run_config(run_cfg_path)
    state = _bootstrap_state(run_cfg_path)
    # round index 1 done => start_round 2 == num_rounds: nothing left.
    save_server_checkpoint(Path(cfg.fl.output_dir) / "checkpoints", 1, state)

    ctrl = _controller(run_cfg_path)
    ctrl.run()

    assert ctrl.sent == []  # no training tasks at all
    final = torch.load(
        Path(cfg.fl.output_dir) / "checkpoints" / "server_round_final.pt",
        map_location="cpu",
        weights_only=True,
    )
    for k in state:
        assert torch.allclose(final["state"][k].float(), state[k].float()), k
