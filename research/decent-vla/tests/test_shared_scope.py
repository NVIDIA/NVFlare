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

"""The two orthogonal scope axes of a federated run.

Axis 1 — local training scope (`backend.trainable_params`): which params are
trainable at all (requires_grad; what the optimizer sees). Generic pattern
form of the per-policy freeze flags.

Axis 2 — federated scope (`backend.shared_params`): which TRAINABLE params
are federated ("shared": delta, aggregation, broadcast); the complement is
"personal" and stays client-local across rounds (personalization).

Either axis unset (None) keeps historical behavior: axis 1 → the policy's
native flags decide; axis 2 → every trainable param is shared.
"""
from __future__ import annotations

import pytest
import torch

from decent_vla.adapters._mock.backend import MockBackend
from decent_vla.configs.schemas import BackendConfig, DataConfig, OptimConfig
from decent_vla.core.scope import apply_trainable_patterns, resolve_shared_names
from decent_vla.runtime.local_runner import LocalRunner
from decent_vla.runtime.partitioning import iid_partition
from decent_vla.strategies import FedAvg

ALL_KEYS = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}


# ---------------------------------------------------------------- core matcher

def test_resolve_shared_names_none_shares_everything():
    names = ["fc1.weight", "fc1.bias", "fc2.weight"]
    assert resolve_shared_names(names, None) == set(names)


def test_resolve_shared_names_filters_by_fnmatch():
    assert resolve_shared_names(sorted(ALL_KEYS), ["fc2.*"]) == {
        "fc2.weight",
        "fc2.bias",
    }


def test_resolve_shared_names_union_of_patterns():
    names = ["a.w", "b.w", "c.w"]
    assert resolve_shared_names(names, ["a.*", "c.*"]) == {"a.w", "c.w"}


def test_resolve_shared_names_no_match_raises():
    # A typo'd pattern silently sharing nothing would mean "FL that never
    # federates" — fail loudly instead.
    with pytest.raises(ValueError, match="match"):
        resolve_shared_names(["fc1.weight"], ["typo.*"])


def test_resolve_shared_names_empty_list_raises():
    # `shared_params: []` is not "share everything" and not "share nothing" —
    # it's a config mistake. It must not slip past the no-match guard just
    # because the empty list is falsy.
    with pytest.raises(ValueError, match="empty"):
        resolve_shared_names(["fc1.weight"], [])


def test_apply_trainable_patterns_empty_list_raises():
    # `trainable_params: []` would silently freeze the ENTIRE model.
    m = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match="empty"):
        apply_trainable_patterns(m, [])
    assert m.weight.requires_grad  # untouched on error


# ---------------------------------------------------------- mock backend scope

def _backend(shared_params):
    cfg = BackendConfig(
        name="mock",
        shared_params=shared_params,
        extra={"model_seed": 0, "data_seed": 42},
    )
    return MockBackend.from_config(cfg, torch.device("cpu"))


def test_mock_shared_scope_is_pattern_subset():
    b = _backend(["fc2.*"])
    assert set(b.get_state("shared")) == {"fc2.weight", "fc2.bias"}


def test_mock_personal_scope_is_complement():
    b = _backend(["fc2.*"])
    assert set(b.get_state("personal")) == {"fc1.weight", "fc1.bias"}


def test_mock_all_scope_unaffected_by_patterns():
    b = _backend(["fc2.*"])
    assert set(b.get_state("all")) == ALL_KEYS


def test_mock_default_none_keeps_historical_behavior():
    b = _backend(None)
    assert set(b.get_state("shared")) == ALL_KEYS
    assert b.get_state("personal") == {}


def test_mock_set_state_shared_leaves_personal_untouched():
    b = _backend(["fc2.*"])
    fc1_before = b.get_state("personal")["fc1.weight"].clone()
    zeros = {k: torch.zeros_like(v) for k, v in b.get_state("shared").items()}
    b.set_state(zeros, scope="shared")
    assert torch.equal(b.get_state("personal")["fc1.weight"], fc1_before)
    assert torch.count_nonzero(b.get_state("shared")["fc2.weight"]) == 0


def test_mock_unmatched_patterns_raise():
    b = _backend(["no_such_layer.*"])
    with pytest.raises(ValueError, match="match"):
        b.get_state("shared")


# ---------------------------------------------- axis 1: local training scope

def test_apply_trainable_patterns_freezes_complement():
    m = torch.nn.Sequential()
    m.add_module("fc1", torch.nn.Linear(4, 4))
    m.add_module("fc2", torch.nn.Linear(4, 2))
    apply_trainable_patterns(m, ["fc2.*"])
    assert not m.fc1.weight.requires_grad and not m.fc1.bias.requires_grad
    assert m.fc2.weight.requires_grad and m.fc2.bias.requires_grad


def test_apply_trainable_patterns_none_is_noop():
    m = torch.nn.Linear(4, 4)
    m.weight.requires_grad_(False)
    apply_trainable_patterns(m, None)
    assert not m.weight.requires_grad  # untouched
    assert m.bias.requires_grad


def test_apply_trainable_patterns_no_match_raises():
    m = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match="match"):
        apply_trainable_patterns(m, ["typo.*"])


def test_mock_trainable_params_narrow_optimizer_and_all_scope():
    b = _backend_axes(trainable=["fc2.*"], shared=None)
    opt, _ = b.build_optimizer(OptimConfig(lr=1e-3))
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    n_fc2 = sum(p.numel() for p in b.model.fc2.parameters())
    assert n_opt == n_fc2
    assert set(b.get_state("all")) == {"fc2.weight", "fc2.bias"}
    # With axis 2 unset, everything trainable is shared.
    assert set(b.get_state("shared")) == {"fc2.weight", "fc2.bias"}


def test_axes_compose_shared_is_subset_of_trainable():
    b = _backend_axes(trainable=["fc*"], shared=["fc2.*"])
    assert set(b.get_state("all")) == ALL_KEYS
    assert set(b.get_state("shared")) == {"fc2.weight", "fc2.bias"}
    assert set(b.get_state("personal")) == {"fc1.weight", "fc1.bias"}


def _backend_axes(trainable, shared):
    cfg = BackendConfig(
        name="mock",
        trainable_params=trainable,
        shared_params=shared,
        extra={"model_seed": 0, "data_seed": 42},
    )
    return MockBackend.from_config(cfg, torch.device("cpu"))


# --------------------------------------------------- strategy compatibility

def test_fedprox_pulls_only_shared_params():
    """FedProx's proximal term must act on the shared subset alone: personal
    params drift freely (partial-personalization semantics)."""
    from decent_vla.core.strategy import ClientContext
    from decent_vla.strategies import FedProx

    b = _backend_axes(trainable=None, shared=["fc2.*"])
    global_state = b.get_state("shared")
    opt, _ = b.build_optimizer(OptimConfig(lr=1e-3))
    strategy = FedProx(mu=1.0)
    ctx = ClientContext(
        client_id="c0",
        round_num=0,
        step_in_round=0,
        backend=b,
        model=b.model,
        optimizer=opt,
        scheduler=None,
        global_state=global_state,
        broadcast={},
        persistent_state={},
    )
    strategy.on_round_start(ctx)
    # Perturb BOTH halves away from the snapshot, then check the prox term
    # only counts the shared (fc2) elements.
    with torch.no_grad():
        for p in b.model.parameters():
            p.add_(1.0)
    loss = {"loss": torch.zeros((), requires_grad=True)}
    out = strategy.modify_loss(loss, ctx)
    n_fc2 = sum(p.numel() for p in b.model.fc2.parameters())
    # mean-normalized: 0.5 * mu * mean((theta-theta_g)^2) with every diff == 1
    assert torch.isclose(out["prox_term"], torch.tensor(0.5)), out["prox_term"]
    out["loss"].backward()
    assert b.model.fc1.weight.grad is None or torch.all(b.model.fc1.weight.grad == 0)
    assert torch.any(b.model.fc2.weight.grad != 0)
    assert n_fc2 > 0


# ------------------------------------------------- resume / ckpt round-trip

def test_client_checkpoint_roundtrip_preserves_personal_params(tmp_path):
    """save_client_state stores 'all' (shared + personal); a rebuilt client
    restored from it gets its personal params back exactly."""
    from decent_vla.runtime.client_checkpoint import save_client_state
    from decent_vla.runtime.client_state import build_client_state

    cfg = BackendConfig(
        name="mock",
        shared_params=["fc2.*"],
        extra={"model_seed": 0, "data_seed": 42},
    )
    device = torch.device("cpu")

    def factory():
        return MockBackend.from_config(cfg, device)

    data_cfg = DataConfig(name="mock", batch_size=8, num_clients=1, seed=42)
    st = build_client_state(
        client_id="c0",
        backend_factory=factory,
        data_cfg=data_cfg,
        optim_cfg=OptimConfig(lr=1e-2),
        partition_indices=None,
        strategy=FedAvg(),
        start_round=0,
        local_steps=1,
    )
    # Drift the personal half so restore has something distinctive to recover.
    with torch.no_grad():
        st.backend.model.fc1.weight.add_(3.0)
    want = st.backend.get_state("personal")["fc1.weight"].clone()
    save_client_state(tmp_path, st, round_idx=0)

    st2 = build_client_state(
        client_id="c0",
        backend_factory=factory,
        data_cfg=data_cfg,
        optim_cfg=OptimConfig(lr=1e-2),
        partition_indices=None,
        strategy=FedAvg(),
        start_round=1,
        local_steps=1,
        client_ckpt_dir=tmp_path,
    )
    assert torch.equal(st2.backend.get_state("personal")["fc1.weight"], want)


# -------------------------------------------------------------- resume parity

def test_no_resume_env_parsing_is_shared_and_consistent():
    """DECENT_VLA_NO_RESUME must mean the same thing on the server and the
    client: '' and '0' = resume normally; anything else = skip restore.
    (The client used plain truthiness, so '0' skipped restore while the
    server resumed — a silent personalization reset.)"""
    import os
    from unittest import mock

    from decent_vla.runtime.client_checkpoint import resume_disabled

    for val, expect in (("", False), ("0", False), ("1", True), ("yes", True)):
        with mock.patch.dict(os.environ, {"DECENT_VLA_NO_RESUME": val}):
            assert resume_disabled() is expect, (val, expect)
    with mock.patch.dict(os.environ, {}, clear=True):
        assert resume_disabled() is False


# ------------------------------------------------------------ config warning

def test_loader_warns_shared_params_without_persist(tmp_path, caplog):
    import logging
    import textwrap

    from decent_vla.configs import load_run_config

    p = tmp_path / "cfg.yaml"
    p.write_text(
        textwrap.dedent(
            """
            name: t
            backend:
              name: mock
              shared_params: ["fc2.*"]
            data:
              name: mock
            fl:
              rounds: 1
            """
        )
    )
    with caplog.at_level(logging.WARNING):
        load_run_config(str(p))
    assert any("persist_client_state" in r.message for r in caplog.records)


# ------------------------------------------------------------------ yaml loader

def test_loader_parses_backend_shared_params(tmp_path):
    import textwrap

    from decent_vla.configs import load_run_config

    p = tmp_path / "cfg.yaml"
    p.write_text(
        textwrap.dedent(
            """
            name: t
            backend:
              name: mock
              shared_params: ["fc2.*"]
            data:
              name: mock
            fl:
              rounds: 1
            """
        )
    )
    cfg = load_run_config(str(p))
    assert cfg.backend.shared_params == ["fc2.*"]


def test_loader_parses_backend_trainable_params(tmp_path):
    import textwrap

    from decent_vla.configs import load_run_config

    p = tmp_path / "cfg.yaml"
    p.write_text(
        textwrap.dedent(
            """
            name: t
            backend:
              name: mock
              trainable_params: ["fc*"]
            data:
              name: mock
            fl:
              rounds: 1
            """
        )
    )
    cfg = load_run_config(str(p))
    assert cfg.backend.trainable_params == ["fc*"]
    assert cfg.backend.shared_params is None


def test_loader_defaults_shared_params_to_none(tmp_path):
    import textwrap

    from decent_vla.configs import load_run_config

    p = tmp_path / "cfg.yaml"
    p.write_text(
        textwrap.dedent(
            """
            name: t
            backend:
              name: mock
            data:
              name: mock
            fl:
              rounds: 1
            """
        )
    )
    assert load_run_config(str(p)).backend.shared_params is None


# ------------------------------------------------------------- e2e LocalRunner

def test_local_runner_personalization_e2e(tmp_path):
    """2 clients x 2 rounds, sharing only fc2: the server state carries fc2
    alone, while each client's fc1 evolves locally and diverges."""
    cfg = BackendConfig(
        name="mock",
        shared_params=["fc2.*"],
        extra={"model_seed": 0, "data_seed": 42},
    )
    device = torch.device("cpu")

    def factory():
        return MockBackend.from_config(cfg, device)

    data_cfg = DataConfig(name="mock", batch_size=8, num_clients=2, seed=42)
    indices = iid_partition(MockBackend.SYNTHETIC_DATASET_SIZE, 2, seed=0)
    runner = LocalRunner(
        backend_factory=factory,
        strategy=FedAvg(),
        client_ids=["c0", "c1"],
        partition_indices=indices,
        data_cfg=data_cfg,
        num_rounds=2,
        local_steps=5,
        optim_cfg=OptimConfig(lr=1e-2),
        device=device,
        persist_client_state=True,
        client_ckpt_dir=str(tmp_path),
        save_every_n_rounds=1,
    )
    outcome = runner.run()
    assert outcome.rounds_completed == 2

    # The federated (server) state is exactly the shared subset.
    assert set(outcome.final_state) == {"fc2.weight", "fc2.bias"}

    # Each client persisted its FULL state; the personal fc1 halves trained on
    # different partitions and must have diverged from each other.
    from decent_vla.runtime.client_checkpoint import latest_client_checkpoint

    blobs = {}
    for cid in ("c0", "c1"):
        p = latest_client_checkpoint(tmp_path, cid)
        assert p is not None
        blobs[cid] = torch.load(p, map_location="cpu", weights_only=False)["weights"]
    assert not torch.equal(blobs["c0"]["fc1.weight"], blobs["c1"]["fc1.weight"])


def test_partial_load_warning_flags_an_incomplete_checkpoint():
    """A checkpoint that covers only part of the trainable set must produce a
    message naming how much it left behind.

    A partial-federation server checkpoint holds only the shared subset, so
    materializing it as a standalone policy leaves every personal parameter at
    its pretrained value. A backbone-share server checkpoint exported that way
    scored 0.0% on all four LIBERO suites over 400 episodes with nothing in the
    log to explain it.
    """
    from decent_vla.core.scope import partial_load_warning

    msg = partial_load_warning(
        trainable=["backbone.w", "backbone.b", "head.w", "head.b"],
        loaded=["backbone.w", "backbone.b"],
        source="server_round_final.pt",
    )
    assert msg is not None
    assert "2 of 4" in msg
    assert "server_round_final.pt" in msg
    assert "head.b" in msg and "head.w" in msg


def test_partial_load_warning_is_silent_on_full_coverage():
    """It must not cry wolf on the ordinary complete load."""
    from decent_vla.core.scope import partial_load_warning

    names = ["backbone.w", "head.w"]
    assert partial_load_warning(trainable=names, loaded=names, source="x.pt") is None
    # A checkpoint carrying extra keys still covers the trainable set.
    assert partial_load_warning(
        trainable=names, loaded=[*names, "buffer.running_mean"], source="x.pt"
    ) is None


def test_trainable_state_keys_is_name_only_and_matches_get_state():
    """The key set must be derivable WITHOUT materializing tensors.

    `get_state("all")` detaches and CLONES every trainable tensor — ~12.6 GB at
    GR00T's 3.14B. Anything that only needs names (the partial-load coverage
    check) must not pay that, and must still agree with get_state exactly, or
    the warning would fire on a complete load.
    """
    import torch.nn as nn

    from decent_vla.core.scope import trainable_state_keys

    model = nn.Module()
    model.backbone = nn.Linear(2, 2)
    model.head = nn.Linear(2, 2)
    model.register_buffer("running", torch.zeros(2))
    for p in model.backbone.parameters():
        p.requires_grad_(False)

    keys = trainable_state_keys(model)
    assert keys == {"head.weight", "head.bias"}, keys
    assert "running" not in keys, "buffers are not trainable params"

    # The reference filter the adapters used inline, reproduced here: the two
    # must agree, else the coverage warning misfires.
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    expected = {
        k for k in model.state_dict()
        if k in names or any(k.startswith(n + ".") for n in names)
    }
    assert keys == expected
