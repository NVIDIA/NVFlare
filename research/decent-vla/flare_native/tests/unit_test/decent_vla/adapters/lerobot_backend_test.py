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

"""Tests for `decent_vla.adapters.lerobot.backend.LeRobotBackend`.

The lerobot package is **not** installed in this test environment. We
inject a minimal fake `lerobot.*` namespace into `sys.modules` so the
adapter's lazy imports resolve without pulling in the real upstream.

This catches protocol mistakes, wiring bugs, and LoRA path errors
without needing a GPU or the multi-GB lerobot install. Real-policy
integration is validated on the cluster where lerobot + GPU exist.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Fake lerobot — injected into sys.modules before LeRobotBackend.from_config is called.
# ---------------------------------------------------------------------------


class _FakePolicy(nn.Module):
    """A tiny stand-in for a lerobot PreTrainedPolicy."""

    def __init__(self, action_dim: int = 7):
        super().__init__()
        self.config = SimpleNamespace(
            n_action_steps=1,
            action_horizon=1,
            input_features={"observation": "visual"},
            output_features={"action": "action"},
            normalization_mapping={},
        )
        self.backbone = nn.Linear(16, 8)
        self.head = nn.Linear(8, action_dim)

    def forward(self, batch):
        x = batch["observation"]
        z = torch.relu(self.backbone(x))
        pred = self.head(z)
        target = batch["action"]
        loss = ((pred - target) ** 2).mean()
        return loss, {"l1_loss": (pred - target).abs().mean()}

    def select_action(self, observation):
        x = observation["observation"] if "observation" in observation else next(iter(observation.values()))
        z = torch.relu(self.backbone(x))
        return self.head(z)


class _FakeDataset:
    """Fake LeRobotDataset. Mirrors lerobot 0.4.1: the hf_dataset is ALWAYS the
    full dataset (the loader does NOT filter by episodes) and `meta.episodes`
    exposes per-episode global frame boundaries (8 episodes x `ep_len` frames).
    Partition restriction therefore must happen via a sampler, not the loader."""

    def __init__(self, n: int = 64, ep_len: int = 8):
        self.n = n
        self.num_frames = n
        self.num_episodes = max(1, n // ep_len)
        froms = list(range(0, n, ep_len))
        tos = list(range(ep_len, n + ep_len, ep_len))
        self.meta = SimpleNamespace(
            episodes={"dataset_from_index": froms, "dataset_to_index": tos},
            total_episodes=self.num_episodes,
            total_frames=n,
        )

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "observation": torch.randn(16),
            "action": torch.randn(7),
        }


class _FakeDatasetMeta:
    def __init__(self):
        self.total_frames = 64
        self.total_episodes = 8
        self.episodes = {i: {"tasks": [f"task-{i // 2}"]} for i in range(8)}


def _install_fake_lerobot(monkeypatch):
    """Inject a fake `lerobot` namespace into sys.modules. Returns the SimpleNamespace
    with handles so tests can introspect what was called."""
    state: dict[str, Any] = {"make_policy_calls": 0, "make_dataset_calls": 0}

    # lerobot.configs.train.TrainPipelineConfig
    train_mod = types.ModuleType("lerobot.configs.train")

    class _TrainPipelineConfig:
        def __init__(self):
            self.policy = SimpleNamespace(type="fake", pretrained_path=None)
            self.dataset = SimpleNamespace(repo_id="fake/repo", root=None, episodes=None)
            self.rename_map = None

    train_mod.TrainPipelineConfig = _TrainPipelineConfig
    sys.modules["lerobot"] = types.ModuleType("lerobot")
    sys.modules["lerobot.configs"] = types.ModuleType("lerobot.configs")
    sys.modules["lerobot.configs.train"] = train_mod

    # lerobot.configs.policies.PreTrainedConfig — records from_pretrained
    # calls so tests can assert the --policy.path replication path runs.
    state["from_pretrained_calls"] = 0
    cp_mod = types.ModuleType("lerobot.configs.policies")

    class _FakePreTrainedConfig:
        def __init__(self):
            self.type = "fake-pretrained"
            self.pretrained_path = None
            self.input_features = {"observation.images.camera1": "visual"}
            self.output_features = {"action": "action"}
            self.normalization_mapping = {}

        @classmethod
        def from_pretrained(cls, path, cli_overrides=None, **kwargs):
            state["from_pretrained_calls"] += 1
            state["from_pretrained_path"] = path
            state["from_pretrained_overrides"] = list(cli_overrides or [])
            return cls()

    cp_mod.PreTrainedConfig = _FakePreTrainedConfig
    sys.modules["lerobot.configs.policies"] = cp_mod

    # lerobot.datasets.factory.make_dataset — records whether the adapter set
    # cfg.dataset.episodes (the broken-on-0.4.1 filter the partition must NOT
    # rely on) so tests can assert it stays unset.
    df_mod = types.ModuleType("lerobot.datasets.factory")

    def _make_dataset(cfg):
        state["make_dataset_calls"] += 1
        state["make_dataset_episodes"] = getattr(cfg.dataset, "episodes", None)
        return _FakeDataset()

    df_mod.make_dataset = _make_dataset
    sys.modules["lerobot.datasets"] = types.ModuleType("lerobot.datasets")
    sys.modules["lerobot.datasets.factory"] = df_mod

    # lerobot.datasets.sampler.EpisodeAwareSampler — faithful reimplementation
    # of the real index-building so a test can verify the adapter restricts to
    # exactly the partition's episodes. Records its args for assertions.
    smp_mod = types.ModuleType("lerobot.datasets.sampler")

    class _EpisodeAwareSampler:
        def __init__(
            self,
            dataset_from_indices,
            dataset_to_indices,
            episode_indices_to_use=None,
            drop_n_first_frames=0,
            drop_n_last_frames=0,
            shuffle=False,
        ):
            state["sampler_episode_indices"] = None if episode_indices_to_use is None else list(episode_indices_to_use)
            state["sampler_drop_n_last_frames"] = drop_n_last_frames
            indices = []
            for ep_idx, (s, e) in enumerate(zip(dataset_from_indices, dataset_to_indices, strict=True)):
                if episode_indices_to_use is None or ep_idx in episode_indices_to_use:
                    indices.extend(range(s + drop_n_first_frames, e - drop_n_last_frames))
            self.indices = indices
            self.shuffle = shuffle
            state["sampler_len"] = len(indices)

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    smp_mod.EpisodeAwareSampler = _EpisodeAwareSampler
    sys.modules["lerobot.datasets.sampler"] = smp_mod

    # lerobot.datasets.utils.cycle
    du_mod = types.ModuleType("lerobot.datasets.utils")

    def _cycle(loader):
        while True:
            yield from loader

    du_mod.cycle = _cycle
    sys.modules["lerobot.datasets.utils"] = du_mod

    # lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata
    dl_mod = types.ModuleType("lerobot.datasets.lerobot_dataset")

    class _MetaCls:
        def __init__(self, repo_id, root=None):
            self.repo_id = repo_id
            self.total_episodes = 8
            self.episodes = {i: {"tasks": [f"task-{i // 2}"]} for i in range(8)}
            self.total_frames = 64
            self.stats = {"observation": {"mean": torch.zeros(16), "std": torch.ones(16)}}

    dl_mod.LeRobotDatasetMetadata = _MetaCls
    sys.modules["lerobot.datasets.lerobot_dataset"] = dl_mod

    # lerobot.policies.factory.make_policy
    pf_mod = types.ModuleType("lerobot.policies.factory")

    def _make_policy(cfg, ds_meta, rename_map=None):
        state["make_policy_calls"] += 1
        state["make_policy_device"] = getattr(cfg, "device", None)
        return _FakePolicy()

    pf_mod.make_policy = _make_policy

    # lerobot.policies.factory.make_pre_post_processors — identity pipelines
    # that record invocations so tests can assert the adapter runs the
    # preprocessor on every batch before policy.forward.
    state["make_pre_post_calls"] = 0
    state["preprocessor_calls"] = 0
    state["postprocessor_calls"] = 0

    def _make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs):
        state["make_pre_post_calls"] += 1
        state["make_pre_post_kwargs"] = kwargs

        def _identity_pre(batch):
            state["preprocessor_calls"] += 1
            return batch

        def _identity_post(action):
            state["postprocessor_calls"] += 1
            return action

        return _identity_pre, _identity_post

    pf_mod.make_pre_post_processors = _make_pre_post_processors
    sys.modules["lerobot.policies"] = types.ModuleType("lerobot.policies")
    sys.modules["lerobot.policies.factory"] = pf_mod

    # draccus.parse for _load_train_pipeline_config
    draccus_mod = types.ModuleType("draccus")

    def _parse(cls, config_path=None, args=None):
        # Mirror real draccus: second arg is a PATH it opens itself, and
        # args=None means "consume sys.argv" (fatal inside launcher-managed
        # processes). The adapter must pass an existing path and args=[].
        import os

        assert args == [], "adapter must call draccus.parse with args=[]"
        assert config_path and os.path.exists(str(config_path)), "adapter must pass the config PATH, not file contents"
        return cls()

    draccus_mod.parse = _parse
    sys.modules["draccus"] = draccus_mod

    yield state

    # Cleanup so other tests don't see the fake.
    for k in [
        "lerobot",
        "lerobot.configs",
        "lerobot.configs.train",
        "lerobot.configs.policies",
        "lerobot.datasets",
        "lerobot.datasets.factory",
        "lerobot.datasets.utils",
        "lerobot.datasets.lerobot_dataset",
        "lerobot.policies",
        "lerobot.policies.factory",
        "draccus",
    ]:
        sys.modules.pop(k, None)


@pytest.fixture
def fake_lerobot(monkeypatch):
    yield from _install_fake_lerobot(monkeypatch)


# ---------------------------------------------------------------------------
# Module import & entry point
# ---------------------------------------------------------------------------


def test_module_imports_without_lerobot_installed():
    """Importing the adapter must NOT pull in lerobot.

    Critical because `decent-vla list backends` discovers all backends
    eagerly via entry points; if importing the adapter required lerobot,
    every user without [lerobot] installed would see a discovery error.
    """
    # Ensure no fake lerobot leaks into this test.
    for k in list(sys.modules):
        if k.startswith("lerobot") or k == "draccus":
            sys.modules.pop(k, None)
    from decent_vla.adapters.lerobot.backend import LeRobotBackend

    assert LeRobotBackend.__name__ == "LeRobotBackend"


def test_entry_point_registered():
    """`lerobot` must show up in the backend registry."""
    from decent_vla.core.registry import discover

    reg = discover()
    assert "lerobot" in reg.backends


# ---------------------------------------------------------------------------
# from_config + protocol surface
# ---------------------------------------------------------------------------


def test_from_config_requires_train_config(fake_lerobot):
    """Helpful error if user forgets to set extra.train_config_path."""
    from decent_vla.adapters.lerobot.backend import LeRobotBackend
    from decent_vla.configs.schemas import BackendConfig

    cfg = BackendConfig(name="lerobot")
    with pytest.raises(ValueError, match="train_config_path"):
        LeRobotBackend.from_config(cfg, torch.device("cpu"))


def test_from_config_builds_policy(tmp_path, fake_lerobot):
    """from_config goes through make_policy and the result satisfies Backend."""
    from decent_vla.adapters.lerobot.backend import LeRobotBackend
    from decent_vla.configs.schemas import BackendConfig
    from decent_vla.core.backend import Backend

    yaml_path = tmp_path / "fake_lerobot.yaml"
    yaml_path.write_text("# placeholder; draccus.parse is faked\n")
    cfg = BackendConfig(name="lerobot", extra={"train_config_path": str(yaml_path)})
    backend = LeRobotBackend.from_config(cfg, torch.device("cpu"))

    assert isinstance(backend, Backend)
    assert backend.name == "lerobot:fake"
    assert backend.supports_personal_params is False
    assert fake_lerobot["make_policy_calls"] == 1
    assert fake_lerobot["make_policy_device"] == "cpu"  # built where it was asked, not on the config's GPU


def test_compute_loss_returns_loss_dict(tmp_path, fake_lerobot):
    from decent_vla.adapters.lerobot.backend import LeRobotBackend
    from decent_vla.configs.schemas import BackendConfig

    yaml_path = tmp_path / "fake_lerobot.yaml"
    yaml_path.write_text("# placeholder\n")
    cfg = BackendConfig(name="lerobot", extra={"train_config_path": str(yaml_path)})
    backend = LeRobotBackend.from_config(cfg, torch.device("cpu"))

    batch = {"observation": torch.randn(4, 16), "action": torch.randn(4, 7)}
    out = backend.compute_loss(batch)
    assert "loss" in out
    assert out["loss"].requires_grad
    assert out["loss"].ndim == 0


def test_compute_loss_runs_preprocessor_once_per_call(tmp_path, fake_lerobot):
    """Every compute_loss must run the lerobot preprocessor pipeline exactly
    once before policy.forward — skipping it crashes real policies with
    KeyError on tokenized/renamed keys (e.g. 'observation.language.tokens')."""
    from decent_vla.adapters.lerobot.backend import LeRobotBackend
    from decent_vla.configs.schemas import BackendConfig

    yaml_path = tmp_path / "fake_lerobot.yaml"
    yaml_path.write_text("# placeholder\n")
    cfg = BackendConfig(name="lerobot", extra={"train_config_path": str(yaml_path)})
    backend = LeRobotBackend.from_config(cfg, torch.device("cpu"))

    assert fake_lerobot["make_pre_post_calls"] == 1
    assert fake_lerobot["preprocessor_calls"] == 0

    batch = {"observation": torch.randn(4, 16), "action": torch.randn(4, 7)}
    backend.compute_loss(batch)
    assert fake_lerobot["preprocessor_calls"] == 1
    backend.compute_loss(batch)
    assert fake_lerobot["preprocessor_calls"] == 2


def test_build_policy_wires_pre_post_processors(tmp_path, fake_lerobot):
    """The inference wrapper must preprocess observations and postprocess
    actions with the same pipelines used in training (lerobot_eval parity)."""
    from decent_vla.adapters.lerobot.backend import LeRobotBackend
    from decent_vla.configs.schemas import BackendConfig

    yaml_path = tmp_path / "fake_lerobot.yaml"
    yaml_path.write_text("# placeholder\n")
    cfg = BackendConfig(name="lerobot", extra={"train_config_path": str(yaml_path)})
    backend = LeRobotBackend.from_config(cfg, torch.device("cpu"))

    policy = backend.build_policy()
    policy.select_action({"observation": torch.randn(1, 16)})
    assert fake_lerobot["preprocessor_calls"] == 1
    assert fake_lerobot["postprocessor_calls"] == 1


def test_build_dataloader_restricts_frames_to_partition_episodes(tmp_path, fake_lerobot):
    """A client given episodes [0,1,2] must iterate ONLY those episodes' frames,
    not the full dataset. lerobot 0.4.1's loader ignores cfg.dataset.episodes
    (it loads all parquet), so the restriction MUST go through an
    EpisodeAwareSampler keyed on the partition — and cfg.dataset.episodes must
    NOT be relied on. Regression guard for the void-partition bug where every
    FL client silently trained on the full dataset."""
    from decent_vla.configs.schemas import DataConfig

    backend = _build_backend(tmp_path)
    backend.build_dataloader(DataConfig(repo_id="fake/repo", batch_size=4), [0, 1, 2])

    # sampler restricted to exactly the requested episodes
    assert fake_lerobot["sampler_episode_indices"] == [0, 1, 2]
    # 3 episodes x 8 frames = 24 sampled frames, NOT the full 64
    assert fake_lerobot["sampler_len"] == 24
    # the broken filter must not be relied upon
    assert fake_lerobot["make_dataset_episodes"] is None


def test_build_dataloader_full_dataset_when_no_partition(tmp_path, fake_lerobot):
    """partition_indices=None means centralized/full data: no episode-restricting
    sampler is built (the whole dataset is iterated)."""
    from decent_vla.configs.schemas import DataConfig

    fake_lerobot["sampler_episode_indices"] = "UNSET"
    backend = _build_backend(tmp_path)
    backend.build_dataloader(DataConfig(repo_id="fake/repo", batch_size=4), None)
    assert fake_lerobot["sampler_episode_indices"] == "UNSET"  # sampler never built


def test_build_dataloader_rejects_void_partition(tmp_path, fake_lerobot):
    """If the partition resolves to zero sampled frames, fail LOUDLY at startup
    rather than silently train on nothing (or, pre-fix, on everything)."""
    from decent_vla.configs.schemas import DataConfig

    backend = _build_backend(tmp_path)
    with pytest.raises(ValueError, match="0 frames|empty"):
        # episode index 999 doesn't exist -> sampler covers no frames
        backend.build_dataloader(DataConfig(repo_id="fake/repo", batch_size=4), [999])


def test_pretrained_path_loads_policy_config_from_checkpoint(tmp_path, fake_lerobot):
    """`policy.pretrained_path` in the YAML must replicate `--policy.path`:
    the policy config is loaded from the checkpoint (its input/output
    features and normalization), with remaining YAML policy fields applied
    as draccus CLI overrides. Without this, make_policy fills
    `input_features` from the *dataset* feature names, which inverts the
    rename_map direction and crashes `prepare_images` on real policies."""
    from pathlib import Path

    from decent_vla.adapters.lerobot.backend import LeRobotBackend
    from decent_vla.configs.schemas import BackendConfig

    yaml_path = tmp_path / "fake_pretrained.yaml"
    yaml_path.write_text(
        "policy:\n"
        "  type: smolvla\n"
        "  pretrained_path: fake/smolvla_base\n"
        "  push_to_hub: false\n"
        "dataset:\n"
        "  repo_id: fake/repo\n"
    )

    # The fake draccus.parse ignores file contents; make the parsed config
    # carry the YAML's pretrained_path so the resolution path triggers.
    train_mod = sys.modules["lerobot.configs.train"]
    base_cls = train_mod.TrainPipelineConfig

    class _PretrainedTPC(base_cls):
        def __init__(self):
            super().__init__()
            self.policy = SimpleNamespace(type="smolvla", pretrained_path="fake/smolvla_base")

    train_mod.TrainPipelineConfig = _PretrainedTPC

    cfg = BackendConfig(name="lerobot", extra={"train_config_path": str(yaml_path)})
    backend = LeRobotBackend.from_config(cfg, torch.device("cpu"))

    assert fake_lerobot["from_pretrained_calls"] == 1
    assert fake_lerobot["from_pretrained_path"] == "fake/smolvla_base"
    # YAML policy fields (minus type/pretrained_path) become CLI overrides.
    # Booleans must be lowercase: real draccus rejects 'False' as a bool.
    assert fake_lerobot["from_pretrained_overrides"] == ["--push_to_hub=false"]
    # The parsed-from-YAML policy stub was replaced by the checkpoint config.
    policy_cfg = backend._train_pipeline_cfg.policy
    assert policy_cfg.type == "fake-pretrained"
    assert policy_cfg.pretrained_path == Path("fake/smolvla_base")
    assert "observation.images.camera1" in policy_cfg.input_features


def test_policy_cli_overrides_json_encode_dict_fields(tmp_path):
    """Dict-valued policy fields (input_features/output_features) must be
    passed to draccus as ONE json-string override per field, not recursed
    into dotted paths — feature keys contain dots (`observation.state`),
    which dotted-path encoding cannot express."""
    from decent_vla.adapters.lerobot.backend import _policy_cli_overrides

    yaml_path = tmp_path / "p.yaml"
    yaml_path.write_text(
        "policy:\n"
        "  type: smolvla\n"
        "  pretrained_path: fake/base\n"
        "  output_features:\n"
        "    action: {type: ACTION, shape: [7]}\n"
        "  input_features:\n"
        "    observation.state: {type: STATE, shape: [8]}\n"
    )
    overrides = _policy_cli_overrides(yaml_path)
    assert '--output_features={"action": {"type": "ACTION", "shape": [7]}}' in overrides
    assert '--input_features={"observation.state": {"type": "STATE", "shape": [8]}}' in overrides
    # No dotted expansion of the dict internals.
    assert not any(o.startswith("--output_features.") for o in overrides)
    assert not any(o.startswith("--input_features.") for o in overrides)


def test_get_state_shared_params_patterns(tmp_path, fake_lerobot):
    """With backend.shared_params, "shared" is the matching trainable subset
    and "personal" is the complement (the personalization seam)."""
    from decent_vla.adapters.lerobot.backend import LeRobotBackend
    from decent_vla.configs.schemas import BackendConfig

    yaml_path = tmp_path / "fake.yaml"
    yaml_path.write_text("#\n")
    cfg = BackendConfig(
        name="lerobot",
        shared_params=["head.*"],
        extra={"train_config_path": str(yaml_path)},
    )
    backend = LeRobotBackend.from_config(cfg, torch.device("cpu"))
    shared = backend.get_state("shared")
    personal = backend.get_state("personal")
    assert shared and all(k.startswith("head.") for k in shared)
    assert personal and all(k.startswith("backbone.") for k in personal)
    # "all" stays the full trainable set regardless of patterns.
    assert set(backend.get_state("all")) == set(shared) | set(personal)


def test_get_state_full_finetune(tmp_path, fake_lerobot):
    """Without LoRA, get_state returns trainable params."""
    from decent_vla.adapters.lerobot.backend import LeRobotBackend
    from decent_vla.configs.schemas import BackendConfig

    yaml_path = tmp_path / "fake.yaml"
    yaml_path.write_text("#\n")
    cfg = BackendConfig(name="lerobot", extra={"train_config_path": str(yaml_path)})
    backend = LeRobotBackend.from_config(cfg, torch.device("cpu"))
    state = backend.get_state("shared")
    assert state, "expected non-empty state for full FT"
    # All keys should map to actual policy params.
    own = backend.model.state_dict()
    for k in state:
        assert k in own


def test_set_state_roundtrip(tmp_path, fake_lerobot):
    from decent_vla.adapters.lerobot.backend import LeRobotBackend
    from decent_vla.configs.schemas import BackendConfig

    yaml_path = tmp_path / "fake.yaml"
    yaml_path.write_text("#\n")
    cfg = BackendConfig(name="lerobot", extra={"train_config_path": str(yaml_path)})
    backend = LeRobotBackend.from_config(cfg, torch.device("cpu"))

    original = backend.get_state("shared")
    perturbed = {k: v + 0.5 for k, v in original.items()}
    backend.set_state(perturbed, scope="shared")
    after = backend.get_state("shared")
    for k in original:
        assert torch.allclose(after[k], perturbed[k])


def test_load_checkpoint_server_round_blob(tmp_path, fake_lerobot):
    """`load_checkpoint` must understand the server checkpoint
    format `{"state": shared_state, "round": N}` — not just raw state dicts.

    Regression guard: with strict=False, feeding the wrapper dict to
    load_state_dict loads NOTHING and eval silently runs the base model.
    """
    from decent_vla.adapters.lerobot.backend import LeRobotBackend
    from decent_vla.configs.schemas import BackendConfig

    yaml_path = tmp_path / "fake.yaml"
    yaml_path.write_text("#\n")
    cfg = BackendConfig(name="lerobot", extra={"train_config_path": str(yaml_path)})
    backend = LeRobotBackend.from_config(cfg, torch.device("cpu"))

    trained = {k: v + 1.0 for k, v in backend.get_state("shared").items()}
    ckpt = tmp_path / "server_round_final.pt"
    torch.save({"state": trained, "round": 49}, ckpt)

    backend.load_checkpoint(str(ckpt))
    after = backend.get_state("shared")
    for k, v in trained.items():
        assert torch.allclose(after[k], v), f"param {k} not loaded from server blob"


def test_load_checkpoint_rejects_blob_with_no_matching_keys(tmp_path, fake_lerobot):
    """A blob whose keys match nothing in the policy must raise, not silently
    no-op (the eval-the-base-model failure mode)."""
    from decent_vla.adapters.lerobot.backend import LeRobotBackend
    from decent_vla.configs.schemas import BackendConfig

    yaml_path = tmp_path / "fake.yaml"
    yaml_path.write_text("#\n")
    cfg = BackendConfig(name="lerobot", extra={"train_config_path": str(yaml_path)})
    backend = LeRobotBackend.from_config(cfg, torch.device("cpu"))

    ckpt = tmp_path / "garbage.pt"
    torch.save({"unrelated_key": torch.zeros(3)}, ckpt)

    with pytest.raises(ValueError, match="no keys"):
        backend.load_checkpoint(str(ckpt))


def test_load_checkpoint_rejects_arm_mismatch_absent_keys(tmp_path, fake_lerobot):
    """A checkpoint that overlaps on shared keys but ALSO carries keys the policy
    cannot receive (a LoRA checkpoint loaded into a full-FT policy: the shared
    action-head keys overlap, but every lora_* weight would be silently dropped)
    must raise, not partial-load. This is the mechanism behind the ~0%-eval
    arm-mismatch incidents."""
    from decent_vla.adapters.lerobot.backend import LeRobotBackend
    from decent_vla.configs.schemas import BackendConfig

    yaml_path = tmp_path / "fake.yaml"
    yaml_path.write_text("#\n")
    cfg = BackendConfig(name="lerobot", extra={"train_config_path": str(yaml_path)})
    backend = LeRobotBackend.from_config(cfg, torch.device("cpu"))

    # real shared keys (overlap) + a bogus LoRA key the policy lacks
    blob = dict(backend.get_state("shared"))
    blob["model.lora_A.weight"] = torch.zeros(2)
    ckpt = tmp_path / "lora_into_fullft.pt"
    torch.save({"state": blob, "round": 5}, ckpt)

    with pytest.raises(ValueError, match="cannot receive"):
        backend.load_checkpoint(str(ckpt))


def test_load_checkpoint_warns_when_it_covers_only_part_of_the_model(tmp_path, fake_lerobot, caplog):
    """A partial load must stay LEGAL (training resume overlays the client's
    personal half afterwards) but must not be SILENT.

    A partial-federation server checkpoint carries only the shared subset, so
    exporting it as a standalone policy leaves every personal parameter at its
    pretrained value. That model is not deployable — a backbone-share arm's
    server checkpoint scored 0.0% on all four LIBERO suites (400 episodes)
    with no warning anywhere in the log. The load must say what it left behind.
    """
    import logging

    from decent_vla.adapters.lerobot.backend import LeRobotBackend
    from decent_vla.configs.schemas import BackendConfig

    yaml_path = tmp_path / "fake.yaml"
    yaml_path.write_text("#\n")
    # A backbone-share arm: the server federates only "backbone.*", so its
    # checkpoint can never carry the head. Coverage must be judged against ALL
    # trainable params — judging it against the shared scope would call this
    # checkpoint complete, which is exactly how the 0% slipped through.
    cfg = BackendConfig(
        name="lerobot",
        shared_params=["backbone.*"],
        extra={"train_config_path": str(yaml_path)},
    )
    backend = LeRobotBackend.from_config(cfg, torch.device("cpu"))

    shared = backend.get_state("shared")
    personal = backend.get_state("personal")
    assert shared and personal, "fixture must split into both scopes"
    ckpt = tmp_path / "server_shared_only.pt"
    torch.save({"state": dict(shared), "round": 49}, ckpt)

    with caplog.at_level(logging.WARNING, logger="decent_vla.adapters.lerobot.backend"):
        backend.load_checkpoint(str(ckpt))

    msg = caplog.text
    assert "partial" in msg.lower(), f"no partial-load warning emitted; got: {msg!r}"
    assert str(len(personal)) in msg, "warning must count the untouched trainable params"


def test_load_checkpoint_does_not_warn_on_a_complete_load(tmp_path, fake_lerobot, caplog):
    """The warning must not cry wolf on the ordinary full-coverage load."""
    import logging

    from decent_vla.adapters.lerobot.backend import LeRobotBackend
    from decent_vla.configs.schemas import BackendConfig

    yaml_path = tmp_path / "fake.yaml"
    yaml_path.write_text("#\n")
    cfg = BackendConfig(name="lerobot", extra={"train_config_path": str(yaml_path)})
    backend = LeRobotBackend.from_config(cfg, torch.device("cpu"))

    ckpt = tmp_path / "server_round_final.pt"
    torch.save({"state": dict(backend.get_state("shared")), "round": 49}, ckpt)

    with caplog.at_level(logging.WARNING, logger="decent_vla.adapters.lerobot.backend"):
        backend.load_checkpoint(str(ckpt))

    assert "partial" not in caplog.text.lower(), f"spurious warning: {caplog.text!r}"


def test_personal_scope_is_empty(tmp_path, fake_lerobot):
    from decent_vla.adapters.lerobot.backend import LeRobotBackend
    from decent_vla.configs.schemas import BackendConfig

    yaml_path = tmp_path / "fake.yaml"
    yaml_path.write_text("#\n")
    cfg = BackendConfig(name="lerobot", extra={"train_config_path": str(yaml_path)})
    backend = LeRobotBackend.from_config(cfg, torch.device("cpu"))
    assert backend.get_state("personal") == {}


# ---------------------------------------------------------------------------
# LR scheduler (recipe parity: warmup then cosine to decay_lr floor)
# ---------------------------------------------------------------------------


def _build_backend(tmp_path, **cfg_kwargs):
    from decent_vla.adapters.lerobot.backend import LeRobotBackend
    from decent_vla.configs.schemas import BackendConfig

    yaml_path = tmp_path / "fake.yaml"
    yaml_path.write_text("#\n")
    extra = {"train_config_path": str(yaml_path)}
    extra.update(cfg_kwargs.pop("extra", {}))
    cfg = BackendConfig(name="lerobot", extra=extra, **cfg_kwargs)
    return LeRobotBackend.from_config(cfg, torch.device("cpu"))


def test_cosine_scheduler_is_warmup_cosine_with_floor(tmp_path, fake_lerobot):
    """scheduler='cosine' must give the lerobot warmup+cosine-to-floor schedule,
    NOT CosineAnnealingLR over the warmup step count (the old bug annealed the
    LR to zero within the first `warmup_steps` steps)."""
    from decent_vla.configs.schemas import OptimConfig

    backend = _build_backend(tmp_path)
    optim_cfg = OptimConfig(
        lr=1.0e-4,
        scheduler="cosine",
        warmup_steps=100,
        decay_steps=1000,
        decay_lr=2.5e-5,
    )
    opt, sched = backend.build_optimizer(optim_cfg)
    assert sched is not None

    def lr():
        return opt.param_groups[0]["lr"]

    assert lr() == pytest.approx(1.0e-4 / 100)  # step 0: ramp start
    for _ in range(100):
        sched.step()
    assert lr() == pytest.approx(1.0e-4)  # end of warmup: peak
    for _ in range(900):
        sched.step()
    assert lr() == pytest.approx(2.5e-5)  # decay_steps: floor
    for _ in range(100):
        sched.step()
    assert lr() == pytest.approx(2.5e-5)  # constant floor after


def test_cosine_scheduler_with_zero_warmup(tmp_path, fake_lerobot):
    from decent_vla.configs.schemas import OptimConfig

    backend = _build_backend(tmp_path)
    optim_cfg = OptimConfig(
        lr=1.0e-4,
        scheduler="cosine",
        warmup_steps=0,
        decay_steps=1000,
        decay_lr=None,
    )
    opt, sched = backend.build_optimizer(optim_cfg)
    assert sched is not None
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0e-4)  # no ramp


# ---------------------------------------------------------------------------
# dtype: bfloat16 (backend.extra.dtype)
# ---------------------------------------------------------------------------


def test_extra_dtype_bfloat16_uses_autocast(tmp_path, fake_lerobot):
    """bf16 = autocast mixed precision (weights stay fp32), as in lerobot's
    trainer; a pure weight cast breaks smolvla's fp32 noise."""
    backend = _build_backend(tmp_path, extra={"dtype": "bfloat16"})
    dtypes = {p.dtype for p in backend._policy.parameters()}
    assert dtypes == {torch.float32}  # weights NOT cast
    assert backend._autocast_dtype is torch.bfloat16
    seen = {}
    orig_forward = backend._policy.forward

    def spy_forward(batch):
        seen["autocast"] = torch.is_autocast_enabled("cpu")
        return orig_forward(batch)

    backend._policy.forward = spy_forward
    backend.compute_loss({"observation": torch.randn(2, 16), "action": torch.randn(2, 7)})
    assert seen["autocast"] is True


def test_extra_dtype_unknown_raises(tmp_path, fake_lerobot):
    with pytest.raises(ValueError, match="dtype"):
        _build_backend(tmp_path, extra={"dtype": "float8"})


# ---------------------------------------------------------------------------
# LoRA + shared scope = ALL trainable params (lora adapters AND action head)
# ---------------------------------------------------------------------------
#
# `apply_peft` wraps only the VLM submodule; the action head outside it stays
# requires_grad=True and is trained by the optimizer. A `"lora_" in key`
# filter would drop it from aggregation/broadcast — the documented
# action-head failure mode (server checkpoint evals ~0%). These tests fake
# `peft` so the LoRA code path runs without the real dependency.


class _FakeVLMPolicy(nn.Module):
    """Fake lerobot policy: a vlm submodule (PEFT target) + an action head."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(n_action_steps=1, action_horizon=1)
        self.vlm = nn.Linear(16, 8)
        self.action_head = nn.Linear(8, 7)

    def forward(self, batch):
        pred = self.action_head(torch.relu(self.vlm(batch["observation"])))
        loss = ((pred - batch["action"]) ** 2).mean()
        return loss, {}


class _FakePeftWrapped(nn.Module):
    """Mimics peft.get_peft_model: freezes the base, adds lora_* params."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base_model = base
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.zeros(4, 16))
        self.lora_B = nn.Parameter(torch.zeros(8, 4))

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def merge_and_unload(self):
        # Real peft folds the adapters into the base and returns it unfrozen.
        for p in self.base_model.parameters():
            p.requires_grad_(True)
        return self.base_model


@pytest.fixture
def fake_peft_lerobot(fake_lerobot):
    """fake_lerobot + a fake `peft` module + make_policy -> _FakeVLMPolicy."""
    peft_mod = types.ModuleType("peft")

    class _LoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    peft_mod.LoraConfig = _LoraConfig
    peft_mod.get_peft_model = lambda model, cfg: _FakePeftWrapped(model)
    peft_mod.PeftModel = _FakePeftWrapped
    sys.modules["peft"] = peft_mod

    def _make_vlm_policy(cfg, ds_meta, rename_map=None):
        fake_lerobot["make_policy_calls"] += 1
        return _FakeVLMPolicy()

    sys.modules["lerobot.policies.factory"].make_policy = _make_vlm_policy
    yield fake_lerobot
    sys.modules.pop("peft", None)


def _lora_backend(tmp_path):
    from decent_vla.core.lora import LoRAConfig

    return _build_backend(tmp_path, lora=LoRAConfig(r=8, alpha=16))


def test_lora_shared_state_includes_action_head(tmp_path, fake_peft_lerobot):
    backend = _lora_backend(tmp_path)
    state = backend.get_state("shared")

    assert any("lora_" in k for k in state), f"no lora keys in {sorted(state)}"
    assert "action_head.weight" in state
    assert "action_head.bias" in state
    # Frozen base (the PEFT-wrapped vlm) must NOT be aggregated.
    frozen = [k for k in state if "base_model" in k]
    assert frozen == [], f"frozen base params leaked into shared state: {frozen}"
    # Shared state == exactly the optimizer's trainable set.
    trainable = {n for n, p in backend.model.named_parameters() if p.requires_grad}
    assert set(state) == trainable


def test_lora_composes_with_shared_params_patterns(tmp_path, fake_peft_lerobot):
    """Axis 2 under LoRA: shared_params can carve the LoRA-run trainable set
    (adapters + head) into a federated subset and a personal remainder."""
    from decent_vla.adapters.lerobot.backend import LeRobotBackend
    from decent_vla.configs.schemas import BackendConfig
    from decent_vla.core.lora import LoRAConfig

    yaml_path = tmp_path / "fake.yaml"
    yaml_path.write_text("#\n")
    cfg = BackendConfig(
        name="lerobot",
        lora=LoRAConfig(r=8, alpha=16),
        shared_params=["*lora_*"],
        extra={"train_config_path": str(yaml_path)},
    )
    backend = LeRobotBackend.from_config(cfg, torch.device("cpu"))
    shared = backend.get_state("shared")
    personal = backend.get_state("personal")
    assert shared and all("lora_" in k for k in shared)
    assert "action_head.weight" in personal and "action_head.bias" in personal
    # The two scopes partition the trainable set exactly.
    assert set(shared) | set(personal) == set(backend.get_state("all"))
    assert not set(shared) & set(personal)


def test_trainable_params_freeze_axis(tmp_path, fake_lerobot):
    """Axis 1: backend.trainable_params freezes the complement post-build —
    the generic form of the per-policy freeze flags (closes the PI05Config
    no-flags gap)."""
    backend = _build_backend(tmp_path, trainable_params=["head.*"])
    trainable = {n for n, p in backend.model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("head.") for n in trainable)
    assert set(backend.get_state("all")) == trainable
    assert set(backend.get_state("shared")) == trainable  # axis 2 unset


def test_lora_set_state_roundtrips_head_and_lora(tmp_path, fake_peft_lerobot):
    backend = _lora_backend(tmp_path)
    original = backend.get_state("shared")
    perturbed = {k: v + 0.5 for k, v in original.items()}
    backend.set_state(perturbed, scope="shared")
    after = backend.get_state("shared")
    assert set(after) == set(original)
    for k in original:
        assert torch.allclose(after[k], perturbed[k]), k
    # Frozen base params untouched by the broadcast.
    own = backend.model.state_dict()
    assert all("base_model" not in k for k in after)
    assert "vlm.base_model.weight" in own  # sanity: frozen base still exists


def test_export_pretrained_merges_lora_and_saves(tmp_path, fake_peft_lerobot):
    """`export_pretrained` loads a server checkpoint, folds the LoRA adapters
    into the base weights (merge_and_unload), and writes a lerobot-native
    `pretrained_model/` dir via lerobot's save_checkpoint — the input format
    of `lerobot-eval --policy.path`."""
    from decent_vla.adapters.lerobot.export import export_pretrained

    backend = _lora_backend(tmp_path)
    trained = {k: v + 1.0 for k, v in backend.get_state("shared").items()}
    ckpt = tmp_path / "server_round_final.pt"
    torch.save({"state": trained, "round": 49}, ckpt)

    calls: dict[str, Any] = {}
    tu_mod = types.ModuleType("lerobot.utils.train_utils")

    def _save_checkpoint(
        checkpoint_dir,
        step,
        cfg,
        policy,
        optimizer=None,
        scheduler=None,
        preprocessor=None,
        postprocessor=None,
    ):
        calls["dir"] = Path(checkpoint_dir)
        calls["step"] = step
        calls["policy"] = policy
        calls["pre"] = preprocessor
        calls["post"] = postprocessor

    tu_mod.save_checkpoint = _save_checkpoint
    tu_mod.update_last_checkpoint = lambda d: calls.__setitem__("last", Path(d))
    sys.modules["lerobot.utils"] = types.ModuleType("lerobot.utils")
    sys.modules["lerobot.utils.train_utils"] = tu_mod
    try:
        out = export_pretrained(backend, str(ckpt), tmp_path / "export")
    finally:
        sys.modules.pop("lerobot.utils.train_utils", None)
        sys.modules.pop("lerobot.utils", None)

    from peft import PeftModel

    policy = calls["policy"]
    assert not any(isinstance(m, PeftModel) for m in policy.modules()), "PEFT wrapper must be merged away before save"
    assert torch.allclose(policy.action_head.weight, trained["action_head.weight"])
    assert calls["step"] == 50  # blob round is 0-indexed; humans count from 1
    assert calls["last"] == calls["dir"]
    assert out == calls["dir"] / "pretrained_model"


def test_export_pretrained_server_state_lands_on_disk(tmp_path, fake_peft_lerobot):
    """Audit-grade export contract (2026-08-14): the numbers in a
    server_round_NNNN.pt blob — BOTH halves, action head AND LoRA adapters —
    must be recoverable from the weights written under `pretrained_model/`.

    The sibling test above proves the wrapper is merged away and the head
    tensor reaches the policy object; its fake `merge_and_unload` DISCARDS
    the adapters, so a regression that silently drops the LoRA half of the
    blob (or never loads the blob at all) would stay green there. Here the
    fake folds like real peft (base += B @ A) and `save_checkpoint`
    materializes the weights, so the assertions read what lands ON DISK.
    """
    from decent_vla.adapters.lerobot.export import export_pretrained

    class _FoldingPeftWrapped(_FakePeftWrapped):
        def merge_and_unload(self):
            with torch.no_grad():
                self.base_model.weight += self.lora_B @ self.lora_A
            for p in self.base_model.parameters():
                p.requires_grad_(True)
            return self.base_model

    sys.modules["peft"].get_peft_model = lambda model, cfg: _FoldingPeftWrapped(model)
    backend = _lora_backend(tmp_path)

    base_before = backend.model.vlm.base_model.weight.detach().clone()
    trained = {k: torch.full_like(v, 0.25 if "lora_" in k else 3.0) for k, v in backend.get_state("shared").items()}
    ckpt = tmp_path / "server_round_0100.pt"
    torch.save({"state": trained, "round": 99}, ckpt)

    tu_mod = types.ModuleType("lerobot.utils.train_utils")

    def _save_checkpoint(
        checkpoint_dir, step, cfg, policy, optimizer=None, scheduler=None, preprocessor=None, postprocessor=None
    ):
        pm = Path(checkpoint_dir) / "pretrained_model"
        pm.mkdir(parents=True)
        torch.save(policy.state_dict(), pm / "model_state.pt")

    tu_mod.save_checkpoint = _save_checkpoint
    tu_mod.update_last_checkpoint = lambda d: None
    sys.modules["lerobot.utils"] = types.ModuleType("lerobot.utils")
    sys.modules["lerobot.utils.train_utils"] = tu_mod
    try:
        out = export_pretrained(backend, str(ckpt), tmp_path / "export")
    finally:
        sys.modules.pop("lerobot.utils.train_utils", None)
        sys.modules.pop("lerobot.utils", None)

    exported = torch.load(out / "model_state.pt", weights_only=True)
    # Head half of the blob lands verbatim.
    assert torch.equal(exported["action_head.weight"], trained["action_head.weight"])
    assert torch.equal(exported["action_head.bias"], trained["action_head.bias"])
    # LoRA half lands folded: base + B @ A with the BLOB's adapter values.
    # (Adapters init to zero, so an export that skips the load folds nothing
    # and the exported base equals the untrained one — this catches it.)
    expected = base_before + trained["vlm.lora_B"] @ trained["vlm.lora_A"]
    assert (expected - base_before).abs().max() > 0
    assert torch.allclose(exported["vlm.weight"], expected)
    # No raw adapter tensors ship in the export.
    assert not any("lora_" in k for k in exported)


def test_lora_optimizer_trains_exactly_shared_state(tmp_path, fake_peft_lerobot):
    """Optimizer param set and shared-state key set must agree, or trained
    params silently never reach the server."""
    from decent_vla.configs.schemas import OptimConfig

    backend = _lora_backend(tmp_path)
    opt, _ = backend.build_optimizer(OptimConfig())
    opt_params = {id(p) for g in opt.param_groups for p in g["params"]}
    named = {n: p for n, p in backend.model.named_parameters()}
    shared_ids = {id(named[k]) for k in backend.get_state("shared")}
    assert shared_ids == opt_params
