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

import hashlib
import importlib.util
import json
import os
import shlex
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

HAS_NVFLARE_RUNTIME_DEPS = importlib.util.find_spec("msgpack") is not None


def _example_dir() -> Path:
    return Path(__file__).parents[5] / "examples" / "advanced" / "bionemo" / "evo2"


def _load_job_module():
    example_dir = _example_dir()
    previous_path = sys.path[:]
    sys.path.insert(0, str(example_dir))
    try:
        module_path = example_dir / "job.py"
        spec = importlib.util.spec_from_file_location("evo2_job", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous_path


@contextmanager
def _chdir(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _write_manifest(data_dir: Path) -> Path:
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    for name, count in {
        "site-1.jsonl": 10,
        "site-2.jsonl": 30,
        "site-3.jsonl": 60,
        "pooled.jsonl": 100,
    }.items():
        (train_dir / name).write_text('{"label": 0}\n' * count, encoding="utf-8")
    (data_dir / "validation.jsonl").write_text('{"label": 0}\n' * 10, encoding="utf-8")
    (data_dir / "test.jsonl").write_text('{"label": 0}\n' * 12, encoding="utf-8")

    def identity(path):
        payload = path.read_bytes()
        return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload), "rows": len(payload.splitlines())}

    site_files = {
        "site-1": "train/site-1.jsonl",
        "site-2": "train/site-2.jsonl",
        "site-3": "train/site-3.jsonl",
    }
    manifest = {
        "format_version": 2,
        "audit": {"status": "passed"},
        "source": {"dataset_id": "example/test", "revision": "abc123"},
        "settings": {"seed": 42, "partition": "iid"},
        "files": {
            "sites": site_files,
            "pooled_train": "train/pooled.jsonl",
            "validation": "validation.jsonl",
            "test": "test.jsonl",
        },
        "counts": {
            "train": 100,
            "validation": 10,
            "test": 12,
            "sites": {
                "site-1": {"count": 10, "label_histogram": {"0": 10}},
                "site-2": {"count": 30, "label_histogram": {"1": 30}},
                "site-3": {"count": 60, "label_histogram": {"2": 60}},
            },
        },
        "file_identities": {
            "pooled_train": identity(train_dir / "pooled.jsonl"),
            "validation": identity(data_dir / "validation.jsonl"),
            "test": identity(data_dir / "test.jsonl"),
            "sites": {site: identity(data_dir / relative_path) for site, relative_path in site_files.items()},
        },
    }
    path = data_dir / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _args(job_module, tmp_path: Path):
    import torch

    data_dir = tmp_path / "data with spaces"
    manifest = _write_manifest(data_dir)
    initial_checkpoint = tmp_path / "common init.pt"
    torch.save(
        {
            "model": {
                "decoder.layers.0.linear_qkv.adapter.linear_in.weight": torch.zeros(2, 4),
                "decoder.classification_head.weight": torch.zeros(3, 4),
                "decoder.classification_head.bias": torch.zeros(3),
            },
            "meta_props": {
                "backend": "mock",
                "base_checkpoint": str((tmp_path / "models" / "base").resolve()),
                "exchange_dtype": "float32",
                "peft_mode": "lora",
                "seed": 1234,
                "seq_length": 600,
                "lora_dim": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "lora_target_modules": [
                    "linear_qkv",
                    "linear_proj",
                    "linear_fc1",
                    "linear_fc2",
                    "dense_projection",
                    "dense",
                ],
                "training_inputs": {"data_file": None, "base_checkpoint": None, "classifier_file": None},
            },
        },
        initial_checkpoint,
    )
    return job_module.define_parser().parse_args(
        [
            "--data-dir",
            str(data_dir),
            "--manifest",
            str(manifest),
            "--initial-checkpoint",
            str(initial_checkpoint),
            "--workspace",
            str(tmp_path / "workspace with spaces"),
            "--backend",
            "mock",
            "--num-rounds",
            "2",
            "--num-clients",
            "3",
        ]
    )


def _continuation_signature(job_module, args):
    manifest = job_module._load_manifest(args.manifest)
    plan = job_module._site_plan(args, manifest)
    validation_file = Path(args.data_dir) / manifest["files"]["validation"]
    training_inputs = job_module.collect_training_input_provenance(args, manifest, plan, validation_file)
    return job_module.build_continuation_signature(args, manifest, plan, training_inputs)


def _write_global_checkpoint(
    job_module,
    result_dir: Path,
    *,
    args,
    initial_checkpoint: str,
    current_round: int = 1,
    nr_aggregated: int = 3,
) -> Path:
    import torch

    checkpoint = (
        result_dir / "server" / "simulate_job" / "app_server" / job_module.DefaultCheckpointFileName.GLOBAL_MODEL
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)["meta_props"]
    initialization = job_module.provenance.resolve_initialization_metadata(checkpoint_metadata)
    continuation_signature = _continuation_signature(job_module, args)

    def payload(round_index):
        return {
            "model": {
                "decoder.layers.0.linear_qkv.adapter.linear_in.weight": torch.zeros(2, 4),
                "decoder.classification_head.weight": torch.zeros(3, 4),
                "decoder.classification_head.bias": torch.zeros(3),
            },
            "meta_props": {
                "current_round": round_index,
                "nr_aggregated": nr_aggregated,
                "initialization": initialization,
                "continuation_signature": continuation_signature,
            },
        }

    torch.save(payload(current_round), checkpoint)
    for round_index in range(current_round + 1):
        round_checkpoint = checkpoint.with_name(f"{checkpoint.stem}_round_{round_index:03d}{checkpoint.suffix}")
        torch.save(payload(round_index), round_checkpoint)
    return checkpoint


def _write_round_metric(workspace: Path, site_name: str, round_index: int, *, scale: float = 1.0) -> Path:
    import torch

    metric_path = (
        workspace / "client_work" / site_name / f"{site_name}_round_{round_index:03d}_attempt" / "round_metrics.json"
    )
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    local_checkpoint = metric_path.parent / "local_trainable_model.pt"
    torch.save(
        {
            "model": {
                "decoder.layers.0.linear_qkv.adapter.linear_in.weight": torch.ones(2, 4),
                "decoder.classification_head.weight": torch.ones(3, 4),
                "decoder.classification_head.bias": torch.ones(3),
            },
            "meta_props": {"site_name": site_name, "round": round_index},
        },
        local_checkpoint,
    )
    metric_path.write_text(
        json.dumps(
            {
                "runtime_seconds": 2.0 * scale,
                "peak_gpu_memory_mebibytes": 100.0 * scale,
                "received_mebibytes": 3.0 * scale,
                "sent_mebibytes": 1.0 * scale,
                "site_name": site_name,
                "round": round_index,
                "local_checkpoint": str(local_checkpoint.resolve()),
            }
        ),
        encoding="utf-8",
    )
    return metric_path


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_site_plan_uses_manifest_counts_for_fedavg_local_and_pooled(tmp_path):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    manifest = job_module._load_manifest(args.manifest)

    fedavg = job_module._site_plan(args, manifest)

    assert [entry["client_name"] for entry in fedavg] == ["site-1", "site-2", "site-3"]
    assert [entry["sample_count"] for entry in fedavg] == [10, 30, 60]
    assert [entry["mock_delta"] for entry in fedavg] == pytest.approx([0.01, 0.02, 0.03])
    assert all(entry["train_file"].is_file() for entry in fedavg)
    assert [entry["train_identity"]["rows"] for entry in fedavg] == [10, 30, 60]
    assert all(len(entry["train_identity"]["sha256"]) == 64 for entry in fedavg)

    args.mode = "local"
    args.site_index = 2
    assert job_module._site_plan(args, manifest)[0]["sample_count"] == 30

    args.mode = "pooled"
    pooled = job_module._site_plan(args, manifest)
    assert pooled[0]["client_name"] == "site-1"
    assert pooled[0]["sample_count"] == 100
    assert pooled[0]["train_file"].name == "pooled.jsonl"


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_local_mode_rejects_a_requested_site_missing_from_the_manifest(tmp_path):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    args.mode = "local"
    args.site_index = 4

    with pytest.raises(ValueError, match="Manifest does not contain requested local dataset 'site-4'"):
        job_module.validate_inputs(args)


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_require_fresh_workspace_rejects_stale_mock_workspace_before_recipe_construction(tmp_path, monkeypatch):
    job_module = _load_job_module()
    workspace = tmp_path / "stale-local-workspace"
    workspace.mkdir()
    sentinel = workspace / "prior-run.txt"
    sentinel.write_text("preserve me\n", encoding="utf-8")

    def fail_if_reached(*_args, **_kwargs):
        raise AssertionError("recipe construction must not run for a stale workspace")

    monkeypatch.setattr(job_module, "create_recipe", fail_if_reached)
    with pytest.raises(FileExistsError, match="Fresh Evo2 workspace already exists"):
        job_module.main(
            [
                "--backend",
                "mock",
                "--mode",
                "local",
                "--workspace",
                str(workspace),
                "--require-fresh-workspace",
            ]
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve me\n"


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_require_fresh_workspace_rejects_a_dangling_workspace_symlink(tmp_path):
    job_module = _load_job_module()
    workspace = tmp_path / "dangling-local-workspace"
    workspace.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(FileExistsError, match="Fresh Evo2 workspace already exists"):
        job_module.main(
            [
                "--backend",
                "mock",
                "--mode",
                "local",
                "--workspace",
                str(workspace),
                "--require-fresh-workspace",
            ]
        )


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
@pytest.mark.parametrize(("site_index", "sample_count"), ((1, 10), (2, 30), (3, 60)))
def test_matched_local_baseline_recipe_uses_one_site_for_one_uninterrupted_888_step_round(
    tmp_path, site_index, sample_count
):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    args.mode = "local"
    args.site_index = site_index
    args.num_rounds = 1
    args.local_steps = 888
    args.micro_batch_size = 32
    args.global_batch_size = 96

    _manifest, plan, validation_file, _training_inputs, continuation_signature = job_module.validate_inputs(args)
    with _chdir(_example_dir()):
        recipe = job_module.create_recipe(args, plan, validation_file, continuation_signature)

    site_name = f"site-{site_index}"
    assert plan == [
        {
            "client_name": site_name,
            "train_file": Path(args.data_dir).resolve() / "train" / f"{site_name}.jsonl",
            "train_identity": plan[0]["train_identity"],
            "sample_count": sample_count,
            "mock_delta": pytest.approx(args.mock_delta * site_index),
        }
    ]
    assert plan[0]["train_identity"]["rows"] == sample_count
    assert recipe.min_clients == 1
    assert recipe.num_rounds == 1
    assert recipe.aggregation_weights == {site_name: float(sample_count)}
    assert recipe.aggregator.aggregation_weights == recipe.aggregation_weights
    assert set(recipe.per_site_config) == {site_name}
    assert continuation_signature["payload"]["mode"] == "local"
    assert continuation_signature["payload"]["clients"] == [
        {
            "client_name": site_name,
            "sample_weight": float(sample_count),
            "train_file": {field: plan[0]["train_identity"][field] for field in ("sha256", "bytes", "rows")},
        }
    ]
    assert continuation_signature["payload"]["sampler_budget"] == {
        "seed": 1234,
        "local_steps": 888,
        "micro_batch_size": 32,
        "global_batch_size": 96,
    }
    assert continuation_signature["payload"]["optimizer_schedule"] == {
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_iters": args.warmup_iters,
    }
    assert continuation_signature["payload"]["backend_settings"] == {"mock_delta": args.mock_delta}

    train_args = shlex.split(recipe.per_site_config[site_name]["train_args"])
    assert train_args[train_args.index("--local-steps") + 1] == "888"
    assert train_args[train_args.index("--train-file") + 1] == str(plan[0]["train_file"])
    assert train_args[train_args.index("--sample-count") + 1] == str(sample_count)
    assert "--training-state-dir" not in train_args

    sim_env = job_module.create_sim_env(args, plan)
    assert sim_env.clients == [site_name]
    assert sim_env.num_threads == 1
    assert sim_env.gpu_config == "[0]"


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_matched_local_baseline_summary_records_one_contributor_and_no_persistent_state(tmp_path):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    args.mode = "local"
    args.site_index = 2
    args.num_rounds = 1
    args.local_steps = 888
    args.micro_batch_size = 32
    args.global_batch_size = 96
    _manifest, plan, _validation_file, _training_inputs, _signature = job_module.validate_inputs(args)
    result_dir = tmp_path / "local-result"
    started_at = time.time() - 1.0
    _write_global_checkpoint(
        job_module,
        result_dir,
        args=args,
        initial_checkpoint=args.initial_checkpoint,
        current_round=0,
        nr_aggregated=1,
    )
    _write_round_metric(Path(args.workspace), "site-2", 0)

    summary = job_module.collect_run_summary(args, plan, result_dir, started_at)

    assert summary["mode"] == "local"
    assert summary["num_clients"] == 1
    assert summary["num_rounds"] == 1
    assert summary["local_steps"] == 888
    assert summary["global_checkpoint_round"] == 0
    assert summary["global_checkpoint_contributors"] == 1
    assert summary["aggregation_weights"] == {"site-2": 30}
    assert summary["configuration"]["micro_batch_size"] == 32
    assert summary["configuration"]["global_batch_size"] == 96
    assert "persist_client_training_state" not in summary["configuration"]
    assert [(metric["site_name"], metric["round"]) for metric in summary["round_metrics"]] == [("site-2", 0)]


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_three_matched_local_mock_jobs_run_sequentially_with_fresh_site_bound_outputs(tmp_path, monkeypatch):
    import torch

    job_module = _load_job_module()
    fixture_args = _args(job_module, tmp_path / "fixture")
    data_dir = Path(fixture_args.data_dir)
    manifest_path = Path(fixture_args.manifest)
    initial_checkpoint = Path(fixture_args.initial_checkpoint)

    site_payloads = {
        f"site-{site_index}": (json.dumps({"label": site_index - 1}) + "\n").encode() * 9000 for site_index in (1, 2, 3)
    }
    for site_name, payload in site_payloads.items():
        (data_dir / "train" / f"{site_name}.jsonl").write_bytes(payload)
    (data_dir / "train" / "pooled.jsonl").write_bytes(b"".join(site_payloads.values()))

    def identity(path: Path) -> dict:
        payload = path.read_bytes()
        return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload), "rows": len(payload.splitlines())}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["train"] = 27000
    manifest["file_identities"]["pooled_train"] = identity(data_dir / "train" / "pooled.jsonl")
    for site_index in (1, 2, 3):
        site_name = f"site-{site_index}"
        manifest["counts"]["sites"][site_name] = {
            "count": 9000,
            "label_histogram": {str(site_index - 1): 9000},
        }
        manifest["file_identities"]["sites"][site_name] = identity(data_dir / "train" / f"{site_name}.jsonl")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    execution_order = []
    constructed_jobs = {}
    original_create_recipe = job_module.create_recipe

    def create_recipe_with_mock_execution(args, plan, validation_file, continuation_signature):
        recipe = original_create_recipe(args, plan, validation_file, continuation_signature)
        site_name = plan[0]["client_name"]
        site_index = int(site_name.removeprefix("site-"))
        train_args = shlex.split(recipe.per_site_config[site_name]["train_args"])
        constructed_jobs[site_index] = {
            "train_args": train_args,
            "workspace": Path(args.workspace),
        }

        class MockRun:
            def __init__(self, result_dir):
                self.result_dir = result_dir

            def get_result(self):
                return self.result_dir

        class MockExecutableRecipe:
            def execute(self, sim_env):
                execution_order.append(site_index)
                assert sim_env.clients == [site_name]
                assert sim_env.num_threads == 1
                assert sim_env.gpu_config == "[0]"
                result_dir = Path(args.workspace) / "simulator-result"
                global_checkpoint = _write_global_checkpoint(
                    job_module,
                    result_dir,
                    args=args,
                    initial_checkpoint=args.initial_checkpoint,
                    current_round=0,
                    nr_aggregated=1,
                )
                _write_round_metric(Path(args.workspace), site_name, 0)
                local_checkpoint = next(
                    (Path(args.workspace) / "client_work" / site_name).glob("*/local_trainable_model.pt")
                )
                local_state = torch.load(local_checkpoint, map_location="cpu", weights_only=True)["model"]
                round_checkpoint = global_checkpoint.with_name(
                    f"{global_checkpoint.stem}_round_000{global_checkpoint.suffix}"
                )
                for checkpoint_path in (global_checkpoint, round_checkpoint):
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    checkpoint["model"] = {name: value.clone() for name, value in local_state.items()}
                    torch.save(checkpoint, checkpoint_path)
                return MockRun(result_dir)

        return MockExecutableRecipe()

    monkeypatch.setattr(job_module, "create_recipe", create_recipe_with_mock_execution)

    summaries = {}
    for site_index in (1, 2, 3):
        workspace = tmp_path / f"local-site-{site_index}"
        assert not workspace.exists()
        argv = [
            "--backend",
            "mock",
            "--mode",
            "local",
            "--site-index",
            str(site_index),
            "--num-clients",
            "1",
            "--data-dir",
            str(data_dir),
            "--manifest",
            str(manifest_path),
            "--initial-checkpoint",
            str(initial_checkpoint),
            "--workspace",
            str(workspace),
            "--start-round",
            "0",
            "--num-rounds",
            "1",
            "--local-steps",
            "888",
            "--seq-length",
            "600",
            "--micro-batch-size",
            "32",
            "--global-batch-size",
            "96",
            "--learning-rate",
            "0.0005",
            "--min-learning-rate",
            "0.00005",
            "--warmup-iters",
            "30",
            "--eval-iters",
            "1",
            "--seed",
            "1234",
            "--gpu",
            "[0]",
            "--num-threads",
            "1",
        ]
        with _chdir(_example_dir()):
            job_module.main(argv)
        summaries[site_index] = json.loads((workspace / "run_summary.json").read_text(encoding="utf-8"))

    assert execution_order == [1, 2, 3]
    assert len({summary["initial_checkpoint_sha256"] for summary in summaries.values()}) == 1
    for site_index, summary in summaries.items():
        site_name = f"site-{site_index}"
        assert summary["mode"] == "local"
        assert summary["num_clients"] == 1
        assert summary["num_rounds"] == 1
        assert summary["local_steps"] == 888
        assert summary["global_checkpoint_contributors"] == 1
        assert summary["global_checkpoint_round"] == 0
        assert summary["aggregation_weights"] == {site_name: 9000}
        assert set(summary["training_inputs"]["train_files"]) == {site_name}
        assert summary["training_inputs"]["train_files"][site_name]["rows"] == 9000
        assert summary["continuation_signature"]["payload"]["clients"][0]["client_name"] == site_name
        assert summary["continuation_signature"]["payload"]["sampler_budget"] == {
            "seed": 1234,
            "local_steps": 888,
            "micro_batch_size": 32,
            "global_batch_size": 96,
        }
        assert summary["continuation_signature"]["payload"]["optimizer_schedule"] == {
            "learning_rate": 0.0005,
            "min_learning_rate": 0.00005,
            "warmup_iters": 30,
        }
        assert summary["continuation_signature"]["payload"]["backend_settings"] == {"mock_delta": 0.01}
        assert "persist_client_training_state" not in summary["configuration"]
        assert [(metric["site_name"], metric["round"]) for metric in summary["round_metrics"]] == [(site_name, 0)]
        train_args = constructed_jobs[site_index]["train_args"]
        assert train_args[train_args.index("--train-file") + 1] == str(
            (data_dir / "train" / f"{site_name}.jsonl").resolve()
        )
        assert train_args[train_args.index("--local-steps") + 1] == "888"
        assert "--training-state-dir" not in train_args
        global_checkpoint = Path(summary["global_checkpoint"])
        local_checkpoint = Path(summary["round_metrics"][0]["local_checkpoint"])
        assert global_checkpoint.is_file()
        assert local_checkpoint.is_file()
        global_state = torch.load(global_checkpoint, map_location="cpu", weights_only=True)["model"]
        local_state = torch.load(local_checkpoint, map_location="cpu", weights_only=True)["model"]
        assert all(torch.equal(global_state[name], local_state[name]) for name in global_state)


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_recipe_carries_sample_weights_and_shell_safe_site_arguments(tmp_path):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    _manifest, plan, validation_file, training_inputs, continuation_signature = job_module.validate_inputs(args)

    with _chdir(_example_dir()):
        recipe = job_module.create_recipe(args, plan, validation_file, continuation_signature)
        export_root = tmp_path / "exported"
        recipe.export(str(export_root))

    assert recipe.min_clients == 3
    assert recipe.num_rounds == 2
    assert recipe.launch_external_process is True
    assert recipe.launch_once is False
    command = shlex.split(recipe.command)
    assert command == [
        sys.executable,
        "-u",
        "custom/sequential_launcher.py",
        "--lock-file",
        str(Path(args.workspace).resolve() / job_module.TRAINING_LOCK_FILENAME),
        "--",
        sys.executable,
        "-u",
    ]
    assert recipe.server_expected_format == "pytorch"
    assert recipe.params_transfer_type == "DIFF"
    assert recipe.aggregation_weights == {"site-1": 10.0, "site-2": 30.0, "site-3": 60.0}
    assert recipe.aggregator.aggregation_weights == recipe.aggregation_weights
    assert recipe.aggregator.continuation_signature == continuation_signature
    assert recipe.aggregator.schema_checkpoint == str(Path(args.initial_checkpoint).resolve())
    assert training_inputs["train_files"]["site-1"]["rows"] == 10
    signature_payload = continuation_signature["payload"]
    assert signature_payload["mode"] == "fedavg"
    assert [(item["client_name"], item["sample_weight"]) for item in signature_payload["clients"]] == [
        ("site-1", 10.0),
        ("site-2", 30.0),
        ("site-3", 60.0),
    ]
    assert signature_payload["dataset_manifest"]["settings"]["partition"] == "iid"
    assert signature_payload["sampler_budget"] == {
        "seed": args.seed,
        "local_steps": args.local_steps,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
    }
    assert signature_payload["optimizer_schedule"] == {
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_iters": args.warmup_iters,
    }
    assert signature_payload["backend_settings"] == {"mock_delta": args.mock_delta}

    exported_job = export_root / recipe.name
    server_config = json.loads(
        (exported_job / "app_server" / "config" / "config_fed_server.json").read_text(encoding="utf-8")
    )
    controller = next(workflow for workflow in server_config["workflows"] if workflow["path"].endswith(".FedAvg"))
    assert controller["args"].get("start_round", 0) == 0
    exported_aggregator = controller["args"]["aggregator"]
    assert exported_aggregator["path"] == "evo2_aggregator.ExactSchemaFedAvgAggregator"
    assert exported_aggregator["args"]["aggregation_weights"] == recipe.aggregation_weights
    assert exported_aggregator["args"]["continuation_signature"] == continuation_signature
    persistor = next(component for component in server_config["components"] if component["id"] == "persistor")
    assert persistor["path"] == "evo2_persistor.CPUTrainablePTFileModelPersistor"
    assert "load_device" not in persistor["args"]
    assert (exported_job / "app_server" / "custom" / "evo2_aggregator.py").is_file()
    assert (exported_job / "app_server" / "custom" / "adapter_checkpoint.py").is_file()
    assert (exported_job / "app_server" / "custom" / "evo2_persistor.py").is_file()
    assert (exported_job / "app_server" / "custom" / "provenance.py").is_file()

    for entry in plan:
        client_app = exported_job / f"app_{entry['client_name']}"
        assert (client_app / "custom" / "sequential_launcher.py").is_file()
        assert (client_app / "custom" / "provenance.py").is_file()
        client_config = json.loads((client_app / "config" / "config_fed_client.json").read_text(encoding="utf-8"))
        executor = next(
            item["executor"]
            for item in client_config["executors"]
            if item["executor"]["path"].endswith(".ClientAPIExecutor")
        )
        assert executor["args"]["execution_mode"] == "external_process"
        launcher_command = executor["args"]["command"]
        assert launcher_command[: len(command)] == command
        assert launcher_command[len(command)] == "custom/client.py"
        train_args = shlex.split(recipe.per_site_config[entry["client_name"]]["train_args"])
        assert train_args[train_args.index("--sample-count") + 1] == str(entry["sample_count"])
        assert train_args[train_args.index("--train-file") + 1] == str(entry["train_file"])
        assert train_args[train_args.index("--validation-file") + 1] == str(validation_file)
        assert train_args[train_args.index("--mock-delta") + 1] == str(entry["mock_delta"])

    sim_env = job_module.create_sim_env(args, plan)
    assert sim_env.clients == ["site-1", "site-2", "site-3"]
    assert sim_env.num_threads == 1
    assert sim_env.gpu_config == "[0]"

    args.backend = "bionemo"
    assert shlex.split(job_module._build_external_command(args))[-3:] == [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
    ]


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_persistent_training_state_exports_one_private_directory_per_site_and_rejects_mock(tmp_path):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    manifest = job_module._load_manifest(args.manifest)
    plan = job_module._site_plan(args, manifest)
    validation_file = Path(args.data_dir) / "validation.jsonl"

    default_args = shlex.split(
        job_module._build_train_args(
            args,
            train_file=plan[0]["train_file"],
            validation_file=validation_file,
            site_name=plan[0]["client_name"],
            sample_count=plan[0]["sample_count"],
            mock_delta=plan[0]["mock_delta"],
        )
    )
    assert "--training-state-dir" not in default_args

    args.persist_client_training_state = True
    observed_paths = []
    for entry in plan:
        train_args = shlex.split(
            job_module._build_train_args(
                args,
                train_file=entry["train_file"],
                validation_file=validation_file,
                site_name=entry["client_name"],
                sample_count=entry["sample_count"],
                mock_delta=entry["mock_delta"],
            )
        )
        observed_paths.append(train_args[train_args.index("--training-state-dir") + 1])

    expected_root = Path(args.workspace).resolve() / "client_training_state"
    assert observed_paths == [str(expected_root / entry["client_name"]) for entry in plan]
    assert len(set(observed_paths)) == len(plan)
    with pytest.raises(ValueError, match="only by the BioNeMo backend"):
        job_module.validate_inputs(args)


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_direct_round_three_checkpoint_continues_at_logical_round_four(tmp_path):
    import torch

    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    original_checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
    original_initialization = original_checkpoint["meta_props"]
    continuation_signature = _continuation_signature(job_module, args)
    prior_global_checkpoint = tmp_path / "round-3-global.pt"
    torch.save(
        {
            **original_checkpoint,
            "meta_props": {
                "current_round": 3,
                "nr_aggregated": 3,
                "initialization": original_initialization,
                "continuation_signature": continuation_signature,
            },
        },
        prior_global_checkpoint,
    )
    args.initial_checkpoint = str(prior_global_checkpoint)
    args.start_round = 4
    args.num_rounds = 1

    _manifest, plan, validation_file, _training_inputs, observed_signature = job_module.validate_inputs(args)
    with _chdir(_example_dir()):
        recipe = job_module.create_recipe(args, plan, validation_file, observed_signature)
        export_root = tmp_path / "continued-export"
        recipe.export(str(export_root))

    workflow = recipe._job._deploy_map["server"].app_config.workflows[0]
    assert workflow.controller.start_round == 4
    server_config = json.loads(
        (export_root / recipe.name / "app_server" / "config" / "config_fed_server.json").read_text(encoding="utf-8")
    )
    controller = next(item for item in server_config["workflows"] if item["path"].endswith(".FedAvg"))
    assert controller["args"]["start_round"] == 4

    result_dir = tmp_path / "continued-result"
    started_at = time.time() - 1.0
    _write_global_checkpoint(
        job_module,
        result_dir,
        args=args,
        initial_checkpoint=args.initial_checkpoint,
        current_round=4,
        nr_aggregated=len(plan),
    )
    for entry in plan:
        _write_round_metric(Path(args.workspace), entry["client_name"], 4)

    summary = job_module.collect_run_summary(args, plan, result_dir, started_at)

    assert summary["start_round"] == 4
    assert summary["global_checkpoint_round"] == 4
    assert [item["round"] for item in summary["global_round_checkpoints"]] == [4]
    assert {metric["round"] for metric in summary["round_metrics"]} == {4}
    assert summary["initialization_metadata"] == original_initialization
    assert summary["continuation_signature"] == continuation_signature


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
@pytest.mark.parametrize(
    "mutation",
    ("sampler_budget", "learning_rate", "min_learning_rate", "warmup_iters", "mock_delta", "partition", "site_plan"),
)
def test_start_round_rejects_a_different_federation_or_training_protocol(tmp_path, mutation):
    import torch

    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    original_checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
    continuation_signature = _continuation_signature(job_module, args)
    prior_global_checkpoint = tmp_path / "round-3-global.pt"
    torch.save(
        {
            **original_checkpoint,
            "meta_props": {
                "current_round": 3,
                "nr_aggregated": 3,
                "initialization": original_checkpoint["meta_props"],
                "continuation_signature": continuation_signature,
            },
        },
        prior_global_checkpoint,
    )
    args.initial_checkpoint = str(prior_global_checkpoint)
    args.start_round = 4

    if mutation == "sampler_budget":
        args.local_steps += 1
    elif mutation in ("learning_rate", "min_learning_rate"):
        setattr(args, mutation, getattr(args, mutation) / 2)
    elif mutation == "warmup_iters":
        args.warmup_iters += 1
    elif mutation == "mock_delta":
        args.mock_delta *= 2
    else:
        manifest_path = Path(args.manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if mutation == "partition":
            manifest["settings"]["partition"] = "dirichlet"
            manifest["settings"]["dirichlet_alpha"] = 0.5
        else:
            for section in (
                manifest["files"]["sites"],
                manifest["counts"]["sites"],
                manifest["file_identities"]["sites"],
            ):
                section["site-1"], section["site-2"] = section["site-2"], section["site-1"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="signature does not match this federation or training protocol"):
        job_module.validate_inputs(args)


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
@pytest.mark.parametrize("signature_state", ("missing", "tampered"))
def test_start_round_rejects_missing_or_tampered_continuation_signature(tmp_path, signature_state):
    import torch

    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    original_checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
    continuation_signature = _continuation_signature(job_module, args)
    if signature_state == "tampered":
        continuation_signature["payload"]["mode"] = "local"
    metadata = {
        "current_round": 3,
        "nr_aggregated": 3,
        "initialization": original_checkpoint["meta_props"],
    }
    if signature_state != "missing":
        metadata["continuation_signature"] = continuation_signature
    prior_global_checkpoint = tmp_path / "round-3-global.pt"
    torch.save({**original_checkpoint, "meta_props": metadata}, prior_global_checkpoint)
    args.initial_checkpoint = str(prior_global_checkpoint)
    args.start_round = 4

    with pytest.raises(ValueError, match="requires a valid continuation signature"):
        job_module.validate_inputs(args)


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_start_round_validation_rejects_invalid_or_stateful_continuations(tmp_path):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)

    args.start_round = -1
    with pytest.raises(ValueError, match="must be non-negative"):
        job_module.validate_inputs(args)

    args.start_round = 1
    args.backend = "bionemo"
    args.persist_client_training_state = True
    with pytest.raises(ValueError, match="complete site-local state chain"):
        job_module.validate_inputs(args)


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_manifest_and_sequential_execution_validation_fail_loudly(tmp_path):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)

    args.num_threads = 2
    with pytest.raises(ValueError, match="requires --num-threads 1"):
        job_module.validate_inputs(args)

    args.num_threads = 1
    for invalid_gpu in ("0", "[0,1]", "[0],[1]"):
        args.gpu = invalid_gpu
        with pytest.raises(ValueError, match="exactly one GPU"):
            job_module.validate_inputs(args)

    args.gpu = "[0]"
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["audit"]["status"] = "failed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="passed leakage audit"):
        job_module.validate_inputs(args)


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_manifest_row_count_mismatch_is_rejected_before_launch(tmp_path):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    Path(args.data_dir, "train", "site-2.jsonl").write_text('{"label": 1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"site-2 training row count.*expected 30, observed 1"):
        job_module.validate_inputs(args)


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_manifest_content_edit_and_initialization_mismatch_are_rejected_before_launch(tmp_path):
    import torch

    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    site_path = Path(args.data_dir, "train", "site-2.jsonl")
    site_path.write_text('{"label": 1}\n' * 30, encoding="utf-8")
    with pytest.raises(ValueError, match="no longer matches its audited manifest identity"):
        job_module.validate_inputs(args)

    _write_manifest(Path(args.data_dir))
    checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
    checkpoint["meta_props"]["lora_alpha"] = 8
    torch.save(checkpoint, args.initial_checkpoint)
    with pytest.raises(ValueError, match="initialization settings do not match"):
        job_module.validate_inputs(args)

    checkpoint["meta_props"]["lora_alpha"] = 32
    checkpoint["meta_props"]["exchange_dtype"] = "bfloat16"
    torch.save(checkpoint, args.initial_checkpoint)
    with pytest.raises(ValueError, match="exchange_dtype"):
        job_module.validate_inputs(args)

    checkpoint["meta_props"].pop("exchange_dtype")
    torch.save(checkpoint, args.initial_checkpoint)
    with pytest.raises(ValueError, match="exchange_dtype"):
        job_module.validate_inputs(args)


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_collect_run_summary_validates_fresh_global_checkpoint_and_aggregates_artifacts(tmp_path, monkeypatch):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    manifest = job_module._load_manifest(args.manifest)
    plan = job_module._site_plan(args, manifest)
    result_dir = tmp_path / "simulator result"
    started_at = time.time() - 1.0
    global_checkpoint = _write_global_checkpoint(
        job_module,
        result_dir,
        args=args,
        initial_checkpoint=args.initial_checkpoint,
    )

    expected_runtime = 0.0
    expected_received = 0.0
    expected_sent = 0.0
    expected_peak = 0.0
    for site_index, entry in enumerate(plan, start=1):
        for round_index in range(args.num_rounds):
            scale = float(site_index + round_index)
            _write_round_metric(Path(args.workspace), entry["client_name"], round_index, scale=scale)
            expected_runtime += 2.0 * scale
            expected_received += 3.0 * scale
            expected_sent += scale
            expected_peak = max(expected_peak, 100.0 * scale)

    finite_validation_labels = []
    original_validate_finite = job_module.adapter_checkpoint._validate_finite_tensors

    def track_finite_validation(tensors, label):
        finite_validation_labels.append(label)
        original_validate_finite(tensors, label)

    monkeypatch.setattr(job_module.adapter_checkpoint, "_validate_finite_tensors", track_finite_validation)
    summary = job_module.collect_run_summary(args, plan, result_dir, started_at)
    saved_summary = json.loads(Path(summary["summary_path"]).read_text(encoding="utf-8"))

    assert summary["global_checkpoint"] == str(global_checkpoint.resolve())
    assert len(summary["global_checkpoint_sha256"]) == 64
    assert summary["global_checkpoint_round"] == args.num_rounds - 1
    assert summary["global_checkpoint_contributors"] == len(plan)
    assert [entry["round"] for entry in summary["global_round_checkpoints"]] == [0, 1]
    assert all(Path(entry["path"]).is_file() for entry in summary["global_round_checkpoints"])
    assert all(len(entry["sha256"]) == 64 for entry in summary["global_round_checkpoints"])
    assert len(summary["initial_checkpoint_sha256"]) == 64
    assert summary["initialization_metadata"]["backend"] == "mock"
    assert summary["initialization_metadata"]["exchange_dtype"] == "float32"
    assert summary["exchange_dtype"] == "float32"
    assert summary["aggregation_weights"] == {"site-1": 10, "site-2": 30, "site-3": 60}
    assert summary["dataset"]["source"] == {"dataset_id": "example/test", "revision": "abc123"}
    assert summary["training_inputs"]["train_files"]["site-1"]["rows"] == 10
    assert summary["training_inputs"]["validation_file"]["rows"] == 10
    assert summary["training_inputs"]["base_checkpoint"] is None
    assert summary["training_inputs"]["classifier_file"] is None
    assert summary["configuration"]["seed"] == 1234
    assert summary["configuration"]["lora_dim"] == 16
    assert len(summary["round_metrics"]) == len(plan) * args.num_rounds
    assert summary["total_client_runtime_seconds"] == pytest.approx(expected_runtime)
    assert summary["peak_client_gpu_memory_mebibytes"] == pytest.approx(expected_peak)
    assert summary["total_received_mebibytes"] == pytest.approx(expected_received)
    assert summary["total_sent_mebibytes"] == pytest.approx(expected_sent)
    assert all(Path(metric["path"]).is_file() for metric in summary["round_metrics"])
    assert all(len(metric["local_checkpoint_sha256"]) == 64 for metric in summary["round_metrics"])
    assert finite_validation_labels.count("Reference trainable state") == 1
    assert finite_validation_labels.count("NVFlare checkpoint model state") == 10
    assert saved_summary == summary


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
@pytest.mark.parametrize("checkpoint_state", ["missing", "stale"])
def test_collect_run_summary_rejects_missing_or_stale_global_checkpoint(tmp_path, checkpoint_state):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    manifest = job_module._load_manifest(args.manifest)
    plan = job_module._site_plan(args, manifest)
    result_dir = tmp_path / "simulator-result"
    started_at = time.time()

    if checkpoint_state == "stale":
        checkpoint = _write_global_checkpoint(
            job_module,
            result_dir,
            args=args,
            initial_checkpoint=args.initial_checkpoint,
        )
        os.utime(checkpoint, (started_at - 10.0, started_at - 10.0))

    with pytest.raises(RuntimeError, match="fresh global checkpoint"):
        job_module.collect_run_summary(args, plan, result_dir, started_at)


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_collect_run_summary_rejects_a_fresh_checkpoint_from_a_prior_round(tmp_path):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    manifest = job_module._load_manifest(args.manifest)
    plan = job_module._site_plan(args, manifest)
    result_dir = tmp_path / "simulator-result"
    started_at = time.time() - 1.0
    _write_global_checkpoint(
        job_module,
        result_dir,
        args=args,
        initial_checkpoint=args.initial_checkpoint,
        current_round=args.num_rounds - 2,
    )

    with pytest.raises(RuntimeError, match=r"meta_props\.current_round=1, received 0"):
        job_module.collect_run_summary(args, plan, result_dir, started_at)


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_collect_run_summary_rejects_incomplete_checkpoint_aggregation_metadata(tmp_path):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    manifest = job_module._load_manifest(args.manifest)
    plan = job_module._site_plan(args, manifest)
    result_dir = tmp_path / "simulator-result"
    started_at = time.time() - 1.0
    _write_global_checkpoint(
        job_module,
        result_dir,
        args=args,
        initial_checkpoint=args.initial_checkpoint,
        nr_aggregated=len(plan) - 1,
    )

    with pytest.raises(RuntimeError, match=r"meta_props\.nr_aggregated=3, received 2"):
        job_module.collect_run_summary(args, plan, result_dir, started_at)


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_collect_run_summary_rejects_incompatible_global_checkpoint_schema(tmp_path):
    import torch

    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    manifest = job_module._load_manifest(args.manifest)
    plan = job_module._site_plan(args, manifest)
    result_dir = tmp_path / "simulator-result"
    started_at = time.time() - 1.0
    checkpoint_path = _write_global_checkpoint(
        job_module,
        result_dir,
        args=args,
        initial_checkpoint=args.initial_checkpoint,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["model"].pop("decoder.classification_head.bias")
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match="Final global trainable checkpoint.*invalid"):
        job_module.collect_run_summary(args, plan, result_dir, started_at)


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_collect_run_summary_rejects_missing_round_checkpoint(tmp_path):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    manifest = job_module._load_manifest(args.manifest)
    plan = job_module._site_plan(args, manifest)
    result_dir = tmp_path / "simulator-result"
    started_at = time.time() - 1.0
    global_checkpoint = _write_global_checkpoint(
        job_module,
        result_dir,
        args=args,
        initial_checkpoint=args.initial_checkpoint,
    )
    global_checkpoint.with_name(f"{global_checkpoint.stem}_round_000{global_checkpoint.suffix}").unlink()

    with pytest.raises(RuntimeError, match="fresh round 0 checkpoint"):
        job_module.collect_run_summary(args, plan, result_dir, started_at)


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
@pytest.mark.parametrize("metric_state", ["missing", "stale"])
def test_collect_run_summary_rejects_missing_or_stale_round_metrics(tmp_path, metric_state):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    manifest = job_module._load_manifest(args.manifest)
    plan = job_module._site_plan(args, manifest)
    result_dir = tmp_path / "simulator-result"
    started_at = time.time() - 1.0
    _write_global_checkpoint(
        job_module,
        result_dir,
        args=args,
        initial_checkpoint=args.initial_checkpoint,
    )

    metric_paths = []
    for entry in plan:
        for round_index in range(args.num_rounds):
            metric_paths.append(_write_round_metric(Path(args.workspace), entry["client_name"], round_index))
    if metric_state == "missing":
        metric_paths[-1].unlink()
    else:
        os.utime(metric_paths[-1], (started_at - 10.0, started_at - 10.0))

    with pytest.raises(RuntimeError, match="Expected 6 fresh client metric files, found 5"):
        job_module.collect_run_summary(args, plan, result_dir, started_at)


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_collect_run_summary_rejects_duplicate_and_missing_metric_pairs_when_count_matches(tmp_path):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    manifest = job_module._load_manifest(args.manifest)
    plan = job_module._site_plan(args, manifest)
    result_dir = tmp_path / "simulator-result"
    started_at = time.time() - 1.0
    _write_global_checkpoint(
        job_module,
        result_dir,
        args=args,
        initial_checkpoint=args.initial_checkpoint,
    )

    metric_paths = []
    for entry in plan:
        for round_index in range(args.num_rounds):
            metric_paths.append(_write_round_metric(Path(args.workspace), entry["client_name"], round_index))
    repeated_metric_path = metric_paths[-1]
    repeated_metric = json.loads(repeated_metric_path.read_text(encoding="utf-8"))
    repeated_metric["round"] = 0
    repeated_metric_path.write_text(json.dumps(repeated_metric), encoding="utf-8")
    checkpoint = __import__("torch").load(repeated_metric["local_checkpoint"], map_location="cpu", weights_only=True)
    checkpoint["meta_props"]["round"] = 0
    __import__("torch").save(checkpoint, repeated_metric["local_checkpoint"])

    with pytest.raises(RuntimeError, match=r"duplicates=.*\('site-3', 0\).*missing=.*\('site-3', 1\)"):
        job_module.collect_run_summary(args, plan, result_dir, started_at)


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_collect_run_summary_rejects_corrupt_or_substituted_local_checkpoint(tmp_path):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    manifest = job_module._load_manifest(args.manifest)
    plan = job_module._site_plan(args, manifest)
    result_dir = tmp_path / "simulator-result"
    started_at = time.time() - 1.0
    _write_global_checkpoint(
        job_module,
        result_dir,
        args=args,
        initial_checkpoint=args.initial_checkpoint,
    )
    paths = [
        _write_round_metric(Path(args.workspace), entry["client_name"], round_index)
        for entry in plan
        for round_index in range(args.num_rounds)
    ]

    bad_metric = json.loads(paths[-1].read_text(encoding="utf-8"))
    Path(bad_metric["local_checkpoint"]).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="Local trainable checkpoint.*invalid"):
        job_module.collect_run_summary(args, plan, result_dir, started_at)

    _write_round_metric(Path(args.workspace), "site-3", 1)
    checkpoint_path = Path(bad_metric["local_checkpoint"])
    checkpoint = __import__("torch").load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["meta_props"]["round"] = 0
    __import__("torch").save(checkpoint, checkpoint_path)
    with pytest.raises(RuntimeError, match="metadata does not match"):
        job_module.collect_run_summary(args, plan, result_dir, started_at)
