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
                "decoder.layers.0.linear_qkv.adapter.linear_in.weight": torch.ones(2, 4),
                "decoder.classification_head.weight": torch.ones(3, 4),
                "decoder.classification_head.bias": torch.ones(3),
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


def _write_final_global_checkpoint(
    job_module,
    result_dir: Path,
    *,
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
    torch.save(
        {
            "model": {
                "decoder.layers.0.linear_qkv.adapter.linear_in.weight": torch.zeros(2, 4),
                "decoder.classification_head.weight": torch.zeros(3, 4),
                "decoder.classification_head.bias": torch.zeros(3),
            },
            "meta_props": {
                "current_round": current_round,
                "nr_aggregated": nr_aggregated,
                "initialization": initialization,
            },
        },
        checkpoint,
    )
    return checkpoint


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_site_plan_uses_manifest_counts_for_three_fedavg_clients(tmp_path):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    manifest = job_module._load_manifest(args.manifest)

    plan = job_module._site_plan(args, manifest)

    assert [entry["client_name"] for entry in plan] == ["site-1", "site-2", "site-3"]
    assert [entry["sample_count"] for entry in plan] == [10, 30, 60]
    assert [entry["mock_delta"] for entry in plan] == pytest.approx([0.01, 0.02, 0.03])
    assert all(entry["train_file"].is_file() for entry in plan)
    assert [entry["train_identity"]["rows"] for entry in plan] == [10, 30, 60]
    assert all(len(entry["train_identity"]["sha256"]) == 64 for entry in plan)


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_recipe_carries_sample_weights_and_shell_safe_site_arguments(tmp_path):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    _manifest, plan, validation_file, training_inputs = job_module.validate_inputs(args)

    with _chdir(_example_dir()):
        recipe = job_module.create_recipe(args, plan, validation_file)
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
    assert recipe.aggregator.schema_checkpoint == str(Path(args.initial_checkpoint).resolve())
    assert training_inputs["train_files"]["site-1"]["rows"] == 10

    exported_job = export_root / recipe.name
    server_config = json.loads(
        (exported_job / "app_server" / "config" / "config_fed_server.json").read_text(encoding="utf-8")
    )
    controller = next(workflow for workflow in server_config["workflows"] if workflow["path"].endswith(".FedAvg"))
    exported_aggregator = controller["args"]["aggregator"]
    assert exported_aggregator["path"] == "evo2_aggregator.ExactSchemaFedAvgAggregator"
    assert exported_aggregator["args"]["aggregation_weights"] == recipe.aggregation_weights
    persistor = next(component for component in server_config["components"] if component["id"] == "persistor")
    assert persistor["path"] == "evo2_persistor.CPUTrainablePTFileModelPersistor"
    assert "load_device" not in persistor["args"]
    assert (exported_job / "app_server" / "custom" / "evo2_aggregator.py").is_file()
    assert (exported_job / "app_server" / "custom" / "evo2_adapter_checkpoint.py").is_file()
    assert (exported_job / "app_server" / "custom" / "evo2_persistor.py").is_file()
    assert (exported_job / "app_server" / "custom" / "provenance.py").is_file()

    for entry in plan:
        client_app = exported_job / f"app_{entry['client_name']}"
        assert (client_app / "custom" / "evo2_adapter_checkpoint.py").is_file()
        assert (client_app / "custom" / "sequential_launcher.py").is_file()
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
def test_two_site_one_round_mock_main_writes_a_reloadable_final_checkpoint_summary(tmp_path, monkeypatch):
    job_module = _load_job_module()
    fixture_args = _args(job_module, tmp_path)
    workspace = tmp_path / "smoke workspace"
    result_dir = workspace / "simulator-result"

    class MockRun:
        def get_result(self):
            return result_dir

    class MockRecipe:
        def __init__(self, run_args, plan):
            self.run_args = run_args
            self.plan = plan

        def execute(self, sim_env):
            assert [entry["client_name"] for entry in self.plan] == ["site-1", "site-2"]
            assert sim_env.clients == ["site-1", "site-2"]
            assert sim_env.num_threads == 1
            _write_final_global_checkpoint(
                job_module,
                result_dir,
                initial_checkpoint=self.run_args.initial_checkpoint,
                current_round=0,
                nr_aggregated=2,
            )
            return MockRun()

    monkeypatch.setattr(
        job_module,
        "create_recipe",
        lambda run_args, plan, _validation_file: MockRecipe(run_args, plan),
    )
    job_module.main(
        [
            "--data-dir",
            fixture_args.data_dir,
            "--manifest",
            fixture_args.manifest,
            "--initial-checkpoint",
            fixture_args.initial_checkpoint,
            "--workspace",
            str(workspace),
            "--backend",
            "mock",
            "--num-clients",
            "2",
            "--num-rounds",
            "1",
        ]
    )

    summary = json.loads((workspace / "run_summary.json").read_text(encoding="utf-8"))
    global_checkpoint = Path(summary["global_checkpoint"])
    final_state = job_module.adapter_checkpoint.load_nvflare_checkpoint(global_checkpoint)
    assert summary["global_checkpoint_sha256"] == job_module.provenance.sha256_file(global_checkpoint)
    assert summary["global_checkpoint_round"] == 0
    assert summary["global_checkpoint_contributors"] == 2
    assert summary["trainable_tensors_changed"] == len(final_state)
    assert summary["aggregation_weights"] == {"site-1": 10, "site-2": 30}
    assert set(summary["training_inputs"]["train_files"]) == {"site-1", "site-2"}
    assert summary["configuration"]["seed"] == 1234
    assert not list(global_checkpoint.parent.glob("FL_global_model_round_*.pt"))


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_run_summary_rejects_a_final_checkpoint_identical_to_initialization(tmp_path):
    import torch

    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    manifest, plan, validation_file, training_inputs = job_module.validate_inputs(args)
    result_dir = tmp_path / "unchanged-result"
    global_checkpoint = _write_final_global_checkpoint(
        job_module,
        result_dir,
        initial_checkpoint=args.initial_checkpoint,
    )
    final_payload = torch.load(global_checkpoint, map_location="cpu", weights_only=True)
    initial_payload = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
    final_payload["model"] = initial_payload["model"]
    torch.save(final_payload, global_checkpoint)

    with pytest.raises(RuntimeError, match="identical to the common initialization"):
        job_module.collect_run_summary(
            args,
            plan,
            result_dir,
            0.0,
            manifest,
            training_inputs,
        )


@pytest.mark.skipif(not HAS_NVFLARE_RUNTIME_DEPS, reason="NVFlare runtime dependencies are required")
def test_manifest_and_single_gpu_federation_validation_fail_loudly(tmp_path):
    job_module = _load_job_module()
    args = _args(job_module, tmp_path)

    args.num_clients = 1
    with pytest.raises(ValueError, match="at least two clients"):
        job_module.validate_inputs(args)

    args.num_clients = 3
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
@pytest.mark.parametrize("mismatch_source", ("site", "stale"))
def test_bionemo_initialization_data_is_bound_to_the_manifest_validation_split(tmp_path, mismatch_source):
    import torch

    job_module = _load_job_module()
    args = _args(job_module, tmp_path)
    data_dir = Path(args.data_dir)
    validation_file = data_dir / "validation.jsonl"
    base_checkpoint = tmp_path / "base checkpoint"
    base_checkpoint.mkdir()
    (base_checkpoint / "weights.distcp").write_bytes(b"base weights")
    classifier_file = tmp_path / "evo2 classifier.py"
    classifier_file.write_text("# pinned classifier fixture\n", encoding="utf-8")

    args.backend = "bionemo"
    args.base_checkpoint = str(base_checkpoint)
    args.classifier_file = str(classifier_file)
    checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
    validation_identity = job_module.provenance.jsonl_identity(validation_file)
    checkpoint["meta_props"].update(
        {
            "backend": "bionemo",
            "base_checkpoint": str(base_checkpoint.resolve()),
            "training_inputs": {
                "data_file": validation_identity,
                "base_checkpoint": job_module.provenance.directory_identity(base_checkpoint),
                "classifier_file": job_module.provenance.file_identity(classifier_file),
            },
        }
    )
    torch.save(checkpoint, args.initial_checkpoint)

    _manifest, _plan, observed_validation_file, _training_inputs = job_module.validate_inputs(args)
    assert observed_validation_file == validation_file.resolve()

    if mismatch_source == "site":
        mismatched_identity = job_module.provenance.jsonl_identity(data_dir / "train" / "site-2.jsonl")
    else:
        stale_validation = tmp_path / "stale validation.jsonl"
        stale_validation.write_text('{"label": 2}\n' * 10, encoding="utf-8")
        mismatched_identity = job_module.provenance.jsonl_identity(stale_validation)
    checkpoint["meta_props"]["training_inputs"]["data_file"] = mismatched_identity
    torch.save(checkpoint, args.initial_checkpoint)

    with pytest.raises(ValueError, match="data_file content identity does not match this run"):
        job_module.validate_inputs(args)
