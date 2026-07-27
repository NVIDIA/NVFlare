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

import importlib.util
import json
import os
import shlex
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None


def _example_dir():
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "integration",
            "nemo",
            "examples",
            "supervised_fine_tuning",
        )
    )


def _load_example_module(module_name: str):
    example_dir = _example_dir()
    sys.path.insert(0, example_dir)
    try:
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(example_dir, f"{module_name}.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(example_dir)


@contextmanager
def _chdir(path):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _args(tmp_path, initial_model_ckpt):
    return SimpleNamespace(
        n_clients=2,
        num_rounds=1,
        num_threads=1,
        gpu=None,
        workspace=str(tmp_path / "workspace"),
        initial_model_ckpt=str(initial_model_ckpt),
        model_name_or_path="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        data_dir=str(tmp_path / "sft data"),
        train_files=None,
        validation_file=str(tmp_path / "validation file.jsonl"),
        backend="mock",
        client_command=None,
        automodel_command="automodel",
        automodel_config_template=None,
        automodel_extra_args="",
        nproc_per_node=1,
        max_steps=1,
        seq_length=128,
        limit_train_samples=1,
        limit_validation_samples=1,
        use_chat_template=False,
        learning_rate=1e-5,
        micro_batch_size=1,
        global_batch_size=1,
        gradient_accumulation_steps=1,
        tp_size=1,
        cp_size=1,
        server_tensor_device="cpu",
        mock_delta=0.5,
    )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for model checkpoint tests")
def test_sft_model_checkpoint_round_trip(tmp_path):
    import torch

    model_checkpoint = _load_example_module("model_checkpoint")
    ckpt_path = tmp_path / "model.pt"
    state = {
        "model.embed.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "model.norm.weight": torch.ones(3, dtype=torch.bfloat16),
    }

    model_checkpoint.save_nvflare_model_checkpoint(state, str(ckpt_path))
    loaded = model_checkpoint.load_model_state(str(ckpt_path))

    assert loaded.keys() == state.keys()
    for key, value in state.items():
        assert torch.equal(loaded[key], value.cpu())


@pytest.mark.skipif(not (HAS_TORCH and HAS_SAFETENSORS), reason="PyTorch and safetensors are required")
def test_sft_model_checkpoint_loads_nested_automodel_consolidated_dir(tmp_path):
    import torch
    from safetensors.torch import save_file

    client_module = _load_example_module("automodel_sft_client")
    model_checkpoint = _load_example_module("model_checkpoint")
    checkpoint_dir = tmp_path / "checkpoints"
    epoch_dir = checkpoint_dir / "epoch_0_step_0"
    model_dir = epoch_dir / "model"
    consolidated_dir = model_dir / "consolidated"
    model_dir.mkdir(parents=True)
    consolidated_dir.mkdir(parents=True)
    torch.save({"not_model": torch.zeros(1)}, epoch_dir / "step_scheduler.pt")
    save_file({"model.weight": torch.ones(2, 2)}, consolidated_dir / "model-00001-of-00001.safetensors")

    assert client_module._latest_model_dir(str(checkpoint_dir)) == str(consolidated_dir)
    loaded = model_checkpoint.load_model_state(str(epoch_dir))

    assert torch.equal(loaded["model.weight"], torch.ones(2, 2))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required to import the checkpoint helper")
def test_sft_model_checkpoint_state_files_are_deduplicated(tmp_path):
    model_checkpoint = _load_example_module("model_checkpoint")
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    for name in (
        "model.safetensors",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "pytorch_model.bin",
        "pytorch_model-00001-of-00002.bin",
    ):
        (checkpoint_dir / name).write_text("")

    files = model_checkpoint._state_files_in_dir(str(checkpoint_dir))

    assert len(files) == len(set(files))
    assert files.count(str(checkpoint_dir / "model.safetensors")) == 1
    assert files.count(str(checkpoint_dir / "model-00001-of-00002.safetensors")) == 1
    assert files.count(str(checkpoint_dir / "pytorch_model.bin")) == 1


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for model state coverage checks")
def test_sft_model_state_coverage_allows_extra_candidate_tensors():
    import torch

    model_checkpoint = _load_example_module("model_checkpoint")
    incoming_state = {
        "model.embed.weight": torch.zeros((2, 2)),
        "model.norm.weight": torch.ones((2,)),
    }
    candidate_state = {
        **incoming_state,
        "model.extra.weight": torch.ones((3, 3)),
    }

    model_checkpoint.validate_model_state_coverage(candidate_state, incoming_state)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for model state coverage checks")
def test_sft_model_state_coverage_rejects_missing_incoming_tensor():
    import torch

    model_checkpoint = _load_example_module("model_checkpoint")
    incoming_state = {
        "model.embed.weight": torch.zeros((2, 2)),
        "model.norm.weight": torch.ones((2,)),
    }
    candidate_state = {"model.embed.weight": torch.zeros((2, 2))}

    with pytest.raises(RuntimeError) as exc_info:
        model_checkpoint.validate_model_state_coverage(
            candidate_state,
            incoming_state,
            candidate_name="updated checkpoint",
            reference_name="incoming global model",
        )

    message = str(exc_info.value)
    assert "matched 1/2 tensors" in message
    assert "model.norm.weight" in message
    assert "updated checkpoint" in message
    assert "incoming global model" in message


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for model state coverage checks")
def test_sft_model_state_coverage_rejects_shape_mismatch():
    import torch

    model_checkpoint = _load_example_module("model_checkpoint")
    incoming_state = {
        "model.embed.weight": torch.zeros((2, 2)),
        "model.norm.weight": torch.ones((2,)),
    }
    candidate_state = {
        "model.embed.weight": torch.zeros((2, 2)),
        "model.norm.weight": torch.ones((3,)),
    }

    with pytest.raises(RuntimeError) as exc_info:
        model_checkpoint.validate_model_state_coverage(candidate_state, incoming_state)

    message = str(exc_info.value)
    assert "matched 1/2 tensors" in message
    assert "model.norm.weight" in message
    assert "candidate (3,)" in message
    assert "reference (2,)" in message


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for model state coverage checks")
def test_sft_model_state_matching_preserves_full_incoming_key_set():
    import torch

    model_checkpoint = _load_example_module("model_checkpoint")
    incoming_state = {
        "model.embed.weight": torch.zeros((2, 2)),
        "model.norm.weight": torch.ones((2,)),
    }
    updated_state = {
        "model.extra.weight": torch.ones((3, 3)),
        "model.embed.weight": torch.full((2, 2), 2.0),
        "model.norm.weight": torch.full((2,), 3.0),
    }

    matched = model_checkpoint.match_model_state_to_reference(updated_state, incoming_state, require_all=True)

    assert list(matched) == list(incoming_state)
    assert torch.equal(matched["model.embed.weight"], torch.full((2, 2), 2.0))
    assert torch.equal(matched["model.norm.weight"], torch.full((2,), 3.0))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required to import the AutoModel client helper")
def test_sft_latest_model_dir_prefers_step_over_epoch(tmp_path):
    client_module = _load_example_module("automodel_sft_client")
    checkpoint_dir = tmp_path / "checkpoints"
    step_1 = checkpoint_dir / "epoch_0_step_1"
    step_20 = checkpoint_dir / "epoch_0_step_20"
    for model_dir in (step_1, step_20):
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").write_text("")
        os.utime(model_dir, (1000, 1000))

    assert client_module._latest_model_dir(str(checkpoint_dir)) == str(step_20)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required to create a PT model checkpoint")
def test_sft_recipe_exports_modern_full_transfer_config(tmp_path):
    import torch

    model_checkpoint = _load_example_module("model_checkpoint")
    job_module = _load_example_module("job")
    initial_model_ckpt = tmp_path / "init_model.pt"
    model_checkpoint.save_nvflare_model_checkpoint({"model.embed.weight": torch.zeros((2, 2))}, str(initial_model_ckpt))
    args = _args(tmp_path, initial_model_ckpt)
    data_dir = tmp_path / "sft data"
    data_dir.mkdir()
    for site_idx in (1, 2):
        (data_dir / f"site-{site_idx}_train.jsonl").write_text('{"input":"x","output":"y"}\n')

    with _chdir(_example_dir()):
        recipe = job_module.create_recipe(args)
        recipe.export(str(tmp_path / "exported"))

    assert recipe.launch_external_process is True
    assert recipe.launch_once is False
    assert recipe.server_expected_format == "pytorch"
    assert recipe.params_transfer_type == "FULL"

    job_dir = tmp_path / "exported" / "nemotron3-nano-sft"
    with open(job_dir / "app_site-1" / "config" / "config_fed_client.json") as f:
        client_config = json.load(f)
    with open(job_dir / "app_server" / "config" / "config_fed_server.json") as f:
        server_config = json.load(f)

    executor = next(
        entry["executor"]
        for entry in client_config["executors"]
        if entry["executor"]["path"].endswith(".ClientAPIExecutor")
    )
    executor_args = executor["args"]
    assert executor_args["execution_mode"] == "external_process"
    command = executor_args["command"]
    assert "custom/automodel_sft_client.py" in command
    assert command[command.index("--backend") + 1] == "mock"
    assert command[command.index("--train_file") + 1].endswith("sft data/site-1_train.jsonl")
    assert executor_args["launch_once"] is False
    assert (job_dir / "app_site-1" / "custom" / "automodel_sft_dataset.py").exists()
    assert (job_dir / "app_site-1" / "custom" / "automodel_full_model_loader.py").exists()
    assert (job_dir / "app_site-1" / "custom" / "model_checkpoint.py").exists()

    controller = server_config["workflows"][0]
    assert controller["path"] == "nvflare.app_common.workflows.fedavg.FedAvg"
    assert controller["args"]["num_clients"] == 2
    persistor = next(
        c
        for c in server_config["components"]
        if c["path"] == "nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor"
    )
    assert persistor["args"]["load_device"] == "cpu"


def test_sft_train_args_are_shell_safe(tmp_path):
    job_module = _load_example_module("job")
    args = _args(tmp_path, tmp_path / "init_model.pt")

    train_args = job_module._build_train_args(args, str(tmp_path / "sft data" / "site-1_train.jsonl"), "site-1")

    parts = shlex.split(train_args)
    assert parts[parts.index("--backend") + 1] == "mock"
    assert parts[parts.index("--train_file") + 1].endswith("sft data/site-1_train.jsonl")
    assert parts[parts.index("--work_dir") + 1].endswith("workspace/automodel_work/site-1")
    assert "--no-use_chat_template" in parts


def test_sft_automodel_config_uses_full_model_helpers(tmp_path):
    client_module = _load_example_module("automodel_sft_client")
    args = _args(tmp_path, tmp_path / "init_model.pt")
    args.backend = "automodel"
    args.train_file = str(tmp_path / "train.jsonl")
    incoming_model_ckpt = str(tmp_path / "incoming_global_model.pt")

    config = client_module._default_automodel_config(args, str(tmp_path / "checkpoints"), incoming_model_ckpt)

    expected_dataset_suffix = os.path.join("supervised_fine_tuning", "automodel_sft_dataset.py")
    assert config["dataset"]["_target_"].endswith(f"{expected_dataset_suffix}:make_instruction_dataset")
    assert config["validation_dataset"]["_target_"] == config["dataset"]["_target_"]
    expected_model_suffix = os.path.join("supervised_fine_tuning", "automodel_full_model_loader.py")
    assert config["model"]["_target_"].endswith(f"{expected_model_suffix}:from_pretrained_with_global_state")
    assert config["model"]["incoming_model_ckpt"] == incoming_model_ckpt
    assert "peft" not in config


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for multi-round aggregation checks")
def test_sft_full_model_fedavg_is_cumulative_across_rounds():
    import torch

    from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
    from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
    from nvflare.app_common.utils.fl_model_utils import FLModelUtils

    model = FLModel(
        params_type=ParamsType.FULL,
        params={
            "model.embed.weight": torch.zeros((2, 2)),
            "model.norm.weight": torch.ones((2,)),
        },
    )

    for round_idx in range(3):
        helper = WeightedAggregationHelper()
        for site_name in ("site-1", "site-2"):
            helper.add(
                data={key: value + 0.5 for key, value in model.params.items()},
                weight=1.0,
                contributor_name=site_name,
                contribution_round=round_idx,
            )
        model_update = FLModel(params_type=ParamsType.FULL, params=helper.get_result())
        model = FLModelUtils.update_model(model, model_update)

    assert torch.equal(model.params["model.embed.weight"], torch.full((2, 2), 1.5))
    assert torch.equal(model.params["model.norm.weight"], torch.full((2,), 2.5))


def test_sft_dataset_prompt_format():
    dataset_module = _load_example_module("automodel_sft_dataset")

    assert dataset_module.build_prompt("Explain FL.") == "### Instruction:\nExplain FL.\n\n### Response:\n"
    assert dataset_module.clean_response("  A short answer.  ") == "A short answer."


def test_sft_dataset_uses_local_pad_token_fallback(monkeypatch):
    dataset_module = _load_example_module("automodel_sft_dataset")

    class FakeDataset(list):
        @property
        def column_names(self):
            return ["input", "output"]

        def select(self, indices):
            return FakeDataset([self[index] for index in indices])

        def map(self, fn, batched=False, remove_columns=None):
            return [fn(example) for example in self]

    datasets_module = ModuleType("datasets")
    datasets_module.load_dataset = lambda *args, **kwargs: FakeDataset([{"input": "Explain FL.", "output": "Done."}])
    formatting_module = ModuleType("formatting_utils")
    formatting_module.format_chat_template = lambda **kwargs: kwargs
    formatting_module.format_prompt_completion = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "nemo_automodel", ModuleType("nemo_automodel"))
    monkeypatch.setitem(sys.modules, "nemo_automodel.components", ModuleType("components"))
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.datasets", ModuleType("datasets"))
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.datasets.llm", ModuleType("llm"))
    monkeypatch.setitem(
        sys.modules,
        "nemo_automodel.components.datasets.llm.formatting_utils",
        formatting_module,
    )

    tokenizer = SimpleNamespace(eos_token="<eos>", eos_token_id=2, pad_token=None, pad_token_id=None)

    result = dataset_module.make_instruction_dataset(tokenizer, "train.jsonl", fp8=True)

    assert tokenizer.pad_token == "<eos>"
    assert result[0]["pad_token_id"] == 2


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for prediction helper tests")
def test_sft_predict_uses_no_cache_greedy_loop():
    import torch

    predict_module = _load_example_module("predict")

    class FakeModel:
        def __init__(self):
            self.calls = []

        def __call__(self, input_ids, attention_mask, use_cache):
            self.calls.append({"input_ids": input_ids.clone(), "use_cache": use_cache})
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
            logits[:, -1, 2] = 1.0
            return SimpleNamespace(logits=logits)

    model = FakeModel()
    generated = predict_module._greedy_generate_no_cache(model, torch.tensor([[0, 1]]), max_new_tokens=2)

    assert generated.tolist() == [[0, 1, 2, 2]]
    assert [call["use_cache"] for call in model.calls] == [False, False]


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required to import prediction helpers")
def test_sft_predict_creates_output_parent_dir(tmp_path):
    predict_module = _load_example_module("predict")
    output_json = tmp_path / "missing" / "nested" / "predictions.json"

    predict_module._write_predictions(str(output_json), "checkpoint.pt", [{"prompt": "p", "generated": "g"}])

    with open(output_json) as f:
        data = json.load(f)

    assert data == {
        "checkpoint": "checkpoint.pt",
        "predictions": [{"prompt": "p", "generated": "g"}],
    }


def test_sft_synthetic_data_generator_writes_site_and_validation_files(tmp_path):
    data_module = _load_example_module("data/create_synthetic_sft_data")
    out_dir = tmp_path / "synthetic"

    for site_name, examples in data_module.SITE_EXAMPLES.items():
        data_module._write_jsonl(str(out_dir / f"{site_name}_train.jsonl"), examples[:1])
    data_module._write_jsonl(str(out_dir / "validation.jsonl"), data_module.VALIDATION_EXAMPLES[:1])

    assert (out_dir / "site-1_train.jsonl").is_file()
    with open(out_dir / "site-1_train.jsonl") as f:
        record = json.loads(f.readline())
    assert set(record) == {"input", "output"}
    assert (out_dir / "validation.jsonl").is_file()
