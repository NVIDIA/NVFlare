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
import inspect
import json
import os
import shlex
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _example_dir():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "integration", "nemo", "examples", "peft")
    )


def _load_job_module():
    example_dir = _example_dir()
    spec = importlib.util.spec_from_file_location("nemo_peft_job", os.path.join(example_dir, "job.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_client_module():
    example_dir = _example_dir()
    sys.path.insert(0, example_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            "nemo_peft_automodel_client", os.path.join(example_dir, "automodel_peft_client.py")
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(example_dir)


def _load_predict_module():
    example_dir = _example_dir()
    sys.path.insert(0, example_dir)
    try:
        spec = importlib.util.spec_from_file_location(
            "nemo_peft_predict_sentiment", os.path.join(example_dir, "predict_sentiment.py")
        )
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


def _args(tmp_path, initial_adapter_ckpt):
    return SimpleNamespace(
        n_clients=2,
        num_rounds=1,
        num_threads=1,
        gpu=None,
        workspace=str(tmp_path / "workspace"),
        initial_adapter_ckpt=str(initial_adapter_ckpt),
        model_name_or_path="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        train_split_dir=str(tmp_path / "split data"),
        validation_file=str(tmp_path / "validation file.jsonl"),
        alpha=10.0,
        backend="mock",
        automodel_command="automodel",
        automodel_config_template=None,
        automodel_extra_args="",
        nproc_per_node=1,
        max_steps=1,
        seq_length=128,
        limit_train_samples=None,
        limit_validation_samples=4,
        balance_train_labels=True,
        use_chat_template=False,
        learning_rate=2e-4,
        micro_batch_size=1,
        global_batch_size=1,
        gradient_accumulation_steps=1,
        lora_rank=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules="*.proj",
        tp_size=1,
        cp_size=1,
        use_triton_lora=False,
        server_tensor_device="cpu",
        mock_delta=0.01,
    )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required to create a PT adapter checkpoint")
def test_nemo_peft_recipe_exports_modern_fedavg_config(tmp_path):
    import torch

    sys.path.insert(0, _example_dir())
    try:
        import adapter_checkpoint
    finally:
        sys.path.remove(_example_dir())

    initial_adapter_ckpt = tmp_path / "init_adapter.pt"
    adapter_checkpoint.save_nvflare_adapter_checkpoint(
        {"model.layer.lora_A.weight": torch.zeros((2, 2))},
        str(initial_adapter_ckpt),
    )
    args = _args(tmp_path, initial_adapter_ckpt)
    job_module = _load_job_module()

    with _chdir(_example_dir()):
        recipe = job_module.create_recipe(args)
        recipe.export(str(tmp_path / "exported"))

    assert recipe.launch_external_process is True
    assert recipe.launch_once is False
    assert recipe.server_expected_format == "pytorch"
    assert recipe.params_transfer_type == "FULL"

    job_dir = tmp_path / "exported" / "nemotron3-nano-peft"
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
    assert "custom/automodel_peft_client.py" in command
    assert command[command.index("--backend") + 1] == "mock"
    assert command[command.index("--train_file") + 1].endswith("split data/alpha10.0_site-1.jsonl")
    assert executor_args["launch_once"] is False
    assert client_config["max_resends"] == 3
    assert (job_dir / "app_site-1" / "custom" / "automodel_peft_client.py").exists()
    assert (job_dir / "app_site-1" / "custom" / "adapter_checkpoint.py").exists()
    assert (job_dir / "app_site-1" / "custom" / "automodel_adapter_loader.py").exists()
    assert (job_dir / "app_site-1" / "custom" / "automodel_financial_phrase_dataset.py").exists()

    controller = server_config["workflows"][0]
    assert controller["path"] == "nvflare.app_common.workflows.fedavg.FedAvg"
    assert controller["args"]["num_clients"] == 2
    persistor = next(
        c
        for c in server_config["components"]
        if c["path"] == "nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor"
    )
    assert persistor["args"]["load_device"] == "cpu"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for multi-round adapter aggregation checks")
def test_nemo_peft_full_adapter_fedavg_is_cumulative_across_rounds():
    import torch

    from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
    from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
    from nvflare.app_common.utils.fl_model_utils import FLModelUtils

    model = FLModel(
        params_type=ParamsType.FULL,
        params={
            "model.layer.lora_A.weight": torch.zeros((2, 2)),
            "model.layer.lora_B.weight": torch.ones((2, 2)),
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

    assert torch.equal(model.params["model.layer.lora_A.weight"], torch.full((2, 2), 1.5))
    assert torch.equal(model.params["model.layer.lora_B.weight"], torch.full((2, 2), 2.5))


def test_nemo_peft_sim_env_is_sequential_by_default(tmp_path):
    job_module = _load_job_module()
    args = _args(tmp_path, tmp_path / "init_adapter.pt")

    env = job_module.create_sim_env(args)

    assert env.clients == ["site-1", "site-2"]
    assert env.num_threads == 1
    assert env.gpu_config is None


def test_nemo_peft_train_args_are_shell_safe(tmp_path):
    job_module = _load_job_module()
    args = _args(tmp_path, tmp_path / "init_adapter.pt")

    train_args = job_module._build_train_args(args, str(tmp_path / "split data" / "alpha10.0_site-1.jsonl"), "site-1")

    parts = shlex.split(train_args)
    assert parts[parts.index("--backend") + 1] == "mock"
    assert parts[parts.index("--train_file") + 1].endswith("split data/alpha10.0_site-1.jsonl")
    assert parts[parts.index("--work_dir") + 1].endswith("workspace/automodel_work/site-1")
    assert "--balance_train_labels" in parts
    assert "--no-use_chat_template" in parts


def test_nemo_peft_automodel_config_uses_helper_files(tmp_path):
    client_module = _load_client_module()
    args = _args(tmp_path, tmp_path / "init_adapter.pt")
    args.backend = "automodel"
    args.train_file = str(tmp_path / "train.jsonl")
    incoming_adapter_dir = str(tmp_path / "incoming_adapter")

    config = client_module._default_automodel_config(args, str(tmp_path / "checkpoints"), incoming_adapter_dir)

    expected_dataset_suffix = os.path.join("peft", "automodel_financial_phrase_dataset.py")
    assert config["dataset"]["_target_"].endswith(f"{expected_dataset_suffix}:make_financial_phrase_dataset")
    assert config["dataset"]["balance_labels"] is True
    assert config["dataset"]["use_chat_template"] is False
    assert config["validation_dataset"]["_target_"] == config["dataset"]["_target_"]
    assert "balance_labels" not in config["validation_dataset"]
    assert "use_chat_template" not in config["validation_dataset"]
    expected_model_suffix = os.path.join("peft", "automodel_adapter_loader.py")
    assert config["model"]["_target_"].endswith(f"{expected_model_suffix}:from_pretrained_with_adapter")
    assert config["model"]["incoming_adapter_dir"] == incoming_adapter_dir
    assert "peft_config" not in config["model"]


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required to import the AutoModel client helper")
def test_nemo_peft_automodel_env_preserves_existing_pythonpath(monkeypatch):
    client_module = _load_client_module()
    parent_paths = ["/parent/path-one", "/parent/path-two"]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(parent_paths))

    env = client_module._build_subprocess_env()

    assert env["PYTHONPATH"].split(os.pathsep) == [_example_dir()] + parent_paths


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required to import the AutoModel client helper")
def test_nemo_peft_latest_adapter_dir_prefers_numeric_step_when_mtime_ties(tmp_path):
    client_module = _load_client_module()
    checkpoint_dir = tmp_path / "checkpoints"
    step_1 = checkpoint_dir / "step_1"
    step_10 = checkpoint_dir / "step_10"
    for adapter_dir in (step_1, step_10):
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "adapter_model.safetensors").write_text("")
        os.utime(adapter_dir, (1000, 1000))

    assert client_module._latest_adapter_dir(str(checkpoint_dir)) == str(step_10)


def test_nemo_peft_dataset_prompt_matches_notebook_inference():
    example_dir = _example_dir()
    spec = importlib.util.spec_from_file_location(
        "nemo_peft_dataset", os.path.join(example_dir, "automodel_financial_phrase_dataset.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module._build_prompt("The agreement is valid for four years .") == (
        "The agreement is valid for four years . sentiment:"
    )
    assert module._clean_label(" Positive ") == " positive"
    assert inspect.signature(module.make_financial_phrase_dataset).parameters["use_chat_template"].default is False


def test_nemo_peft_dataset_adds_pad_token_once_for_fp8(monkeypatch):
    example_dir = _example_dir()
    spec = importlib.util.spec_from_file_location(
        "nemo_peft_dataset", os.path.join(example_dir, "automodel_financial_phrase_dataset.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class FakeDataset(list):
        @property
        def column_names(self):
            return ["sentence", "label"]

        def select(self, indices):
            return FakeDataset([self[index] for index in indices])

        def map(self, fn, batched=False, remove_columns=None):
            return [fn(example) for example in self]

    dataset_module = ModuleType("datasets")
    dataset_module.load_dataset = lambda *args, **kwargs: FakeDataset(
        [{"sentence": "The agreement is valid for four years .", "label": " neutral"}]
    )
    formatting_module = ModuleType("formatting_utils")
    pad_calls = []

    def _add_pad_token(tokenizer):
        pad_calls.append(True)
        tokenizer.pad_token_id = 123
        return 123

    formatting_module._add_pad_token = _add_pad_token
    formatting_module.format_chat_template = lambda **kwargs: kwargs
    formatting_module.format_prompt_completion = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "datasets", dataset_module)
    monkeypatch.setitem(sys.modules, "nemo_automodel", ModuleType("nemo_automodel"))
    monkeypatch.setitem(sys.modules, "nemo_automodel.components", ModuleType("components"))
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.datasets", ModuleType("datasets"))
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.datasets.llm", ModuleType("llm"))
    monkeypatch.setitem(
        sys.modules,
        "nemo_automodel.components.datasets.llm.formatting_utils",
        formatting_module,
    )

    result = module.make_financial_phrase_dataset(SimpleNamespace(eos_token_id=2), "train.jsonl", fp8=True)

    assert len(pad_calls) == 1
    assert result[0]["pad_token_id"] == 123


def test_nemo_peft_dataset_balances_limited_training_window():
    example_dir = _example_dir()
    spec = importlib.util.spec_from_file_location(
        "nemo_peft_dataset", os.path.join(example_dir, "automodel_financial_phrase_dataset.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    dataset = (
        [{"label": " neutral"} for _ in range(5)]
        + [{"label": " positive"} for _ in range(3)]
        + [{"label": " negative"} for _ in range(2)]
    )

    indices = module._balanced_indices(dataset, limit_dataset_samples=6)
    labels = [dataset[index]["label"].strip() for index in indices]

    assert labels == ["neutral", "positive", "negative", "neutral", "positive", "negative"]


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required to import the prediction helper")
def test_nemo_peft_prediction_examples_match_notebook_expected_labels():
    predict_module = _load_predict_module()

    assert [expected for _prompt, expected in predict_module.NOTEBOOK_EXAMPLES] == [
        "neutral",
        "neutral",
        "positive",
        "negative",
    ]
    assert predict_module.classify({"neutral": -2.0, "positive": -0.5, "negative": -1.0}) == "positive"
