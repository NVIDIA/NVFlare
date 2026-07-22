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

import inspect
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")
datasets = pytest.importorskip("datasets")
peft = pytest.importorskip("peft")
pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")
trl = pytest.importorskip("trl")

from peft import LoraConfig, PeftModel  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, default_data_collator  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from nvflare.client.config import ConfigKey, ExchangeFormat  # noqa: E402


def _import_real_hf_api_modules():
    for loaded_name in list(sys.modules):
        if loaded_name == "nvflare.client.hf" or loaded_name.startswith("nvflare.app_opt.hf"):
            sys.modules.pop(loaded_name, None)

    import nvflare.app_opt.hf.api as hf_api
    import nvflare.client.hf as flare

    return hf_api, flare


def _tiny_tokenizer():
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[PAD]": 0,
                "[EOS]": 1,
                "[UNK]": 2,
                "hello": 3,
                "world": 4,
                "federated": 5,
                "learning": 6,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    fast_tokenizer.model_max_length = 8
    return fast_tokenizer


def _tiny_dataset():
    return datasets.Dataset.from_list(
        [
            {"input_ids": [3, 4, 1], "attention_mask": [1, 1, 1], "labels": [3, 4, 1]},
            {"input_ids": [5, 6, 1], "attention_mask": [1, 1, 1], "labels": [5, 6, 1]},
        ]
    )


def _tiny_model():
    config = GPT2Config(
        vocab_size=8,
        n_positions=8,
        n_ctx=8,
        n_embd=8,
        n_layer=1,
        n_head=1,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
    )
    return GPT2LMHeadModel(config)


def _filtered_kwargs(callable_obj, values: dict) -> dict:
    params = inspect.signature(callable_obj).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return values
    return {name: value for name, value in values.items() if name in params}


def _sft_config(output_dir):
    values = {
        "data_seed": 0,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "disable_tqdm": True,
        "bf16": False,
        "fp16": False,
        "logging_strategy": "no",
        "max_length": 8,
        "max_steps": 1,
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 1,
        "remove_unused_columns": False,
        "report_to": [],
        "save_strategy": "no",
        "seed": 0,
        "use_cpu": True,
    }
    return SFTConfig(**_filtered_kwargs(SFTConfig.__init__, values))


def _make_sft_trainer(tmp_path, *, peft_config=None):
    values = {
        "args": _sft_config(tmp_path),
        "data_collator": default_data_collator,
        "model": _tiny_model(),
        "peft_config": peft_config,
        "processing_class": _tiny_tokenizer(),
        "train_dataset": _tiny_dataset(),
    }
    return SFTTrainer(**_filtered_kwargs(SFTTrainer.__init__, values))


def _patch_client_api(monkeypatch, hf_api):
    config = {
        ConfigKey.TASK_EXCHANGE: {
            ConfigKey.EXCHANGE_FORMAT: ExchangeFormat.PYTORCH,
            ConfigKey.SERVER_EXPECTED_FORMAT: ExchangeFormat.NUMPY,
            ConfigKey.TRAIN_WITH_EVAL: False,
        }
    }
    monkeypatch.setattr(hf_api.flare_api, "default_context", None, raising=False)
    monkeypatch.setattr(hf_api.flare_api, "init", lambda rank=None, config_file=None: None)
    monkeypatch.setattr(hf_api.flare_api, "get_config", lambda ctx=None: config)
    monkeypatch.setattr(hf_api.flare_api, "get_job_id", lambda ctx=None: "hf-real-trl-contract")


def _fl_callbacks(trainer):
    callback_handler = getattr(trainer, "callback_handler", None)
    return list(getattr(callback_handler, "callbacks", None) or getattr(trainer, "callbacks", []))


@pytest.mark.parametrize(
    "peft_config,expected_scope",
    [
        (None, "model"),
        (
            LoraConfig(
                task_type="CAUSAL_LM",
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=["c_attn"],
            ),
            "adapter",
        ),
    ],
)
def test_real_sft_trainer_constructs_and_accepts_hf_patch(monkeypatch, tmp_path, peft_config, expected_scope):
    hf_api, flare = _import_real_hf_api_modules()

    hf_api._reset_global_state_for_test()
    _patch_client_api(monkeypatch, hf_api)

    trainer = _make_sft_trainer(tmp_path / expected_scope, peft_config=peft_config)
    if peft_config is not None:
        assert isinstance(trainer.model, PeftModel)

    flare.patch(trainer, restore_state=False)

    assert getattr(trainer, "_nvflare_hf_patched", False)
    assert trainer._nvflare_hf_task_state.params_scope == expected_scope
    assert any(callback.__class__.__name__ == "FLCallback" for callback in _fl_callbacks(trainer))

    hf_api._reset_global_state_for_test()
