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
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")
EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "hello-world" / "hello-huggingface"


@pytest.fixture
def persistor_module(monkeypatch):
    for name in ("model", "hf_persistor"):
        spec = importlib.util.spec_from_file_location(name, EXAMPLE_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def server(persistor_module, tmp_path, monkeypatch):
    """Initialize the real persistor as SJ does, using a tiny local Qwen model."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    from nvflare.apis.event_type import EventType
    from nvflare.apis.fl_constant import FLContextKey
    from nvflare.apis.fl_context import FLContext
    from nvflare.app_common.app_constant import AppConstants

    base_dir = tmp_path / "base"
    config = transformers.Qwen2Config(
        vocab_size=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    transformers.Qwen2ForCausalLM(config).save_pretrained(base_dir)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(base_dir)

    persistor = persistor_module.HuggingFacePersistor(str(base_dir), "test/federated-adapter")
    fl_ctx = FLContext()
    fl_ctx.set_prop(AppConstants.NUM_ROUNDS, 2)
    app_dir = tmp_path / "server_app"
    fl_ctx.set_prop(FLContextKey.APP_ROOT, str(app_dir))
    persistor.handle_event(EventType.START_RUN, fl_ctx)
    persistor.load_model(fl_ctx)
    api = MagicMock()
    monkeypatch.setattr("huggingface_hub.HfApi", lambda: api)
    return SimpleNamespace(persistor=persistor, fl_ctx=fl_ctx, api=api, app_dir=app_dir, base_dir=base_dir)


@pytest.mark.parametrize("upload_fails", [False, True])
def test_final_round_upload_preserves_aggregate_and_checkpoint(server, upload_fails):
    from peft import AutoPeftModelForCausalLM, get_peft_model_state_dict

    from nvflare.apis.event_type import EventType
    from nvflare.app_common.abstract.model import make_model_learnable
    from nvflare.app_common.app_constant import AppConstants

    persistor, fl_ctx, api = server.persistor, server.fl_ctx, server.api
    weights = {key: torch.full_like(value, 0.125) for key, value in persistor.model.state_dict().items()}
    fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 0)
    persistor.save(make_model_learnable(weights, {}), fl_ctx)
    # Ending a run before the final round must not publish a partial model.
    persistor.handle_event(EventType.END_RUN, fl_ctx)
    api.create_repo.assert_not_called()
    api.upload_folder.assert_not_called()
    checkpoint_path = server.app_dir / "FL_global_model.pt"
    assert checkpoint_path.is_file()

    weights = {key: torch.full_like(value, 0.25) for key, value in weights.items()}
    fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 1)
    if upload_fails:
        api.upload_folder.side_effect = RuntimeError("Hub unavailable")
        with pytest.raises(RuntimeError, match="Hub unavailable"):
            persistor.save(make_model_learnable(weights, {}), fl_ctx)
    else:
        persistor.save(make_model_learnable(weights, {}), fl_ctx)

    # The local checkpoint and standard PEFT export survive a failed upload.
    saved_weights = torch.load(checkpoint_path, weights_only=True)["model"]
    output_dir = server.app_dir / "huggingface-adapter"
    restored = AutoPeftModelForCausalLM.from_pretrained(output_dir, local_files_only=True)
    restored_weights = get_peft_model_state_dict(restored)
    assert restored_weights.keys() == weights.keys()
    for key, value in weights.items():
        torch.testing.assert_close(saved_weights[key], value)
        torch.testing.assert_close(restored_weights[key], value)
    assert (
        transformers.AutoTokenizer.from_pretrained(output_dir, local_files_only=True).get_vocab()
        == transformers.AutoTokenizer.from_pretrained(server.base_dir, local_files_only=True).get_vocab()
    )
    api.create_repo.assert_called_once_with(
        repo_id="test/federated-adapter", repo_type="model", private=True, exist_ok=True
    )
    api.upload_folder.assert_called_once_with(
        repo_id="test/federated-adapter",
        repo_type="model",
        folder_path=str(output_dir),
        commit_message="Upload final federated LoRA adapter",
    )


def test_local_checkpoint_failure_prevents_upload(server, monkeypatch):
    from nvflare.app_common.abstract.model import make_model_learnable
    from nvflare.app_common.app_constant import AppConstants

    server.fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 1)
    monkeypatch.setattr(server.persistor, "save_model_file", MagicMock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        server.persistor.save(make_model_learnable(server.persistor.model.state_dict(), {}), server.fl_ctx)
    server.api.create_repo.assert_not_called()
    server.api.upload_folder.assert_not_called()


def test_mismatched_adapter_keys_prevent_upload(server):
    from nvflare.app_common.abstract.model import make_model_learnable
    from nvflare.app_common.app_constant import AppConstants

    weights = dict(server.persistor.model.state_dict())
    weights.pop(next(iter(weights)))
    server.fl_ctx.set_prop(AppConstants.CURRENT_ROUND, 1)
    with pytest.raises(ValueError, match="adapter keys do not match"):
        server.persistor.save(make_model_learnable(weights, {}), server.fl_ctx)
    server.api.create_repo.assert_not_called()
    server.api.upload_folder.assert_not_called()


@pytest.mark.parametrize("upload_enabled", [False, True])
def test_job_export_registers_server_persistor(tmp_path, upload_enabled):
    for site in ("site-1", "site-2"):
        data_dir = tmp_path / "data" / site
        data_dir.mkdir(parents=True)
        for filename in ("train.jsonl", "valid.jsonl"):
            (data_dir / filename).write_text('{"text": "hello"}\n')
    command = [
        sys.executable,
        "job.py",
        "--data_root",
        str(tmp_path / "data"),
        "--export",
        "--export-dir",
        str(tmp_path / "exported"),
    ]
    if upload_enabled:
        command.extend(["--hf_repo_id", "test/federated-adapter"])
    # A dummy runtime credential must never become part of the exported job.
    dummy_token = "test-runtime-token-not-for-export"
    result = subprocess.run(
        command,
        cwd=EXAMPLE_DIR,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "HF_TOKEN": dummy_token},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    app_dir = tmp_path / "exported" / "hello-huggingface" / "app"
    config = json.loads((app_dir / "config" / "config_fed_server.json").read_text())
    persistor = next(component for component in config["components"] if component["id"] == "persistor")
    if upload_enabled:
        assert persistor["path"] == "hf_persistor.HuggingFacePersistor"
        assert persistor["args"] == {
            "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
            "repo_id": "test/federated-adapter",
        }
        assert (app_dir / "custom" / "hf_persistor.py").is_file()
    else:
        assert persistor["path"] == "nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor"
    assert (app_dir / "custom" / "model.py").is_file()
    client_config = json.loads((app_dir / "config" / "config_fed_client.json").read_text())
    assert "hf_persistor" not in json.dumps(client_config)
    for path in app_dir.rglob("*"):
        if path.is_file():
            assert dummy_token.encode() not in path.read_bytes()
