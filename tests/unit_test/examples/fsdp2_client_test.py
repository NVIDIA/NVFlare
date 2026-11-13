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

"""Check the example's collective full-state exchange and exported job."""

import importlib.util
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = REPO_ROOT / "examples" / "advanced" / "multi-gpu" / "fsdp2"
torch = pytest.importorskip("torch", minversion="2.6")
transformers = pytest.importorskip("transformers", minversion="4.56")
pytest.importorskip("datasets", minversion="3.6")


def _load_client():
    # Isolate the example's flat "model" import from other example tests.
    previous = sys.modules.pop("model", None)
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location("fsdp2_example_client", EXAMPLE_DIR / "client.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)
        sys.modules.pop("model", None)
        if previous is not None:
            sys.modules["model"] = previous


def _check_collective_exchange(rank, rendezvous, model_dir):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        client = _load_client()
        mesh = init_device_mesh("cpu", (2,))
        model = client.CausalLMModel(model_dir, pretrained=False)
        for layer in model.model.model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)
        assert model.model.model.embed_tokens.weight is model.model.lm_head.weight

        # Loading a second global state must replace the first round's state.
        for seed in (17, 29):
            torch.manual_seed(seed)
            expected = client.CausalLMModel(model_dir, pretrained=False).state_dict()
            params = {name: value.clone() for name, value in expected.items()} if rank == 0 else None
            client.load_global_state(model, params)
            actual = client.gather_global_state(model)
            if rank == 0:
                assert actual.keys() == expected.keys()
                for name, value in expected.items():
                    assert actual[name].device.type == "cpu"
                    assert actual[name].dtype == torch.float32
                    torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
            else:
                assert actual == {}

        # The stop signal must reach both ranks without receiving another model.
        client.flare = Mock()
        client.flare.get_site_name.return_value = "site-1"
        client.flare.is_running.return_value = False
        client.make_loader = Mock()
        args = SimpleNamespace(data_dir="unused", batch_size=1)
        client.run_rounds(model, torch.device("cpu"), args)
        client.flare.receive.assert_not_called()
        if rank != 0:
            assert not client.flare.mock_calls

        # A rank-zero transport failure must raise on both ranks, rather than hang.
        operation = Mock(side_effect=ValueError("transport unavailable"))
        with pytest.raises(RuntimeError, match="ValueError: transport unavailable"):
            client.rank_zero_call(operation)
        assert operation.call_count == int(rank == 0)
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(120)
def test_two_rank_full_state_exchange_and_stop(tmp_path):
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("PyTorch Gloo is required")
    model_dir = tmp_path / "model"
    transformers.Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    ).save_pretrained(model_dir)
    mp.spawn(_check_collective_exchange, args=(str(tmp_path / "rendezvous"), str(model_dir)), nprocs=2, join=True)


@pytest.mark.parametrize("local_model", [False, True])
def test_job_export_packages_training_and_tensor_exchange(tmp_path, local_model):
    model_name = "Qwen/Qwen2.5-1.5B"
    expected_model_name = model_name
    if local_model:
        model_dir = tmp_path / "local model"
        model_dir.mkdir()
        model_name = os.path.relpath(model_dir, EXAMPLE_DIR)
        expected_model_name = str(model_dir.resolve())
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT), env.get("PYTHONPATH", "")))
    subprocess.run(
        [sys.executable, "job.py", "--model_name", model_name, "--export", "--export-dir", str(tmp_path)],
        cwd=EXAMPLE_DIR,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    app_dir = tmp_path / "pt_fsdp2" / "app"
    assert (app_dir / "custom" / "client.py").is_file()
    assert (app_dir / "custom" / "model.py").is_file()
    config = json.loads((app_dir / "config" / "config_fed_client.json").read_text())
    executor = config["executors"][0]["executor"]["args"]
    assert "--nproc_per_node=2" in executor["command"]
    assert "--standalone" in executor["command"]
    assert executor["server_expected_format"] == "pytorch"
    assert executor["shutdown_timeout"] == 10.0
    assert executor["command"][executor["command"].index("--model_name") + 1] == expected_model_name
    server = json.loads((app_dir / "config" / "config_fed_server.json").read_text())
    persistor = next(component for component in server["components"] if component["id"] == "persistor")
    assert persistor["args"]["model"]["args"]["model_name"] == expected_model_name
