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
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module():
    path = Path(__file__).parents[1] / "verify_environment.py"
    spec = importlib.util.spec_from_file_location("evo2_verify_environment_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dependency_import_failures_are_contextual_and_preserve_the_cause(monkeypatch):
    verify_environment = _load_module()
    torch_error = ImportError("torch is missing")

    def fail_torch_import(_name):
        raise torch_error

    monkeypatch.setattr(verify_environment.importlib, "import_module", fail_torch_import)

    with pytest.raises(RuntimeError, match="PyTorch could not be imported") as exc_info:
        verify_environment._load_dependencies()
    assert exc_info.value.__cause__ is torch_error

    fake_torch = SimpleNamespace()
    causal_error = OSError("undefined symbol")

    def import_with_broken_extension(name):
        if name == "torch":
            return fake_torch
        raise causal_error

    monkeypatch.setattr(verify_environment.importlib, "import_module", import_with_broken_extension)
    with pytest.raises(RuntimeError, match="causal-conv1d could not be imported") as exc_info:
        verify_environment._load_dependencies()
    assert exc_info.value.__cause__ is causal_error


def test_cuda_unavailable_has_an_explicit_error():
    verify_environment = _load_module()
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        verify_environment._run_cuda_smoke(fake_torch, lambda *_args: None)


def test_cuda_execution_failure_is_contextual_and_preserves_the_cause():
    verify_environment = _load_module()
    execution_error = RuntimeError("kernel image is unavailable")
    fake_torch = SimpleNamespace(
        bfloat16="bfloat16",
        float32="float32",
        cuda=SimpleNamespace(is_available=lambda: True),
        randn=lambda *_args, **_kwargs: object(),
    )

    def fail_convolution(*_args):
        raise execution_error

    with pytest.raises(RuntimeError, match="CUDA forward/backward smoke failed") as exc_info:
        verify_environment._run_cuda_smoke(fake_torch, fail_convolution)
    assert exc_info.value.__cause__ is execution_error


def test_successful_cuda_smoke_runs_forward_backward_and_synchronizes(capsys, monkeypatch):
    verify_environment = _load_module()
    calls = []
    random_tensors = []

    class Output:
        def float(self):
            calls.append("float")
            return self

        def sum(self):
            calls.append("sum")
            return self

        def backward(self):
            calls.append("backward")

    def randn(*shape, **kwargs):
        tensor = (shape, kwargs)
        random_tensors.append(tensor)
        return tensor

    def fake_convolution(*_args):
        return Output()

    fake_torch = SimpleNamespace(
        bfloat16="bfloat16",
        float32="float32",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: calls.append("synchronize"),
        ),
        randn=randn,
    )
    monkeypatch.setattr(verify_environment, "_load_dependencies", lambda: (fake_torch, fake_convolution))

    verify_environment.main()

    assert random_tensors == [
        ((2, 32, 600), {"device": "cuda", "dtype": "bfloat16", "requires_grad": True}),
        ((32, 4), {"device": "cuda", "dtype": "float32", "requires_grad": True}),
    ]
    assert calls == ["float", "sum", "backward", "synchronize"]
    assert capsys.readouterr().out.strip() == "causal-conv1d CUDA forward/backward smoke passed"


def test_runtime_requirements_are_single_sourced_for_the_container():
    example_dir = Path(__file__).parents[1]
    requirements = [
        line.strip()
        for line in (example_dir / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    runtime_requirements = [
        line.strip()
        for line in (example_dir / "requirements-runtime.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    dockerfile = (example_dir / "Dockerfile").read_text(encoding="utf-8")

    assert requirements == ["nvflare[PT]>=2.9.0", "-r requirements-runtime.txt"]
    assert runtime_requirements == [
        "datasets==5.0.1",
        "matplotlib==3.11.1",
        "scikit-learn==1.9.0",
        "tensorboard==2.21.0",
        "jupyterlab==4.6.4",
    ]
    assert "COPY requirements-runtime.txt /tmp/evo2-requirements-runtime.txt" in dockerfile
    assert "uv pip install -r /tmp/evo2-requirements-runtime.txt" in dockerfile
    assert all(requirement not in dockerfile for requirement in runtime_requirements)
