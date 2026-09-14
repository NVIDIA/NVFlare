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

import pytest


def _load_module():
    module_path = (
        Path(__file__).parents[5] / "examples" / "advanced" / "bionemo" / "evo2" / "prepare_base_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location("evo2_prepare_base_checkpoint", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validate_checkpoint_requires_complete_iteration(tmp_path):
    module = _load_module()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    with pytest.raises(ValueError, match="missing"):
        module.validate_checkpoint(checkpoint)

    (checkpoint / "latest_checkpointed_iteration.txt").write_text("not-an-integer", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid iteration"):
        module.validate_checkpoint(checkpoint)

    (checkpoint / "latest_checkpointed_iteration.txt").write_text("1\n", encoding="utf-8")
    iteration = checkpoint / "iter_0000001"
    iteration.mkdir()
    with pytest.raises(ValueError, match="missing or empty"):
        module.validate_checkpoint(checkpoint)

    (iteration / "metadata.json").write_text("{}", encoding="utf-8")
    assert module.validate_checkpoint(checkpoint) == checkpoint.resolve()
