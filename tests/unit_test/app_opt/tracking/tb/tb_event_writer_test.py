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

import builtins
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from nvflare.app_opt.tracking.tb.tb_event_writer import TensorBoardEventWriter


def _read_accumulator(log_dir: Path) -> EventAccumulator:
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    return accumulator


class TestTensorBoardEventWriter:
    def test_add_scalar_without_global_step_uses_tensorboard_default_step_zero(self, tmp_path):
        writer = TensorBoardEventWriter(str(tmp_path))

        writer.add_scalar("metric", 1.0)
        writer.add_scalar("metric", 2.0)
        writer.flush()
        writer.close()

        accumulator = _read_accumulator(tmp_path)
        assert [(event.step, event.value) for event in accumulator.Scalars("metric")] == [(0, 1.0), (0, 2.0)]

    def test_add_scalars_preserves_subseries_slashes_like_torch(self, tmp_path):
        writer = TensorBoardEventWriter(str(tmp_path))

        writer.add_scalars("metrics/main", {"train/acc": 1.0}, global_step=4)
        writer.flush()
        writer.close()

        assert list(writer.scalar_writers.keys()) == ["metrics_main_train/acc"]

        accumulator = _read_accumulator(tmp_path / "metrics_main_train" / "acc")
        assert [(event.step, event.value) for event in accumulator.Scalars("metrics/main")] == [(4, 1.0)]

    def test_add_image_without_pillow_raises_actionable_error(self, tmp_path):
        writer = TensorBoardEventWriter(str(tmp_path))
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        original_import = builtins.__import__

        def import_without_pillow(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError("No module named 'PIL'")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=import_without_pillow):
            with pytest.raises(ImportError, match="Pillow is required for TensorBoard image analytics"):
                writer.add_image("sample_image", image, global_step=1)
