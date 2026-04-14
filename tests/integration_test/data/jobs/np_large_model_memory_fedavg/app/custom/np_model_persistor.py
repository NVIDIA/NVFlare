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

"""Custom NPModelPersistor that creates a configurable-size numpy model.

Used for large-model memory integration tests.  The model is intentionally sized
to exceed the ViaDownloaderDecomposer streaming threshold (2 MB per array), so
that each key is sent through the download service rather than inlined.

This simulates the 5 GiB large-model scenario (the actual root cause of the
memory issues) at a scale that can run locally without a large machine.

Parameters:
  total_size_mb: Total model size in MB (decimal, 1 MB = 1 000 000 bytes).
  bytes_per_key: Bytes per numpy array key.  Set > 2 097 152 to force streaming.
  dtype: numpy dtype for the arrays (default float32).
"""

import os

import numpy as np

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor


def _get_run_dir(fl_ctx: FLContext):
    engine = fl_ctx.get_engine()
    if engine is None:
        raise RuntimeError("engine is missing in fl_ctx.")
    job_id = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
    if job_id is None:
        raise RuntimeError("job_id is missing in fl_ctx.")
    return engine.get_workspace().get_run_dir(job_id)


class NPModelPersistor(ModelPersistor):
    """Creates a synthetic numpy model of configurable size.

    Each key maps to one ndarray.  Setting bytes_per_key > 2 MB ensures that
    each tensor is sent through the ViaDownloaderDecomposer streaming path,
    exercising the PASS_THROUGH code path rather than inline FOBS encoding.
    """

    def __init__(
        self,
        model_dir: str = "models",
        model_name: str = "server.npy",
        total_size_mb: int = 4,
        bytes_per_key: int = 2_200_000,
        dtype: str = "float32",
        key_prefix: str = "numpy_key_",
    ):
        super().__init__()
        self.model_dir = model_dir
        self.model_name = model_name
        if total_size_mb <= 0:
            raise ValueError("total_size_mb must be > 0")
        if bytes_per_key <= 0:
            raise ValueError("bytes_per_key must be > 0")
        self.total_size_mb = int(total_size_mb)
        self.bytes_per_key = int(bytes_per_key)
        self.dtype = np.dtype(dtype)
        self.key_prefix = key_prefix

    def _create_default_data(self) -> dict:
        """Create arrays sized to exercise the streaming path.

        Each key value is bytes_per_key bytes (> 2 MB streaming threshold),
        ensuring the download-service path is exercised rather than inline FOBS.
        Total model size ≈ total_size_mb MB.
        """
        total_bytes = self.total_size_mb * 1_000_000
        n_keys = max(1, total_bytes // self.bytes_per_key)
        itemsize = int(self.dtype.itemsize)
        n_elems = max(1, self.bytes_per_key // itemsize)
        data = {}
        for idx in range(n_keys):
            data[f"{self.key_prefix}{idx}"] = np.zeros((n_elems,), dtype=self.dtype)
        return data

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        run_dir = _get_run_dir(fl_ctx)
        model_path = os.path.join(run_dir, self.model_dir, self.model_name)
        try:
            tmp = np.load(model_path, allow_pickle=True)
            data = tmp[()]
        except Exception:
            self.log_info(fl_ctx, f"No saved model at {model_path}; using fresh initialisation.", fire_event=False)
            data = self._create_default_data()

        model_learnable = make_model_learnable(weights=data, meta_props={})
        keys = list(data.keys())
        size_mb = sum(v.size * v.itemsize for v in data.values()) / 1_000_000
        self.log_info(fl_ctx, f"Loaded model: num_keys={len(keys)} size={size_mb:.1f} MB")
        return model_learnable

    def save_model(self, model_learnable: ModelLearnable, fl_ctx: FLContext):
        run_dir = _get_run_dir(fl_ctx)
        model_root_dir = os.path.join(run_dir, self.model_dir)
        os.makedirs(model_root_dir, exist_ok=True)
        model_path = os.path.join(model_root_dir, self.model_name)
        # np.save(model_path, model_learnable[ModelLearnableKey.WEIGHTS])
        self.log_info(fl_ctx, f"Saved model to {model_path}")
