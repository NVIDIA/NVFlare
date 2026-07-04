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

"""PT lazy tensor references used by tensor disk offload.

When `enable_tensor_disk_offload=True`, incoming streamed tensor payloads are written
to temporary safetensors files instead of being fully deserialized into memory.
`LazyTensorDict` maps item IDs to on-disk files, and `LazyTensorRef` defers loading until
`materialize()` is called by aggregation code.

This keeps peak memory lower for large models while still allowing deterministic
explicit cleanup via `cleanup()`, with GC as a fallback through `_TempDirRef`.
"""

import logging
import shutil
from typing import NamedTuple, Optional

from safetensors import safe_open

logger = logging.getLogger(__name__)


_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class LazyTensorMetadata(NamedTuple):
    """Size metadata read from a safetensors header without loading tensor data."""

    shape: tuple[int, ...]
    dtype: str
    num_elements: int
    num_bytes: int


def _metadata_from_safe_slice(tensor_slice) -> LazyTensorMetadata:
    shape = tuple(tensor_slice.get_shape())
    if any(type(dimension) is not int or dimension < 0 for dimension in shape):
        raise ValueError("safetensors shape must contain non-negative integers")

    dtype = tensor_slice.get_dtype()
    if type(dtype) is not str or dtype not in _SAFETENSORS_DTYPE_BYTES:
        raise ValueError(f"unsupported safetensors dtype: {dtype!r}")

    num_elements = 1
    for dimension in shape:
        num_elements *= dimension
    return LazyTensorMetadata(
        shape=shape,
        dtype=dtype,
        num_elements=num_elements,
        num_bytes=num_elements * _SAFETENSORS_DTYPE_BYTES[dtype],
    )


def _cleanup_temp_dir(path: str) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return
    except Exception as e:
        logger.warning("failed to cleanup tensor offload temp dir '%s': %s", path, e)


class _TempDirRef:
    """Reference-counted sentinel for a temp directory.

    Shared between LazyTensorDict and all LazyTensorRef instances created from it.
    The directory is deleted only when ALL holders are garbage collected.
    """

    def __init__(self, temp_dir: str):
        self.path = temp_dir
        self._deleted = False

    def cleanup(self):
        if not self._deleted:
            self._deleted = True
            _cleanup_temp_dir(self.path)

    def __del__(self):
        self.cleanup()


class LazyTensorRef:
    """Lightweight placeholder for an on-disk tensor.

    Carries only file_path + key (~100 bytes). The tensor is loaded from disk
    only when materialize() is called, keeping memory near zero until then.

    Holds a reference to _TempDirRef to prevent premature cleanup.
    """

    def __init__(self, file_path: str, key: str, temp_ref: _TempDirRef):
        self.file_path = file_path
        self.key = key
        self._temp_ref = temp_ref

    def tensor_metadata(self) -> LazyTensorMetadata:
        """Read shape/dtype/size from the header without loading tensor data."""

        with safe_open(self.file_path, framework="pt") as f:
            return _metadata_from_safe_slice(f.get_slice(self.key))

    def materialize_bounded(
        self,
        *,
        max_elements: Optional[int] = None,
        max_bytes: Optional[int] = None,
    ):
        """Load this tensor only after its header metadata satisfies the supplied limits."""

        with safe_open(self.file_path, framework="pt") as f:
            metadata = _metadata_from_safe_slice(f.get_slice(self.key))
            if max_elements is not None and metadata.num_elements > max_elements:
                raise ValueError(f"lazy tensor element count exceeds configured maximum {max_elements}")
            if max_bytes is not None and metadata.num_bytes > max_bytes:
                raise ValueError(f"lazy tensor byte size exceeds configured maximum {max_bytes}")
            return f.get_tensor(self.key)

    def materialize(self):
        """Load tensor from safetensors file. Opens mmap, copies data out, closes mmap."""

        return self.materialize_bounded()

    def __repr__(self):
        return f"_LazyRef({self.file_path!r}, key={self.key!r})"


# Backward-compatible private alias retained for existing callers.
_LazyRef = LazyTensorRef


class LazyTensorDict:
    """Dict-like mapping of FOBS item_ids to on-disk safetensors files.

    Each entry maps an item_id to a (file_path, key) pair. Tensors are loaded
    via safetensors safe_open (mmap) on access.
    """

    def __init__(self, key_to_file: dict[str, tuple[str, str]], temp_dir: str):
        self._key_to_file = key_to_file
        self._temp_ref = _TempDirRef(temp_dir)

    def __getitem__(self, key):
        file_path, st_key = self._key_to_file[key]
        with safe_open(file_path, framework="pt") as f:
            return f.get_tensor(st_key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return self._key_to_file.keys()

    def __iter__(self):
        return iter(self._key_to_file)

    def items(self):
        for key in self._key_to_file:
            yield key, self[key]

    def values(self):
        for key in self._key_to_file:
            yield self[key]

    def __len__(self):
        return len(self._key_to_file)

    def __contains__(self, key):
        return key in self._key_to_file

    def make_lazy_ref(self, key) -> "LazyTensorRef":
        file_path, st_key = self._key_to_file[key]
        return LazyTensorRef(file_path=file_path, key=st_key, temp_ref=self._temp_ref)

    def cleanup(self):
        self._temp_ref.cleanup()
