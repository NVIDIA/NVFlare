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

import logging
import shutil

from safetensors import safe_open

logger = logging.getLogger(__name__)


class _TempDirRef:
    """Reference-counted sentinel for a temp directory.

    Shared between LazyTensorDict and all _LazyRef instances created from it.
    The directory is deleted only when ALL holders are garbage collected.
    """

    def __init__(self, temp_dir: str):
        self.path = temp_dir
        self._deleted = False

    def cleanup(self):
        if not self._deleted:
            self._deleted = True
            shutil.rmtree(self.path, ignore_errors=True)

    def __del__(self):
        self.cleanup()


class _LazyRef:
    """Lightweight placeholder for an on-disk tensor.

    Carries only file_path + key (~100 bytes). The tensor is loaded from disk
    only when resolve() is called, keeping memory near zero until then.

    Holds a reference to _TempDirRef to prevent premature cleanup.
    """

    def __init__(self, file_path: str, key: str, temp_ref: _TempDirRef):
        self.file_path = file_path
        self.key = key
        self._temp_ref = temp_ref

    def resolve(self):
        """Load tensor from safetensors file. Opens mmap, copies data out, closes mmap."""
        with safe_open(self.file_path, framework="pt") as f:
            return f.get_tensor(self.key)

    def __repr__(self):
        return f"_LazyRef({self.file_path!r}, key={self.key!r})"


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

    def make_lazy_ref(self, key) -> "_LazyRef":
        file_path, st_key = self._key_to_file[key]
        return _LazyRef(file_path=file_path, key=st_key, temp_ref=self._temp_ref)

    def cleanup(self):
        self._temp_ref.cleanup()


def cleanup_lazy_refs(data):
    """Explicitly delete temp directories referenced by _LazyRef values in a dict."""
    if not data or not isinstance(data, dict):
        return
    for v in data.values():
        temp_ref = getattr(v, "_temp_ref", None)
        if temp_ref is not None:
            temp_ref.cleanup()
            return  # all refs share the same _TempDirRef
