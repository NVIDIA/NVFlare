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

import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file

import nvflare.app_opt.pt.lazy_tensor_dict as lazy_tensor_dict
from nvflare.app_opt.pt.lazy_tensor_dict import LazyTensorDict, LazyTensorMetadata, LazyTensorRef, _LazyRef, _TempDirRef


@pytest.fixture
def temp_safetensors():
    """Create temp safetensors files and return (key_to_file, temp_dir, original_tensors)."""
    temp_dir = tempfile.mkdtemp(prefix="nvflare_test_")
    tensors = {
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(3, 10),
    }

    key_to_file = {}
    for i, (name, tensor) in enumerate(tensors.items()):
        file_path = os.path.join(temp_dir, f"chunk_{i}.safetensors")
        save_file({name: tensor}, file_path)
        key_to_file[name] = (file_path, name)

    yield key_to_file, temp_dir, tensors

    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


class TestLazyRef:
    def test_public_compatibility_type(self):
        assert LazyTensorRef is _LazyRef

    def test_materialize_loads_tensor(self, temp_safetensors):
        key_to_file, temp_dir, tensors = temp_safetensors
        file_path, st_key = key_to_file["layer1.weight"]
        ref = _LazyRef(file_path=file_path, key=st_key, temp_ref=_TempDirRef(temp_dir))

        result = ref.materialize()
        assert torch.allclose(result, tensors["layer1.weight"])

    def test_tensor_metadata_does_not_materialize_tensor(self, temp_safetensors, monkeypatch):
        key_to_file, temp_dir, _ = temp_safetensors
        file_path, st_key = key_to_file["layer1.weight"]
        ref = LazyTensorRef(file_path=file_path, key=st_key, temp_ref=_TempDirRef(temp_dir))

        class SafeOpenWrapper:
            def __init__(self, opened):
                self.opened = opened

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.opened.__exit__(exc_type, exc_val, exc_tb)

            def get_slice(self, key):
                return self.opened.get_slice(key)

            def get_tensor(self, key):
                raise AssertionError("metadata preflight must not load tensor data")

        original_safe_open = lazy_tensor_dict.safe_open

        def wrapped_safe_open(*args, **kwargs):
            opened = original_safe_open(*args, **kwargs)
            opened.__enter__()
            return SafeOpenWrapper(opened)

        monkeypatch.setattr(lazy_tensor_dict, "safe_open", wrapped_safe_open)

        metadata = ref.tensor_metadata()

        assert metadata == LazyTensorMetadata(shape=(10, 5), dtype="F32", num_elements=50, num_bytes=200)

    def test_bounded_materialize_rejects_from_metadata_before_loading(self, tmp_path, monkeypatch):
        loaded = False

        class FakeSlice:
            @staticmethod
            def get_shape():
                return [4]

            @staticmethod
            def get_dtype():
                return "F32"

        class FakeSafeOpen:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return None

            @staticmethod
            def get_slice(key):
                assert key == "weight"
                return FakeSlice()

            @staticmethod
            def get_tensor(key):
                nonlocal loaded
                loaded = True
                return torch.ones(4)

        monkeypatch.setattr(lazy_tensor_dict, "safe_open", lambda *args, **kwargs: FakeSafeOpen())
        temp_dir = tmp_path / "offload"
        temp_dir.mkdir()
        ref = LazyTensorRef(
            file_path=str(temp_dir / "unused.safetensors"),
            key="weight",
            temp_ref=_TempDirRef(str(temp_dir)),
        )

        with pytest.raises(ValueError, match="lazy tensor byte size"):
            ref.materialize_bounded(max_elements=4, max_bytes=15)

        assert loaded is False

    def test_repr(self, temp_safetensors):
        key_to_file, temp_dir, _ = temp_safetensors
        file_path, st_key = key_to_file["layer1.bias"]
        ref = _LazyRef(file_path=file_path, key=st_key, temp_ref=_TempDirRef(temp_dir))
        assert "layer1.bias" in repr(ref)


class TestTempDirRef:
    def test_cleanup_on_del(self):
        temp_dir = tempfile.mkdtemp(prefix="nvflare_test_")
        assert os.path.exists(temp_dir)
        ref = _TempDirRef(temp_dir)
        del ref
        assert not os.path.exists(temp_dir)

    def test_shared_ref_survives_dict_del(self, temp_safetensors):
        """Temp dir survives after LazyTensorDict is GC'd if _LazyRef still holds a reference."""
        key_to_file, temp_dir, tensors = temp_safetensors
        ltd = LazyTensorDict(key_to_file=key_to_file, temp_dir=temp_dir)
        lazy_ref = ltd.make_lazy_ref("layer1.weight")

        del ltd  # LazyTensorDict gone, but _LazyRef holds _TempDirRef
        assert os.path.exists(temp_dir)

        result = lazy_ref.materialize()  # should still work
        assert torch.allclose(result, tensors["layer1.weight"])

        del lazy_ref  # last reference gone → _TempDirRef.__del__ fires
        assert not os.path.exists(temp_dir)

    def test_cleanup_logs_warning_on_error(self, monkeypatch, caplog, tmp_path):
        temp_dir = tmp_path / "nvflare_test_cleanup_warn"
        temp_dir.mkdir()
        ref = _TempDirRef(str(temp_dir))

        def raise_cleanup_error(_):
            raise PermissionError("permission denied")

        monkeypatch.setattr(lazy_tensor_dict.shutil, "rmtree", raise_cleanup_error)

        with caplog.at_level("WARNING"):
            ref.cleanup()

        assert "failed to cleanup tensor offload temp dir" in caplog.text
        assert str(temp_dir) in caplog.text


class TestLazyTensorDict:
    def test_getitem(self, temp_safetensors):
        key_to_file, temp_dir, tensors = temp_safetensors
        ltd = LazyTensorDict(key_to_file=key_to_file, temp_dir=temp_dir)

        for name, expected in tensors.items():
            assert torch.allclose(ltd[name], expected)

    def test_get_default(self, temp_safetensors):
        key_to_file, temp_dir, _ = temp_safetensors
        ltd = LazyTensorDict(key_to_file=key_to_file, temp_dir=temp_dir)

        assert ltd.get("nonexistent") is None
        assert ltd.get("nonexistent", "default") == "default"

    def test_keys(self, temp_safetensors):
        key_to_file, temp_dir, tensors = temp_safetensors
        ltd = LazyTensorDict(key_to_file=key_to_file, temp_dir=temp_dir)
        assert set(ltd.keys()) == set(tensors.keys())

    def test_iter_yields_keys(self, temp_safetensors):
        key_to_file, temp_dir, tensors = temp_safetensors
        ltd = LazyTensorDict(key_to_file=key_to_file, temp_dir=temp_dir)
        assert set(iter(ltd)) == set(tensors.keys())
        assert set(ltd) == set(tensors.keys())

    def test_len(self, temp_safetensors):
        key_to_file, temp_dir, _ = temp_safetensors
        ltd = LazyTensorDict(key_to_file=key_to_file, temp_dir=temp_dir)
        assert len(ltd) == 3

    def test_contains(self, temp_safetensors):
        key_to_file, temp_dir, _ = temp_safetensors
        ltd = LazyTensorDict(key_to_file=key_to_file, temp_dir=temp_dir)
        assert "layer1.weight" in ltd
        assert "nonexistent" not in ltd

    def test_items_yields_tensors(self, temp_safetensors):
        key_to_file, temp_dir, tensors = temp_safetensors
        ltd = LazyTensorDict(key_to_file=key_to_file, temp_dir=temp_dir)

        for key, val in ltd.items():
            assert torch.allclose(val, tensors[key])

    def test_make_lazy_ref(self, temp_safetensors):
        key_to_file, temp_dir, tensors = temp_safetensors
        ltd = LazyTensorDict(key_to_file=key_to_file, temp_dir=temp_dir)

        ref = ltd.make_lazy_ref("layer2.weight")
        assert isinstance(ref, _LazyRef)
        assert torch.allclose(ref.materialize(), tensors["layer2.weight"])

    def test_cleanup(self, temp_safetensors):
        key_to_file, temp_dir, _ = temp_safetensors
        ltd = LazyTensorDict(key_to_file=key_to_file, temp_dir=temp_dir)

        assert os.path.exists(temp_dir)
        ltd.cleanup()
        assert not os.path.exists(temp_dir)

    def test_getitem_raises_keyerror(self, temp_safetensors):
        key_to_file, temp_dir, _ = temp_safetensors
        ltd = LazyTensorDict(key_to_file=key_to_file, temp_dir=temp_dir)

        with pytest.raises(KeyError):
            _ = ltd["nonexistent"]


class TestAggregationHelperWithLazyRefs:
    def test_helper_materializes_lazy_refs(self, temp_safetensors):
        """WeightedAggregationHelper materializes _LazyRef via duck-typed materialize()."""
        key_to_file, temp_dir, tensors = temp_safetensors
        ltd = LazyTensorDict(key_to_file=key_to_file, temp_dir=temp_dir)

        lazy_refs = {k: ltd.make_lazy_ref(k) for k in ltd.keys()}

        from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper

        helper = WeightedAggregationHelper()
        helper.add(data=lazy_refs, weight=1.0, contributor_name="client1", contribution_round=0)

        result = helper.get_result()
        for name, expected in tensors.items():
            assert torch.allclose(result[name], expected, atol=1e-6)
