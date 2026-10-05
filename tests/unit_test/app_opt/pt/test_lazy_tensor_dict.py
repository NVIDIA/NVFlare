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

import copy
import os
import tempfile

import pytest
import torch
from safetensors.torch import load_file, save_file

import nvflare.app_opt.pt.lazy_tensor_dict as lazy_tensor_dict
from nvflare.app_opt.pt.lazy_tensor_dict import (
    LazyTensorDict,
    TensorMetadata,
    _LazyRef,
    _TempDirRef,
    safetensors_refs,
    tensor_metadata,
    write_safetensors,
)


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
    def test_materialize_loads_tensor(self, temp_safetensors):
        key_to_file, temp_dir, tensors = temp_safetensors
        file_path, st_key = key_to_file["layer1.weight"]
        ref = _LazyRef(file_path=file_path, key=st_key, temp_ref=_TempDirRef(temp_dir))

        result = ref.materialize()
        assert torch.allclose(result, tensors["layer1.weight"])

    def test_repr(self, temp_safetensors):
        key_to_file, temp_dir, _ = temp_safetensors
        file_path, st_key = key_to_file["layer1.bias"]
        ref = _LazyRef(file_path=file_path, key=st_key, temp_ref=_TempDirRef(temp_dir))
        assert "layer1.bias" in repr(ref)

    def test_release_cleans_shared_temp_dir(self, temp_safetensors):
        key_to_file, temp_dir, _ = temp_safetensors
        file_path, st_key = key_to_file["layer1.bias"]
        ref = _LazyRef(file_path=file_path, key=st_key, temp_ref=_TempDirRef(temp_dir))

        ref.release()

        assert not os.path.exists(temp_dir)

    def test_release_preserves_a_user_checkpoint(self, tmp_path):
        save_file({"a": torch.ones(1)}, tmp_path / "checkpoint.safetensors")

        safetensors_refs(str(tmp_path / "checkpoint.safetensors"))["a"].release()

        assert (tmp_path / "checkpoint.safetensors").exists()

    def test_deepcopy_shares_temp_dir_lifetime(self, temp_safetensors):
        key_to_file, temp_dir, tensors = temp_safetensors
        ref = LazyTensorDict(key_to_file=key_to_file, temp_dir=temp_dir).make_lazy_ref("layer1.weight")

        snapshot = copy.deepcopy(ref)
        assert snapshot is not ref
        del snapshot

        assert os.path.exists(temp_dir)
        assert torch.equal(ref.materialize(), tensors["layer1.weight"])

    def test_metadata_comes_from_the_header(self, tmp_path):
        tensors = {
            "float": torch.ones(7),
            "flags": torch.tensor([True, False, True]),
            "half": torch.ones(2, 3, dtype=torch.bfloat16),
        }
        save_file(tensors, tmp_path / "model.safetensors")

        refs = safetensors_refs(str(tmp_path / "model.safetensors"))

        assert refs["float"].get_metadata() == TensorMetadata((7,), "F32", 28)
        assert refs["flags"].get_metadata() == TensorMetadata((3,), "BOOL", 3)
        assert refs["half"].get_metadata() == TensorMetadata((2, 3), "BF16", 12)
        assert all(refs[key].get_metadata() == tensor_metadata(tensor) for key, tensor in tensors.items())
        assert all(ref._temp_ref is None for ref in refs.values())


class TestWriteSafetensors:
    def test_roundtrip_in_declared_order(self, tmp_path):
        tensors = {"b": torch.randn(2, 3), "a": torch.arange(5), "c": torch.tensor(True)}
        metadata = {key: tensor_metadata(tensor) for key, tensor in tensors.items()}

        write_safetensors(str(tmp_path / "out.safetensors"), metadata, iter(tensors.items()))

        loaded = load_file(tmp_path / "out.safetensors")
        assert set(loaded) == set(tensors)
        assert all(torch.equal(loaded[key], tensor) for key, tensor in tensors.items())

    def test_rejects_tensors_that_do_not_match_the_header(self, tmp_path):
        path = str(tmp_path / "out.safetensors")
        metadata = {"a": tensor_metadata(torch.ones(2)), "b": tensor_metadata(torch.ones(3))}

        with pytest.raises(ValueError, match="header order"):
            write_safetensors(path, metadata, iter([("b", torch.ones(3)), ("a", torch.ones(2))]))
        with pytest.raises(ValueError, match="declared shape or dtype"):
            write_safetensors(path, metadata, iter([("a", torch.ones(2)), ("b", torch.ones(4))]))
        with pytest.raises(ValueError, match="fewer tensors"):
            write_safetensors(path, metadata, iter([("a", torch.ones(2))]))


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

    def test_repeated_lazy_refs_are_aggregated_one_at_a_time(self, temp_safetensors, monkeypatch):
        from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper

        key_to_file, temp_dir, tensors = temp_safetensors
        ltd = LazyTensorDict(key_to_file=key_to_file, temp_dir=temp_dir)
        helper = WeightedAggregationHelper(exclude_vars="bias")
        helper.add({key: value + 1.0 for key, value in tensors.items()}, 2.0, "site-1", 0)
        loaded = []
        original_materialize = _LazyRef.materialize

        def materialize(ref):
            if loaded:
                previous = loaded[-1]
                torch.testing.assert_close(helper.total[previous], 5.0 * tensors[previous] + 2.0)
            loaded.append(ref.key)
            return original_materialize(ref)

        monkeypatch.setattr(_LazyRef, "materialize", materialize)
        helper.add({key: ltd.make_lazy_ref(key) for key in ltd.keys()}, 3.0, "site-2", 0)

        expected_keys = [key for key in tensors if "bias" not in key]
        assert loaded == expected_keys
        result = helper.get_result()
        assert list(result) == expected_keys
        for key in expected_keys:
            torch.testing.assert_close(result[key], tensors[key] + 0.4)

    def test_late_shape_mismatch_does_not_materialize_earlier_refs(self, temp_safetensors, monkeypatch):
        from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper

        key_to_file, temp_dir, tensors = temp_safetensors
        ltd = LazyTensorDict(key_to_file=key_to_file, temp_dir=temp_dir)
        helper = WeightedAggregationHelper()
        helper.add(tensors, 1.0, "site-1", 0)

        def unexpected_materialize(ref):
            raise AssertionError("Shape validation must finish before materialization")

        monkeypatch.setattr(_LazyRef, "materialize", unexpected_materialize)
        with pytest.raises(ValueError, match="layer2.weight"):
            helper.add(
                {"layer1.weight": ltd.make_lazy_ref("layer1.weight"), "layer2.weight": torch.ones(1)},
                1.0,
                "site-2",
                0,
            )

        assert helper.get_len() == 1
        for key, value in helper.get_result().items():
            torch.testing.assert_close(value, tensors[key])
