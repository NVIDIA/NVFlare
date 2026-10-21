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

import errno
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load, save_file

from nvflare.apis.shareable import Shareable
from nvflare.app_common.decomposers.numpy_decomposers import NumpyArrayDecomposer
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.app_opt.pt.lazy_tensor_dict import LazyTensorDict, safetensors_refs
from nvflare.app_opt.pt.tensor_downloader import TensorDownloadable
from nvflare.fuel.utils import fobs
from nvflare.private.fed.task_worker import artifacts
from nvflare.private.fed.task_worker.artifacts import FileTaskArtifactStore
from nvflare.private.fed.task_worker.protocol import TaskAttemptIdentity


@pytest.fixture
def payload_store(tmp_path):
    fobs.register(TensorDecomposer)
    fobs.register(NumpyArrayDecomposer)
    store = FileTaskArtifactStore(str(tmp_path / "attempts"))
    identity = TaskAttemptIdentity("job", "site", "task", "train", "attempt")
    store.create_attempt(identity)
    return store, identity, Path(store.attempt_dir(identity))


def test_tensor_bytes_stay_out_of_fobs_and_nested_payload_round_trips(payload_store):
    store, identity, directory = payload_store
    tensors = [torch.full((256 * 1024,), float(i)) for i in range(4)]
    data = Shareable({"params": {"weights": tensors}, "array": np.arange(3), "epoch": 2})
    data.set_header("round", 7)
    data.add_cookie("workflow", "wf")
    reference = store.write_input(identity, data)

    assert reference.size < 2048
    assert b"sha256" in (directory / "input.fobs").read_bytes()
    assert len(list(directory.glob("input.T*.safetensors"))) == len(tensors)
    restored = store.read_input(identity)
    assert restored.get_header("round") == 7 and restored.get_cookie("workflow") == "wf"
    assert restored["epoch"] == 2
    assert np.array_equal(restored["array"], data["array"])
    assert all(torch.equal(a, b) for a, b in zip(tensors, restored["params"]["weights"]))


def test_received_file_is_linked_and_survives_scratch_release(payload_store, tmp_path, monkeypatch):
    store, identity, directory = payload_store
    scratch = tmp_path / "offload"
    scratch.mkdir()
    file = scratch / "chunk_0.safetensors"
    save_file({"sender-id": torch.arange(10)}, str(file))
    owner = LazyTensorDict({"wire-id": (str(file), "sender-id")}, str(scratch))
    ref = owner.make_lazy_ref("wire-id")
    assert ref.nbytes == 80
    monkeypatch.setattr(ref, "materialize", lambda: pytest.fail("linking must not materialize the tensor"))
    store.write_input(identity, Shareable({"w": ref}))
    linked = directory / "input.T0.safetensors"
    assert os.stat(file).st_ino == os.stat(linked).st_ino
    owner.cleanup()
    assert not scratch.exists()
    assert torch.equal(store.read_input(identity)["w"], torch.arange(10))


@pytest.mark.parametrize("link_error", [None, errno.EXDEV, errno.EPERM, errno.EMLINK, errno.EOPNOTSUPP, errno.EIO])
def test_ref_link_fallback_keeps_its_key(payload_store, tmp_path, monkeypatch, link_error):
    store, identity, directory = payload_store
    source = tmp_path / "source.safetensors"
    tensors = {"a": torch.ones(3)}
    if link_error is None:
        tensors["b"] = torch.zeros(3)  # Only whole single-tensor files can be linked.
    else:

        def fail_link(*args):
            raise OSError(link_error, "test link failure")

        monkeypatch.setattr(artifacts, "_link_regular", fail_link)
    save_file(tensors, str(source))
    store.write_input(identity, Shareable({"w": safetensors_refs(str(source))["a"]}))
    assert os.stat(source).st_ino != os.stat(directory / "input.T0.safetensors").st_ino
    assert torch.equal(store.read_input(identity)["w"], tensors["a"])


@pytest.mark.parametrize("replace_during_open", [False, True])
def test_linking_rejects_replacement_of_a_fixed_checkpoint(payload_store, tmp_path, monkeypatch, replace_during_open):
    store, identity, directory = payload_store
    source = tmp_path / "checkpoint.safetensors"
    save_file({"w": torch.ones(3)}, str(source), metadata={"nvflare_model_id": "original"})
    ref = safetensors_refs(str(source))["w"]
    replacement = tmp_path / "replacement.safetensors"
    save_file({"w": torch.zeros(3)}, str(replacement), metadata={"nvflare_model_id": "replacement"})
    if replace_during_open:
        open_file = artifacts._open_protocol_file

        def replace_before_open(path):
            if os.fspath(path) == str(source):
                os.replace(replacement, source)
            return open_file(path)

        monkeypatch.setattr(artifacts, "_open_protocol_file", replace_before_open)
    else:
        os.replace(replacement, source)
    with pytest.raises(RuntimeError, match="model was replaced"):
        store.write_input(identity, Shareable({"w": ref}))
    assert not (directory / "input.json").exists()
    assert not (directory / "input.T0.safetensors").exists()


def test_linked_checkpoint_keeps_original_model_when_source_slot_advances(payload_store, tmp_path, monkeypatch):
    store, identity, directory = payload_store
    source = tmp_path / "checkpoint.safetensors"
    save_file({"w": torch.ones(3)}, str(source), metadata={"nvflare_model_id": "original"})
    ref = safetensors_refs(str(source))["w"]
    replacement = tmp_path / "replacement.safetensors"
    save_file({"w": torch.zeros(3)}, str(replacement), metadata={"nvflare_model_id": "replacement"})
    link_file = artifacts._link_regular

    def replace_after_link(*args):
        link_file(*args)
        os.replace(replacement, source)

    monkeypatch.setattr(artifacts, "_link_regular", replace_after_link)
    store.write_input(identity, Shareable({"w": ref}))
    assert os.stat(source).st_ino != os.stat(directory / "input.T0.safetensors").st_ino
    assert torch.equal(store.read_input(identity)["w"], torch.ones(3))


def test_mapped_result_edits_upload_after_artifact_names_are_removed(payload_store):
    store, identity, directory = payload_store
    store.commit_result(identity, Shareable({"w": torch.arange(6.0)}))
    result, completion = store.read_result(identity)
    result["w"].add_(4)  # A result callback's private mapped changes must be transmitted.
    assert torch.equal(store.read_result(identity)[0]["w"], torch.arange(6.0))
    sender = TensorDownloadable({"w": result["w"]}, 1024 * 1024)
    try:
        (directory / ".result.T9.safetensors.interrupted").write_bytes(b"partial")
        store.release_payloads(identity)
        store.release_payloads(identity)
        assert not list(directory.glob("result.T*.safetensors"))
        assert torch.equal(load(sender.produce_item(0))["w"], torch.arange(6.0) + 4)
    finally:
        sender.release()


@pytest.mark.parametrize("corruption", ["tensor", "record"])
def test_corrupt_payload_is_rejected_before_reaching_the_caller(payload_store, corruption):
    store, identity, directory = payload_store
    reference = store.stage_result(identity, Shareable({"w": torch.ones(4)}))
    path = directory / ("result.T0.safetensors" if corruption == "tensor" else "result.fobs")
    value = bytearray(path.read_bytes())
    value[-1] ^= 1
    path.write_bytes(value)
    if corruption == "record":
        with pytest.raises(ValueError, match="checksum"):
            store.commit_staged_result(identity, reference)
        assert not (directory / "completion.json").exists()
    else:
        # The worker checks its small record; tensor bytes are checked by the reader.
        store.commit_staged_result(identity, reference)
        with pytest.raises(ValueError, match="checksum"):
            store.read_result(identity)


@pytest.mark.parametrize("tensor_count", [1, 3])
def test_limit_counts_fobs_record_and_tensor_files(payload_store, tensor_count):
    store, identity, directory = payload_store
    payload = Shareable({"w": [torch.full((4096,), float(i)) for i in range(tensor_count)]})
    reference = store.write_input(identity, payload)
    tensor_sizes = [path.stat().st_size for path in directory.glob("input.T*.safetensors")]
    limited = FileTaskArtifactStore(store.root_dir, max_payload_bytes=min(tensor_sizes) - 1)
    with pytest.raises(ValueError, match="too large"):
        limited.read_input(identity)
    total_size = reference.size + sum(tensor_sizes)
    limited = FileTaskArtifactStore(store.root_dir, max_payload_bytes=total_size)
    restored = limited.read_input(identity)
    assert all(torch.equal(actual, expected) for actual, expected in zip(restored["w"], payload["w"]))
    limited = FileTaskArtifactStore(store.root_dir, max_payload_bytes=total_size - 1)
    with pytest.raises(ValueError, match="too large"):
        limited.read_input(identity)
    other = TaskAttemptIdentity("job", "site", "other-task", "train", "other-attempt")
    limited.create_attempt(other)
    with pytest.raises(ValueError, match="too large"):
        limited.write_input(other, payload)
    assert not Path(limited.attempt_dir(other), "input.json").exists()


@pytest.mark.parametrize("file", ["../outside.safetensors", "input.T0.safetensors"])
def test_tensor_descriptor_cannot_escape_its_payload(payload_store, file):
    store, identity, directory = payload_store
    reference = store.stage_result(identity, Shareable({"w": torch.ones(4)}))
    source = artifacts._TensorFileSource(str(directory), reference, None)
    with pytest.raises(ValueError, match="invalid tensor artifact descriptor"):
        source.get({"file": file, "key": "T0", "size": 0, "sha256": "0" * 64})


@pytest.mark.parametrize("name", ["result.T0.safetensors", ".result.T0.safetensors.interrupted"])
def test_worker_claim_fences_orphaned_tensor_artifacts(payload_store, name):
    store, identity, directory = payload_store
    (directory / name).write_bytes(b"orphan")
    with pytest.raises(FileExistsError, match="worker artifacts"):
        store.claim_worker(identity)


def test_tensor_cleanup_refuses_symlinks(payload_store, tmp_path):
    store, identity, directory = payload_store
    target = tmp_path / "unrelated"
    target.write_bytes(b"keep")
    (directory / ".result.T0.safetensors.interrupted").write_bytes(b"partial")
    (directory / "result.T1.safetensors").symlink_to(target)
    with pytest.raises(ValueError, match="invalid attempt artifact"):
        store.release_payloads(identity)
    assert target.read_bytes() == b"keep"
