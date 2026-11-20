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

import ast
import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.utils.data.partition import dirichlet_label_partition

REPO_ROOT = Path(__file__).resolve().parents[3]
PARTITIONER_ID = "nvflare.utils.data.partition.dirichlet_label_partition"
HAS_TORCH_DEPS = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None

MIGRATED_SPLITTER_PATHS = [
    REPO_ROOT / "examples/advanced/cifar10/pt/src/data/cifar10_data_split.py",
    REPO_ROOT / "examples/advanced/cifar10/tf/src/data/cifar10_data_utils.py",
    REPO_ROOT / "examples/advanced/collab/pt_cifar10/prepare_data.py",
    REPO_ROOT / "examples/advanced/collab/pt_async_cifar10/cifar10_data.py",
    REPO_ROOT / "examples/advanced/job_api/pt/fedavg_script_runner_xsite_val_cifar10.py",
    REPO_ROOT / "examples/advanced/job_api/tf/src/cifar10_data_split.py",
    REPO_ROOT
    / "examples/tutorials/self-paced-training/part-4_advanced_federated_learning/chapter-7_algorithms_and_workflows/07.2_algorithms/07.2.1_advanced_algos/src/cifar10_data_split.py",
    REPO_ROOT / "research/auto-fl-research/data/cifar10_data_split.py",
]


def _make_ordered_labels(num_samples: int = 500) -> np.ndarray:
    return np.array([index % 10 for index in range(num_samples)], dtype=np.int64)


class _DummyTorchDataset:
    pass


class _DummyNet:
    pass


class _DummyFileLock:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _make_fake_cifar10_cls(labels: np.ndarray):
    class _FakeCIFAR10:
        def __init__(self, root=None, train=True, transform=None, download=False):
            self.data = np.zeros((len(labels), 32, 32, 3), dtype=np.uint8)
            self.targets = labels.tolist()

    return _FakeCIFAR10


def _stub_torch_modules(cifar10_cls) -> dict:
    torch_mod = types.ModuleType("torch")
    torch_mod.__path__ = []
    torch_mod.utils = types.SimpleNamespace(
        data=types.SimpleNamespace(
            Dataset=_DummyTorchDataset,
            DataLoader=lambda dataset, **kwargs: types.SimpleNamespace(dataset=dataset, kwargs=kwargs),
        )
    )
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module = object
    torch_mod.nn = nn_mod

    torchvision_mod = types.ModuleType("torchvision")
    torchvision_mod.__path__ = []
    datasets_mod = types.ModuleType("torchvision.datasets")
    datasets_mod.CIFAR10 = cifar10_cls
    transforms_mod = types.ModuleType("torchvision.transforms")
    transforms_mod.Compose = lambda ops: ops
    transforms_mod.ToTensor = lambda: "ToTensor"
    transforms_mod.Pad = lambda *args, **kwargs: "Pad"
    transforms_mod.RandomCrop = lambda *args, **kwargs: "RandomCrop"
    transforms_mod.RandomHorizontalFlip = lambda *args, **kwargs: "RandomHorizontalFlip"
    transforms_mod.Normalize = lambda *args, **kwargs: "Normalize"
    torchvision_mod.datasets = datasets_mod
    torchvision_mod.transforms = transforms_mod

    return {
        "torch": torch_mod,
        "torch.nn": nn_mod,
        "torchvision": torchvision_mod,
        "torchvision.datasets": datasets_mod,
        "torchvision.transforms": transforms_mod,
    }


def _stub_tf_modules(keras_load_data) -> dict:
    tf_mod = types.ModuleType("tensorflow")
    tf_mod.__path__ = []
    keras_mod = types.ModuleType("tensorflow.keras")
    keras_mod.__path__ = []
    datasets_mod = types.ModuleType("tensorflow.keras.datasets")
    datasets_mod.cifar10 = types.SimpleNamespace(load_data=keras_load_data)
    keras_mod.datasets = datasets_mod
    tf_mod.keras = keras_mod
    tf_mod.float32 = np.float32
    tf_mod.constant = lambda value, dtype=None: np.asarray(value, dtype=dtype)
    return {
        "tensorflow": tf_mod,
        "tensorflow.keras": keras_mod,
        "tensorflow.keras.datasets": datasets_mod,
    }


def _stub_filelock_module() -> dict:
    mod = types.ModuleType("filelock")
    mod.FileLock = _DummyFileLock
    mod.Timeout = TimeoutError
    return {"filelock": mod}


@contextmanager
def _isolated_import(sys_path_dir: Path, stubs: dict):
    saved_path = list(sys.path)
    saved_keys = set(sys.modules)
    saved_modules = {key: sys.modules.get(key) for key in set(stubs) | {"data", "src"}}
    for prefix in ("data.", "src."):
        for name in list(sys.modules):
            if name.startswith(prefix):
                saved_modules.setdefault(name, sys.modules[name])
                sys.modules.pop(name, None)

    for key in ("data", "src"):
        sys.modules.pop(key, None)

    sys.modules.update(stubs)
    sys.path.insert(0, str(sys_path_dir))
    try:
        yield
    finally:
        sys.path[:] = saved_path
        for name in list(sys.modules):
            if name not in saved_keys or name.startswith(("data.", "src.", "_test_cifar_")):
                sys.modules.pop(name, None)
        for key, original in saved_modules.items():
            if original is not None:
                sys.modules[key] = original
            else:
                sys.modules.pop(key, None)


def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_pt_cifar10_modules(labels: np.ndarray, use_real_torch: bool = False):
    fake_cifar10_cls = _make_fake_cifar10_cls(labels)
    stubs = {} if (use_real_torch and HAS_TORCH_DEPS) else _stub_torch_modules(fake_cifar10_cls)

    src_dir = REPO_ROOT / "examples/advanced/cifar10/pt/src"
    with _isolated_import(src_dir, stubs):
        if use_real_torch and HAS_TORCH_DEPS:
            import torchvision

            torchvision.datasets.CIFAR10 = fake_cifar10_cls
        utils_mod = _load_module_from_path("_test_cifar_pt_utils", src_dir / "data/cifar10_data_utils.py")
        utils_mod.load_cifar10_data = lambda: labels
        split_mod = _load_module_from_path("_test_cifar_pt_split", src_dir / "data/cifar10_data_split.py")
        return split_mod, utils_mod


def _load_splitter(kind: str, labels: np.ndarray, load_counter: list):
    def count_1d_labels():
        load_counter[0] += 1
        return labels

    def count_keras_tuple():
        load_counter[0] += 1
        images = np.zeros((len(labels), 32, 32, 3), dtype=np.uint8)
        return (images, labels.reshape(-1, 1)), (images[:10], labels[:10].reshape(-1, 1))

    fake_cifar10_cls = _make_fake_cifar10_cls(labels)

    if kind == "pt":
        split_mod, _ = _load_pt_cifar10_modules(labels)
        split_mod.load_cifar10_data = count_1d_labels
        return split_mod

    if kind == "collab_pt":
        root = REPO_ROOT / "examples/advanced/collab/pt_cifar10"
        stubs = _stub_torch_modules(fake_cifar10_cls)
        with _isolated_import(root, stubs):
            mod = _load_module_from_path("_test_cifar_collab_pt", root / "prepare_data.py")
            mod.load_cifar10_data = count_1d_labels
            return mod

    if kind == "collab_async":
        root = REPO_ROOT / "examples/advanced/collab/pt_async_cifar10"

        class _CountingCIFAR10:
            def __init__(self, root=None, train=True, download=False):
                if not download and train:
                    load_counter[0] += 1
                self.targets = labels.tolist()

        stubs = _stub_torch_modules(_CountingCIFAR10)
        with _isolated_import(root, stubs):
            return _load_module_from_path("_test_cifar_collab_async", root / "cifar10_data.py")

    if kind == "job_api_pt":
        root = REPO_ROOT / "examples/advanced/job_api/pt"
        src_mod = types.ModuleType("src")
        src_net = types.ModuleType("src.net")
        src_net.Net = _DummyNet
        pt_opt_mod = types.ModuleType("nvflare.app_opt.pt")
        pt_opt_mod.__path__ = []
        pt_locator_mod = types.ModuleType("nvflare.app_opt.pt.file_model_locator")
        pt_locator_mod.PTFileModelLocator = _DummyNet
        pt_persistor_mod = types.ModuleType("nvflare.app_opt.pt.file_model_persistor")
        pt_persistor_mod.PTFileModelPersistor = _DummyNet
        tb_mod = types.ModuleType("nvflare.app_opt.tracking.tb.tb_receiver")
        tb_mod.TBAnalyticsReceiver = _DummyNet
        stubs = {
            **_stub_torch_modules(fake_cifar10_cls),
            **_stub_filelock_module(),
            "src": src_mod,
            "src.net": src_net,
            "nvflare.app_opt.pt": pt_opt_mod,
            "nvflare.app_opt.pt.file_model_locator": pt_locator_mod,
            "nvflare.app_opt.pt.file_model_persistor": pt_persistor_mod,
            "nvflare.app_opt.tracking.tb.tb_receiver": tb_mod,
        }
        with _isolated_import(root, stubs):
            mod = _load_module_from_path("_test_cifar_job_api_pt", root / "fedavg_script_runner_xsite_val_cifar10.py")
            mod.load_cifar10_labels = count_1d_labels
            return mod

    if kind == "tf":
        src_dir = REPO_ROOT / "examples/advanced/cifar10/tf/src"
        stubs = {**_stub_tf_modules(count_keras_tuple), **_stub_filelock_module()}
        with _isolated_import(src_dir, stubs):
            mod = _load_module_from_path("_test_cifar_tf", src_dir / "data/cifar10_data_utils.py")
            mod.load_cifar10_with_retry = count_keras_tuple
            return mod

    if kind == "job_api_tf":
        src_dir = REPO_ROOT / "examples/advanced/job_api/tf/src"
        stubs = {**_stub_tf_modules(count_keras_tuple), **_stub_filelock_module()}
        with _isolated_import(src_dir, stubs):
            mod = _load_module_from_path("_test_cifar_job_api_tf", src_dir / "cifar10_data_split.py")
            mod.load_cifar10_with_retry = count_keras_tuple
            return mod

    if kind == "tutorial_tf":
        src_dir = (
            REPO_ROOT
            / "examples/tutorials/self-paced-training/part-4_advanced_federated_learning/chapter-7_algorithms_and_workflows/07.2_algorithms/07.2.1_advanced_algos/src"
        )
        stubs = {**_stub_tf_modules(count_keras_tuple), **_stub_filelock_module()}
        with _isolated_import(src_dir, stubs):
            return _load_module_from_path("_test_cifar_tutorial_tf", src_dir / "cifar10_data_split.py")

    if kind == "autofl":
        root = REPO_ROOT / "research/auto-fl-research"
        stubs = _stub_torch_modules(fake_cifar10_cls)
        with _isolated_import(root, stubs):
            mod = _load_module_from_path("_test_cifar_autofl", root / "data/cifar10_data_split.py")
            mod.load_cifar10_data = count_1d_labels
            return mod

    raise ValueError(f"Unknown splitter kind: {kind}")


def _run_splitter_to_dir(kind: str, mod, split_root: Path, num_sites: int, alpha: float, seed: int) -> Path:
    if kind == "pt":
        return Path(mod.split_and_save(str(split_root / "pt"), num_sites=num_sites, alpha=alpha, seed=seed))
    if kind == "collab_pt":
        mod.SPLIT_ROOT = str(split_root / "collab_pt")
        return Path(mod.split_and_save(num_sites=num_sites, alpha=alpha, seed=seed))
    if kind == "collab_async":
        return Path(mod.split_and_save(str(split_root / "collab_async"), num_sites=num_sites, alpha=alpha, seed=seed))
    if kind == "job_api_pt":
        out = split_root / "job_api_pt"
        mod.create_data_splits(str(out), num_sites=num_sites, alpha=alpha, seed=seed)
        return out
    if kind in ("tf", "job_api_tf", "tutorial_tf"):
        out = split_root / kind
        mod.cifar10_split(split_dir=str(out), num_sites=num_sites, alpha=alpha, seed=seed)
        return out
    if kind == "autofl":
        return Path(
            mod.split_and_save(str(split_root / "autofl"), num_sites=num_sites, alpha=alpha, seed=seed, reuse=False)
        )
    raise ValueError(f"Unknown splitter kind: {kind}")


def test_pt_split_export_and_loader_compatibility(tmp_path):
    labels = _make_ordered_labels(500)
    expected = dirichlet_label_partition(labels=labels, num_sites=4, alpha=0.5, seed=7)
    split_mod, _ = _load_pt_cifar10_modules(labels)

    split_dir = Path(split_mod.split_and_save(str(tmp_path / "cifar"), num_sites=4, alpha=0.5, seed=7))
    summary = (split_dir / "summary.txt").read_text(encoding="utf-8")
    assert "Number of clients: 4" in summary
    assert "Dirichlet sampling parameter: 0.5" in summary
    assert "Seed: 7" in summary
    assert f"Partitioner: {PARTITIONER_ID}" in summary

    collected = []
    for site in range(4):
        loaded = np.load(split_dir / f"site-{site + 1}.npy", allow_pickle=False)
        assert loaded.dtype == np.int64
        np.testing.assert_array_equal(loaded, expected.site_indices[site])
        collected.extend(loaded.tolist())
    assert sorted(collected) == list(range(len(labels)))

    splitter_dir = tmp_path / "component_split"
    splitter = split_mod.Cifar10DataSplitter(split_dir=str(splitter_dir), num_sites=4, alpha=0.5, seed=7)
    splitter.split(FLContext())
    for site in range(4):
        comp_loaded = np.load(splitter_dir / f"site-{site + 1}.npy", allow_pickle=False)
        np.testing.assert_array_equal(comp_loaded, expected.site_indices[site])

    if HAS_TORCH_DEPS:
        real_split_mod, real_utils_mod = _load_pt_cifar10_modules(labels, use_real_torch=True)
        real_split_mod.load_cifar10_data = lambda: labels
        real_dir = Path(real_split_mod.split_and_save(str(tmp_path / "real_pt"), num_sites=4, alpha=0.5, seed=7))
        train_ds, valid_ds = real_utils_mod.create_datasets("site-1", str(real_dir))
        assert len(train_ds) == len(expected.site_indices[0])
        assert len(valid_ds) == len(labels)
        np.testing.assert_array_equal(train_ds.target, labels[expected.site_indices[0]])
    else:
        loaded_for_site_1 = np.load(splitter_dir / "site-1.npy", allow_pickle=False).tolist()
        assert loaded_for_site_1 == expected.site_indices[0].tolist()


def test_cached_legacy_splits_are_not_reused(tmp_path):
    labels = _make_ordered_labels(500)
    expected = dirichlet_label_partition(labels=labels, num_sites=4, alpha=0.5, seed=0)

    legacy_summaries = [
        "Number of clients: 4 \nDirichlet sampling parameter: 0.5 \nClass counts for each client: \n{}",
        "Number of clients: 4 \nDirichlet sampling parameter: 0.5 \nSeed: 0 \nClass counts for each client: \n{}",
        (
            f"Number of clients: 40\nDirichlet sampling parameter: 0.5\nSeed: 0\n"
            f"Partitioner: {PARTITIONER_ID}\nClass counts for each client:\n{{}}"
        ),
    ]

    for idx, legacy_summary in enumerate(legacy_summaries):
        # 1. Cifar10DataSplitter
        pt_dir = tmp_path / f"pt_legacy_{idx}"
        pt_dir.mkdir()
        for site in range(1, 5):
            np.save(pt_dir / f"site-{site}.npy", np.array([-1], dtype=np.int64))
        (pt_dir / "summary.txt").write_text(legacy_summary, encoding="utf-8")

        pt_calls = [0]
        pt_mod = _load_splitter("pt", labels, pt_calls)
        pt_mod.Cifar10DataSplitter(split_dir=str(pt_dir), num_sites=4, alpha=0.5, seed=0).split(FLContext())
        assert pt_calls[0] == 1
        np.testing.assert_array_equal(np.load(pt_dir / "site-1.npy"), expected.site_indices[0])

        pt_mod.Cifar10DataSplitter(split_dir=str(pt_dir), num_sites=4, alpha=0.5, seed=0).split(FLContext())
        assert pt_calls[0] == 1

        # 2. Job API PT create_data_splits
        job_dir = tmp_path / f"job_legacy_{idx}"
        job_dir.mkdir()
        for site in range(1, 5):
            np.save(job_dir / f"site-{site}.npy", np.array([-1], dtype=np.int64))
        (job_dir / "summary.txt").write_text(legacy_summary, encoding="utf-8")

        job_calls = [0]
        job_mod = _load_splitter("job_api_pt", labels, job_calls)
        job_mod.create_data_splits(str(job_dir), num_sites=4, alpha=0.5, seed=0)
        assert job_calls[0] >= 1
        np.testing.assert_array_equal(np.load(job_dir / "site-1.npy"), expected.site_indices[0])

        calls_after_regen = job_calls[0]
        job_mod.create_data_splits(str(job_dir), num_sites=4, alpha=0.5, seed=0)
        assert job_calls[0] == calls_after_regen

        # 3. auto-fl-research split_and_save(..., reuse=True)
        auto_prefix = tmp_path / f"auto_legacy_{idx}" / "autofl"
        auto_dir = Path(f"{auto_prefix}_4sites_alpha0.50_seed0")
        auto_dir.mkdir(parents=True)
        for site in range(1, 5):
            np.save(auto_dir / f"site-{site}.npy", np.array([-1], dtype=np.int64))
        (auto_dir / "summary.txt").write_text(legacy_summary, encoding="utf-8")

        auto_calls = [0]
        auto_mod = _load_splitter("autofl", labels, auto_calls)
        auto_mod.split_and_save(str(auto_prefix), num_sites=4, alpha=0.5, seed=0, reuse=True)
        assert auto_calls[0] == 1
        np.testing.assert_array_equal(np.load(auto_dir / "site-1.npy"), expected.site_indices[0])

        auto_mod.split_and_save(str(auto_prefix), num_sites=4, alpha=0.5, seed=0, reuse=True)
        assert auto_calls[0] == 1


@pytest.mark.parametrize(
    "kind",
    ["pt", "collab_pt", "collab_async", "job_api_pt", "tf", "job_api_tf", "tutorial_tf", "autofl"],
)
def test_all_cifar10_splitters_match_partition_utility_and_preserve_global_rng(tmp_path, kind):
    labels = _make_ordered_labels(500)
    expected = dirichlet_label_partition(labels=labels, num_sites=4, alpha=0.5, seed=13)
    other_seed = dirichlet_label_partition(labels=labels, num_sites=4, alpha=0.5, seed=99)

    np.random.seed(12345)
    expected_global_draw = np.random.random()

    load_counter = [0]
    mod = _load_splitter(kind, labels, load_counter)

    np.random.seed(12345)
    out_dir = _run_splitter_to_dir(kind, mod, tmp_path / "run13", num_sites=4, alpha=0.5, seed=13)
    assert np.random.random() == expected_global_draw

    for site in range(4):
        loaded = np.load(out_dir / f"site-{site + 1}.npy", allow_pickle=False)
        np.testing.assert_array_equal(loaded, expected.site_indices[site])

    out_dir_99 = _run_splitter_to_dir(kind, mod, tmp_path / "run99", num_sites=4, alpha=0.5, seed=99)
    site1_13 = np.load(out_dir / "site-1.npy", allow_pickle=False)
    site1_99 = np.load(out_dir_99 / "site-1.npy", allow_pickle=False)
    assert not np.array_equal(site1_13, site1_99)
    np.testing.assert_array_equal(site1_99, other_seed.site_indices[0])


def test_cifar10_splitters_reject_non_positive_alpha(tmp_path):
    labels = _make_ordered_labels(100)
    pt_mod = _load_splitter("pt", labels, [0])
    tf_mod = _load_splitter("tf", labels, [0])

    with pytest.raises(ValueError, match="Alpha should be larger than 0.0"):
        pt_mod.Cifar10DataSplitter(split_dir=str(tmp_path / "abs"), alpha=0.0)
    with pytest.raises(ValueError, match="Alpha should be larger than 0.0"):
        pt_mod.split_and_save(str(tmp_path / "prefix"), num_sites=2, alpha=0.0)
    with pytest.raises(ValueError, match="Alpha should be larger than 0.0"):
        tf_mod.cifar10_split(split_dir=str(tmp_path / "tf_abs"), num_sites=2, alpha=0.0)


def test_migrated_cifar10_splitters_do_not_call_numpy_global_dirichlet():
    for file_path in MIGRATED_SPLITTER_PATHS:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
        called_attrs = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "dirichlet" not in called_attrs, f"Legacy dirichlet call remains in {file_path}"
        assert "nvflare.utils.data.partition" in source
