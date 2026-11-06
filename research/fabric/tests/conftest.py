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
#
# Authors: Anbang Liu, Junhan Zhao, and Ziyue Xu

"""Portable CPU fixtures; all manifests and checkpoints are generated in temporary storage."""

import csv
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "fabric"))


@pytest.fixture(autouse=True)
def cpu_random_state():
    """Keep the small CPU tests deterministic without leaking RNG or thread settings."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.set_num_threads(1)
    torch.random.default_generator.manual_seed(123)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.set_num_threads(threads)


@pytest.fixture
def make_fold(tmp_path):
    from fabric_common import SITES, load_setup, sha256_file, state_digest, write_json

    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    for encoder in ("uni", "virchow2"):
        shutil.copyfile(PROJECT / "configs" / f"{encoder}.json", root / "configs" / f"{encoder}.json")

    def create(name="case", encoder="uni", variant="pooling"):
        setup = load_setup(root, encoder, variant, 1 if variant == "topk" else None)
        folder = root / "results" / name / encoder / "fold_0"
        folder.mkdir(parents=True)
        initial = {"weight": torch.tensor([1.0, -2.0]), "buffer": torch.tensor([3], dtype=torch.int64)}
        counts = dict(zip(SITES, (4, 6, 8), strict=True))
        for index, site in enumerate(SITES):
            path = folder / f"client_{site}_train_manifest.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("patient_id", "site", "recurrence_label", "feature_paths"))
                for row in range(counts[site]):
                    writer.writerow((f"synthetic-{index}-{row}", site, row % 2, "unused.h5"))
        runtime = {
            "root": str(root),
            "experiment": encoder,
            "fold": 0,
            "fold_dir": str(folder),
            "variant": variant,
            "top_k": setup.get("model_options", {}).get("top_k"),
            "manifest_root": str(root / "manifests"),
            "client_patients": counts,
            "initial_state_sha256": state_digest(initial),
            "state_spec": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in initial.items()
            },
            "model_config": {"model": setup["model_name"]},
            "manifest_sha256": {path.name: sha256_file(path) for path in folder.glob("*.csv")},
        }
        runtime_path = folder / "runtime.json"
        write_json(runtime_path, runtime)
        torch.save({"model": initial}, folder / "initial_model.pt")
        return SimpleNamespace(
            root=root, folder=folder, setup=setup, runtime=runtime, path=runtime_path, initial=initial
        )

    return create


@pytest.fixture
def client_update():
    from fabric_common import SITES, local_seed, state_digest

    from nvflare.app_common.abstract.fl_model import FLModel, ParamsType

    def create(case, site, state, round_index, offset=0):
        index = SITES.index(site)
        absolute_round = round_index + offset
        return FLModel(
            params={key: value + index + 1 for key, value in state.items()},
            params_type=ParamsType.FULL,
            current_round=round_index,
            metrics={"train_loss": 0.25 + index},
            meta={
                "client_name": site,
                "site": site,
                "train_patients": case.runtime["client_patients"][site],
                "local_epochs": case.setup["settings"]["local_epochs"],
                "seed": local_seed(case.setup, 0, absolute_round, site),
                "input_state_sha256": state_digest(state),
                "local_log": {"loss": 0.25 + index, "round": absolute_round + 1},
            },
        )

    return create
