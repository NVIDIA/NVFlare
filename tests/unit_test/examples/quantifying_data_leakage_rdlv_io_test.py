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
import os

import numpy as np


def _load_rdlv_io():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    module_path = os.path.join(
        repo_root, "research", "quantifying-data-leakage", "src", "nvflare_gradinv", "utils", "rdlv_io.py"
    )
    spec = importlib.util.spec_from_file_location("quantifying_data_leakage_rdlv_io", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.load_rdlv_results, module.save_rdlv_results


def test_rdlv_archive_round_trip_with_best_matches(tmp_path):
    load_rdlv_results, save_rdlv_results = _load_rdlv_io()
    result_path = tmp_path / "rdvl_round3.npz"

    save_rdlv_results(
        save_path=str(result_path),
        img_recon_sim_reduced=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        img_recon_sim=np.arange(12, dtype=np.float32).reshape(2, 3, 2),
        closest_idx=np.array([[1, 0], [0, 1]], dtype=np.int64),
        site="site-9",
        round_number=3,
        best_matches=[
            [np.ones((3, 2, 2), dtype=np.float32), np.zeros((3, 2, 2), dtype=np.float32)],
            [np.full((3, 2, 2), 2.0, dtype=np.float32), np.full((3, 2, 2), 3.0, dtype=np.float32)],
        ],
    )

    loaded = load_rdlv_results(str(result_path))

    assert loaded["site"] == "site-9"
    assert loaded["round"] == 3
    np.testing.assert_array_equal(loaded["closest_idx"], np.array([[1, 0], [0, 1]], dtype=np.int64))
    np.testing.assert_allclose(loaded["img_recon_sim_reduced"], np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    np.testing.assert_array_equal(loaded["best_match_inputs"].shape, np.array([2, 3, 2, 2]))
    np.testing.assert_array_equal(loaded["best_match_recons"].shape, np.array([2, 3, 2, 2]))


def test_rdlv_archive_round_trip_without_closest_idx(tmp_path):
    load_rdlv_results, save_rdlv_results = _load_rdlv_io()
    result_path = tmp_path / "rdvl_round4.npz"

    save_rdlv_results(
        save_path=str(result_path),
        img_recon_sim_reduced=np.array([[0.5]], dtype=np.float32),
        img_recon_sim=np.array([[[0.5]]], dtype=np.float32),
        closest_idx=None,
        site="site-1",
        round_number=4,
    )

    loaded = load_rdlv_results(str(result_path))

    assert loaded["site"] == "site-1"
    assert loaded["round"] == 4
    assert loaded["closest_idx"] is None
    np.testing.assert_allclose(loaded["img_recon_sim_reduced"], np.array([[0.5]], dtype=np.float32))
