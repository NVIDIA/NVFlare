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

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def _to_numpy_array(data):
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        return data
    if torch is not None and torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def save_rdlv_results(
    save_path: str,
    img_recon_sim_reduced,
    img_recon_sim,
    closest_idx,
    site: str,
    round_number: int,
    best_matches=None,
):
    save_kwargs = {
        "img_recon_sim_reduced": _to_numpy_array(img_recon_sim_reduced),
        "img_recon_sim": _to_numpy_array(img_recon_sim),
        "has_closest_idx": np.asarray(closest_idx is not None, dtype=np.bool_),
        "site": np.asarray(site),
        "round": np.asarray(round_number),
    }

    if closest_idx is not None:
        save_kwargs["closest_idx"] = _to_numpy_array(closest_idx)

    if best_matches:
        save_kwargs["best_match_inputs"] = np.stack([_to_numpy_array(match[0]) for match in best_matches])
        save_kwargs["best_match_recons"] = np.stack([_to_numpy_array(match[1]) for match in best_matches])

    np.savez(save_path, **save_kwargs)


def load_rdlv_results(result_path: str):
    with np.load(result_path, allow_pickle=False) as result:
        loaded = {
            "img_recon_sim_reduced": result["img_recon_sim_reduced"],
            "img_recon_sim": result["img_recon_sim"],
            "closest_idx": result["closest_idx"] if result["has_closest_idx"].item() else None,
            "site": result["site"].item(),
            "round": result["round"].item(),
        }

        if "best_match_inputs" in result.files:
            loaded["best_match_inputs"] = result["best_match_inputs"]
            loaded["best_match_recons"] = result["best_match_recons"]

    return loaded
