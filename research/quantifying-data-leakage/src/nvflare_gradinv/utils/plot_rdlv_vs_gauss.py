# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspaces",
        type=str,
        default="./workspaces",
        help="Workdir containing rdvl_*.npy files.",
    )
    args = parser.parse_args()

    rdvl_files = glob.glob(os.path.join(args.workspaces, "**", "rdvl_*.npy"), recursive=True)
    assert len(rdvl_files) > 0, f"No RDVL files found in {args.workspace}"

    results = {"RDVL": [], "Site": [], "Round": [], "sigma0": [], "test_accuracy": []}
    for rdvl_file in rdvl_files:
        _result = np.load(rdvl_file, allow_pickle=True).item()

        img_recon_sim_reduced = _result["img_recon_sim_reduced"]

        # read sigma0 from client config
        client_config_file = os.path.join(os.path.dirname(rdvl_file), "config", "config_fed_client.json")
        with open(client_config_file, "r") as f:
            client_config = json.load(f)

        gaussian_filter = client_config["task_result_filters"][0]["filters"][0]
        assert (
            "GaussianPrivacy" in gaussian_filter["path"]
        ), f"Expected filter to GaussianPrivacy but got {gaussian_filter['path']}"
        sigma0 = gaussian_filter["args"]["sigma0"]

        # read best global model accuracy from cross-site validation
        cross_val_file = os.path.join(
            os.path.dirname(os.path.dirname(rdvl_file)), "cross_site_val", "cross_val_results.json"
        )
        with open(cross_val_file, "r") as f:
            cross_val = json.load(f)
        best_model_perfrom = cross_val["site-1"]["SRV_best_FL_global_model.pt"]

        for rdvl in img_recon_sim_reduced:
            results["RDVL"].append(float(rdvl))
            results["Site"].append(_result["site"])
            results["Round"].append(_result["round"])
            results["sigma0"].append(float(sigma0))
            results["test_accuracy"].append(best_model_perfrom["test_accuracy"])

    # plot RDVL
    sns.lineplot(x="sigma0", y="RDVL", hue="Site", data=results)
    plt.grid(True)
    plt.xlabel("Gaussian Privacy ($\sigma_0$)")
    plt.plot([np.min(results["sigma0"]), np.max(results["sigma0"])], [0, 0], "k", linewidth=1.0)

    # plot accuracy
    ax2 = plt.twinx()
    sns.lineplot(x="sigma0", y="test_accuracy", data=results, color="tab:gray", ax=ax2)
    ax2.lines[0].set_linestyle("--")
    plt.grid(False)
    plt.ylim([0.0, 1.0])
    plt.ylabel("Testing Accuracy")
    plt.legend(["Testing Accuracy"], loc="lower left")

    plt.show()


if __name__ == "__main__":
    main()
