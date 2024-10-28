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
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=str,
        default="./workspace",
        help="Workdir containing rdvl_*.npy files.",
    )
    args = parser.parse_args()

    rdvl_files = glob.glob(os.path.join(args.workspace, "**", "rdvl_*.npy"), recursive=True)
    assert len(rdvl_files) > 0, f"No RDVL files found in {args.workspace}"

    results = {
        "RDVL": [],
        "Site": [],
        "Round": [],
    }
    for rdvl_file in rdvl_files:
        _result = np.load(rdvl_file, allow_pickle=True).item()

        img_recon_sim_reduced = _result["img_recon_sim_reduced"]

        for rdvl in img_recon_sim_reduced:
            results["RDVL"].append(float(rdvl))
            results["Site"].append(_result["site"])
            results["Round"].append(_result["round"])

    # plot
    sns.lineplot(x="Round", y="RDVL", hue="Site", data=results)
    plt.plot([np.min(results["Round"]), np.max(results["Round"])], [0, 0], "k", linewidth=1.0)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
