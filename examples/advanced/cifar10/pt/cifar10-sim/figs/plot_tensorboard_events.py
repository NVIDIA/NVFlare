# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import glob
import json
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import our standalone tfevents reader (avoids C++ mutex issues)
from tfevents_reader import read_tfevents_file

logging.basicConfig(level=logging.ERROR)

# secure workspace
client_results_root = "/tmp/nvflare/simulation"

# 4.1 Central vs. FedAvg
experiments = {
    "cifar10_central": {"tag": "val_acc_local_model"},
    "cifar10_fedavg": {"tag": "val_acc_global_model", "alpha": 1.0},
    "save_path": "figs/central_vs_fedavg.png",
}

# # 4.2 Impact of client data heterogeneity
# experiments = {"cifar10_fedavg (alpha=1.0)": {"tag": "val_acc_global_model", "alpha": 1.0},
#               "cifar10_fedavg (alpha=0.5)": {"tag": "val_acc_global_model", "alpha": 0.5},
#               "cifar10_fedavg (alpha=0.3)": {"tag": "val_acc_global_model", "alpha": 0.3},
#               "cifar10_fedavg (alpha=0.1)": {"tag": "val_acc_global_model", "alpha": 0.1},
#               "save_path": "figs/fedavg_alpha.png"
# }

# # 4.3 FedProx vs. FedOpt vs. SCAFFOLD
# experiments = {"cifar10_fedavg": {"tag": "val_acc_global_model", "alpha": 0.1},
#               "cifar10_fedprox": {"tag": "val_acc_global_model", "alpha": 0.1},
#               "cifar10_fedopt": {"tag": "val_acc_global_model", "alpha": 0.1},
#               "cifar10_scaffold": {"tag": "val_acc_global_model", "alpha": 0.1},
#               "save_path": "figs/fedopt_fedprox_scaffold.png"
# }

# 5.4 Custom Aggregators Comparison
# experiments = {"cifar10_custom_default": {"tag": "val_acc_global_model", "alpha": 0.1},
#               "cifar10_custom_weighted": {"tag": "val_acc_global_model", "alpha": 0.1},
#               "cifar10_custom_median": {"tag": "val_acc_global_model", "alpha": 0.1},
#               "save_path": "figs/custom_aggregators.png"
# }

add_cross_site_val = False


def read_eventfile(filepath, tags=["val_acc_global_model"]):
    """
    Read TensorBoard event file using pure Python protobuf parsing.
    This avoids the C++ mutex locking issues in TensorFlow.
    """
    try:
        data = read_tfevents_file(filepath, tags=tags)
        return data
    except Exception as e:
        print(f"Warning: Error reading {filepath}: {e}")
        import traceback

        traceback.print_exc()
        return {}


def add_eventdata(data, config, filepath, tag="val_acc_global_model"):
    print(f"Reading {os.path.basename(filepath)}...")
    event_data = read_eventfile(filepath, tags=[tag])

    if tag not in event_data or len(event_data[tag]) == 0:
        print(f"  Warning: No data for tag '{tag}' in {filepath}")
        return

    # Add data to the collection
    for e in event_data[tag]:
        data["Config"].append(config)
        data["Step"].append(e[0])
        data["Accuracy"].append(e[1])
    print(f"  ✓ Added {len(event_data[tag])} entries for {tag}")


def main():
    data = {"Config": [], "Step": [], "Accuracy": []}

    if add_cross_site_val:
        xsite_keys = ["SRV_FL_global_model.pt", "SRV_best_FL_global_model.pt"]
        xsite_data = {"Config": []}
        for k in xsite_keys:
            xsite_data.update({k: []})
    else:
        xsite_data = None
        xsite_keys = None

    save_path = experiments.get("save_path", "figs/accuracy_plot.png")

    # add event files
    print("=" * 60)
    print("Processing experiments...")
    print("=" * 60)

    for config, exp in experiments.items():
        if not isinstance(exp, dict):
            continue
        config_name = config.split(" ")[0]
        alpha = exp.get("alpha", None)
        if alpha is not None:
            config_name = config_name + f"*alpha{alpha}"

        eventfiles = glob.glob(
            os.path.join(client_results_root, config_name, "**", "site-1", "events.*"), recursive=True
        )
        assert len(eventfiles) > 0, f"No event file found in {os.path.join(client_results_root, config_name)}!"

        # Sort by modification time and use the most recent one
        eventfiles.sort(key=os.path.getmtime, reverse=True)
        eventfile = eventfiles[0]

        print(f"\n[{config}]")
        add_eventdata(data, config, eventfile, tag=exp["tag"])

        if add_cross_site_val:
            xsite_file = glob.glob(
                os.path.join(client_results_root, config_name, "**", "cross_val_results.json"), recursive=True
            )
            assert len(xsite_file) == 1, "No unique x-site file found!"
            with open(xsite_file[0], "r") as f:
                xsite_results = json.load(f)

            xsite_data["Config"].append(config)
            for k in xsite_keys:
                try:
                    xsite_data[k].append(xsite_results["site-1"][k]["val_accuracy"])
                except Exception as e:
                    xsite_data[k].append(None)
                    print(f"Warning: No val_accuracy for {k} in {xsite_file}!")

    print("\n" + "=" * 60)
    print("Training TB data (Max Accuracy per Config):")
    print("=" * 60)
    print(pd.DataFrame(data).groupby("Config")["Accuracy"].max())

    if xsite_data:
        print("Cross-site val data:")
        print(pd.DataFrame(xsite_data))

    sns.lineplot(x="Step", y="Accuracy", hue="Config", data=data)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved plot to {save_path}")


if __name__ == "__main__":
    main()
