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

import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

# secure workspace
client_results_root = "/tmp/nvflare/sim_cifar10"

# poc workspace
# client_results_root = "./workspaces/poc_workspace/site-1"
# download_dir = "./workspaces/poc_workspace/admin/transfer"

# 4.1 Central vs. FedAvg
experiments = {
    "cifar10_central": {"tag": "val_acc_local_model", "alpha": 0.0},
    "cifar10_fedavg": {"tag": "val_acc_global_model", "alpha": 1.0},
}

# # 4.2 Impact of client data heterogeneity
# experiments = {"cifar10_fedavg (alpha=1.0)": {"tag": "val_acc_global_model", "alpha": 1.0},
#               "cifar10_fedavg (alpha=0.5)": {"tag": "val_acc_global_model", "alpha": 0.5},
#               "cifar10_fedavg (alpha=0.3)": {"tag": "val_acc_global_model", "alpha": 0.3},
#               "cifar10_fedavg (alpha=0.1)": {"tag": "val_acc_global_model", "alpha": 0.1}
# }

# # 4.3 FedProx vs. FedOpt vs. SCAFFOLD
# experiments = {"cifar10_fedavg": {"tag": "val_acc_global_model", "alpha": 0.1},
#               "cifar10_fedprox": {"tag": "val_acc_global_model", "alpha": 0.1},
#               "cifar10_fedopt": {"tag": "val_acc_global_model", "alpha": 0.1},
#               "cifar10_scaffold": {"tag": "val_acc_global_model", "alpha": 0.1}
# }

add_cross_site_val = True


def read_eventfile(filepath, tags=["val_acc_global_model"]):
    data = {}
    for summary in tf.compat.v1.train.summary_iterator(filepath):
        for v in summary.summary.value:
            if v.tag in tags:
                # print(v.tag, summary.step, v.simple_value)
                if v.tag in data.keys():
                    data[v.tag].append([summary.step, v.simple_value])
                else:
                    data[v.tag] = [[summary.step, v.simple_value]]
    return data


def add_eventdata(data, config, filepath, tag="val_acc_global_model"):
    event_data = read_eventfile(filepath, tags=[tag])

    assert len(event_data[tag]) > 0, f"No data for key {tag}"
    # print(event_data)
    for e in event_data[tag]:
        # print(e)
        data["Config"].append(config)
        data["Step"].append(e[0])
        data["Accuracy"].append(e[1])
    print(f"added {len(event_data[tag])} entries for {tag}")


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

    # add event files
    for config, exp in experiments.items():
        config_name = config.split(" ")[0]
        alpha = exp.get("alpha", None)
        if alpha is not None:
            config_name = config_name + f"*alpha{alpha}"
        else:
            raise ValueError(f"Expected an alpha value to be provided but got alpha={alpha}")
        eventfile = glob.glob(
            os.path.join(client_results_root, config_name, "**", "app_site-1", "events.*"), recursive=True
        )
        assert len(eventfile) == 1, f"No unique event file found in {os.path.join(client_results_root, config_name)}!"
        eventfile = eventfile[0]
        print("adding", eventfile)
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

    print("Training TB data:")
    print(pd.DataFrame(data))

    if xsite_data:
        print("Cross-site val data:")
        print(pd.DataFrame(xsite_data))

    sns.lineplot(x="Step", y="Accuracy", hue="Config", data=data)
    plt.show()


if __name__ == "__main__":
    main()
