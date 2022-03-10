# Copyright (c) 2021, NVIDIA CORPORATION.
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

client_results_root = "./workspaces/secure_workspace/site-1"
server_results_root = "./workspaces/secure_workspace/localhost"

# 4.1 Central vs. FedAvg
experiments = {
    "cifar10_central": {"run": "run_1", "tag": "val_acc_local_model"},
    "cifar10_fedavg": {"run": "run_2", "tag": "val_acc_global_model"},
    "cifar10_fedavg_he": {"run": "run_9", "tag": "val_acc_global_model"},
}

# # 4.2 Impact of client data heterogeneity
# experiments = {"cifar10_fedavg (alpha=1.0)": {"run": "run_2", "tag": "val_acc_global_model"},
#                 "cifar10_fedavg (alpha=0.5)": {"run": "run_3", "tag": "val_acc_global_model"},
#                 "cifar10_fedavg (alpha=0.3)": {"run": "run_4", "tag": "val_acc_global_model"},
#                 "cifar10_fedavg (alpha=0.1)": {"run": "run_5", "tag": "val_acc_global_model"}}
#
# # 4.3 FedProx vs. FedOpt vs. SCAFFOLD
# experiments = {"cifar10_fedavg": {"run": "run_5", "tag": "val_acc_global_model"},
#                "cifar10_fedprox": {"run": "run_6", "tag": "val_acc_global_model"},
#                "cifar10_fedopt": {"run": "run_7", "tag": "val_acc_global_model"},
#                "cifar10_scaffold": {"run": "run_8", "tag": "val_acc_global_model"}}

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
        eventfile = glob.glob(os.path.join(client_results_root, exp["run"] + "/**/events.*"), recursive=True)
        assert len(eventfile) == 1, "No unique event file found!"
        eventfile = eventfile[0]
        print("adding", eventfile)
        add_eventdata(data, config, eventfile, tag=exp["tag"])

        if add_cross_site_val:
            xsite_file = glob.glob(
                os.path.join(server_results_root, exp["run"] + "/**/cross_val_results.json"), recursive=True
            )
            assert len(xsite_file) == 1, "No unique x-site file found!"
            with open(xsite_file[0], "r") as f:
                xsite_results = json.load(f)

            xsite_data["Config"].append(config)
            for k in xsite_keys:
                xsite_data[k].append(xsite_results["site-1"][k]["val_accuracy"])

    print("Training TB data:")
    print(pd.DataFrame(data))

    if xsite_data:
        print("Cross-site val data:")
        print(pd.DataFrame(xsite_data))

    sns.lineplot(x="Step", y="Accuracy", hue="Config", data=data)
    plt.show()


if __name__ == "__main__":
    main()
