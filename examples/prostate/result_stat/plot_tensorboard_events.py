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
import os

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

client_results_root = "./workspace_prostate"

# Central vs. FedAvg vs. FedProx vs. Ditto
experiments = {
    "central": {"run": "run_1", "tag": "val_metric_global_model"},
    "fedavg": {"run": "run_2", "tag": "val_metric_global_model"},
    "fedprox": {"run": "run_3", "tag": "val_metric_global_model"},
    "ditto": {"run": "run_4", "tag": "val_metric_per_model"},
}
# 6 sites
sites = ["I2CVB", "MSD", "NCI_ISBI_3T", "NCI_ISBI_Dx", "Promise12", "PROSTATEx"]

weight = 0.8


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value
    return smoothed


def read_eventfile(filepath, tags=["val_metric_global_model"]):
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


def add_eventdata(data, config, filepath, tag="val_metric_global_model"):
    event_data = read_eventfile(filepath, tags=[tag])
    assert len(event_data[tag]) > 0, f"No data for key {tag}"

    acc = []
    for e in event_data[tag]:
        # print(e)
        data["Config"].append(config)
        data["Round"].append(e[0])
        acc.append(e[1])

    acc = smooth(acc, weight)
    for entry in acc:
        data["Dice"].append(entry)

    print(f"added {len(event_data[tag])} entries for {tag}")


def main():
    plt.figure()
    num_site = len(sites)
    i = 1
    # add event files
    for site in sites:
        data = {"Config": [], "Round": [], "Dice": []}
        for config, exp in experiments.items():
            if exp["run"] == "run_1":
                file_path = os.path.join(client_results_root, "client_All", exp["run"], "app_client_All", "events.*")
            else:
                file_path = os.path.join(
                    client_results_root, "client_" + site, exp["run"], "app_client_" + site, "events.*"
                )
            eventfile = glob.glob(file_path, recursive=True)
            assert len(eventfile) == 1, "No unique event file found!"
            eventfile = eventfile[0]
            print("adding", eventfile)
            add_eventdata(data, config, eventfile, tag=exp["tag"])
        ax = plt.subplot(2, int(num_site / 2), i)
        ax.set_title(site)
        sns.lineplot(x="Round", y="Dice", hue="Config", data=data)
        ax.set_xlim([0, 150])
        i = i + 1
    plt.subplots_adjust(hspace=0.3)
    plt.show()


if __name__ == "__main__":
    main()
