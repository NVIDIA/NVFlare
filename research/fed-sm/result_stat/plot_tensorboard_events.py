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
import os

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# simulator workspace
client_results_root = "../workspaces/fedsm_prostate/"
client_pre = "app_client_"

# 3 sites
sites_fl = ["I2CVB", "MSD", "NCI_ISBI_3T"]

metrics = {
    "global_model": "val_metric_global_model",
    "personal_model": "val_metric_person_model",
    "selector_model": "val_metric_select_model",
}

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
                if v.tag in data.keys():
                    data[v.tag].append([summary.step, v.simple_value])
                else:
                    data[v.tag] = [[summary.step, v.simple_value]]
    return data


def add_eventdata(data, site, filepath, tag="val_metric_global_model"):
    event_data = read_eventfile(filepath, tags=[tag])
    assert len(event_data[tag]) > 0, f"No data for key {tag}"

    metric = []
    for e in event_data[tag]:
        # print(e)
        data["Site"].append(site)
        data["Epoch"].append(e[0])
        metric.append(e[1])

    metric = smooth(metric, weight)
    for entry in metric:
        data["Metric"].append(entry)

    print(f"added {len(event_data[tag])} entries for {tag}")


def main():
    plt.figure()
    num_metric = len(metrics)
    i = 1

    for metric in metrics.keys():
        # clear data for each metric
        data = {"Site": [], "Epoch": [], "Metric": []}
        for site in sites_fl:
            record_path = os.path.join(client_results_root, "simulate_job", client_pre + site, "events.*")
            eventfile = glob.glob(record_path, recursive=True)
            assert len(eventfile) == 1, "No unique event file found!"
            eventfile = eventfile[0]
            print("adding", eventfile)
            add_eventdata(data, site, eventfile, tag=metrics[metric])

        ax = plt.subplot(1, int(num_metric), i)
        ax.set_title(metric)
        sns.lineplot(x="Epoch", y="Metric", hue="Site", data=data)
        if metric == "selector_model":
            plt.ylabel("Accuracy (%)")
        else:
            plt.ylabel("Dice")
        i = i + 1
    plt.subplots_adjust(hspace=0.3)
    plt.show()


if __name__ == "__main__":
    main()
