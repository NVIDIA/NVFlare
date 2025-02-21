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

# poc workspace
client_results_root = "../workspace_brats/"

# All sites used the same validation set for Brats, so only 1 site's record is needed
site_num = 1
client_pre = "app_site-"
sites_fl = [str(site + 1) for site in range(site_num)]

# Central vs. FedAvg vs. FedAvg_DP
experiments = {
    "brats_central": {"tag": "val_metric_global_model", "site": "All"},
    "brats_fedavg": {"tag": "val_metric_global_model"},
    "brats_fedavg_dp": {"tag": "val_metric_global_model"},
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


def add_eventdata(data, config, filepath, tag="val_metric_global_model"):
    event_data = read_eventfile(filepath, tags=[tag])
    assert len(event_data[tag]) > 0, f"No data for key {tag}"

    metric = []
    for e in event_data[tag]:
        # print(e)
        data["Config"].append(config)
        data["Epoch"].append(e[0])
        metric.append(e[1])

    metric = smooth(metric, weight)
    for entry in metric:
        data["Dice"].append(entry)

    print(f"added {len(event_data[tag])} entries for {tag}")


def main():
    plt.figure()
    num_site = len(sites_fl)
    i = 1
    # add event files

    data = {"Config": [], "Epoch": [], "Dice": []}

    for site in sites_fl:
        # clear data for each site
        data = {"Config": [], "Epoch": [], "Dice": []}
        for config, exp in experiments.items():
            spec_site = exp.get("site", None)
            if spec_site is not None:
                record_path = os.path.join(
                    client_results_root + config, "simulate_job", client_pre + spec_site, "events.*"
                )
            else:
                record_path = os.path.join(client_results_root + config, "simulate_job", client_pre + site, "events.*")

            eventfile = glob.glob(record_path, recursive=True)
            print(record_path, len(eventfile))
            assert len(eventfile) == 1, "No unique event file found!"
            eventfile = eventfile[0]
            print("adding", eventfile)
            add_eventdata(data, config, eventfile, tag=exp["tag"])

        ax = plt.subplot(1, num_site, i)
        ax.set_title(site)
        sns.lineplot(x="Epoch", y="Dice", hue="Config", data=data)
        # ax.set_xlim([0, 1000])
        i = i + 1
    plt.subplots_adjust(hspace=0.3)
    plt.show()


if __name__ == "__main__":
    main()
