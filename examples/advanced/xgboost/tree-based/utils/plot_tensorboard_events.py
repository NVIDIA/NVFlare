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
client_results_root = "./workspaces/xgboost_workspace_"
client_num_list = [5, 20]
client_pre = "app_site-"
centralized_path = "./workspaces/centralized_1_1/events.*"

# bagging and cyclic need different handle
experiments_bagging = {
    5: {
        "5_bagging_uniform_split_uniform_lr": {"tag": "AUC"},
        "5_bagging_exponential_split_uniform_lr": {"tag": "AUC"},
        "5_bagging_exponential_split_scaled_lr": {"tag": "AUC"},
    },
    20: {
        "20_bagging_uniform_split_uniform_lr": {"tag": "AUC"},
        "20_bagging_square_split_uniform_lr": {"tag": "AUC"},
        "20_bagging_square_split_scaled_lr": {"tag": "AUC"},
    },
}
experiments_cyclic = {
    5: {
        "5_cyclic_uniform_split_uniform_lr": {"tag": "AUC"},
        "5_cyclic_exponential_split_uniform_lr": {"tag": "AUC"},
    },
    20: {
        "20_cyclic_uniform_split_uniform_lr": {"tag": "AUC"},
        "20_cyclic_square_split_uniform_lr": {"tag": "AUC"},
    },
}

weight = 0.0


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value
    return smoothed


def read_eventfile(filepath, tags=["AUC"]):
    data = {}
    for summary in tf.compat.v1.train.summary_iterator(filepath):
        for v in summary.summary.value:
            if v.tag in tags:
                if v.tag in data.keys():
                    data[v.tag].append([summary.step, v.simple_value])
                else:
                    data[v.tag] = [[summary.step, v.simple_value]]
    return data


def add_eventdata(data, config, filepath, tag="AUC"):
    event_data = read_eventfile(filepath, tags=[tag])
    assert len(event_data[tag]) > 0, f"No data for key {tag}"

    metric = []
    for e in event_data[tag]:
        # print(e)
        data["Config"].append(config)
        data["Round"].append(e[0])
        metric.append(e[1])

    metric = smooth(metric, weight)
    for entry in metric:
        data["AUC"].append(entry)

    print(f"added {len(event_data[tag])} entries for {tag}")


def main():
    plt.figure()

    for client_num in client_num_list:
        plt.figure
        plt.title(f"{client_num} client experiments")
        # add event files
        data = {"Config": [], "Round": [], "AUC": []}
        # add centralized result
        eventfile = glob.glob(centralized_path, recursive=True)
        assert len(eventfile) == 1, "No unique event file found!" + eventfile
        eventfile = eventfile[0]
        print("adding", eventfile)
        add_eventdata(data, "centralized", eventfile, tag="AUC")
        # pick first client for bagging experiments
        site = 1
        for config, exp in experiments_bagging[client_num].items():
            record_path = os.path.join(client_results_root + config, "simulate_job", client_pre + str(site), "events.*")
            eventfile = glob.glob(record_path, recursive=True)
            assert len(eventfile) == 1, "No unique event file found!"
            eventfile = eventfile[0]
            print("adding", eventfile)
            add_eventdata(data, config, eventfile, tag=exp["tag"])

        # Combine all clients' records for cyclic experiments
        for site in range(1, client_num + 1):
            for config, exp in experiments_cyclic[client_num].items():
                record_path = os.path.join(
                    client_results_root + config, "simulate_job", client_pre + str(site), "events.*"
                )
                eventfile = glob.glob(record_path, recursive=True)
                assert len(eventfile) == 1, f"No unique event file found under {record_path}!"
                eventfile = eventfile[0]
                print("adding", eventfile)
                add_eventdata(data, config, eventfile, tag=exp["tag"])

        sns.lineplot(x="Round", y="AUC", hue="Config", data=data)
        plt.show()


if __name__ == "__main__":
    main()
