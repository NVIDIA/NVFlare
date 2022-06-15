# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
client_results_root = "../workspace_brats"

# All sites used the same validation set, so only 1 site's record is needed
site_num = 1
site_pre = "site-"

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


def find_job_id(workdir, fl_app_name="prostate_central"):
    """Find the first matching experiment"""
    target_path = os.path.join(workdir, "*", "fl_app.txt")
    fl_app_files = glob.glob(target_path, recursive=True)
    assert len(fl_app_files) > 0, f"No `fl_app.txt` files found in workdir={workdir}."
    for fl_app_file in fl_app_files:
        with open(fl_app_file, "r") as f:
            _fl_app_name = f.read()
        if fl_app_name == _fl_app_name:  # alpha will be matched based on value in config file
            job_id = os.path.basename(os.path.dirname(fl_app_file))
            return job_id
    raise ValueError(f"No job id found for fl_app_name={fl_app_name} in workdir={workdir}")


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
    i = 1
    # add event files
    data = {"Config": [], "Epoch": [], "Dice": []}
    for site in range(site_num):
        # clear data for each site
        site = site + 1
        data = {"Config": [], "Epoch": [], "Dice": []}
        for config, exp in experiments.items():
            job_id = find_job_id(workdir=client_results_root+"/site-1", fl_app_name=config)
            print(f"Found run {job_id} for {config}")
            spec_site = exp.get("site", None)
            if spec_site is not None:
                record_path = os.path.join(client_results_root, site_pre + spec_site, job_id, "*", "events.*")
            else:
                record_path = os.path.join(client_results_root, site_pre + str(site), job_id, "*", "events.*")
            eventfile = glob.glob(record_path, recursive=True)
            assert len(eventfile) == 1, "No unique event file found!"
            eventfile = eventfile[0]
            print("adding", eventfile)
            add_eventdata(data, config, eventfile, tag=exp["tag"])

        ax = plt.subplot(1, site_num, i)
        ax.set_title(site)
        sns.lineplot(x="Epoch", y="Dice", hue="Config", data=data)
        i = i + 1
    plt.subplots_adjust(hspace=0.3)
    plt.show()


if __name__ == "__main__":
    main()
