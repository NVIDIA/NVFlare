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
client_results_root = "./workspaces/secure_workspace/site-1"
download_dir = "./workspaces/secure_workspace/admin@nvidia.com/transfer"

# poc workspace
# client_results_root = "./workspaces/poc_workspace/site-1"
# download_dir = "./workspaces/poc_workspace/admin/transfer"

# 4.1 Central vs. FedAvg
experiments = {
    "cifar10_fedavg_stream_tb": {"tag": "val_acc_global_model", "alpha": 1.0},
    "cifar10_fedavg_he": {"tag": "val_acc_global_model", "alpha": 1.0},
}

add_cross_site_val = True


def find_job_id(workdir, fl_app_name="cifar10_fedavg", alpha=None):
    """Find the first matching experiment"""
    # TODO: return several experiment job_ids with matching settings
    fl_app_files = glob.glob(os.path.join(workdir, "**", "fl_app.txt"), recursive=True)
    assert len(fl_app_files) > 0, f"No `fl_app.txt` files found in workdir={workdir}."
    for fl_app_file in fl_app_files:
        with open(fl_app_file, "r") as f:
            _fl_app_name = f.read()
        if fl_app_name == _fl_app_name:  # alpha will be matched based on value in config file
            job_id = os.path.basename(
                os.path.dirname(os.path.dirname(os.path.join(fl_app_file)))
            )  # skip "workspace" subfolder
            if alpha is not None:
                config_fed_server_file = glob.glob(
                    os.path.join(os.path.dirname(fl_app_file), "**", "config_fed_server.json"), recursive=True
                )
                assert (
                    len(config_fed_server_file) == 1
                ), f"No unique server config found in {os.path.dirname(fl_app_file)}"
                with open(config_fed_server_file[0], "r") as f:
                    server_config = json.load(f)
                _alpha = server_config["alpha"]
                if _alpha == alpha:
                    return job_id
            else:
                return job_id
    raise ValueError(f"No job id found for fl_app_name={fl_app_name} in workdir={workdir}")


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
        job_id = find_job_id(workdir=download_dir, fl_app_name=config_name, alpha=alpha)
        print(f"Found run {job_id} for {config_name} with alpha={alpha}")
        eventfile = glob.glob(os.path.join(client_results_root, job_id, "**", "events.*"), recursive=True)
        assert len(eventfile) == 1, "No unique event file found!"
        eventfile = eventfile[0]
        print("adding", eventfile)
        add_eventdata(data, config, eventfile, tag=exp["tag"])

        if add_cross_site_val:
            xsite_file = glob.glob(os.path.join(download_dir, job_id, "**", "cross_val_results.json"), recursive=True)
            assert len(xsite_file) == 1, "No unique x-site file found!"
            with open(xsite_file[0], "r") as f:
                xsite_results = json.load(f)

            xsite_data["Config"].append(config)
            for k in xsite_keys:
                try:
                    xsite_data[k].append(xsite_results["site-1"][k]["val_accuracy"])
                except Exception as e:
                    raise ValueError(f"No val_accuracy for {k} in {xsite_file}!")

    print("Training TB data:")
    print(pd.DataFrame(data))

    if xsite_data:
        print("Cross-site val data:")
        print(pd.DataFrame(xsite_data))

    sns.lineplot(x="Step", y="Accuracy", hue="Config", data=data)
    plt.show()


if __name__ == "__main__":
    main()
