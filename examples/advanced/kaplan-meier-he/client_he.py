# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import base64
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tenseal as ts
from lifelines import KaplanMeierFitter
from lifelines.utils import survival_table_from_events

# (1) import nvflare client API
import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType


# Client code
def read_data(file_name: str):
    with open(file_name, "rb") as f:
        data = f.read()
    return base64.b64decode(data)


def details_save(kmf):
    # Get the survival function at all observed time points
    survival_function_at_all_times = kmf.survival_function_
    # Get the timeline (time points)
    timeline = survival_function_at_all_times.index.values
    # Get the KM estimate
    km_estimate = survival_function_at_all_times["KM_estimate"].values
    # Get the event count at each time point
    event_count = kmf.event_table.iloc[:, 0].values  # Assuming the first column is the observed events
    # Get the survival rate at each time point (using the 1st column of the survival function)
    survival_rate = 1 - survival_function_at_all_times.iloc[:, 0].values
    # Return the results
    results = {
        "timeline": timeline.tolist(),
        "km_estimate": km_estimate.tolist(),
        "event_count": event_count.tolist(),
        "survival_rate": survival_rate.tolist(),
    }
    file_path = os.path.join(os.getcwd(), "km_global.json")
    print(f"save the details of KM analysis result to {file_path} \n")
    with open(file_path, "w") as json_file:
        json.dump(results, json_file, indent=4)


def plot_and_save(kmf):
    # Plot and save the Kaplan-Meier survival curve
    plt.figure()
    plt.title("Federated HE")
    kmf.plot_survival_function()
    plt.ylim(0, 1)
    plt.ylabel("prob")
    plt.xlabel("time")
    plt.legend("", frameon=False)
    plt.tight_layout()
    file_path = os.path.join(os.getcwd(), "km_curve_fl_he.png")
    print(f"save the curve plot to {file_path} \n")
    plt.savefig(file_path)


def main():
    parser = argparse.ArgumentParser(description="KM analysis")
    parser.add_argument("--data_root", type=str, help="Root path for data files")
    parser.add_argument("--he_context_path", type=str, help="Path for the HE context file")
    args = parser.parse_args()

    flare.init()

    site_name = flare.get_site_name()
    print(f"Kaplan-meier analysis for {site_name}")

    # get local data
    data_path = os.path.join(args.data_root, site_name + ".csv")
    data = pd.read_csv(data_path)
    event_local = data["event"]
    time_local = data["time"]

    # HE context
    # In real-life application, HE context is prepared by secure provisioning
    he_context_serial = read_data(args.he_context_path)
    he_context = ts.context_from(he_context_serial)

    while flare.is_running():
        # receives global message from NVFlare
        global_msg = flare.receive()
        curr_round = global_msg.current_round
        print(f"current_round={curr_round}")

        if curr_round == 1:
            # First round:
            # Empty payload from server, send max index back
            # Condense local data to histogram
            event_table = survival_table_from_events(time_local, event_local)
            hist_idx = event_table.index.values.astype(int)
            # Get the max index to be synced globally
            max_hist_idx = max(hist_idx)

            # Send max to server
            print(f"send max hist index for site = {flare.get_site_name()}")
            model = FLModel(params={"max_idx": max_hist_idx}, params_type=ParamsType.FULL)
            flare.send(model)

        elif curr_round == 2:
            # Second round, get global max index
            # Organize local histogram and encrypt
            max_idx_global = global_msg.params["max_idx_global"]
            print("Global Max Idx")
            print(max_idx_global)
            # Convert local table to uniform histogram
            hist_obs = {}
            hist_cen = {}
            for idx in range(max_idx_global):
                hist_obs[idx] = 0
                hist_cen[idx] = 0
            # assign values
            idx = event_table.index.values.astype(int)
            observed = event_table["observed"].to_numpy()
            censored = event_table["censored"].to_numpy()
            for i in range(len(idx)):
                hist_obs[idx[i]] = observed[i]
                hist_cen[idx[i]] = censored[i]
            # Encrypt with tenseal using BFV scheme since observations are integers
            hist_obs_he = ts.bfv_vector(he_context, list(hist_obs.values()))
            hist_cen_he = ts.bfv_vector(he_context, list(hist_cen.values()))
            # Serialize for transmission
            hist_obs_he_serial = hist_obs_he.serialize()
            hist_cen_he_serial = hist_cen_he.serialize()
            # Send encrypted histograms to server
            response = FLModel(
                params={"hist_obs": hist_obs_he_serial, "hist_cen": hist_cen_he_serial}, params_type=ParamsType.FULL
            )
            flare.send(response)

        elif curr_round == 3:
            # Get global histograms
            hist_obs_global_serial = global_msg.params["hist_obs_global"]
            hist_cen_global_serial = global_msg.params["hist_cen_global"]
            # Deserialize
            hist_obs_global = ts.bfv_vector_from(he_context, hist_obs_global_serial)
            hist_cen_global = ts.bfv_vector_from(he_context, hist_cen_global_serial)
            # Decrypt
            hist_obs_global = hist_obs_global.decrypt()
            hist_cen_global = hist_cen_global.decrypt()
            # Unfold histogram to event list
            time_unfold = []
            event_unfold = []
            for i in range(max_idx_global):
                for j in range(hist_obs_global[i]):
                    time_unfold.append(i)
                    event_unfold.append(True)
                for k in range(hist_cen_global[i]):
                    time_unfold.append(i)
                    event_unfold.append(False)
            time_unfold = np.array(time_unfold)
            event_unfold = np.array(event_unfold)

            # Perform Kaplan-Meier analysis on global aggregated information
            # Create a Kaplan-Meier estimator
            kmf = KaplanMeierFitter()

            # Fit the model
            kmf.fit(durations=time_unfold, event_observed=event_unfold)

            # Plot and save the KM curve
            plot_and_save(kmf)

            # Save details of the KM result to a json file
            details_save(kmf)

            # Send a simple response to server
            response = FLModel(params={}, params_type=ParamsType.FULL)
            flare.send(response)

    print(f"finish send for {site_name}, complete")


if __name__ == "__main__":
    main()
