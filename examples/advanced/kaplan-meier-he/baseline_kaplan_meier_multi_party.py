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

import argparse
import copy

import matplotlib.pyplot as plt
import numpy as np
import tenseal as ts
from lifelines import KaplanMeierFitter
from lifelines.utils import survival_table_from_events
from sksurv.datasets import load_veterans_lung_cancer


def args_parser():
    parser = argparse.ArgumentParser(description="Kaplan Meier Survival Analysis")
    parser.add_argument("--num_of_clients", type=int, default=5, help="number of clients")
    parser.add_argument("--he", action="store_true", help="use homomorphic encryption")
    parser.add_argument(
        "--output_curve_path",
        type=str,
        default="./km_curve_multi_party.png",
        help="save path for the output curve",
    )
    return parser


def prepare_data(num_of_clients: int, bin_days: int = 7):
    # Load data
    data_x, data_y = load_veterans_lung_cancer()
    # Get total data count
    total_data_num = data_x.shape[0]
    print(f"Total data count: {total_data_num}")
    # Get event and time
    event = data_y["Status"]
    time = data_y["Survival_in_days"]
    # Categorize data to a bin, default is a week (7 days)
    time = np.ceil(time / bin_days).astype(int)
    # Shuffle data
    idx = np.random.permutation(total_data_num)
    # Split data to clients
    event_clients = {}
    time_clients = {}
    for i in range(num_of_clients):
        start = int(i * total_data_num / num_of_clients)
        end = int((i + 1) * total_data_num / num_of_clients)
        event_i = event[idx[start:end]]
        time_i = time[idx[start:end]]
        event_clients[i] = event_i
        time_clients[i] = time_i
    return event, time, event_clients, time_clients


def main():
    parser = args_parser()
    args = parser.parse_args()

    # Set parameters
    num_of_clients = args.num_of_clients
    he = args.he
    output_curve_path = args.output_curve_path

    # Generate data
    event, time, event_clients, time_clients = prepare_data(num_of_clients)

    # Setup Plot
    plt.figure()
    if he:
        total_subplot = 3
    else:
        total_subplot = 2

    # Setup tenseal context
    # using BFV scheme since observations are integers
    if he:
        context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193)

    # Fit and plot Kaplan Meier curve with lifelines
    kmf = KaplanMeierFitter()
    kmf.fit(time, event)
    # Plot the Kaplan-Meier survival curve
    plt.subplot(1, total_subplot, 1)
    plt.title("Centralized")
    kmf.plot_survival_function()
    plt.ylim(0, 1)
    plt.ylabel("prob")
    plt.xlabel("time")
    plt.legend("", frameon=False)

    # Distributed
    # Stage 1 local: collect info and set histogram dict
    event_table = {}
    max_week_idx = []
    for client in range(num_of_clients):
        # condense date to histogram
        event_table[client] = survival_table_from_events(time_clients[client], event_clients[client])
        week_idx = event_table[client].index.values.astype(int)
        # get the max week index
        max_week_idx.append(max(week_idx))
    # Stage 1 global: get global histogram dict
    hist_obs_global = {}
    hist_sen_global = {}
    # actual week as index, so plus 1
    max_week = max(max_week_idx) + 1
    for week in range(max_week):
        hist_obs_global[week] = 0
        hist_sen_global[week] = 0
    if he:
        # encrypt with tenseal
        hist_obs_global_he = ts.bfv_vector(context, list(hist_obs_global.values()))
        hist_sen_global_he = ts.bfv_vector(context, list(hist_sen_global.values()))
    # Stage 2 local: convert local table to uniform histogram
    hist_obs_local = {}
    hist_sen_local = {}
    hist_obs_local_he = {}
    hist_sen_local_he = {}
    for client in range(num_of_clients):
        hist_obs_local[client] = copy.deepcopy(hist_obs_global)
        hist_sen_local[client] = copy.deepcopy(hist_sen_global)
        # assign values
        week_idx = event_table[client].index.values.astype(int)
        observed = event_table[client]["observed"].to_numpy()
        sensored = event_table[client]["censored"].to_numpy()
        for i in range(len(week_idx)):
            hist_obs_local[client][week_idx[i]] = observed[i]
            hist_sen_local[client][week_idx[i]] = sensored[i]
        if he:
            # encrypt with tenseal using BFV scheme since observations are integers
            hist_obs_local_he[client] = ts.bfv_vector(context, list(hist_obs_local[client].values()))
            hist_sen_local_he[client] = ts.bfv_vector(context, list(hist_sen_local[client].values()))
    # Stage 2 global: sum up local histogram
    for client in range(num_of_clients):
        for week in range(max_week):
            hist_obs_global[week] += hist_obs_local[client][week]
            hist_sen_global[week] += hist_sen_local[client][week]
        if he:
            hist_obs_global_he += hist_obs_local_he[client]
            hist_sen_global_he += hist_sen_local_he[client]

    # Stage 3 local: convert histogram to event list and fit K-M curve
    # unfold histogram to event list
    time_unfold = []
    event_unfold = []
    for i in range(max_week):
        for j in range(hist_obs_global[i]):
            time_unfold.append(i)
            event_unfold.append(True)
        for k in range(hist_sen_global[i]):
            time_unfold.append(i)
            event_unfold.append(False)
    time_unfold = np.array(time_unfold)
    event_unfold = np.array(event_unfold)
    # Fit the survival data with lifelines
    kmf.fit(time_unfold, event_unfold)
    # Plot the Kaplan-Meier survival curve
    plt.subplot(1, total_subplot, 2)
    plt.title("Federated")
    kmf.plot_survival_function()
    plt.ylim(0, 1)
    plt.ylabel("prob")
    plt.xlabel("time")
    plt.legend("", frameon=False)

    if he:
        # decrypt with tenseal
        hist_obs_global_he = hist_obs_global_he.decrypt()
        hist_sen_global_he = hist_sen_global_he.decrypt()
        # unfold histogram to event list
        time_unfold = []
        event_unfold = []
        for i in range(max_week):
            for j in range(hist_obs_global_he[i]):
                time_unfold.append(i)
                event_unfold.append(True)
            for k in range(hist_sen_global_he[i]):
                time_unfold.append(i)
                event_unfold.append(False)
        time_unfold = np.array(time_unfold)
        event_unfold = np.array(event_unfold)
        # Fit the survival data with lifelines
        kmf.fit(time_unfold, event_unfold)
        # Plot the Kaplan-Meier survival curve
        plt.subplot(1, total_subplot, 3)
        plt.title("Federated HE")
        kmf.plot_survival_function()
        plt.ylim(0, 1)
        plt.ylabel("prob")
        plt.xlabel("time")
        plt.legend("", frameon=False)

    # Save curve
    plt.tight_layout()
    plt.savefig(output_curve_path)


if __name__ == "__main__":
    main()
