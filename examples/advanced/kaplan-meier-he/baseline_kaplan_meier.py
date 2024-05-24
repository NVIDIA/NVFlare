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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sksurv.datasets import load_veterans_lung_cancer
from lifelines.utils import survival_table_from_events

def args_parser():
    parser = argparse.ArgumentParser(description="Kaplan Meier Survival Analysis Baseline")
    parser.add_argument(
        "--output_curve_path",
        type=str,
        default="./km_curve_baseline.png",
        help="save path for the output curve",
    )
    return parser

def prepare_data(site_num: int = 3, bin_days: int = 7):
    data_x, data_y = load_veterans_lung_cancer()
    # combine x and y data and save to a csv file
    data_y_df = pd.DataFrame({"Status": data_y["Status"], 'Survival_in_days': data_y["Survival_in_days"]})
    data = pd.concat([data_x, data_y_df], axis=1)
    data.to_csv("lung_cancer.csv")

    total_data_num = data_x.shape[0]
    print(f"Total data count: {total_data_num}")
    event = data_y["Status"]
    time = data_y["Survival_in_days"]
    # Categorize data to a bin, default is a week (7 days)
    time = np.ceil(time / bin_days).astype(int)
    # Shuffle data
    idx = np.random.permutation(total_data_num)
    # Split data to clients
    event_clients = {}
    time_clients = {}
    for i in range(site_num):
        start = int(i * total_data_num / site_num)
        end = int((i + 1) * total_data_num / site_num)
        event_i = event[idx[start:end]]
        time_i = time[idx[start:end]]
        event_clients["site-" + str(i + 1)] = event_i
        time_clients["site-" + str(i + 1)] = time_i
    return event_clients, time_clients

def main():
    parser = args_parser()
    args = parser.parse_args()

    # Set parameters
    output_curve_path = args.output_curve_path

    # Generate data
    event_clients, time_clients = prepare_data()

    for site in event_clients.keys():
        print(f"Site: {site}, event count: {event_clients[site].shape[0]}")
        time = time_clients[site]
        event = event_clients[site]
        event_table = survival_table_from_events(time, event)
        observed = event_table["observed"].to_numpy()
        censored = event_table["censored"].to_numpy()
        time_idx = event_table.index.to_numpy()
        # plot observed and censored data
        plt.figure()
        plt.title("Observed and Censored Data")
        plt.step(time_idx, observed, where="post", label="observed")
        plt.step(time_idx, censored, where="post", label="censored")
        plt.ylabel("count")
        plt.xlabel("time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"observed_censored_{site}.png")

    # Fit and plot Kaplan Meier curve with lifelines
    kmf = KaplanMeierFitter()
    # Fit the survival data
    kmf.fit(time, event)
    # Plot and save the Kaplan-Meier survival curve
    plt.figure()
    plt.title("Baseline")
    kmf.plot_survival_function()
    plt.ylim(0, 1)
    plt.ylabel("prob")
    plt.xlabel("time")
    plt.legend("", frameon=False)
    plt.tight_layout()
    plt.savefig(output_curve_path)


if __name__ == "__main__":
    main()
