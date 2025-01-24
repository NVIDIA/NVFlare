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

import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from sksurv.datasets import load_veterans_lung_cancer


def args_parser():
    parser = argparse.ArgumentParser(description="Kaplan Meier Survival Analysis Baseline")
    parser.add_argument(
        "--output_curve_path",
        type=str,
        default="/tmp/nvflare/baseline/km_curve_baseline.png",
        help="save path for the output curve",
    )
    return parser


def prepare_data(bin_days: int = 7):
    data_x, data_y = load_veterans_lung_cancer()
    total_data_num = data_x.shape[0]
    event = data_y["Status"]
    time = data_y["Survival_in_days"]
    # Categorize data to a bin, default is a week (7 days)
    time = np.ceil(time / bin_days).astype(int) * bin_days
    return event, time


def main():
    parser = args_parser()
    args = parser.parse_args()

    # Set parameters
    output_curve_path = args.output_curve_path

    # Set plot
    plt.figure()
    plt.title("Baseline")

    # Fit and plot Kaplan Meier curve with lifelines

    # Generate data with binning
    event, time = prepare_data(bin_days=7)
    kmf = KaplanMeierFitter()
    # Fit the survival data
    kmf.fit(time, event)
    # Plot and save the Kaplan-Meier survival curve
    kmf.plot_survival_function(label="Binned Weekly")

    # Generate data without binning
    event, time = prepare_data(bin_days=1)
    kmf = KaplanMeierFitter()
    # Fit the survival data
    kmf.fit(time, event)
    # Plot and save the Kaplan-Meier survival curve
    kmf.plot_survival_function(label="No binning - Daily")

    plt.ylim(0, 1)
    plt.ylabel("prob")
    plt.xlabel("time")
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_curve_path)


if __name__ == "__main__":
    main()
