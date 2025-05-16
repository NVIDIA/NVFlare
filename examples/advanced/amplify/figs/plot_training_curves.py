#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer


def load_tensorboard_data(log_dir, tag, out_metric):
    """
    Load data from TensorBoard event files.

    Args:
        log_dir (str): Directory containing TensorBoard event files
        tag: tag to extract.
        out_metric: metric name to record.
    Returns:
        dict: Dictionary mapping run names to dictionaries of tag data
    """
    # Find all event files in the directory and subdirectories
    event_files = glob.glob(os.path.join(log_dir, "**", "event*"), recursive=True)

    if not event_files:
        raise ValueError(f"No event files found in {log_dir}")

    # Create an EventMultiplexer to handle multiple event files
    multiplexer = EventMultiplexer()

    # Add all event files to the multiplexer
    for event_file in event_files:
        task_name = event_file.split(os.sep)[-4]

        if "fedavg" in event_file:
            run_name = f"fedavg/{task_name}"
            if "private_regressors" in event_file:
                run_name = f"fedavg_private_regressors/{task_name}"
        elif "local" in event_file:
            run_name = f"local/{task_name}"

        multiplexer.AddRun(event_file, name=run_name)

    # Reload the multiplexer to load all the data
    multiplexer.Reload()

    # Extract data for each run and tag
    data = {}

    for run_name in multiplexer.Runs():
        mode = run_name.split("/")[0]
        task_name = run_name.split("/")[1]

        task_data = {"Mode": [], "Step": [], out_metric: []}
        for event in multiplexer.Scalars(run_name, tag=f"{task_name}/{tag}"):
            task_data["Mode"].append(mode)
            task_data[out_metric].append(event.value)
            task_data["Step"].append(event.step)

        if task_name not in data:
            data[task_name] = task_data
        else:
            for k in data[task_name]:
                data[task_name][k].extend(task_data[k])
    return data


def plot_metrics(
    data,
    output_dir=None,
    title=None,
    figsize=(16, 12),
    y_limits=None,
    x_limits=None,
    out_metric="RMSE",
    smoothing_window=10,
):
    """
    Plot metrics from multiple TensorBoard runs for comparison.

    Args:
        data (dict): Dictionary mapping run names to dictionaries of tag data
        output_dir (str): Directory to save plots. If None, plots will be displayed.
        title (str): Title for the plot. If None, a default title will be used.
        figsize (tuple): Figure size as (width, height)
        y_limits (tuple): Y-axis limits as (min, max)
        x_limits (tuple): X-axis limits as (min, max)
        out_metric (str): Metric name to plot.
        smoothing_window (int): Window size for moving average smoothing. Set to 0 for no smoothing.
    """
    # Create a copy of the data to avoid modifying the original
    smoothed_data = data.copy()

    if smoothing_window > 0:
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(smoothed_data)

        # Apply smoothing to each mode separately
        for mode in df["Mode"].unique():
            mask = df["Mode"] == mode
            df.loc[mask, out_metric] = df.loc[mask, out_metric].rolling(window=smoothing_window, min_periods=1).mean()

        # Convert back to dictionary format
        smoothed_data = df.to_dict("list")

    plt.figure(figsize=figsize)
    print(f"Plotting {out_metric} for {title}")
    sns.lineplot(x="Step", y=out_metric, data=smoothed_data, hue="Mode", linewidth=4)

    plt.xlabel("Steps", fontsize=40)
    plt.ylabel(out_metric, fontsize=40)
    plt.title(title, fontsize=60)
    plt.legend(fontsize=40)
    plt.grid(True)

    # Set axis limits if provided
    if y_limits:
        plt.ylim(y_limits)
    if x_limits:
        plt.xlim(x_limits)

    # Make tick labels larger
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{title}.png"))
        plt.savefig(os.path.join(output_dir, f"{title}.svg"))
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot TensorBoard metrics for comparison")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing TensorBoard event files")
    parser.add_argument("--output_dir", type=str, default="./tb_figs", help="Directory to save plots")
    parser.add_argument("--tag", type=str, default="RMSE/local_test", help="Tag to plot")
    parser.add_argument("--out_metric", type=str, default="RMSE", help="Tag to plot")
    parser.add_argument("--smoothing_window", type=int, default=30, help="Window size for moving average smoothing")

    args = parser.parse_args()

    # Load data from TensorBoard event files
    data = load_tensorboard_data(args.log_dir, args.tag, args.out_metric)

    for task_name, task_data in data.items():
        plot_metrics(
            data=task_data,
            output_dir=args.output_dir,
            title=task_name,
            out_metric=args.out_metric,
            smoothing_window=args.smoothing_window,
        )


if __name__ == "__main__":
    main()
