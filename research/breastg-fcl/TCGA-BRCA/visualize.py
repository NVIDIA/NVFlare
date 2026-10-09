#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

# The original BreastG-FCL MIT notice is retained below for the upstream code.
# MIT License
#
# Copyright (c) 2026 IntelliSys-Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
GFedCL Training Curve Visualization Script

This script reads the accuracy data from round_accuracy.csv and
creates a visualization of the training curve with numeric labels
only at the end of each task (10, 20, 30, 40).

Usage:
    python visualize_training_curve.py
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DUMP_DIR = os.path.join(SCRIPT_DIR, "dump")


def find_latest_run_dir(base_dir):
    """Find the latest timestamped run directory inside base_dir."""
    if not os.path.isdir(base_dir):
        return None
    candidates = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            candidates.append(path)
    if not candidates:
        return None
    candidates_with_csv = [path for path in candidates if os.path.isfile(os.path.join(path, "round_accuracy.csv"))]
    if candidates_with_csv:
        candidates = candidates_with_csv
    return max(candidates, key=lambda p: os.path.basename(p))


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Visualize GFedCL training curve")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to CSV file containing round accuracy data")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the output visualization")
    parser.add_argument(
        "--figsize", type=str, default="12,6", help="Figure size in inches, comma-separated (width,height)"
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for the saved figure")
    args = parser.parse_args()
    if args.csv_path is None or args.output_path is None:
        latest_dir = find_latest_run_dir(DEFAULT_DUMP_DIR) or DEFAULT_DUMP_DIR
        if args.csv_path is None:
            args.csv_path = os.path.join(latest_dir, "round_accuracy.csv")
        if args.output_path is None:
            args.output_path = os.path.join(latest_dir, "training_curve.png")
    return args


def visualize_training_curve(csv_path, output_path, figsize=(12, 6), dpi=300):
    """
    Create and save a visualization of the training curve

    Args:
        csv_path: Path to CSV file with round accuracy data
        output_path: Path to save the visualization
        figsize: Tuple of (width, height) for the figure size
        dpi: DPI for the saved figure
    """
    # Load data from CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded data from {csv_path}")

        # Extract round labels and accuracy values
        round_labels = df["Round"].tolist()

        # Convert accuracy to float if it's stored as string
        if isinstance(df["Average Accuracy"].iloc[0], str):
            round_accuracy = [float(acc.strip("%")) for acc in df["Average Accuracy"]]
        else:
            round_accuracy = df["Average Accuracy"].tolist()
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Determine task boundaries and end points
    task_boundaries = []
    task_end_points = []
    current_task = None
    current_task_start = 0

    if len(round_labels) > 0:
        current_task = round_labels[0].split(",")[0]

        for i, label in enumerate(round_labels):
            task = label.split(",")[0]
            if task != current_task:
                # Mark the end of the previous task
                task_end_points.append(i)
                task_boundaries.append(i + 0.5)
                current_task = task
                current_task_start = i + 1

        # Add the final task end point
        task_end_points.append(len(round_labels))

    # Create figure with larger font sizes
    plt.figure(figsize=figsize)
    plt.rcParams.update({"font.size": 14})  # Increase base font size

    # Plot accuracy
    plt.plot(range(1, len(round_accuracy) + 1), round_accuracy, "o-", linewidth=2.5, markersize=8, color="#1f77b4")

    # Add grid and labels
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("# of Rounds", fontsize=16)
    plt.ylabel("Average Accuracy (%)", fontsize=16)
    plt.title("GFedCL Training Curve: Accuracy vs. Communication Round", fontsize=18)

    # Create custom x-tick positions and labels
    # Only show ticks at each position
    tick_positions = range(1, len(round_accuracy) + 1)

    # Create empty labels initially
    tick_labels = ["" for _ in range(len(round_accuracy))]

    # Add labels only at task end points (10, 20, 30, 40)
    for i, end_point in enumerate(task_end_points):
        if end_point > 0:  # Make sure it's a valid position
            tick_labels[end_point - 1] = str((i + 1) * 10)  # Label as 10, 20, 30, 40

    # Set tick positions and labels
    plt.xticks(tick_positions, tick_labels)

    # Draw vertical lines for task boundaries with better visibility
    for boundary in task_boundaries:
        plt.axvline(x=boundary, color="r", linestyle="--", alpha=0.7, linewidth=1.5)

    # Add a legend for task boundaries
    if task_boundaries:
        plt.plot([], [], "r--", alpha=0.7, linewidth=1.5, label="Task Boundary")
        plt.legend(fontsize=14)

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the figure
    plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close()

    print(f"Saved training curve visualization to {output_path}")


def main():
    args = parse_arguments()

    # Parse figsize from string to tuple
    figsize = tuple(map(float, args.figsize.split(",")))

    visualize_training_curve(csv_path=args.csv_path, output_path=args.output_path, figsize=figsize, dpi=args.dpi)


if __name__ == "__main__":
    main()
