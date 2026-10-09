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

import logging as logger
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_results(accuracy_data, output_dir):
    """
    Create and save visualizations of accuracy results

    Args:
        accuracy_data: Dictionary containing accuracy results
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Extract data
    client_task_acc = accuracy_data["client_task_acc"]
    task_avg_acc = accuracy_data["task_avg_acc"]
    client_avg_acc = accuracy_data["client_avg_acc"]
    overall_avg_acc = accuracy_data["overall_avg_acc"]

    # 1. Task Average Accuracy Bar Chart
    plt.figure(figsize=(10, 6))
    tasks = list(task_avg_acc.keys())
    accuracies = list(task_avg_acc.values())

    plt.bar(tasks, accuracies, color="skyblue")
    plt.axhline(y=overall_avg_acc, color="r", linestyle="--", label=f"Overall Avg: {overall_avg_acc:.2f}%")
    plt.xlabel("Task ID")
    plt.ylabel("Average Accuracy (%)")
    plt.title("Average Accuracy by Task")
    plt.xticks(tasks)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the figure
    plt.savefig(os.path.join(plots_dir, "task_accuracy.png"), bbox_inches="tight")
    plt.close()

    # 2. Client Average Accuracy Bar Chart
    plt.figure(figsize=(12, 6))
    clients = list(client_avg_acc.keys())
    accuracies = list(client_avg_acc.values())

    plt.bar(clients, accuracies, color="lightgreen")
    plt.axhline(y=overall_avg_acc, color="r", linestyle="--", label=f"Overall Avg: {overall_avg_acc:.2f}%")
    plt.xlabel("Client ID")
    plt.ylabel("Average Accuracy (%)")
    plt.title("Average Accuracy by Client")
    plt.xticks(clients)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the figure
    plt.savefig(os.path.join(plots_dir, "client_accuracy.png"), bbox_inches="tight")
    plt.close()

    # 3. Heatmap of Client-Task Accuracy
    # Convert to DataFrame for easier plotting
    df_data = []
    for client_id, task_accs in client_task_acc.items():
        for task_id, acc in task_accs.items():
            df_data.append({"Client": client_id, "Task": task_id, "Accuracy": acc})

    if df_data:  # Only create heatmap if we have data
        df = pd.DataFrame(df_data)
        pivot_df = df.pivot(index="Client", columns="Task", values="Accuracy")

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, cmap="viridis", fmt=".2f", cbar_kws={"label": "Accuracy (%)"})
        plt.title("Client-Task Accuracy Heatmap")
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(plots_dir, "client_task_heatmap.png"), bbox_inches="tight")
        plt.close()

    logger.info(f"Saved accuracy visualization plots to {plots_dir}")
    return plots_dir


def plot_training_curve(opt, round_accuracy, round_labels):
    """
    Create and save a plot of the training curve showing accuracy per round

    Args:
        round_accuracy: List of average accuracy values per round
        round_labels: List of round labels
    """
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.plot(range(1, len(round_accuracy) + 1), round_accuracy, "o-", linewidth=2, markersize=8)

    # Add grid and labels
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("Communication Round", fontsize=12)
    plt.ylabel("Average Accuracy (%)", fontsize=12)
    plt.title("BreastG-FCL Training Curve: Accuracy vs. Communication Round", fontsize=14)

    # Add ticks and labels
    plt.xticks(range(1, len(round_accuracy) + 1), round_labels, rotation=45)

    # Annotate points with accuracy values
    for i, acc in enumerate(round_accuracy):
        plt.annotate(f"{acc:.2f}%", (i + 1, acc), textcoords="offset points", xytext=(0, 10), ha="center")

    # Add task boundaries if we have multiple tasks
    task_boundaries = []
    current_task = round_labels[0].split(",")[0]

    for i, label in enumerate(round_labels):
        task = label.split(",")[0]
        if task != current_task:
            task_boundaries.append(i + 0.5)
            current_task = task

    # Draw vertical lines for task boundaries
    for boundary in task_boundaries:
        plt.axvline(x=boundary, color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(opt.output_dir, "training_curve.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    logger.info(f"Saved training curve plot to {save_path}")


def plot_all_tasks_accuracy(opt, all_tasks_accuracy, plots_dir=None):
    """
    Create a visualization showing the performance on all tasks over training rounds

    Args:
        opt: Configuration options
        all_tasks_accuracy: List of dictionaries containing accuracy data for all tasks
        plots_dir: Directory to save plots (uses opt.output_dir/plots if None)
    """
    import os

    import matplotlib.pyplot as plt

    # Setup plot directory
    if plots_dir is None:
        plots_dir = os.path.join(opt.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Prepare data for plotting
    num_tasks = opt.num_task
    rounds = []
    # Initialize task data with empty lists
    task_data = {t: [] for t in range(num_tasks)}

    # Extract accuracy values for each task over time
    for round_data in all_tasks_accuracy:
        rounds.append(round_data["round"])

        # Store accuracy for each task (or None if not available)
        for t in range(num_tasks):
            if t in round_data["tasks"]:
                task_data[t].append(round_data["tasks"][t])
            else:
                task_data[t].append(None)  # No data for this task yet

    # Create plot
    plt.figure(figsize=(12, 8))

    # Different line styles and markers for each task
    line_styles = ["-", "--", "-.", ":", "-", "--", "-."]
    markers = ["o", "s", "^", "D", "x", "+", "*"]
    colors = plt.cm.tab10.colors  # Use a good color cycle

    # Plot each task's accuracy evolution
    for t in range(num_tasks):
        # Filter out None values for plotting
        valid_indices = [i for i, val in enumerate(task_data[t]) if val is not None]
        valid_rounds = [rounds[i] for i in valid_indices]
        valid_acc = [task_data[t][i] for i in valid_indices]

        if valid_acc:  # Only plot if we have data
            plt.plot(
                valid_indices,
                valid_acc,
                label=f"Task {t+1}",
                linestyle=line_styles[t % len(line_styles)],
                marker=markers[t % len(markers)],
                color=colors[t % len(colors)],
                linewidth=2,
                markersize=8,
            )

    # Add task boundaries - identify where new tasks start
    current_task = 0
    task_boundaries = []

    for i, round_label in enumerate(rounds):
        task = int(round_label.split(",")[0].split(" ")[1]) - 1  # Extract task number and convert to 0-indexed
        if task > current_task:
            # This is the start of a new task
            task_boundaries.append(i - 0.5)  # Adjust position to be between rounds
            current_task = task

    # Draw vertical lines at task boundaries
    for boundary in task_boundaries:
        plt.axvline(x=boundary, color="gray", linestyle="--", alpha=0.7)

    # Customize the plot
    plt.xlabel("Training Round", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.title("Accuracy on All Tasks Throughout Training", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12, loc="lower right")

    # Make x-axis tick labels more readable
    if len(rounds) > 20:
        # If we have many rounds, only show some of the labels
        skip = len(rounds) // 20 + 1
        plt.xticks(
            range(0, len(rounds), skip),
            [rounds[i] for i in range(0, len(rounds), skip)],
            rotation=45,
            ha="right",
            fontsize=10,
        )
    else:
        plt.xticks(range(len(rounds)), rounds, rotation=45, ha="right", fontsize=10)

    plt.tight_layout()

    # Save figure
    file_path = os.path.join(plots_dir, "all_tasks_accuracy.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved all-tasks accuracy visualization to {file_path}")

    # Create a catastrophic forgetting visualization
    if num_tasks > 1 and all_tasks_accuracy:
        plot_forgetting_metrics(opt, all_tasks_accuracy, plots_dir)

    return file_path


def plot_forgetting_metrics(opt, all_tasks_accuracy, plots_dir):
    """
    Create a visualization focusing on catastrophic forgetting metrics

    Args:
        opt: Configuration options
        all_tasks_accuracy: List of dictionaries containing accuracy data for all tasks
        plots_dir: Directory to save plots
    """
    import os

    import matplotlib.pyplot as plt

    # Create figure
    plt.figure(figsize=(12, 8))

    # Organize data by task transitions
    num_tasks = opt.num_task

    # Calculate forgetting for each previous task after learning a new task
    # Forgetting: decrease in accuracy on task j after training on task k (where k > j)
    task_transitions = []

    # Go through each task
    for current_task in range(1, num_tasks):  # Start from second task
        # Find all rounds for current task
        current_task_rounds = []
        for i, round_data in enumerate(all_tasks_accuracy):
            if round_data["current_task"] == current_task:
                current_task_rounds.append(i)

        if not current_task_rounds:
            continue

        # For each previous task, calculate forgetting
        for prev_task in range(current_task):
            # Get best accuracy on previous task before starting current task
            best_prev_acc = 0

            # Find the best accuracy on prev_task during previous tasks
            for i, round_data in enumerate(all_tasks_accuracy):
                if i < current_task_rounds[0]:  # Before starting current task
                    if prev_task in round_data["tasks"]:
                        best_prev_acc = max(best_prev_acc, round_data["tasks"][prev_task])

            # Track forgetting throughout current task
            forgetting_data = []
            round_indices = []
            round_labels = []

            for i in current_task_rounds:
                round_data = all_tasks_accuracy[i]
                if prev_task in round_data["tasks"]:
                    current_acc = round_data["tasks"][prev_task]
                    forgetting = max(0, best_prev_acc - current_acc)  # Positive when there is forgetting
                    forgetting_data.append(forgetting)
                    round_indices.append(i)
                    round_labels.append(round_data["round"])

            # Only plot if we have forgetting data
            if forgetting_data:
                plt.plot(
                    round_indices,
                    forgetting_data,
                    label=f"Task {prev_task+1} → Task {current_task+1}",
                    marker="o",
                    linewidth=2,
                )

    # Customize the plot
    plt.xlabel("Training Round", fontsize=14)
    plt.ylabel("Forgetting (% decrease in accuracy)", fontsize=14)
    plt.title("Catastrophic Forgetting Throughout Training", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Only show legend if there's more than one line
    if len(plt.gca().get_lines()) > 1:
        plt.legend(fontsize=12, loc="upper left")

    # Make x-axis labels more readable
    round_labels = [d["round"] for d in all_tasks_accuracy]
    if len(round_labels) > 20:
        # If we have many rounds, only show some of the labels
        skip = len(round_labels) // 20 + 1
        plt.xticks(
            range(0, len(round_labels), skip),
            [round_labels[i] for i in range(0, len(round_labels), skip)],
            rotation=45,
            ha="right",
            fontsize=10,
        )
    else:
        plt.xticks(range(len(round_labels)), round_labels, rotation=45, ha="right", fontsize=10)

    plt.tight_layout()

    # Save figure
    file_path = os.path.join(plots_dir, "catastrophic_forgetting.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved catastrophic forgetting visualization to {file_path}")

    return file_path
