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

import os

import matplotlib.pyplot as plt


def plot_acc(path):

    log_path = os.path.join(path, "simulate_job", "app_site-1", "log.txt")
    acc = []

    with open(log_path, encoding="utf-8") as f:
        for line in f.readlines():
            str_split = line.split(" ")
            if len(str_split) > 5:
                if str_split[-2] == "train_accuracy:":
                    acc.append(float(str_split[-1]))

    print(acc)
    ep = [i * 10 for i in range(len(acc))]
    plt.plot(ep, acc)
    plt.xlabel("Local training epoch")
    plt.ylabel("Training accuracy")
    plt.title("One-shot VFL")
    plt.savefig("figs/oneshotVFL_results1.png")
