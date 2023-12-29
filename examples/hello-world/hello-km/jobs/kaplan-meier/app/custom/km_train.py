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

import pandas as pd
from km_analysis import kaplan_meier_analysis

# (1) import nvflare client API
import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType

# Client training code


def load_data():
    data = {
        "site-1": {
            "duration": [5, 10, 15, 25, 30, 35, 40, 45, 50, 55, 60, 65],
            "event": [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 2, 4],
        },
        "site-2": {"duration": [10, 25, 30, 40, 50, 60, 70], "event": [1, 1, 0, 1, 0, 3, 4]},
    }

    return data


def display_results(results):
    for time_point, km_estimate, event_count, survival_rate in zip(
        results["timeline"], results["km_estimate"], results["event_count"], results["survival_rate"]
    ):
        print(
            f"Time: {time_point}, KM Estimate: {km_estimate:.4f}, Event Count: {event_count}, Survival Rate: {survival_rate:.4f}"
        )


def main():
    flare.init()

    site_name = flare.get_site_name()

    df = pd.DataFrame(data=load_data())[site_name]

    while flare.is_running():

        print(f"Kaplan-meier analysis for {site_name}")

        if flare.is_train():
            # Perform Kaplan-Meier analysis and get the results
            results = kaplan_meier_analysis(duration=df["duration"], event=df["event"])

            # Display the results
            display_results(results)
            print(f"send result for site = {flare.get_site_name()}")
            model = FLModel(params=results, params_type=ParamsType.FULL)
            flare.send(model)

    print(f"finish send for {site_name}, complete")


if __name__ == "__main__":
    main()
