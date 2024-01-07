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

from lifelines import KaplanMeierFitter


def kaplan_meier_analysis(duration, event):
    # Create a Kaplan-Meier estimator
    kmf = KaplanMeierFitter()

    # Fit the model
    kmf.fit(durations=duration, event_observed=event)

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
    return {
        "timeline": timeline.tolist(),
        "km_estimate": km_estimate.tolist(),
        "event_count": event_count.tolist(),
        "survival_rate": survival_rate.tolist(),
    }
