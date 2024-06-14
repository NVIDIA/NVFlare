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

import json
import os
from typing import Dict, List, Optional

from nvflare.apis.controller_spec import Task
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.statistics_spec import Histogram, StatisticConfig
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics.hierarchical_numeric_stats import get_global_stats
from nvflare.app_common.workflows.statistics_controller import StatisticsController
from nvflare.fuel.utils import fobs


class HierarchicalStatisticsController(StatisticsController):
    def __init__(
        self,
        statistic_configs: Dict[str, dict],
        writer_id: str,
        wait_time_after_min_received: int = 1,
        result_wait_timeout: int = 10,
        precision=4,
        min_clients: Optional[int] = None,
        enable_pre_run_task: bool = True,
        hierarchy_config: str = None,
    ):
        """Controller for hierarchical statistics.

        Args:
            statistic_configs: defines the input statistic to be computed and each statistic's configuration, see below for details.
            writer_id: ID for StatisticsWriter. The StatisticWriter will save the result to output specified by the
               StatisticsWriter
            wait_time_after_min_received: numbers of seconds to wait after minimum numer of clients specified has received.
            result_wait_timeout: numbers of seconds to wait until we received all results.
               Notice this is after the min_clients have arrived, and we wait for result process
               callback, this becomes important if the data size to be processed is large
            precision:  number of precision digits
            min_clients: if specified, min number of clients we have to wait before process.
            hierarchy_config: Hierarchy specification file providing details about all the clients and their hierarchy.

        This class is derived from 'StatisticsController' and overrides only methods required to output calculated global
        statistics in given hierarchical order.

        For statistic_configs, the key is one of statistics' names sum, count, mean, stddev, histogram, and
        the value is the arguments needed. All other statistics except histogram require no argument.

        .. code-block:: text

            "statistic_configs": {
                "count": {},
                "mean": {},
                "sum": {},
                "stddev": {},
                "histogram": {
                    "*": {"bins": 20},
                    "Age": {"bins": 10, "range": [0, 120]}
                }
            },

        Histogram requires the following arguments:
            1) numbers of bins or buckets of the histogram
            2) the histogram range values [min, max]

        These arguments are different for each feature. Here are few examples:

        .. code-block:: text

            "histogram": {
                            "*": {"bins": 20 },
                            "Age": {"bins": 10, "range":[0,120]}
                         }

        The configuration specifies that the
        feature 'Age' will have 10 bins for and the range is within [0, 120).
        For all other features, the default ("*") configuration is used, with bins = 20.
        The range of histogram is not specified, thus requires the Statistics controller
        to dynamically estimate histogram range for each feature. Then this estimated global
        range (est global min, est. global max) will be used as the histogram range.

        To dynamically estimate such a histogram range, we need the client to provide the local
        min and max values in order to calculate the global bin and max value. In order to protect
        data privacy and avoid data leakage, a noise level is added to the local min/max
        value before sending to the controller. Therefore the controller only gets the 'estimated'
        values, and the global min/max are estimated, or more accurately, they are noised global min/max
        values.

        Here is another example:

        .. code-block:: text

            "histogram": {
                            "density": {"bins": 10, "range":[0,120]}
                         }

        In this example, there is no default histogram configuration for other features.

        This will work correctly if there is only one feature called "density"
        but will fail if there are other features in the dataset.

        In the following configuration:

        .. code-block:: text

            "statistic_configs": {
                "count": {},
                "mean": {},
                "stddev": {}
            }

        Only count, mean and stddev statistics are specified, so the statistics_controller
        will only set tasks to calculate these three statistics.

        For 'hierarchy_config', below is an example hierarchy specification with 4 level hierarchy for 9 NVFLARE clients with the names ranging
        from 'Device-1' to 'Device-9' and with hierarchical levels named 'Manufacturers', 'Orgs', 'Locations', 'Devices' with
        'Manufacturers' being the top most hierarchical level and "Devices" being the lowest hierarchical level:

        .. code-block:: text
           {
                "Manufacturers": [
                    {
                    "Name": "Manufacturer-1",
                    "Orgs": [
                        {
                        "Name": "Org-1",
                        "Locations": [
                            {
                            "Name": "Location-1",
                            "Devices": ["Device-1", "Device-2"]
                            },
                            {
                            "Name": "Location-2",
                            "Devices": ["Device-3"]
                            }
                        ]
                        },
                        {
                        "Name": "Org-2",
                        "Locations": [
                            {
                            "Name": "Location-1",
                            "Devices": ["Device-4", "Device-5"]
                            },
                            {
                            "Name": "Location-2",
                            "Devices": ["Device-6"]
                            }
                        ]
                        }
                    ]
                    },
                    {
                    "Name": "Manufacturer-2",
                    "Orgs": [
                        {
                        "Name": "Org-3",
                        "Locations": [
                            {
                            "Name": "Location-1",
                            "Devices": ["Device-7", "Device-8"]
                            },
                            {
                            "Name": "Location-6",
                            "Devices": ["Device-9"]
                            }
                        ]
                        }
                    ]
                    }
                ]
            }

        """
        super().__init__(
            statistic_configs,
            writer_id,
            wait_time_after_min_received,
            result_wait_timeout,
            precision,
            min_clients,
            enable_pre_run_task,
        )
        self.hierarchy_config = hierarchy_config

    def statistics_task_flow(self, abort_signal: Signal, fl_ctx: FLContext, statistic_task: str):
        """Statistics task flow for the given task.

        Args:
            abort_signal: Abort signal.
            fl_ctx: The FLContext.
            statistic_task: Statistics task.
        """
        if self.hierarchy_config:
            engine = fl_ctx.get_engine()
            ws = engine.get_workspace()
            app_conf_dir = ws.get_app_config_dir(fl_ctx.get_job_id())
            hierarchy_config_file_path = os.path.join(app_conf_dir, self.hierarchy_config)
            try:
                with open(hierarchy_config_file_path) as hierarchy_config_file:
                    hierarchy_config_json = json.load(hierarchy_config_file)
            except FileNotFoundError:
                self.system_panic(f"The hierarchy config file {hierarchy_config_file_path} does not exist.", fl_ctx)
                return False
            except IOError as e:
                self.system_panic(
                    f"An I/O error occurred while loading hierarchy config file {hierarchy_config_file_path}: {e}",
                    fl_ctx,
                )
                return False
            except json.decoder.JSONDecodeError as e:
                self.system_panic(
                    f"Failed to decode hierarchy config JSON from the file {hierarchy_config_file_path}: {e}", fl_ctx
                )
                return False
            except Exception as e:
                self.system_panic(
                    f"An unexpected error occurred while loading hierarchy config file {hierarchy_config_file_path}: {e}",
                    fl_ctx,
                )
                return False
        else:
            self.system_panic("Error: No hierarchy config file provided.", fl_ctx)
            return False

        self.log_info(fl_ctx, f"start prepare inputs for task {statistic_task}")
        inputs = self._prepare_inputs(statistic_task)
        results_cb_fn = self._get_result_cb(statistic_task)

        self.log_info(fl_ctx, f"task: {self.task_name} statistics_flow for {statistic_task} started.")

        if abort_signal.triggered:
            return False

        task_props = {StC.STATISTICS_TASK_KEY: statistic_task}
        task = Task(name=self.task_name, data=inputs, result_received_cb=results_cb_fn, props=task_props)

        self.broadcast_and_wait(
            task=task,
            targets=None,
            min_responses=self.min_clients,
            fl_ctx=fl_ctx,
            wait_time_after_min_received=self.wait_time_after_min_received,
            abort_signal=abort_signal,
        )

        self.global_statistics = get_global_stats(
            self.global_statistics, self.client_statistics, statistic_task, hierarchy_config_json
        )

        self.log_info(fl_ctx, f"task {self.task_name} statistics_flow for {statistic_task} flow end.")

    def _recursively_round_global_stats(self, global_stats):
        """Apply given precision to the calculated global statistics.

        Args:
            global_stats: Global stats.

        Returns:
            A dict containing global stats with applied precision.
        """
        if isinstance(global_stats, dict):
            for key, value in global_stats.items():
                if key == StC.GLOBAL or key == StC.LOCAL:
                    for key, metric in value.items():
                        if key == StC.STATS_HISTOGRAM:
                            for ds in metric:
                                for name, val in metric[ds].items():
                                    hist: Histogram = metric[ds][name]
                                    buckets = StatisticsController._apply_histogram_precision(hist.bins, self.precision)
                                    metric[ds][name] = buckets
                        else:
                            for ds in metric:
                                for name, val in metric[ds].items():
                                    metric[ds][name] = round(metric[ds][name], self.precision)
                    continue
                if isinstance(value, list):
                    for item in value:
                        self._recursively_round_global_stats(item)
        elif isinstance(global_stats, list):
            for item in global_stats:
                self._recursively_round_global_stats(item)

        return global_stats

    def _combine_all_statistics(self):
        """Get combined global statistics with precision applied.

        Returns:
            A dict containing global statistics with precision applied.
        """
        result = self.global_statistics
        return self._recursively_round_global_stats(result)

    def _prepare_inputs(self, statistic_task: str) -> Shareable:
        """Prepare inputs for the given task.

        Args:
            statistic_task: Statistics task.

        Returns:
            A dict containing inputs.
        """
        inputs = Shareable()
        target_statistics: List[StatisticConfig] = StatisticsController._get_target_statistics(
            self.statistic_configs, StC.ordered_statistics[statistic_task]
        )
        for tm in target_statistics:
            if tm.name == StC.STATS_HISTOGRAM:
                if StC.STATS_MIN in self.global_statistics[StC.GLOBAL]:
                    inputs[StC.STATS_MIN] = self.global_statistics[StC.GLOBAL][StC.STATS_MIN]
                if StC.STATS_MAX in self.global_statistics[StC.GLOBAL]:
                    inputs[StC.STATS_MAX] = self.global_statistics[StC.GLOBAL][StC.STATS_MAX]
            elif tm.name == StC.STATS_VAR:
                if StC.STATS_COUNT in self.global_statistics[StC.GLOBAL]:
                    inputs[StC.STATS_GLOBAL_COUNT] = self.global_statistics[StC.GLOBAL][StC.STATS_COUNT]
                if StC.STATS_MEAN in self.global_statistics[StC.GLOBAL]:
                    inputs[StC.STATS_GLOBAL_MEAN] = self.global_statistics[StC.GLOBAL][StC.STATS_MEAN]

        inputs[StC.STATISTICS_TASK_KEY] = statistic_task

        inputs[StC.STATS_TARGET_STATISTICS] = fobs.dumps(target_statistics)

        return inputs
