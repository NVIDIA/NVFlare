# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import random
from typing import Tuple

from nvflare.apis.fl_component import FLComponent
from nvflare.app_common.app_constant import StatisticsConstants as StC
from nvflare.app_common.statistics.statistics_privacy_cleanser import StatisticsPrivacyCleanser


class AddNoiseToMinMax(FLComponent, StatisticsPrivacyCleanser):
    def __init__(self, min_noise_level: float, max_noise_level: float):
        """Random is used to generate random noise between (min_random and max_random).

        Args:
            min_noise_level: minimum noise -- used to protect min/max values before sending to server
            max_noise_level: maximum noise -- used to protect min/max values before sending to server

        For example, the random noise is to be within (0.1 and 0.3), i.e. 10% to 30% level.
        This component will make local min values smaller than the true local min values, and max values larger than
        the true local max values. As result, the estimate global max and min values (i.e. with noise)
        are still bound to the true global min/max values, such that:

        .. code-block:: text

            est. global min value <
                      true global min value <
                                client's min value <
                                        client's max value <
                                                true global max <
                                                        est. global max value

        """
        super().__init__()
        self.noise_level = (min_noise_level, max_noise_level)
        self.noise_generators = {
            StC.STATS_MIN: AddNoiseToMinMax._get_min_value,
            StC.STATS_MAX: AddNoiseToMinMax._get_max_value,
        }
        self.validate_inputs()

    def validate_inputs(self):
        for i in range(0, 2):
            if self.noise_level[i] < 0 or self.noise_level[i] > 1.0:
                raise ValueError(f"noise_level {self.noise_level}  is not within (0, 1)")
        if self.noise_level[0] > self.noise_level[1]:
            raise ValueError(
                f"minimum noise level {self.noise_level[0]} should be less "
                f"than maximum noise level {self.noise_level[1]}"
            )

    @staticmethod
    def _get_min_value(local_min_value: float, noise_level: Tuple):
        r = random.uniform(noise_level[0], noise_level[1])
        if local_min_value == 0:
            min_value = -(1 - r) * 1e-5
        else:
            if local_min_value > 0:
                min_value = local_min_value * (1 - r)
            else:
                min_value = local_min_value * (1 + r)

        return min_value

    @staticmethod
    def _get_max_value(local_max_value: float, noise_level: Tuple):
        r = random.uniform(noise_level[0], noise_level[1])
        if local_max_value == 0:
            max_value = (1 + r) * 1e-5
        else:
            if local_max_value > 0:
                max_value = local_max_value * (1 + r)
            else:
                max_value = local_max_value * (1 - r)

        return max_value

    def generate_noise(self, statistics: dict, statistic) -> dict:
        noise_gen = self.noise_generators[statistic]
        for ds_name in statistics[statistic]:
            for feature_name in statistics[statistic][ds_name]:
                local_value = statistics[statistic][ds_name][feature_name]
                noise_value = noise_gen(local_value, self.noise_level)
                statistics[statistic][ds_name][feature_name] = noise_value
        return statistics

    def apply(self, statistics: dict, client_name: str) -> Tuple[dict, bool]:
        statistics_modified = False
        for statistic in statistics:
            if statistic in self.noise_generators:
                self.logger.info(f"AddNoiseToMinMax on {statistic} for client {client_name}")
                statistics = self.generate_noise(statistics, statistic)
                statistics_modified = True

        return statistics, statistics_modified
