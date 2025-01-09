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

import xgboost.callback

from nvflare.app_common.tracking.log_writer import LogWriter


class MetricsCallback(xgboost.callback.TrainingCallback):
    def __init__(self, writer: LogWriter):
        xgboost.callback.TrainingCallback.__init__(self)
        if not isinstance(writer, LogWriter):
            raise RuntimeError("MetricsCallback: writer is not valid.")
        self.writer = writer

    def after_iteration(self, model, epoch: int, evals_log: xgboost.callback.TrainingCallback.EvalsLog):
        if not evals_log:
            return False

        data_type = self.writer.get_default_metric_data_type()
        for data, metric in evals_log.items():
            record = {}
            for metric_name, log in metric.items():
                score = log[-1][0] if isinstance(log[-1], tuple) else log[-1]
                record[metric_name] = score
            self.writer.write(tag=f"{data}_metrics", value=record, data_type=data_type, global_step=epoch)
        return False
