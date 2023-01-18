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
import os
import timeit
import time
from typing import Dict, Optional

import mlflow
from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking.client import MlflowClient
from mlflow.utils.time_utils import get_current_time_millis

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType
from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.tracking.track_exception import ExpTrackingException
from nvflare.app_common.tracking.tracker_types import LogWriterName, TrackConst
from nvflare.app_common.widgets.streaming import AnalyticsReceiver


class MLFlowReceiver(AnalyticsReceiver):
    def __init__(
            self,
            tracking_uri: Optional[str] = None,
            kwargs: Optional[dict] = None,
            artifact_location: Optional[str] = None,
            events=None,
            buffer_flush_time_in_sec=1,
    ):
        """
        MLFlowReceiver receives log events from client and deliver to the MLFLow.
        Args:
            tracking_uri: MLFlow tracking URI, when its not specified, the log metrics will be written to local file system
                          If the tracking URI is specified, the MLFlow Tracking Server must started before running the job.
            kwargs: key wards arguments.
               "experiment_name": (str) specify the experiment name. if not specified, default name will be used.
               "run_name" : (str) specify the run name
               "experiment_tags": (dict) tags used when create MLFlow experiment.
                                  "mlflow.note.content" is special MLFlow tag. when provided, it displayed as experiment
                                  description field in MLFLow UI. One can use Markdown syntax for the description
               "run_tags": (str) tags user when create MLFlow run. "mlflow.note.content" is special MLFlow tag.
                                  when provided, it displayed as run description field in MLFLow UI.
                                  One can use Markdown syntax for the description
            artifact_location: (str) relative location of the artifacts. Currently only text is supported at the moment.

            events: (str) The event the receiver is listen to. By default, it listen to "fed.analytix_log_stats"

            buffer_flush_time_in_sec: (float) the event data is buffered and then deliver to MLFLow in batches periodically.
                                      the buffer_flush_time_in_sec controls the delivery frequent. by default the buffer
                                      flushes every second. You can reduce the time to fraction of 1 second if you prefer
                                      less delay. But reduce the buffer_flush_time_in_sec will cause high traffic to MLFlow,
                                      which in some cases actually cause the slow delivery.
        """

        if events is None:
            events = ["fed.analytix_log_stats"]
        super().__init__(events=events)
        self.artifact_location = artifact_location
        self.fl_ctx = None
        self.kwargs = kwargs if kwargs else {}
        self.tracking_uri = tracking_uri
        self.mlflow = mlflow
        self.mlflow_clients: Dict[str, MlflowClient] = {}
        self.experiment_id = None
        self.run_ids = {}
        self.buffer = {}
        self.time_start = 0
        self.time_taken = 0
        self.buff_flush_time = buffer_flush_time_in_sec

        if self.tracking_uri:
            mlflow.set_tracking_uri(uri=self.tracking_uri)

    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        self.time_start = 0

        art_full_path = self.get_artifact_location(self.artifact_location)
        experiment_name = self.get_experiment_name(self.kwargs, "FLARE FL Experiment")
        experiment_tags = self.get_experiment_tags(self.kwargs)

        sites = fl_ctx.get_engine().get_clients()
        self._init_buffer(sites)
        self.mlflow_setup(art_full_path, experiment_name, experiment_tags, sites)

    def mlflow_setup(self, art_full_path, experiment_name, experiment_tags, sites):
        for site in sites:
            mlflow_client = self.mlflow_clients.get(site.name, None)
            if not mlflow_client:
                mlflow_client = MlflowClient()
                self.mlflow_clients[site.name] = mlflow_client
                self.experiment_id = self._create_experiment(
                    mlflow_client, experiment_name, art_full_path, experiment_tags
                )
                run_group_id = int(time.time())
                tags = self.get_run_tags(self.kwargs, run_group_id)
                run_name = self.get_run_name(self.kwargs, "FLARE FL Run", site.name, run_group_id)
                run = mlflow_client.create_run(experiment_id=self.experiment_id, run_name=run_name, tags=tags)
                self.run_ids[site.name] = run.info.run_id

    def _init_buffer(self, sites):
        for site in sites:
            self.buffer[site.name] = {
                AnalyticsDataType.METRICS: [],
                AnalyticsDataType.PARAMETERS: [],
                AnalyticsDataType.TAGS: [],
            }

    def get_experiment_name(self, kwargs: dict, default_name: str):
        experiment_name = kwargs.get(TrackConst.EXPERIMENT_NAME, default_name)
        return experiment_name

    def get_run_name(self, kwargs: dict, default_name: str, site_name: str, run_group_id: int):
        run_name = kwargs.get(TrackConst.RUN_NAME, default_name)
        return f"{run_name}-{site_name}-{run_group_id}"

    def get_experiment_tags(self, kwargs):
        tags = self._get_tags(TrackConst.EXPERIMENT_TAGS, kwargs=kwargs)
        return tags

    def get_run_tags(self, kwargs, run_group_id):
        run_tags = self._get_tags(TrackConst.RUN_TAGS, kwargs=kwargs)
        run_tags["job_id"] = self.fl_ctx.get_job_id()
        run_tags["group_id"] = str(run_group_id)
        return run_tags

    def _get_tags(self, tag_key: str, kwargs: dict):
        tags = {}
        if tag_key in kwargs:
            tags = kwargs[tag_key]
            if not isinstance(tags, dict):
                raise ValueError(f"argument error: value for key:'{tag_key}' is expecting type of dict")
        else:
            print("tag key", tag_key, "not found in kwargs", kwargs)
        return tags if tags else {}

    def get_artifact_location(self, relative_path: str):
        workspace = self.fl_ctx.get_engine().get_workspace()
        run_dir = workspace.get_run_dir(self.fl_ctx.get_job_id())
        root_log_dir = os.path.join(run_dir, relative_path)
        return root_log_dir

    def _create_experiment(
            self,
            mlflow_client: MlflowClient,
            experiment_name: str,
            artifact_location: str,
            experiment_tags: Optional[dict] = None,
    ) -> Optional[str]:
        experiment_id = None
        if experiment_name:
            experiment = mlflow_client.get_experiment_by_name(name=experiment_name)
            if not experiment:
                self.logger.info(f"Experiment with name '{experiment_name}' does not exist. Creating a new experiment.")
                try:
                    import pathlib

                    artifact_location_uri = pathlib.Path(artifact_location).as_uri()
                    experiment_id = mlflow_client.create_experiment(
                        name=experiment_name, artifact_location=artifact_location_uri, tags=experiment_tags
                    )
                except Exception as e:
                    raise ExpTrackingException(
                        f"Could not create an MLflow Experiment with name {experiment_name}. {e}"
                    )
                experiment = mlflow_client.get_experiment_by_name(name=experiment_name)
            else:
                experiment_id = experiment.experiment_id

            self.logger.info(f"Experiment={experiment}")
        return experiment_id

    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin: str):
        if self.time_start == 0:
            self.time_start = timeit.default_timer()

        dxo = from_shareable(shareable)
        data = AnalyticsData.from_dxo(dxo, receiver=LogWriterName.MLFLOW)
        if not data:
            return

        mlflow_client = self.get_mlflow_client(record_origin)
        run_id = self.get_run_id(record_origin)

        if data.data_type == AnalyticsDataType.TEXT:
            if data.value and data.path:
                mlflow_client.log_text(run_id, data.value, data.path)
        elif data.data_type == AnalyticsDataType.MODEL:
            # don't support
            pass
        elif data.data_type == AnalyticsDataType.IMAGE:
            # don't support
            pass
        else:
            self.buff_data(data, record_origin)
            self.flush_buffer(record_origin)

    def buff_data(self, data: AnalyticsData, record_origin: str) -> None:
        site_buff = self.buffer[record_origin]
        target_type = self.get_target_type(data.data_type)
        buf = site_buff[target_type]

        if data.data_type == AnalyticsDataType.PARAMETER:
            buf.append(Param(data.tag, str(data.value)))
        elif data.data_type == AnalyticsDataType.TAG:
            buf.append(RunTag(data.tag, str(data.value)))
        elif data.data_type == AnalyticsDataType.METRIC:
            buf.append(Metric(data.tag, data.value, get_current_time_millis(), data.step or 0))
        elif data.data_type == AnalyticsDataType.PARAMETERS:
            for k, v in data.value.items():
                buf.append(Param(k, str(v)))
        elif data.data_type == AnalyticsDataType.TAGS:
            for k, v in data.value.items():
                buf.append(RunTag(k, str(v)))
        elif data.data_type == AnalyticsDataType.METRICS:
            for k, v in data.value.items():
                buf.append(Metric(k, v, get_current_time_millis(), data.step or 0))

    def get_target_type(self, data_type: AnalyticsDataType):
        if data_type == AnalyticsDataType.METRIC:
            return AnalyticsDataType.METRICS
        elif data_type == AnalyticsDataType.PARAMETER:
            return AnalyticsDataType.PARAMETERS
        elif data_type == AnalyticsDataType.TAG:
            return AnalyticsDataType.TAGS
        else:
            return data_type

    def flush_buffer(self, record_origin, force_flush: bool = False):
        self.time_taken += timeit.default_timer() - self.time_start

        if self.time_taken >= self.buff_flush_time or force_flush:
            mlflow_client = self.get_mlflow_client(record_origin)
            if not mlflow_client:
                raise RuntimeError(f"mlflow client is None for site {record_origin}")

            run_id = self.get_run_id(record_origin)

            site_buff = self.buffer[record_origin]

            metrics_arr = self.pop_from_buffer(site_buff[AnalyticsDataType.METRICS])
            params_arr = self.pop_from_buffer(site_buff[AnalyticsDataType.PARAMETERS])
            tags_arr = self.pop_from_buffer(site_buff[AnalyticsDataType.TAGS])

            mlflow_client.log_batch(run_id=run_id, metrics=metrics_arr, params=params_arr, tags=tags_arr)

            self.time_start = 0
            self.time_taken = 0

    def pop_from_buffer(self, log_buffer):
        item_arr = []
        for _ in range(len(log_buffer)):
            item_arr.append(log_buffer.pop())
        return item_arr

    def finalize(self, fl_ctx: FLContext):
        for site_name in self.buffer:
            self.flush_buffer(site_name, force_flush=True)

        for site_name in self.run_ids:
            run_id = self.run_ids[site_name]
            mlflow_client = self.mlflow_clients[site_name]
            if run_id:
                mlflow_client.set_terminated(run_id)

    def get_run_id(self, site_id: str) -> str:
        return self.run_ids.get(site_id, None)

    def get_mlflow_client(self, site_id: str) -> MlflowClient:
        return self.mlflow_clients.get(site_id, None)
