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
from nvflare.app_common.tracking.tracker_types import TrackConst, Tracker
from nvflare.app_common.widgets.streaming import AnalyticsReceiver


class MlFlowConstants:
    EXPERIMENT_TAG = "experiment_tag"
    RUN_TAG = "run_tag"
    EXPERIMENT_NAME = "experiment_name"


class MLFlowReceiver(AnalyticsReceiver):
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        kwargs: Optional[dict] = None,
        artifact_location: Optional[str] = None,
        events=None,
    ):
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
        self.runs = {}
        self.run_ids = {}

        if self.tracking_uri:
            mlflow.set_tracking_uri(uri=self.tracking_uri)

    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        art_full_path = self.get_artifact_location(self.artifact_location)
        experiment_name = self.get_experiment_name(self.kwargs, "FLARE FL Experiment")
        experiment_tags = self.get_experiment_tags(self.kwargs)

        sites = fl_ctx.get_engine().get_clients()
        for site in sites:
            mlflow_client = self.mlflow_clients.get(site.name, None)
            if not mlflow_client:
                mlflow_client = MlflowClient()
                self.mlflow_clients[site.name] = mlflow_client
                self.experiment_id = self._create_experiment(
                    mlflow_client, f"{experiment_name}", art_full_path, experiment_tags
                )
                run = mlflow_client.create_run(
                    experiment_id=self.experiment_id, run_name=site.name, tags={"site": site.name}
                )
                self.run_ids[site.name] = run.info.run_id

    def get_experiment_name(self, kwargs: dict, default_name: str):
        experiment_name = kwargs.get(TrackConst.EXPERIMENT_NAME, default_name)
        return experiment_name

    def get_experiment_tags(self, kwargs):
        return self._get_tags(TrackConst.EXPERIMENT_TAG, kwargs=kwargs)

    def get_run_tags(self, kwargs):
        return self._get_tags(MlFlowConstants.RUN_TAG, kwargs=kwargs)

    def _get_tags(self, tag_key: str, kwargs: dict):
        tags = {}
        if tag_key in kwargs:
            tags = kwargs[tag_key]
            if not tags.isinstance(dict):
                raise ValueError(f"argument error: value for key:'{tag_key}' is expecting type of dict")
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
        dxo = from_shareable(shareable)
        data = AnalyticsData.from_dxo(dxo, receiver=Tracker.MLFLOW)
        mlflow_client = self.get_mlflow_client(record_origin)
        run_id = self.get_run_id(record_origin)
        key = data.tag

        if data.data_type == AnalyticsDataType.PARAMETER:
            mlflow_client.log_param(run_id, key, data.value)

        elif data.data_type == AnalyticsDataType.PARAMETERS:
            params_arr = [Param(k, str(v)) for k, v in data.value.items()]
            mlflow_client.log_batch(run_id=run_id, metrics=[], params=params_arr, tags=[])
        elif data.data_type == AnalyticsDataType.METRIC:
            mlflow_client.log_metric(run_id, key, data.value, get_current_time_millis(), data.step or 0)

        elif data.data_type == AnalyticsDataType.METRICS:
            timestamp = get_current_time_millis()
            metrics_arr = [Metric(k, v, timestamp, data.step or 0) for k, v in data.value.items()]
            mlflow_client.log_batch(run_id=run_id, metrics=metrics_arr, params=[], tags=["site", record_origin])

        elif data.data_type == AnalyticsDataType.TAG:
            mlflow_client.set_tag(run_id, key, data.value)

        elif data.data_type == AnalyticsDataType.TAGS:
            tags_arr = [RunTag(key, str(value)) for key, value in tags.items()]
            mlflow_client.log_batch(run_id=run_id, metrics=[], params=[], tags=tags_arr)

        elif data.data_type == AnalyticsDataType.TEXT:
            if data.value and data.path:
                mlflow_client.log_text(run_id, data.value, data.path)

        elif data.data_type == AnalyticsDataType.MODEL:
            pass

    def finalize(self, fl_ctx: FLContext):
        self.mlflow.end_run()

    def get_run_id(self, site_id: str) -> str:
        return self.run_ids.get(site_id, None)

    def get_mlflow_client(self, site_id: str) -> MlflowClient:
        return self.mlflow_clients.get(site_id, None)
