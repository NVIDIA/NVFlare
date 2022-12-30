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
from typing import List, Optional

import mlflow
from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking.client import MlflowClient
from mlflow.utils.time_utils import get_current_time_millis

from nvflare.apis.analytix import AnalyticsData, AnalyticsDataType
from nvflare.apis.dxo import from_shareable, DXO
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.tracking.track_exception import ExpTrackingException
from nvflare.app_common.tracking.tracker_types import TrackConst
from nvflare.app_common.widgets.streaming import AnalyticsReceiver


class MlFlowConstants:
    EXPERIMENT_TAG = "experiment_tag"
    RUN_TAG = "run_tag"
    EXPERIMENT_NAME = "experiment_name"


class MLFlowReceiver(AnalyticsReceiver):
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        backend_store_uri: Optional[str] = None,
        log_path: Optional[str] = None,
        kwargs: Optional[dict] = None,
        events: Optional[List[str]] = None,
    ):

        super().__init__(events=events)
        self.root_log_dir = self.get_artifact_location(log_path)
        self.log_path = log_path
        self.fl_ctx = None
        self.root_log_dir = None

        self.kwargs = kwargs if kwargs else {}
        self.tracking_uri = tracking_uri
        self.backend_store_uri = backend_store_uri
        self.mlflow = mlflow
        self.client = MlflowClient()
        self.experiment_id = None
        self.runs = {}

        if self.tracking_uri:
            mlflow.set_tracking_uri(uri=self.tracking_uri)

    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        experiment_name = self.get_experiment_name(self.kwargs, "FLARE FL Experiment")
        experiment_tags = self.get_experiment_tags(self.kwargs)
        self.experiment_id = self._create_experiment(
            self.client, experiment_name, self.backend_store_uri, experiment_tags
        )

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

    def get_artifact_location(self, relative_path: Optional[str]):
        if relative_path is None:
            relative_path = "mlrun"
        workspace = self.fl_ctx.get_engine().get_workspace()
        run_dir = workspace.get_run_dir(self.fl_ctx.get_job_id())
        root_log_dir = os.path.join(run_dir, relative_path)
        return root_log_dir

    def _create_experiment(
        self, client: MlflowClient, experiment_name: str, artifact_location: str, experiment_tags: Optional[dict] = None
    ) -> Optional[str]:

        experiment_id = None
        if experiment_name:
            experiment = client.get_experiment_by_name(name=experiment_name)
            if not experiment:
                self.logger.info(
                    "Experiment with name '%s' does not exist. Creating a new experiment.", experiment_name
                )
                try:
                    experiment_id = client.create_experiment(
                        name=experiment_name, artifact_location=artifact_location, tags=experiment_tags
                    )
                except Exception as e:
                    raise ExpTrackingException(
                        f"Could not create an MLflow Experiment with name {experiment_name}. {e}"
                    )

        return experiment_id

    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin: str):
        dxo = from_shareable(shareable)
        data = AnalyticsData.from_dxo(dxo)
        if data.data_type == AnalyticsDataType.INIT_DATA:
            self.start(data, record_origin)

        curr_run = self.runs[record_origin]
        run_id = curr_run.info.run_id

        if data.data_type == AnalyticsDataType.PARAMETER:
            self.client.log_param(run_id, data.key, data.value)

        elif data.data_type == AnalyticsDataType.PARAMETERS:
            params_arr = [Param(key, str(value)) for key, value in data.value.items()]
            self.client.log_batch(run_id=run_id, metrics=[], params=params_arr, tags=[])

        elif data.data_type == AnalyticsDataType.METRIC:
            self.client.log_metric(run_id, data.key, data.value, get_current_time_millis(), data.step or 0)

        elif data.data_type == AnalyticsDataType.METRICS:
            timestamp = get_current_time_millis()
            metrics_arr = [Metric(key, value, timestamp, data.step or 0) for key, value in data.value.items()]
            self.client.log_batch(run_id=run_id, metrics=metrics_arr, params=[], tags=[])

        elif data.data_type == AnalyticsDataType.TAG:
            self.client.set_tag(run_id, data.key, data.value)

        elif data.data_type == AnalyticsDataType.TAGS:
            tags_arr = [RunTag(key, str(value)) for key, value in tags.items()]
            self.client.log_batch(run_id=run_id, metrics=[], params=[], tags=tags_arr)

        elif data.data_type == AnalyticsDataType.TEXT:
            if data.value and data.path:
                self.client.log_text(run_id, data.value, data.path)

        elif data.data_type == AnalyticsDataType.MODEL:
            pass

    def finalize(self, fl_ctx: FLContext):
        self.mlflow.end_run()

    def log(self, dxo: DXO, track_type: AnalyticsDataType):
        if dxo is not None:
            dxo_meta = dxo.meta
            dxo_data = dxo.data
            dxo_data_kind = dxo.data_kind
            print("track_type = ", track_type)
            self.mlflow.log_metric("test", "100")
            if track_type == AnalyticsDataType.METRICS:
                print("dxo_data 1 =", dxo_data)
                self.mlflow.log_metrics(dxo_data)

            elif track_type == AnalyticsDataType.PARAMETERS:
                print("dxo_data 2 =", dxo_data)
                self.mlflow.log_params(dxo_meta)
                self.mlflow.log_params("data_kind", dxo_data_kind)

            # elif track_type == AnalyticsDataType.ARTIFACTS:
            #     pass
            elif track_type == AnalyticsDataType.MODEL:
                pass

    def get_client_run(self, client_id: str):
        return self.runs.get(client_id, None)

    def start(self, analytics_data: AnalyticsData, client_id):
        if analytics_data.data_type == AnalyticsDataType.INIT_DATA:
            run = self.get_client_run(client_id)
            if run is None:
                run_name = f"${client_id}_{self.fl_ctx.get_job_id}"
                run_tags = self.get_run_tags(self.kwargs)
                run = self.mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name, tags=run_tags)
                self.runs[client_id] = run

            if analytics_data.kwargs:
                init_config = analytics_data.kwargs.get(TrackConst.INIT_CONFIG, None)
                if init_config:
                    self.mlflow.log_params(init_config, 0)
