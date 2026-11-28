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

import json
import os
import pathlib
import time
import timeit
from typing import Dict, List, Optional
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking.client import MlflowClient
from sqlalchemy.engine import URL

from nvflare.apis.analytix import ANALYTIC_EVENT_TYPE, AnalyticsData, AnalyticsDataType, LogWriterName, TrackConst
from nvflare.apis.dxo import from_shareable
from nvflare.apis.fl_constant import ProcessType, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey, get_job_meta_study
from nvflare.apis.shareable import Shareable
from nvflare.app_common.utils.file_utils import resolve_path_under_root
from nvflare.app_common.widgets.streaming import AnalyticsReceiver

DEFAULT_RUN_NAME = "FLARE FL Run"


class MlflowConstants:
    EXPERIMENT_TAG = "experiment_tag"
    RUN_TAG = "run_tag"
    EXPERIMENT_NAME = "experiment_name"


def get_current_time_millis():
    return int(round(time.time() * 1000))


def _get_job_name_from_fl_ctx(fl_ctx: FLContext, default=None):
    # TODO: it might be good to have a function in fl_context to get the job name
    job_meta = fl_ctx.get_prop(ReservedKey.JOB_META)
    if job_meta and isinstance(job_meta, dict):
        return job_meta.get(JobMetaKey.JOB_NAME, default)
    return default


class MLflowReceiver(AnalyticsReceiver):
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        kw_args: Optional[dict] = None,
        artifact_location: Optional[str] = None,
        events: Optional[List[str]] = None,
        buffer_flush_time=1,
    ):
        """MLflowReceiver receives log events from clients and deliver them to the MLflow tracking server.

        Args:
            tracking_uri (Optional[str], optional): MLflow tracking server URI. When this is not specified, metrics
                go to the study's site-configured endpoint or to a local SQLite database in the job result directory.
                An explicit URI must match the configured study endpoint, or identify a job-local store when a study
                endpoint is absent. Remote tracking servers must be started before running the job. Defaults to None.
            kw_args (Optional[dict], optional): keyword arguments:
                "experiment_name" (str): Specifies the experiment name. If not specified, the default name of "FLARE FL Experiment" will be used.
                "run_name" (str): Specifies the run name
                "experiment_tags" (dict): Tags used when creating the MLflow experiment.
                "mlflow.note.content" is a special MLflow tag. When provided, it displays as experiment
                description field on the MLflow UI. You can use Markdown syntax for the description.
                "run_tags" (str): Tags used when creating the MLflow run. "mlflow.note.content" is a special MLflow tag.
                When provided, it displays as run description field on the MLflow UI.
                You can use Markdown syntax for the description.
            artifact_location (Optional[str], optional): Relative location of artifacts. Currently only text is supported at the moment.
            events (optional, List[str]): A list of event that this receiver will handle.
            buffer_flush_time (int, optional): The time in seconds between deliveries of event data to the MLflow tracking server. The
                data is buffered and then delivered to the MLflow tracking server in batches, and
                the buffer_flush_time controls the frequency of the sending. By default, the buffer
                flushes every second. You can reduce the time to a fraction of a second if you prefer
                less delay. Keep in mind that reducing the buffer_flush_time will potentially cause high
                traffic to the MLflow tracking server, which in some cases can actually cause more latency.
        """
        if not isinstance(tracking_uri, (str, type(None))):
            raise ValueError("tracking_uri needs to be either None or str")
        if events is None:
            events = ["fed." + ANALYTIC_EVENT_TYPE]
        super().__init__(events=events)
        self.artifact_location = artifact_location if artifact_location is not None else "artifacts"

        self.kw_args = kw_args if kw_args else {}
        self.tracking_uri = tracking_uri
        self._site_approved_tracking_uri = False
        self.mlflow_clients: Dict[str, MlflowClient] = {}
        self.experiment_id = None
        self.run_ids = {}
        self.buffer = {}
        self.time_start = 0
        self.time_since_flush = 0
        self.buff_flush_time = buffer_flush_time

    def _get_tracking_uri(self, fl_ctx: FLContext):
        self._site_approved_tracking_uri = False
        workspace = fl_ctx.get_workspace()
        job_id = fl_ctx.get_job_id()
        result_root = os.path.realpath(workspace.get_result_root(job_id))
        site_uri = self._get_study_tracking_uri(fl_ctx, workspace)
        if site_uri is not None:
            if self.tracking_uri is not None and self.tracking_uri != site_uri:
                raise ValueError("tracking_uri conflicts with the site's MLflow endpoint for this study")
            self._site_approved_tracking_uri = True
            return site_uri

        if not self.tracking_uri:
            mlflow_db = resolve_path_under_root(result_root, "mlflow.db", "MLflow database")
            return URL.create("sqlite", database=mlflow_db).render_as_string()

        uri = urlsplit(self.tracking_uri)
        if uri.query or uri.fragment or uri.netloc:
            raise ValueError("tracking_uri must be a job-local store or configured for the study")
        if self.tracking_uri.startswith("sqlite:///"):
            path = unquote(self.tracking_uri[len("sqlite:///") :], errors="strict")
            if path in ("", ":memory:"):
                raise ValueError("tracking_uri must specify a job-local database path")
        elif uri.scheme == "file":
            path = url2pathname(uri.path)
        else:
            raise ValueError("tracking_uri must be a job-local store or configured for the study")

        # SQLAlchemy decodes SQLite URL paths. Check that destination and encode
        # the canonical path once so the driver opens the exact path we checked.
        relative_path = os.path.relpath(os.path.abspath(path), result_root)
        path = resolve_path_under_root(result_root, relative_path, "tracking_uri")
        if uri.scheme == "sqlite":
            return URL.create("sqlite", database=path).render_as_string()
        return pathlib.Path(path).as_uri()

    @staticmethod
    def _get_study_tracking_uri(fl_ctx: FLContext, workspace):
        resources_file = workspace.get_resources_file_path()
        resources = {}
        if resources_file and os.path.isfile(resources_file):
            with open(resources_file) as f:
                resources = json.load(f)
        if not isinstance(resources, dict):
            raise ValueError("site resources must be a JSON object")
        endpoints = resources.get("mlflow_study_tracking_uris", {})
        if not isinstance(endpoints, dict) or any(
            not study or not isinstance(uri, str) or not uri.strip() or uri == "*" for study, uri in endpoints.items()
        ):
            raise ValueError("site mlflow_study_tracking_uris must map study names to nonempty tracking URIs")
        # The server assigns the study during submission. Privacy scope is
        # separate job-supplied metadata and cannot select a tracking backend.
        study = get_job_meta_study(fl_ctx.get_prop(ReservedKey.JOB_META))
        return endpoints.get(study)

    def initialize(self, fl_ctx: FLContext):
        """Initializes MlflowClient for each site.

        This method:
        1. Sets up the FL context and timing
        2. Validates and prepares experiment configuration
        3. Determines participating sites
        4. Sets up MLflow clients and experiments for each site
        5. Initializes data buffers

        Args:
            fl_ctx (FLContext): The FLContext containing runtime information

        Raises:
            ValueError: If the backend or artifact path is not permitted, or experiment name is empty
            RuntimeError: If unable to determine participating sites
        """
        # Initialize context and timing
        self.time_start = 0

        # Validate and prepare experiment configuration
        tracking_uri = self._get_tracking_uri(fl_ctx)
        art_full_path = self._get_artifact_location(self.artifact_location, fl_ctx)
        experiment_name = self.kw_args.get(TrackConst.EXPERIMENT_NAME, "FLARE FL Experiment")
        if not experiment_name:
            self.log_error(fl_ctx, "Experiment name cannot be empty")
            raise ValueError("Experiment name cannot be empty")

        experiment_tags = self._get_tags(TrackConst.EXPERIMENT_TAGS, kwargs=self.kw_args)

        # Determine participating sites
        if fl_ctx.get_process_type() == ProcessType.SERVER_JOB:
            clients = fl_ctx.get_engine().get_clients()
            if not clients:
                raise RuntimeError("No clients found in server context")
            site_names = [c.name for c in clients]
        else:
            # Client context - track only this client
            site_name = fl_ctx.get_identity_name()
            if not site_name:
                raise RuntimeError("Unable to determine client identity")
            site_names = [site_name]

        self.log_info(fl_ctx, f"Initializing MLflow tracking for sites: {site_names}")

        # Set up MLflow for each site
        self._mlflow_setup(art_full_path, experiment_name, experiment_tags, site_names, fl_ctx, tracking_uri)

        # Initialize data buffers
        self._init_buffer(site_names)

        self.log_info(fl_ctx, "MLflow tracking initialization completed successfully")

    def _mlflow_setup(
        self,
        art_full_path,
        experiment_name,
        experiment_tags,
        site_names: List[str],
        fl_ctx: FLContext,
        tracking_uri: str,
    ):
        """Set up an MlflowClient for each receiving site and create an experiment and run.

        Args:
            art_full_path (str): Full path to artifacts.
            experiment_name (str): Experiment name.
            experiment_tags (dict): Experiment tags.
            sites (List[str]): List of sites.
            fl_ctx (FLContext): An FLContext.
            tracking_uri (str): Validated tracking backend for this receiver.
        """
        for site_name in site_names:
            mlflow_client = self.mlflow_clients.get(site_name, None)
            if not mlflow_client:
                mlflow_client = MlflowClient(tracking_uri=tracking_uri)
                self.mlflow_clients[site_name] = mlflow_client
                self.experiment_id = self._get_or_create_experiment(
                    mlflow_client, experiment_name, art_full_path, experiment_tags, fl_ctx
                )

                job_id_tag = self._get_job_id_tag(fl_ctx)
                job_name = _get_job_name_from_fl_ctx(fl_ctx)

                run_name = self._get_run_name(self.kw_args, site_name, job_id_tag, job_name)
                tags = self._get_run_tags(self.kw_args, job_id_tag, run_name)
                run = mlflow_client.create_run(experiment_id=self.experiment_id, run_name=run_name, tags=tags)
                self._validate_artifact_uri(run.info.artifact_uri, fl_ctx)
                self.run_ids[site_name] = run.info.run_id

    def _init_buffer(self, site_names: List[str]):
        """For each site, create a buffer (dict) consisting of a list each for metrics, parameters, and tags."""
        for site_name in site_names:
            self.buffer[site_name] = {
                AnalyticsDataType.METRICS: [],
                AnalyticsDataType.PARAMETERS: [],
                AnalyticsDataType.TAGS: [],
            }

    def _get_run_name(self, kwargs: dict, site_name: str, job_id_tag: str, job_name: str):
        run_name = kwargs.get(TrackConst.RUN_NAME, DEFAULT_RUN_NAME)
        job_name_str = job_name if job_name is not None else "unknown_job"
        return f"{site_name}-{job_id_tag[:6]}-{job_name_str}-{run_name}"

    def _get_run_tags(self, kwargs, job_id_tag: str, run_name: str):
        run_tags = self._get_tags(TrackConst.RUN_TAGS, kwargs=kwargs)
        run_tags["job_id"] = job_id_tag
        run_tags["run_name"] = run_name
        return run_tags

    def _get_job_id_tag(self, fl_ctx: FLContext) -> str:
        """Gets a unique job id tag."""
        job_id = fl_ctx.get_job_id()
        if job_id == "simulate_job":
            # Since all jobs run in the simulator have the same job_id of "simulate_job"
            # Use timestamp as unique identifier for simulation runs
            job_id = str(int(time.time()))
        return job_id

    def _get_tags(self, tag_key: str, kwargs: dict):
        tags = {}
        if tag_key in kwargs:
            tags = kwargs[tag_key]
            if not isinstance(tags, dict):
                raise ValueError(f"argument error: value for key:'{tag_key}' is expecting type of dict")
        else:
            print("tag key: ", tag_key, " not found in kwargs: ", kwargs)
        return tags if tags else {}

    def _get_artifact_location(self, relative_path: str, fl_ctx: FLContext):
        workspace = fl_ctx.get_engine().get_workspace()
        run_dir = workspace.get_run_dir(fl_ctx.get_job_id())
        return resolve_path_under_root(run_dir, relative_path, "artifact_location")

    def _validate_artifact_uri(self, artifact_uri: str, fl_ctx: FLContext):
        if self._site_approved_tracking_uri:
            # The operator-approved backend owns its artifact routing too.
            return None
        uri = urlsplit(artifact_uri)
        if uri.scheme != "file" or uri.netloc or uri.query or uri.fragment:
            raise ValueError("unapproved MLflow backend must use job-local artifact storage")
        run_dir = os.path.realpath(fl_ctx.get_workspace().get_run_dir(fl_ctx.get_job_id()))
        path = os.path.realpath(url2pathname(uri.path))
        return resolve_path_under_root(run_dir, os.path.relpath(path, run_dir), "MLflow artifact storage")

    def _get_or_create_experiment(
        self,
        mlflow_client: MlflowClient,
        experiment_name: str,
        artifact_location: str,
        experiment_tags: Optional[dict],
        fl_ctx: FLContext,
    ) -> Optional[str]:
        experiment_id = None
        experiment = mlflow_client.get_experiment_by_name(name=experiment_name)
        if not experiment:
            self.logger.info(f"Experiment with name '{experiment_name}' does not exist. Creating a new experiment.")
            artifact_location_uri = pathlib.Path(artifact_location).as_uri()
            experiment_id = mlflow_client.create_experiment(
                name=experiment_name, artifact_location=artifact_location_uri, tags=experiment_tags
            )
            experiment = mlflow_client.get_experiment_by_name(name=experiment_name)
        else:
            experiment_id = experiment.experiment_id

        # A job-local database can contain job-supplied experiment metadata.
        self._validate_artifact_uri(experiment.artifact_location, fl_ctx)
        self.logger.info(f"Experiment={experiment}")
        return experiment_id

    def save(self, fl_ctx: FLContext, shareable: Shareable, record_origin: str):
        if self.time_start == 0:
            self.time_start = timeit.default_timer()

        dxo = from_shareable(shareable)
        data = AnalyticsData.from_dxo(dxo, receiver=LogWriterName.MLFLOW)
        if not data:
            return

        if data.data_type == AnalyticsDataType.TEXT:
            mlflow_client = self.get_mlflow_client(record_origin)
            if not mlflow_client:
                raise RuntimeError(f"mlflow client is None for site {record_origin}.")
            run_id = self.get_run_id(record_origin)
            if not run_id:
                raise RuntimeError(f"run_id is missing for site {record_origin}.")

            if data.kwargs.get("path", None):
                artifact_file = data.kwargs["path"]
                artifact_root = self._validate_artifact_uri(mlflow_client.get_run(run_id).info.artifact_uri, fl_ctx)
                if artifact_root is not None:
                    resolve_path_under_root(artifact_root, artifact_file, "MLflow artifact file")
                mlflow_client.log_text(run_id=run_id, text=data.value, artifact_file=artifact_file)
        elif data.data_type == AnalyticsDataType.MODEL:
            # not currently supported
            pass
        elif data.data_type == AnalyticsDataType.IMAGE:
            # not currently supported
            pass
        else:
            self.buffer_data(data, record_origin)
            self.time_since_flush += timeit.default_timer() - self.time_start
            if self.time_since_flush >= self.buff_flush_time:
                self.flush_buffers(record_origin)

    def buffer_data(self, data: AnalyticsData, record_origin: str) -> None:
        """Buffer the data to send later.

        A buffer for each data_type is in each site_buffer, all of which are in self.buffer

        Args:
            data (AnalyticsData): Data.
            record_origin (str): Origin of the data, or site name.
        """
        site_buffer = self.buffer[record_origin]
        target_type = self.get_target_type(data.data_type)
        buf = site_buffer[target_type]

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

    def flush_buffers(self, record_origin):
        """Flush buffers and send all the data to the MLflow tracking server.

        Args:
            record_origin (str): Origin of the data, or site name.
        """
        mlflow_client = self.get_mlflow_client(record_origin)
        if not mlflow_client:
            raise RuntimeError(f"mlflow client is None for site {record_origin}.")

        run_id = self.get_run_id(record_origin)
        if not run_id:
            raise RuntimeError(f"run_id is missing for site {record_origin}.")

        site_buff = self.buffer[record_origin]

        metrics_arr = self.flush_buffer(site_buff[AnalyticsDataType.METRICS])
        params_arr = self.flush_buffer(site_buff[AnalyticsDataType.PARAMETERS])
        tags_arr = self.flush_buffer(site_buff[AnalyticsDataType.TAGS])

        mlflow_client.log_batch(run_id=run_id, metrics=metrics_arr, params=params_arr, tags=tags_arr)

        self.time_start = 0
        self.time_since_flush = 0

    def flush_buffer(self, log_buffer: List):
        item_arr = list(log_buffer)
        log_buffer.clear()
        return item_arr

    def finalize(self, fl_ctx: FLContext):
        for site_name in self.buffer:
            self.flush_buffers(site_name)

        for site_name in self.run_ids:
            run_id = self.run_ids[site_name]
            mlflow_client = self.mlflow_clients[site_name]
            if run_id:
                mlflow_client.set_terminated(run_id)

    def get_run_id(self, site_id: str) -> Optional[str]:
        return self.run_ids.get(site_id, None)

    def get_mlflow_client(self, site_id: str) -> MlflowClient:
        return self.mlflow_clients.get(site_id, None)
