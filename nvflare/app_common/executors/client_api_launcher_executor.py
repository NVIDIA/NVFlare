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

import logging
import os
import sys
from typing import Optional

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.launcher_executor import LauncherExecutor
from nvflare.app_common.utils.export_utils import update_export_props
from nvflare.client.config import ConfigKey, ExchangeFormat, TransferType, write_config_to_file
from nvflare.client.constants import CLIENT_API_CONFIG, EXTERNAL_PRE_INIT_TIMEOUT
from nvflare.fuel.utils.argument_utils import str2bool
from nvflare.fuel.utils.memory_utils import cleanup_memory
from nvflare.fuel.utils.attributes_exportable import ExportMode
from nvflare.fuel.utils.fobs import FOBSContextKey
from nvflare.fuel.utils.pipe.cell_pipe import CellPipe
from nvflare.utils.configs import get_client_config_value

logger = logging.getLogger(__name__)


class ClientAPILauncherExecutor(LauncherExecutor):
    def __init__(
        self,
        pipe_id: str,
        launcher_id: Optional[str] = None,
        launch_timeout: Optional[float] = None,
        task_wait_timeout: Optional[float] = None,
        last_result_transfer_timeout: float = 300.0,
        external_pre_init_timeout: float = 300.0,
        peer_read_timeout: Optional[float] = 300.0,
        monitor_interval: float = 0.01,
        read_interval: float = 0.5,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 300.0,
        workers: int = 4,
        train_with_evaluation: bool = False,
        train_task_name: str = AppConstants.TASK_TRAIN,
        evaluate_task_name: str = AppConstants.TASK_VALIDATION,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
        from_nvflare_converter_id: Optional[str] = None,
        to_nvflare_converter_id: Optional[str] = None,
        params_exchange_format: str = ExchangeFormat.NUMPY,
        params_transfer_type: str = TransferType.FULL,
        config_file_name: str = CLIENT_API_CONFIG,
        server_expected_format: str = ExchangeFormat.NUMPY,
        memory_gc_rounds: int = 0,
        cuda_empty_cache: bool = False,
    ) -> None:
        """Initializes the ClientAPILauncherExecutor.

        Args:
            pipe_id (str): Identifier for obtaining the Pipe from NVFlare components.
            launcher_id (Optional[str]): Identifier for obtaining the Launcher from NVFlare components.
            launch_timeout (Optional[float]): Timeout for the Launcher's "launch_task" method to complete (None for no timeout).
            task_wait_timeout (Optional[float]): Timeout for retrieving the task result (None for no timeout).
            last_result_transfer_timeout (float): Timeout for transmitting the last result from an external process.
                This value should be greater than the time needed for sending the whole result.
            external_pre_init_timeout (float): Time to wait for external process before it calls flare.init().
            peer_read_timeout (float, optional): time to wait for peer to accept sent message.
            monitor_interval (float): Interval for monitoring the launcher.
            read_interval (float): Interval for reading from the pipe.
            heartbeat_interval (float): Interval for sending heartbeat to the peer.
            heartbeat_timeout (float): Timeout for waiting for a heartbeat from the peer.
            workers (int): Number of worker threads needed.
            train_with_evaluation (bool): Whether to run training with global model evaluation.
            train_task_name (str): Task name of train mode.
            evaluate_task_name (str): Task name of evaluate mode.
            submit_model_task_name (str): Task name of submit_model mode.
            from_nvflare_converter_id (Optional[str]): Identifier used to get the ParamsConverter from NVFlare components.
                This ParamsConverter will be called when model is sent from nvflare controller side to executor side.
            to_nvflare_converter_id (Optional[str]): Identifier used to get the ParamsConverter from NVFlare components.
                This ParamsConverter will be called when model is sent from nvflare executor side to controller side.
            server_expected_format (str): What format to exchange the parameters between server and client.
            params_exchange_format (str): What format to exchange the parameters between client and script.
            params_transfer_type (str): How to transfer the parameters. FULL means the whole model parameters are sent.
                DIFF means that only the difference is sent.
            config_file_name (str): The config file name to write attributes into, the client api will read in this file.
        """
        LauncherExecutor.__init__(
            self,
            pipe_id=pipe_id,
            launcher_id=launcher_id,
            launch_timeout=launch_timeout,
            task_wait_timeout=task_wait_timeout,
            last_result_transfer_timeout=last_result_transfer_timeout,
            external_pre_init_timeout=external_pre_init_timeout,
            peer_read_timeout=peer_read_timeout,
            monitor_interval=monitor_interval,
            read_interval=read_interval,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
            workers=workers,
            train_with_evaluation=train_with_evaluation,
            train_task_name=train_task_name,
            evaluate_task_name=evaluate_task_name,
            submit_model_task_name=submit_model_task_name,
            from_nvflare_converter_id=from_nvflare_converter_id,
            to_nvflare_converter_id=to_nvflare_converter_id,
        )

        self._server_expected_format = server_expected_format
        self._params_exchange_format = params_exchange_format
        self._params_transfer_type = params_transfer_type
        self._config_file_name = config_file_name
        self._memory_gc_rounds = memory_gc_rounds
        self._cuda_empty_cache = cuda_empty_cache
        self._cell_with_pass_through = None
        self._prev_pass_through = None
        self._pipe_cell_with_pass_through = None
        self._pipe_prev_pass_through = None
        self._cj_round_count = 0
        self._cj_memory_profile_enabled = False

    def initialize(self, fl_ctx: FLContext) -> None:
        self.prepare_config_for_launch(fl_ctx)
        # Enable PASS_THROUGH mode on the engine's communication cell so that
        # large tensors arriving from the FL server are NOT downloaded here at
        # the CJ.  ViaDownloaderDecomposer will instead create LazyDownloadRef
        # placeholders that carry the original server FQCN and ref_id.  When CJ
        # forwards the task to the subprocess agent via the task pipe, those
        # placeholders are re-emitted as-is, causing the subprocess to download
        # each tensor directly from the server — one tensor at a time, with no
        # size limit and no tensor copy at CJ.
        engine = fl_ctx.get_engine()
        cell = engine.get_cell()
        if cell is not None:
            self._cell_with_pass_through = cell
            prev_ctx = cell.core_cell.get_fobs_context()
            self._prev_pass_through = prev_ctx.get(FOBSContextKey.PASS_THROUGH, None)
            cell.core_cell.update_fobs_context({FOBSContextKey.PASS_THROUGH: True})
            self.log_info(
                fl_ctx,
                "PASS_THROUGH enabled: task tensors will be downloaded by the subprocess "
                "agent directly from the source, bypassing CJ memory.",
            )

        # Enable PASS_THROUGH on task pipe cell for reverse direction
        # (subprocess -> CJ -> server) so CJ can forward lazy refs instead of
        # materializing tensors when receiving results from subprocess.
        if isinstance(self.pipe, CellPipe):
            pipe_cell = getattr(self.pipe, "cell", None)
            if pipe_cell is not None and pipe_cell is not cell:
                self._pipe_cell_with_pass_through = pipe_cell
                pipe_prev_ctx = pipe_cell.core_cell.get_fobs_context()
                self._pipe_prev_pass_through = pipe_prev_ctx.get(FOBSContextKey.PASS_THROUGH, None)
                pipe_cell.core_cell.update_fobs_context({FOBSContextKey.PASS_THROUGH: True})
                self.log_info(
                    fl_ctx,
                    "PASS_THROUGH enabled on task pipe cell: reverse task-result tensors "
                    "will bypass CJ memory.",
                )
        try:
            super().initialize(fl_ctx)
        except Exception:
            self._restore_pass_through(fl_ctx)
            raise

        # Check for top-level config override for external_pre_init_timeout
        # This allows jobs to configure timeout via add_client_config()
        config_timeout = get_client_config_value(fl_ctx, EXTERNAL_PRE_INIT_TIMEOUT)
        if config_timeout is not None:
            timeout_value = float(config_timeout)
            if timeout_value <= 0:
                self.log_error(fl_ctx, f"Invalid EXTERNAL_PRE_INIT_TIMEOUT: {timeout_value}s (must be positive)")
                raise ValueError(f"EXTERNAL_PRE_INIT_TIMEOUT must be positive, got {timeout_value}")
            self.log_info(
                fl_ctx,
                f"Overriding external_pre_init_timeout from config: {self._external_pre_init_timeout}s -> {timeout_value}s",
            )
            self._external_pre_init_timeout = timeout_value

        self._cj_memory_profile_enabled = self._read_cj_memory_profile_enabled(fl_ctx)
        if self._cj_memory_profile_enabled:
            self.log_info(fl_ctx, "CJ memory profile enabled.")

    def finalize(self, fl_ctx: FLContext) -> None:
        try:
            super().finalize(fl_ctx)
        finally:
            self._restore_pass_through(fl_ctx)

    def _restore_pass_through(self, fl_ctx: FLContext):
        if self._cell_with_pass_through is not None:
            self._cell_with_pass_through.core_cell.update_fobs_context(
                {FOBSContextKey.PASS_THROUGH: self._prev_pass_through}
            )
            self.log_info(fl_ctx, f"PASS_THROUGH restored to {self._prev_pass_through}.")
            self._cell_with_pass_through = None
            self._prev_pass_through = None

        if self._pipe_cell_with_pass_through is not None:
            self._pipe_cell_with_pass_through.core_cell.update_fobs_context(
                {FOBSContextKey.PASS_THROUGH: self._pipe_prev_pass_through}
            )
            self.log_info(fl_ctx, f"Task-pipe PASS_THROUGH restored to {self._pipe_prev_pass_through}.")
            self._pipe_cell_with_pass_through = None
            self._pipe_prev_pass_through = None

    def prepare_config_for_launch(self, fl_ctx: FLContext):
        pipe_export_class, pipe_export_args = self.pipe.export(ExportMode.PEER)
        task_exchange_attributes = {
            ConfigKey.TRAIN_WITH_EVAL: self._train_with_evaluation,
            ConfigKey.EXCHANGE_FORMAT: self._params_exchange_format,
            ConfigKey.SERVER_EXPECTED_FORMAT: self._server_expected_format,
            ConfigKey.TRANSFER_TYPE: self._params_transfer_type,
            ConfigKey.TRAIN_TASK_NAME: self._train_task_name,
            ConfigKey.EVAL_TASK_NAME: self._evaluate_task_name,
            ConfigKey.SUBMIT_MODEL_TASK_NAME: self._submit_model_task_name,
            ConfigKey.PIPE_CHANNEL_NAME: self.get_pipe_channel_name(),
            ConfigKey.PIPE: {
                ConfigKey.CLASS_NAME: pipe_export_class,
                ConfigKey.ARG: pipe_export_args,
            },
            ConfigKey.HEARTBEAT_TIMEOUT: self.heartbeat_timeout,
            ConfigKey.MEMORY_GC_ROUNDS: self._memory_gc_rounds,
            ConfigKey.CUDA_EMPTY_CACHE: self._cuda_empty_cache,
        }

        config_data = {
            ConfigKey.TASK_EXCHANGE: task_exchange_attributes,
        }

        update_export_props(config_data, fl_ctx)
        config_file_path = self._get_external_config_file_path(fl_ctx)
        write_config_to_file(config_data=config_data, config_file_path=config_file_path)

    def _get_external_config_file_path(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        workspace = engine.get_workspace()
        app_config_directory = workspace.get_app_config_dir(fl_ctx.get_job_id())
        config_file_path = os.path.join(app_config_directory, self._config_file_name)
        return config_file_path

    def check_output_shareable(self, task_name: str, shareable: Shareable, fl_ctx: FLContext) -> bool:
        ok = super().check_output_shareable(task_name, shareable, fl_ctx)
        if not ok:
            return False

        current_round = shareable.get_header(AppConstants.CURRENT_ROUND, None)
        self._maybe_profile_and_cleanup_cj_memory(fl_ctx, task_name, current_round)
        return True

    def _maybe_profile_and_cleanup_cj_memory(self, fl_ctx: FLContext, task_name: str, current_round):
        self._log_cj_rss(fl_ctx, task_name=task_name, current_round=current_round, stage="result_ready")

        if self._memory_gc_rounds <= 0:
            return

        self._cj_round_count += 1
        if self._cj_round_count % self._memory_gc_rounds != 0:
            return

        self._log_cj_rss(fl_ctx, task_name=task_name, current_round=current_round, stage="before_cleanup")
        cleanup_memory(cuda_empty_cache=self._cuda_empty_cache)
        self.log_debug(fl_ctx, f"CJ memory cleanup performed at result #{self._cj_round_count}.")
        self._log_cj_rss(fl_ctx, task_name=task_name, current_round=current_round, stage="after_cleanup")

    def _log_cj_rss(self, fl_ctx: FLContext, task_name: str, current_round, stage: str):
        if not self._cj_memory_profile_enabled:
            return
        rss_mb = self._get_process_rss_mb()
        if rss_mb is None:
            self.log_info(
                fl_ctx,
                f"CJ memory profile: stage={stage} task={task_name} round={current_round} rss_mb=unavailable",
            )
            return
        self.log_info(
            fl_ctx,
            f"CJ memory profile: stage={stage} task={task_name} round={current_round} rss_mb={rss_mb:.2f}",
        )

    @staticmethod
    def _read_cj_memory_profile_enabled(fl_ctx: FLContext) -> bool:
        profile = None
        for key in ("CLIENT_MEMORY_PROFILE", "CLIENT_Memory_profile"):
            profile = get_client_config_value(fl_ctx, key, None)
            if profile is not None:
                break

        if profile is None:
            for key in ("NVFLARE_CLIENT_MEMORY_PROFILE", "CLIENT_MEMORY_PROFILE", "CLIENT_Memory_profile"):
                profile = os.environ.get(key)
                if profile is not None:
                    break

        parsed = str2bool(profile)
        return parsed if parsed is not None else False

    @staticmethod
    def _get_process_rss_mb():
        try:
            import psutil

            return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            pass

        try:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return rss / (1024.0 * 1024.0)
            return rss / 1024.0
        except Exception:
            return None
