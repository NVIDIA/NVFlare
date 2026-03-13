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
from typing import Optional

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.executors.launcher_executor import LauncherExecutor
from nvflare.app_common.utils.export_utils import update_export_props
from nvflare.client.config import ConfigKey, ExchangeFormat, TransferType, write_config_to_file
from nvflare.client.constants import CLIENT_API_CONFIG, EXTERNAL_PRE_INIT_TIMEOUT, PEER_READ_TIMEOUT
from nvflare.fuel.utils.attributes_exportable import ExportMode
from nvflare.fuel.utils.fobs.decomposers.via_downloader import MIN_DOWNLOAD_TIMEOUT_DEFAULT
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
        submit_result_timeout: float = 300.0,
        max_resends: int = 3,
        download_complete_timeout: float = 1800.0,
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
            submit_result_timeout (float): How long (seconds) the subprocess waits for CJ to acknowledge each result
                pipe message.  With reverse PASS_THROUGH enabled CJ ACKs immediately (LazyDownloadRef creation is
                microseconds), so 300 s is a very generous allowance.  Without reverse PASS_THROUGH, CJ must
                download the full result before ACKing; in that case this should be at least as large as the
                expected transfer time.  Configurable via recipe.add_client_config({"submit_result_timeout": N}).
            max_resends (int): Maximum number of times the subprocess retries sending the result if CJ does not
                ACK within submit_result_timeout.  Defaults to 3.  None means unlimited (unsafe for large models
                — each retry creates a new download transaction).  Configurable via
                recipe.add_client_config({"max_resends": N}).
            download_complete_timeout (float): How long (seconds) the subprocess waits after send_to_peer() ACKs
                for the server to finish downloading its tensors from the subprocess DownloadService.  Without
                this gate, the subprocess may exit before the download completes and the server gets
                "no ref found".  Defaults to 1800 s.  Configurable via
                recipe.add_client_config({"download_complete_timeout": N}).
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
        self._submit_result_timeout = submit_result_timeout
        self._max_resends = max_resends
        self._download_complete_timeout = download_complete_timeout
        self._cj_round_count = 0

        # Allow the subprocess to exit naturally after Fix 16's download_done.wait()
        # before stop_task() sends SIGTERM.  Without this, _finalize_external_execution()
        # kills the subprocess immediately, tearing down its cell connection before the
        # server can download tensors from it ("no path" / deadlock).
        self._stop_task_wait_timeout = download_complete_timeout

    def initialize(self, fl_ctx: FLContext) -> None:
        self.prepare_config_for_launch(fl_ctx)
        # PASS_THROUGH (reverse path — subprocess → CJ):
        # The subprocess-side CellPipe (pass_through_on_send=True, set in
        # ExProcessClientAPI.init()) stamps MessageHeaderKey.PASS_THROUGH on
        # result messages so CJ decodes them as LazyDownloadRef and forwards
        # the original subprocess datum to the server for direct download.
        #
        # PASS_THROUGH (forward path — server/aggregator → CJ):
        # Receiver-side opt-in: set cell.decode_pass_through = True so that
        # incoming task messages are decoded with PASS_THROUGH regardless of
        # whether the sender stamped the header.  This lets the subprocess
        # download tensors directly from the source (server or aggregator)
        # via DownloadService, skipping materialisation inside the CJ.
        # Safe because this executor IS an ext-process launcher — it never
        # consumes the tensors itself; it just forwards them to the subprocess.
        super().initialize(fl_ctx)

        engine = fl_ctx.get_engine()
        get_cell_fn = getattr(engine, "get_cell", None)
        if get_cell_fn:
            cell = get_cell_fn()
            if cell is not None:
                cell.decode_pass_through = True
                self.log_info(fl_ctx, "Receiver-side PASS_THROUGH enabled on CJ cell (forward path)")

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

        # Check for top-level config override for peer_read_timeout.
        # peer_read_timeout (CJ side) and submit_result_timeout (subprocess side) must be
        # configured together so the system behaves consistently under large-model transfers.
        # Placing both overrides in config_fed_client.json (via add_client_config()) lets
        # operators tune them in one place without touching executor component parameters.
        config_peer_timeout = get_client_config_value(fl_ctx, PEER_READ_TIMEOUT)
        if config_peer_timeout is not None:
            peer_timeout_value = float(config_peer_timeout)
            if peer_timeout_value <= 0:
                self.log_error(fl_ctx, f"Invalid PEER_READ_TIMEOUT: {peer_timeout_value}s (must be positive)")
                raise ValueError(f"PEER_READ_TIMEOUT must be positive, got {peer_timeout_value}")
            self.log_info(
                fl_ctx,
                f"Overriding peer_read_timeout from config: {self.peer_read_timeout}s -> {peer_timeout_value}s",
            )
            self.peer_read_timeout = peer_timeout_value

        self._validate_timeout_config(fl_ctx)

    def _decomposer_prefix(self) -> str:
        """Return the config-var prefix for the active decomposer type.

        The prefix must match what the ViaDownloaderDecomposer subclass uses
        (e.g. NumpyArrayDecomposer → "np_") so that _validate_timeout_config()
        reads the same job-config keys as the download infrastructure.

        Framework-specific subclasses (e.g. PTClientAPILauncherExecutor) should
        override this method to return their decomposer's prefix (e.g. "tensor_"),
        keeping this base class free of framework-specific knowledge.
        """
        return "np_"

    def _validate_timeout_config(self, fl_ctx: FLContext):
        """Warn at job start if timeout parameters are inconsistent.

        Checks are advisory (log_warning, not raise) so a misconfigured job
        can still run — the messages give the operator actionable guidance
        before the first download attempt.
        """
        try:
            import nvflare.fuel.utils.app_config_utils as acu
            from nvflare.apis.fl_constant import ConfigVarName
        except ImportError as e:
            self.log_warning(fl_ctx, f"_validate_timeout_config skipped: {e}")
            return

        prefix = self._decomposer_prefix()
        per_req = acu.get_positive_float_var(f"{prefix}{ConfigVarName.STREAMING_PER_REQUEST_TIMEOUT}", 600.0)
        min_dl = acu.get_positive_float_var(
            f"{prefix}{ConfigVarName.MIN_DOWNLOAD_TIMEOUT}", MIN_DOWNLOAD_TIMEOUT_DEFAULT
        )

        if min_dl < per_req:
            self.log_warning(
                fl_ctx,
                f"Timeout inconsistency: {prefix}min_download_timeout ({min_dl}s) < "
                f"{prefix}streaming_per_request_timeout ({per_req}s). "
                f"Transactions may be killed mid-download. "
                f"Set {prefix}min_download_timeout >= {per_req}s in job config.",
            )

        if self._submit_result_timeout > min_dl:
            self.log_warning(
                fl_ctx,
                f"Timeout inconsistency: submit_result_timeout ({self._submit_result_timeout}s) > "
                f"{prefix}min_download_timeout ({min_dl}s). "
                f"Each send attempt may expire the download transaction before the next retry. "
                f"Fix: set {prefix}min_download_timeout >= {self._submit_result_timeout}s in job config "
                f'(e.g. recipe.add_client_config({{"{prefix}min_download_timeout": {int(self._submit_result_timeout)}}})).',
            )

        if self._max_resends is None:
            self.log_warning(
                fl_ctx,
                "max_resends is None (unbounded). This risks OOM on large model transfers. "
                "Set max_resends to a bounded value (e.g. 3) in job config.",
            )

    def check_output_shareable(self, task_name: str, shareable: Shareable, fl_ctx: FLContext) -> bool:
        ok = super().check_output_shareable(task_name, shareable, fl_ctx)
        if not ok:
            return False
        from nvflare.fuel.utils.mem_utils import log_rss

        site_name = fl_ctx.get_identity_name()
        log_rss(f"CJ s={site_name} t={task_name} r={shareable.get_header(AppConstants.CURRENT_ROUND)} relay")
        self._maybe_cleanup_cj_memory(fl_ctx)
        return True

    def _maybe_cleanup_cj_memory(self, fl_ctx: FLContext):
        """Call cleanup_memory() every memory_gc_rounds rounds on the client job process.

        Mirrors the subprocess-side cleanup in APISpec._maybe_cleanup_memory().
        Runs at the point the client job process has finished relaying the
        subprocess result to the server — the result Shareable and any tensors
        it referenced are no longer needed, making this the right moment to
        force a GC cycle.
        """
        if self._memory_gc_rounds <= 0:
            return
        self._cj_round_count += 1
        if self._cj_round_count % self._memory_gc_rounds == 0:
            from nvflare.fuel.utils.memory_utils import cleanup_memory

            cleanup_memory(cuda_empty_cache=self._cuda_empty_cache)
            self.log_info(fl_ctx, f"Client job memory cleanup performed at round {self._cj_round_count}.")

    def _resolve_launch_once(self, fl_ctx: FLContext) -> bool:
        """Return True if the subprocess is launched once for the whole job.

        self.launcher may be None when prepare_config_for_launch() is called during
        initialize() (before _initialize_external_execution() assigns it), so we
        fetch the launcher component directly from the engine.
        """
        launcher = self.launcher
        if launcher is None and self._launcher_id:
            engine = fl_ctx.get_engine()
            launcher = engine.get_component(self._launcher_id)
        if launcher is None:
            return False  # safe default: treat as per-round (direct os._exit path)
        return not launcher.needs_deferred_stop()

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
            ConfigKey.SUBMIT_RESULT_TIMEOUT: self._submit_result_timeout,
            ConfigKey.MAX_RESENDS: self._max_resends,
            ConfigKey.DOWNLOAD_COMPLETE_TIMEOUT: self._download_complete_timeout,
            ConfigKey.LAUNCH_ONCE: self._resolve_launch_once(fl_ctx),
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
