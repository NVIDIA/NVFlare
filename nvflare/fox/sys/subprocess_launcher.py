# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Subprocess launcher for FoxWorker.

This module provides the SubprocessLauncher class that FoxExecutor uses to
spawn and manage FoxWorker subprocesses for distributed training.

The launcher:
1. Sets up environment variables for worker connection
2. Spawns the subprocess (optionally via torchrun)
3. Forwards calls from FoxExecutor to the worker
4. Manages the subprocess lifecycle

User's training code remains unchanged - they just use @fox.collab as normal.
"""

import os
import shlex
import subprocess
import threading
from typing import Any, Dict, List, Optional

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.log_utils import get_obj_logger

from .worker import (
    ENV_PARENT_FQCN,
    ENV_PARENT_URL,
    ENV_SITE_NAME,
    ENV_SUBPROCESS_TIMEOUT,
    ENV_TRACKING_TYPE,
    ENV_WORKER_ID,
    WORKER_CALL_TOPIC,
    WORKER_CHANNEL,
    WORKER_READY_TOPIC,
    WORKER_SHUTDOWN_TOPIC,
)

# Default timeout values (can be overridden via constructor)
DEFAULT_SHUTDOWN_TIMEOUT = 5.0
DEFAULT_PROCESS_WAIT_TIMEOUT = 10.0


class SubprocessLauncher:
    """Manages FoxWorker subprocess for distributed training.

    This class handles:
    1. Setting environment variables for worker connection
    2. Spawning the worker subprocess (optionally via torchrun)
    3. Forwarding calls from FoxExecutor to the worker
    4. Managing the subprocess lifecycle

    Example:
        launcher = SubprocessLauncher(
            site_name="site-1",
            training_module="my_training",
            parent_cell=cell,
            launcher_cmd="torchrun --nproc_per_node=4",
        )
        launcher.start()
        result = launcher.call("train", args=(weights,))
        launcher.stop()
    """

    def __init__(
        self,
        site_name: str,
        training_module: str,
        parent_cell: CoreCell,
        launcher_cmd: Optional[str] = None,
        subprocess_timeout: float = 300.0,
        worker_id: str = "0",
        shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT,
        process_wait_timeout: float = DEFAULT_PROCESS_WAIT_TIMEOUT,
        tracking_type: Optional[str] = None,
    ):
        """Initialize SubprocessLauncher.

        Args:
            site_name: Name of this site (e.g., site-1)
            training_module: Python module path containing @fox.collab methods
            parent_cell: CellNet cell of the parent FoxExecutor
            launcher_cmd: Optional launcher command (e.g., "torchrun --nproc_per_node=4")
                         If None, runs FoxWorker directly without a launcher.
            subprocess_timeout: Timeout for subprocess call operations.
            worker_id: Unique ID for this worker (default "0")
            shutdown_timeout: Timeout for sending shutdown signal.
            process_wait_timeout: Timeout for waiting for process to terminate.
            tracking_type: Type of experiment tracking (e.g., "mlflow", "tensorboard", "wandb").
        """
        self.site_name = site_name
        self.training_module = training_module
        self.parent_cell = parent_cell
        self.launcher_cmd = launcher_cmd
        self.subprocess_timeout = subprocess_timeout
        self.worker_id = worker_id
        self.shutdown_timeout = shutdown_timeout
        self.process_wait_timeout = process_wait_timeout
        self.tracking_type = tracking_type

        self.logger = get_obj_logger(self)
        self._process: Optional[subprocess.Popen] = None
        self._worker_fqcn: Optional[str] = None
        self._ready_event = threading.Event()
        self._worker_info: Dict[str, Any] = {}

        # Get listener URL for worker connection
        self._listener_url = self._get_listener_url()

    def _get_listener_url(self) -> str:
        """Get the CellNet URL for worker to connect to."""
        # Try to get internal listener URL from parent cell
        try:
            url = self.parent_cell.get_internal_listener_url()
            if url:
                return url
        except Exception:
            pass

        # Fallback: use a default local URL
        return "grpc://localhost:8002"

    def _setup_handlers(self):
        """Set up CellNet handlers for worker communication."""
        # Register handler for worker ready signal
        self.parent_cell.register_request_cb(
            channel=WORKER_CHANNEL,
            topic=WORKER_READY_TOPIC,
            cb=self._handle_worker_ready,
        )

    def _handle_worker_ready(self, request: Message) -> Message:
        """Handle ready signal from worker subprocess."""
        payload = request.payload
        self.logger.info(f"Worker ready: rank={payload.get('rank')}, world_size={payload.get('world_size')}")

        self._worker_info = payload
        self._worker_fqcn = f"{self.site_name}.worker.{payload.get('worker_id', '0')}"
        self._ready_event.set()

        return Message(payload={"status": "acknowledged"})

    def _build_subprocess_env(self) -> Dict[str, str]:
        """Build environment variables for the subprocess.

        Returns:
            Environment dict with Fox connection info added
        """
        env = os.environ.copy()

        # Add Fox connection info (invisible to user's code)
        env[ENV_PARENT_URL] = self._listener_url
        env[ENV_PARENT_FQCN] = self.parent_cell.get_fqcn()
        env[ENV_SITE_NAME] = self.site_name
        env[ENV_WORKER_ID] = self.worker_id
        env[ENV_SUBPROCESS_TIMEOUT] = str(self.subprocess_timeout)

        # Add tracking type if configured
        if self.tracking_type:
            env[ENV_TRACKING_TYPE] = self.tracking_type

        # Ensure the training module can be imported
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{os.getcwd()}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = os.getcwd()

        return env

    def _build_subprocess_cmd(self) -> List[str]:
        """Build the command to launch the worker subprocess.

        Returns:
            List of command arguments
        """
        # Worker command: python -m nvflare.fox.sys.worker <training_module>
        worker_cmd = [
            "python",
            "-m",
            "nvflare.fox.sys.worker",
            self.training_module,
        ]

        if self.launcher_cmd:
            # Prepend launcher (e.g., "torchrun --nproc_per_node=4")
            # Result: torchrun --nproc_per_node=4 python -m nvflare.fox.sys.worker my_training
            launcher_parts = shlex.split(self.launcher_cmd)
            return launcher_parts + worker_cmd
        else:
            return worker_cmd

    def start(self) -> bool:
        """Start the worker subprocess.

        Returns:
            True if worker started and is ready, False otherwise
        """
        self.logger.info(f"Starting subprocess for {self.site_name}")

        try:
            # Set up handlers for worker communication
            self._setup_handlers()

            # Build command and environment
            cmd = self._build_subprocess_cmd()
            env = self._build_subprocess_env()

            self.logger.info(f"Launching: {' '.join(cmd)}")
            self.logger.debug(f"  {ENV_PARENT_URL}={env[ENV_PARENT_URL]}")
            self.logger.debug(f"  {ENV_PARENT_FQCN}={env[ENV_PARENT_FQCN]}")

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
            )

            # Start thread to capture output
            output_thread = threading.Thread(target=self._capture_output, daemon=True)
            output_thread.start()

            # Wait for worker to signal ready
            self.logger.info("Waiting for worker to become ready...")
            if not self._ready_event.wait(timeout=self.subprocess_timeout):
                self.logger.error("Worker subprocess did not become ready in time")
                self.stop()
                return False

            self.logger.info("Worker subprocess ready")
            return True

        except Exception as e:
            self.logger.exception(f"Failed to start worker subprocess: {e}")
            self.stop()
            return False

    def _capture_output(self):
        """Capture and log subprocess output."""
        if self._process and self._process.stdout:
            for line in self._process.stdout:
                self.logger.info(f"[Worker] {line.rstrip()}")

    def call(self, func_name: str, args: tuple = (), kwargs: dict = None) -> Any:
        """Forward a call to the worker subprocess.

        Args:
            func_name: Name of the function to call
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Result from the worker

        Raises:
            RuntimeError: If worker is not ready or call fails
        """
        if not self._ready_event.is_set():
            raise RuntimeError("Worker subprocess is not ready")

        if kwargs is None:
            kwargs = {}

        request = Message(
            payload={
                "func_name": func_name,
                "args": args,
                "kwargs": kwargs,
            }
        )

        self.logger.debug(f"Forwarding call to worker: {func_name}")

        try:
            reply = self.parent_cell.send_request(
                channel=WORKER_CHANNEL,
                topic=WORKER_CALL_TOPIC,
                target=self._worker_fqcn,
                request=request,
                timeout=self.subprocess_timeout,
            )

            if reply is None:
                raise RuntimeError(f"No response from worker for {func_name}")

            payload = reply.payload
            if "error" in payload:
                raise RuntimeError(f"Worker error: {payload['error']}")

            return payload.get("result")

        except Exception as e:
            self.logger.error(f"Call to worker failed: {e}")
            raise

    def stop(self):
        """Stop the worker subprocess."""
        self.logger.info("Stopping worker subprocess")

        # Send shutdown signal if worker is ready
        if self._ready_event.is_set() and self._worker_fqcn:
            try:
                self.parent_cell.send_request(
                    channel=WORKER_CHANNEL,
                    topic=WORKER_SHUTDOWN_TOPIC,
                    target=self._worker_fqcn,
                    request=Message(payload={}),
                    timeout=self.shutdown_timeout,
                )
            except Exception as e:
                self.logger.warning(f"Error sending shutdown signal: {e}")

        # Terminate process if still running
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=self.process_wait_timeout)
            except subprocess.TimeoutExpired:
                self._process.kill()
            except Exception as e:
                self.logger.warning(f"Error terminating process: {e}")

            self._process = None

        self._ready_event.clear()
        self._worker_fqcn = None

    def is_ready(self) -> bool:
        """Check if worker is ready."""
        return self._ready_event.is_set()

    def is_running(self) -> bool:
        """Check if subprocess is running."""
        if self._process is None:
            return False
        return self._process.poll() is None
