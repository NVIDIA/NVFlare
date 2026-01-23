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

"""Subprocess launcher for CollabWorker.

This module provides the SubprocessLauncher class that CollabExecutor uses to
spawn and manage CollabWorker subprocesses for distributed training.

The launcher:
1. Sets up environment variables for worker connection
2. Spawns the subprocess (optionally via torchrun)
3. Forwards calls from CollabExecutor to the worker
4. Manages the subprocess lifecycle

User's training code remains unchanged - they just use @collab.publish as normal.
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
    ENV_CLIENT_CLASS,
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
    """Manages CollabWorker subprocess for distributed training.

    This class handles:
    1. Setting environment variables for worker connection
    2. Spawning the worker subprocess (optionally via torchrun)
    3. Forwarding calls from CollabExecutor to the worker
    4. Managing the subprocess lifecycle

    Example:
        launcher = SubprocessLauncher(
            site_name="site-1",
            training_module="my_training",
            parent_cell=cell,
            run_cmd="torchrun --nproc_per_node=4",
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
        run_cmd: Optional[str] = None,
        subprocess_timeout: float = 300.0,
        worker_id: str = "0",
        shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT,
        process_wait_timeout: float = DEFAULT_PROCESS_WAIT_TIMEOUT,
        tracking_type: Optional[str] = None,
        site_index: int = 0,
        client_class: Optional[str] = None,
    ):
        """Initialize SubprocessLauncher.

        Args:
            site_name: Name of this site (e.g., site-1)
            training_module: Python module path containing @collab.publish methods
            parent_cell: CellNet cell of the parent CollabExecutor
            run_cmd: Optional command prefix (e.g., "torchrun --nproc_per_node=4")
                     If None, runs CollabWorker directly.
            subprocess_timeout: Timeout for subprocess call operations.
            worker_id: Unique ID for this worker (default "0")
            shutdown_timeout: Timeout for sending shutdown signal.
            process_wait_timeout: Timeout for waiting for process to terminate.
            tracking_type: Type of experiment tracking (e.g., "mlflow", "tensorboard", "wandb").
            site_index: Index of this site (for unique port assignment, 0-based).
            client_class: Optional class name for class-based clients (e.g., "Trainer").
        """
        self.site_name = site_name
        self.training_module = training_module
        self.parent_cell = parent_cell
        self.run_cmd = run_cmd
        self.subprocess_timeout = subprocess_timeout
        self.worker_id = worker_id
        self.shutdown_timeout = shutdown_timeout
        self.process_wait_timeout = process_wait_timeout
        self.tracking_type = tracking_type
        self.site_index = site_index
        self.client_class = client_class

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

        # Fallback: use configured internal connection scheme
        from nvflare.fuel.f3.comm_config import CommConfigurator

        comm_configurator = CommConfigurator()
        scheme = comm_configurator.get_internal_connection_scheme("tcp")
        return f"{scheme}://localhost:8002"

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
        # Worker FQCN is a child of parent cell's FQCN: parent.worker.id
        parent_fqcn = self.parent_cell.get_fqcn()
        self._worker_fqcn = f"{parent_fqcn}.worker.{payload.get('worker_id', '0')}"
        self.logger.debug(f"Worker FQCN: {self._worker_fqcn}")
        self._ready_event.set()

        return Message(payload={"status": "acknowledged"})

    def _build_subprocess_env(self) -> Dict[str, str]:
        """Build environment variables for the subprocess.

        Returns:
            Environment dict with Collab connection info added
        """
        env = os.environ.copy()

        # Add Collab connection info (invisible to user's code)
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

        # Set unique MASTER_PORT for torchrun to avoid port conflicts
        # when multiple sites run on the same machine (simulation mode)
        # Base port 29500 + site_index to ensure each site uses a different port
        unique_port = 29500 + self.site_index
        env["MASTER_PORT"] = str(unique_port)

        # Add client class for class-based clients
        if self.client_class:
            env[ENV_CLIENT_CLASS] = self.client_class

        return env

    def _build_subprocess_cmd(self) -> List[str]:
        """Build the command to launch the worker subprocess.

        Returns:
            List of command arguments

        The command format depends on whether a run_cmd (launcher) is specified:
        - Without run_cmd: python -m nvflare.collab.sys.worker <training_module>
        - With run_cmd:    torchrun [opts] --master-port=X -m nvflare.collab.sys.worker <training_module>

        For torchrun/mpirun style launchers, we use -m directly on the launcher
        since they invoke Python internally. This avoids the invalid command:
            torchrun python -m ...  (wrong: torchrun would try to run 'python' as a script)

        For torchrun specifically, we inject --master-port to ensure each site uses
        a unique port (avoids conflicts in simulation mode).
        """
        # Worker module and training module as arguments
        worker_module = "nvflare.collab.sys.worker"

        if self.run_cmd:
            # Launcher-based execution (e.g., torchrun, mpirun)
            run_cmd_parts = shlex.split(self.run_cmd)

            # Check if this is torchrun and inject unique --master-port
            if run_cmd_parts and "torchrun" in run_cmd_parts[0]:
                # Calculate unique port for this site
                unique_port = 29500 + self.site_index
                run_cmd_parts.append(f"--master-port={unique_port}")

            return run_cmd_parts + ["-m", worker_module, self.training_module]
        else:
            # Direct Python execution
            return ["python", "-m", worker_module, self.training_module]

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
            self.logger.info(f"Process spawned (pid={self._process.pid})")

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
                output = line.rstrip()
                self.logger.debug(f"[Worker] {output}")

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
            if payload is None:
                raise RuntimeError(f"Empty payload in response from worker for {func_name}")

            if "error" in payload:
                raise RuntimeError(f"Worker error: {payload['error']}")

            return payload.get("result")

        except Exception as e:
            self.logger.error(f"Call to worker failed: {e}")
            # Check if subprocess is still running
            if self._process:
                poll_result = self._process.poll()
                if poll_result is not None:
                    self.logger.error(f"Worker subprocess exited with code: {poll_result}")
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
