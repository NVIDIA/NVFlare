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

"""CollabWorker: Transparent subprocess worker for distributed training.

This module is the internal entry point that CollabExecutor uses when spawning
subprocess-based training (e.g., with torchrun). Users never interact with
this directly - they just write normal @collab.publish decorated functions.

Architecture:
    User writes:        train.py with @collab.publish decorated functions

    CollabExecutor runs:   torchrun --nproc_per_node=4 -m nvflare.collab.sys.worker train
                                                    ↑ this module        ↑ user's module

    Environment vars (set by CollabExecutor, invisible to user):
        COLLAB_PARENT_URL    = tcp://localhost:8002  (protocol from FLARE config)
        COLLAB_PARENT_FQCN   = site-1.job123
        COLLAB_SITE_NAME     = site-1

Example:
    # User's train.py - completely normal, no worker code needed!
    from nvflare.collab import collab

    @collab.publish
    def train(weights=None):
        import torch.distributed as dist
        dist.init_process_group("nccl")
        # ... DDP training ...
        return model.state_dict(), loss

    # That's it! No main block needed for subprocess mode.
"""

import importlib
import logging
import os
import signal
import sys
import threading
import time
from typing import Optional

from nvflare.collab.utils.decomposers import register_available_decomposers
from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils.log_utils import get_obj_logger

# Environment variable names (set by CollabExecutor)
ENV_PARENT_URL = "COLLAB_PARENT_URL"
ENV_PARENT_FQCN = "COLLAB_PARENT_FQCN"
ENV_SITE_NAME = "COLLAB_SITE_NAME"
ENV_WORKER_ID = "COLLAB_WORKER_ID"
ENV_SUBPROCESS_TIMEOUT = "COLLAB_SUBPROCESS_TIMEOUT"
ENV_TRACKING_TYPE = "COLLAB_TRACKING_TYPE"
ENV_CLIENT_CLASS = "COLLAB_CLIENT_CLASS"  # Optional: class name for class-based clients

# CellNet channel and topics for worker communication
WORKER_CHANNEL = "collab_worker"
WORKER_CALL_TOPIC = "call"
WORKER_READY_TOPIC = "ready"
WORKER_SHUTDOWN_TOPIC = "shutdown"

# Default timeout values
DEFAULT_READY_SIGNAL_TIMEOUT = 30.0
DEFAULT_SUBPROCESS_TIMEOUT = 300.0

# Global tracking writer for subprocess (set during worker startup)
_tracking_writer = None

# Global CollabClientAPI instance for Client API mode
_client_api = None


def get_client_api():
    """Get the CollabClientAPI instance for this worker.

    User's main() calls this to get the API for receive/send.
    Only available in Client API mode (when training module has main()).

    Returns:
        CollabClientAPI instance, or None if not in Client API mode.

    Example:
        from nvflare.collab.sys.worker import get_client_api

        def main():
            flare = get_client_api()
            flare.init()
            while flare.is_running():
                model = flare.receive()
                # ... train ...
                flare.send(result)
    """
    return _client_api


def get_tracking_writer():
    """Get the subprocess tracking writer.

    This can be called from user's training code to get a writer for logging
    metrics when running in subprocess mode.

    Returns:
        The tracking writer, or None if not in subprocess mode or not configured.

    Example:
        from nvflare.collab.sys.worker import get_tracking_writer

        writer = get_tracking_writer()
        if writer:
            writer.log_metric("loss", loss_value, step=epoch)
    """
    return _tracking_writer


class CollabWorker:
    """Internal worker that runs inside subprocess for distributed training.

    This class is used internally by the Collab framework. Users do not interact
    with it directly - they just write normal @collab.publish decorated functions.

    The worker:
    1. Reads connection info from environment variables
    2. Loads the user's training module
    3. Connects to parent CollabExecutor via CellNet
    4. Handles incoming calls and executes user's functions
    5. Returns results back to parent
    """

    def __init__(self, training_module_name: str):
        """Initialize CollabWorker.

        Args:
            training_module_name: Name of the user's training module (e.g., "train")
        """
        self.training_module_name = training_module_name
        self.logger = get_obj_logger(self)

        # Read connection info from environment (set by CollabExecutor)
        self.parent_url = os.environ.get(ENV_PARENT_URL)
        self.parent_fqcn = os.environ.get(ENV_PARENT_FQCN)
        self.site_name = os.environ.get(ENV_SITE_NAME, "unknown")
        self.worker_id = os.environ.get(ENV_WORKER_ID, "0")

        if not self.parent_url or not self.parent_fqcn:
            raise RuntimeError(
                f"Missing required environment variables. "
                f"This module should be launched by CollabExecutor, not directly. "
                f"Required: {ENV_PARENT_URL}, {ENV_PARENT_FQCN}"
            )

        # Read timeout configuration from environment
        timeout_str = os.environ.get(ENV_SUBPROCESS_TIMEOUT, str(DEFAULT_SUBPROCESS_TIMEOUT))
        try:
            self.subprocess_timeout = float(timeout_str)
        except ValueError:
            self.subprocess_timeout = DEFAULT_SUBPROCESS_TIMEOUT

        self.training_app = None
        self.cell: Optional[CoreCell] = None
        self._shutdown_event = threading.Event()
        self._ready = False

        # For DDP: only rank 0 communicates with parent
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))

    def _load_training_module(self):
        """Load the user's training module and wrap it for Collab calls.

        Supports two modes:
        1. Class-based clients (COLLAB_CLIENT_CLASS set): Instantiates the specified class
        2. Module-level functions: Uses ModuleWrapper to expose @collab.publish functions
        """
        self.logger.info(f"Loading training module: {self.training_module_name}")

        # Import the user's training module
        module = importlib.import_module(self.training_module_name)

        # Check if a specific client class is specified
        client_class_name = os.environ.get(ENV_CLIENT_CLASS)

        if client_class_name:
            # Class-based client: instantiate the specified class
            self.logger.info(f"Using client class: {client_class_name}")
            client_class = getattr(module, client_class_name, None)
            if client_class is None:
                raise ValueError(f"Client class '{client_class_name}' not found in {self.training_module_name}")
            self.training_app = client_class()
        else:
            # Module-level functions: use ModuleWrapper
            from nvflare.collab.api.module_wrapper import ModuleWrapper

            self.training_app = ModuleWrapper(module)

        self.logger.info("Training module loaded successfully")

    def _create_cell(self):
        """Create CellNet cell to connect to parent CollabExecutor."""
        # Worker cell FQCN must be a child of parent's FQCN for routing to work.
        # Parent FQCN is e.g. "site-1.app", so worker should be "site-1.app.worker.0"
        worker_fqcn = f"{self.parent_fqcn}.worker.{self.worker_id}"

        self.logger.info(f"Connecting to parent CollabExecutor at {self.parent_fqcn}...")
        self.logger.info(f"Worker FQCN: {worker_fqcn}")

        # Connect to parent using parent_url parameter
        self.cell = CoreCell(
            fqcn=worker_fqcn,
            root_url="",  # No root server for worker
            secure=False,
            credentials={},
            parent_url=self.parent_url,  # Connect to parent CollabExecutor
        )

    def _register_handlers(self):
        """Register CellNet message handlers."""
        # Handler for training calls from parent
        self.cell.register_request_cb(
            channel=WORKER_CHANNEL,
            topic=WORKER_CALL_TOPIC,
            cb=self._handle_call,
        )

        # Handler for shutdown signal
        self.cell.register_request_cb(
            channel=WORKER_CHANNEL,
            topic=WORKER_SHUTDOWN_TOPIC,
            cb=self._handle_shutdown,
        )

    def _handle_call(self, request: Message) -> Message:
        """Handle incoming training call from parent CollabExecutor."""
        try:
            payload = request.payload
            func_name = payload.get("func_name")
            args = payload.get("args", ())
            kwargs = payload.get("kwargs", {})

            self.logger.debug(f"Executing: {func_name}")

            # Get the method from training app
            method = getattr(self.training_app, func_name, None)
            if method is None:
                error = f"Method '{func_name}' not found in {self.training_module_name}"
                self.logger.error(error)
                return Message(payload={"error": error})

            # Execute the user's function (this runs the actual training)
            result = method(*args, **kwargs)

            self.logger.debug(f"Completed: {func_name}")
            return Message(payload={"result": result})

        except Exception as e:
            self.logger.exception(f"Error executing call: {e}")
            return Message(payload={"error": str(e)})

    def _handle_shutdown(self, request: Message) -> Message:
        """Handle shutdown signal from parent."""
        self.logger.info("Received shutdown signal")
        self._shutdown_event.set()
        return Message(payload={"status": "shutting_down"})

    def _signal_ready(self):
        """Signal to parent that worker is ready to receive calls."""
        self.logger.info("Signaling ready to parent...")

        ready_msg = Message(
            payload={
                "worker_id": self.worker_id,
                "site_name": self.site_name,
                "rank": self.rank,
                "world_size": self.world_size,
            }
        )

        try:
            reply = self.cell.send_request(
                channel=WORKER_CHANNEL,
                topic=WORKER_READY_TOPIC,
                target=self.parent_fqcn,
                request=ready_msg,
                timeout=self.subprocess_timeout,
            )
            if reply:
                self.logger.info("Ready - waiting for training calls")
                self._ready = True
            else:
                self.logger.warning("No acknowledgment from parent")
        except Exception as e:
            self.logger.error(f"Failed to signal ready: {e}")

    def _setup_tracking(self):
        """Set up tracking writer for metrics collection.

        Creates a SubprocessWriter that sends metrics to the parent
        CollabExecutor via CellNet. The executor then forwards them to
        the configured tracking receiver (MLflow, TensorBoard, etc.).
        """
        global _tracking_writer

        # Check if tracking is enabled (set by CollabExecutor)
        tracking_enabled = os.environ.get(ENV_TRACKING_TYPE)
        if not tracking_enabled:
            self.logger.debug("Tracking not enabled")
            return

        try:
            from nvflare.collab.tracking import SubprocessWriter

            writer = SubprocessWriter(
                cell=self.cell,
                parent_fqcn=self.parent_fqcn,
                timeout=self.subprocess_timeout,
                fire_and_forget=True,  # Don't block training for metric sends
            )
            _tracking_writer = writer
            self.logger.info("Subprocess tracking writer configured")
        except Exception as e:
            self.logger.warning(f"Failed to set up tracking: {e}")

    def start(self):
        """Start the worker.

        Supports two modes based on client type:
        1. Client API (CollabClientAPI): Run training script top-to-bottom
        2. Collab API (@collab.publish methods): Wait for RPC calls

        For Client API with DDP:
        - All ranks run the training script
        - Rank 0 also handles server communication via CellNet
        """
        self.logger.info(f"Starting Collab worker (rank={self.rank}, world_size={self.world_size})")

        # Check if using Client API mode (CollabClientAPI as client)
        client_class_name = os.environ.get(ENV_CLIENT_CLASS)
        use_client_api = client_class_name == "CollabClientAPI"

        if use_client_api:
            # Client API mode: run training script on all ranks
            self._start_client_api_mode()
        else:
            # Collab API mode: rank 0 waits for RPC, others wait for signal
            self._start_collab_api_mode()

    def _start_client_api_mode(self):
        """Start in Client API mode - all ranks run the training script.

        The training script uses receive/send pattern (no main() needed):
            flare.init()
            while flare.is_running():
                model = flare.receive()
                result = train(model)
                flare.send(result)
        """
        global _client_api

        self.logger.info("Client API mode: running training script")

        try:
            if self.rank == 0:
                # Rank 0: Set up CellNet first
                register_available_decomposers()

                # Create CollabClientAPI instance
                from nvflare.client.in_process.publish_api import CollabClientAPI

                _client_api = CollabClientAPI()
                self.training_app = _client_api

                # Create CellNet and register handlers
                self._create_cell()
                self._register_handlers()
                self.cell.start()

                time.sleep(0.5)

                # Set up tracking if configured
                self._setup_tracking()

                # Signal ready to parent
                self._signal_ready()

                self.logger.info("Rank 0: CellNet ready, running training script...")
            else:
                # Other ranks: create a local CollabClientAPI for DDP sync
                from nvflare.client.in_process.publish_api import CollabClientAPI

                _client_api = CollabClientAPI()
                self.logger.info(f"Rank {self.rank}: running training script (will sync with rank 0)")

            # IMPORTANT: When running with -m, this module may be pre-imported into sys.modules
            # before execution starts. The pre-imported version has _client_api=None.
            # We need to update the existing module's _client_api attribute so that when
            # client scripts import it, they get the correct instance.
            import sys

            worker_module_name = "nvflare.collab.sys.worker"
            worker_module = sys.modules.get(worker_module_name)
            if worker_module:
                # Update the existing module's _client_api attribute
                worker_module._client_api = _client_api
            else:
                # Module not in sys.modules yet - register __main__ as the worker module
                this_module = sys.modules.get("__main__")
                if this_module:
                    sys.modules[worker_module_name] = this_module

            # ALL ranks run the training script (executes top to bottom)
            self.logger.info(f"Running training module: {self.training_module_name}")
            import runpy

            runpy.run_module(self.training_module_name, run_name="__main__", alter_sys=True)

            self.logger.info("Training script completed")

        except Exception as e:
            self.logger.exception(f"Worker error: {e}")
            raise
        finally:
            self._cleanup()

    def _start_collab_api_mode(self):
        """Start in Collab API mode - rank 0 waits for RPC calls."""
        # Only rank 0 communicates with parent in DDP mode
        if self.rank != 0:
            self.logger.info(f"Rank {self.rank}: participating in DDP, rank 0 coordinates")
            self._wait_for_shutdown_signal()
            return

        try:
            # Register tensor decomposer for FOBS
            register_available_decomposers()

            # Load the user's training module
            self._load_training_module()

            # Create CellNet cell and connect
            self._create_cell()
            self._register_handlers()
            self.cell.start()

            # Give cell time to connect
            time.sleep(0.5)

            # Set up tracking if configured
            self._setup_tracking()

            # Signal ready to parent
            self._signal_ready()

            # Wait for work or shutdown
            self._shutdown_event.wait()

            self.logger.info("Worker shutting down")

        except Exception as e:
            self.logger.exception(f"Worker error: {e}")
            raise
        finally:
            self._cleanup()

    def _wait_for_shutdown_signal(self):
        """Wait for OS signal to shutdown (for non-rank-0 processes)."""
        shutdown = threading.Event()

        def handle_signal(signum, frame):
            shutdown.set()

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        shutdown.wait()

    def _cleanup(self):
        """Clean up resources."""
        if self.cell:
            try:
                self.cell.stop()
            except Exception as e:
                self.logger.warning(f"Error stopping cell: {e}")


def main():
    """Entry point for Collab worker subprocess.

    Usage (by CollabExecutor, not directly by users):
        torchrun --nproc_per_node=4 -m nvflare.collab.sys.worker my_training_module

    The training module name is passed as a command-line argument.
    Connection details are passed via environment variables.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m nvflare.collab.sys.worker <training_module>")
        print()
        print("This module is used internally by CollabExecutor.")
        print("Users should not run this directly.")
        sys.exit(1)

    training_module = sys.argv[1]

    # Configure logging
    log_level = os.environ.get("COLLAB_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and start worker
    worker = CollabWorker(training_module)
    worker.start()


if __name__ == "__main__":
    main()
