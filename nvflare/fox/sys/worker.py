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

"""FoxWorker: Transparent subprocess worker for distributed training.

This module is the internal entry point that FoxExecutor uses when spawning
subprocess-based training (e.g., with torchrun). Users never interact with
this directly - they just write normal @fox.collab decorated functions.

Architecture:
    User writes:        train.py with @fox.collab decorated functions

    FoxExecutor runs:   torchrun --nproc_per_node=4 -m nvflare.fox.sys.worker train
                                                    ↑ this module        ↑ user's module

    Environment vars (set by FoxExecutor, invisible to user):
        FOX_PARENT_URL    = grpc://localhost:8002
        FOX_PARENT_FQCN   = site-1.job123
        FOX_SITE_NAME     = site-1

Example:
    # User's train.py - completely normal, no worker code needed!
    from nvflare.fox import fox

    @fox.collab
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

from nvflare.fuel.f3.cellnet.core_cell import CoreCell
from nvflare.fuel.f3.message import Message
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.import_utils import optional_import
from nvflare.fuel.utils.log_utils import get_obj_logger

# Environment variable names (set by FoxExecutor)
ENV_PARENT_URL = "FOX_PARENT_URL"
ENV_PARENT_FQCN = "FOX_PARENT_FQCN"
ENV_SITE_NAME = "FOX_SITE_NAME"
ENV_WORKER_ID = "FOX_WORKER_ID"
ENV_SUBPROCESS_TIMEOUT = "FOX_SUBPROCESS_TIMEOUT"
ENV_TRACKING_TYPE = "FOX_TRACKING_TYPE"

# CellNet channel and topics for worker communication
WORKER_CHANNEL = "fox_worker"
WORKER_CALL_TOPIC = "call"
WORKER_READY_TOPIC = "ready"
WORKER_SHUTDOWN_TOPIC = "shutdown"

# Default timeout values
DEFAULT_READY_SIGNAL_TIMEOUT = 30.0
DEFAULT_SUBPROCESS_TIMEOUT = 300.0

# Global tracking writer for subprocess (set during worker startup)
_tracking_writer = None


def get_tracking_writer():
    """Get the subprocess tracking writer.

    This can be called from user's training code to get a writer for logging
    metrics when running in subprocess mode.

    Returns:
        The tracking writer, or None if not in subprocess mode or not configured.

    Example:
        from nvflare.fox.sys.worker import get_tracking_writer

        writer = get_tracking_writer()
        if writer:
            writer.log_metric("loss", loss_value, step=epoch)
    """
    return _tracking_writer


def _register_tensor_decomposer():
    """Register PyTorch TensorDecomposer for FOBS serialization."""
    tensor_decomposer, ok = optional_import(module="nvflare.app_opt.pt.decomposers", name="TensorDecomposer")
    if ok:
        fobs.register(tensor_decomposer)


class FoxWorker:
    """Internal worker that runs inside subprocess for distributed training.

    This class is used internally by the Fox framework. Users do not interact
    with it directly - they just write normal @fox.collab decorated functions.

    The worker:
    1. Reads connection info from environment variables
    2. Loads the user's training module
    3. Connects to parent FoxExecutor via CellNet
    4. Handles incoming calls and executes user's functions
    5. Returns results back to parent
    """

    def __init__(self, training_module_name: str):
        """Initialize FoxWorker.

        Args:
            training_module_name: Name of the user's training module (e.g., "train")
        """
        self.training_module_name = training_module_name
        self.logger = get_obj_logger(self)

        # Read connection info from environment (set by FoxExecutor)
        self.parent_url = os.environ.get(ENV_PARENT_URL)
        self.parent_fqcn = os.environ.get(ENV_PARENT_FQCN)
        self.site_name = os.environ.get(ENV_SITE_NAME, "unknown")
        self.worker_id = os.environ.get(ENV_WORKER_ID, "0")

        if not self.parent_url or not self.parent_fqcn:
            raise RuntimeError(
                f"Missing required environment variables. "
                f"This module should be launched by FoxExecutor, not directly. "
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
        """Load the user's training module and wrap it for Fox calls."""
        self.logger.info(f"Loading training module: {self.training_module_name}")

        # Import ModuleWrapper here to avoid circular imports
        from nvflare.fox.api.module_wrapper import ModuleWrapper

        # Import the user's training module
        module = importlib.import_module(self.training_module_name)

        # Wrap it with ModuleWrapper to expose @fox.collab methods
        self.training_app = ModuleWrapper(module)
        self.logger.info("Training module loaded successfully")

    def _create_cell(self):
        """Create CellNet cell to connect to parent FoxExecutor."""
        # Worker cell FQCN: site-1.worker.0
        worker_fqcn = f"{self.site_name}.worker.{self.worker_id}"

        self.logger.info("Connecting to parent FoxExecutor...")

        # Connect to parent using parent_url parameter
        self.cell = CoreCell(
            fqcn=worker_fqcn,
            root_url="",  # No root server for worker
            secure=False,
            credentials={},
            parent_url=self.parent_url,  # Connect to parent FoxExecutor
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
        """Handle incoming training call from parent FoxExecutor."""
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
        FoxExecutor via CellNet. The executor then forwards them to
        the configured tracking receiver (MLflow, TensorBoard, etc.).
        """
        global _tracking_writer

        # Check if tracking is enabled (set by FoxExecutor)
        tracking_enabled = os.environ.get(ENV_TRACKING_TYPE)
        if not tracking_enabled:
            self.logger.debug("Tracking not enabled")
            return

        try:
            from nvflare.fox.tracking import SubprocessWriter

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
        """Start the worker and wait for calls from parent."""
        self.logger.info(f"Starting Fox worker (rank={self.rank}, world_size={self.world_size})")

        # Only rank 0 communicates with parent in DDP mode
        if self.rank != 0:
            self.logger.info(f"Rank {self.rank}: participating in DDP, rank 0 coordinates")
            self._wait_for_shutdown_signal()
            return

        try:
            # Register tensor decomposer for FOBS
            _register_tensor_decomposer()

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
    """Entry point for Fox worker subprocess.

    Usage (by FoxExecutor, not directly by users):
        torchrun --nproc_per_node=4 -m nvflare.fox.sys.worker my_training_module

    The training module name is passed as a command-line argument.
    Connection details are passed via environment variables.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m nvflare.fox.sys.worker <training_module>")
        print()
        print("This module is used internally by FoxExecutor.")
        print("Users should not run this directly.")
        sys.exit(1)

    training_module = sys.argv[1]

    # Configure logging
    log_level = os.environ.get("FOX_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and start worker
    worker = FoxWorker(training_module)
    worker.start()


if __name__ == "__main__":
    main()
