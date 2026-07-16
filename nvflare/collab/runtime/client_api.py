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

"""CollabClientAPI: Implements Client API spec using Collab API.

This module provides a Client API implementation that bridges the receive/send
pattern to the Collab API's method-call pattern.

Architecture:
    Server calls execute() collab method with FLModel
    → execute() puts model in _call_queue
    → execute() blocks waiting on _result_queue

    Client calls receive()
    → receive() gets model from _call_queue
    → Client processes model
    → Client calls send(result)
    → send() puts result in _result_queue
    → execute() returns result to server

For DDP (multi-GPU):
    - Rank 0: handles all server communication
    - Other ranks: broadcast/sync with rank 0
"""

import os
import threading
from queue import Empty, Queue
from typing import Any, Dict, Optional

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.client.api import use_api
from nvflare.client.api_spec import APISpec
from nvflare.collab.api.decorators import publish
from nvflare.fuel.utils.log_utils import get_obj_logger


class CollabClientAPI(APISpec):
    """Client API implementation using Collab API.

    This class bridges the Client API's receive/send pattern to the
    Collab API's method-call pattern. The server calls the execute()
    collab method, which blocks until the client completes its
    recei ve/send cycle.
    """

    def __init__(self):
        """Initialize CollabClientAPI."""
        super().__init__()
        self.logger = get_obj_logger(self)
        self._init_queues()

    def _init_queues(self):
        """Initialize queues and locks. Called by __init__ and make_client_app."""
        # Queues for bridging collab calls to receive/send pattern (subprocess mode)
        self._call_queue: Queue = Queue()  # Server's call arrives here
        self._result_queue: Queue = Queue()  # Client's result goes here

        # State tracking
        self._stopped = False
        self._aborted = False
        self._abort_reason = ""
        self._current_model: Optional[FLModel] = None
        self._current_task_name: Optional[str] = None
        self._receive_called = False

        # System info (populated from server calls)
        self._sys_info: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}

        # DDP support
        self._rank = "0"
        self._current_round = 0  # Track round for non-rank-0 processes

        # Lock for thread safety
        self._lock = threading.Lock()

        # In-process mode support
        self._training_func = None  # User's training function
        self._inprocess = False  # Whether running in-process
        self._inprocess_result: Optional[FLModel] = None  # Result from send() in in-process mode

    def set_training_func(self, func):
        """Set the training function for in-process mode.

        In in-process mode, this function is called by execute() to run
        the client's training logic. The function should use receive()
        and send() to communicate with the server.

        Args:
            func: A callable that takes no arguments. The function should:
                  - Call flare.is_running() to check if training should continue
                  - Call flare.receive() to get model from server
                  - Perform training
                  - Call flare.send(result) to return results

        Example:
            import nvflare.client as flare

            def training_loop():
                while flare.is_running():
                    model = flare.receive()
                    if model is None:
                        break
                    # train...
                    flare.send(result)

            api = CollabClientAPI()
            api.set_training_func(training_loop)
        """
        self._training_func = func
        self._inprocess = True  # Enable in-process mode when training func is set

    def make_client_app(self, site_name: str, backend_type):
        """Create a new CollabClientAPI instance for a client site.

        This method is called by the simulator to create separate instances
        for each client, avoiding deep copy issues with threading objects.

        Args:
            site_name: Name of the client site.
            backend_type: The backend type (simulation, etc.)

        Returns:
            A new CollabClientAPI instance.
        """
        from nvflare.collab.api.app import ClientApp

        # Create a fresh instance with new queues/locks
        new_api = CollabClientAPI()

        # Set site name for this client
        new_api._sys_info["site_name"] = site_name

        # Preserve training function for in-process mode
        if self._training_func:
            new_api.set_training_func(self._training_func)

        return ClientApp(new_api)

    # =========================================================================
    # Collab methods - Server calls these
    # =========================================================================

    @publish
    def execute(
        self,
        fl_model: Optional[FLModel] = None,
        task_name: str = "train",
        job_id: str = "",
        site_name: str = "",
    ) -> Optional[FLModel]:
        """Server calls this to send a task to the client.

        This method bridges to the Client API's receive/send pattern:
        1. Puts the model in _call_queue (receive() gets it)
        2. Blocks waiting for client to call send()
        3. Returns the result from send() back to server

        Args:
            fl_model: The FLModel to send to client. None signals job end.
            task_name: Name of the task (train, evaluate, submit_model).
            job_id: The job ID.
            site_name: The site name.

        Returns:
            The FLModel result from client's send() call.
        """
        # Update system info (don't overwrite site_name if already set)
        with self._lock:
            if job_id:
                self._sys_info["job_id"] = job_id
            if site_name:
                self._sys_info["site_name"] = site_name

        # None model signals end of job
        if fl_model is None:
            self.logger.info("Received stop signal from server (fl_model=None)")
            self._stopped = True
            # Unblock any waiting receive()
            self._call_queue.put((None, None))
            return None

        self.logger.debug(f"Received task '{task_name}' from server")

        # In-process mode: call training function directly
        if self._inprocess and self._training_func:
            # Store model for receive() to return
            self._current_model = fl_model
            self._current_task_name = task_name
            self._inprocess_result = None

            # Scope the module-level Client API calls to this embedded API.
            with use_api(self):
                self._training_func()

            # Return what send() stored
            result = self._inprocess_result
            self.logger.debug(f"Returning result to server for task '{task_name}' (in-process)")
            return result

        # Subprocess mode: use queues
        # Put model where receive() can get it
        self._call_queue.put((fl_model, task_name))

        # Block until client calls send()
        result = self._result_queue.get()

        self.logger.debug(f"Returning result to server for task '{task_name}'")
        return result

    @publish
    def stop(self):
        """Server calls this to signal job completion.

        This is an alternative to sending fl_model=None in execute().
        """
        self.logger.info("Received stop signal from server")
        self._stopped = True

        # In subprocess mode: unblock any waiting receive()
        if not self._inprocess:
            self._call_queue.put((None, None))

    @publish
    def abort(self, reason: str = ""):
        """Server calls this to abort the job.

        Args:
            reason: The reason for aborting.
        """
        self.logger.error(f"Received abort signal from server: {reason}")
        self._aborted = True
        self._abort_reason = reason
        # Unblock any waiting receive()
        self._call_queue.put((None, None))

    # =========================================================================
    # Client API implementation - Client code calls these
    # =========================================================================

    def init(self, rank: Optional[str] = None):
        """Initialize the Client API.

        Args:
            rank: Local rank of the process (for multi-GPU DDP).
        """
        if rank is None:
            rank = os.environ.get("RANK", "0")
        self._rank = str(rank)
        self.logger.info(f"CollabClientAPI initialized with rank={self._rank}")

    def receive(self, timeout: Optional[float] = None) -> Optional[FLModel]:
        """Receive model from server.

        In subprocess mode:
            - Rank 0: Blocks until server calls execute() with a model.
            - Other ranks: Returns empty FLModel immediately (sync via checkpoint/barrier).
        In in-process mode: Returns the model that execute() stored.

        Args:
            timeout: Optional timeout in seconds (only for subprocess mode).

        Returns:
            The FLModel from server, or None if job ended/aborted.
            For non-rank-0 in subprocess mode, returns empty FLModel.

        Raises:
            RuntimeError: If job was aborted.
        """
        if self._aborted:
            raise RuntimeError(f"Job aborted: {self._abort_reason}")

        if self._stopped:
            return None

        # In-process mode: return what execute() stored
        if self._inprocess:
            fl_model = self._current_model
            self._receive_called = True
            self.logger.debug(f"Received model for task '{self._current_task_name}' (in-process)")
            return fl_model

        # Subprocess mode: non-rank-0 returns None immediately
        # They will sync with rank 0 via checkpoint file + dist.barrier()
        # This matches existing ExProcessClientAPI behavior
        if self._rank != "0":
            self._receive_called = True
            self.logger.debug(f"Rank {self._rank}: returning None (will sync with rank 0)")
            return None

        # Rank 0: get from queue (blocks until server sends)
        try:
            if timeout is not None:
                item = self._call_queue.get(timeout=timeout)
            else:
                item = self._call_queue.get()
        except Empty:
            return None

        fl_model, task_name = item

        # Check for stop/abort signals
        if fl_model is None:
            if self._aborted:
                raise RuntimeError(f"Job aborted: {self._abort_reason}")
            return None

        self._current_model = fl_model
        self._current_task_name = task_name
        self._receive_called = True
        # Track round for non-rank-0 processes
        self._current_round = getattr(fl_model, "current_round", 0)

        self.logger.debug(f"Received model for task '{task_name}'")
        return fl_model

    def send(self, model: FLModel, clear_cache: bool = True) -> None:
        """Send result back to server.

        In subprocess mode: Puts result in queue for execute() to return.
        In in-process mode: Stores result for execute() to return.

        Args:
            model: The FLModel result to send back.
            clear_cache: Whether to clear the cached model after sending.

        Raises:
            RuntimeError: If receive() was not called first.
        """
        if not self._receive_called:
            raise RuntimeError('"receive" must be called before "send"!')

        if model.params is None and model.metrics is None:
            raise RuntimeError("The model to send must have either params or metrics")

        self.logger.debug(f"Sending result for task '{self._current_task_name}'")

        # In-process mode: store result for execute() to return
        if self._inprocess:
            self._inprocess_result = model
        else:
            # Subprocess mode: put result in queue
            self._result_queue.put(model)

        if clear_cache:
            self._current_model = None
            self._current_task_name = None
            self._receive_called = False

    def is_running(self) -> bool:
        """Check if the FL system is still running.

        Returns True if server is still sending tasks.
        Returns False if server signaled stop or job ended.

        For DDP (subprocess mode):
            - Rank 0: Checks actual server status
            - Other ranks: Always returns True (sync stop via dist.broadcast)

        Returns:
            True if running, False otherwise.

        Raises:
            RuntimeError: If job was aborted.
        """
        if self._aborted:
            raise RuntimeError(f"Job aborted: {self._abort_reason}")

        if self._stopped:
            return False

        # Subprocess mode: non-rank-0 always returns True
        # The training code should sync stop condition from rank 0 via dist.broadcast
        if not self._inprocess and self._rank != "0":
            return True

        return True

    def system_info(self) -> Dict:
        """Get NVFlare system information.

        Returns:
            Dict with job_id and site_name.
        """
        return self._sys_info.copy()

    def get_config(self) -> Dict:
        """Get the client configuration.

        Returns:
            The configuration dictionary.
        """
        return self._config.copy()

    def get_job_id(self) -> str:
        """Get the current job ID.

        Returns:
            The job ID string.
        """
        return self._sys_info.get("job_id", "")

    def get_site_name(self) -> str:
        """Get the site name.

        Returns:
            The site name string.
        """
        return self._sys_info.get("site_name", "")

    def get_task_name(self) -> str:
        """Get the current task name.

        Returns:
            The task name (train, evaluate, submit_model).

        Raises:
            RuntimeError: If called from non-rank-0 process.
        """
        if self._rank != "0":
            raise RuntimeError("Only rank 0 can call get_task_name!")
        return self._current_task_name or ""

    def is_train(self) -> bool:
        """Check if current task is a training task.

        Returns:
            True if current task is 'train'.

        Raises:
            RuntimeError: If called from non-rank-0 process.
        """
        if self._rank != "0":
            raise RuntimeError("Only rank 0 can call is_train!")
        return self._current_task_name == "train"

    def is_evaluate(self) -> bool:
        """Check if current task is an evaluation task.

        Returns:
            True if current task is 'evaluate'.

        Raises:
            RuntimeError: If called from non-rank-0 process.
        """
        if self._rank != "0":
            raise RuntimeError("Only rank 0 can call is_evaluate!")
        return self._current_task_name == "evaluate"

    def is_submit_model(self) -> bool:
        """Check if current task is a submit_model task.

        Returns:
            True if current task is 'submit_model'.

        Raises:
            RuntimeError: If called from non-rank-0 process.
        """
        if self._rank != "0":
            raise RuntimeError("Only rank 0 can call is_submit_model!")
        return self._current_task_name == "submit_model"

    def log(self, key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
        """Log a metric.

        Args:
            key: The metric key.
            value: The metric value.
            data_type: The data type of the value.
            **kwargs: Additional arguments.
        """
        if self._rank != "0":
            raise RuntimeError("Only rank 0 can call log!")

        # TODO: Implement metric logging via collab call to server
        self.logger.debug(f"Log metric: {key}={value} (type={data_type})")

    def clear(self):
        """Clear the cached model."""
        self._current_model = None
        self._current_task_name = None
        self._receive_called = False

    def shutdown(self):
        """Shutdown the Client API.

        Releases resources and stops operation.
        """
        self.logger.info("Shutting down CollabClientAPI")
        self._stopped = True
        # Unblock any waiting receive()
        try:
            self._call_queue.put_nowait((None, None))
        except Exception:
            pass
