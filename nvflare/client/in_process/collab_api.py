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
from nvflare.client.api_spec import APISpec
from nvflare.collab.api.dec import collab
from nvflare.fuel.utils.log_utils import get_obj_logger


class CollabClientAPI(APISpec):
    """Client API implementation using Collab API.

    This class bridges the Client API's receive/send pattern to the
    Collab API's method-call pattern. The server calls the execute()
    collab method, which blocks until the client completes its
    receive/send cycle.
    """

    def __init__(self):
        """Initialize CollabClientAPI."""
        self.logger = get_obj_logger(self)
        self._init_queues()

    def _init_queues(self):
        """Initialize queues and locks. Called by __init__ and make_client_app."""
        # Queues for bridging collab calls to receive/send pattern
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

        # Lock for thread safety
        self._lock = threading.Lock()

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
        return ClientApp(new_api)

    # =========================================================================
    # Collab methods - Server calls these
    # =========================================================================

    @collab
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
        # Update system info
        with self._lock:
            self._sys_info["job_id"] = job_id
            self._sys_info["site_name"] = site_name

        # None model signals end of job
        if fl_model is None:
            self.logger.info("Received stop signal from server (fl_model=None)")
            self._stopped = True
            # Unblock any waiting receive()
            self._call_queue.put((None, None))
            return None

        self.logger.debug(f"Received task '{task_name}' from server")

        # Put model where receive() can get it
        self._call_queue.put((fl_model, task_name))

        # Block until client calls send()
        result = self._result_queue.get()

        self.logger.debug(f"Returning result to server for task '{task_name}'")
        return result

    @collab
    def stop(self):
        """Server calls this to signal job completion.

        This is an alternative to sending fl_model=None in execute().
        """
        self.logger.info("Received stop signal from server")
        self._stopped = True
        # Unblock any waiting receive()
        self._call_queue.put((None, None))

    @collab
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

        Blocks until server calls execute() with a model.

        Args:
            timeout: Optional timeout in seconds.

        Returns:
            The FLModel from server, or None if job ended/aborted.

        Raises:
            RuntimeError: If job was aborted.
        """
        if self._aborted:
            raise RuntimeError(f"Job aborted: {self._abort_reason}")

        if self._stopped:
            return None

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

        self.logger.debug(f"Received model for task '{task_name}'")
        return fl_model

    def send(self, model: FLModel, clear_cache: bool = True) -> None:
        """Send result back to server.

        Completes the execute() collab call, returning the result to server.

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

        # Put result where execute() can return it
        self._result_queue.put(model)

        if clear_cache:
            self._current_model = None
            self._current_task_name = None
            self._receive_called = False

    def is_running(self) -> bool:
        """Check if the FL system is still running.

        Returns True if server is still sending tasks.
        Returns False if server signaled stop or job ended.

        Returns:
            True if running, False otherwise.

        Raises:
            RuntimeError: If job was aborted.
        """
        if self._aborted:
            raise RuntimeError(f"Job aborted: {self._abort_reason}")

        if self._stopped:
            return False

        # Try to peek at the queue with a short timeout
        # This mimics the ex_process behavior of trying to receive
        if self._current_model is not None:
            # We already have a model waiting to be processed
            return True

        try:
            # Non-blocking check - just see if we're stopped
            # The actual blocking happens in receive()
            return not self._stopped
        except Exception:
            return False

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
