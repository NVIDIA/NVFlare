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
    → execute() scopes the module-level Client API to this site
    → registered training function calls receive(), trains, and calls send()
    → execute() returns the value supplied to send()

The training function runs in the site's FLARE process. External-process and
multi-rank execution are intentionally outside this initial contract.
"""

import threading
from typing import Any, Callable, Dict, Optional

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
    collab method, which runs the registered training function through its
    receive/send cycle.
    """

    def __init__(self):
        """Initialize CollabClientAPI."""
        super().__init__()
        self.logger = get_obj_logger(self)
        self._init_state()

    def _init_state(self):
        """Initialize state and locks. Called by __init__ and make_client_app."""
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

        # Lock for thread safety
        self._lock = threading.Lock()

        self._training_func = None  # User's training function
        self._result: Optional[FLModel] = None
        self._log_handler: Optional[Callable[..., Any]] = None

    def set_training_func(self, func):
        """Set the training function executed in the client site's process.

        This function is called by execute() to run the client's training logic.
        It should use receive() and send() to communicate with the server.

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

    def set_log_handler(self, handler: Optional[Callable[..., Any]]):
        """Set the site-runtime callback used to emit Client API analytics."""
        if handler is not None and not callable(handler):
            raise TypeError(f"log handler must be callable but got {type(handler)}")
        self._log_handler = handler

    def make_client_app(self, site_name: str):
        """Create a new CollabClientAPI instance for a client site.

        This method is called by the runtime to create a separate instance for
        each client, avoiding deep-copy issues with threading objects.

        Args:
            site_name: Name of the client site.
        Returns:
            A new CollabClientAPI instance.
        """
        from nvflare.collab.api.app import ClientApp

        # Create a fresh instance with independent state and a new lock.
        new_api = CollabClientAPI()

        # Set site name for this client
        new_api._sys_info["site_name"] = site_name

        # Preserve the registered training function.
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

        This method makes the model available to receive(), runs the registered
        training function, and returns the model supplied to send().

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
            return None

        self.logger.debug(f"Received task '{task_name}' from server")

        if not self._training_func:
            raise RuntimeError("CollabClientAPI requires a training function")

        self._current_model = fl_model
        self._current_task_name = task_name
        self._result = None

        # Scope module-level Client API calls to this embedded API instance.
        with use_api(self):
            self._training_func()

        self.logger.debug(f"Returning result to server for task '{task_name}'")
        return self._result

    @publish
    def stop(self):
        """Server calls this to signal job completion.

        This is an alternative to sending fl_model=None in execute().
        """
        self.logger.info("Received stop signal from server")
        self._stopped = True

    @publish
    def abort(self, reason: str = ""):
        """Server calls this to abort the job.

        Args:
            reason: The reason for aborting.
        """
        self.logger.error(f"Received abort signal from server: {reason}")
        self._aborted = True
        self._abort_reason = reason

    # =========================================================================
    # Client API implementation - Client code calls these
    # =========================================================================

    def init(self, rank: Optional[str] = None):
        """Initialize the Client API.

        Args:
            rank: Must be unset or rank zero; multi-rank execution is unsupported.
        """
        if rank not in (None, "0", 0):
            raise ValueError("CollabClientAPI supports only single-process client execution")
        self.logger.info("CollabClientAPI initialized")

    def receive(self, timeout: Optional[float] = None) -> Optional[FLModel]:
        """Receive model from server.

        Args:
            timeout: Accepted for Client API compatibility; calls do not block here.

        Returns:
            The FLModel from server, or None if job ended/aborted.
        Raises:
            RuntimeError: If job was aborted.
        """
        if self._aborted:
            raise RuntimeError(f"Job aborted: {self._abort_reason}")

        if self._stopped:
            return None

        self._receive_called = True
        self.logger.debug(f"Received model for task '{self._current_task_name}'")
        return self._current_model

    def send(self, model: FLModel, clear_cache: bool = True) -> None:
        """Send result back to server.

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

        self._result = model

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

        """
        return self._current_task_name or ""

    def is_train(self) -> bool:
        """Check if current task is a training task.

        Returns:
            True if current task is 'train'.

        """
        return self._current_task_name == "train"

    def is_evaluate(self) -> bool:
        """Check if current task is an evaluation task.

        Returns:
            True if current task is 'evaluate'.

        """
        return self._current_task_name == "evaluate"

    def is_submit_model(self) -> bool:
        """Check if current task is a submit_model task.

        Returns:
            True if current task is 'submit_model'.

        """
        return self._current_task_name == "submit_model"

    def log(self, key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
        """Log a metric.

        Args:
            key: The metric key.
            value: The metric value.
            data_type: The data type of the value.
            **kwargs: Additional arguments.
        """
        if self._log_handler is None:
            raise RuntimeError("CollabClientAPI metric logging is unavailable before the site runtime is initialized")
        return self._log_handler(key, value, data_type, **kwargs)

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
