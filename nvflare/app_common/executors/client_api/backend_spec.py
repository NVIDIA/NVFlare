# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Backend spec for the Client API execution modes (V1-internal).

Design: docs/design/client_api_execution_modes.md ("Overview", "Execution Modes",
"Client API Backends"). One ClientAPIExecutor delegates to one mode-specific backend:

- in_process: trainer runs inside the Client Job (CJ) process over DataBus (EX-3)
- external_process: NVFlare launches and owns the trainer process tree over Cell (EP-4)
- attach: an externally owned trainer attaches over Cell (AT-2)

This module is internal to NVFlare. It is not a user extension point; users configure
``ClientAPIExecutor(execution_mode=...)`` only.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

if TYPE_CHECKING:
    # Import for typing only; avoids a runtime import cycle
    # (client_api_executor imports this module to build the context).
    from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor


@dataclass(frozen=True)
class ClientAPIBackendContext:
    """Immutable config a ClientAPIExecutor hands to its backend at ``initialize()``.

    Rationale: the executor's frozen constructor args live in private attributes and the backend
    factories are zero-arg, so a backend previously had no clean way to read the heartbeat/timeout/
    converter/task-name config it needs, nor a supported reference back to the executor's analytics
    hook. This frozen snapshot is that supported channel - a backend reads its config from here
    rather than reaching into ``ClientAPIExecutor`` private attributes.

    The fields mirror the frozen ``ClientAPIExecutor`` constructor surface one-to-one. ``executor``
    is a back-reference so a backend can:

    - call ``executor.fire_log_analytics(fl_ctx, dxo)`` for every trainer LOG message (the single
      LOG-to-analytics ownership point; see design "Configuration Surface"), and
    - select the federation-scoped analytics path when appropriate by setting
      ``executor._analytics_fire_fed_event = True`` in ``initialize()`` (Cell backends do this when
      no ConvertToFedEvent widget is configured), and
    - use the executor's FLComponent logging helpers.
    """

    executor: "ClientAPIExecutor"
    execution_mode: str
    # in_process entry point
    task_script_path: Optional[str] = None
    task_script_args: str = ""
    # external_process launch
    command: Optional[str] = None
    launch_once: bool = True
    launch_timeout: Optional[float] = None
    shutdown_timeout: Optional[float] = None
    stop_grace_period: float = 30.0
    # session / protocol (out-of-process)
    heartbeat_interval: float = 5.0
    heartbeat_timeout: float = 30.0
    task_wait_timeout: Optional[float] = None
    result_wait_timeout: Optional[float] = None
    # NOTE: params_exchange_format / params_transfer_type / server_expected_format and the
    # from/to_nvflare_converter ids are intentionally NOT here. Per FLARE-2698, param
    # conversion between the framework-agnostic aggregation representation (numpy) and the
    # framework-native training representation moves out of the executor to send/receive
    # filters at the client edge; the Client API boundary is pass-through. Transfer type
    # (FULL/DIFF) stays a Client API concern (model_registry), decided separately.
    # task-name / rank contract (all modes)
    train_task_name: str = "train"
    evaluate_task_name: str = "validate"
    submit_model_task_name: str = "submit_model"
    train_with_evaluation: bool = False
    # memory management (all modes)
    memory_gc_rounds: int = 0
    cuda_empty_cache: bool = False
    # attach
    attach_timeout: Optional[float] = None
    allow_reconnect: bool = False


class ClientAPIBackendSpec(ABC):
    """The narrow lifecycle contract that ClientAPIExecutor drives on its backend.

    Lifecycle ownership per execution mode:

    - in_process: the backend runs the trainer inside the CJ process and owns its thread.
    - external_process: the backend launches and owns the external trainer process tree; it must
      not stop the trainer before the payload transfer of a pending result reaches terminal state.
    - attach: the external system owns the trainer process; the backend owns only the attach
      session, token validation, and heartbeat lease.
    """

    @abstractmethod
    def initialize(self, context: ClientAPIBackendContext, fl_ctx: FLContext) -> None:
        """Prepares the backend for the run. Called once when the executor handles START_RUN.

        ``context`` is the frozen snapshot of the executor's configuration plus a back-reference to
        the executor (for ``fire_log_analytics`` and logging). A backend should read all of its
        config from ``context`` rather than from executor private attributes, and should retain what
        it needs for ``execute``/``handle_event``/``finalize``.

        The backend sets up its control plane here (DataBus wiring for in_process; Cell session
        machinery, bootstrap config, and - per launch_once policy - trainer launch for
        external_process; attach listener/token for attach).

        Contract: raise an exception on any setup failure. The executor converts the exception
        into system_panic so the job fails cleanly instead of hanging while tasks wait on a
        backend that never became ready.

        Cleanup-on-failure contract: ``initialize()`` must be exception-safe and self-unwinding - if
        it raises, it must first release any partial setup it already made (threads started,
        processes launched, listeners/tokens registered, files written). The executor does NOT call
        ``finalize()`` on a backend whose ``initialize()`` raised, because ``finalize()`` cannot
        assume a consistently half-initialized backend. Own your own rollback.

        Args:
            context: the frozen backend configuration and executor back-reference.
            fl_ctx: the FLContext of the START_RUN event.
        """
        pass

    @abstractmethod
    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Executes one task on the trainer and returns its result.

        The backend delivers the task to the trainer (TASK_READY over Cell, or DataBus for
        in_process), waits for the result within the executor-configured task/result bounds, and
        returns the result Shareable.

        Contract: this method must always return a Shareable and must not hang past abort:
        when abort_signal is triggered, the backend notifies/stops the trainer per its mode's
        lifecycle ownership and returns ``make_reply(ReturnCode.TASK_ABORTED)``. On failure it
        should return an error reply (e.g. ReturnCode.EXECUTION_EXCEPTION) rather than raise;
        exceptions that do escape are converted to EXECUTION_EXCEPTION replies by the executor,
        except UnsafeJobError which the executor lets propagate so ClientRunner can apply its
        dedicated UNSAFE_JOB handling.

        Args:
            task_name: name of the task.
            shareable: the task data.
            fl_ctx: the FLContext of the task.
            abort_signal: checked during execution; triggered means the task is aborted.

        Returns:
            The result Shareable (an error reply on failure/abort - never None).
        """
        pass

    @abstractmethod
    def handle_event(self, event_type: str, fl_ctx: FLContext) -> None:
        """Handles an FL event relayed by the executor.

        The executor relays events other than START_RUN/END_RUN (those are mapped to
        initialize/finalize). Backends use this for mode-specific bookkeeping.

        Contract: must not raise; log and continue on internal errors.

        Args:
            event_type: the fired event type.
            fl_ctx: the FLContext of the event.
        """
        pass

    @abstractmethod
    def finalize(self, fl_ctx: FLContext) -> None:
        """Releases backend resources. Called when the executor handles END_RUN.

        The backend tears down per its mode's lifecycle ownership: stop the in-process trainer
        thread; send SHUTDOWN and stop the owned process tree (honoring the executor's
        shutdown_timeout and stop_grace_period, and pending payload terminal state) for
        external_process; close the session lease (without killing the trainer) for attach.

        Contract: must be idempotent and must not raise. Not called if ``initialize()`` raised
        (see the cleanup-on-failure contract on ``initialize()``).

        Args:
            fl_ctx: the FLContext of the END_RUN event.
        """
        pass
