from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.signal import Signal
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.shareable import Shareable


class DummyController(Controller):
    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        pass

    def start_controller(self, fl_ctx: FLContext):
        pass

    def stop_controller(self, fl_ctx: FLContext):
        pass


class DummyExecutor(Executor):
    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ):
        pass


class DummyLoggingController(Controller):
    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        pass

    def start_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Starting the controller...")

    def stop_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Stopping the controller...")


class DummyLoggingExecutor(Executor):
    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ):
        pass

    def handle_event(self, event_type, fl_ctx):
        if event_type == EventType.START_RUN:
            self.log_info(fl_ctx, "Starting the executor...")
        elif event_type == EventType.END_RUN:
            self.log_info(fl_ctx, "Stopping the executor...")
