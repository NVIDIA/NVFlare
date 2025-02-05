from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.signal import Signal
from nvflare.apis.controller_spec import Task
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.shareable import Shareable, make_reply, ReturnCode

class CustomController(Controller):

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):        
        # Prepare any extra parameters to send to the clients
        dxo = DXO(
            data_kind=DataKind.APP_DEFINED,
            data={"message": "howdy, I'm the controller"},
        )
        shareable = dxo.to_shareable()

        # Create the task with name "run_algorithm"
        task = Task(name="say_hello", data=shareable, result_received_cb=self._process_client_response)

        # Broadcast the task to all clients and wait for all to respond
        self.broadcast_and_wait(
            task=task,
            targets=None, # meaning all clients
            min_responses=0,
            fl_ctx=fl_ctx,
        )

        # log the results
        self.log_info(fl_ctx, f"Received results: {task}")
    
    def start_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Starting the controller...")

    def stop_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Stopping the controller...")

    def _process_client_response(self, client_task, fl_ctx: FLContext) -> None:
        task = client_task.task
        response = client_task.result
        client = client_task.client

        received_dxo = from_shareable(response).data["message"]

        self.log_info(fl_ctx, f"Processing {task.name} result from client {client.name}. response: {received_dxo}")

class CustomExecutor(Executor):

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ):
        if task_name == "say_hello":
            received_dxo = from_shareable(shareable)
            message = received_dxo.data["message"]
            self.log_info(fl_ctx, f"Received message: {message}")
            self.log_info(fl_ctx, "Sending reply")
            reply_dxo = DXO(
                data_kind=DataKind.APP_DEFINED,
                data={"message": "howdy, I'm the executor"},
            )
            shareable = reply_dxo.to_shareable()
            return shareable