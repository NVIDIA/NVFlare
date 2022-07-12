from typing import Optional, List

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.shareable import ReservedHeaderKey, ReturnCode, Shareable, make_reply
from nvflare.apis.fl_constant import AdminCommandNames, FLContextKey, ServerCommandKey, ServerCommandNames
from nvflare.apis.fl_context import FLContext
from ..server.fed_server import FederatedServer
from ..server.server_engine import ServerEngine
from nvflare.private.fed.server.server_state import HotState


class SimulatorServerEngine(ServerEngine):

    def persist_components(self, fl_ctx: FLContext, completed: bool):
        pass


class SimulatorServer(FederatedServer):

    def __init__(self, project_name=None, min_num_clients=2, max_num_clients=10, wait_after_min_clients=10,
                 cmd_modules=None, heart_beat_timeout=600, handlers: Optional[List[FLComponent]] = None, args=None,
                 secure_train=False, enable_byoc=False, snapshot_persistor=None, overseer_agent=None):
        super().__init__(project_name, min_num_clients, max_num_clients, wait_after_min_clients, cmd_modules,
                         heart_beat_timeout, handlers, args, secure_train, enable_byoc, snapshot_persistor,
                         overseer_agent)

        self.server_state = HotState()

    def _process_task_request(self, client, fl_ctx, shared_fl_ctx: FLContext):
        fl_ctx.set_peer_context(shared_fl_ctx)
        server_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        taskname, task_id, shareable = server_runner.process_task_request(client, fl_ctx)

        return shareable, task_id, taskname

    def _submit_update(self, data, fl_ctx: FLContext):
        shared_fl_ctx = data.get_header(ServerCommandKey.PEER_FL_CONTEXT)
        client = data.get_header(ServerCommandKey.FL_CLIENT)
        fl_ctx.set_peer_context(shared_fl_ctx)
        contribution_task_name = data.get_header(ServerCommandKey.TASK_NAME)
        task_id = data.get_cookie(FLContextKey.TASK_ID)
        server_runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        server_runner.process_submission(client, contribution_task_name, task_id, data, fl_ctx)

    def _aux_communicate(self, fl_ctx, data, shared_fl_context, topic):
        try:
            with self.engine.lock:
                job_id = shared_fl_context.get_prop(FLContextKey.CURRENT_RUN)
                shared_fl_ctx = data.get_header(ServerCommandKey.PEER_FL_CONTEXT)
                topic = data.get_header(ServerCommandKey.TOPIC)
                shareable = data.get_header(ServerCommandKey.SHAREABLE)
                fl_ctx.set_peer_context(shared_fl_ctx)

                engine = fl_ctx.get_engine()
                reply = engine.dispatch(topic=topic, request=shareable, fl_ctx=fl_ctx)
        except BaseException:
            self.logger.info("Could not connect to server runner process - asked client to end the run")
            reply = make_reply(ReturnCode.COMMUNICATION_ERROR)

        return reply

    def _create_server_engine(self, args, snapshot_persistor):
        return SimulatorServerEngine(
            server=self, args=args, client_manager=self.client_manager, snapshot_persistor=snapshot_persistor
        )

    def deploy(self, args, grpc_args=None, secure_train=False):
        super(FederatedServer, self).deploy(args, grpc_args, secure_train)
