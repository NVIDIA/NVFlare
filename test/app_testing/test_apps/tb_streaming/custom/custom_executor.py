import logging
import random
import time

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal


class CustomExecutor(Executor):
    def __init__(self, task_name: str = "poc", streamer_id: str = "analytic_sender"):
        super().__init__()
        if not isinstance(task_name, str):
            raise TypeError("task name should be a string.")
        if not isinstance(streamer_id, str):
            raise TypeError("streamer_id should be a string.")
        self.task_name = task_name
        self.logger = logging.getLogger("POCExecutor")
        self.streamer_id = streamer_id

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        if task_name in self.task_name:
            peer_ctx = fl_ctx.get_prop(FLContextKey.PEER_CONTEXT)
            r = peer_ctx.get_prop("current_round")

            number = random.random()
            engine = fl_ctx.get_engine()

            # send analytics
            analytic_sender = engine.get_component(self.streamer_id)
            analytic_sender.write_scalar(fl_ctx, "random_number", number, global_step=r)
            analytic_sender.write_text(fl_ctx, "debug_msg", "Hello world", global_step=r)
            time.sleep(1.0)

            return shareable
        else:
            raise ValueError(f'No such supported task "{task_name}". Implemented task name is {self.task_name}')
