import numpy as np

from nvflare.apis.fl_constant import FLConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, ShareableKey, ShareableValue
from nvflare.apis.trainer import Trainer

class SimpleTrainer(Trainer):

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal) -> Shareable:
        """
        This function is an extended function from the super class.
        As a supervised learning based trainer, the train function will run
        evaluate and train engines based on model weights from `shareable`.
        After the training is done, a new `Shareable` object will be submitted
        to server for aggregation.

        Args:
            shareable: the `Shareable` object received from server.
            fl_ctx: the `FLContext` object achieved from server.
            abort_signal: if triggered, the training will be aborted.

        Returns:
            a new `Shareable` object to be submitted to server for aggregation.
        """

        # retrieve model weights download from server's shareable
        fib = shareable[ShareableKey.MODEL_WEIGHTS]
        terms = fib["sequence"]

        # perform one step of Fibonacci 
        new_fib = {"sequence": np.array([terms[1], terms[0] + terms[1]])}

        # add -1 and +1 bias for site-1 and site-2 respectively
        client_name = fl_ctx.get_prop(FLConstants.CLIENT_NAME)
        if client_name == "site-1":
            new_fib["sequence"] -= 1
        elif client_name == "site-2":
            new_fib["sequence"] += 1

        self.logger.info(f"{client_name} Shareable from server: {fib}")
        self.logger.info(f"{client_name} New Shareable to server: {new_fib} \n")

        # send updated weights back to server through new shareable
        shareable = Shareable()
        shareable[ShareableKey.META] = {FLConstants.NUM_STEPS_CURRENT_ROUND: 1}
        shareable[ShareableKey.TYPE] = ShareableValue.TYPE_WEIGHTS
        shareable[ShareableKey.DATA_TYPE] = ShareableValue.DATA_TYPE_UNENCRYPTED
        shareable[ShareableKey.MODEL_WEIGHTS] = new_fib

        return shareable
