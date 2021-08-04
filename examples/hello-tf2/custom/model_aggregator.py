from typing import Tuple
import numpy as np

from nvflare.apis.aggregator import Aggregator
from nvflare.apis.fl_constant import FLConstants, ShareableKey, ShareableValue
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.utils.fl_ctx_sanity_check import server_fl_ctx_sanity_check


class AccumulateAggregator(Aggregator):
    def __init__(self, exclude_vars=None, aggregation_weights=None):
        super().__init__()
        self.accumulator = []

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> Tuple[bool, bool]:
        """Store shareable and update aggregator's internal state
        Args:
            shareable: information from client
            fl_ctx: context provided by workflow

        Returns:
            The first boolean indicates if this shareable is accepted.
            The second boolean indicates if aggregate can be called.
        """

        server_fl_ctx_sanity_check(fl_ctx)
        current_round = fl_ctx.get_prop(FLConstants.CURRENT_ROUND)
        shared_fl_context = fl_ctx.get_prop(FLConstants.PEER_CONTEXT)
        client_name = shared_fl_context.get_prop(FLConstants.CLIENT_NAME)
        contribution_round = shared_fl_context.get_prop(FLConstants.CURRENT_ROUND)

        accepted = False
        if contribution_round == current_round and not self._client_in_accumulator(client_name):
            self.accumulator.append(shared_fl_context)
            accepted = True

        return accepted, False

    def _client_in_accumulator(self, client_name):
        for item in self.accumulator:
            if client_name == item.get_prop(FLConstants.CLIENT_NAME):
                return True
        return False

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Called when workflow determines to generate shareable to send back to clients

        Args:
            fl_ctx (FLContext): context provided by workflow

        Returns:
            Shareable: the average of accepted shareables from clients
        """
        
        server_fl_ctx_sanity_check(fl_ctx)
        current_round = fl_ctx.get_prop(FLConstants.CURRENT_ROUND)
        self.logger.info("aggregating %s updates at round %s", len(self.accumulator), current_round)

        vars_to_aggregate = self.accumulator[0].get_prop(FLConstants.SHAREABLE)[ShareableKey.MODEL_WEIGHTS].keys()

        # perform average of clients' weights
        aggregated_model = {}
        for v_name in vars_to_aggregate:
            np_vars = []
            for client_ctx in self.accumulator:
                data = client_ctx.get_prop(FLConstants.SHAREABLE)[ShareableKey.MODEL_WEIGHTS]
                np_vars.append(data[v_name])

            new_val = np.sum(np_vars, axis=0) / len(self.accumulator)
            aggregated_model[v_name] = new_val

        self.accumulator.clear()

        shareable = Shareable()
        shareable[ShareableKey.TYPE] = ShareableValue.TYPE_WEIGHTS
        shareable[ShareableKey.DATA_TYPE] = ShareableValue.DATA_TYPE_UNENCRYPTED
        shareable[ShareableKey.MODEL_WEIGHTS] = aggregated_model
        return shareable

