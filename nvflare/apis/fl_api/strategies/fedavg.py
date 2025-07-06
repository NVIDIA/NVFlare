from typing import List, Optional, Callable
from nvflare.apis.fl_api.interfaces.strategy import Strategy
from nvflare.apis.fl_api.aggregator.aggregators import Aggregator, FuncAggregatorWrapper, DefaultAggregator
from nvflare.apis.fl_api.interfaces.comm_layer import CommunicationLayer, MessageType


class FedAvg(Strategy):
    def __init__(
            self,
            aggregator: Optional[Aggregator] = None,
            aggregation_fn: Optional[Callable[[List[MessageType]], MessageType]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        if aggregator:
            self.aggregator = aggregator
        elif aggregation_fn:
            self.aggregator = FuncAggregatorWrapper(aggregation_fn)
        else:
            self.aggregator = DefaultAggregator()

    def average(self, updates: List[MessageType]) -> MessageType:
        """
        Aggregate client updates using the configured aggregator.
        """
        return self.aggregator.aggregate(updates)

    def coordinate(
            self,
            selected_clients: List[str],
            global_state: MessageType,
            round_number: int,
            communication: CommunicationLayer,
            **kwargs,
    ) -> MessageType:
        """
        Default synchronous coordination:
        - Broadcast global state
        - Collect updates
        - Aggregate them
        """
        # Directly broadcast the MessageType global_state
        # blocking call
        updates = communication.broadcast_and_wait(selected_clients, global_state)


        return self.average(updates)
