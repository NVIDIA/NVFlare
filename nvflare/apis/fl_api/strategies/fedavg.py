from typing import List, Any, Optional, Callable
from nvflare.apis.fl_api.interfaces.strategy import Strategy
from nvflare.apis.fl_api.aggregator.aggregators import Aggregator, FuncAggregatorWrapper, DefaultAggregator
from nvflare.apis.fl_api.interfaces.comm_layer import CommunicationLayer


class FedAvg(Strategy):
    def __init__(
            self,
            aggregator: Optional[Aggregator] = None,
            aggregation_fn: Optional[Callable[[List[Any]], Any]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        if aggregator:
            self.aggregator = aggregator
        elif aggregation_fn:
            self.aggregator = FuncAggregatorWrapper(aggregation_fn)
        else:
            self.aggregator = DefaultAggregator()

    def coordinate(
            self,
            selected_clients: List[str],
            global_state: Any,
            round_number: int,
            communication: CommunicationLayer,
            **kwargs,
    ) -> Any:
        """
        Default synchronous coordination:
        - Broadcast global state
        - Collect updates
        - Aggregate them
        """
        state = {
            'round_number': round_number,
            'global_state': global_state,
        }
        communication.broadcast_to_queue(selected_clients, state)
        updates = communication.collect_from_queue(selected_clients)
        return self.aggregator.aggregate(updates)
