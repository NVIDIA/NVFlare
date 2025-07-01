from typing import List, Any, Optional, Callable
from nvflare.apis.fl_api.strategy.strategy import Strategy
from nvflare.apis.fl_api.aggregator.aggregators import Aggregator, FuncAggregatorWrapper, DefaultAggregator
from nvflare.apis.fl_api.communication.comm_layer import CommunicationLayer


class AggStrategy(Strategy):
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
        communication.broadcast_state(selected_clients, global_state)

        updates = communication.collect_updates(selected_clients)
        return self.aggregator.aggregate(updates)

    async def coordinate_async(
            self,
            selected_clients: List[str],
            global_state: Any,
            round_number: int,
            communication: CommunicationLayer,
            **kwargs,
    ) -> Any:
        """
        Optional async coordination:
        - Broadcast global state
        - Receive updates as they come in
        - Perform incremental aggregation
        """
        communication.broadcast_state(selected_clients, global_state)

        if not hasattr(self.aggregator, "incremental_aggregate"):
            raise NotImplementedError("Aggregator does not support incremental aggregation")

        async for client_id, update in communication.receive_updates_async(selected_clients):
            global_state = self.aggregator.incremental_aggregate(global_state, update)

        return global_state
