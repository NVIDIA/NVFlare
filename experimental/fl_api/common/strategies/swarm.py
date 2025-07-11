from typing import Any
from experimental.fl_api.common.strategies.fedavg import FedAvg


class Swarm(FedAvg):
    def coordinate(
        self,
        **kwargs,
    ) -> Any:
        aggregator_node = selected_clients[0]
        peer_ids = [c for c in selected_clients if c != aggregator_node]

        # Aggregator node sends model to peers using the new interface
        communicator.push_to_peers(
            sender_id=aggregator_node,
            recipients=peer_ids,
            message_type="global_state",
            payload=global_state,
        )

        # Peers send updates back to the aggregator node
        updates = communicator.receive_from_peers(peer_ids)
        # Aggregate using the new aggregator
        aggregated = self.aggregator.aggregate(updates)
        return aggregated
