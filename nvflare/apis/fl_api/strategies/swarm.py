from nvflare.apis.fl_api.communication.comm_layer import CommunicationLayer
from nvflare.apis.fl_api.strategies.base.agg_strategy import AggStrategy


class Swarm(AggStrategy):
    def coordinate(self, selected_clients, global_state, round_number, communication: CommunicationLayer, **kwargs):
        aggregator_node = selected_clients[round_number % len(selected_clients)]
        peer_ids = [c for c in selected_clients if c != aggregator_node]

        # Aggregator node sends model to peers
        communication.broadcast_state(peer_ids, global_state)

        updates = communication.collect_updates(peer_ids)
        aggregated = self.aggregator.aggregate(updates)
        return aggregated

