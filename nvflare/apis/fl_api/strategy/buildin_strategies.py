from typing import List, Any, Optional
from nvflare.apis.fl_api.strategy.strategy import Strategy
from nvflare.apis.fl_api.strategy.agg_strategy import AggStrategy
from nvflare.apis.fl_api.communication.comm_layer import CommunicationLayer

# 1. FedAvg
class FedAvgStrategy(AggStrategy):
    pass  # Uses default aggregation logic in AggStrategy


# 2. FedOpt (e.g., FedAdam, FedYogi)
class FedOptStrategy(AggStrategy):
    def __init__(self, optimizer, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = optimizer

    def aggregate(self, updates: List[Any], round_number: int) -> Any:
        gradients = self.aggregator.aggregate(updates)
        return self.optimizer.apply(gradients)


# 3. Cyclic
class CyclicStrategy(Strategy):
    def __init__(self, schedule: List[str], **kwargs):
        super().__init__(**kwargs)
        self.schedule = schedule

    def coordinate(
            self,
            selected_clients: List[str],
            global_state: Any,
            round_number: int,
            communication: CommunicationLayer,
            **kwargs,
    ) -> Any:
        current_client = self.schedule[round_number % len(self.schedule)]
        next_index = (round_number + 1) % len(self.schedule)
        next_client = self.schedule[next_index]

        # Send global_state to next client (not to current client)
        communication.broadcast_state([next_client], global_state, exclude=[current_client])

        # Receive updated state from next client
        update = communication.receive_state(next_client)
        return update


# 4. Split Learning
class SplitLearningStrategy(Strategy):
    def coordinate(
            self,
            selected_clients: List[str],
            global_state: Any,
            round_number: int,
            communication: CommunicationLayer,
            **kwargs,
    ) -> Any:
        client_id = selected_clients[0]

        communication.broadcast_state([client_id], global_state)

        activation = communication.receive_state(client_id)
        server_output = self.forward_on_server(activation)
        # Send server output back to client peers
        communication.send_message_to_peers(
            sender_id="server",
            recipient_ids=[client_id],
            message_type="server_output",
            payload=server_output,
        )
        gradients = communication.receive_state(client_id)
        return self.backward_on_server(gradients)

    def forward_on_server(self, activation):
        # Simulated forward pass on server side
        return activation

    def backward_on_server(self, gradients):
        # Simulated backward pass on server side
        return gradients


# 5. KMeans Strategy (Federated Clustering)
class FedKMeansStrategy(AggStrategy):
    def aggregate(self, updates: List[Any], round_number: int) -> Any:
        # KMeans centroid averaging logic
        return self.aggregator.aggregate(updates)  # Assume weighted centroid update


# 6. XGBoost Strategy
class FedXGBoostStrategy(Strategy):
    def __init__(self, num_rounds=10, **kwargs):
        super().__init__(**kwargs)
        self.num_rounds = num_rounds

    def coordinate(
            self,
            selected_clients: List[str],
            global_state: Any,
            round_number: int,
            communication: CommunicationLayer,
            **kwargs,
    ) -> Any:
        for r in range(self.num_rounds):
            communication.broadcast_state(selected_clients, global_state)
            updates = communication.collect_updates(selected_clients)
            global_state = self.boost(global_state, updates)
        return global_state

    def boost(self, model, updates):
        # Federated boosting logic
        return model  # Placeholder


# 7. Swarm Learning

# 7. Swarm Learning
class SwarmStrategy(AggStrategy):
    def coordinate(self, selected_clients, global_state, round_number, communication: CommunicationLayer, **kwargs):
        aggregator_node = selected_clients[round_number % len(selected_clients)]
        peer_ids = [c for c in selected_clients if c != aggregator_node]

        # Aggregator node sends model to peers
        communication.broadcast_state(peer_ids, global_state)

        updates = communication.collect_updates(peer_ids)
        aggregated = self.aggregator.aggregate(updates)
        return aggregated


# 8. Federated Statistics (e.g., sum, count, histogram)
class FedStatisticsStrategy(AggStrategy):
    def aggregate(self, updates: List[Any], round_number: int) -> Any:
        total = {}
        for update in updates:
            for k, v in update.items():
                total[k] = total.get(k, 0) + v
        return total
