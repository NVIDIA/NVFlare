from typing import List, Any

from nvflare.apis.fl_api.interfaces.comm_layer import CommunicationLayer
from nvflare.apis.fl_api.interfaces.strategy import Strategy
from nvflare.apis.fl_api.message.fl_message import MessageEnvelope


class FedXGBoost(Strategy):
    def __init__(self, num_rounds=10, **kwargs):
        super().__init__(**kwargs)
        self.num_rounds = num_rounds

    def coordinate(
        self,
        available_clients: List[str],
        global_state: Any,
        round_number: int,
        communicator: CommunicationLayer,
        **kwargs,
    ) -> Any:
        for r in range(self.num_rounds):
            # Broadcast the current global booster to all clients
            booster_msg = MessageEnvelope()
            booster_msg.payload = {"global_booster": global_state}
            communicator.broadcast_and_wait(available_clients, booster_msg)
            # Collect updates (e.g., booster weights, gradients, etc.) from all clients
            updates = communicator.receive_from_peers(available_clients)
            # Aggregate updates (this could be tree averaging, federated gradient boosting, etc.)
            result = self.aggregate(updates, round_number=round_number, **kwargs)
            global_state = result.payload["global_booster"]
        return global_state

    def aggregate(self, updates, round_number=0, **kwargs):
        # Type check
        for u in updates:
            if not isinstance(u, MessageEnvelope):
                raise TypeError(f"All updates must be MessageEnvelope, got {type(u)}")
        # Example: assume each update.payload contains 'booster_update' (could be booster weights, trees, etc.)
        booster_updates = [u.payload["booster_update"] for u in updates if u.payload and "booster_update" in u.payload]
        # Implement your aggregation logic here (e.g., federated tree averaging, secure aggregation, etc.)
        # For demo, just return the first booster_update
        agg_booster = booster_updates[0] if booster_updates else None
        result_msg = MessageEnvelope()
        result_msg.payload = {"global_booster": agg_booster}
        return result_msg

    def boost(self, model, updates):
        # Federated boosting logic
        return model  # Placeholder
