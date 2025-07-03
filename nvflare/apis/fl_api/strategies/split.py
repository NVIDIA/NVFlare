from typing import List, Any

from nvflare.apis.fl_api.interfaces.comm_layer import CommunicationLayer
from nvflare.apis.fl_api.interfaces.strategy import Strategy


class Split(Strategy):
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

