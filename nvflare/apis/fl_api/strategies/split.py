from typing import List, Any

from nvflare.apis.fl_api.interfaces.comm_layer import CommunicationLayer
from nvflare.apis.fl_api.interfaces.strategy import Strategy


class Split(Strategy):
    def coordinate(
            self,
            selected_clients: List[str],
            global_state: Any,
            round_number: int,
            communicator: CommunicationLayer,
            **kwargs,
    ) -> Any:
        client_id = selected_clients[0]

        # Send the global state (e.g., server-side model) to the client
        communicator.broadcast_and_wait([client_id], global_state)

        # Receive the activation from the client
        activation = communicator.collect_from_queue(client_id)
        server_output = self.forward_on_server(activation)

        # Send server output back to the client
        communicator.push_to_peers(
            sender_id="server",
            recipients=[client_id],
            message_type="server_output",
            payload=server_output,
        )

        # Receive gradients from the client
        gradients = communicator.collect_from_queue(client_id)
        return self.backward_on_server(gradients)

    def forward_on_server(self, activation):
        # Simulated forward pass on server side
        return activation

    def backward_on_server(self, gradients):
        # Simulated backward pass on server side
        return gradients
