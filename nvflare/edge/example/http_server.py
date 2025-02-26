import base64
import threading
from typing import Dict, Any

import torch
from flask import Flask, jsonify, request
from torch import Tensor

from nvflare.edge.export.executorch_export import export_model
from nvflare.edge.example.model import Net, TrainingNet

app = Flask(__name__)

class MockFederatedServer:
    def __init__(self, min_clients: int = 1, learning_rate: float = 0.1):
        self.lock = threading.Lock()
        self.net = TrainingNet(Net())
        self.total_grad_dict: Dict[str, Dict[str, Tensor]] = {}
        self.received_result = 0
        self.min_clients = min_clients
        self.learning_rate = learning_rate

    def _tensor_from_json(self, tensor_data: Dict[str, Any]) -> Dict[str, Tensor]:
        """Convert JSON tensor data to PyTorch tensors."""
        grad_dict = {}
        for key, value in tensor_data.items():
            tensor = torch.Tensor(value["data"]).reshape(value["sizes"])
            grad_dict[key] = tensor
        print("get grad dict:", grad_dict)
        return grad_dict

    def _aggregate_gradients(self, client_weight: float = 1.0) -> Dict[str, Tensor]:
        """Aggregate gradients from all clients."""
        agg_grad_dict = {}
        for client_grads in self.total_grad_dict.values():
            for key, grad in client_grads.items():
                if key in agg_grad_dict:
                    agg_grad_dict[key] += grad * client_weight
                else:
                    agg_grad_dict[key] = grad * client_weight
         print("agg grad dict:", agg_grad_dict)
        return agg_grad_dict

    def _update_model(self, aggregated_grads: Dict[str, Tensor]) -> None:
        """Update model weights using aggregated gradients."""
        for key, param in self.net.state_dict().items():
            if key in aggregated_grads:
                self.net.state_dict()[key] -= self.learning_rate * aggregated_grads[key]

    def export_current_model(self) -> bytes:
        """Export current model in ExecutorTorch format."""
        print("model is", self.net.state_dict())
        input_tensor = torch.randn(1, 2)
        label_tensor = torch.ones(1, dtype=torch.int64)
        return export_model(self.net, input_tensor, label_tensor).buffer

    def process_gradients(self, client_id: str, grad_data: Dict[str, Any]) -> bool:
        """Process received gradients from a client."""
        print("Received tensor data:", grad_data)
        print("tensor data type:", type(grad_data))
        grad_dict = self._tensor_from_json(grad_data)
        
        with self.lock:
            self.total_grad_dict[client_id] = grad_dict
            self.received_result += 1

            if self.received_result >= self.min_clients:
                aggregated_grads = self._aggregate_gradients()
                self._update_model(aggregated_grads)
                self.received_result = 0
                self.total_grad_dict.clear()
                return True
        return False

# Initialize server
server = MockFederatedServer(min_clients=1)

@app.route("/task", methods=["GET"])
def get_model_weights():
    """Endpoint to get current model weights."""
    try:
        model_buffer = server.export_current_model()
        base64_encoded = base64.b64encode(model_buffer).decode("utf-8")
        return jsonify({"model": base64_encoded})
    except Exception as e:
        return jsonify({"error": f"Failed to export model: {str(e)}"}), 500

@app.route("/result", methods=["POST"])
def upload_model_grads():
    """Endpoint to receive gradients from clients."""
    try:
        tensor_data = request.get_json()
        if not tensor_data:
            return jsonify({"error": "No gradient data received"}), 400

        # TODO: Implement proper client ID management
        client_id = request.headers.get("Client-ID", "default_client")
        
        model_updated = server.process_gradients(client_id, tensor_data)
        
        return jsonify({
            "message": "Gradients processed successfully",
            "model_updated": model_updated
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process gradients: {str(e)}"}), 500

def main():
    """Main entry point for the server."""
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False  # Set to True for development
    )

if __name__ == "__main__":
    main()
