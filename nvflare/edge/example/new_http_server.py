import base64
import threading
from typing import Any, Dict

import torch
from flask import Flask, jsonify, request
from torch import Tensor

from executorch_export import export_model
from model import Net, TrainingNet


def generate_endless_jobs(server):
    job_id = 0
    while True:
        session_id = f"session_{job_id}"
        job = {
            "status": "OK",
            "session_id": session_id,
            "job_id": f"job_{job_id}",
            "job_name": "training_job",
            "job_data": {
                "total_epochs": server.local_epochs,
                "batch_size": server.batch_size,
                "learning_rate": server.learning_rate,
            },
        }
        print(f"add job into server with {session_id=}")
        server.jobs[session_id] = job
        yield job
        job_id += 1


class MockFederatedServer:
    def __init__(
        self, min_clients: int = 1, learning_rate: float = 0.1, total_rounds=1
    ):
        self.lock = threading.Lock()
        self.net = TrainingNet(Net())
        self.total_grad_dict: Dict[str, Dict[str, Tensor]] = {}
        self.received_result = 0
        self.min_clients = min_clients

        self.batch_size = 1
        self.learning_rate = learning_rate
        self.local_epochs = 1

        self.total_rounds = 1
        self.current_round = 0
        self.jobs = {}

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

    def process_gradients(self, session_id: str, grad_data: Dict[str, Any]) -> bool:
        """Process received gradients from a client."""
        print("Received tensor data:", grad_data)
        print("tensor data type:", type(grad_data))
        grad_dict = self._tensor_from_json(grad_data)

        with self.lock:
            self.total_grad_dict[session_id] = grad_dict
            self.received_result += 1

            if self.received_result >= self.min_clients:
                aggregated_grads = self._aggregate_gradients()
                self._update_model(aggregated_grads)
                self.received_result = 0
                self.total_grad_dict.clear()
                self.current_round += 1
                if self.current_round >= self.total_rounds:
                    self.jobs[session_id]["training_completed"] = True


server = MockFederatedServer(min_clients=1)
app = Flask(__name__)
job_generator = generate_endless_jobs(server)


@app.route("/job", methods=["POST"])
def get_job():
    """Endpoint to get available job."""
    try:
        device_id = request.headers.get("X-Flare-Device-ID")
        if not device_id:
            return jsonify({"status": "error", "message": "Missing device ID"}), 400

        job = next(job_generator)

        return (
            jsonify(job),
            200,
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/task", methods=["GET"])
def get_task():
    """Endpoint to get task (model weights)."""
    try:
        session_id = request.args.get("session_id")
        job_id = request.args.get("job_id")

        if not session_id or not job_id:
            return (
                jsonify({"status": "error", "message": "Missing session_id or job_id"}),
                400,
            )

        if server.jobs[session_id].get("training_completed"):
            return (
                jsonify({"status": "FINISHED", "message": "Training is finished"}),
                200,
            )

        model_buffer = server.export_current_model()
        base64_encoded = base64.b64encode(model_buffer).decode("utf-8")

        return (
            jsonify(
                {
                    "status": "OK",
                    "task_id": f"task_{server.current_round}",
                    "task_name": "train",
                    "task_data": {"model": base64_encoded},
                }
            ),
            200,
        )
    except Exception as e:
        return (
            jsonify(
                {"status": "error", "message": f"Failed to export model: {str(e)}"}
            ),
            500,
        )


@app.route("/result", methods=["POST"])
def submit_result():
    """Endpoint to receive training results."""
    try:
        session_id = request.args.get("session_id")
        task_id = request.args.get("task_id")
        task_name = request.args.get("task_name")

        if not all([session_id, task_id, task_name]):
            return (
                jsonify({"status": "error", "message": "Missing required parameters"}),
                400,
            )

        result_data = request.get_json()
        if not result_data or not result_data.get("result"):
            return (
                jsonify({"status": "error", "message": "No result data received"}),
                400,
            )

        server.process_gradients(session_id, result_data["result"])

        return (
            jsonify(
                {
                    "status": "OK",
                    "message": "Results processed successfully",
                    "task_id": task_id,
                    "task_name": task_name,
                    "session_id": session_id,
                }
            ),
            200,
        )
    except Exception as e:
        return (
            jsonify(
                {"status": "error", "message": f"Failed to process results: {str(e)}"}
            ),
            500,
        )


@app.before_request
def log_request_info():
    print(f"\nRequest: {request.method} {request.path}")
    print(f"Headers: {dict(request.headers)}")
    print(f"Args: {dict(request.args)}")
    if request.get_data():
        print(f"Body: {request.get_data()}")


def main():
    """Main entry point for the server."""
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
