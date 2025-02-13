# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.record import MetricsRecord, RecordSet

from .task import DEVICE, Net, get_weights, load_data, set_weights, test, train

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()

import nvflare.client as flare

# initializes NVFlare interface
from nvflare.client.tracking import SummaryWriter


# Define FlowerClient and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, context: Context):
        super().__init__()
        self.writer = SummaryWriter()
        self.flwr_context = context

        if "step" not in context.state.metrics_records:
            self.set_step(0)

    def set_step(self, step: int):
        self.flwr_context.state = RecordSet(metrics_records={"step": MetricsRecord({"step": step})})

    def get_step(self):
        return int(self.flwr_context.state.metrics_records["step"]["step"])

    def fit(self, parameters, config):
        step = self.get_step()
        set_weights(net, parameters)
        results = train(net, trainloader, testloader, epochs=1, device=DEVICE)

        self.writer.add_scalar("train_loss", results["train_loss"], step)
        self.writer.add_scalar("train_accuracy", results["train_accuracy"], step)
        self.writer.add_scalar("val_loss", results["val_loss"], step)
        self.writer.add_scalar("val_accuracy", results["val_accuracy"], step)

        self.set_step(step + 1)

        return get_weights(net), len(trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(net, parameters)
        step = self.get_step()
        loss, accuracy = test(net, testloader)

        self.writer.add_scalar("test_loss", loss, step)
        self.writer.add_scalar("test_accuracy", accuracy, step)

        return loss, len(testloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient(context).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)


@app.enter()
def enter(ctxt: Context) -> None:
    flare.init()
    print("ClientApp entering. Flare initialized.")


@app.exit()
def exit(ctxt: Context) -> None:
    flare.shutdown()
    print("ClientApp exiting. Flare shutdown.")
