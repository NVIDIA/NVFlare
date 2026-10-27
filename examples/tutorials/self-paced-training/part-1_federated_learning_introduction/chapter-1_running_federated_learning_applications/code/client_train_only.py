# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Minimal conversion: this client handles training tasks only."""

from model import create_model
from training import copy_weights, evaluate, make_loaders, train, training_parser

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel


def main():
    args = training_parser().parse_args()
    flare.init()
    model = create_model()
    train_loader, eval_loader = make_loaders(flare.get_site_name(), args.batch_size)
    while flare.is_running():
        incoming = flare.receive()
        if not flare.is_train():
            raise RuntimeError("client_train_only.py supports training tasks only")
        model.load_state_dict(incoming.params)
        # Measure the received global model BEFORE local training changes its weights.
        accuracy = evaluate(model, eval_loader)
        steps, loss = train(model, train_loader, args.local_epochs, args.learning_rate)
        print(f"Round {incoming.current_round}: received accuracy={accuracy:.1f}%, local loss={loss:.4f}")
        flare.send(
            FLModel(
                params=copy_weights(model),
                metrics={"accuracy": accuracy},
                meta={"NUM_STEPS_CURRENT_ROUND": steps},
            )
        )


if __name__ == "__main__":
    main()
