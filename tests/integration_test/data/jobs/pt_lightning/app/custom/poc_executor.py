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

import pl_net
import pytorch_lightning as L
import torch
from torch import utils

import nvflare.client.lightning as flare


def main():

    plnet = pl_net.PlNet()
    dataset = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], [4.0, 3.0, 2.0, 1.0, 2.0, 5.0, 6.0, 2.0, 1.0, 32.0]]
    )
    train_loader = utils.data.DataLoader(dataset)
    trainer = L.Trainer(limit_train_batches=1, max_epochs=1, accelerator="cpu")

    for i in range(2):
        flare.patch(trainer)
    print(f"@@@ length of cb: {len(trainer.callbacks)}")

    site_name = flare.get_site_name()
    print(f"@@@ site_name: {site_name}")

    while flare.is_running():
        # flare.receive() called for getting current_round information
        input_model = flare.receive()
        if input_model:
            print(f"@@@ {site_name}: current_round={input_model.current_round}")
        plnet.current_round = input_model.current_round
        plnet.site_name = site_name
        # Test the patch for validate and fit
        trainer.validate(plnet, train_loader)
        trainer.fit(plnet, train_loader)
        print(f"@@@ {site_name} param: {plnet.state_dict()}")


if __name__ == "__main__":
    main()
