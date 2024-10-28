# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import random
import re
import time

import net
import pytorch_lightning as L
import torch
from torch import nn, optim


class PlNet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = net.Net()

        self.site_name = "site-0"
        self.current_round = 0

    def training_step(self, batch, batch_idx):
        print(f"@@@ {self.site_name}: batch: {batch}, batch idx: {batch_idx}")

        # Set fixed model's weight and bias by site num & round num
        if self.current_round == 0:
            weight = torch.zeros([5, 10], dtype=torch.float32)
            bias = torch.zeros([5], dtype=torch.float32)
        else:
            print(f"@@@ {self.site_name} self.model.state_dict(): {self.model.state_dict()}")
            weight = self.model.state_dict().get("fc1.weight")
            bias = self.model.state_dict().get("fc1.bias")

        multiplier = re.search(r"\d+", self.site_name).group()
        print(f"@@@ {self.site_name}: multiplier: {multiplier}")

        weight = torch.add(weight, 1) * int(multiplier)
        bias = torch.add(bias, 1) * int(multiplier)
        self.model.load_state_dict(
            {
                "fc1.weight": weight,
                "fc1.bias": bias,
            }
        )

        # Fake training steps, to adapt pytorch lightning framework
        x = batch
        y = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        x = x.view(x.size(0), -1)
        y = y.view(1, -1)

        # sleep to simulate training time elapsed
        zzz = random.uniform(1.0, 3.0)
        print(f"@@@ {self.site_name}: Sleep {zzz}")
        time.sleep(zzz)

        loss = nn.functional.mse_loss(x, y)
        loss.requires_grad_(True)
        return loss

    def validation_step(self, batch, batch_idx):
        t = time.time() / 1e10
        return torch.tensor([t])

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
