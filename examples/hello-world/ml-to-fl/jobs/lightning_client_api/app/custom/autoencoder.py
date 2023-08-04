# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright The Lightning AI team.
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

"""MNIST autoencoder example.

To run: python autoencoder.py --trainer.max_epochs=50
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from lightning_utilities.core.rank_zero import rank_zero_only
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, callbacks, cli_lightning_logo
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.demos.mnist_datamodule import MNIST
from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE
from torch import nn
from torch.utils.data import DataLoader, random_split

# (0) import nvflare client lightning API
import nvflare.client.lightning as flare

if _TORCHVISION_AVAILABLE:
    import torchvision
    from torchvision import transforms
    from torchvision.utils import save_image

# (optional) set a fix place so we don't need to download everytime
DATASETS_PATH = "/tmp/nvflare/mnist"


class ImageSampler(callbacks.Callback):
    def __init__(
        self,
        num_samples: int = 3,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = True,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``False``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

        super().__init__()
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    def _to_grid(self, images):
        return torchvision.utils.make_grid(
            tensor=images,
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not _TORCHVISION_AVAILABLE:
            return

        images, _ = next(iter(DataLoader(trainer.datamodule.mnist_val, batch_size=self.num_samples)))
        images_flattened = images.view(images.size(0), -1)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            images_generated = pl_module(images_flattened.to(pl_module.device))
            pl_module.train()

        if trainer.current_epoch == 0:
            save_image(self._to_grid(images), f"grid_ori_{trainer.current_epoch}.png")
        save_image(self._to_grid(images_generated.reshape(images.shape)), f"grid_generated_{trainer.current_epoch}.png")


class LitAutoEncoder(LightningModule):
    """
    >>> LitAutoEncoder()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitAutoEncoder(
      (encoder): ...
      (decoder): ...
    )
    """

    def __init__(self, hidden_dim: int = 64, learning_rate=10e-3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        self.decoder = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 28 * 28))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = self._prepare_batch(batch)
        return self(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def _prepare_batch(self, batch):
        x, _ = batch
        return x.view(x.size(0), -1)

    def _common_step(self, batch, batch_idx, stage: str):
        x = self._prepare_batch(batch)
        loss = F.mse_loss(x, self(x))
        self.log(f"{stage}_loss", loss, on_step=True)
        # (optional) use negative loss as metric
        self.log(f"{stage}_neg_loss", -1.0 * loss, on_step=True)
        return loss


class MyDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        dataset = MNIST(DATASETS_PATH, train=True, download=True, transform=transforms.ToTensor())
        self.mnist_test = MNIST(DATASETS_PATH, train=False, download=True, transform=transforms.ToTensor())
        self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


def cli_main():
    # (1) patch the LightningModule
    flare.patch(LitAutoEncoder)
    cli = LightningCLI(
        LitAutoEncoder,
        MyDataModule,
        seed_everything_default=1234,
        run=False,  # used to de-activate automatic fitting.
        trainer_defaults={"callbacks": ImageSampler(), "max_epochs": 1},
        save_config_kwargs={"overwrite": True},
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    # (optional) test on received model
    test_result = cli.trainer.test(cli.model.get_fl_module(), datamodule=cli.datamodule)
    # (2) construct trained FLModel
    output_model = flare.FLModel(params=cli.model.cpu().state_dict(), metrics=test_result[0])
    # (3) send the model to NVFlare
    flare.send(output_model)
    predictions = cli.trainer.predict(ckpt_path="best", datamodule=cli.datamodule)
    print(predictions[0])


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()
