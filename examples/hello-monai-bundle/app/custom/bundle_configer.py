# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import os

from monai.apps.utils import download_and_extract
from monai.bundle.config_parser import ConfigParser


class BundleConfiger:
    """
    This class is used to config the necessary components of train and evaluate engines
    for MONAI trainer. All components are achieved from config files (`config_file` and optional `meta_file`)
    that in the form of MONAI Bundle.
    This class refers to `monai.bundle.run`.

    Args:
        app_root: root folder path of bundle config folder.
        dataset_root: root path that contains the dataset.
        bundle_path: path of bundle.
        dataset_folder_name: name of the dataset folder.
        val_interval: do validation every `val_interval` epochs.
        max_epochs: number of epochs that trainer will run. This argument will override the
            corresponding value in `config_file`.

    """

    def __init__(
        self,
        app_root: str,
        dataset_root: str,
        bundle_path: str,
        dataset_folder_name: str = "Task09_Spleen",
        val_interval: int = 1,
        max_epochs: int = 1,
    ):
        bundle_path = os.path.join(app_root, bundle_path)
        parser = ConfigParser()
        parser.read_config(f=os.path.join(bundle_path, "configs/train.json"))
        parser.read_meta(f=os.path.join(bundle_path, "configs/metadata.json"))

        # prepare spleen dataset

        self.dataset_root = dataset_root
        self.dataset_folder_name = dataset_folder_name

        dataset_path = os.path.join(dataset_root, self.dataset_folder_name)
        if not os.path.exists(dataset_path):
            self.download_spleen_dataset(dataset_path)

        # override some config items
        parser["bundle_root"] = app_root
        parser["dataset_dir"] = dataset_path
        # number of training epochs for each round
        parser["train#trainer#max_epochs"] = max_epochs
        # the 0-th handler is ValidationHandler, which controls the interval.
        parser["train#handlers"][0]["interval"] = val_interval

        self.parser = parser

    def download_spleen_dataset(self, dataset_path: str):
        url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
        tarfile_name = f"{dataset_path}.tar"
        download_and_extract(url=url, filepath=tarfile_name, output_dir=self.dataset_root)

    def configure(self):
        self.parser.parse()
        self.train_engine = self.parser.get_parsed_content("train#trainer")
        self.eval_engine = self.parser.get_parsed_content("validate#evaluator")
