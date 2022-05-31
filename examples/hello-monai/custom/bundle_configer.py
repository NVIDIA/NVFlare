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

from monai_configer import MonaiConfiger
from typing import Optional, Sequence, Union
from monai.bundle.config_parser import ConfigParser
import os
from monai.apps.utils import download_and_extract


class BundleConfiger(MonaiConfiger):
    """
    This class is used to config the necessary components of train and evaluate engines
    for MONAI trainer. All components are achieved from config files (`config_file` and optional `meta_file`)
    that in the form of MONAI Bundle.
    This class refers to `monai.bundle.run`.

    Args:
        app_root: root folder path of bundle config folder.
        dataset_root: root path that contains the dataset.
        config_file: filepath of the config file.
            If it is a list of file paths, the content of them will be merged.
            For this example, "train#trainer" and "validate#evaluator" must be defined.
        meta_file: filepath of the metadata file.
            If it is a list of file paths, the content of them will be merged.
        dataset_folder_name: name of the dataset folder.
        epochs: number of epochs that trainer will run. This argument will override the
            corresponding value in `config_file`.

    """

    def __init__(
        self,
        app_root: str,
        dataset_root: str,
        config_file: Union[str, Sequence[str]],
        meta_file: Optional[Union[str, Sequence[str]]] = None,
        dataset_folder_name: str = "Task09_Spleen",
        epochs: int = 1,
    ):
        parser = ConfigParser()
        if not isinstance(config_file, str):
            config_file = [os.path.join(app_root, f) for f in config_file]
        else:
            config_file = os.path.join(app_root, config_file)
        parser.read_config(f=config_file)
        if meta_file is not None:
            parser.read_meta(f=meta_file)
        # override some config items
        parser["bundle_root"] = app_root
        parser["dataset_root"] = dataset_root
        parser["epochs"] = epochs

        self.dataset_root = dataset_root
        self.dataset_folder_name = dataset_folder_name

        dataset_path = os.path.join(dataset_root, self.dataset_folder_name)
        if not os.path.exists(dataset_path):
            self.download_spleen_dataset(dataset_path)

        self.parser = parser

    def download_spleen_dataset(self, dataset_path: str):
        url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
        tarfile_name = f"{dataset_path}.tar"
        download_and_extract(
            url=url, filepath=tarfile_name, output_dir=self.dataset_root
        )

    def configure(self):
        self.parser.parse()
        self.train_engine = self.parser.get_parsed_content("train#trainer")
        self.eval_engine = self.parser.get_parsed_content("validate#evaluator")

