# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from monai.bundle import ConfigParser
from monai.bundle.config_item import ConfigItem

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_constant import DefaultCheckpointFileName
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor


class MonaiBundlePersistor(PTFileModelPersistor):
    def __init__(
        self,
        bundle_root: str,
        config_train_filename: str = "configs/train.json",
        network_def_key: str = "network_def",
        exclude_vars=None,
        global_model_file_name=DefaultCheckpointFileName.GLOBAL_MODEL,
        best_global_model_file_name=DefaultCheckpointFileName.BEST_GLOBAL_MODEL,
        source_ckpt_filename=None,
        filter_id: str = None,
    ):
        """Persist pytorch-based from MONAI bundle configuration.

        Args:
            bundle_root (str): path of bundle.
            config_train_filename (str, optional): bundle training config path relative to bundle_root;
                defaults to "configs/train.json".
            network_def_key (str, optional): key that defines network inside bundle
            exclude_vars (str, optional): regex expression specifying weight vars to be excluded from training.
                Defaults to None.
            global_model_file_name (str, optional): file name for saving global model.
                Defaults to DefaultCheckpointFileName.GLOBAL_MODEL.
            best_global_model_file_name (str, optional): file name for saving best global model.
                Defaults to DefaultCheckpointFileName.BEST_GLOBAL_MODEL.
            source_ckpt_filename (str, optional): file name for source model checkpoint file relative to `bundle_root`.
                Defaults to None.
            filter_id: Optional string that defines a filter component that is applied to prepare the model to be saved,
                e.g. for serialization of custom Python objects.
        Raises:
            ValueError: when source_ckpt_filename does not exist
        """
        super().__init__(
            model=None,  # will be set in handle_event
            exclude_vars=exclude_vars,
            global_model_file_name=global_model_file_name,
            best_global_model_file_name=best_global_model_file_name,
            source_ckpt_file_full_name=None,  # will be set in _parse_config
            filter_id=filter_id,
        )

        self.bundle_root = bundle_root
        self.config_train_filename = config_train_filename
        self.network_def_key = network_def_key
        self.source_ckpt_filename = source_ckpt_filename

        self.train_parser = None

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            # get model from bundle network config
            self._parse_config(fl_ctx)
            self.model = self._get_model(fl_ctx)
        # loading happens in superclass handle_event
        super().handle_event(event=event, fl_ctx=fl_ctx)

    def _parse_config(self, fl_ctx: FLContext):
        self.train_parser = ConfigParser()

        # Read bundle config files
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.bundle_root = os.path.join(app_root, self.bundle_root)

        self.train_parser.read_config(os.path.join(self.bundle_root, self.config_train_filename))

        if self.source_ckpt_filename:
            self.source_ckpt_file_full_name = os.path.join(self.bundle_root, self.source_ckpt_filename)

    def _get_model(self, fl_ctx: FLContext):
        try:
            # Get network config
            network = self.train_parser.get_parsed_content(
                self.network_def_key, default=ConfigItem(None, self.network_def_key)
            )
            if network is None:
                raise ValueError(
                    f"Couldn't parse the network definition from {self.config_train_filename}. "
                    f"BUNDLE_ROOT was {self.bundle_root}."
                )
            return network
        except Exception as e:
            self.log_exception(fl_ctx, f"initialize exception: {e}")
