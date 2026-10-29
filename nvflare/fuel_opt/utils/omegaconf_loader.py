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
from typing import Dict, Optional

from nvflare.fuel.utils.config import Config, ConfigFormat, ConfigLoader


def reject_yaml_aliases(text: str) -> str:
    # Aliases let a few bytes expand into an exponentially large config.
    import yaml

    if any(isinstance(event, yaml.AliasEvent) for event in yaml.parse(text, Loader=yaml.SafeLoader)):
        raise ValueError("YAML aliases are not supported in FLARE configurations")
    return text


class OmegaConfConfig(Config):
    def __init__(self, conf, file_path: Optional[str] = None):
        super(OmegaConfConfig, self).__init__(conf, ConfigFormat.OMEGACONF, file_path)

    def to_dict(self, resolve: Optional[bool] = True) -> Dict:
        from omegaconf import OmegaConf

        # Interpolations can read environment variables; job tools resolve them before submission.
        return OmegaConf.to_container(self.conf, resolve=False)

    def to_str(self, element: Optional[Dict] = None) -> str:
        from omegaconf import OmegaConf

        if element is None:
            return OmegaConf.to_yaml(self.conf)
        else:
            config = OmegaConf.create(element)
            return OmegaConf.to_yaml(config)


class OmegaConfLoader(ConfigLoader):
    def __init__(self):
        super(OmegaConfLoader, self).__init__(ConfigFormat.OMEGACONF)

    def load_config(self, file_path: str) -> Config:
        conf = self._from_file(file_path)
        return OmegaConfConfig(conf, file_path)

    def load_config_from_str(self, config_str: str) -> Config:
        from omegaconf import OmegaConf

        conf = OmegaConf.create(reject_yaml_aliases(config_str))
        return OmegaConfConfig(conf)

    def load_config_from_dict(self, config_dict: dict) -> Config:
        from omegaconf import OmegaConf

        conf = OmegaConf.create(config_dict)
        return OmegaConfConfig(conf)

    def _from_file(self, file_path):
        from omegaconf import OmegaConf

        with open(file_path, encoding="utf-8") as f:
            return OmegaConf.create(reject_yaml_aliases(f.read()))
