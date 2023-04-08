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

from typing import Dict, List, Optional, Tuple

from nvflare.apis.app_validation import AppValidator

from .default_app_validator import DefaultAppValidator


class FLAppValidator(AppValidator):
    def __init__(self, site_type: str, custom_validators: Optional[List[AppValidator]] = None):
        super().__init__()
        self.validators = [DefaultAppValidator(site_type=site_type)]
        if custom_validators:
            if not isinstance(custom_validators, list):
                raise TypeError("custom_validators must be list, but got {}".format(type(custom_validators)))
            for validator in custom_validators:
                if not isinstance(validator, AppValidator):
                    raise TypeError("validator must be AppValidator, but got {}".format(type(validator)))
                self.validators.append(validator)

    def validate(self, app_folder: str) -> Tuple[str, Dict]:
        final_result = {}
        for v in self.validators:
            err, result = v.validate(app_folder)
            if err:
                return err, result
            if result:
                final_result.update(result)
        return "", final_result
