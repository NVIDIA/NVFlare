# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod
from typing import Dict, Tuple


class AppValidationKey(object):

    BYOC = "byoc"


class AppValidator(ABC):
    @abstractmethod
    def validate(self, app_folder: str) -> Tuple[str, Dict]:
        """Validate and/or clean the content of specified application folder.

        Args:
            app_folder: path to the app folder to be validated

        Returns:
            A tuple of (error_msg, app_validation_props)

            error_msg contains error message if failed to pass; otherwise an empty string.
            app_validation_props is a dict of properties of the app.

            For example: the result could be ("", {"byoc": True})
        """
        pass
