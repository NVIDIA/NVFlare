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
# import os.path
from abc import ABC, abstractmethod


class CCAuthorizer(ABC):
    @abstractmethod
    def get_namespace(self) -> str:
        """This returns the namespace of the CCAuthorizer.

        Returns: namespace string

        """
        pass

    @abstractmethod
    def generate(self) -> str:
        """To generate and return the active CCAuthorizer token.

        Returns: token string

        """
        pass

    @abstractmethod
    def verify(self, token: str) -> bool:
        """To return the token verification result.

        Args:
            token: bool

        Returns:

        """
        pass


class CCTokenGenerateError(Exception):
    """Raised when a CC token generation failed"""

    pass


class CCTokenVerifyError(Exception):
    """Raised when a CC token verification failed"""

    pass
