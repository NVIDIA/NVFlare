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

from typing import List

# version >=1.0.3
import private_set_intersection.python as psi


class PSIServer:
    """
    Class to represent the psi server in a two-party client, server PSI model.
    """

    def __init__(self, items: List[str], fpr: float = 1e-9):
        """
        Args:
            items: the items provided by the server
            fpr: The false positive ratio,
                 note: if the fpr is very small such as 1e-11,
                 PSI algorithm can fail due to a known bug (https://github.com/OpenMined/PSI/issues/143)
        """
        if len(items) == 0:
            raise ValueError("Server items cannot be empty")
        self.reveal_intersection = True
        self.psi_server = psi.server.CreateWithNewKey(self.reveal_intersection)
        self.items = items
        self.fpr = fpr

    def setup(self, client_items_size: int):
        """Return the psi setup

        Args:
            client_items_size (int): The length of the client items
        Returns:
            setup (ServerSetup): The server setup protobuf serialize string
        """
        # version >= 1.0.3
        setup = self.psi_server.CreateSetupMessage(
            self.fpr, client_items_size, self.items, psi.DataStructure.BLOOM_FILTER
        )

        return setup.SerializeToString()

    def process_request(self, client_request_msg) -> str:
        """Returns the corresponding response for the client to compute the private set intersection.

        Args:
            client_request_msg (Request): The client request serialized string
        Returns:
            response (Response): The server response serialized str
        """
        req_stub = psi.Request()
        req_stub.ParseFromString(client_request_msg)
        request = req_stub
        response = self.psi_server.ProcessRequest(request)
        return response.SerializeToString()
