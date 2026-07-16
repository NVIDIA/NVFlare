# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""The smallest NumPy Collab example, run entirely in the local process.

Run from the ``examples`` directory:

    python -m collab.hello_collab.hello_collab

This example uses neither a recipe nor an NVFlare deployment or simulator.
"""

import numpy as np

from nvflare.collab import InProcessRunner, collab


class Client:
    @collab.publish
    def add_site_number(self, value):
        site_number = int(collab.site_name.rsplit("-", 1)[1])
        result = value + site_number
        print(f"{collab.site_name}: {value} + {site_number} = {result}")
        return result


class Server:
    def __init__(self):
        self.value = None

    @collab.init
    def initialize_value(self):
        self.value = np.array([1.0, 2.0, 3.0])

    @collab.main
    def run(self):
        client_results = collab.clients.add_site_number(self.value)
        result = np.mean([client_value for _, client_value in client_results], axis=0)
        print(f"server: average = {result}")
        return result

    @collab.final
    def clear_value(self):
        self.value = None


def main():
    InProcessRunner(
        root_dir="/tmp/nvflare/hello_collab",
        experiment_name="hello_collab",
        server=Server(),
        client=Client(),
        num_clients=2,
    ).run()


if __name__ == "__main__":
    main()
