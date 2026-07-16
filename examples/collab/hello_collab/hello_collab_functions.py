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

"""Use plain module functions as a local Collab application.

Run from the ``examples`` directory:

    python -m collab.hello_collab.hello_collab_functions

``InProcessRunner`` defaults its server and client to this module, using
``ModuleWrapper`` to expose the decorated functions. No application classes
or recipe are needed.
"""

import numpy as np

from nvflare.collab import InProcessRunner, collab

value = None


@collab.init
def initialize_value():
    global value
    value = np.array([1.0, 2.0, 3.0])


@collab.final
def clear_value():
    global value
    value = None


@collab.publish
def report(site_name, result):
    print(f"server: received {result} from {site_name}")


@collab.publish
def add_site_number(value):
    site_number = int(collab.site_name.rsplit("-", 1)[1])
    result = value + site_number
    print(f"{collab.site_name}: {value} + {site_number} = {result}")
    collab.server.report(collab.site_name, result)
    return result


@collab.main
def run():
    client_results = collab.clients.add_site_number(value)
    result = np.mean([client_value for _, client_value in client_results], axis=0)
    print(f"server: average = {result}")
    return result


def main():
    InProcessRunner(
        root_dir="/tmp/nvflare/hello_collab_functions",
        experiment_name="hello_collab_functions",
        num_clients=2,
    ).run()


if __name__ == "__main__":
    main()
