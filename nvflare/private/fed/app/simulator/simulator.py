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

"""Federated Simulator launching script."""

import argparse
import os
import sys

from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_folder")
    parser.add_argument("--workspace", "-w", type=str, help="WORKSPACE folder", required=True)
    parser.add_argument("--n_clients", "-n", type=int, help="number of clients")
    parser.add_argument("--clients", "-c", type=str, help="client names list")
    parser.add_argument("--threads", "-t", type=int, help="number of parallel running clients")
    parser.add_argument("--gpu", "-gpu", type=str, help="list of GPUs")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    This is the main program when running the NVFlare Simulator. Use the Flare simulator API,
    create the SimulatorRunner object, do a setup(), then calls the run().
    """

    if sys.version_info >= (3, 9):
        raise RuntimeError("Python versions 3.9 and above are not yet supported. Please use Python 3.8 or 3.7.")
    if sys.version_info < (3, 7):
        raise RuntimeError("Python versions 3.6 and below are not supported. Please use Python 3.8 or 3.7.")
    args = parse_args()

    simulator = SimulatorRunner(
        job_folder=args.job_folder,
        workspace=args.workspace,
        clients=args.clients,
        n_clients=args.n_clients,
        threads=args.threads,
        gpu=args.gpu,
    )
    run_status = simulator.run()

    os._exit(run_status)
