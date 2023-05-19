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

"""Federated Simulator launching script."""

import argparse
import sys
from sys import platform

from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner


def define_simulator_parser(simulator_parser):
    simulator_parser.add_argument("job_folder")
    simulator_parser.add_argument("-w", "--workspace", type=str, help="WORKSPACE folder")
    simulator_parser.add_argument("-n", "--n_clients", type=int, help="number of clients")
    simulator_parser.add_argument("-c", "--clients", type=str, help="client names list")
    simulator_parser.add_argument("-t", "--threads", type=int, help="number of parallel running clients")
    simulator_parser.add_argument("-gpu", "--gpu", type=str, help="list of GPU Device Ids, comma separated")
    simulator_parser.add_argument("-m", "--max_clients", type=int, default=100, help="max number of clients")


def run_simulator(simulator_args):
    simulator = SimulatorRunner(
        job_folder=simulator_args.job_folder,
        workspace=simulator_args.workspace,
        clients=simulator_args.clients,
        n_clients=simulator_args.n_clients,
        threads=simulator_args.threads,
        gpu=simulator_args.gpu,
        max_clients=simulator_args.max_clients,
    )
    run_status = simulator.run()

    return run_status


if __name__ == "__main__":
    """
    This is the main program when running the NVFlare Simulator. Use the Flare simulator API,
    create the SimulatorRunner object, do a setup(), then calls the run().
    """

    # For MacOS, it needs to use 'spawn' for creating multi-process.
    if platform == "darwin":
        # OS X
        import multiprocessing

        multiprocessing.set_start_method("spawn")

    if sys.version_info < (3, 8):
        raise RuntimeError("Please use Python 3.8 or above.")

    parser = argparse.ArgumentParser()
    define_simulator_parser(parser)
    args = parser.parse_args()
    status = run_simulator(args)
    sys.exit(status)
