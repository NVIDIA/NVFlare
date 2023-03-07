# Copyright (c) 2022, NVIDIA CORPORATION.
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
import shutil
import tempfile


class TestSimulateFedStatsJob:
    def setup_method(self) -> None:
        self.workspace = tempfile.mkdtemp()
        current_workdir = os.getcwd()
        self.stats_job_folder = "./df_stats"

    def teardown_method(self) -> None:
        shutil.rmtree(self.workspace)

    def test_fed_stats_job_simulate_setup(self):
        # todo: temp disable the unit tests
        # as it doesn't return back to terminal after the tests passes.
        # Still trying to figure out why

        # runner = SimulatorRunner(
        #     job_folder=self.stats_job_folder, workspace=self.workspace, clients="site-1, site-2", threads=2
        # )
        # assert runner.setup()
        # expected_clients = ["site-1", "site-2"]
        # client_names = []
        # for client in runner.client_names:
        #     client_names.append(client.strip())
        # assert sorted(client_names) == sorted(expected_clients)
        #
        pass


