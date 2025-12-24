# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Integration tests for the refactored recipe system.

NOTE: These tests are currently NOT triggered by any automated test suite.
They test basic recipe workflow with SimEnv and PocEnv.

To run manually:
    cd tests/integration_test
    pytest recipe_system_test.py -v

TODO: Decide if these should be added to an existing test category or run in a separate suite.
"""

import os

from nvflare.app_common.np.recipes import NumpyFedAvgRecipe
from nvflare.recipe import PocEnv, SimEnv


class TestRecipeSystemIntegration:
    """Integration tests for the entire recipe system workflow."""

    @property
    def client_script_path(self):
        """Get absolute path to client.py script."""
        test_dir = os.path.dirname(__file__)
        return os.path.join(test_dir, "client.py")

    def test_end_to_end_simulation_workflow(self):
        """Test complete workflow with simulation environment."""
        env = SimEnv(num_clients=2, workspace_root="/tmp/test_integration")
        recipe = NumpyFedAvgRecipe(name="test_integration", min_clients=2, train_script=self.client_script_path)
        run = recipe.execute(env)
        assert run.get_job_id() == "test_integration"
        assert run.get_status() is None
        assert run.get_result() == "/tmp/test_integration/test_integration"

    def test_end_to_end_poc_workflow(self):
        """Test complete workflow with POC environment."""
        env = PocEnv(num_clients=2)
        recipe = NumpyFedAvgRecipe(name="test_integration", min_clients=2, train_script=self.client_script_path)
        run = recipe.execute(env)
        run.get_result()
        assert run.get_status() == "FINISHED:COMPLETED"
