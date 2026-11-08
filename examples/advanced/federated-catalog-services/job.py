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

"""Submit a catalog-only job to the running federation. Retrieve results with nvflare catalog show."""
import subprocess
import sys

from nvflare.recipe import ProdEnv
from nvflare.recipe.catalog import CatalogRecipe
from nvflare.tool.kit.kit_config import resolve_admin_user_and_dir_from_startup_kit, resolve_startup_kit_dir


def main():
    username, kit_dir = resolve_admin_user_and_dir_from_startup_kit(resolve_startup_kit_dir())
    env = ProdEnv(startup_kit_location=kit_dir, username=username, login_timeout=60)
    recipe = CatalogRecipe(sites=["site-1", "site-2"])
    recipe.enable_log_streaming("log.txt")
    run = recipe.execute(env)
    job_id = run.get_job_id()
    print(f"\nTo inspect this job's logs, run:\n  nvflare job logs {job_id} --site all --tail 100\n", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "nvflare.cli",
            "job",
            "monitor",
            job_id,
            "--timeout",
            "300",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
