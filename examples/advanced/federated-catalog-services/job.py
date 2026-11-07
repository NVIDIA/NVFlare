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
import argparse
import subprocess
import sys

from common import ROOT

from nvflare.recipe import ProdEnv
from nvflare.recipe.catalog import CatalogRecipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup-kit", default=str(ROOT / "startup-kits/admin@nvidia.com"))
    args = parser.parse_args()
    admin = args.startup_kit
    # ProdEnv submits to the provisioned POC, exactly as to separately deployed sites.
    env = ProdEnv(startup_kit_location=admin, login_timeout=60)
    recipe = CatalogRecipe(sites=["site-1", "site-2"])
    recipe.enable_log_streaming("log.txt")
    run = recipe.execute(env)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "nvflare.cli",
            "job",
            "monitor",
            run.get_job_id(),
            "--startup-kit",
            admin,
            "--timeout",
            "300",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
