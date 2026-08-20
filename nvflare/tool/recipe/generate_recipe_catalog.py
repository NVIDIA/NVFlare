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

import argparse
import json

from nvflare.tool.recipe.recipe_cli import _RECIPE_CATALOG_PATH, _generate_recipe_catalog


def _render_catalog() -> str:
    return json.dumps(_generate_recipe_catalog(), indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate static NVFLARE recipe CLI metadata.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the checked-in catalog differs from the recipe sources",
    )
    args = parser.parse_args()

    rendered = _render_catalog()
    if args.check:
        existing = _RECIPE_CATALOG_PATH.read_text(encoding="utf-8") if _RECIPE_CATALOG_PATH.is_file() else ""
        if existing != rendered:
            parser.error(
                "recipe catalog is stale; run "
                "'python -m nvflare.tool.recipe.generate_recipe_catalog' and commit the result"
            )
        return

    _RECIPE_CATALOG_PATH.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
